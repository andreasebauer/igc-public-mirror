from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .checkpoints import current_process_identity, _process_start_token, _boot_id
from .records import utc_now
from .safety import validate_identifier

RUNTIME_CONTRACT_VERSION = "1.0.0"
RUNTIME_SCOPE_SCHEMA = "IG_DECODER_V05_P3_RUNTIME_SCOPE_V1"
TASK_TABLE_SCHEMA = "IG_DECODER_V05_P3_TASK_TABLE_V1"
CLAIM_SCHEMA = "IG_DECODER_V05_P3_SHARD_CLAIM_V1"
ATTEMPT_SCHEMA = "IG_DECODER_V05_P3_SHARD_ATTEMPT_V1"
COMMIT_SCHEMA = "IG_DECODER_V05_P3_SHARD_COMMIT_V1"
STATUS_SCHEMA = "IG_DECODER_V05_P3_RUNTIME_STATUS_V1"
TELEMETRY_SPAN_SCHEMA = "IG_DECODER_V05_P3_TELEMETRY_SPAN_V1"

REQUIRED_TELEMETRY_CATEGORIES = (
    "MATERIALIZATION", "CENSUS", "COMPOSITION", "CANONICALIZATION",
    "SERIALIZATION", "SCHEDULING", "MERGE", "CHECKPOINT_IO",
    "REPLAY", "VERIFICATION", "PUBLICATION",
)


class V05RuntimeError(RuntimeError):
    pass


class CompetingShardOwner(V05RuntimeError):
    pass


class CorruptShardState(V05RuntimeError):
    pass


class IncompatibleRuntimeState(V05RuntimeError):
    pass


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _read_json(path: Path) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise CorruptShardState(f"cannot read JSON state: {path}") from exc
    if not isinstance(obj, dict):
        raise CorruptShardState(f"state is not an object: {path}")
    return obj


def _age_seconds(utc_text: str | None) -> float | None:
    if not utc_text:
        return None
    try:
        dt = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None


def _pid_identity_live(identity: dict) -> bool:
    """True only if a local process still matches pid/boot/start token.

    A missing heartbeat alone is deliberately not enough to steal a live owner.
    Foreign-host claims are never auto-stolen by this local runtime.
    """
    if identity.get("host") != socket.gethostname():
        return True
    try:
        pid = int(identity.get("pid", 0))
    except Exception:
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    locked_boot = identity.get("boot_id")
    locked_start = identity.get("process_start_token")
    current_boot = _boot_id()
    current_start = _process_start_token(pid)
    if None in (locked_boot, locked_start, current_boot, current_start):
        return True
    return locked_boot == current_boot and locked_start == current_start


def _validate_sha(v: str, field: str) -> str:
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise V05RuntimeError(f"{field} must be lowercase sha256")
    return v


def shard_descriptor(
    *, task_id: str, generator_sha256: str, source_sha256: str, spec_sha256: str,
    input_sha256: str, dependency_sha256: list[str] | tuple[str, ...], task_payload: Any,
) -> dict:
    validate_identifier(task_id, field="task_id")
    deps = sorted(_validate_sha(x, "dependency_sha256") for x in dependency_sha256)
    base = {
        "task_id": task_id,
        "generator_sha256": _validate_sha(generator_sha256, "generator_sha256"),
        "source_sha256": _validate_sha(source_sha256, "source_sha256"),
        "spec_sha256": _validate_sha(spec_sha256, "spec_sha256"),
        "input_sha256": _validate_sha(input_sha256, "input_sha256"),
        "dependency_sha256": deps,
        "task_payload_sha256": canonical_sha256(task_payload),
    }
    return dict(base, shard_identity_sha256=canonical_sha256(base))


@dataclass(frozen=True)
class Claim:
    task_id: str
    claim_token: str
    attempt: int
    retry_count: int
    reclaimed_stale_owner: bool


class V05ShardRuntime:
    """P3 process-safe exactly-once logical shard runtime.

    Task descriptors are immutable and bind generator/source/spec/input/dependencies.
    Claims are operational and replaceable only when the old local process identity is
    proven dead. Commits are create-only immutable records that bind exact output bytes
    (through canonical output payload SHA-256). Re-execution is allowed; contribution is
    exactly once because only one immutable commit can exist per task id.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve(strict=True)
        self.scope_path = self.root / "scope.json"
        self.tasks_path = self.root / "tasks.json"
        self.lock_path = self.root / "runtime.lock"
        self.claims = self.root / "claims"
        self.commits = self.root / "commits"
        self.attempts = self.root / "attempts"
        self.progress = self.root / "progress"
        self.cancel_path = self.root / "cancel.json"
        scope = _read_json(self.scope_path)
        if scope.get("schema_id") != RUNTIME_SCOPE_SCHEMA or scope.get("runtime_contract_version") != RUNTIME_CONTRACT_VERSION:
            raise IncompatibleRuntimeState("runtime scope schema/version mismatch; old state is not resumable")
        observed = canonical_sha256({k: v for k, v in scope.items() if k != "scope_sha256"})
        if observed != scope.get("scope_sha256"):
            raise CorruptShardState("runtime scope hash mismatch")
        table = _read_json(self.tasks_path)
        if table.get("schema_id") != TASK_TABLE_SCHEMA:
            raise IncompatibleRuntimeState("task table schema mismatch")
        tasks = table.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise CorruptShardState("task table empty")
        if canonical_sha256(tasks) != table.get("tasks_sha256"):
            raise CorruptShardState("task table hash mismatch")
        self.scope = scope
        self.task_map = {x["task_id"]: x for x in tasks}
        if len(self.task_map) != len(tasks):
            raise CorruptShardState("duplicate task ids in task table")

    @classmethod
    def prepare(cls, root: Path, *, scope: dict, tasks: list[dict], writable_uid: int | None = None, writable_gid: int | None = None) -> "V05ShardRuntime":
        root = Path(root).resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        scope_base = dict(scope)
        scope_base.update(schema_id=RUNTIME_SCOPE_SCHEMA, runtime_contract_version=RUNTIME_CONTRACT_VERSION)
        scope_obj = dict(scope_base, scope_sha256=canonical_sha256(scope_base))
        task_sorted = sorted(tasks, key=lambda x: x["task_id"])
        for t in task_sorted:
            validate_identifier(t.get("task_id", ""), field="task_id")
            expected = canonical_sha256({k: v for k, v in t.items() if k != "shard_identity_sha256"})
            if t.get("shard_identity_sha256") != expected:
                raise V05RuntimeError(f"bad shard identity: {t.get('task_id')}")
        table = {"schema_id": TASK_TABLE_SCHEMA, "tasks": task_sorted, "tasks_sha256": canonical_sha256(task_sorted)}
        for path, obj in ((root / "scope.json", scope_obj), (root / "tasks.json", table)):
            if path.exists():
                if _read_json(path) != obj:
                    raise IncompatibleRuntimeState(f"existing immutable runtime state differs: {path.name}")
            else:
                write_json_atomic(path, obj)
                try: os.chmod(path, 0o444)
                except OSError: pass
        for d in (root / "claims", root / "commits", root / "attempts", root / "progress"):
            d.mkdir(parents=True, exist_ok=True)
            if writable_uid is not None and writable_gid is not None:
                try: os.chown(d, int(writable_uid), int(writable_gid)); os.chmod(d, 0o700)
                except OSError: pass
        lock = root / "runtime.lock"
        lock.touch(exist_ok=True)
        if writable_uid is not None and writable_gid is not None:
            try: os.chown(lock, int(writable_uid), int(writable_gid)); os.chmod(lock, 0o600)
            except OSError: pass
        _fsync_dir(root)
        return cls(root)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("r+") as h:
            fcntl.flock(h.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(h.fileno(), fcntl.LOCK_UN)

    def _descriptor(self, task_id: str) -> dict:
        validate_identifier(task_id, field="task_id")
        try: return self.task_map[task_id]
        except KeyError as exc: raise V05RuntimeError(f"undeclared shard: {task_id}") from exc

    def _commit_path(self, task_id: str) -> Path: return self.commits / f"{task_id}.json"
    def _claim_path(self, task_id: str) -> Path: return self.claims / f"{task_id}.json"
    def _progress_path(self, task_id: str) -> Path: return self.progress / f"{task_id}.json"

    def load_commit(self, task_id: str) -> dict | None:
        desc = self._descriptor(task_id); p = self._commit_path(task_id)
        if not p.exists(): return None
        obj = _read_json(p)
        if obj.get("schema_id") != COMMIT_SCHEMA or obj.get("task_id") != task_id:
            raise CorruptShardState(f"invalid commit schema/binding: {task_id}")
        if obj.get("shard_identity_sha256") != desc["shard_identity_sha256"]:
            raise CorruptShardState(f"commit shard identity mismatch: {task_id}")
        if canonical_sha256({k: v for k, v in obj.items() if k != "commit_content_sha256"}) != obj.get("commit_content_sha256"):
            raise CorruptShardState(f"commit content hash mismatch: {task_id}")
        if canonical_sha256(obj.get("output")) != obj.get("output_sha256"):
            raise CorruptShardState(f"commit output hash mismatch: {task_id}")
        return obj

    def _attempt_files(self, task_id: str) -> list[Path]:
        d = self.attempts / task_id
        return sorted(d.glob("*.json")) if d.exists() else []

    def claim(self, task_id: str, *, owner_id: str, worker_id: str) -> Claim:
        validate_identifier(owner_id, field="owner_id"); validate_identifier(worker_id, field="worker_id")
        desc = self._descriptor(task_id)
        with self._locked():
            if self.load_commit(task_id) is not None:
                return Claim(task_id, "ALREADY_COMMITTED", 0, max(0, len(self._attempt_files(task_id)) - 1), False)
            cp = self._claim_path(task_id); reclaimed = False
            if cp.exists():
                old = _read_json(cp)
                if old.get("schema_id") != CLAIM_SCHEMA or old.get("shard_identity_sha256") != desc["shard_identity_sha256"]:
                    raise CorruptShardState(f"claim binding corrupt: {task_id}")
                if _pid_identity_live(old.get("process_identity", {})):
                    raise CompetingShardOwner(f"live shard owner exists: {task_id}")
                reclaimed = True
                hist = self.attempts / task_id; hist.mkdir(parents=True, exist_ok=True)
                stale = dict(old, stale_recovered_utc=utc_now(), terminal_state="STALE_OWNER_RECLAIMED")
                stale_path = hist / f"stale-{uuid.uuid4().hex}.json"; write_json_atomic(stale_path, stale)
                cp.unlink(missing_ok=True)
            attempt = len([p for p in self._attempt_files(task_id) if p.name.startswith("attempt-")]) + 1
            token = uuid.uuid4().hex
            ident = current_process_identity()
            claim_base = {
                "schema_id": CLAIM_SCHEMA, "task_id": task_id,
                "shard_identity_sha256": desc["shard_identity_sha256"],
                "claim_token": token, "attempt": attempt, "owner_id": owner_id, "worker_id": worker_id,
                "process_identity": ident, "claimed_utc": utc_now(), "heartbeat_utc": utc_now(),
            }
            claim_obj = dict(claim_base, claim_content_sha256=canonical_sha256(claim_base))
            write_json_atomic(cp, claim_obj)
            hist = self.attempts / task_id; hist.mkdir(parents=True, exist_ok=True)
            attempt_base = {
                "schema_id": ATTEMPT_SCHEMA, "task_id": task_id, "attempt": attempt,
                "claim_token": token, "owner_id": owner_id, "worker_id": worker_id,
                "shard_identity_sha256": desc["shard_identity_sha256"], "started_utc": utc_now(),
                "retry": attempt > 1 or reclaimed,
            }
            write_json_atomic(hist / f"attempt-{attempt:06d}.json", dict(attempt_base, attempt_content_sha256=canonical_sha256(attempt_base)))
            return Claim(task_id, token, attempt, max(0, attempt - 1), reclaimed)

    def heartbeat(self, claim: Claim, *, completed_units: int | None = None, total_units: int | None = None, message: str | None = None) -> dict:
        if claim.claim_token == "ALREADY_COMMITTED":
            raise V05RuntimeError("cannot heartbeat an already committed shard")
        with self._locked():
            cp = self._claim_path(claim.task_id); obj = _read_json(cp)
            if obj.get("claim_token") != claim.claim_token:
                raise CompetingShardOwner("shard claim token changed")
            obj["heartbeat_utc"] = utc_now()
            obj["progress"] = {"completed_units": completed_units, "total_units": total_units, "message": message}
            content = {k: v for k, v in obj.items() if k != "claim_content_sha256"}; obj["claim_content_sha256"] = canonical_sha256(content)
            write_json_atomic(cp, obj)
            pbase = {
                "task_id": claim.task_id, "claim_token": claim.claim_token, "utc": obj["heartbeat_utc"],
                "completed_units": completed_units, "total_units": total_units, "message": message,
            }
            write_json_atomic(self._progress_path(claim.task_id), pbase)
            return pbase

    def commit(self, claim: Claim, output: Any) -> dict:
        desc = self._descriptor(claim.task_id); p = self._commit_path(claim.task_id)
        output_sha = canonical_sha256(output)
        with self._locked():
            existing = self.load_commit(claim.task_id)
            if existing is not None:
                if existing.get("output_sha256") != output_sha or existing.get("output") != output:
                    raise CorruptShardState(f"idempotent commit collision: {claim.task_id}")
                return dict(existing, reused_existing_commit=True)
            if claim.claim_token == "ALREADY_COMMITTED":
                raise CorruptShardState("commit reported missing after already-committed claim")
            cp = self._claim_path(claim.task_id); owner = _read_json(cp)
            if owner.get("claim_token") != claim.claim_token:
                raise CompetingShardOwner(f"claim ownership changed: {claim.task_id}")
            base = {
                "schema_id": COMMIT_SCHEMA, "task_id": claim.task_id,
                "shard_identity_sha256": desc["shard_identity_sha256"],
                "attempt": claim.attempt, "output_sha256": output_sha, "output": output,
                "committed_utc": utc_now(),
            }
            obj = dict(base, commit_content_sha256=canonical_sha256(base))
            tmp = self.commits / f".{claim.task_id}.{uuid.uuid4().hex}.tmp"
            data = canonical_text(obj, pretty=True).encode("utf-8")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as h:
                    h.write(data); h.flush(); os.fsync(h.fileno())
                try: os.link(tmp, p)
                except FileExistsError: pass
                _fsync_dir(self.commits)
            finally:
                tmp.unlink(missing_ok=True)
            final = self.load_commit(claim.task_id)
            if final is None or final.get("output_sha256") != output_sha:
                raise CorruptShardState(f"commit did not seal correctly: {claim.task_id}")
            cp.unlink(missing_ok=True); self._progress_path(claim.task_id).unlink(missing_ok=True)
            return final

    def cancel(self, *, reason: str, requested_by: str) -> dict:
        if not reason or not requested_by: raise V05RuntimeError("cancellation needs reason/requested_by")
        base = {"requested_utc": utc_now(), "reason": reason, "requested_by": requested_by}
        obj = dict(base, cancel_sha256=canonical_sha256(base))
        if self.cancel_path.exists():
            old = _read_json(self.cancel_path)
            if old != obj and old.get("reason") != reason:
                raise V05RuntimeError("runtime already cancelled with a different reason")
            return old
        write_json_atomic(self.cancel_path, obj); return obj

    def is_cancelled(self) -> bool: return self.cancel_path.is_file()

    def status(self, *, no_progress_threshold_seconds: float = 30.0, no_progress_action: str = "REPORT_ONLY") -> dict:
        completed = 0; running = []; retries = 0; last_times = []
        for task_id in sorted(self.task_map):
            c = self.load_commit(task_id)
            if c is not None:
                completed += 1; last_times.append(c.get("committed_utc"))
            cp = self._claim_path(task_id)
            if cp.exists():
                cl = _read_json(cp); live = _pid_identity_live(cl.get("process_identity", {}))
                running.append({"task_id": task_id, "owner_id": cl.get("owner_id"), "worker_id": cl.get("worker_id"), "attempt": cl.get("attempt"), "live": live, "heartbeat_utc": cl.get("heartbeat_utc"), "progress": cl.get("progress")})
                last_times.append(cl.get("heartbeat_utc"))
            attempts = [p for p in self._attempt_files(task_id) if p.name.startswith("attempt-")]
            retries += max(0, len(attempts) - 1)
        total = len(self.task_map); running_ids = {x["task_id"] for x in running}
        queued = total - completed - len([x for x in running_ids if self.load_commit(x) is None])
        last = max((x for x in last_times if x), default=None)
        age = _age_seconds(last)
        no_progress = age is not None and age > float(no_progress_threshold_seconds) and completed < total
        base = {
            "schema_id": STATUS_SCHEMA, "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
            "scope_sha256": self.scope["scope_sha256"], "tasks_total": total,
            "queued": max(0, queued), "running": len(running), "completed": completed,
            "retried": retries, "active_workers": running, "last_progress_utc": last,
            "seconds_since_last_progress": age, "no_progress_threshold_seconds": float(no_progress_threshold_seconds),
            "no_progress": no_progress, "no_progress_action": no_progress_action,
            "cancelled": self.is_cancelled(), "science_progress_fraction": (completed / total if total else 1.0),
            "heartbeat_semantics": "LIVENESS_SEPARATE_FROM_SCIENCE_PROGRESS",
        }
        return dict(base, status_sha256=canonical_sha256(base))

    def verify_complete(self) -> dict:
        commits = [self.load_commit(t) for t in sorted(self.task_map)]
        missing = [t for t, c in zip(sorted(self.task_map), commits) if c is None]
        if missing:
            return {"status": "INCOMPLETE", "missing": missing, "completed": len(commits)-len(missing), "total": len(commits)}
        bindings = [{"task_id": c["task_id"], "shard_identity_sha256": c["shard_identity_sha256"], "output_sha256": c["output_sha256"], "commit_content_sha256": c["commit_content_sha256"]} for c in commits if c]
        return {"status": "PASS", "completed": len(bindings), "total": len(bindings), "bindings": bindings, "bindings_sha256": canonical_sha256(bindings)}


class V05TelemetryLedger:
    """Operational span ledger. Spans never enter scientific projection hashes."""
    def __init__(self, path: Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, category: str, *, component: str, wall_seconds: float = 0.0, process_cpu_seconds: float = 0.0, status: str = "PASS", details: dict | None = None, error: str | None = None) -> dict:
        if category not in REQUIRED_TELEMETRY_CATEGORIES:
            raise V05RuntimeError(f"unknown telemetry category: {category}")
        base = {
            "schema_id": TELEMETRY_SPAN_SCHEMA, "category": category, "component": component,
            "started_utc": None, "finished_utc": utc_now(), "wall_seconds": float(wall_seconds),
            "process_cpu_seconds": float(process_cpu_seconds), "status": status, "details": details or {}, "error": error,
        }
        rec = dict(base, span_sha256=canonical_sha256(base))
        with self.path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n"); h.flush()
            try: os.fsync(h.fileno())
            except OSError: pass
        return rec

    @contextlib.contextmanager
    def span(self, category: str, *, component: str, details: dict | None = None):
        if category not in REQUIRED_TELEMETRY_CATEGORIES:
            raise V05RuntimeError(f"unknown telemetry category: {category}")
        start = time.perf_counter(); cpu0 = time.process_time(); started = utc_now()
        status = "PASS"; error = None
        try:
            yield
        except Exception as exc:
            status = "FAIL"; error = f"{type(exc).__name__}: {exc}"; raise
        finally:
            base = {
                "schema_id": TELEMETRY_SPAN_SCHEMA, "category": category, "component": component,
                "started_utc": started, "finished_utc": utc_now(), "wall_seconds": time.perf_counter()-start,
                "process_cpu_seconds": time.process_time()-cpu0, "status": status, "details": details or {}, "error": error,
            }
            rec = dict(base, span_sha256=canonical_sha256(base))
            with self.path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n"); h.flush()
                try: os.fsync(h.fileno())
                except OSError: pass

    def coverage(self) -> dict:
        seen = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip(): seen.add(json.loads(line)["category"])
        return {"required": list(REQUIRED_TELEMETRY_CATEGORIES), "seen": sorted(seen), "missing": [x for x in REQUIRED_TELEMETRY_CATEGORIES if x not in seen], "complete": all(x in seen for x in REQUIRED_TELEMETRY_CATEGORIES)}
