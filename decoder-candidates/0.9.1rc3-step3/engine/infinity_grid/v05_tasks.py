from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .records import utc_now
from .safety import validate_identifier


TASK_JOURNAL_SCHEMA = "IG_DECODER_V05_TASK_JOURNAL_V1_3"
TASK_COMMIT_SCHEMA = "IG_DECODER_V05_LOGICAL_TASK_COMMIT_V1_3"


class V05TaskJournalError(RuntimeError):
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


def prepare_task_state(root: Path, *, scope: dict, uid: int, gid: int) -> dict:
    """Controller-owned preparation of a durable worker task journal.

    The scope file is immutable and root-owned; only the commits directory is
    writable by the scientific worker.  Re-entry requires byte-equivalent scope.
    """
    root = Path(root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    scope_base = dict(scope)
    scope_base["schema_id"] = TASK_JOURNAL_SCHEMA
    scope_base["task_scope_sha256"] = canonical_sha256({k: v for k, v in scope_base.items() if k != "task_scope_sha256"})
    scope_path = root / "scope.json"
    if scope_path.exists():
        old = json.loads(scope_path.read_text(encoding="utf-8"))
        if old != scope_base:
            raise V05TaskJournalError("durable task scope mismatch; refusing stale resume state")
    else:
        write_json_atomic(scope_path, scope_base)
    try:
        os.chmod(scope_path, 0o444)
    except OSError:
        pass
    commits = root / "commits"
    commits.mkdir(parents=True, exist_ok=True)
    try:
        os.chown(commits, int(uid), int(gid))
        os.chmod(commits, 0o700)
        os.chmod(root, 0o755)
    except OSError:
        pass
    _fsync_dir(root)
    return scope_base


@dataclass(frozen=True)
class TaskCommitSummary:
    task_id: str
    task_payload_sha256: str
    commit_sha256: str
    reused: bool


class V05TaskJournal:
    """Worker-side append-only logical-task commit journal.

    A task id may be committed at most once to one immutable payload.  A later
    restart may read/reuse the same commit, but a different payload for the same
    task id is rejected.  This is the unit used to prove no duplicate logical
    commit across crash/resume.
    """

    def __init__(self, root: Path, *, expected_scope_sha256: str, max_tasks: int | None = None):
        self.root = Path(root).resolve(strict=True)
        self.commits = self.root / "commits"
        scope = json.loads((self.root / "scope.json").read_text(encoding="utf-8"))
        if scope.get("schema_id") != TASK_JOURNAL_SCHEMA:
            raise V05TaskJournalError("invalid task journal scope schema")
        observed = canonical_sha256({k: v for k, v in scope.items() if k != "task_scope_sha256"})
        if scope.get("task_scope_sha256") != observed or observed != expected_scope_sha256:
            raise V05TaskJournalError("task journal scope hash mismatch")
        self.scope = scope
        self.max_tasks = None if max_tasks is None else int(max_tasks)
        if self.max_tasks is not None and self.max_tasks < 1:
            raise V05TaskJournalError("max_tasks must be positive")

    def _path(self, task_id: str) -> Path:
        validate_identifier(task_id, field="task_id")
        return self.commits / f"{task_id}.json"

    def load(self, task_id: str) -> dict | None:
        p = self._path(task_id)
        if not p.is_file():
            return None
        raw = p.read_bytes()
        obj = json.loads(raw)
        expected_keys = {
            "schema_id", "task_scope_sha256", "task_id", "task_payload_sha256",
            "payload", "committed_utc", "commit_content_sha256",
        }
        if set(obj) != expected_keys:
            raise V05TaskJournalError(f"invalid task commit shape: {task_id}")
        if obj.get("schema_id") != TASK_COMMIT_SCHEMA or obj.get("task_id") != task_id:
            raise V05TaskJournalError(f"invalid task commit binding: {task_id}")
        if obj.get("task_scope_sha256") != self.scope.get("task_scope_sha256"):
            raise V05TaskJournalError(f"task commit scope mismatch: {task_id}")
        observed_payload_sha = canonical_sha256(obj.get("payload"))
        if obj.get("task_payload_sha256") != observed_payload_sha:
            raise V05TaskJournalError(f"task commit payload hash mismatch: {task_id}")
        content = {k: v for k, v in obj.items() if k != "commit_content_sha256"}
        if canonical_sha256(content) != obj.get("commit_content_sha256"):
            raise V05TaskJournalError(f"task commit content hash mismatch: {task_id}")
        return obj

    def commit(self, task_id: str, payload: Any) -> TaskCommitSummary:
        p = self._path(task_id)
        payload_sha = canonical_sha256(payload)
        existing = self.load(task_id)
        if existing is not None:
            if existing.get("task_payload_sha256") != payload_sha or existing.get("payload") != payload:
                raise V05TaskJournalError(f"duplicate task id with different payload: {task_id}")
            return TaskCommitSummary(
                task_id=task_id,
                task_payload_sha256=payload_sha,
                commit_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                reused=True,
            )
        if self.max_tasks is not None and len(list(self.commits.glob("*.json"))) >= self.max_tasks:
            raise V05TaskJournalError("logical task budget exceeded")
        base = {
            "schema_id": TASK_COMMIT_SCHEMA,
            "task_scope_sha256": self.scope["task_scope_sha256"],
            "task_id": task_id,
            "task_payload_sha256": payload_sha,
            "payload": payload,
            "committed_utc": utc_now(),
        }
        obj = dict(base, commit_content_sha256=canonical_sha256(base))
        tmp = self.commits / f".{task_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        data = canonical_text(obj, pretty=True).encode("utf-8")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as h:
                h.write(data)
                h.flush()
                os.fsync(h.fileno())
            # create-only semantics: if another writer somehow committed first,
            # never overwrite it; verify/reuse through the normal path instead.
            try:
                os.link(tmp, p)
                _fsync_dir(self.commits)
            except FileExistsError:
                pass
        finally:
            tmp.unlink(missing_ok=True)
        final = self.load(task_id)
        if final is None or final.get("task_payload_sha256") != payload_sha or final.get("payload") != payload:
            raise V05TaskJournalError(f"task commit collision/inconsistency: {task_id}")
        # Engineering fault-injection aid.  The controller passes this only to
        # TEST_ONLY workers; production/replay workers never receive it.
        delay = os.environ.get("IG_V05_WORKER_TEST_TASK_DELAY_SECONDS")
        if delay:
            time.sleep(max(0.0, float(delay)))
        return TaskCommitSummary(
            task_id=task_id,
            task_payload_sha256=payload_sha,
            commit_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            reused=False,
        )

    def verified_commits(self) -> list[dict]:
        out = []
        for p in sorted(self.commits.glob("*.json")):
            out.append(self.load(p.stem))
        return [x for x in out if x is not None]

    def summary(self) -> dict:
        commits = self.verified_commits()
        bindings = [
            {
                "task_id": x["task_id"],
                "task_payload_sha256": x["task_payload_sha256"],
                "commit_content_sha256": x["commit_content_sha256"],
            }
            for x in commits
        ]
        return {
            "schema_id": "IG_DECODER_V05_TASK_JOURNAL_SUMMARY_V1_3",
            "task_scope_sha256": self.scope["task_scope_sha256"],
            "committed_tasks": len(bindings),
            "task_bindings": bindings,
            "task_bindings_sha256": canonical_sha256(bindings),
        }
