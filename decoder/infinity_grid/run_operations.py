from __future__ import annotations

"""Crash-safe operational state for long Regime Maturation runs.

This layer is deliberately non-scientific. It never changes Test evidence, Maturation,
Plateau, Lift, or ResearchFrontier semantics. Its job is only to make long runs observable,
resumable, and diagnosable:

* RUN_STATUS.json: one compact authoritative operational snapshot.
* RUN_TIMINGS.csv: lightweight per-test/per-depth timing ledger.
* checkpoints/Oxxxxx/: immutable, hash-verified depth checkpoints.
* strict run identity: decoder source, RunPlan, ResearchFrontier, Regime and Test selection.

Checkpoint commit order is fail-closed:
  write temporary directory -> fsync files -> manifest/hash -> COMPLETE marker -> atomic rename
  -> update RUN_STATUS.json.

Therefore a crash can lose at most the currently uncommitted depth. Existing committed
scientific depth results are never silently recomputed under a different identity.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
import threading
import zipfile
import re

from . import __version__, build_meta
from .canon import canonical_sha256, write_json_atomic
from .records import source_sha256
from .scientific_architecture import TestExecution
from .maturation_auditor import LevelTestRecord


class RunOperationsError(RuntimeError):
    pass


class RunIdentityMismatch(RunOperationsError):
    pass


class CorruptDepthCheckpoint(RunOperationsError):
    pass


class ExternalDurabilityAckRequired(RunOperationsError):
    """A committed depth is waiting for externally verified durable storage."""

    def __init__(self, depth: int, request_path: str | Path):
        self.depth = int(depth)
        self.request_path = str(request_path)
        super().__init__(f"external durability acknowledgement required for O{self.depth:05d}")


RUN_STATUS_SCHEMA = "IG_MATURATION_RUN_STATUS_V1"
CHECKPOINT_SCHEMA = "IG_MATURATION_DEPTH_CHECKPOINT_V1"
RUN_IDENTITY_SCHEMA = "IG_MATURATION_RUN_IDENTITY_V1"
TIMING_COLUMNS = (
    "sequence", "controller_session_id", "depth_attempt_id", "attempt_disposition",
    "depth", "stage", "test_ref", "start_utc", "end_utc",
    "elapsed_seconds", "provider_ref", "scientific_status", "candidate_like",
    "checkpoint_ref", "outcome",
)
CANDIDATE_LIKE_STATUSES = frozenset({"OBSERVED_VARIATION", "CANDIDATE", "COUNTEREXAMPLE", "REOPEN_TRIGGER"})
REQUIRED_CHECKPOINT_PAYLOAD_FILES = frozenset({
    "SOURCE_INPUT.json", "CAPABILITY_SNAPSHOT.json", "LEVEL_RECORD.json",
    "FINDINGS.json", "MATURATION_AUDIT_THROUGH_DEPTH.json",
})
MIRROR_SEGMENT_SCHEMA = "IG_MATURATION_DURABLE_MIRROR_SEGMENT_V1"
MIRROR_LATEST_SCHEMA = "IG_MATURATION_DURABLE_MIRROR_LATEST_V1"
EXTERNAL_REQUEST_SCHEMA = "IG_MATURATION_EXTERNAL_DURABILITY_REQUEST_V1"
EXTERNAL_ACK_SCHEMA = "IG_MATURATION_EXTERNAL_DURABILITY_ACK_V1"
EXTERNAL_REQUIRED_PROVIDERS = ("GOOGLE_DRIVE", "GITHUB")
SAFE_MIRROR_CORE_MEMBERS = frozenset({
    "RUN_IDENTITY.json", "RUN_PLAN.json", "RUN_TIMINGS.csv",
    "O13_O14_PROVIDER_SEAM_CALIBRATION.json", "REVIEW_ACKNOWLEDGEMENTS.jsonl",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        h.write(text)
        h.flush()
        os.fsync(h.fileno())


def _safe_relative_member(rel: str) -> Path:
    """Return a normalized safe relative path or fail closed.

    ZIP/member manifests are external artifacts.  They must never be allowed to escape a
    restore root, even if both the ZIP and its internal manifest are self-consistent.
    """
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        raise RunOperationsError(f"unsafe mirror member path: {rel!r}")
    p = Path(rel)
    if p.is_absolute() or p.drive or any(part in {"", ".", ".."} for part in p.parts):
        raise RunOperationsError(f"unsafe mirror member path: {rel!r}")
    normalized = Path(*p.parts)
    if normalized.as_posix() != rel:
        raise RunOperationsError(f"non-canonical mirror member path: {rel!r}")
    return normalized


def _read_sidecar_sha(path: Path, expected_name: str) -> str:
    try:
        parts = path.read_text(encoding="utf-8").strip().split()
    except Exception as exc:
        raise RunOperationsError(f"cannot read mirror checksum sidecar {path}: {exc}") from exc
    if len(parts) != 2 or parts[1] != expected_name or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise RunOperationsError(f"malformed mirror checksum sidecar: {path}")
    return parts[0]


def _tree_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _process_rss_bytes() -> int | None:
    # Linux current RSS; deterministic enough for an operational ceiling, never a science value.
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _json_read(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RunOperationsError(f"cannot parse {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RunOperationsError(f"expected JSON object: {path}")
    return obj


def _test_execution_to_json(exe: TestExecution) -> dict[str, Any]:
    return {
        "test_ref": exe.test_ref,
        "evidence_payload": exe.evidence_payload,
        "evidence_sha256": exe.evidence_sha256,
        "finding": dict(exe.finding),
        "observation_view": None if exe.observation_view is None else dict(exe.observation_view),
    }


def _test_execution_from_json(obj: Mapping[str, Any]) -> TestExecution:
    return TestExecution(
        test_ref=str(obj["test_ref"]),
        evidence_payload=obj["evidence_payload"],
        evidence_sha256=str(obj["evidence_sha256"]),
        finding=dict(obj["finding"]),
        observation_view=None if obj.get("observation_view") is None else dict(obj["observation_view"]),
    )


def level_record_to_json(record: LevelTestRecord) -> dict[str, Any]:
    obj = {
        "depth": record.depth,
        "provider_ref": record.provider_ref,
        "provider_kind": record.provider_kind,
        "source_science_sha256": record.source_science_sha256,
        "executions": {k: _test_execution_to_json(v) for k, v in sorted(record.executions.items())},
        "stabilization_signature_sha256": record.stabilization_signature_sha256,
        "complexity_projection": dict(record.complexity_projection),
        "complexity_growth_from_previous": record.complexity_growth_from_previous,
        "provider_seam": record.provider_seam,
        "provider_seam_calibrated": record.provider_seam_calibrated,
        "provider_seam_calibration_sha256": record.provider_seam_calibration_sha256,
        "longitudinal_candidate_cohort": None if record.longitudinal_candidate_cohort is None else dict(record.longitudinal_candidate_cohort),
        "longitudinal_discriminator": None if record.longitudinal_discriminator is None else dict(record.longitudinal_discriminator),
        "stabilization_basis": record.stabilization_basis,
    }
    obj["record_sha256"] = canonical_sha256(obj)
    return obj


def level_record_from_json(obj: Mapping[str, Any]) -> LevelTestRecord:
    raw = dict(obj)
    expected = raw.pop("record_sha256", None)
    observed = canonical_sha256(raw)
    if expected != observed:
        raise CorruptDepthCheckpoint(f"LevelTestRecord identity mismatch: {expected} != {observed}")
    return LevelTestRecord(
        depth=int(raw["depth"]),
        provider_ref=str(raw["provider_ref"]),
        provider_kind=str(raw["provider_kind"]),
        source_science_sha256=str(raw["source_science_sha256"]),
        executions={k: _test_execution_from_json(v) for k, v in raw["executions"].items()},
        stabilization_signature_sha256=str(raw["stabilization_signature_sha256"]),
        complexity_projection=dict(raw["complexity_projection"]),
        complexity_growth_from_previous=raw.get("complexity_growth_from_previous"),
        provider_seam=bool(raw["provider_seam"]),
        provider_seam_calibrated=bool(raw.get("provider_seam_calibrated", False)),
        provider_seam_calibration_sha256=raw.get("provider_seam_calibration_sha256"),
        longitudinal_candidate_cohort=None if raw.get("longitudinal_candidate_cohort") is None else dict(raw.get("longitudinal_candidate_cohort")),
        longitudinal_discriminator=None if raw.get("longitudinal_discriminator") is None else dict(raw.get("longitudinal_discriminator")),
        stabilization_basis=str(raw.get("stabilization_basis", "SELECTED_PANEL_TESTPACK")),
    )


def _verify_checkpoint_directory(
    cp: Path,
    *,
    depth: int,
    run_identity_sha256: str,
    run_id: str | None = None,
    expected_previous_checkpoint_sha256: str | None | object = ...,
) -> tuple[dict[str, Any], LevelTestRecord]:
    """Verify a committed depth as a closed, cross-linked transaction."""
    cp = Path(cp)
    manifest_path = cp / "CHECKPOINT_MANIFEST.json"
    complete_path = cp / "COMPLETE.json"
    if not cp.is_dir() or not manifest_path.is_file() or not complete_path.is_file():
        raise CorruptDepthCheckpoint(f"incomplete checkpoint O{depth}")
    manifest = _json_read(manifest_path)
    complete = _json_read(complete_path)
    if manifest.get("schema_id") != CHECKPOINT_SCHEMA:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} schema mismatch")
    if manifest.get("run_identity_sha256") != run_identity_sha256:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} run identity mismatch")
    if run_id is not None and manifest.get("run_id") != run_id:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} run_id mismatch")
    if int(manifest.get("depth", -1)) != int(depth):
        raise CorruptDepthCheckpoint(f"checkpoint depth mismatch at O{depth}")
    observed_content = canonical_sha256({k: v for k, v in manifest.items() if k != "checkpoint_content_sha256"})
    if manifest.get("checkpoint_content_sha256") != observed_content:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} manifest content identity mismatch")
    if complete.get("schema_id") != "IG_MATURATION_DEPTH_CHECKPOINT_COMPLETE_V1":
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} COMPLETE schema mismatch")
    if int(complete.get("depth", -1)) != int(depth):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} COMPLETE depth mismatch")
    if complete.get("manifest_file_sha256") != _sha_file(manifest_path):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} COMPLETE marker does not pin manifest")
    if complete.get("checkpoint_content_sha256") != manifest.get("checkpoint_content_sha256"):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} COMPLETE marker does not pin checkpoint content")
    if expected_previous_checkpoint_sha256 is not ... and manifest.get("previous_checkpoint_content_sha256") != expected_previous_checkpoint_sha256:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} predecessor pointer mismatch")

    file_records = manifest.get("files")
    if not isinstance(file_records, list):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} manifest files is not a list")
    declared: list[str] = []
    for rec in file_records:
        if not isinstance(rec, dict):
            raise CorruptDepthCheckpoint(f"checkpoint O{depth} malformed file record")
        rel = rec.get("path")
        if not isinstance(rel, str) or not rel or Path(rel).name != rel or "/" in rel or "\\" in rel:
            raise CorruptDepthCheckpoint(f"checkpoint O{depth} unsafe file path: {rel!r}")
        declared.append(rel)
    if set(declared) != set(REQUIRED_CHECKPOINT_PAYLOAD_FILES) or len(declared) != len(set(declared)):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} payload allowlist mismatch: {declared}")
    allowed_entries = set(REQUIRED_CHECKPOINT_PAYLOAD_FILES) | {"CHECKPOINT_MANIFEST.json", "COMPLETE.json"}
    observed_entries = {x.name for x in cp.iterdir()}
    if observed_entries != allowed_entries:
        raise CorruptDepthCheckpoint(
            f"checkpoint O{depth} undeclared/missing entries: extra={sorted(observed_entries-allowed_entries)} missing={sorted(allowed_entries-observed_entries)}"
        )
    for rec in file_records:
        fp = cp / rec["path"]
        try:
            expected_size = int(rec["size_bytes"])
            expected_sha = str(rec["sha256"])
        except Exception as exc:
            raise CorruptDepthCheckpoint(f"checkpoint O{depth} malformed file metadata") from exc
        if not fp.is_file() or fp.stat().st_size != expected_size or _sha_file(fp) != expected_sha:
            raise CorruptDepthCheckpoint(f"checkpoint O{depth} file verification failed: {rec['path']}")

    level_obj = _json_read(cp / "LEVEL_RECORD.json")
    record = level_record_from_json(level_obj)
    if not record.executions:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} has no scientific executions")
    for tref, exe in record.executions.items():
        if exe.evidence_sha256 != canonical_sha256(exe.evidence_payload):
            raise CorruptDepthCheckpoint(f"checkpoint O{depth} evidence payload hash mismatch: {tref}")
        finding_ref = exe.finding.get("evidence_ref") if isinstance(exe.finding, Mapping) else None
        if isinstance(finding_ref, Mapping) and finding_ref.get("sha256") not in {None, exe.evidence_sha256}:
            raise CorruptDepthCheckpoint(f"checkpoint O{depth} finding/evidence cross-link mismatch: {tref}")
    if record.depth != depth:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} LevelTestRecord depth mismatch")
    if manifest.get("level_record_sha256") != level_obj.get("record_sha256"):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} manifest level_record_sha256 mismatch")
    if manifest.get("source_science_sha256") != record.source_science_sha256:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} manifest source_science_sha256 mismatch")
    if manifest.get("provider_ref") != record.provider_ref or manifest.get("provider_kind") != record.provider_kind:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} manifest provider metadata mismatch")

    snapshot = _json_read(cp / "CAPABILITY_SNAPSHOT.json")
    if int(snapshot.get("regime_depth", -1)) != depth:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} capability snapshot depth mismatch")
    if snapshot.get("provider_ref") != record.provider_ref or snapshot.get("provider_kind") != record.provider_kind:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} capability snapshot provider mismatch")
    if snapshot.get("source_science_sha256") != record.source_science_sha256:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} capability snapshot source mismatch")
    snapshot_sha = snapshot.get("snapshot_science_sha256")
    if snapshot_sha != canonical_sha256({k: v for k, v in snapshot.items() if k != "snapshot_science_sha256"}):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} capability snapshot identity mismatch")

    findings = _json_read(cp / "FINDINGS.json")
    expected_findings = {k: dict(v.finding) for k, v in sorted(record.executions.items())}
    if findings != expected_findings:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} findings do not match LevelTestRecord")

    maturation = _json_read(cp / "MATURATION_AUDIT_THROUGH_DEPTH.json")
    mat_sha = maturation.get("science_sha256")
    if mat_sha is not None and mat_sha != canonical_sha256({k: v for k, v in maturation.items() if k != "science_sha256"}):
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} maturation audit identity mismatch")
    mat_record = maturation.get("maturation_record")
    if not isinstance(mat_record, Mapping) or int(mat_record.get("depth_end", -1)) != depth:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} maturation audit depth_end mismatch")
    gate = maturation.get("review_gate")
    if not isinstance(gate, Mapping) or int(gate.get("evaluated_through_depth", -1)) != depth:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} review gate depth mismatch")

    source_input = _json_read(cp / "SOURCE_INPUT.json")
    source_sha = source_input.get("science_sha256")
    if source_sha is not None and source_sha != record.source_science_sha256:
        raise CorruptDepthCheckpoint(f"checkpoint O{depth} source input science pointer mismatch")
    return manifest, record


def _finding_counts(records: Iterable[LevelTestRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        for exe in rec.executions.values():
            status = str(exe.finding.get("scientific_status", "UNKNOWN"))
            out[status] = out.get(status, 0) + 1
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class RunIdentity:
    payload: Mapping[str, Any]
    science_sha256: str


class RunOperations:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        plan: Mapping[str, Any],
        plan_science_sha256: str,
        frontier_science_sha256: str,
        entity_class_ref: str,
        regime_ref: str,
        selected_test_refs: Iterable[str],
        start_depth: int,
        max_depth: int,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.plan = dict(plan)
        self.plan_sha = str(plan_science_sha256)
        self.frontier_sha = str(frontier_science_sha256)
        self.entity_class_ref = str(entity_class_ref)
        self.regime_ref = str(regime_ref)
        self.selected_test_refs = tuple(selected_test_refs)
        self.start_depth = int(start_depth)
        self.max_depth = int(max_depth)
        operational = dict(self.plan.get("operational_integrity") or {})
        self.external_ack_required = bool(operational.get("external_ack_required", False))
        providers = operational.get("external_required_providers", list(EXTERNAL_REQUIRED_PROVIDERS))
        self.external_required_providers = tuple(sorted(str(x) for x in providers))
        if self.external_ack_required and tuple(self.external_required_providers) != tuple(sorted(EXTERNAL_REQUIRED_PROVIDERS)):
            raise RunOperationsError(
                "certified external durability currently requires GOOGLE_DRIVE and GITHUB acknowledgements"
            )
        self.resource_budget = dict(operational.get("resource_budget") or {})
        self.status_path = self.run_dir / "RUN_STATUS.json"
        self.plan_path = self.run_dir / "RUN_PLAN.json"
        self.identity_path = self.run_dir / "RUN_IDENTITY.json"
        self.timings_path = self.run_dir / "RUN_TIMINGS.csv"
        self.checkpoint_root = self.run_dir / "checkpoints"
        mirror_override = os.environ.get("IG_DECODER_DURABLE_MIRROR_ROOT")
        if mirror_override:
            self.mirror_root = Path(mirror_override) / self.run_id
        else:
            self.mirror_root = self.run_dir.parent / f"{self.run_dir.name}_durable_mirror"
        try:
            if self.mirror_root.resolve().is_relative_to(self.run_dir.resolve()):
                raise RunOperationsError("durable mirror root must be outside the live run directory")
        except AttributeError:  # pragma: no cover
            pass
        self.mirror_latest_path = self.mirror_root / "LATEST.json"
        self.external_root = self.run_dir / "external_durability"
        self.external_request_root = self.external_root / "requests"
        self.external_ack_root = self.external_root / "acks"
        self.started_perf: float | None = None
        self._test_starts: dict[str, tuple[float, str]] = {}
        self._mutation_lock = threading.RLock()
        self.controller_lock_payload: dict[str, Any] | None = None
        self.controller_session_id: str | None = None
        self._active_attempts: dict[int, str] = {}
        self._checkpoint_cache: dict[int, tuple[dict[str, Any], tuple[int, int], tuple[int, int]]] = {}
        self._verified_depths: list[int] | None = None
        # v0.28.5: status/progress telemetry must be O(1) in historical depth.
        # These caches are seeded only after the explicit startup full-chain audit.
        self._status_records_cache: list[LevelTestRecord] | None = None
        self._external_ack_status_cache: tuple[int | None, list[int]] | None = None
        self._verified_source_sha256: str | None = None
        self._expected_identity_cache: RunIdentity | None = None
        self._run_started_perf = time.perf_counter()

    def bind_controller_lock(self, payload: Mapping[str, Any] | None) -> None:
        self.controller_lock_payload = None if payload is None else dict(payload)
        self.controller_session_id = None if payload is None else str(payload.get("lock_token") or "") or None

    def expected_identity(self) -> RunIdentity:
        # Mandatory live verification happens at startup and again before every commit.
        # Progress/status calls reuse that verified identity so telemetry never re-hashes
        # the entire executable source tree.
        if self._expected_identity_cache is not None:
            return self._expected_identity_cache
        live_source_sha = self._verified_source_sha256 or source_sha256(verify_build=True)
        self._verified_source_sha256 = live_source_sha
        payload = {
            "schema_id": RUN_IDENTITY_SCHEMA,
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "decoder_version": __version__,
            "decoder_source_sha256": live_source_sha,
            "plan_science_sha256": self.plan_sha,
            "frontier_science_sha256": self.frontier_sha,
            "entity_class_ref": self.entity_class_ref,
            "regime_ref": self.regime_ref,
            "selected_test_refs": list(self.selected_test_refs),
            "start_depth": self.start_depth,
            "max_depth": self.max_depth,
            "checkpoint_semantics": "COMMIT_AFTER_HASH_VERIFY_ATOMIC_RENAME",
            "resume_semantics": "STRICT_IDENTITY_MATCH_FAIL_CLOSED",
            "external_durability": {
                "required": self.external_ack_required,
                "required_providers": list(self.external_required_providers),
                "ack_schema": EXTERNAL_ACK_SCHEMA,
                "advance_gate": "NO_NEXT_DEPTH_BEFORE_VERIFIED_EXTERNAL_ACK" if self.external_ack_required else "NOT_REQUIRED",
            },
            "resource_budget": dict(self.resource_budget),
        }
        ident = RunIdentity(payload=payload, science_sha256=canonical_sha256(payload))
        self._expected_identity_cache = ident
        return ident

    def _verify_source_unchanged(self) -> str:
        live = source_sha256(verify_build=True)
        if self._verified_source_sha256 is not None and live != self._verified_source_sha256:
            raise RunOperationsError(
                f"executable source changed during run: expected {self._verified_source_sha256}, observed {live}"
            )
        self._verified_source_sha256 = live
        return live

    def initialize_or_verify(self) -> dict[str, Any]:
        # Verify executable source before creating or mutating a run directory.
        self._verify_source_unchanged()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        expected = self.expected_identity()
        if self.identity_path.exists():
            observed = _json_read(self.identity_path)
            got_sha = observed.get("run_identity_sha256")
            raw = {k: v for k, v in observed.items() if k != "run_identity_sha256"}
            if got_sha != canonical_sha256(raw):
                raise RunIdentityMismatch("stored RUN_IDENTITY.json self-hash mismatch")
            if raw != dict(expected.payload):
                raise RunIdentityMismatch("run identity changed; allocate a new run directory/run_id")
            if not self.plan_path.exists() or canonical_sha256(_json_read(self.plan_path)) != self.plan_sha:
                raise RunIdentityMismatch("stored RUN_PLAN.json does not match frozen plan identity")
        else:
            if any(self.run_dir.iterdir()) and not all(p.name in {"checkpoints", "controller.lock", "lock_recovery", "CONTINUATION_SEED.json"} for p in self.run_dir.iterdir()):
                raise RunOperationsError("run directory is nonempty without RUN_IDENTITY.json")
            write_json_atomic(self.plan_path, self.plan)
            obj = dict(expected.payload)
            obj["run_identity_sha256"] = expected.science_sha256
            write_json_atomic(self.identity_path, obj)
            self._init_timing_ledger()
        self._init_timing_ledger()
        # Mirror catch-up runs before any new scientific depth can execute.  Thus a crash after
        # checkpoint commit but before mirror creation cannot silently leave the next resume
        # advancing with only one copy.
        self.full_checkpoint_audit()
        self.ensure_durable_mirror()
        self._write_missing_external_requests()
        return dict(expected.payload) | {"run_identity_sha256": expected.science_sha256}

    def _init_timing_ledger(self) -> None:
        with self._mutation_lock:
            if self.timings_path.exists():
                try:
                    first = self.timings_path.open("r", encoding="utf-8", newline="").readline().strip()
                except Exception as exc:
                    raise RunOperationsError(f"cannot read timing ledger header: {exc}") from exc
                if first != ",".join(TIMING_COLUMNS):
                    raise RunOperationsError("timing ledger schema/header mismatch for this Decoder release")
                return
            _write_text_fsync(self.timings_path, ",".join(TIMING_COLUMNS) + "\n")
            _fsync_dir(self.timings_path.parent)

    def _append_timing(self, row: Mapping[str, Any]) -> None:
        with self._mutation_lock:
            self._init_timing_ledger()
            seq = sum(1 for _ in self.timings_path.open("r", encoding="utf-8"))
            values = {k: row.get(k, "") for k in TIMING_COLUMNS}
            values["sequence"] = seq
            values["controller_session_id"] = row.get("controller_session_id", self.controller_session_id or "")
            depth_value = row.get("depth")
            if depth_value not in (None, ""):
                try:
                    attempt = self._active_attempts.get(int(depth_value))
                except Exception:
                    attempt = None
                values["depth_attempt_id"] = row.get("depth_attempt_id", attempt or "")
            with self.timings_path.open("a", encoding="utf-8", newline="") as h:
                w = csv.DictWriter(h, fieldnames=TIMING_COLUMNS)
                w.writerow(values)
                h.flush()
                os.fsync(h.fileno())

    def begin_depth_attempt(self, depth: int) -> str:
        depth = int(depth)
        if depth in self._active_attempts:
            raise RunOperationsError(f"depth O{depth:05d} already has an active attempt")
        attempt = f"{self.controller_session_id or 'sessionless'}-O{depth:05d}-{uuid.uuid4().hex}"
        self._active_attempts[depth] = attempt
        self._append_timing({
            "depth": depth,
            "stage": "DEPTH_ATTEMPT",
            "test_ref": "",
            "start_utc": _utc_now(),
            "end_utc": "",
            "elapsed_seconds": "",
            "provider_ref": "",
            "scientific_status": "",
            "candidate_like": "",
            "checkpoint_ref": "",
            "outcome": "STARTED",
            "attempt_disposition": "IN_PROGRESS",
            "depth_attempt_id": attempt,
        })
        return attempt

    def finish_depth_attempt(self, depth: int, disposition: str) -> None:
        depth = int(depth)
        attempt = self._active_attempts.pop(depth, None)
        if attempt is None:
            return
        self._append_timing({
            "depth": depth,
            "stage": "DEPTH_ATTEMPT",
            "test_ref": "",
            "start_utc": "",
            "end_utc": _utc_now(),
            "elapsed_seconds": "",
            "provider_ref": "",
            "scientific_status": "",
            "candidate_like": "",
            "checkpoint_ref": self.checkpoint_path(depth).name if self.checkpoint_path(depth).exists() else "",
            "outcome": disposition,
            "attempt_disposition": disposition,
            "depth_attempt_id": attempt,
        })

    def abort_active_attempts(self) -> None:
        for depth in list(self._active_attempts):
            self.finish_depth_attempt(depth, "ABORTED")

    def check_resource_budget(self, *, depth: int | None = None, depth_elapsed_seconds: float | None = None) -> dict[str, Any]:
        if not self.resource_budget:
            return {"status": "NOT_CONFIGURED"}
        observed = {
            "wall_seconds_total": max(0.0, time.perf_counter() - self._run_started_perf),
            "run_dir_bytes": _tree_size_bytes(self.run_dir),
            "mirror_dir_bytes": _tree_size_bytes(self.mirror_root),
            "rss_bytes": _process_rss_bytes(),
            "depth": depth,
            "depth_elapsed_seconds": depth_elapsed_seconds,
        }
        limits = self.resource_budget
        no_deadline = os.environ.get('IG_DECODER_EXECUTION_POLICY') == 'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
        checks = (
            ("max_wall_seconds_total", observed["wall_seconds_total"]),
            ("max_run_dir_bytes", observed["run_dir_bytes"]),
            ("max_mirror_dir_bytes", observed["mirror_dir_bytes"]),
            ("max_rss_bytes", observed["rss_bytes"]),
            ("max_wall_seconds_per_depth", depth_elapsed_seconds),
        )
        for key, value in checks:
            if no_deadline and key in {'max_wall_seconds_total','max_wall_seconds_per_depth'}:
                continue
            limit = limits.get(key)
            if limit is None or value is None:
                continue
            if float(value) > float(limit):
                raise RunOperationsError(f"resource budget exceeded: {key} observed={value} limit={limit}")
        return {"status": "PASS", "limits": dict(limits), "observed": observed}

    def _mirror_segment_path(self, depth: int) -> Path:
        return self.mirror_root / "segments" / f"O{int(depth):05d}.zip"

    def _verify_mirror_segment(self, depth: int) -> dict[str, Any]:
        depth = int(depth)
        target = self._mirror_segment_path(depth)
        sha_path = target.with_suffix(target.suffix + ".sha256.txt")
        if not target.is_file() or not sha_path.is_file():
            raise RunOperationsError(f"durable mirror O{depth} missing ZIP or checksum sidecar")
        actual_sha = _sha_file(target)
        if _read_sidecar_sha(sha_path, target.name) != actual_sha:
            raise RunOperationsError(f"durable mirror O{depth} checksum sidecar mismatch")
        with zipfile.ZipFile(target, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RunOperationsError(f"durable mirror ZIP CRC failure at {bad}")
            names = zf.namelist()
            if names.count("MIRROR_SEGMENT_MANIFEST.json") != 1:
                raise RunOperationsError(f"durable mirror O{depth} has invalid manifest multiplicity")
            sm = json.loads(zf.read("MIRROR_SEGMENT_MANIFEST.json"))
            raw = dict(sm); expected = raw.pop("segment_content_sha256", None)
            if expected != canonical_sha256(raw):
                raise RunOperationsError(f"durable mirror O{depth} manifest identity mismatch")
            if int(sm.get("depth", -1)) != depth or target.name != f"O{depth:05d}.zip":
                raise RunOperationsError(f"durable mirror O{depth} filename/depth mismatch")
            declared = {}
            for rec in sm.get("files", []):
                rel = str(rec.get("path", ""))
                _safe_relative_member(rel)
                if rel in declared:
                    raise RunOperationsError(f"durable mirror O{depth} duplicate declared member: {rel}")
                declared[rel] = rec
            expected_names = set(declared) | {"MIRROR_SEGMENT_MANIFEST.json"}
            if set(names) != expected_names or len(names) != len(set(names)):
                raise RunOperationsError(f"durable mirror O{depth} ZIP member allowlist mismatch")
            for rel, rec in declared.items():
                data = zf.read(rel)
                if hashlib.sha256(data).hexdigest() != rec.get("sha256") or len(data) != int(rec.get("size_bytes", -1)):
                    raise RunOperationsError(f"durable mirror O{depth} member hash/size mismatch: {rel}")
        cp = self.verify_checkpoint(depth)
        if sm.get("run_identity_sha256") != self.expected_identity().science_sha256:
            raise RunOperationsError(f"durable mirror O{depth} run identity mismatch")
        if sm.get("checkpoint_content_sha256") != cp.get("checkpoint_content_sha256"):
            raise RunOperationsError(f"durable mirror O{depth} checkpoint binding mismatch")
        return sm

    def _verify_latest(self, *, allow_missing: bool = False) -> dict[str, Any] | None:
        if not self.mirror_latest_path.is_file():
            if allow_missing:
                return None
            raise RunOperationsError("durable mirror LATEST.json missing")
        latest = _json_read(self.mirror_latest_path)
        raw = dict(latest); expected = raw.pop("latest_content_sha256", None)
        if expected != canonical_sha256(raw):
            raise RunOperationsError("durable mirror LATEST.json self-hash mismatch")
        if latest.get("run_id") != self.run_id or latest.get("run_identity_sha256") != self.expected_identity().science_sha256:
            raise RunOperationsError("durable mirror LATEST.json run identity mismatch")
        depth = int(latest.get("latest_mirrored_depth", -1))
        target = self._mirror_segment_path(depth)
        if latest.get("latest_segment") != str(target.relative_to(self.mirror_root)):
            raise RunOperationsError("durable mirror LATEST.json segment path mismatch")
        if not target.is_file() or latest.get("latest_segment_sha256") != _sha_file(target):
            raise RunOperationsError("durable mirror LATEST.json segment hash mismatch")
        sm = self._verify_mirror_segment(depth)
        if latest.get("checkpoint_content_sha256") != sm.get("checkpoint_content_sha256"):
            raise RunOperationsError("durable mirror LATEST.json checkpoint binding mismatch")
        return latest

    def _write_durable_mirror_segment(self, depth: int) -> Path:
        """Write one immutable compact mirror segment for a committed depth.

        The segment lives outside the live run directory and contains the exact checkpoint plus
        the frozen run identity/plan and the timing ledger as of the commit.  A failed mirror
        write is operationally fatal: the scientific checkpoint remains committed, but the
        controller must stop rather than advance without the promised second durable copy.
        """
        depth = int(depth)
        manifest = self.verify_checkpoint(depth)
        self.mirror_root.mkdir(parents=True, exist_ok=True)
        segdir = self.mirror_root / "segments"
        segdir.mkdir(parents=True, exist_ok=True)
        target = self._mirror_segment_path(depth)
        sha_path = target.with_suffix(target.suffix + ".sha256.txt")
        checkpoint = self.checkpoint_path(depth)
        core_files = [self.identity_path, self.plan_path, self.timings_path]
        continuation = self.run_dir / "CONTINUATION_SEED.json"
        if continuation.is_file():
            core_files.append(continuation)
        seam = self.run_dir / "O13_O14_PROVIDER_SEAM_CALIBRATION.json"
        if seam.is_file():
            core_files.append(seam)
        ack = self.run_dir / "REVIEW_ACKNOWLEDGEMENTS.jsonl"
        if ack.is_file():
            core_files.append(ack)
        file_rows: list[dict[str, Any]] = []
        zip_members: list[tuple[Path, str]] = []
        for src in core_files:
            if not src.is_file():
                raise RunOperationsError(f"durable mirror missing required live file: {src}")
            arc = src.name
            zip_members.append((src, arc))
            file_rows.append({"path": arc, "sha256": _sha_file(src), "size_bytes": src.stat().st_size})
        for src in sorted(x for x in checkpoint.iterdir() if x.is_file()):
            arc = f"checkpoints/{checkpoint.name}/{src.name}"
            zip_members.append((src, arc))
            file_rows.append({"path": arc, "sha256": _sha_file(src), "size_bytes": src.stat().st_size})
        seg_manifest = {
            "schema_id": MIRROR_SEGMENT_SCHEMA,
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "run_identity_sha256": self.expected_identity().science_sha256,
            "depth": depth,
            "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
            "previous_checkpoint_content_sha256": manifest.get("previous_checkpoint_content_sha256"),
            "created_utc": _utc_now(),
            "files": sorted(file_rows, key=lambda r: r["path"]),
            "restore_semantics": "EXTRACT_SEGMENTS_IN_ASCENDING_DEPTH_OVER_SAME_RUN_ROOT_THEN_STRICT_VERIFY",
        }
        seg_manifest["segment_content_sha256"] = canonical_sha256(seg_manifest)
        tmp = segdir / f".{target.name}.tmp-{uuid.uuid4().hex}"
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
                for src, arc in sorted(zip_members, key=lambda x: x[1]):
                    zf.write(src, arc)
                zf.writestr("MIRROR_SEGMENT_MANIFEST.json", json.dumps(seg_manifest, sort_keys=True, separators=(",", ":")) + "\n")
            with zipfile.ZipFile(tmp, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise RunOperationsError(f"durable mirror ZIP CRC failure at {bad}")
                observed = json.loads(zf.read("MIRROR_SEGMENT_MANIFEST.json"))
                expected_sha = observed.pop("segment_content_sha256", None)
                if expected_sha != canonical_sha256(observed):
                    raise RunOperationsError("durable mirror segment manifest self-hash mismatch")
            observed_sha = _sha_file(tmp)
            if target.exists():
                if _sha_file(target) != observed_sha:
                    raise RunOperationsError(f"durable mirror segment collision at O{depth}")
                tmp.unlink()
            else:
                tmp.replace(target)
                _fsync_dir(segdir)
            _write_text_fsync(sha_path, f"{_sha_file(target)}  {target.name}\n")
            self._verify_mirror_segment(depth)
            prior_latest = self._verify_latest(allow_missing=True)
            if prior_latest is not None and int(prior_latest["latest_mirrored_depth"]) > depth:
                raise RunOperationsError("durable mirror LATEST would move backwards")
            latest = {
                "schema_id": MIRROR_LATEST_SCHEMA,
                "schema_version": "1.0.0",
                "run_id": self.run_id,
                "run_identity_sha256": self.expected_identity().science_sha256,
                "latest_mirrored_depth": depth,
                "latest_segment": str(target.relative_to(self.mirror_root)),
                "latest_segment_sha256": _sha_file(target),
                "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
                "updated_utc": _utc_now(),
            }
            latest["latest_content_sha256"] = canonical_sha256(latest)
            write_json_atomic(self.mirror_latest_path, latest)
            _fsync_dir(self.mirror_root)
            self._verify_latest()
        except Exception:
            if tmp.exists():
                # Preserve failed mirror temp for forensic inspection, mirroring checkpoint policy.
                pass
            raise
        return target

    def ensure_durable_mirror(self) -> list[int]:
        mirrored: list[int] = []
        for depth in self.valid_checkpoint_depths():
            target = self._mirror_segment_path(depth)
            if not target.is_file():
                self._write_durable_mirror_segment(depth)
            else:
                self._verify_mirror_segment(depth)
            mirrored.append(depth)
        if mirrored:
            depth = mirrored[-1]
            target = self._mirror_segment_path(depth)
            cp = self.verify_checkpoint(depth)
            latest = {
                "schema_id": MIRROR_LATEST_SCHEMA,
                "schema_version": "1.0.0",
                "run_id": self.run_id,
                "run_identity_sha256": self.expected_identity().science_sha256,
                "latest_mirrored_depth": depth,
                "latest_segment": str(target.relative_to(self.mirror_root)),
                "latest_segment_sha256": _sha_file(target),
                "checkpoint_content_sha256": cp["checkpoint_content_sha256"],
                "updated_utc": _utc_now(),
            }
            latest["latest_content_sha256"] = canonical_sha256(latest)
            prior_latest = self._verify_latest(allow_missing=True)
            if prior_latest is not None and int(prior_latest["latest_mirrored_depth"]) > depth:
                raise RunOperationsError("durable mirror LATEST would move backwards")
            write_json_atomic(self.mirror_latest_path, latest)
            _fsync_dir(self.mirror_root)
            self._verify_latest()
        return mirrored

    def external_request_path(self, depth: int) -> Path:
        return self.external_request_root / f"O{int(depth):05d}.json"

    def external_ack_path(self, depth: int) -> Path:
        return self.external_ack_root / f"O{int(depth):05d}.json"

    def _external_request_payload(self, depth: int) -> dict[str, Any]:
        depth = int(depth)
        cp = self.verify_checkpoint(depth)
        sm = self._verify_mirror_segment(depth)
        segment = self._mirror_segment_path(depth)
        obj = {
            "schema_id": EXTERNAL_REQUEST_SCHEMA,
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "run_identity_sha256": self.expected_identity().science_sha256,
            "depth": depth,
            "checkpoint_content_sha256": cp["checkpoint_content_sha256"],
            "segment_filename": segment.name,
            "segment_sha256": _sha_file(segment),
            "segment_size_bytes": segment.stat().st_size,
            "segment_content_sha256": sm["segment_content_sha256"],
            "required_providers": list(self.external_required_providers),
            "required_protocol": (
                "UPLOAD_SEGMENT_TO_GOOGLE_DRIVE_READBACK_HASH_VERIFY_THEN_APPEND_CONTENT_ADDRESSED_GITHUB_MANIFEST"
            ),
            "created_utc": _utc_now(),
        }
        obj["request_sha256"] = canonical_sha256(obj)
        return obj

    def write_external_request(self, depth: int) -> dict[str, Any]:
        depth = int(depth)
        self.external_request_root.mkdir(parents=True, exist_ok=True)
        path = self.external_request_path(depth)
        expected = self._external_request_payload(depth)
        if path.is_file():
            observed = _json_read(path)
            raw = dict(observed); got = raw.pop("request_sha256", None)
            if got != canonical_sha256(raw):
                raise RunOperationsError(f"external durability request O{depth} self-hash mismatch")
            # created_utc is intentionally stable once written. Compare all identity-bearing fields.
            for key in (
                "schema_id", "schema_version", "run_id", "run_identity_sha256", "depth",
                "checkpoint_content_sha256", "segment_filename", "segment_sha256",
                "segment_size_bytes", "segment_content_sha256", "required_providers",
                "required_protocol",
            ):
                if observed.get(key) != expected.get(key):
                    raise RunOperationsError(f"external durability request O{depth} binding mismatch: {key}")
            return observed
        write_json_atomic(path, expected)
        _fsync_dir(self.external_request_root)
        return expected

    def _write_missing_external_requests(self) -> list[int]:
        if not self.external_ack_required:
            return []
        out = []
        for depth in self.valid_checkpoint_depths():
            self.write_external_request(depth)
            out.append(depth)
        return out

    def verify_external_ack(self, depth: int) -> dict[str, Any]:
        depth = int(depth)
        path = self.external_ack_path(depth)
        if not path.is_file():
            raise ExternalDurabilityAckRequired(depth, self.external_request_path(depth))
        ack = _json_read(path)
        raw = dict(ack); got = raw.pop("ack_sha256", None)
        if got != canonical_sha256(raw):
            raise RunOperationsError(f"external durability ack O{depth} self-hash mismatch")
        req = self.write_external_request(depth)
        checks = {
            "schema_id": EXTERNAL_ACK_SCHEMA,
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "run_identity_sha256": self.expected_identity().science_sha256,
            "depth": depth,
            "request_sha256": req["request_sha256"],
            "checkpoint_content_sha256": req["checkpoint_content_sha256"],
            "segment_sha256": req["segment_sha256"],
        }
        for key, expected in checks.items():
            if ack.get(key) != expected:
                raise RunOperationsError(f"external durability ack O{depth} binding mismatch: {key}")
        providers = ack.get("providers")
        if not isinstance(providers, list):
            raise RunOperationsError(f"external durability ack O{depth} providers must be a list")
        by_name = {str(x.get("provider")): x for x in providers if isinstance(x, Mapping)}
        if sorted(by_name) != list(self.external_required_providers):
            raise RunOperationsError(f"external durability ack O{depth} provider set mismatch")
        drive = by_name.get("GOOGLE_DRIVE", {})
        if not drive.get("file_id") or drive.get("artifact_sha256") != req["segment_sha256"] or drive.get("readback_verified") is not True:
            raise RunOperationsError(f"external durability ack O{depth} Drive proof incomplete")
        github = by_name.get("GITHUB", {})
        required_github = ("repository", "branch", "path", "commit_sha", "manifest_sha256")
        if any(not github.get(k) for k in required_github):
            raise RunOperationsError(f"external durability ack O{depth} GitHub proof incomplete")
        if github.get("drive_file_id") != drive.get("file_id") or github.get("artifact_sha256") != req["segment_sha256"]:
            raise RunOperationsError(f"external durability ack O{depth} GitHub/Drive cross-binding mismatch")
        return ack

    def record_external_ack(self, depth: int, receipt: Mapping[str, Any]) -> dict[str, Any]:
        """Store a connector-produced external durability receipt after validating its bindings."""
        depth = int(depth)
        req = self.write_external_request(depth)
        obj = dict(receipt)
        obj.setdefault("schema_id", EXTERNAL_ACK_SCHEMA)
        obj.setdefault("schema_version", "1.0.0")
        obj.setdefault("run_id", self.run_id)
        obj.setdefault("run_identity_sha256", self.expected_identity().science_sha256)
        obj.setdefault("depth", depth)
        obj.setdefault("request_sha256", req["request_sha256"])
        obj.setdefault("checkpoint_content_sha256", req["checkpoint_content_sha256"])
        obj.setdefault("segment_sha256", req["segment_sha256"])
        obj.setdefault("acknowledged_utc", _utc_now())
        obj.pop("ack_sha256", None)
        obj["ack_sha256"] = canonical_sha256(obj)
        self.external_ack_root.mkdir(parents=True, exist_ok=True)
        path = self.external_ack_path(depth)
        if path.exists():
            existing = _json_read(path)
            if existing != obj:
                # Immutable receipt: never rewrite history in place.
                raise RunOperationsError(f"external durability ack O{depth} already exists with different content")
        else:
            write_json_atomic(path, obj)
            _fsync_dir(self.external_ack_root)
        verified = self.verify_external_ack(depth)
        if self._external_ack_status_cache is not None:
            _, pending = self._external_ack_status_cache
            pending = [d for d in pending if d != depth]
            completed = [] if self._status_records_cache is None else [r.depth for r in self._status_records_cache]
            latest = depth if (not pending and completed and depth == completed[-1]) else self._external_ack_status_cache[0]
            self._external_ack_status_cache = (latest, pending)
        return verified

    def pending_external_ack_depths(self) -> list[int]:
        if not self.external_ack_required:
            self._external_ack_status_cache = (None, [])
            return []
        pending = []
        latest = None
        for depth in self.valid_checkpoint_depths():
            self.write_external_request(depth)
            try:
                self.verify_external_ack(depth)
                if not pending:
                    latest = depth
            except ExternalDurabilityAckRequired:
                pending.append(depth)
        self._external_ack_status_cache = (latest, list(pending))
        return pending

    def latest_external_ack_depth(self) -> int | None:
        if not self.external_ack_required:
            return None
        if self._external_ack_status_cache is None:
            self.pending_external_ack_depths()
        assert self._external_ack_status_cache is not None
        return self._external_ack_status_cache[0]

    def valid_checkpoint_depths(self) -> list[int]:
        depths: list[int] = []
        if not self.checkpoint_root.exists():
            return depths
        for p in sorted(self.checkpoint_root.glob("O[0-9][0-9][0-9][0-9][0-9]")):
            depth = int(p.name[1:])
            depths.append(depth)
        expected = list(range(self.start_depth, (max(depths) if depths else self.start_depth - 1) + 1))
        if depths != expected:
            raise CorruptDepthCheckpoint(f"non-contiguous committed checkpoint sequence: {depths}")
        expected_prev = None
        for depth in depths:
            manifest = self.verify_checkpoint(depth, expected_previous=expected_prev)
            expected_prev = manifest["checkpoint_content_sha256"]
        self._verified_depths = list(depths)
        return depths

    def checkpoint_path(self, depth: int) -> Path:
        return self.checkpoint_root / f"O{int(depth):05d}"

    def _checkpoint_stat_key(self, depth: int) -> tuple[tuple[int, int], tuple[int, int]]:
        cp = self.checkpoint_path(depth)
        m = (cp / "CHECKPOINT_MANIFEST.json").stat()
        c = (cp / "COMPLETE.json").stat()
        return (int(m.st_size), int(m.st_mtime_ns)), (int(c.st_size), int(c.st_mtime_ns))

    def verify_checkpoint(self, depth: int, *, expected_previous: str | None | object = ...) -> dict[str, Any]:
        depth = int(depth)
        try:
            stats = self._checkpoint_stat_key(depth)
        except OSError as exc:
            raise CorruptDepthCheckpoint(f"incomplete checkpoint O{depth}") from exc
        # Never trust a payload-blind stat cache for scientific verification.
        # Payload files are hashed by _verify_checkpoint_directory on every call.
        if expected_previous is ...:
            if depth == self.start_depth:
                expected_prev = None
            else:
                prev_manifest = self.verify_checkpoint(depth - 1)
                expected_prev = prev_manifest["checkpoint_content_sha256"]
        else:
            expected_prev = expected_previous
        manifest, _ = _verify_checkpoint_directory(
            self.checkpoint_path(depth),
            depth=depth,
            run_identity_sha256=self.expected_identity().science_sha256,
            run_id=self.run_id,
            expected_previous_checkpoint_sha256=expected_prev,
        )
        self._checkpoint_cache[depth] = (manifest, stats[0], stats[1])
        return manifest

    def full_checkpoint_audit(self) -> list[int]:
        """Perform one explicit full-chain verification and seed the controller cache."""
        self._checkpoint_cache.clear()
        depths = []
        if self.checkpoint_root.exists():
            depths = [int(p.name[1:]) for p in sorted(self.checkpoint_root.glob("O[0-9][0-9][0-9][0-9][0-9]"))]
        expected = list(range(self.start_depth, (max(depths) if depths else self.start_depth - 1) + 1))
        if depths != expected:
            raise CorruptDepthCheckpoint(f"non-contiguous committed checkpoint sequence: {depths}")
        prev = None
        for depth in depths:
            manifest = self.verify_checkpoint(depth, expected_previous=prev)
            prev = manifest["checkpoint_content_sha256"]
        self._verified_depths = list(depths)
        return depths

    def load_level_record(self, depth: int) -> LevelTestRecord:
        self.verify_checkpoint(depth)
        return level_record_from_json(_json_read(self.checkpoint_path(depth) / "LEVEL_RECORD.json"))

    def load_committed_records(self) -> list[LevelTestRecord]:
        records = [self.load_level_record(d) for d in self.valid_checkpoint_depths()]
        # A full load is an explicit integrity operation.  Once loaded under the
        # single-writer lock, progress/status reads consume this immutable in-memory
        # snapshot and later commits append exactly one record.
        self._status_records_cache = list(records)
        return records

    def commit_depth(
        self,
        *,
        depth: int,
        source_input: Mapping[str, Any],
        capability_snapshot: Mapping[str, Any],
        level_record: LevelTestRecord,
        maturation_audit: Mapping[str, Any],
        depth_started_utc: str,
        depth_elapsed_seconds: float,
    ) -> Path:
        depth = int(depth)
        # Fail closed on source mutation, but do this once per depth rather than on
        # every TEST_START/TEST_COMPLETE status update.
        self._verify_source_unchanged()
        self.check_resource_budget(depth=depth, depth_elapsed_seconds=float(depth_elapsed_seconds))
        target = self.checkpoint_path(depth)
        if level_record.depth != depth or not level_record.executions:
            raise RunOperationsError("scientific checkpoint requires a nonempty LevelTestRecord bound to depth")
        missing_tests = set(self.selected_test_refs) - set(level_record.executions)
        if missing_tests:
            raise RunOperationsError(f"scientific checkpoint missing selected tests: {sorted(missing_tests)}")
        for tref, exe in level_record.executions.items():
            if exe.evidence_sha256 != canonical_sha256(exe.evidence_payload):
                raise RunOperationsError(f"scientific checkpoint evidence hash mismatch: {tref}")
        mat_record = maturation_audit.get("maturation_record") if isinstance(maturation_audit, Mapping) else None
        if not isinstance(mat_record, Mapping) or int(mat_record.get("depth_end", -1)) != depth:
            raise RunOperationsError("scientific checkpoint maturation audit must end at committed depth")
        gate = maturation_audit.get("review_gate")
        if not isinstance(gate, Mapping) or int(gate.get("evaluated_through_depth", -1)) != depth:
            raise RunOperationsError("scientific checkpoint review gate must bind committed depth")
        findings = {k: dict(v.finding) for k, v in sorted(level_record.executions.items())}
        payloads = {
            "SOURCE_INPUT.json": dict(source_input),
            "CAPABILITY_SNAPSHOT.json": dict(capability_snapshot),
            "LEVEL_RECORD.json": level_record_to_json(level_record),
            "FINDINGS.json": findings,
            "MATURATION_AUDIT_THROUGH_DEPTH.json": dict(maturation_audit),
        }
        if target.exists():
            self.verify_checkpoint(depth)
            for name, expected_obj in payloads.items():
                observed_obj = _json_read(target / name)
                if canonical_sha256(observed_obj) != canonical_sha256(expected_obj):
                    raise CorruptDepthCheckpoint(
                        f"checkpoint O{depth} already exists with different computed semantic payload: {name}"
                    )
            return target
        previous_depths = self.valid_checkpoint_depths()
        expected_next = self.start_depth if not previous_depths else previous_depths[-1] + 1
        if depth != expected_next:
            raise RunOperationsError(f"checkpoint O{depth} is not next expected depth O{expected_next}")

        tmp = self.checkpoint_root / f".O{depth:05d}.tmp-{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            for name, obj in payloads.items():
                write_json_atomic(tmp / name, obj)
            files = []
            for p in sorted(x for x in tmp.iterdir() if x.is_file()):
                files.append({"path": p.name, "sha256": _sha_file(p), "size_bytes": p.stat().st_size})
            previous_ref = None
            if previous_depths:
                prev_manifest = self.verify_checkpoint(previous_depths[-1])
                previous_ref = prev_manifest["checkpoint_content_sha256"]
            manifest = {
                "schema_id": CHECKPOINT_SCHEMA,
                "schema_version": "1.0.0",
                "run_id": self.run_id,
                "run_identity_sha256": self.expected_identity().science_sha256,
                "depth": depth,
                "previous_checkpoint_content_sha256": previous_ref,
                "level_record_sha256": level_record_to_json(level_record)["record_sha256"],
                "source_science_sha256": level_record.source_science_sha256,
                "provider_ref": level_record.provider_ref,
                "provider_kind": level_record.provider_kind,
                "files": files,
                "depth_started_utc": depth_started_utc,
                "depth_completed_utc": _utc_now(),
                "depth_elapsed_seconds": float(depth_elapsed_seconds),
                "commit_protocol": "FILES_FSYNC_MANIFEST_COMPLETE_ATOMIC_RENAME",
            }
            manifest["checkpoint_content_sha256"] = canonical_sha256(manifest)
            write_json_atomic(tmp / "CHECKPOINT_MANIFEST.json", manifest)
            complete = {
                "schema_id": "IG_MATURATION_DEPTH_CHECKPOINT_COMPLETE_V1",
                "depth": depth,
                "manifest_file_sha256": _sha_file(tmp / "CHECKPOINT_MANIFEST.json"),
                "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
                "completed_utc": _utc_now(),
            }
            write_json_atomic(tmp / "COMPLETE.json", complete)
            _fsync_dir(tmp)
            tmp.replace(target)
            _fsync_dir(self.checkpoint_root)
            self.verify_checkpoint(depth)
        except Exception:
            # Deliberately preserve failed temp directory for forensic inspection; it is ignored on resume.
            raise
        self._append_timing({
            "depth": depth,
            "stage": "DEPTH_TOTAL",
            "test_ref": "",
            "start_utc": depth_started_utc,
            "end_utc": _utc_now(),
            "elapsed_seconds": f"{float(depth_elapsed_seconds):.9f}",
            "provider_ref": level_record.provider_ref,
            "scientific_status": "",
            "candidate_like": "",
            "checkpoint_ref": target.name,
            "outcome": "COMMITTED",
            "attempt_disposition": "LOCAL_COMMITTED",
        })
        self._write_durable_mirror_segment(depth)
        if self._status_records_cache is not None:
            if self._status_records_cache and self._status_records_cache[-1].depth >= depth:
                if self._status_records_cache[-1].depth != depth:
                    raise RunOperationsError("status record cache depth regression")
            else:
                self._status_records_cache.append(level_record)
        if self.external_ack_required:
            self.write_external_request(depth)
            latest, pending = self._external_ack_status_cache or (None, [])
            if depth not in pending:
                pending = list(pending) + [depth]
            self._external_ack_status_cache = (latest, pending)
        return target

    def _timing_depth_durations(self) -> list[float]:
        out: list[float] = []
        if not self.timings_path.is_file():
            return out
        with self.timings_path.open("r", encoding="utf-8", newline="") as h:
            for row in csv.DictReader(h):
                if row.get("stage") == "DEPTH_TOTAL" and row.get("outcome") == "COMMITTED":
                    try:
                        out.append(float(row["elapsed_seconds"]))
                    except Exception:
                        pass
        return out

    def status_payload(
        self,
        *,
        state: str,
        current_depth: int | None,
        current_test_ref: str | None = None,
        last_completed_test_ref: str | None = None,
        failure: Mapping[str, Any] | None = None,
        review_gate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = (
            list(self._status_records_cache)
            if self._status_records_cache is not None
            else self.load_committed_records()
        )
        completed = [r.depth for r in records]
        remaining = [d for d in range(self.start_depth, self.max_depth + 1) if d not in set(completed)]
        durations = self._timing_depth_durations()
        recent = durations[-5:]
        mean_recent = sum(recent) / len(recent) if recent else None
        eta = mean_recent * len(remaining) if mean_recent is not None else None
        counts = _finding_counts(records)
        candidate_like = sum(v for k, v in counts.items() if k in CANDIDATE_LIKE_STATUSES)
        seams = [r.depth for r in records if r.provider_seam]
        last_cp = self.checkpoint_path(completed[-1]).name if completed else None
        started_utc = None
        started_local = None
        if self.status_path.exists():
            prior = _json_read(self.status_path)
            started_utc = prior.get("started_utc")
            started_local = prior.get("started_local")
        now_utc, now_local = _utc_now(), _local_now()
        obj = {
            "schema_id": RUN_STATUS_SCHEMA,
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "state": state,
            "run_identity_sha256": self.expected_identity().science_sha256,
            "plan_id": self.plan.get("plan_id"),
            "plan_science_sha256": self.plan_sha,
            "frontier_science_sha256": self.frontier_sha,
            "decoder_version": __version__,
            "decoder_source_sha256": self._verified_source_sha256 or self._verify_source_unchanged(),
            "controller_pid": os.getpid(),
            "controller_host": socket.gethostname(),
            "controller_lock_held": self.controller_lock_payload is not None,
            "controller_lock_created_utc": None if self.controller_lock_payload is None else self.controller_lock_payload.get("created_utc"),
            "controller_process_start_token": None if self.controller_lock_payload is None else self.controller_lock_payload.get("process_start_token"),
            "controller_boot_id": None if self.controller_lock_payload is None else self.controller_lock_payload.get("boot_id"),
            "controller_lock_token": None if self.controller_lock_payload is None else self.controller_lock_payload.get("lock_token"),
            "entity_class_ref": self.entity_class_ref,
            "regime_ref": self.regime_ref,
            "planned_depth_range": {"start": self.start_depth, "end": self.max_depth},
            "completed_depths": completed,
            "completed_count": len(completed),
            "remaining_depths": remaining,
            "remaining_count": len(remaining),
            "current_depth": current_depth,
            "current_test_ref": current_test_ref,
            "last_completed_test_ref": last_completed_test_ref,
            "selected_test_refs": list(self.selected_test_refs),
            "finding_counts_by_status": counts,
            "candidate_like_finding_count": candidate_like,
            "provider_seam_depths": seams,
            "last_checkpoint": last_cp,
            "checkpoint_policy": "ATOMIC_VERIFIED_EVERY_COMPLETED_DEPTH",
            "resume_policy": "STRICT_IDENTITY_MATCH_RESUME_FROM_LAST_VALID_DEPTH",
            "durable_mirror_policy": "IMMUTABLE_COMPACT_SEGMENT_EVERY_COMMITTED_DEPTH_FAIL_CLOSED",
            "durable_mirror_root": str(self.mirror_root),
            "durable_mirror_latest_depth": (completed[-1] if completed and self._mirror_segment_path(completed[-1]).is_file() else None),
            "external_durability_required": self.external_ack_required,
            "external_required_providers": list(self.external_required_providers),
            "external_ack_latest_depth": (
                (self._external_ack_status_cache or (None, []))[0]
                if self.external_ack_required else None
            ),
            "external_ack_pending_depths": (
                list((self._external_ack_status_cache or (None, []))[1])
                if self.external_ack_required else []
            ),
            "controller_session_id": self.controller_session_id,
            "active_depth_attempt_ids": {str(k): v for k, v in sorted(self._active_attempts.items())},
            "resource_budget": dict(self.resource_budget),
            "started_utc": started_utc or now_utc,
            "started_local": started_local or now_local,
            "last_update_utc": now_utc,
            "last_update_local": now_local,
            "mean_recent_depth_seconds": mean_recent,
            "estimated_remaining_seconds": eta,
            "timing_ledger": self.timings_path.name,
            "failure": None if failure is None else dict(failure),
            "review_required": bool(review_gate and review_gate.get("required")),
            "review_gate": None if review_gate is None else dict(review_gate),
            "scientific_frontier_mutation": False,
            "plateau_automatic_promotion": False,
            "lift_automatic_promotion": False,
        }
        obj["status_sha256"] = canonical_sha256(obj)
        return obj

    def write_status(self, **kwargs: Any) -> dict[str, Any]:
        with self._mutation_lock:
            obj = self.status_payload(**kwargs)
            write_json_atomic(self.status_path, obj)
            return obj

    def progress_callback(self, *, depth: int, provider_ref: str):
        last_completed: dict[str, str | None] = {"test_ref": None}

        def callback(event: str, payload: Mapping[str, Any]) -> None:
            tref = str(payload.get("test_ref") or "") or None
            if event == "TEST_START" and tref:
                self._test_starts[tref] = (time.perf_counter(), _utc_now())
                self.write_status(state="RUNNING", current_depth=depth, current_test_ref=tref, last_completed_test_ref=last_completed["test_ref"])
            elif event == "TEST_COMPLETE" and tref:
                started = self._test_starts.pop(tref, (time.perf_counter(), _utc_now()))
                elapsed = time.perf_counter() - started[0]
                sci = str(payload.get("scientific_status") or "")
                self._append_timing({
                    "depth": depth,
                    "stage": "TEST",
                    "test_ref": tref,
                    "start_utc": started[1],
                    "end_utc": _utc_now(),
                    "elapsed_seconds": f"{elapsed:.9f}",
                    "provider_ref": provider_ref,
                    "scientific_status": sci,
                    "candidate_like": "1" if sci in CANDIDATE_LIKE_STATUSES else "0",
                    "checkpoint_ref": "",
                    "outcome": "PASS",
                })
                last_completed["test_ref"] = tref
                self.write_status(state="RUNNING", current_depth=depth, current_test_ref=None, last_completed_test_ref=tref)
        return callback

    def latest_maturation_audit(self) -> dict[str, Any] | None:
        depths = self.valid_checkpoint_depths()
        if not depths:
            return None
        return _json_read(self.checkpoint_path(depths[-1]) / "MATURATION_AUDIT_THROUGH_DEPTH.json")

    def record_review_acknowledgement(self, review_gate: Mapping[str, Any]) -> dict[str, Any]:
        """Append an auditable operational acknowledgement without mutating science artifacts."""
        gate_sha = str(review_gate.get("science_sha256") or canonical_sha256(dict(review_gate)))
        obj = {
            "schema_id": "IG_SCIENTIFIC_REVIEW_ACKNOWLEDGEMENT_V1",
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "acknowledged_utc": _utc_now(),
            "acknowledged_review_gate_sha256": gate_sha,
            "evaluated_through_depth": review_gate.get("evaluated_through_depth"),
            "automatic_promotion_authorized": False,
        }
        obj["acknowledgement_sha256"] = canonical_sha256(obj)
        path = self.run_dir / "REVIEW_ACKNOWLEDGEMENTS.jsonl"
        with self._mutation_lock:
            with path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
                h.flush()
                os.fsync(h.fileno())
            _fsync_dir(path.parent)
        return obj

    def repair_status_from_checkpoints(self, *, state: str = "PAUSED") -> dict[str, Any]:
        depths = self.valid_checkpoint_depths()
        next_depth = (depths[-1] + 1) if depths else self.start_depth
        current = next_depth if next_depth <= self.max_depth else None
        return self.write_status(state=state if current is not None else "COMPLETE", current_depth=current)


def restore_run_from_durable_mirror(mirror_root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Reconstruct a resumable live run directory from immutable mirror segments.

    Segments are verified in ascending depth, including run identity, predecessor checkpoint
    chain, ZIP CRC, per-file hashes and segment self-hashes.  Restoration is into a fresh
    destination only and is committed by atomic directory rename.
    """
    mirror_root = Path(mirror_root)
    destination = Path(destination)
    segments = sorted((mirror_root / "segments").glob("O[0-9][0-9][0-9][0-9][0-9].zip"))
    if not segments:
        raise RunOperationsError(f"no durable mirror segments found under {mirror_root}")
    if destination.exists():
        raise RunOperationsError(f"restore destination already exists: {destination}")
    tmp = destination.parent / f".{destination.name}.restore-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=False)
    expected_run_identity = None
    expected_prev = None
    depths: list[int] = []
    try:
        for seg in segments:
            sidecar = seg.with_suffix(seg.suffix + ".sha256.txt")
            if not sidecar.is_file() or _read_sidecar_sha(sidecar, seg.name) != _sha_file(seg):
                raise RunOperationsError(f"mirror checksum sidecar mismatch: {seg.name}")
            with zipfile.ZipFile(seg, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise RunOperationsError(f"mirror segment CRC failure {seg.name}: {bad}")
                names = zf.namelist()
                if names.count("MIRROR_SEGMENT_MANIFEST.json") != 1 or len(names) != len(set(names)):
                    raise RunOperationsError(f"mirror segment invalid member multiplicity: {seg.name}")
                sm = json.loads(zf.read("MIRROR_SEGMENT_MANIFEST.json"))
                raw = dict(sm); expected_sm = raw.pop("segment_content_sha256", None)
                if expected_sm != canonical_sha256(raw):
                    raise RunOperationsError(f"mirror segment manifest identity mismatch: {seg.name}")
                depth = int(sm["depth"])
                if seg.name != f"O{depth:05d}.zip":
                    raise RunOperationsError(f"mirror segment filename/depth mismatch: {seg.name}")
                if expected_run_identity is None:
                    expected_run_identity = sm["run_identity_sha256"]
                elif sm["run_identity_sha256"] != expected_run_identity:
                    raise RunOperationsError("mirror segments span different run identities")
                if sm.get("previous_checkpoint_content_sha256") != expected_prev:
                    raise RunOperationsError(f"mirror predecessor chain mismatch at O{depth}")
                declared = {r["path"]: r for r in sm.get("files", [])}
                if len(declared) != len(sm.get("files", [])):
                    raise RunOperationsError(f"mirror segment duplicate declared member: {seg.name}")
                for rel in declared:
                    safe = _safe_relative_member(rel)
                    parts = safe.parts
                    allowed = (
                        rel in SAFE_MIRROR_CORE_MEMBERS
                        or (
                            len(parts) == 3
                            and parts[0] == "checkpoints"
                            and parts[1] == f"O{depth:05d}"
                            and parts[2] in (set(REQUIRED_CHECKPOINT_PAYLOAD_FILES) | {"CHECKPOINT_MANIFEST.json", "COMPLETE.json"})
                        )
                    )
                    if not allowed:
                        raise RunOperationsError(f"mirror segment undeclared path class {seg.name}:{rel}")
                expected_names = set(declared) | {"MIRROR_SEGMENT_MANIFEST.json"}
                if set(names) != expected_names:
                    raise RunOperationsError(f"mirror segment ZIP member allowlist mismatch: {seg.name}")
                for rel, rec in sorted(declared.items()):
                    if rel not in zf.namelist():
                        raise RunOperationsError(f"mirror segment missing declared member {rel}")
                    data = zf.read(rel)
                    if hashlib.sha256(data).hexdigest() != rec.get("sha256") or len(data) != int(rec.get("size_bytes", -1)):
                        raise RunOperationsError(f"mirror member hash/size mismatch {seg.name}:{rel}")
                    safe = _safe_relative_member(rel)
                    dest = tmp / safe
                    resolved = dest.resolve()
                    root_resolved = tmp.resolve()
                    try:
                        contained = resolved.is_relative_to(root_resolved)
                    except AttributeError:  # pragma: no cover
                        contained = str(resolved).startswith(str(root_resolved) + os.sep)
                    if not contained:
                        raise RunOperationsError(f"mirror member escapes restore root {seg.name}:{rel}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                expected_prev = sm["checkpoint_content_sha256"]
                depths.append(depth)
        expected_depths = list(range(depths[0], depths[-1] + 1))
        if depths != expected_depths:
            raise RunOperationsError(f"mirror segment sequence is not contiguous: {depths}")
        latest_path = mirror_root / "LATEST.json"
        if not latest_path.is_file():
            raise RunOperationsError("mirror LATEST.json missing")
        latest = _json_read(latest_path)
        raw_latest = dict(latest); latest_hash = raw_latest.pop("latest_content_sha256", None)
        if latest_hash != canonical_sha256(raw_latest):
            raise RunOperationsError("mirror LATEST.json self-hash mismatch")
        if int(latest.get("latest_mirrored_depth", -1)) != depths[-1]:
            raise RunOperationsError("mirror LATEST.json does not point to final contiguous segment")
        final_seg = segments[-1]
        if latest.get("latest_segment_sha256") != _sha_file(final_seg):
            raise RunOperationsError("mirror LATEST.json final segment hash mismatch")
        if latest.get("checkpoint_content_sha256") != expected_prev:
            raise RunOperationsError("mirror LATEST.json final checkpoint binding mismatch")
        for required_core in ("RUN_IDENTITY.json", "RUN_PLAN.json", "RUN_TIMINGS.csv"):
            if not (tmp / required_core).is_file():
                raise RunOperationsError(f"mirror restore missing required core file: {required_core}")
        # A segment can be internally hash-consistent while embedding a checkpoint
        # whose own manifest/payload relation is corrupt.  Re-audit the restored
        # checkpoint chain before atomic publication.
        prev_cp = None
        run_id = _json_read(tmp / "RUN_IDENTITY.json").get("run_id")
        for d in depths:
            manifest, _record = _verify_checkpoint_directory(
                tmp / "checkpoints" / f"O{d:05d}",
                depth=d,
                run_identity_sha256=expected_run_identity,
                run_id=run_id,
                expected_previous_checkpoint_sha256=prev_cp,
            )
            prev_cp = manifest["checkpoint_content_sha256"]
        tmp.replace(destination)
        _fsync_dir(destination.parent)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "restored_run_dir": str(destination),
        "run_identity_sha256": expected_run_identity,
        "restored_depths": depths,
        "latest_checkpoint_content_sha256": expected_prev,
    }


def summarize_run_status(run_dir: str | Path) -> dict[str, Any]:
    """Return status reconciled against committed checkpoints and controller liveness.

    This reader intentionally does not require the currently installed Decoder version to equal
    the run version; old runs remain inspectable.  Resume itself remains strict.
    """
    run_dir = Path(run_dir)
    path = run_dir / "RUN_STATUS.json"
    obj = _json_read(path)
    raw = {k: v for k, v in obj.items() if k != "status_sha256"}
    if obj.get("status_sha256") != canonical_sha256(raw):
        raise RunOperationsError("RUN_STATUS.json identity mismatch")
    rid = obj.get("run_identity_sha256")
    depths = []
    records: list[LevelTestRecord] = []
    checkpoint_hashes: dict[int, str] = {}
    cp_root = run_dir / "checkpoints"
    start = int(obj["planned_depth_range"]["start"]); end = int(obj["planned_depth_range"]["end"])
    prev_sha = None
    for cp in sorted(cp_root.glob("O[0-9][0-9][0-9][0-9][0-9]")) if cp_root.exists() else []:
        depth = int(cp.name[1:])
        manifest, record = _verify_checkpoint_directory(
            cp, depth=depth, run_identity_sha256=rid, run_id=obj.get("run_id"),
            expected_previous_checkpoint_sha256=prev_sha,
        )
        prev_sha = manifest["checkpoint_content_sha256"]
        checkpoint_hashes[depth] = str(manifest["checkpoint_content_sha256"])
        records.append(record)
        depths.append(depth)
    start = int(obj["planned_depth_range"]["start"]); end = int(obj["planned_depth_range"]["end"])
    expected = list(range(start, (max(depths) if depths else start - 1) + 1))
    if depths != expected:
        raise CorruptDepthCheckpoint(f"non-contiguous checkpoint sequence {depths}")
    remaining = [d for d in range(start, end + 1) if d not in set(depths)]
    reconciled = depths != obj.get("completed_depths")
    obj["completed_depths"] = depths
    obj["completed_count"] = len(depths)
    obj["remaining_depths"] = remaining
    obj["remaining_count"] = len(remaining)
    obj["last_checkpoint"] = f"O{depths[-1]:05d}" if depths else None
    # Recompute scientific-status summaries from verified committed LevelTestRecords so a
    # crash after checkpoint commit but before RUN_STATUS update cannot leave stale counts.
    counts = _finding_counts(records)
    obj["finding_counts_by_status"] = counts
    obj["candidate_like_finding_count"] = sum(
        v for k, v in counts.items() if k in CANDIDATE_LIKE_STATUSES
    )
    obj["provider_seam_depths"] = [r.depth for r in records if r.provider_seam]
    # Reconcile connector-backed external durability from immutable requests/acks.
    # This is deliberately read-only: inspection must never create or rewrite evidence.
    external_required = bool(obj.get("external_durability_required"))
    external_pending: list[int] = []
    external_latest: int | None = None
    external_reconciled = False
    if external_required:
        required_providers = tuple(str(x) for x in (obj.get("external_required_providers") or []))
        request_root = run_dir / "external_durability" / "requests"
        ack_root = run_dir / "external_durability" / "acks"
        mirror_root_raw = obj.get("durable_mirror_root")
        mirror_root = Path(str(mirror_root_raw)) if mirror_root_raw else None
        prefix_open = True
        for depth in depths:
            req_path = request_root / f"O{depth:05d}.json"
            ack_path = ack_root / f"O{depth:05d}.json"
            if not req_path.is_file() or not ack_path.is_file():
                external_pending.append(depth)
                prefix_open = False
                continue
            req = _json_read(req_path)
            req_raw = dict(req); req_sha = req_raw.pop("request_sha256", None)
            if req_sha != canonical_sha256(req_raw):
                raise RunOperationsError(f"external durability request O{depth} self-hash mismatch")
            expected_req = {
                "schema_id": EXTERNAL_REQUEST_SCHEMA,
                "schema_version": "1.0.0",
                "run_id": obj.get("run_id"),
                "run_identity_sha256": rid,
                "depth": depth,
                "checkpoint_content_sha256": checkpoint_hashes[depth],
            }
            for key, expected_value in expected_req.items():
                if req.get(key) != expected_value:
                    raise RunOperationsError(f"external durability request O{depth} binding mismatch: {key}")
            if tuple(str(x) for x in (req.get("required_providers") or [])) != required_providers:
                raise RunOperationsError(f"external durability request O{depth} provider set mismatch")
            seg_name = str(req.get("segment_filename") or "")
            if not seg_name or mirror_root is None:
                raise RunOperationsError(f"external durability request O{depth} missing mirror binding")
            seg_path = mirror_root / "segments" / seg_name
            if not seg_path.is_file() or _sha_file(seg_path) != req.get("segment_sha256"):
                raise RunOperationsError(f"external durability request O{depth} local segment binding mismatch")

            ack = _json_read(ack_path)
            ack_raw = dict(ack); ack_sha = ack_raw.pop("ack_sha256", None)
            if ack_sha != canonical_sha256(ack_raw):
                raise RunOperationsError(f"external durability ack O{depth} self-hash mismatch")
            expected_ack = {
                "schema_id": EXTERNAL_ACK_SCHEMA,
                "schema_version": "1.0.0",
                "run_id": obj.get("run_id"),
                "run_identity_sha256": rid,
                "depth": depth,
                "request_sha256": req_sha,
                "checkpoint_content_sha256": checkpoint_hashes[depth],
                "segment_sha256": req.get("segment_sha256"),
            }
            for key, expected_value in expected_ack.items():
                if ack.get(key) != expected_value:
                    raise RunOperationsError(f"external durability ack O{depth} binding mismatch: {key}")
            providers = ack.get("providers")
            if not isinstance(providers, list):
                raise RunOperationsError(f"external durability ack O{depth} providers must be a list")
            by_name = {str(x.get("provider")): x for x in providers if isinstance(x, Mapping)}
            if tuple(sorted(by_name)) != tuple(sorted(required_providers)):
                raise RunOperationsError(f"external durability ack O{depth} provider set mismatch")
            drive = by_name.get("GOOGLE_DRIVE", {})
            github = by_name.get("GITHUB", {})
            if (
                not drive.get("file_id")
                or drive.get("artifact_sha256") != req.get("segment_sha256")
                or drive.get("readback_verified") is not True
            ):
                raise RunOperationsError(f"external durability ack O{depth} Drive proof incomplete")
            if any(not github.get(k) for k in ("repository", "branch", "path", "commit_sha", "manifest_sha256")):
                raise RunOperationsError(f"external durability ack O{depth} GitHub proof incomplete")
            if (
                github.get("drive_file_id") != drive.get("file_id")
                or github.get("artifact_sha256") != req.get("segment_sha256")
            ):
                raise RunOperationsError(f"external durability ack O{depth} GitHub/Drive cross-binding mismatch")
            if prefix_open:
                external_latest = depth
        external_reconciled = (
            obj.get("external_ack_latest_depth") != external_latest
            or obj.get("external_ack_pending_depths") != external_pending
        )
        obj["external_ack_latest_depth"] = external_latest
        obj["external_ack_pending_depths"] = external_pending
        obj["external_ack_status_reconciled"] = external_reconciled
    latest_review_gate = None
    if depths:
        audit_path = cp_root / f"O{depths[-1]:05d}" / "MATURATION_AUDIT_THROUGH_DEPTH.json"
        if audit_path.is_file():
            latest_audit = _json_read(audit_path)
            if isinstance(latest_audit.get("review_gate"), dict):
                latest_review_gate = dict(latest_audit["review_gate"])
    obj["review_gate"] = latest_review_gate
    obj["review_required"] = bool(latest_review_gate and latest_review_gate.get("required"))
    if obj["review_required"] and obj.get("state") not in {"FAILED_SAFE"}:
        obj["state"] = "REVIEW_REQUIRED"
        obj["current_depth"] = remaining[0] if remaining else None
    elif not remaining and obj.get("state") not in {"FAILED_SAFE"}:
        obj["state"] = "COMPLETE"
        obj["current_depth"] = None
    elif remaining and obj.get("state") == "RUNNING":
        host = obj.get("controller_host"); pid = int(obj.get("controller_pid") or -1)
        if host == socket.gethostname() and pid > 0:
            alive = True
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
            if not alive:
                obj["state"] = "INTERRUPTED_RECOVERABLE"
                obj["current_depth"] = remaining[0]
                obj["controller_liveness"] = "DEAD_SAME_HOST"
            else:
                from .checkpoints import _process_start_token, _boot_id
                expected_start=obj.get("controller_process_start_token")
                expected_boot=obj.get("controller_boot_id")
                actual_start=_process_start_token(pid); actual_boot=_boot_id()
                if expected_start and actual_start and expected_start != actual_start or expected_boot and actual_boot and expected_boot != actual_boot:
                    obj["state"] = "INTERRUPTED_RECOVERABLE"
                    obj["current_depth"] = remaining[0]
                    obj["controller_liveness"] = "PID_REUSED_OR_IDENTITY_MISMATCH"
                elif expected_start is None or expected_boot is None or actual_start is None or actual_boot is None:
                    obj["controller_liveness"] = "UNKNOWN_SAME_HOST_IDENTITY_UNRESOLVED"
                else:
                    obj["controller_liveness"] = "ALIVE_SAME_HOST"
        else:
            obj["controller_liveness"] = "UNKNOWN_FOREIGN_OR_UNRESOLVED"
    if external_required and not obj.get("review_required") and obj.get("state") != "FAILED_SAFE":
        if external_pending and remaining:
            obj["state"] = "EXTERNAL_ACK_REQUIRED"
            obj["current_depth"] = remaining[0]
        elif not external_pending and remaining and obj.get("state") == "EXTERNAL_ACK_REQUIRED":
            obj["state"] = "READY_TO_RESUME"
            obj["current_depth"] = remaining[0]
    obj["status_reconciled_from_checkpoints"] = bool(reconciled or external_reconciled)
    # This is a read-time view; do not rewrite historical RUN_STATUS.json merely by inspecting it.
    return obj


def open_existing_run_operations(run_dir: str | Path) -> RunOperations:
    """Open an existing maturation run under the currently executing, source-verified release."""
    run_dir = Path(run_dir)
    ident = _json_read(run_dir / "RUN_IDENTITY.json")
    plan = _json_read(run_dir / "RUN_PLAN.json")
    ops = RunOperations(
        run_dir,
        run_id=str(ident["run_id"]),
        plan=plan,
        plan_science_sha256=str(ident["plan_science_sha256"]),
        frontier_science_sha256=str(ident["frontier_science_sha256"]),
        entity_class_ref=str(ident["entity_class_ref"]),
        regime_ref=str(ident["regime_ref"]),
        selected_test_refs=tuple(ident["selected_test_refs"]),
        start_depth=int(ident["start_depth"]),
        max_depth=int(ident["max_depth"]),
    )
    ops.initialize_or_verify()
    return ops
