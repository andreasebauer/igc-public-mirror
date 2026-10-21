from __future__ import annotations

"""Decoder-owned runtime for scientific chain stages.

Scientific handlers describe deterministic tasks and scientific signatures.  This
runtime exclusively owns multiprocessing, CPU leasing, durable checkpoint state,
compact evidence, streaming reduction, resume and runtime telemetry.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import importlib
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .structural_encoding import structural_canonical_bytes, structural_canonical_sha256, STRUCTURAL_ENCODER_ID
from .execution import ExecutionPolicy, TaskSpec, execute_tasks_stream
from .records import live_source_sha256, utc_now
from .runtime_telemetry import RuntimeTelemetry
from .v05_stage_architecture import require_controller_only_callable
from .workflow_guard import runtime_initialization
from .v05_execution_authority import require_runtime_permit, ExecutionAuthorityError

PARTITION_DB_SCHEMA = "IG_DECODER_V05_STRUCTURAL_PARTITION_DB_V1"
PARTITION_SUMMARY_SCHEMA = "IG_DECODER_V05_STRUCTURAL_PARTITION_SUMMARY_V2"
STAGE_RUNTIME_STATUS_SCHEMA = "IG_DECODER_V05_STAGE_RUNTIME_STATUS_V1"
STATE_STORE_SCHEMA = "IG_DECODER_V05_CONTENT_INDEXED_STATE_STORE_V1"
STATE_STORE_SUMMARY_SCHEMA = "IG_DECODER_V05_CONTENT_INDEXED_STATE_STORE_SUMMARY_V1"
GENERATION_STORE_SCHEMA = "IG_DECODER_V05_STREAMING_GENERATION_STORE_V1"
GENERATION_STORE_SUMMARY_SCHEMA = "IG_DECODER_V05_STREAMING_GENERATION_STORE_SUMMARY_V1"


class StageRuntimeError(RuntimeError):
    pass


class ReplayRootPaused(StageRuntimeError):
    """Registered replay root reached a durable, explicitly resumable brake."""

    def __init__(self, disposition: str, status_artifact: Mapping[str, Any]):
        super().__init__(f"REPLAY_ROOT_PAUSED:{disposition}")
        self.disposition = disposition
        self.status_artifact = dict(status_artifact)


_PARTITION_EVALUATOR = None
_PARTITION_ALLOW_TEST_DIGEST_OVERRIDE = False
_GENERATION_EVALUATOR = None

# Resource-only housekeeping.  Scientific bytes remain on disk; this only asks
# Linux to release clean page-cache pages for large, completed phase stores.
_FILE_CACHE_RECLAIM_MIN_BYTES = 64 * 1024 * 1024
_FILE_CACHE_RECLAIM_NAMES = (
    "state_store.sqlite3", "state_store.sqlite3-wal", "state_store.sqlite3-shm",
    "partition.sqlite3", "partition.sqlite3-wal", "partition.sqlite3-shm",
)

def _file_cache_dontneed(path: Path, *, min_bytes: int = _FILE_CACHE_RECLAIM_MIN_BYTES) -> tuple[bool, int, str | None]:
    try:
        size = int(path.stat().st_size)
        if size < int(min_bytes):
            return False, size, None
        if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
            return False, size, "POSIX_FADVISE_UNAVAILABLE"
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True, size, None
    except FileNotFoundError:
        return False, 0, None
    except OSError as exc:
        return False, 0, f"{type(exc).__name__}:{exc}"

def reclaim_completed_phase_file_cache(runtime_root: str | Path, *, min_bytes: int = _FILE_CACHE_RECLAIM_MIN_BYTES) -> dict[str, Any]:
    """Release clean cache pages for large completed Decoder phase stores.

    This is availability/resource housekeeping only: no file is deleted or changed,
    and phase status/structural equality authority is untouched.
    """
    root = Path(runtime_root).resolve()
    phases = root / "science_chains"
    out: dict[str, Any] = {
        "schema_id": "IG_DECODER_COMPLETED_PHASE_FILE_CACHE_RECLAIM_V1",
        "status": "PASS",
        "min_bytes": int(min_bytes),
        "completed_phases_seen": 0,
        "files_advised": 0,
        "bytes_advised": 0,
        "failures": [],
        "scientific_effect": "NONE",
    }
    if not phases.is_dir():
        return out
    for status_path in sorted(phases.glob("*/decoder_stage_runtime/*/phases/*/RUNTIME_STATUS.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("status") != "COMPLETE":
            continue
        out["completed_phases_seen"] += 1
        phase_root = status_path.parent
        for name in _FILE_CACHE_RECLAIM_NAMES:
            advised, size, err = _file_cache_dontneed(phase_root / name, min_bytes=int(min_bytes))
            if advised:
                out["files_advised"] += 1
                out["bytes_advised"] += int(size)
            elif err is not None:
                out["failures"].append({"path": str((phase_root / name).relative_to(root)), "error": err})
    if out["failures"]:
        out["status"] = "PASS_WITH_NONFATAL_ADVISE_FAILURES"
    return out

def cgroup_memory_snapshot() -> dict[str, Any]:
    """Best-effort resource telemetry; never scientific authority."""
    root = Path("/sys/fs/cgroup")
    snap: dict[str, Any] = {"schema_id": "IG_DECODER_CGROUP_MEMORY_SNAPSHOT_V1"}
    try:
        snap["memory_current_bytes"] = int((root / "memory.current").read_text().strip())
        raw = {}
        for line in (root / "memory.stat").read_text().splitlines():
            k, v = line.split()
            raw[k] = int(v)
        for k in ("anon", "file", "slab", "kernel", "sock", "shmem"):
            if k in raw:
                snap[k + "_bytes"] = raw[k]
        ev = {}
        for line in (root / "memory.events").read_text().splitlines():
            k, v = line.split()
            ev[k] = int(v)
        snap["events"] = ev
    except Exception as exc:
        snap["telemetry_error"] = f"{type(exc).__name__}:{exc}"
    return snap


def _resolve_ref(ref: str, project_binding: dict[str, Any] | None = None):
    mod, sep, name = str(ref).partition(":")
    if not sep:
        raise StageRuntimeError(f"invalid callable ref {ref!r}")
    from .v05_stage_registry import (
        CONTROLLER_ONLY_WORKER_EVALUATORS, ENGINEERING_ONLY_WORKER_EVALUATORS,
        require_registered_evaluator_semantics,
    )
    from .v05_stage_architecture import audit_module_source
    from .project_stage import is_project_ref, resolve_callable
    if is_project_ref(ref):
        try:
            return resolve_callable(ref, project_binding)
        except Exception as exc:
            raise StageRuntimeError(f"PROJECT_EVALUATOR_BINDING_FAILED: {ref}: {exc}") from exc
    if ref not in CONTROLLER_ONLY_WORKER_EVALUATORS + ENGINEERING_ONLY_WORKER_EVALUATORS:
        raise StageRuntimeError(f"EVALUATOR_NOT_IN_ACCEPTED_REGISTRY: {ref}")
    if not mod.startswith("infinity_grid."):
        raise StageRuntimeError("EVALUATOR_MODULE_OUTSIDE_ACCEPTED_PACKAGE")
    path = Path(__file__).resolve().parent.joinpath(*mod.split(".")[1:]).with_suffix(".py")
    from .workflow_guard import preflight
    preflight(Path(__file__).resolve().parents[1], [path])
    gate = audit_module_source(path)
    if gate["status"] != "PASS":
        raise StageRuntimeError(f"EVALUATOR_PREFLIGHT_FAILED: {ref}")
    if ref in CONTROLLER_ONLY_WORKER_EVALUATORS:
        try:
            require_registered_evaluator_semantics(ref, path)
        except Exception as exc:
            raise StageRuntimeError(f"EVALUATOR_SEMANTIC_PREFLIGHT_FAILED: {ref}: {exc}") from exc
    fn = getattr(importlib.import_module(mod), name)
    if not callable(fn):
        raise StageRuntimeError(f"callable ref not callable {ref}")
    return fn


@runtime_initialization
def _init_partition_worker(payload: Mapping[str, Any]) -> None:
    global _PARTITION_EVALUATOR, _PARTITION_ALLOW_TEST_DIGEST_OVERRIDE
    ref = str(payload["evaluator_ref"])
    fn = _resolve_ref(ref, payload.get("project_callable_binding"))
    require_controller_only_callable(fn, role="scientific worker evaluator")
    from .v05_stage_registry import CONTROLLER_ONLY_WORKER_EVALUATORS, get_evaluator_spec
    from .v05_kernel_services import bind_kernel_view, clear_kernel_view
    from .v05_kernel_service_providers import build_kernel_service_providers
    # E1/O3B: runtime owns exact kernel lifecycle, then binds only declared services.
    from .exact_tree_relation_kernel import configure_relation_kernel
    configure_relation_kernel(
        scope_identity=str(payload.get("scope_sha256", "ENGINEERING_UNBOUND")),
        **({
            'max_cache_entries':512, 'max_cache_bytes':128*1024*1024,
            'max_relation_cache_entries':512, 'max_relation_cache_bytes':128*1024*1024,
            'max_observer_q_cache_entries':2048, 'max_observer_q_cache_bytes':256*1024*1024,
            'max_observer_decode_cache_entries':1024, 'max_observer_decode_cache_bytes':128*1024*1024,
        } if (ref.startswith('infinity_grid.g6_marker_evaluators:') or ref.startswith('infinity_grid.g6_s7_evaluators:') or ref.startswith('infinity_grid.g6_s8_evaluators:')) else {})
    )
    if ref in CONTROLLER_ONLY_WORKER_EVALUATORS:
        spec=get_evaluator_spec(ref); bind_kernel_view(spec,build_kernel_service_providers(spec))
    else:
        clear_kernel_view()
    _PARTITION_EVALUATOR = fn
    _PARTITION_ALLOW_TEST_DIGEST_OVERRIDE = bool(payload.get("engineering_test_digest_override", False))


def _bind_controller_fallback_kernel_view(evaluator_ref: str, scope_sha256: str) -> None:
    """Bind exactly the evaluator-declared kernel view for controller-side resume fallback.

    A resumed structural partition may need to re-open one durable representative
    to compare exact canonical signature bytes.  That re-open executes in the
    controller process, not a worker, so it must receive the same declared public
    kernel-service view as the worker evaluator.  This does not alter worker/kernel
    implementations or scientific signatures.
    """
    from .v05_stage_registry import CONTROLLER_ONLY_WORKER_EVALUATORS, get_evaluator_spec
    from .v05_kernel_services import bind_kernel_view, clear_kernel_view
    from .v05_kernel_service_providers import build_kernel_service_providers
    from .exact_tree_relation_kernel import configure_relation_kernel
    ref = str(evaluator_ref)
    if ref not in CONTROLLER_ONLY_WORKER_EVALUATORS:
        clear_kernel_view()
        return
    configure_relation_kernel(
        scope_identity=str(scope_sha256) + ':CONTROLLER_RESUME_FALLBACK',
        **({
            'max_cache_entries':512, 'max_cache_bytes':128*1024*1024,
            'max_relation_cache_entries':512, 'max_relation_cache_bytes':128*1024*1024,
            'max_observer_q_cache_entries':2048, 'max_observer_q_cache_bytes':256*1024*1024,
            'max_observer_decode_cache_entries':1024, 'max_observer_decode_cache_bytes':128*1024*1024,
        } if (ref.startswith('infinity_grid.g6_marker_evaluators:') or ref.startswith('infinity_grid.g6_s7_evaluators:') or ref.startswith('infinity_grid.g6_s8_evaluators:')) else {})
    )
    spec = get_evaluator_spec(ref)
    bind_kernel_view(spec, build_kernel_service_providers(spec))


@runtime_initialization
def _init_generation_worker(payload: Mapping[str, Any]) -> None:
    global _GENERATION_EVALUATOR
    ref = str(payload["evaluator_ref"])
    fn = _resolve_ref(ref, payload.get("project_callable_binding"))
    require_controller_only_callable(fn, role="scientific generation evaluator")
    from .v05_stage_registry import CONTROLLER_ONLY_WORKER_EVALUATORS, get_evaluator_spec
    from .v05_kernel_services import bind_kernel_view, clear_kernel_view
    from .v05_kernel_service_providers import build_kernel_service_providers
    # E1/O3B: runtime owns exact kernel lifecycle, then binds only declared services.
    from .exact_tree_relation_kernel import configure_relation_kernel
    configure_relation_kernel(
        scope_identity=str(payload.get("scope_sha256", "ENGINEERING_UNBOUND")),
        **({
            'max_cache_entries':512, 'max_cache_bytes':128*1024*1024,
            'max_relation_cache_entries':512, 'max_relation_cache_bytes':128*1024*1024,
            'max_observer_q_cache_entries':2048, 'max_observer_q_cache_bytes':256*1024*1024,
            'max_observer_decode_cache_entries':1024, 'max_observer_decode_cache_bytes':128*1024*1024,
        } if (ref.startswith('infinity_grid.g6_marker_evaluators:') or ref.startswith('infinity_grid.g6_s7_evaluators:') or ref.startswith('infinity_grid.g6_s8_evaluators:')) else {})
    )
    if ref in CONTROLLER_ONLY_WORKER_EVALUATORS:
        spec=get_evaluator_spec(ref); bind_kernel_view(spec,build_kernel_service_providers(spec))
    else:
        clear_kernel_view()
    _GENERATION_EVALUATOR = fn


def _generation_worker(payload: Any) -> dict[str, Any]:
    if _GENERATION_EVALUATOR is None:
        raise StageRuntimeError("generation worker evaluator not initialized")
    t0 = time.perf_counter(); c0 = time.process_time()
    from .workflow_guard import scientific_call
    with scientific_call(Path(__file__).resolve().parents[1]):
        raw = _GENERATION_EVALUATOR(payload)
    eval_wall = time.perf_counter() - t0; eval_cpu = time.process_time() - c0
    if not isinstance(raw, Mapping) or "states" not in raw:
        raise StageRuntimeError("generation evaluator must return mapping with states")
    encoded = []
    enc_wall = 0.0
    for row in raw["states"]:
        if not isinstance(row, Mapping) or "identity" not in row or "state" not in row:
            raise StageRuntimeError("generation state row must contain identity and state")
        e0 = time.perf_counter()
        b = structural_canonical_bytes(row["identity"])
        digest = hashlib.sha256(b).hexdigest()
        if "_test_identity_digest_override" in row:
            if os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") != "GENERATION_DIGEST_COLLISION_TEST":
                raise StageRuntimeError("generation digest override requires explicit engineering acknowledgement")
            digest = str(row["_test_identity_digest_override"])
        enc_wall += time.perf_counter() - e0
        encoded.append({"identity_sha256": digest, "identity_canonical_bytes": b, "state": row["state"],
                        "engineering_digest_override": "_test_identity_digest_override" in row})
    metrics = dict(raw.get("metrics") or {})
    metrics.update(engine_evaluator_wall_seconds=eval_wall, engine_evaluator_cpu_seconds=eval_cpu,
                   engine_identity_encode_wall_seconds=enc_wall, engine_identity_encoder_id=STRUCTURAL_ENCODER_ID,
                   generated_occurrence_count=len(encoded))
    return {"states": encoded, "metrics": metrics}


def _partition_worker(payload: Any) -> dict[str, Any]:
    if _PARTITION_EVALUATOR is None:
        raise StageRuntimeError("partition worker evaluator not initialized")
    evaluation_start = time.perf_counter()
    evaluation_cpu_start = time.process_time()
    from .workflow_guard import scientific_call
    with scientific_call(Path(__file__).resolve().parents[1]):
        raw = _PARTITION_EVALUATOR(payload)
    evaluation_wall = time.perf_counter() - evaluation_start
    evaluation_cpu = time.process_time() - evaluation_cpu_start
    if not isinstance(raw, Mapping) or "signature" not in raw:
        raise StageRuntimeError("partition evaluator must return mapping with signature")
    sig = raw["signature"]
    encoding_start = time.perf_counter()
    signature_canonical_bytes = structural_canonical_bytes(sig)
    digest = hashlib.sha256(signature_canonical_bytes).hexdigest()
    encoding_wall = time.perf_counter() - encoding_start
    # Deliberate collision testing is available only behind an explicit engineering ack.
    engineering_digest_override = "_test_signature_digest_override" in raw
    if engineering_digest_override:
        if not _PARTITION_ALLOW_TEST_DIGEST_OVERRIDE:
            raise StageRuntimeError("test digest override requires explicit engineering acknowledgement")
        digest = str(raw["_test_signature_digest_override"])
    if len(digest) != 64:
        raise StageRuntimeError("signature digest must be SHA-256 shaped")
    metrics = dict(raw.get("metrics") or {})
    metrics.update(engine_evaluator_wall_seconds=evaluation_wall,
                   engine_evaluator_cpu_seconds=evaluation_cpu,
                   engine_signature_encode_wall_seconds=encoding_wall,
                   engine_signature_encoder_id=STRUCTURAL_ENCODER_ID)
    outcome_count = int(raw.get("outcome_count", 0))
    return {
        "signature_sha256": digest,
        "signature_canonical_bytes": signature_canonical_bytes,
        "engineering_digest_override": engineering_digest_override,
        "outcome_count": outcome_count,
        "metrics": metrics,
    }


def _phase_safe(phase_id: str) -> str:
    s = str(phase_id)
    if not s or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for ch in s):
        raise StageRuntimeError(f"invalid phase_id {phase_id!r}")
    return s


def _rss_bytes() -> int:
    try:
        for line in Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _stage_workspace_bytes(root: Path) -> int:
    """Count publication files and every closed-layout phase in a stage."""
    root = Path(root)
    total = 0
    if not root.exists():
        return total
    for child in root.iterdir():
        if child.is_symlink() or not child.is_dir() or child.name not in {
            "artifacts", "phases", "replay_runner", "replay_reference_data",
        }:
            raise StageRuntimeError(f"unexpected stage workspace entry {child.name}")
        if child.name == "replay_runner":
            total += _closed_json_store_bytes(
                child,
                root_files={"runner_state.json"},
                json_directories={"checkpoints"},
            )
            continue
        if child.name == "replay_reference_data":
            total += _closed_json_store_bytes(
                child,
                root_files={"MANIFEST.json"},
                json_directories={"commits", "records", "transactions"},
            )
            continue
        for item in child.iterdir():
            if item.is_symlink():
                raise StageRuntimeError(f"stage workspace symlink {item.name}")
            if child.name == "artifacts":
                if not item.is_file() or item.suffix != ".json":
                    raise StageRuntimeError(f"unexpected publication entry {item.name}")
                total += item.stat().st_size
            else:
                if not item.is_dir():
                    raise StageRuntimeError(f"unexpected phase entry {item.name}")
                _phase_safe(item.name)
                total += _runtime_workspace_bytes(item)
    return total


def _closed_json_store_bytes(
    root: Path, *, root_files: set[str], json_directories: set[str],
) -> int:
    """Account a replay store with an exact, symlink-free, flat JSON layout."""
    total = 0
    for item in root.iterdir():
        if item.is_symlink():
            raise StageRuntimeError(f"replay workspace symlink {item.name}")
        if item.name == ".replay-index.lock":
            if not item.is_file() or item.stat().st_size != 0:
                raise StageRuntimeError(f"unexpected replay workspace entry {item.name}")
            continue
        if item.name in root_files:
            if not item.is_file():
                raise StageRuntimeError(f"unexpected replay workspace entry {item.name}")
            total += item.stat().st_size
            continue
        if item.name not in json_directories or not item.is_dir():
            raise StageRuntimeError(f"unexpected replay workspace entry {item.name}")
        for record in item.iterdir():
            if record.is_symlink() or not record.is_file() or record.suffix != ".json":
                raise StageRuntimeError(f"unexpected replay record entry {record.name}")
            total += record.stat().st_size
    return total


def _runtime_workspace_bytes(root: Path) -> int:
    """Exact accounting for the runtime-owned phase workspace without recursive scans.

    Controller-only phases have a closed durable layout.  Unknown entries fail
    closed instead of being silently ignored.  Known variable directories are
    flat and tiny (telemetry: two files; collision witnesses: exceptional only).
    """
    root = Path(root)
    allowed_root_files = {
        "partition.sqlite3", "partition.sqlite3-wal", "partition.sqlite3-shm",
        "state_store.sqlite3", "state_store.sqlite3-wal", "state_store.sqlite3-shm",
        "RUNTIME_STATUS.json", "SUMMARY.json",
    }
    allowed_dirs = {"telemetry", "digest_collision_witnesses"}
    total = 0
    if not root.exists():
        return 0
    for child in root.iterdir():
        if child.is_file():
            if child.name not in allowed_root_files:
                raise StageRuntimeError(f"unexpected runtime workspace file {child.name}")
            total += child.stat().st_size
        elif child.is_dir():
            if child.name not in allowed_dirs:
                raise StageRuntimeError(f"unexpected runtime workspace directory {child.name}")
            allowed_names = {"RUNTIME_STATUS.json", "RUNTIME_TIMINGS.json"} if child.name == "telemetry" else None
            for item in child.iterdir():
                if not item.is_file():
                    raise StageRuntimeError(f"unexpected nested runtime workspace entry {item}")
                if allowed_names is not None and item.name not in allowed_names:
                    raise StageRuntimeError(f"unexpected telemetry workspace file {item.name}")
                if child.name == "digest_collision_witnesses" and item.suffix != ".json":
                    raise StageRuntimeError(f"unexpected collision witness file {item.name}")
                total += item.stat().st_size
        else:
            raise StageRuntimeError(f"unexpected runtime workspace entry {child}")
    return total


@dataclass(frozen=True)
class StructuralPartitionResult:
    summary: dict[str, Any]
    execution_metadata: dict[str, Any]


@dataclass(frozen=True)
class ContentIndexedStateStoreResult:
    summary: dict[str, Any]
    execution_metadata: dict[str, Any]


@dataclass(frozen=True)
class StreamingGenerationStoreResult:
    summary: dict[str, Any]
    execution_metadata: dict[str, Any]


class StageScienceRuntime:
    """Capability object passed to controller-only scientific stage handlers."""

    def __init__(
        self,
        *,
        chain_dir: Path,
        chain_id: str,
        stage_id: str,
        question_sha256: str,
        default_workers: int = 4,
        memory_budget_bytes: int | None = None,
        workspace_budget_bytes: int | None = None,
        _execution_permit=None,
        _dependency_loader=None,
        _project_callable_bindings=None,
    ) -> None:
        from .v05_origin_guard import require_controller_execution_origin
        require_controller_execution_origin("StageScienceRuntime.__init__")
        require_runtime_permit(_execution_permit, chain_dir=Path(chain_dir),
            chain_id=str(chain_id), stage_id=str(stage_id), question_sha256=str(question_sha256))
        self._execution_permit = _execution_permit
        self._dependency_loader = _dependency_loader
        self._project_callable_bindings = dict(_project_callable_bindings or {})
        self._chain_dir = Path(chain_dir).resolve(strict=True)
        self.chain_id = str(chain_id)
        self.stage_id = str(stage_id)
        self.question_sha256 = str(question_sha256)
        self.default_workers = max(1, int(default_workers))
        self.memory_budget_bytes = None if memory_budget_bytes is None else int(memory_budget_bytes)
        self.workspace_budget_bytes = None if workspace_budget_bytes is None else int(workspace_budget_bytes)
        self._root = self._chain_dir / "decoder_stage_runtime" / self.stage_id.replace(":", "__")
        from .v05_origin_guard import require_registered_output
        require_registered_output(self._root, "StageScienceRuntime.__init__")
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def runtime_root(self) -> Path:
        # Exposed for diagnostics only. Scientific handlers must not write here;
        # the architecture gate rejects direct durable-write APIs in handler modules.
        self._require_execution()
        return self._root

    def _require_execution(self) -> None:
        from .v05_origin_guard import require_controller_execution_origin
        require_controller_execution_origin("StageScienceRuntime")
        from .v05_origin_guard import require_registered_output
        require_registered_output(self._root, "StageScienceRuntime")
        require_runtime_permit(self._execution_permit, chain_dir=self._chain_dir,
            chain_id=self.chain_id, stage_id=self.stage_id, question_sha256=self.question_sha256)

    def run_registered_engineering_job(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one installed engineering job in a child source workspace."""
        self._require_execution()
        from .v05_engineering_jobs import execute_registered_engineering_job
        return execute_registered_engineering_job(runtime=self, parameters=parameters)

    def accept_registered_engineering_candidate(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Accept and package a previously validated child source candidate."""
        self._require_execution()
        from .v05_engineering_jobs import accept_registered_engineering_candidate
        return accept_registered_engineering_candidate(runtime=self, parameters=parameters)

    def run_registered_replay_root(
        self,
        parameters: Mapping[str, Any],
        input_artifacts: Mapping[str, str],
    ) -> dict[str, Any]:
        """Advance the controller-owned replay root to its next safe boundary.

        P2B remains scheduler-only. P4 binds exactly the four frozen L0
        source-integrity executors. P5 imports that captured frontier and binds
        exactly the four C0_HISTORICAL evidence executors.
        """
        self._require_execution()
        base_fields = {
            "operation", "manifest_input", "manifest_file_sha256",
            "manifest_dag_sha256", "dataset_root_sha256", "root_run_id",
            "authorized_through_layer", "next_wait_layer", "node_executor_bindings",
        }
        operation = parameters.get("operation")
        if operation not in {"SCHEDULE_ONLY_P2B", "EXECUTE_L0_PILOT_P4",
                             "EXECUTE_C0_HISTORICAL_P5", "EXECUTE_L2J3_P6", "EXECUTE_NODE_IN_P7",
                             "EXECUTE_SCOUT_HISTORICAL_P8", "EXECUTE_O1_O3_P9"}:
            raise StageRuntimeError("REPLAY_ROOT_OPERATION_NOT_REGISTERED")
        if operation == "SCHEDULE_ONLY_P2B":
            expected = base_fields
        else:
            expected = base_fields | {"reference_catalogue_input", "reference_catalogue_file_sha256",
                                      "pilot_fault_injection"}
            if operation == "EXECUTE_C0_HISTORICAL_P5":
                expected |= {"p4_frontier_input", "p4_frontier_file_sha256"}
            if operation == "EXECUTE_L2J3_P6":
                expected |= {"p5_frontier_input", "p5_frontier_file_sha256"}
            if operation == "EXECUTE_NODE_IN_P7":
                expected |= {"p6_frontier_input", "p6_frontier_file_sha256"}
            if operation == "EXECUTE_SCOUT_HISTORICAL_P8":
                expected |= {"p7_frontier_input", "p7_frontier_file_sha256"}
            if operation == "EXECUTE_O1_O3_P9":
                expected |= {"p8_frontier_input", "p8_frontier_file_sha256"}
        if set(parameters) != expected:
            raise StageRuntimeError("REPLAY_ROOT_PARAMETER_FIELDS")
        if parameters["authorized_through_layer"] != "G8" or parameters["next_wait_layer"] != "GLOBAL":
            raise StageRuntimeError("REPLAY_ROOT_AUTHORITY_BOUNDARY")
        if operation == "SCHEDULE_ONLY_P2B":
            if parameters["node_executor_bindings"] != []:
                raise StageRuntimeError("P2B_NODE_EXECUTORS_MUST_REMAIN_UNBOUND")
        elif operation == "EXECUTE_L0_PILOT_P4":
            from .replay_l0_executor import L0_NODE_IDS, NODE_EXECUTOR_BINDINGS
            if parameters["node_executor_bindings"] != NODE_EXECUTOR_BINDINGS:
                raise StageRuntimeError("P4_L0_EXECUTOR_BINDINGS")
            injection = parameters["pilot_fault_injection"]
            if injection != "NONE" and injection not in L0_NODE_IDS:
                raise StageRuntimeError("P4_FAULT_INJECTION_NODE")
        elif operation == "EXECUTE_C0_HISTORICAL_P5":
            from .replay_c0_historical_executor import C0_NODE_IDS, NODE_EXECUTOR_BINDINGS
            if parameters["node_executor_bindings"] != NODE_EXECUTOR_BINDINGS:
                raise StageRuntimeError("P5_C0_EXECUTOR_BINDINGS")
            injection = parameters["pilot_fault_injection"]
            if injection != "NONE" and injection not in C0_NODE_IDS:
                raise StageRuntimeError("P5_FAULT_INJECTION_NODE")
        elif operation == "EXECUTE_L2J3_P6":
            from .replay_l2j3_executor import L2J3_NODE_IDS, NODE_EXECUTOR_BINDINGS
            if parameters["node_executor_bindings"] != NODE_EXECUTOR_BINDINGS:
                raise StageRuntimeError("P6_L2J3_EXECUTOR_BINDINGS")
            injection = parameters["pilot_fault_injection"]
            if injection != "NONE" and injection not in L2J3_NODE_IDS:
                raise StageRuntimeError("P6_FAULT_INJECTION_NODE")
        elif operation == "EXECUTE_NODE_IN_P7":
            from .replay_node_in_executor import NODE_IDS, NODE_EXECUTOR_BINDINGS
            if parameters["node_executor_bindings"] != NODE_EXECUTOR_BINDINGS:
                raise StageRuntimeError("P7_NODE_IN_EXECUTOR_BINDINGS")
            injection = parameters["pilot_fault_injection"]
            if injection != "NONE" and injection not in NODE_IDS:
                raise StageRuntimeError("P7_FAULT_INJECTION_NODE")
        elif operation == "EXECUTE_SCOUT_HISTORICAL_P8":
            from .replay_scout_historical_executor import NODE_IDS, NODE_EXECUTOR_BINDINGS
            if parameters["node_executor_bindings"] != NODE_EXECUTOR_BINDINGS:
                raise StageRuntimeError("P8_SCOUT_EXECUTOR_BINDINGS")
            injection = parameters["pilot_fault_injection"]
            if injection != "NONE" and injection not in NODE_IDS:
                raise StageRuntimeError("P8_FAULT_INJECTION_NODE")
        else:
            from .replay_o1_o3_executor import NODE_IDS, NODE_EXECUTOR_BINDINGS
            if parameters["node_executor_bindings"] != NODE_EXECUTOR_BINDINGS:
                raise StageRuntimeError("P9_O1_O3_EXECUTOR_BINDINGS")
            injection = parameters["pilot_fault_injection"]
            if injection != "NONE" and injection not in NODE_IDS:
                raise StageRuntimeError("P9_FAULT_INJECTION_NODE")
        logical = parameters["manifest_input"]
        if not isinstance(logical, str) or logical not in input_artifacts:
            raise StageRuntimeError("REPLAY_ROOT_MANIFEST_INPUT_MISSING")
        manifest_path = Path(input_artifacts[logical]).resolve(strict=True)
        from .v05_origin_guard import require_registered_output
        # Input artifacts are outside the output root by design. Admission has
        # already content-addressed them; independently recheck their bytes here.
        file_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if file_sha != parameters["manifest_file_sha256"]:
            raise StageRuntimeError("REPLAY_ROOT_MANIFEST_FILE_HASH")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageRuntimeError("REPLAY_ROOT_MANIFEST_JSON") from exc
        if manifest.get("dag_sha256") != parameters["manifest_dag_sha256"]:
            raise StageRuntimeError("REPLAY_ROOT_MANIFEST_DAG_HASH")
        if (manifest.get("authorized_replay_through_layer"), manifest.get("next_wait_layer")) != ("G8", "GLOBAL"):
            raise StageRuntimeError("REPLAY_ROOT_COMPILED_AUTHORITY_BOUNDARY")

        catalogue = None
        if operation != "SCHEDULE_ONLY_P2B":
            catalogue_logical = parameters["reference_catalogue_input"]
            if not isinstance(catalogue_logical, str) or catalogue_logical not in input_artifacts:
                raise StageRuntimeError("REPLAY_REFERENCE_CATALOGUE_INPUT_MISSING")
            catalogue_path = Path(input_artifacts[catalogue_logical]).resolve(strict=True)
            if hashlib.sha256(catalogue_path.read_bytes()).hexdigest() != parameters["reference_catalogue_file_sha256"]:
                raise StageRuntimeError("REPLAY_REFERENCE_CATALOGUE_FILE_HASH")
            try:
                catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StageRuntimeError("REPLAY_REFERENCE_CATALOGUE_JSON") from exc
            if (catalogue.get("schema_id"), catalogue.get("version")) != ("IG_REPLAY_OBLIGATION_SOURCE_V3", "9.0"):
                raise StageRuntimeError("REPLAY_REFERENCE_CATALOGUE_SCHEMA")
            from .replay_reference_data import empty_manifest
            if parameters["dataset_root_sha256"] != empty_manifest()["manifest_sha256"]:
                raise StageRuntimeError("REPLAY_EMPTY_REFERENCE_ROOT_IDENTITY")

        state_root = self._root / "replay_runner"
        require_registered_output(state_root, "replay-root-state")
        from .replay_dag_runner import ReplayDagRunner
        runner_exists = (state_root / "runner_state.json").is_file()
        reference_store = None
        if operation == "EXECUTE_L2J3_P6":
            from .replay_p5_frontier_import import P5_STATE_OBJECT_SHA256, P5_COMPLETED_NODE_IDS
            frontier_logical = parameters["p5_frontier_input"]
            if not isinstance(frontier_logical, str) or frontier_logical not in input_artifacts:
                raise StageRuntimeError("P6_FRONTIER_INPUT_MISSING")
            frontier_path = Path(input_artifacts[frontier_logical]).resolve(strict=True)
            if (parameters["p5_frontier_file_sha256"] != P5_STATE_OBJECT_SHA256
                    or hashlib.sha256(frontier_path.read_bytes()).hexdigest() != P5_STATE_OBJECT_SHA256):
                raise StageRuntimeError("P6_FRONTIER_FILE_HASH")
        if operation == "EXECUTE_NODE_IN_P7":
            from .replay_p6_frontier_import import P6_STATE_OBJECT_SHA256, P6_COMPLETED_NODE_IDS
            frontier_logical = parameters["p6_frontier_input"]
            if not isinstance(frontier_logical, str) or frontier_logical not in input_artifacts:
                raise StageRuntimeError("P7_FRONTIER_INPUT_MISSING")
            frontier_path = Path(input_artifacts[frontier_logical]).resolve(strict=True)
            if (parameters["p6_frontier_file_sha256"] != P6_STATE_OBJECT_SHA256
                    or hashlib.sha256(frontier_path.read_bytes()).hexdigest() != P6_STATE_OBJECT_SHA256):
                raise StageRuntimeError("P7_FRONTIER_FILE_HASH")
        if operation == "EXECUTE_SCOUT_HISTORICAL_P8":
            from .replay_p7_frontier_import import P7_STATE_OBJECT_SHA256, P7_COMPLETED_NODE_IDS
            frontier_logical = parameters["p7_frontier_input"]
            if not isinstance(frontier_logical, str) or frontier_logical not in input_artifacts:
                raise StageRuntimeError("P8_FRONTIER_INPUT_MISSING")
            frontier_path = Path(input_artifacts[frontier_logical]).resolve(strict=True)
            if (parameters["p7_frontier_file_sha256"] != P7_STATE_OBJECT_SHA256
                    or hashlib.sha256(frontier_path.read_bytes()).hexdigest() != P7_STATE_OBJECT_SHA256):
                raise StageRuntimeError("P8_FRONTIER_FILE_HASH")
        if operation == "EXECUTE_O1_O3_P9":
            from .replay_p8_frontier_import import P8_STATE_OBJECT_SHA256, P8_COMPLETED_NODE_IDS
            frontier_logical = parameters["p8_frontier_input"]
            if not isinstance(frontier_logical, str) or frontier_logical not in input_artifacts:
                raise StageRuntimeError("P9_FRONTIER_INPUT_MISSING")
            frontier_path = Path(input_artifacts[frontier_logical]).resolve(strict=True)
            if (parameters["p8_frontier_file_sha256"] != P8_STATE_OBJECT_SHA256
                    or hashlib.sha256(frontier_path.read_bytes()).hexdigest() != P8_STATE_OBJECT_SHA256):
                raise StageRuntimeError("P9_FRONTIER_FILE_HASH")
        if operation == "EXECUTE_L2J3_P6" and not runner_exists:
            from .replay_p5_frontier_import import import_p5_frontier
            try:
                runner, reference_store = import_p5_frontier(
                    state_zip=frontier_path,
                    expected_file_sha256=parameters["p5_frontier_file_sha256"],
                    manifest=manifest, state_root=state_root,
                    reference_root=self._root / "replay_reference_data",
                    root_run_id=parameters["root_run_id"],
                    dataset_root_sha256=parameters["dataset_root_sha256"],
                )
            except Exception as exc:
                raise StageRuntimeError(f"P6_FRONTIER_IMPORT_FAILED: {exc}") from exc
            runner_exists = True
        elif operation == "EXECUTE_NODE_IN_P7" and not runner_exists:
            from .replay_p6_frontier_import import import_p6_frontier
            try:
                runner, reference_store = import_p6_frontier(
                    state_zip=frontier_path,
                    expected_file_sha256=parameters["p6_frontier_file_sha256"],
                    manifest=manifest, state_root=state_root,
                    reference_root=self._root / "replay_reference_data",
                    root_run_id=parameters["root_run_id"],
                    dataset_root_sha256=parameters["dataset_root_sha256"],
                )
            except Exception as exc:
                raise StageRuntimeError(f"P7_FRONTIER_IMPORT_FAILED: {exc}") from exc
            runner_exists = True
        elif operation == "EXECUTE_SCOUT_HISTORICAL_P8" and not runner_exists:
            from .replay_p7_frontier_import import import_p7_frontier
            try:
                runner, reference_store = import_p7_frontier(
                    state_zip=frontier_path,
                    expected_file_sha256=parameters["p7_frontier_file_sha256"],
                    manifest=manifest, state_root=state_root,
                    reference_root=self._root / "replay_reference_data",
                    root_run_id=parameters["root_run_id"],
                    dataset_root_sha256=parameters["dataset_root_sha256"],
                )
            except Exception as exc:
                raise StageRuntimeError(f"P8_FRONTIER_IMPORT_FAILED: {exc}") from exc
            runner_exists = True
        elif operation == "EXECUTE_O1_O3_P9" and not runner_exists:
            from .replay_p8_frontier_import import import_p8_frontier
            try:
                runner, reference_store = import_p8_frontier(
                    state_zip=frontier_path,
                    expected_file_sha256=parameters["p8_frontier_file_sha256"],
                    manifest=manifest, state_root=state_root,
                    reference_root=self._root / "replay_reference_data",
                    root_run_id=parameters["root_run_id"],
                    dataset_root_sha256=parameters["dataset_root_sha256"],
                )
            except Exception as exc:
                raise StageRuntimeError(f"P9_FRONTIER_IMPORT_FAILED: {exc}") from exc
            runner_exists = True
        elif operation == "EXECUTE_C0_HISTORICAL_P5" and not runner_exists:
            frontier_logical = parameters["p4_frontier_input"]
            if not isinstance(frontier_logical, str) or frontier_logical not in input_artifacts:
                raise StageRuntimeError("P5_FRONTIER_INPUT_MISSING")
            from .replay_frontier_import import import_p4_frontier
            try:
                runner, reference_store = import_p4_frontier(
                    state_zip=Path(input_artifacts[frontier_logical]).resolve(strict=True),
                    expected_file_sha256=parameters["p4_frontier_file_sha256"],
                    manifest=manifest, state_root=state_root,
                    reference_root=self._root / "replay_reference_data",
                    root_run_id=parameters["root_run_id"],
                    dataset_root_sha256=parameters["dataset_root_sha256"],
                )
            except Exception as exc:
                raise StageRuntimeError(f"P5_FRONTIER_IMPORT_FAILED: {exc}") from exc
            runner_exists = True
        elif runner_exists:
            runner = ReplayDagRunner.resume(manifest, state_root)
            if operation == "EXECUTE_L2J3_P6":
                completed = tuple(runner.state["completed_node_ids"])
                if completed[:len(P5_COMPLETED_NODE_IDS)] != P5_COMPLETED_NODE_IDS:
                    raise StageRuntimeError("P6_RESUME_PREFIX_MISMATCH")
                if any(node_id not in (*P5_COMPLETED_NODE_IDS, *L2J3_NODE_IDS,
                                       "IG/L2J3/GATE/CERTIFY") for node_id in completed):
                    raise StageRuntimeError("P6_RESUME_BEYOND_LAYER")
            if operation == "EXECUTE_NODE_IN_P7":
                completed = tuple(runner.state["completed_node_ids"])
                if completed[:len(P6_COMPLETED_NODE_IDS)] != P6_COMPLETED_NODE_IDS:
                    raise StageRuntimeError("P7_RESUME_PREFIX_MISMATCH")
                if any(node_id not in (*P6_COMPLETED_NODE_IDS, *NODE_IDS,
                                       "IG/NODE_IN/GATE/CERTIFY") for node_id in completed):
                    raise StageRuntimeError("P7_RESUME_BEYOND_LAYER")
            if operation == "EXECUTE_SCOUT_HISTORICAL_P8":
                completed = tuple(runner.state["completed_node_ids"])
                if completed[:len(P7_COMPLETED_NODE_IDS)] != P7_COMPLETED_NODE_IDS:
                    raise StageRuntimeError("P8_RESUME_PREFIX_MISMATCH")
                if any(node_id not in (*P7_COMPLETED_NODE_IDS, *NODE_IDS,
                                       "IG/SCOUT_HISTORICAL/GATE/CERTIFY") for node_id in completed):
                    raise StageRuntimeError("P8_RESUME_BEYOND_LAYER")
            if operation == "EXECUTE_O1_O3_P9":
                completed = tuple(runner.state["completed_node_ids"])
                if completed[:len(P8_COMPLETED_NODE_IDS)] != P8_COMPLETED_NODE_IDS:
                    raise StageRuntimeError("P9_RESUME_PREFIX_MISMATCH")
                if any(node_id not in (*P8_COMPLETED_NODE_IDS, *NODE_IDS,
                                       "IG/O1_O3/GATE/CERTIFY") for node_id in completed):
                    raise StageRuntimeError("P9_RESUME_BEYOND_LAYER")
        else:
            runner = ReplayDagRunner.create(
                manifest,
                state_root,
                root_run_id=parameters["root_run_id"],
                dataset_root={"state": "EMPTY", "sha256": parameters["dataset_root_sha256"]},
            )
        frontier = None
        if operation == "EXECUTE_L0_PILOT_P4":
            from .replay_l0_executor import execute_l0_source_integrity_node, seal_l0_frontier
            from .replay_reference_data import ReplayReferenceDataStore
            reference_root = self._root / "replay_reference_data"
            reference_store = ReplayReferenceDataStore.initialize(reference_root)
            if not runner_exists and reference_store.manifest != empty_manifest():
                raise StageRuntimeError("P4_REFERENCE_ROOT_NOT_EMPTY_AT_START")
        elif operation in {"EXECUTE_C0_HISTORICAL_P5", "EXECUTE_L2J3_P6", "EXECUTE_NODE_IN_P7",
                           "EXECUTE_SCOUT_HISTORICAL_P8", "EXECUTE_O1_O3_P9"}:
            from .replay_reference_data import ReplayReferenceDataStore
            reference_store = reference_store or ReplayReferenceDataStore(
                self._root / "replay_reference_data")
        action = runner.next_action()
        if operation == "EXECUTE_L0_PILOT_P4":
            while action["action"] == "EXECUTE_SCIENTIFIC_NODE" and action["contract"]["layer"] == "L0":
                node = runner.nodes[action["node_id"]]
                result = execute_l0_source_integrity_node(
                    node=node,
                    catalogue=catalogue,
                    repository_root=Path(__file__).resolve().parents[1],
                    attempt=action["attempt"],
                    inject_stop=parameters["pilot_fault_injection"] == action["node_id"],
                )
                records = result.pop("reference_records")
                for record in records:
                    reference_store.put(record)
                result["reference_manifest_sha256"] = reference_store.manifest["manifest_sha256"]
                recorded = runner.record_node_result(result)
                if recorded["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                    action = recorded
                    break
                action = runner.next_action()
            if action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                next_node_id = action["audit_capsule"]["stopped_node_id"]
                frontier = seal_l0_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=next_node_id, waiting_for_audit=True)
                reference_store.put(frontier)
                disposition = "WAITING_FOR_EXTERNAL_AUDIT"
            elif action["action"] == "EXECUTE_SCIENTIFIC_NODE" and action["contract"]["layer"] != "L0":
                frontier = seal_l0_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["node_id"], waiting_for_audit=False)
                reference_store.put(frontier)
                disposition = "L0_PILOT_COMPLETE"
            else:
                raise StageRuntimeError("P4_L0_FRONTIER_NOT_REACHED")
        elif operation == "EXECUTE_C0_HISTORICAL_P5":
            from .replay_c0_historical_executor import execute_c0_historical_node, seal_c0_frontier
            while (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                   and action["contract"]["layer"] == "C0_HISTORICAL"):
                node = runner.nodes[action["node_id"]]
                result = execute_c0_historical_node(
                    node=node, catalogue=catalogue,
                    repository_root=Path(__file__).resolve().parents[1],
                    attempt=action["attempt"],
                    inject_stop=parameters["pilot_fault_injection"] == action["node_id"],
                )
                records = result.pop("reference_records")
                for record in records:
                    reference_store.put(record)
                result["reference_manifest_sha256"] = reference_store.manifest["manifest_sha256"]
                recorded = runner.record_node_result(result)
                if recorded["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                    action = recorded
                    break
                action = runner.next_action()
            if action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                next_node_id = action["audit_capsule"]["stopped_node_id"]
                frontier = seal_c0_frontier(runner_state=runner.state, manifest=manifest,
                                            next_node_id=next_node_id, waiting_for_audit=True)
                reference_store.put(frontier)
                disposition = "WAITING_FOR_EXTERNAL_AUDIT"
            elif (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                  and action["contract"]["layer"] != "C0_HISTORICAL"):
                frontier = seal_c0_frontier(runner_state=runner.state, manifest=manifest,
                                            next_node_id=action["node_id"], waiting_for_audit=False)
                reference_store.put(frontier)
                disposition = "C0_HISTORICAL_PILOT_COMPLETE"
            else:
                raise StageRuntimeError("P5_C0_FRONTIER_NOT_REACHED")
        elif operation == "EXECUTE_L2J3_P6":
            from .replay_l2j3_executor import execute_l2j3_node, seal_l2j3_frontier
            while (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                   and action["contract"]["layer"] == "L2J3"):
                node = runner.nodes[action["node_id"]]
                result = execute_l2j3_node(
                    node=node, catalogue=catalogue,
                    repository_root=Path(__file__).resolve().parents[1],
                    attempt=action["attempt"],
                    inject_stop=parameters["pilot_fault_injection"] == action["node_id"],
                )
                for record in result.pop("reference_records"):
                    reference_store.put(record)
                result["reference_manifest_sha256"] = reference_store.manifest["manifest_sha256"]
                recorded = runner.record_node_result(result)
                if recorded["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                    action = recorded
                    break
                action = runner.next_action()
            if action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                frontier = seal_l2j3_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["audit_capsule"]["stopped_node_id"],
                    waiting_for_audit=True)
                reference_store.put(frontier)
                disposition = "WAITING_FOR_EXTERNAL_AUDIT"
            elif (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                  and action["contract"]["layer"] == "NODE_IN"):
                frontier = seal_l2j3_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["node_id"], waiting_for_audit=False)
                reference_store.put(frontier)
                disposition = "L2J3_HISTORICAL_PILOT_COMPLETE"
            else:
                raise StageRuntimeError("P6_L2J3_FRONTIER_NOT_REACHED")
        elif operation == "EXECUTE_NODE_IN_P7":
            from .replay_node_in_executor import execute_node_in, seal_frontier
            while (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                   and action["contract"]["layer"] == "NODE_IN"):
                node = runner.nodes[action["node_id"]]
                result = execute_node_in(
                    node=node, catalogue=catalogue,
                    repository_root=Path(__file__).resolve().parents[1],
                    attempt=action["attempt"],
                    inject_stop=parameters["pilot_fault_injection"] == action["node_id"],
                )
                for record in result.pop("reference_records"):
                    reference_store.put(record)
                result["reference_manifest_sha256"] = reference_store.manifest["manifest_sha256"]
                recorded = runner.record_node_result(result)
                if recorded["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                    action = recorded
                    break
                action = runner.next_action()
            if action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                frontier = seal_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["audit_capsule"]["stopped_node_id"],
                    waiting_for_audit=True)
                reference_store.put(frontier)
                disposition = "WAITING_FOR_EXTERNAL_AUDIT"
            elif (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                  and action["contract"]["layer"] != "NODE_IN"):
                frontier = seal_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["node_id"], waiting_for_audit=False)
                reference_store.put(frontier)
                disposition = "NODE_IN_PILOT_COMPLETE"
            else:
                raise StageRuntimeError("P7_NODE_IN_FRONTIER_NOT_REACHED")
        elif operation == "EXECUTE_SCOUT_HISTORICAL_P8":
            from .replay_scout_historical_executor import execute_scout_historical, seal_frontier
            while (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                   and action["contract"]["layer"] == "SCOUT_HISTORICAL"):
                node = runner.nodes[action["node_id"]]
                result = execute_scout_historical(
                    node=node, catalogue=catalogue,
                    repository_root=Path(__file__).resolve().parents[1],
                    attempt=action["attempt"],
                    inject_stop=parameters["pilot_fault_injection"] == action["node_id"],
                )
                for record in result.pop("reference_records"):
                    reference_store.put(record)
                result["reference_manifest_sha256"] = reference_store.manifest["manifest_sha256"]
                recorded = runner.record_node_result(result)
                if recorded["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                    action = recorded
                    break
                action = runner.next_action()
            if action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                frontier = seal_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["audit_capsule"]["stopped_node_id"],
                    waiting_for_audit=True)
                reference_store.put(frontier)
                disposition = "WAITING_FOR_EXTERNAL_AUDIT"
            elif (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                  and action["contract"]["layer"] != "SCOUT_HISTORICAL"):
                frontier = seal_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["node_id"], waiting_for_audit=False)
                reference_store.put(frontier)
                disposition = "SCOUT_HISTORICAL_PILOT_COMPLETE"
            else:
                raise StageRuntimeError("P8_SCOUT_FRONTIER_NOT_REACHED")
        elif operation == "EXECUTE_O1_O3_P9":
            from .replay_o1_o3_executor import execute_o1_o3, seal_frontier
            while (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                   and action["contract"]["layer"] == "O1_O3"):
                node = runner.nodes[action["node_id"]]
                result = execute_o1_o3(
                    node=node, catalogue=catalogue,
                    repository_root=Path(__file__).resolve().parents[1],
                    attempt=action["attempt"],
                    inject_stop=parameters["pilot_fault_injection"] == action["node_id"],
                )
                for record in result.pop("reference_records"):
                    reference_store.put(record)
                result["reference_manifest_sha256"] = reference_store.manifest["manifest_sha256"]
                recorded = runner.record_node_result(result)
                if recorded["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                    action = recorded
                    break
                action = runner.next_action()
            if action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
                frontier = seal_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["audit_capsule"]["stopped_node_id"],
                    waiting_for_audit=True)
                reference_store.put(frontier)
                disposition = "WAITING_FOR_EXTERNAL_AUDIT"
            elif (action["action"] == "EXECUTE_SCIENTIFIC_NODE"
                  and action["contract"]["layer"] != "O1_O3"):
                frontier = seal_frontier(
                    runner_state=runner.state, manifest=manifest,
                    next_node_id=action["node_id"], waiting_for_audit=False)
                reference_store.put(frontier)
                disposition = "O1_O3_PILOT_COMPLETE"
            else:
                raise StageRuntimeError("P9_O1_O3_FRONTIER_NOT_REACHED")
        elif action["action"] == "EXECUTE_SCIENTIFIC_NODE":
            disposition = "WAITING_FOR_NODE_EXECUTOR_BINDING"
        elif action["action"] == "WAIT_FOR_EXTERNAL_AUDIT":
            disposition = "WAITING_FOR_EXTERNAL_AUDIT"
        else:
            disposition = action["action"]
        status = {
            "schema_id": "IG_REPLAY_REGISTERED_ROOT_STATUS_V1",
            "outcome": disposition,
            "root_run_id": parameters["root_run_id"],
            "manifest_dag_sha256": manifest["dag_sha256"],
            "runner_state_sha256": runner.state["state_sha256"],
            "runner_action": action,
            "completed_node_ids": list(runner.state["completed_node_ids"]),
            "authorized_replay_through_layer": "G8",
            "next_wait_layer": "GLOBAL",
            "science_executed": (reference_store.manifest["science_executed"]
                                 if reference_store is not None else False),
            "science_authority_effect": "NONE",
            "implementation_status": (
                "P4_L0_SOURCE_INTEGRITY_EXECUTORS_BOUND" if operation == "EXECUTE_L0_PILOT_P4"
                else "P5_C0_HISTORICAL_EVIDENCE_EXECUTORS_BOUND"
                if operation == "EXECUTE_C0_HISTORICAL_P5"
                else "P6_L2J3_HISTORICAL_EVIDENCE_EXECUTORS_BOUND"
                if operation == "EXECUTE_L2J3_P6"
                else "P7_NODE_IN_FRESH_RECOMPUTATION_BOUND"
                if operation == "EXECUTE_NODE_IN_P7"
                else "P8_SCOUT_HISTORICAL_EVIDENCE_BOUND"
                if operation == "EXECUTE_SCOUT_HISTORICAL_P8"
                else "P9_O1_O3_HISTORICAL_EVIDENCE_BOUND"
                if operation == "EXECUTE_O1_O3_P9"
                else "P2B_REGISTERED_ROOT_NO_NODE_EXECUTORS"),
        }
        if reference_store is not None:
            status.update({
                "reference_manifest_sha256": reference_store.manifest["manifest_sha256"],
                "reference_record_count": len(reference_store.manifest["record_ids"]),
                "frontier_record_sha256": frontier["record_sha256"],
                "fresh_empty_root_science_node_count": 0,
            })
            if operation == "EXECUTE_L0_PILOT_P4":
                status["source_integrity_nodes_executed"] = len([
                    node_id for node_id in runner.state["completed_node_ids"] if node_id in L0_NODE_IDS])
            elif operation == "EXECUTE_C0_HISTORICAL_P5":
                status["historical_evidence_nodes_executed"] = len([
                    node_id for node_id in runner.state["completed_node_ids"] if node_id in C0_NODE_IDS])
            elif operation == "EXECUTE_L2J3_P6":
                status["historical_evidence_nodes_executed"] = len([
                    node_id for node_id in runner.state["completed_node_ids"] if node_id in L2J3_NODE_IDS])
            elif operation == "EXECUTE_NODE_IN_P7":
                status["historical_evidence_nodes_executed"] = len([
                    node_id for node_id in runner.state["completed_node_ids"] if node_id in NODE_IDS]) - int(
                        NODE_IDS[1] in runner.state["completed_node_ids"])
                status["fresh_empty_root_science_node_count"] = int(
                    NODE_IDS[1] in runner.state["completed_node_ids"])
            elif operation == "EXECUTE_SCOUT_HISTORICAL_P8":
                status["historical_evidence_nodes_executed"] = len([
                    node_id for node_id in runner.state["completed_node_ids"] if node_id in NODE_IDS])
                status["fresh_empty_root_science_node_count"] = 1
                status["fresh_science_nodes_executed_this_stage"] = 0
            elif operation == "EXECUTE_O1_O3_P9":
                status["historical_evidence_nodes_executed"] = len([
                    node_id for node_id in runner.state["completed_node_ids"] if node_id in NODE_IDS])
                status["fresh_empty_root_science_node_count"] = 1
                status["fresh_science_nodes_executed_this_stage"] = 0
        publication = self.publish_json("REPLAY_ROOT_STATUS", status)
        result = dict(status, status_artifact=publication)
        if disposition.startswith("WAITING_FOR_"):
            # Raising leaves the registered attempt PAUSED rather than writing a
            # terminal completion that future calls would incorrectly reuse.
            raise ReplayRootPaused(disposition, publication)
        return result

    def dependency_commit(self, stage_id: str) -> dict[str, Any]:
        self._require_execution()
        if self._dependency_loader is None:
            raise ExecutionAuthorityError("CONTROLLER_DEPENDENCY_LOADER_REQUIRED")
        obj = self._dependency_loader(stage_id)
        if obj is None:
            raise StageRuntimeError(f"missing dependency commit {stage_id}")
        return obj

    def publish_json(self, logical_name: str, obj: Any) -> dict[str, Any]:
        self._require_execution()
        name = _phase_safe(logical_name)
        p = self._root / "artifacts" / f"{name}.json"
        from .v05_origin_guard import require_registered_output
        require_registered_output(p, "publish_json")
        p.parent.mkdir(parents=True, exist_ok=True)
        from .preservation import poll
        from .canon import canonical_text
        poll(extra_bytes=len(canonical_text(obj,pretty=True).encode('utf-8')))
        write_json_atomic(p, obj)
        self._enforce_execution_budgets(self._root)
        return {"logical_name": logical_name, "path": str(p), "sha256": canonical_sha256(obj), "size_bytes": p.stat().st_size}

    def _phase_root(self, phase_id: str) -> Path:
        self._require_execution()
        p = self._root / "phases" / _phase_safe(phase_id)
        from .v05_origin_guard import require_registered_output
        require_registered_output(p, "stage-phase")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _open_db(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
              task_id TEXT PRIMARY KEY,
              payload_sha256 TEXT NOT NULL,
              signature_sha256 TEXT NOT NULL,
              class_token TEXT NOT NULL,
              outcome_count INTEGER NOT NULL,
              metrics_json TEXT NOT NULL,
              locator_json TEXT,
              committed_utc TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classes (
              class_token TEXT PRIMARY KEY,
              signature_sha256 TEXT NOT NULL,
              class_index INTEGER NOT NULL,
              representative_task_id TEXT NOT NULL,
              size INTEGER NOT NULL,
              representative_signature_bytes BLOB
            )
        """)
        # A19: exact representative bytes are durable engine evidence.  Existing
        # pre-A19 stores migrate additively; NULL means one legacy re-open may be
        # required before the row is backfilled.
        class_columns = {str(r[1]) for r in conn.execute("PRAGMA table_info(classes)")}
        if "representative_signature_bytes" not in class_columns:
            conn.execute("ALTER TABLE classes ADD COLUMN representative_signature_bytes BLOB")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_classes_digest ON classes(signature_sha256)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_class ON task_results(class_token)")
        conn.commit()
        return conn

    @staticmethod
    def _finalize_idle_partition_database(path: Path) -> None:
        """Finalize an idle task database as one rollback-journal image.

        SQLite owns checkpointing and journal retirement. A competing handle
        makes the mode transition fail; never unlink a WAL as a repair.
        Called only before completion sealing, never on admitted old evidence.
        """
        with closing(sqlite3.connect(path, timeout=5)) as finalizer:
            if finalizer.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise StageRuntimeError("task database failed final integrity check")
            mode = finalizer.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            if mode.lower() != "delete":
                raise StageRuntimeError("task database could not leave WAL mode")

    def _status(self, phase_root: Path, **fields: Any) -> None:
        base = {
            "schema_id": STAGE_RUNTIME_STATUS_SCHEMA,
            "chain_id": self.chain_id,
            "stage_id": self.stage_id,
            "question_sha256": self.question_sha256,
            "updated_utc": utc_now(),
            **fields,
        }
        base["status_sha256"] = canonical_sha256(base)
        write_json_atomic(phase_root / "RUNTIME_STATUS.json", base)

    @staticmethod
    def _task_manifest(tasks: list[TaskSpec]) -> list[dict[str, str]]:
        return [
            {"task_id": t.task_id, "binding_sha256": t.binding_sha256, "payload_sha256": canonical_sha256(t.payload)}
            for t in sorted(tasks, key=lambda x: x.task_id)
        ]

    @staticmethod
    def _partition_bindings_sha256(conn: sqlite3.Connection) -> str:
        """Hash sorted task->class bindings without materializing them in RAM."""
        h = hashlib.sha256()
        h.update(b"IG_DECODER_V05_PARTITION_BINDINGS_STREAM_V1\n")
        for task_id, class_token in conn.execute("SELECT task_id,class_token FROM task_results ORDER BY task_id"):
            h.update(canonical_text([task_id, class_token]).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _enforce_execution_budgets(self, phase_root: Path) -> tuple[int, int]:
        rss = _rss_bytes()
        evidence = _stage_workspace_bytes(self._root)
        if self.memory_budget_bytes is not None and rss > self.memory_budget_bytes:
            raise StageRuntimeError(
                f"stage reducer memory budget exceeded {rss} > {self.memory_budget_bytes}"
            )
        if self.workspace_budget_bytes is not None and evidence > self.workspace_budget_bytes:
            raise StageRuntimeError(
                f"stage workspace budget exceeded {evidence} > {self.workspace_budget_bytes}"
            )
        from .preservation import poll, safe_point
        poll();safe_point('COMMITTED_STAGE_BOUNDARY')
        return rss, evidence

    def _bind_scope(self, conn: sqlite3.Connection, *, phase_id: str, tasks: list[TaskSpec], evaluator_ref: str) -> str:
        manifest = self._task_manifest(tasks)
        base = {
            "schema_id": PARTITION_DB_SCHEMA,
            "chain_id": self.chain_id,
            "stage_id": self.stage_id,
            "phase_id": phase_id,
            "question_sha256": self.question_sha256,
            "source_sha256": live_source_sha256(),
            "evaluator_ref": evaluator_ref,
            "task_count": len(tasks),
            "task_manifest_sha256": canonical_sha256(manifest),
        }
        scope_sha = canonical_sha256(base)
        row = conn.execute("SELECT v FROM meta WHERE k='scope'").fetchone()
        encoded = canonical_text(dict(base, scope_sha256=scope_sha))
        if row is None:
            conn.execute("INSERT INTO meta(k,v) VALUES('scope',?)", (encoded,)); conn.commit()
        elif row[0] != encoded:
            raise StageRuntimeError("stage-runtime durable scope mismatch; refusing stale resume state")
        # Verify any existing task bindings against the current deterministic task set.
        by_id = {x["task_id"]: x for x in manifest}
        for tid, psha in conn.execute("SELECT task_id,payload_sha256 FROM task_results"):
            if tid not in by_id or by_id[tid]["payload_sha256"] != psha:
                raise StageRuntimeError(f"durable task binding mismatch for {tid}")
        return scope_sha

    def _evaluate_signature(self, evaluator_ref: str, payload: Any) -> Any:
        self._execution_permit.require_evaluator(evaluator_ref)
        fn = _resolve_ref(evaluator_ref, self._project_callable_bindings.get(evaluator_ref))
        raw = fn(payload)
        if not isinstance(raw, Mapping) or "signature" not in raw:
            raise StageRuntimeError("partition evaluator missing signature during structural fallback")
        return raw["signature"]

    def _evaluate_signature_bytes(self, evaluator_ref: str, payload: Any) -> bytes:
        """Re-open an equal-digest candidate using exact canonical bytes.

        Digest equality is never scientific equality.  This method supplies the
        exact V1 canonical byte string used for the structural fallback without
        retaining the full signature object in the reducer.
        """
        return structural_canonical_bytes(self._evaluate_signature(evaluator_ref, payload))

    @staticmethod
    def _finalize_collision_class_tokens(conn: sqlite3.Connection, signature_bytes_for_task) -> int:
        """Canonicalize class indices for genuine/ejected digest-collision buckets.

        Provisional indices are allowed during streaming commits.  Before a
        scientific partition hash is emitted, every multi-class digest bucket is
        sorted by exact canonical signature bytes, and task bindings are remapped
        to the resulting deterministic ``digest:index`` tokens.  This erases
        worker completion order without making a hash an equality oracle.
        """
        buckets = [r[0] for r in conn.execute(
            "SELECT signature_sha256 FROM classes GROUP BY signature_sha256 HAVING COUNT(*)>1 ORDER BY signature_sha256"
        )]
        remapped = 0
        for digest in buckets:
            rows = list(conn.execute(
                "SELECT class_token,class_index,representative_task_id,size,representative_signature_bytes FROM classes WHERE signature_sha256=?",
                (digest,),
            ))
            keyed = []
            for token, old_index, rep_tid, size, stored_sig in rows:
                sig_bytes = bytes(stored_sig) if stored_sig is not None else signature_bytes_for_task(rep_tid)
                if stored_sig is None:
                    conn.execute(
                        "UPDATE classes SET representative_signature_bytes=? WHERE class_token=?",
                        (sqlite3.Binary(sig_bytes), token),
                    )
                keyed.append((sig_bytes, str(token), int(old_index), str(rep_tid), int(size)))
            keyed.sort(key=lambda x: (x[0], x[3]))
            mapping = []
            for new_index, (_sig_bytes, old_token, _old_index, _rep_tid, _size) in enumerate(keyed):
                new_token = f"{digest}:{new_index}"
                mapping.append((old_token, new_token, new_index))
            if all(old == new and int(old_idx) == idx for (old, new, idx), (_b,_t,old_idx,_r,_sz) in zip(mapping, keyed)):
                continue
            # Two-step remap avoids PRIMARY KEY collisions when indices swap.
            temps = []
            for j, (old_token, new_token, new_index) in enumerate(mapping):
                temp = f"__E2_TMP__:{digest}:{j}"
                conn.execute("UPDATE task_results SET class_token=? WHERE class_token=?", (temp, old_token))
                conn.execute("UPDATE classes SET class_token=? WHERE class_token=?", (temp, old_token))
                temps.append((temp, new_token, new_index))
            for temp, new_token, new_index in temps:
                conn.execute("UPDATE classes SET class_token=?, class_index=? WHERE class_token=?",
                             (new_token, new_index, temp))
                conn.execute("UPDATE task_results SET class_token=? WHERE class_token=?", (new_token, temp))
                remapped += 1
        conn.commit()
        return remapped

    def run_structural_partition(
        self,
        *,
        phase_id: str,
        tasks: Iterable[TaskSpec],
        evaluator_ref: str,
        requested_workers: int | str | None = None,
        max_tasks: int | None = None,
        stream_shard_task_limit: int | None = None,
    ) -> StructuralPartitionResult:
        """Evaluate and durably reduce a structural observer partition.

        Task workers return compact signature digests/counts plus the exact
        canonical signature bytes already computed for that task. Different
        digests imply different canonical encodings. Equal digests never decide
        equality: the controller compares exact canonical bytes before assigning a
        class. One exact representative byte string per class is stored durably in
        the engine-owned partition database. A legacy pre-A19 row with no stored
        bytes may be re-opened once under the evaluator-declared KernelView and is
        then backfilled; fresh current representatives are never re-evaluated by
        the controller.
        """
        self._require_execution()
        runtime_call_started = time.perf_counter()
        tasks = sorted(list(tasks), key=lambda t: t.task_id)
        if max_tasks is not None and len(tasks) > int(max_tasks):
            raise StageRuntimeError(f"task budget exceeded {len(tasks)} > {max_tasks}")
        self._execution_permit.require_evaluator(evaluator_ref)
        evaluator = _resolve_ref(evaluator_ref, self._project_callable_bindings.get(evaluator_ref))
        gate = require_controller_only_callable(evaluator, role="scientific worker evaluator")
        phase_root = self._phase_root(phase_id)
        db_path = phase_root / "partition.sqlite3"
        conn = self._open_db(db_path)
        phase_complete = False
        try:
            admission_started = time.perf_counter()
            scope_sha = self._bind_scope(conn, phase_id=phase_id, tasks=tasks, evaluator_ref=evaluator_ref)
            # Admission is fail-closed before scientific task execution.
            self._enforce_execution_budgets(phase_root)
            admission_wall_seconds = time.perf_counter() - admission_started
            task_by_id = {t.task_id: t for t in tasks}
            done = {r[0] for r in conn.execute("SELECT task_id FROM task_results")}
            pending = [t for t in tasks if t.task_id not in done]
            controller_fallback_kernel_view_bound = False

            def ensure_controller_fallback_kernel_view() -> None:
                nonlocal controller_fallback_kernel_view_bound
                if not controller_fallback_kernel_view_bound:
                    _bind_controller_fallback_kernel_view(evaluator_ref, scope_sha)
                    controller_fallback_kernel_view_bound = True

            rep_signature_cache: dict[str, bytes] = {}
            controller_classification_wall_seconds = 0.0
            controller_structural_fallback_wall_seconds = 0.0
            controller_commit_wall_seconds = 0.0
            controller_representative_durable_hits = 0
            controller_representative_fallback_evaluations = 0
            controller_representative_bytes_backfilled = 0
            commits_since_flush = 0
            committed_this_invocation = 0
            peak_reducer_rss = _rss_bytes()
            peak_evidence_bytes = _runtime_workspace_bytes(phase_root)
            fault_after_raw = os.environ.get("IG_V05_ENGINEERING_ABORT_AFTER_COMMITS")
            fault_after = None
            if fault_after_raw is not None:
                if os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") != "STAGE_RUNTIME_ABORT_TEST":
                    raise StageRuntimeError("engineering abort injection requires explicit acknowledgement")
                fault_after = max(1, int(fault_after_raw))
            started = time.monotonic()
            self._status(
                phase_root, status="RUNNING", task_count=len(tasks), completed=len(done), pending=len(pending),
                workers_requested=requested_workers if requested_workers is not None else self.default_workers,
                architecture_gate=gate,
            )

            def class_for(tid: str, result: Mapping[str, Any]) -> str:
                nonlocal commits_since_flush, committed_this_invocation, peak_reducer_rss, peak_evidence_bytes
                nonlocal controller_classification_wall_seconds, controller_structural_fallback_wall_seconds, controller_commit_wall_seconds
                nonlocal controller_representative_durable_hits, controller_representative_fallback_evaluations, controller_representative_bytes_backfilled
                classification_started = time.perf_counter()
                fallback_started_total = 0.0
                task = task_by_id[tid]
                digest = str(result["signature_sha256"])
                cur_sig = result.get("signature_canonical_bytes")
                if type(cur_sig) is not bytes:
                    raise StageRuntimeError("partition worker missing exact signature canonical bytes")
                if not bool(result.get("engineering_digest_override", False)) and hashlib.sha256(cur_sig).hexdigest() != digest:
                    raise StageRuntimeError("partition worker signature digest/bytes mismatch")
                classes = list(conn.execute(
                    "SELECT class_token,class_index,representative_task_id,representative_signature_bytes FROM classes WHERE signature_sha256=? ORDER BY class_index",
                    (digest,),
                ))
                chosen = None
                if not classes:
                    token = f"{digest}:0"
                    conn.execute(
                        "INSERT INTO classes(class_token,signature_sha256,class_index,representative_task_id,size,representative_signature_bytes) VALUES(?,?,?,?,1,?)",
                        (token, digest, 0, tid, sqlite3.Binary(cur_sig)),
                    )
                    chosen = token
                    if len(rep_signature_cache) < 256:
                        rep_signature_cache[token] = cur_sig
                else:
                    # Hash equality is only an index hit. The exact canonical bytes
                    # produced by the worker are the scientific equality authority.
                    # Only a representative restored from durable pre-existing state
                    # may require one controller-side re-open to seed the ephemeral cache.
                    for token, _idx, rep_tid, stored_sig in classes:
                        rep_sig = rep_signature_cache.get(token)
                        if rep_sig is None:
                            if stored_sig is not None:
                                rep_sig = bytes(stored_sig)
                                controller_representative_durable_hits += 1
                            else:
                                # Legacy pre-A19 row: one exact controller re-open is
                                # permitted, but only under this evaluator's declared view.
                                ensure_controller_fallback_kernel_view()
                                fb0 = time.perf_counter()
                                rep_sig = self._evaluate_signature_bytes(evaluator_ref, task_by_id[rep_tid].payload)
                                fallback_started_total += time.perf_counter() - fb0
                                controller_representative_fallback_evaluations += 1
                                conn.execute(
                                    "UPDATE classes SET representative_signature_bytes=? WHERE class_token=?",
                                    (sqlite3.Binary(rep_sig), token),
                                )
                                controller_representative_bytes_backfilled += len(rep_sig)
                            # Ephemeral cache is a bounded read acceleration only.  Exact
                            # no-repeat behavior comes from the durable representative BLOB.
                            if len(rep_signature_cache) < 256:
                                rep_signature_cache[token] = rep_sig
                        if rep_sig == cur_sig:
                            chosen = token
                            if len(rep_signature_cache) < 256:
                                rep_signature_cache[token] = rep_sig
                            conn.execute(
                                "UPDATE classes SET size=size+1, representative_task_id=CASE WHEN representative_task_id>? THEN ? ELSE representative_task_id END WHERE class_token=?",
                                (tid, tid, token),
                            )
                            break
                    if chosen is None:
                        idx = max(int(x[1]) for x in classes) + 1
                        token = f"{digest}:{idx}"
                        conn.execute(
                            "INSERT INTO classes(class_token,signature_sha256,class_index,representative_task_id,size,representative_signature_bytes) VALUES(?,?,?,?,1,?)",
                            (token, digest, idx, tid, sqlite3.Binary(cur_sig)),
                        )
                        chosen = token
                        if len(rep_signature_cache) < 256:
                            rep_signature_cache[token] = cur_sig
                        # This is a real cryptographic digest collision witness. Persist
                        # exact structures only here, never for the normal unique case.
                        witness = {
                            "schema_id": "IG_DECODER_V05_STRUCTURAL_DIGEST_COLLISION_WITNESS_V1",
                            "signature_sha256": digest,
                            "new_task_id": tid,
                            "new_signature_canonical_utf8": cur_sig.decode("utf-8"),
                            "existing_classes": [x[0] for x in classes],
                        }
                        wp = phase_root / "digest_collision_witnesses"
                        wp.mkdir(parents=True, exist_ok=True)
                        write_json_atomic(wp / f"{digest}-{idx}.json", witness)
                controller_structural_fallback_wall_seconds += fallback_started_total
                classification_elapsed = time.perf_counter() - classification_started
                controller_classification_wall_seconds += classification_elapsed
                row_metrics = dict(result.get("metrics") or {})
                row_metrics["engine_controller_classification_wall_seconds"] = classification_elapsed
                row_metrics["engine_controller_structural_fallback_wall_seconds"] = fallback_started_total
                insert_started = time.perf_counter()
                conn.execute(
                    "INSERT INTO task_results(task_id,payload_sha256,signature_sha256,class_token,outcome_count,metrics_json,locator_json,committed_utc) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        tid, canonical_sha256(task.payload), digest, chosen,
                        int(result.get("outcome_count", 0)),
                        canonical_text(row_metrics), None, utc_now(),
                    ),
                )
                controller_commit_wall_seconds += time.perf_counter() - insert_started
                commits_since_flush += 1
                committed_this_invocation += 1
                if fault_after is not None and committed_this_invocation >= fault_after:
                    c0=time.perf_counter(); conn.commit(); controller_commit_wall_seconds += time.perf_counter()-c0
                    raise StageRuntimeError(
                        f"ENGINEERING_FAULT_INJECTION_ABORT_AFTER_{committed_this_invocation}_COMMITS"
                    )
                if commits_since_flush >= 32:
                    c0=time.perf_counter(); conn.commit(); controller_commit_wall_seconds += time.perf_counter()-c0; commits_since_flush = 0
                    complete = int(conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0])
                    rss, evidence = self._enforce_execution_budgets(phase_root)
                    peak_reducer_rss = max(peak_reducer_rss, rss)
                    peak_evidence_bytes = max(peak_evidence_bytes, evidence)
                    self._status(
                        phase_root, status="RUNNING", task_count=len(tasks), completed=complete,
                        pending=len(tasks)-complete, elapsed_seconds=time.monotonic()-started,
                        reducer_rss_bytes=rss, evidence_bytes=evidence,
                    )
                return chosen

            exec_meta: dict[str, Any]
            telemetry = RuntimeTelemetry(
                phase_root / "telemetry",
                experiment_id=f"{self.chain_id}:{self.stage_id}:{phase_id}",
                workers_requested=requested_workers if requested_workers is not None else self.default_workers,
                wall_interval_seconds=5.0,
            )
            telemetry.phase("SCIENCE_CENSUS", tasks_total=len(pending), kernels_total=len(pending))
            try:
                if pending:
                    policy = ExecutionPolicy(
                        backend="AUTO", requested_workers=requested_workers if requested_workers is not None else self.default_workers,
                        scheduler="COST_WEIGHTED_SHARDS",
                        stream_shard_task_limit=(24 if stream_shard_task_limit is None else max(1, int(stream_shard_task_limit))),
                        owner=f"stage-runtime:{self.chain_id}:{self.stage_id}:{phase_id}",
                    )
                    exec_meta = execute_tasks_stream(
                        pending,
                        worker_ref="infinity_grid.v05_stage_runtime:_partition_worker",
                        initializer_ref="infinity_grid.v05_stage_runtime:_init_partition_worker",
                        initializer_payload={
                            "evaluator_ref": evaluator_ref,
                            "project_callable_binding": self._project_callable_bindings.get(evaluator_ref),
                            "scope_sha256": scope_sha,
                            "engineering_test_digest_override": (
                                os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK")
                                == "STAGE_RUNTIME_DIGEST_COLLISION_TEST"
                            ),
                        },
                        policy=policy,
                        on_result=class_for,
                        telemetry=telemetry,
                    )
                    c0=time.perf_counter(); conn.commit(); controller_commit_wall_seconds += time.perf_counter()-c0
                else:
                    exec_meta = {"backend": "REUSE", "workers": 0, "task_count": 0, "completed_task_count": 0, "wall_seconds": 0.0, "streaming_results": True}
                telemetry.phase("MERGE", tasks_total=len(tasks), kernels_total=0)
            except BaseException as exc:
                telemetry.fail(exc)
                raise

            finalize_started = time.perf_counter()
            def finalize_legacy_signature_bytes(tid: str) -> bytes:
                nonlocal controller_structural_fallback_wall_seconds, controller_representative_fallback_evaluations
                ensure_controller_fallback_kernel_view()
                fb0 = time.perf_counter()
                value = self._evaluate_signature_bytes(evaluator_ref, task_by_id[tid].payload)
                controller_structural_fallback_wall_seconds += time.perf_counter() - fb0
                controller_representative_fallback_evaluations += 1
                return value

            deterministic_collision_class_remaps = self._finalize_collision_class_tokens(
                conn, finalize_legacy_signature_bytes
            )
            if os.environ.get("IG_V05_ENGINEERING_ABORT_AFTER_REDUCE_FINALIZE") is not None:
                if os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") != "STAGE_RUNTIME_REDUCE_ABORT_TEST":
                    raise StageRuntimeError("engineering reduce abort injection requires explicit acknowledgement")
                raise StageRuntimeError("ENGINEERING_FAULT_INJECTION_ABORT_AFTER_REDUCE_FINALIZE")
            total = int(conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0])
            if total != len(tasks):
                raise StageRuntimeError(f"partition coverage mismatch {total} != {len(tasks)}")
            class_count = int(conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0])
            multi_count = int(conn.execute("SELECT COUNT(*) FROM classes WHERE size>1").fetchone()[0])
            max_class_size = int(conn.execute("SELECT COALESCE(MAX(size),0) FROM classes").fetchone()[0])
            first_multi_row = conn.execute(
                "SELECT class_token,size,representative_task_id FROM classes WHERE size>1 ORDER BY class_token LIMIT 1"
            ).fetchone()
            summary_base = {
                "schema_id": PARTITION_SUMMARY_SCHEMA,
                "chain_id": self.chain_id,
                "stage_id": self.stage_id,
                "phase_id": phase_id,
                "question_sha256": self.question_sha256,
                "task_count": total,
                "class_count": class_count,
                "multi_class_count": multi_count,
                "max_class_size": max_class_size,
                "first_multi_class": None if first_multi_row is None else {
                    "class_token": first_multi_row[0], "size": int(first_multi_row[1]), "representative_task_id": first_multi_row[2]
                },
                "partition_bindings_sha256": self._partition_bindings_sha256(conn),
                "partition_binding_hash_schema": "IG_DECODER_V05_PARTITION_BINDINGS_STREAM_V1",
                "structural_equality_used_for_equal_digest": True,
                "different_digest_implies_different_canonical_encoding": True,
                "hash_equality_never_decides_scientific_equality": True,
                "deterministic_collision_class_ordering": "EXACT_CANONICAL_BYTES_ASCENDING_V1",
            }
            summary = dict(summary_base, summary_sha256=canonical_sha256(summary_base))
            rss, evidence = self._enforce_execution_budgets(phase_root)
            peak_reducer_rss = max(peak_reducer_rss, rss)
            peak_evidence_bytes = max(peak_evidence_bytes, evidence)
            finalize_verify_wall_seconds = time.perf_counter() - finalize_started
            worker_metric_totals: dict[str, float] = {}
            worker_metric_categories: dict[str, Any] = {}
            for (metrics_text,) in conn.execute("SELECT metrics_json FROM task_results ORDER BY task_id"):
                for key, value in json.loads(metrics_text).items():
                    if type(value) in (int, float) and not isinstance(value, bool):
                        worker_metric_totals[key] = worker_metric_totals.get(key, 0.0) + float(value)
                    elif key not in worker_metric_categories:
                        worker_metric_categories[key] = value
            phase_timing_seconds = {
                "admission": admission_wall_seconds,
                "engine_execution_including_stream_reduce": float(exec_meta.get("wall_seconds", 0.0)),
                "controller_classification": controller_classification_wall_seconds,
                "controller_structural_fallback": controller_structural_fallback_wall_seconds,
                "controller_commit": controller_commit_wall_seconds,
                "finalize_verify": finalize_verify_wall_seconds,
                "runtime_call_total_to_summary": time.perf_counter() - runtime_call_started,
            }
            exec_meta = dict(exec_meta,
                durable_store="SQLITE_COMPACT_ENGINE_OWNED",
                durable_scope_sha256=scope_sha,
                evidence_bytes=evidence,
                evidence_bytes_peak=peak_evidence_bytes,
                reducer_rss_bytes_final=rss,
                reducer_rss_bytes_peak=peak_reducer_rss,
                structural_signature_encoder_id=STRUCTURAL_ENCODER_ID,
                workspace_accounting_mode="CLOSED_LAYOUT_INCREMENTAL_FILE_SIZES_V1",
                deterministic_collision_class_remaps=deterministic_collision_class_remaps,
                representative_signature_storage="SQLITE_CLASSES_BLOB_V1",
                controller_representative_durable_hits=controller_representative_durable_hits,
                controller_representative_fallback_evaluations=controller_representative_fallback_evaluations,
                controller_representative_bytes_backfilled=controller_representative_bytes_backfilled,
                worker_metric_totals=worker_metric_totals,
                worker_metric_categories=worker_metric_categories,
                phase_timing_seconds=phase_timing_seconds,
            )
            telemetry.phase("RESULT_SEAL", tasks_total=total, kernels_total=0)
            telemetry.set_detail(
                evidence_bytes=evidence, reducer_rss_bytes=rss,
                worker_metric_totals=worker_metric_totals,
                phase_timing_seconds=phase_timing_seconds,
                actual_workers=exec_meta.get("workers", 0),
                max_inflight_shards=exec_meta.get("max_inflight_shards", 0),
            )
            telemetry.complete()
            write_json_atomic(phase_root / "SUMMARY.json", {"science": summary, "execution": exec_meta})
            self._status(
                phase_root, status="COMPLETE", task_count=total, completed=total, pending=0,
                class_count=class_count, multi_class_count=multi_count,
                evidence_bytes=evidence, reducer_rss_bytes=rss,
            )
            phase_complete = True
            return StructuralPartitionResult(summary=summary, execution_metadata=exec_meta)
        finally:
            try:
                if 'controller_fallback_kernel_view_bound' in locals() and controller_fallback_kernel_view_bound:
                    from .v05_kernel_services import clear_kernel_view
                    clear_kernel_view()
            finally:
                conn.close()
                if phase_complete:
                    self._finalize_idle_partition_database(db_path)

    def run_content_indexed_generation(
        self,
        *,
        phase_id: str,
        tasks: Iterable[TaskSpec],
        evaluator_ref: str,
        requested_workers: int | str | None = None,
        max_tasks: int | None = None,
        max_generated_occurrences: int | None = None,
    ) -> StreamingGenerationStoreResult:
        """Execute exact-state generation through the common runtime and store unique states once.

        Workers may return many exact state occurrences.  Each identity is encoded once in the
        worker, then the controller treats SHA-256 only as an index bucket and compares complete
        canonical bytes before reuse.  Task completion and occurrences are committed atomically,
        so an interrupted invocation resumes only missing tasks.  Worker completion order is not
        part of the state-set binding or representative selection.
        """
        self._require_execution()
        runtime_started = time.perf_counter()
        tasks = sorted(list(tasks), key=lambda t: t.task_id)
        if max_tasks is not None and len(tasks) > int(max_tasks):
            raise StageRuntimeError(f"generation task budget exceeded {len(tasks)} > {max_tasks}")
        self._execution_permit.require_evaluator(evaluator_ref)
        evaluator = _resolve_ref(evaluator_ref, self._project_callable_bindings.get(evaluator_ref))
        gate = require_controller_only_callable(evaluator, role="scientific generation evaluator")
        phase_root = self._phase_root(phase_id)
        db_path = phase_root / "state_store.sqlite3"
        generation_complete = False
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS states (
              state_token TEXT PRIMARY KEY,
              index_digest TEXT NOT NULL,
              class_index INTEGER NOT NULL,
              canonical_bytes BLOB NOT NULL,
              canonical_size_bytes INTEGER NOT NULL,
              state_json TEXT NOT NULL,
              representative_occurrence_id TEXT NOT NULL,
              occurrence_count INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gen_state_digest ON states(index_digest)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS occurrences (
              occurrence_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              state_token TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gen_occ_state ON occurrences(state_token)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_tasks (
              task_id TEXT PRIMARY KEY,
              payload_sha256 TEXT NOT NULL,
              state_occurrence_count INTEGER NOT NULL,
              metrics_json TEXT NOT NULL,
              committed_utc TEXT NOT NULL
            )
        """)
        conn.commit()
        try:
            manifest = self._task_manifest(tasks)
            scope_base = {
                "schema_id": GENERATION_STORE_SCHEMA,
                "chain_id": self.chain_id, "stage_id": self.stage_id, "phase_id": phase_id,
                "question_sha256": self.question_sha256, "source_sha256": live_source_sha256(),
                "evaluator_ref": evaluator_ref, "task_count": len(tasks),
                "task_manifest_sha256": canonical_sha256(manifest),
                "structural_encoder_id": STRUCTURAL_ENCODER_ID,
            }
            scope_sha = canonical_sha256(scope_base)
            scope_text = canonical_text(dict(scope_base, scope_sha256=scope_sha))
            old = conn.execute("SELECT v FROM meta WHERE k='scope'").fetchone()
            if old is None:
                conn.execute("INSERT INTO meta(k,v) VALUES('scope',?)", (scope_text,)); conn.commit()
            elif old[0] != scope_text:
                raise StageRuntimeError("streaming generation durable scope mismatch; refusing stale resume state")
            by_id = {x["task_id"]: x for x in manifest}
            for tid, psha in conn.execute("SELECT task_id,payload_sha256 FROM generation_tasks"):
                if tid not in by_id or by_id[tid]["payload_sha256"] != psha:
                    raise StageRuntimeError(f"durable generation task binding mismatch for {tid}")
            self._enforce_execution_budgets(phase_root)
            done = {r[0] for r in conn.execute("SELECT task_id FROM generation_tasks")}
            pending = [t for t in tasks if t.task_id not in done]
            task_by_id = {t.task_id: t for t in tasks}
            peak_rss = _rss_bytes(); peak_evidence = _runtime_workspace_bytes(phase_root)
            committed_tasks = 0; generated_this_invocation = 0
            worker_metric_totals: dict[str, float] = {}
            fault_after_raw = os.environ.get("IG_V05_ENGINEERING_ABORT_AFTER_GENERATION_TASKS")
            fault_after = None
            if fault_after_raw is not None:
                if os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") != "STAGE_RUNTIME_GENERATION_ABORT_TEST":
                    raise StageRuntimeError("generation abort injection requires explicit acknowledgement")
                fault_after = max(1, int(fault_after_raw))
            self._status(phase_root, status="RUNNING_GENERATION", task_count=len(tasks), completed=len(done),
                         pending=len(pending), workers_requested=requested_workers if requested_workers is not None else self.default_workers,
                         architecture_gate=gate)

            def commit_task(tid: str, result: Mapping[str, Any]) -> None:
                nonlocal committed_tasks, generated_this_invocation, peak_rss, peak_evidence
                if tid in done:
                    raise StageRuntimeError(f"duplicate generation completion {tid}")
                rows = list(result.get("states") or [])
                if max_generated_occurrences is not None:
                    existing = int(conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0])
                    if existing + len(rows) > int(max_generated_occurrences):
                        raise StageRuntimeError(f"generated occurrence budget exceeded {existing + len(rows)} > {max_generated_occurrences}")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for j, row in enumerate(rows):
                        oid = f"{tid}:{j:06d}"
                        digest = str(row["identity_sha256"]); b = bytes(row["identity_canonical_bytes"])
                        real_digest = hashlib.sha256(b).hexdigest()
                        if len(digest) != 64:
                            raise StageRuntimeError(f"generation identity digest shape mismatch {oid}")
                        if real_digest != digest:
                            if not bool(row.get("engineering_digest_override")) or os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") != "GENERATION_DIGEST_COLLISION_TEST":
                                raise StageRuntimeError(f"generation identity digest/bytes mismatch {oid}")
                        state_json = canonical_text(row["state"])
                        matches = list(conn.execute(
                            "SELECT state_token,class_index,canonical_bytes,representative_occurrence_id,state_json FROM states WHERE index_digest=? ORDER BY class_index",
                            (digest,),
                        ))
                        chosen = None
                        for token, _idx, old_bytes, rep_oid, old_state_json in matches:
                            if bytes(old_bytes) == b:
                                chosen = str(token)
                                # Stable representative: smallest occurrence ID controls stored reconstruction payload.
                                if oid < str(rep_oid):
                                    conn.execute("UPDATE states SET representative_occurrence_id=?,state_json=? WHERE state_token=?",
                                                 (oid, state_json, chosen))
                                conn.execute("UPDATE states SET occurrence_count=occurrence_count+1 WHERE state_token=?", (chosen,))
                                break
                        if chosen is None:
                            idx = 0 if not matches else max(int(r[1]) for r in matches) + 1
                            chosen = f"{digest}:{idx}"
                            conn.execute(
                                "INSERT INTO states(state_token,index_digest,class_index,canonical_bytes,canonical_size_bytes,state_json,representative_occurrence_id,occurrence_count) VALUES(?,?,?,?,?,?,?,1)",
                                (chosen, digest, idx, sqlite3.Binary(b), len(b), state_json, oid),
                            )
                        conn.execute("INSERT INTO occurrences(occurrence_id,task_id,state_token) VALUES(?,?,?)", (oid, tid, chosen))
                    metrics = dict(result.get("metrics") or {})
                    conn.execute(
                        "INSERT INTO generation_tasks(task_id,payload_sha256,state_occurrence_count,metrics_json,committed_utc) VALUES(?,?,?,?,?)",
                        (tid, canonical_sha256(task_by_id[tid].payload), len(rows), canonical_text(metrics), utc_now()),
                    )
                    conn.commit()
                except BaseException:
                    conn.rollback(); raise
                done.add(tid); committed_tasks += 1; generated_this_invocation += len(rows)
                for key, value in (result.get("metrics") or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        worker_metric_totals[key] = worker_metric_totals.get(key, 0.0) + float(value)
                if committed_tasks % 8 == 0 or committed_tasks == len(pending):
                    rss, evidence = self._enforce_execution_budgets(phase_root)
                    peak_rss = max(peak_rss, rss); peak_evidence = max(peak_evidence, evidence)
                    self._status(phase_root, status="RUNNING_GENERATION", task_count=len(tasks), completed=len(done),
                                 pending=len(tasks)-len(done), distinct_states=int(conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]),
                                 generated_occurrences=int(conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0]),
                                 reducer_rss_bytes=rss, evidence_bytes=evidence)
                if fault_after is not None and committed_tasks >= fault_after:
                    raise StageRuntimeError("ENGINEERING_GENERATION_ABORT_AFTER_COMMITTED_TASKS")

            exec_meta = {
                "backend": "NOOP_RESUME", "workers": 0, "task_count": 0, "completed_task_count": 0,
                "wall_seconds": 0.0, "streaming_results": True,
            }
            if pending:
                policy = ExecutionPolicy(
                    requested_workers=requested_workers if requested_workers is not None else self.default_workers,
                    scheduler="COST_WEIGHTED_SHARDS", stream_shard_task_limit=1,
                    stream_inflight_shards_per_worker=2, owner=f"V05:{self.chain_id}:{self.stage_id}:{phase_id}:GEN",
                )
                exec_meta = execute_tasks_stream(
                    pending, worker_ref="infinity_grid.v05_stage_runtime:_generation_worker", on_result=commit_task,
                    policy=policy, initializer_ref="infinity_grid.v05_stage_runtime:_init_generation_worker",
                    initializer_payload={"evaluator_ref": evaluator_ref, "scope_sha256": scope_sha,
                                         "project_callable_binding": self._project_callable_bindings.get(evaluator_ref)},
                )

            if len(done) != len(tasks):
                raise StageRuntimeError(f"generation task coverage mismatch {len(done)} != {len(tasks)}")
            # Canonicalize the exceptional true digest-collision buckets independent of arrival order.
            collision_buckets = [r[0] for r in conn.execute(
                "SELECT index_digest FROM states GROUP BY index_digest HAVING COUNT(*)>1 ORDER BY index_digest"
            )]
            collision_remaps = 0
            for digest in collision_buckets:
                rows = list(conn.execute("SELECT state_token,class_index,canonical_bytes FROM states WHERE index_digest=?", (digest,)))
                rows.sort(key=lambda r: (bytes(r[2]), str(r[0])))
                temps = []
                for j, (old_token, _old_idx, _b) in enumerate(rows):
                    tmp = f"__E3_GEN_TMP__:{digest}:{j}"
                    conn.execute("UPDATE occurrences SET state_token=? WHERE state_token=?", (tmp, old_token))
                    conn.execute("UPDATE states SET state_token=? WHERE state_token=?", (tmp, old_token))
                    temps.append((tmp, f"{digest}:{j}", j))
                for tmp, new_token, j in temps:
                    conn.execute("UPDATE states SET state_token=?,class_index=? WHERE state_token=?", (new_token, j, tmp))
                    conn.execute("UPDATE occurrences SET state_token=? WHERE state_token=?", (new_token, tmp)); collision_remaps += 1
            conn.commit()
            task_count = int(conn.execute("SELECT COUNT(*) FROM generation_tasks").fetchone()[0])
            occurrence_count = int(conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0])
            distinct = int(conn.execute("SELECT COUNT(*) FROM states").fetchone()[0])
            stored_bytes = int(conn.execute("SELECT COALESCE(SUM(canonical_size_bytes),0) FROM states").fetchone()[0])
            h = hashlib.sha256(); h.update(b"IG_DECODER_V05_GENERATED_STATE_SET_V1\n")
            for token, b in conn.execute("SELECT state_token,canonical_bytes FROM states ORDER BY canonical_bytes,state_token"):
                h.update(bytes(b)); h.update(b"\n")
            state_set_sha = h.hexdigest()
            occurrence_h = hashlib.sha256(); occurrence_h.update(b"IG_DECODER_V05_GENERATION_OCCURRENCE_BINDINGS_V1\n")
            for oid, token in conn.execute("SELECT occurrence_id,state_token FROM occurrences ORDER BY occurrence_id"):
                occurrence_h.update(canonical_text([oid, token]).encode("utf-8")); occurrence_h.update(b"\n")
            summary_base = {
                "schema_id": GENERATION_STORE_SUMMARY_SCHEMA, "chain_id": self.chain_id, "stage_id": self.stage_id,
                "phase_id": phase_id, "question_sha256": self.question_sha256, "task_count": task_count,
                "raw_generated_occurrence_count": occurrence_count, "distinct_exact_state_count": distinct,
                "duplicate_generated_occurrence_count": occurrence_count - distinct,
                "stored_exact_identity_canonical_bytes": stored_bytes,
                "generated_state_set_sha256": state_set_sha,
                "generation_occurrence_bindings_sha256": occurrence_h.hexdigest(),
                "digest_equality_never_decides_state_equality": True,
                "exact_canonical_bytes_are_state_equality_authority": True,
                "representative_selection": "LEXICOGRAPHIC_MIN_OCCURRENCE_ID_V1",
            }
            summary = dict(summary_base, summary_sha256=canonical_sha256(summary_base))
            rss, evidence = self._enforce_execution_budgets(phase_root)
            peak_rss = max(peak_rss, rss); peak_evidence = max(peak_evidence, evidence)
            exec_meta = dict(exec_meta,
                durable_store="SQLITE_STREAMING_GENERATION_ENGINE_OWNED", durable_scope_sha256=scope_sha,
                committed_generation_tasks_this_invocation=committed_tasks,
                generated_occurrences_this_invocation=generated_this_invocation,
                reducer_rss_bytes_final=rss, reducer_rss_bytes_peak=peak_rss,
                evidence_bytes=evidence, evidence_bytes_peak=peak_evidence,
                deterministic_collision_state_remaps=collision_remaps,
                worker_metric_totals=worker_metric_totals,
                runtime_call_total_to_summary=time.perf_counter()-runtime_started,
            )
            write_json_atomic(phase_root / "SUMMARY.json", {"generation": summary, "execution": exec_meta})
            self._status(phase_root, status="COMPLETE", task_count=task_count, completed=task_count, pending=0,
                         distinct_states=distinct, generated_occurrences=occurrence_count,
                         evidence_bytes=evidence, reducer_rss_bytes=rss)
            generation_complete = True
            return StreamingGenerationStoreResult(summary=summary, execution_metadata=exec_meta)
        finally:
            conn.close()
            if generation_complete:
                self._finalize_idle_partition_database(db_path)

    def iter_generated_states(self, *, phase_id: str):
        """Yield deterministic generated exact-state records from an engine-owned generation phase."""
        self._require_execution()
        phase_root = self._phase_root(phase_id)
        db = phase_root / "state_store.sqlite3"
        if not db.is_file():
            raise StageRuntimeError(f"missing generation state store for phase {phase_id}")
        conn = sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True)
        try:
            for token, digest, b, state_json, rep_oid, count in conn.execute(
                "SELECT state_token,index_digest,canonical_bytes,state_json,representative_occurrence_id,occurrence_count FROM states ORDER BY canonical_bytes,state_token"
            ):
                yield {"state_token": str(token), "identity_sha256": str(digest),
                       "identity_canonical_bytes": bytes(b), "state": json.loads(state_json),
                       "representative_occurrence_id": str(rep_oid), "occurrence_count": int(count)}
        finally:
            conn.close()

    def build_content_indexed_state_store(
        self,
        *,
        phase_id: str,
        occurrences: Iterable[Mapping[str, Any]],
        max_occurrences: int | None = None,
    ) -> ContentIndexedStateStoreResult:
        """Store each exact canonical state once and bind every raw occurrence to it.

        Each occurrence mapping must contain ``occurrence_id`` and ``state``; an
        optional ``metadata`` value is stored per occurrence.  SHA-256 is only a
        candidate bucket. Equal-digest states are compared by their complete
        canonical bytes before reuse.
        """
        self._require_execution()
        items = sorted((dict(x) for x in occurrences), key=lambda x: str(x.get("occurrence_id", "")))
        if max_occurrences is not None and len(items) > int(max_occurrences):
            raise StageRuntimeError(f"state-store occurrence budget exceeded {len(items)} > {max_occurrences}")
        ids = [str(x.get("occurrence_id", "")) for x in items]
        if any(not x for x in ids) or len(ids) != len(set(ids)):
            raise StageRuntimeError("state-store occurrence_id values must be unique and nonempty")
        for item in items:
            if "state" not in item:
                raise StageRuntimeError(f"state-store occurrence {item.get('occurrence_id')} missing state")

        # Bind the complete exact input set before any state-store mutation.  The
        # manifest retains only compact exact digests/metadata digests.
        manifest = []
        for item in items:
            manifest.append({
                "occurrence_id": str(item["occurrence_id"]),
                "state_sha256": structural_canonical_sha256(item["state"]),
                "metadata_sha256": canonical_sha256(item.get("metadata")),
            })
        phase_root = self._phase_root(phase_id)
        db_path = phase_root / "state_store.sqlite3"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS states (
              state_token TEXT PRIMARY KEY,
              index_digest TEXT NOT NULL,
              class_index INTEGER NOT NULL,
              canonical_bytes BLOB NOT NULL,
              canonical_size_bytes INTEGER NOT NULL,
              representative_occurrence_id TEXT NOT NULL,
              occurrence_count INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_state_digest ON states(index_digest)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS occurrences (
              occurrence_id TEXT PRIMARY KEY,
              state_token TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              committed_utc TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_occ_state ON occurrences(state_token)")
        conn.commit()
        started=time.monotonic(); peak_rss=_rss_bytes(); peak_evidence=_runtime_workspace_bytes(phase_root)
        try:
            scope_base={
                "schema_id":STATE_STORE_SCHEMA,"chain_id":self.chain_id,"stage_id":self.stage_id,
                "phase_id":phase_id,"question_sha256":self.question_sha256,"source_sha256":live_source_sha256(),
                "structural_encoder_id":STRUCTURAL_ENCODER_ID,"occurrence_count":len(items),
                "occurrence_manifest_sha256":canonical_sha256(manifest),
            }
            scope_sha=canonical_sha256(scope_base); encoded=canonical_text(dict(scope_base,scope_sha256=scope_sha))
            old=conn.execute("SELECT v FROM meta WHERE k='scope'").fetchone()
            if old is None:
                conn.execute("INSERT INTO meta(k,v) VALUES('scope',?)",(encoded,));conn.commit()
            elif old[0] != encoded:
                raise StageRuntimeError("content-indexed state-store durable scope mismatch")
            self._enforce_execution_budgets(phase_root)
            done={r[0] for r in conn.execute("SELECT occurrence_id FROM occurrences")}; committed=0
            raw_state_bytes=0
            collision_override_ack = os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") == "STATE_STORE_DIGEST_COLLISION_TEST"
            for item in items:
                oid=str(item["occurrence_id"]); state_bytes=structural_canonical_bytes(item["state"]); raw_state_bytes += len(state_bytes)
                if oid in done:
                    continue
                real_digest=hashlib.sha256(state_bytes).hexdigest(); digest=real_digest
                if "_test_state_digest_override" in item:
                    if not collision_override_ack:
                        raise StageRuntimeError("state-store digest override requires explicit engineering acknowledgement")
                    digest=str(item["_test_state_digest_override"])
                    if len(digest)!=64:
                        raise StageRuntimeError("state-store test digest override must be SHA-256 shaped")
                rows=list(conn.execute(
                    "SELECT state_token,class_index,canonical_bytes,representative_occurrence_id FROM states WHERE index_digest=? ORDER BY class_index",(digest,)
                ))
                chosen=None
                for token,_idx,rep_bytes,rep_oid in rows:
                    if bytes(rep_bytes)==state_bytes:
                        chosen=str(token)
                        conn.execute(
                            "UPDATE states SET occurrence_count=occurrence_count+1, representative_occurrence_id=CASE WHEN representative_occurrence_id>? THEN ? ELSE representative_occurrence_id END WHERE state_token=?",
                            (oid,oid,chosen),
                        )
                        break
                if chosen is None:
                    idx=0 if not rows else max(int(r[1]) for r in rows)+1
                    chosen=f"{digest}:{idx}"
                    conn.execute(
                        "INSERT INTO states(state_token,index_digest,class_index,canonical_bytes,canonical_size_bytes,representative_occurrence_id,occurrence_count) VALUES(?,?,?,?,?,?,1)",
                        (chosen,digest,idx,sqlite3.Binary(state_bytes),len(state_bytes),oid),
                    )
                conn.execute(
                    "INSERT INTO occurrences(occurrence_id,state_token,metadata_json,committed_utc) VALUES(?,?,?,?)",
                    (oid,chosen,canonical_text(item.get("metadata")),utc_now()),
                )
                done.add(oid); committed += 1
                if committed % 64 == 0:
                    conn.commit(); rss,evidence=self._enforce_execution_budgets(phase_root); peak_rss=max(peak_rss,rss);peak_evidence=max(peak_evidence,evidence)
                    self._status(phase_root,status="BUILDING_STATE_STORE",occurrence_count=len(items),completed=len(done),distinct_states=int(conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]),reducer_rss_bytes=rss,evidence_bytes=evidence)
            conn.commit()

            # Canonicalize true/forced digest-collision state tokens by exact bytes.
            buckets=[r[0] for r in conn.execute("SELECT index_digest FROM states GROUP BY index_digest HAVING COUNT(*)>1 ORDER BY index_digest")]
            remaps=0
            for digest in buckets:
                rows=list(conn.execute("SELECT state_token,class_index,canonical_bytes FROM states WHERE index_digest=?",(digest,)))
                rows.sort(key=lambda r:(bytes(r[2]),str(r[0])))
                temps=[]
                for j,(old_token,_old_idx,_b) in enumerate(rows):
                    temp=f"__E2_STATE_TMP__:{digest}:{j}"
                    conn.execute("UPDATE occurrences SET state_token=? WHERE state_token=?",(temp,old_token))
                    conn.execute("UPDATE states SET state_token=? WHERE state_token=?",(temp,old_token))
                    temps.append((temp,f"{digest}:{j}",j))
                for temp,new_token,j in temps:
                    conn.execute("UPDATE states SET state_token=?,class_index=? WHERE state_token=?",(new_token,j,temp))
                    conn.execute("UPDATE occurrences SET state_token=? WHERE state_token=?",(new_token,temp));remaps+=1
            conn.commit()

            occurrence_count=int(conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0])
            if occurrence_count != len(items):
                raise StageRuntimeError(f"state-store coverage mismatch {occurrence_count} != {len(items)}")
            distinct=int(conn.execute("SELECT COUNT(*) FROM states").fetchone()[0])
            stored_state_bytes=int(conn.execute("SELECT COALESCE(SUM(canonical_size_bytes),0) FROM states").fetchone()[0])
            h=hashlib.sha256();h.update(b"IG_DECODER_V05_STATE_STORE_BINDINGS_V1\n")
            for oid,token in conn.execute("SELECT occurrence_id,state_token FROM occurrences ORDER BY occurrence_id"):
                h.update(canonical_text([oid,token]).encode("utf-8"));h.update(b"\n")
            binding_sha=h.hexdigest()
            summary_base={
                "schema_id":STATE_STORE_SUMMARY_SCHEMA,"chain_id":self.chain_id,"stage_id":self.stage_id,"phase_id":phase_id,
                "question_sha256":self.question_sha256,"raw_occurrence_count":occurrence_count,"distinct_exact_state_count":distinct,
                "duplicate_occurrence_count":occurrence_count-distinct,"stored_exact_state_canonical_bytes":stored_state_bytes,
                "raw_occurrence_state_canonical_bytes":raw_state_bytes,
                "state_payload_compression_ratio":(raw_state_bytes/stored_state_bytes if stored_state_bytes else 1.0),
                "state_store_bindings_sha256":binding_sha,"state_store_binding_hash_schema":"IG_DECODER_V05_STATE_STORE_BINDINGS_V1",
                "digest_equality_never_decides_state_equality":True,"exact_canonical_bytes_are_equality_authority":True,
                "deterministic_collision_state_ordering":"EXACT_CANONICAL_BYTES_ASCENDING_V1",
            }
            summary=dict(summary_base,summary_sha256=canonical_sha256(summary_base))
            rss,evidence=self._enforce_execution_budgets(phase_root);peak_rss=max(peak_rss,rss);peak_evidence=max(peak_evidence,evidence)
            exec_meta={
                "backend":"ENGINE_CONTENT_INDEXED_STATE_STORE","wall_seconds":time.monotonic()-started,"durable_scope_sha256":scope_sha,
                "committed_occurrences_this_invocation":committed,"state_collision_token_remaps":remaps,
                "reducer_rss_bytes_final":rss,"reducer_rss_bytes_peak":peak_rss,"evidence_bytes":evidence,"evidence_bytes_peak":peak_evidence,
                "workspace_accounting_mode":"CLOSED_LAYOUT_INCREMENTAL_FILE_SIZES_V1","structural_signature_encoder_id":STRUCTURAL_ENCODER_ID,
            }
            write_json_atomic(phase_root/"SUMMARY.json",{"state_store":summary,"execution":exec_meta})
            self._status(phase_root,status="COMPLETE",occurrence_count=occurrence_count,distinct_states=distinct,evidence_bytes=evidence,reducer_rss_bytes=rss)
            return ContentIndexedStateStoreResult(summary=summary,execution_metadata=exec_meta)
        finally:
            conn.close()

    def import_legacy_partition_shards(
        self,
        *,
        phase_id: str,
        shard_root: str | Path,
        expected_schema_id: str,
        expected_level: str | None = None,
    ) -> StructuralPartitionResult:
        """Stream a historical full-signature shard set into compact engine evidence.

        This is an execution-only migration. Every old shard hash is verified.
        Full signatures are read one row at a time; only compact class bindings are
        retained. Equal digests are checked by exact structural equality using the
        full legacy row before class assignment.
        """
        self._require_execution()
        root = Path(shard_root).resolve(strict=True)
        files_list = sorted(root.glob("*.json"))
        phase_root = self._phase_root(phase_id)
        conn = self._open_db(phase_root / "partition.sqlite3")
        started = time.monotonic()
        peak_reducer_rss = _rss_bytes()
        peak_evidence_bytes = _runtime_workspace_bytes(phase_root)
        try:
            scope_base = {
                "schema_id": PARTITION_DB_SCHEMA,
                "chain_id": self.chain_id,
                "stage_id": self.stage_id,
                "phase_id": phase_id,
                "question_sha256": self.question_sha256,
                "source_sha256": live_source_sha256(),
                "legacy_shard_root": str(root),
                "legacy_shard_count": len(files_list),
                "legacy_expected_schema_id": expected_schema_id,
                "legacy_expected_level": expected_level,
            }
            scope_sha = canonical_sha256(scope_base)
            scope_encoded = canonical_text(dict(scope_base, scope_sha256=scope_sha))
            old = conn.execute("SELECT v FROM meta WHERE k='scope'").fetchone()
            if old is None:
                conn.execute("INSERT INTO meta(k,v) VALUES('scope',?)", (scope_encoded,)); conn.commit()
            elif old[0] != scope_encoded:
                raise StageRuntimeError("legacy import scope mismatch")
            self._enforce_execution_budgets(phase_root)

            # task id -> (shard path,row index) enables exact fallback without
            # retaining huge representative signatures in RAM.
            def load_signature(locator_json: str) -> Any:
                loc = json.loads(locator_json)
                sh = json.loads(Path(loc["path"]).read_text(encoding="utf-8"))
                return sh["rows"][int(loc["row_index"])]["signature"]

            done = {r[0] for r in conn.execute("SELECT task_id FROM task_results")}
            processed = len(done)
            representative_signature_cache: dict[str, bytes] = {}
            seen_source_task_ids: set[str] = set()
            for sp in files_list:
                sh = json.loads(sp.read_text(encoding="utf-8"))
                if sh.get("schema_id") != expected_schema_id:
                    raise StageRuntimeError(f"legacy shard schema mismatch {sp}")
                if expected_level is not None and sh.get("level") != expected_level:
                    raise StageRuntimeError(f"legacy shard level mismatch {sp}")
                expected = sh.get("science_sha256")
                observed = canonical_sha256({k:v for k,v in sh.items() if k != "science_sha256"})
                if expected != observed:
                    raise StageRuntimeError(f"legacy shard integrity failure {sp}")
                rows = list(sh.get("rows") or [])
                declared_ids = list(sh.get("state_ids") or [])
                row_ids = [str(r.get("state_id")) for r in rows]
                if declared_ids != row_ids:
                    raise StageRuntimeError(f"legacy shard state_ids mismatch {sp}")
                for idx, row in enumerate(rows):
                    tid = str(row["state_id"])
                    if tid in seen_source_task_ids:
                        raise StageRuntimeError(f"duplicate legacy source task_id {tid}")
                    seen_source_task_ids.add(tid)
                    if tid in done:
                        continue
                    sig = row["signature"]
                    sig_bytes = structural_canonical_bytes(sig)
                    digest = hashlib.sha256(sig_bytes).hexdigest()
                    classes = list(conn.execute(
                        "SELECT class_token,class_index,representative_task_id FROM classes WHERE signature_sha256=? ORDER BY class_index",
                        (digest,),
                    ))
                    chosen = None
                    if not classes:
                        chosen = f"{digest}:0"
                        conn.execute("INSERT INTO classes(class_token,signature_sha256,class_index,representative_task_id,size) VALUES(?,?,?,?,1)", (chosen,digest,0,tid))
                    else:
                        for token, _ci, rep_tid in classes:
                            rep_sig = representative_signature_cache.get(token)
                            if rep_sig is None:
                                rr = conn.execute("SELECT locator_json FROM task_results WHERE task_id=?", (rep_tid,)).fetchone()
                                if rr is None or not rr[0]:
                                    raise StageRuntimeError("legacy representative locator missing")
                                rep_sig = structural_canonical_bytes(load_signature(rr[0]))
                                if len(representative_signature_cache) < 256:
                                    representative_signature_cache[token] = rep_sig
                            if rep_sig == sig_bytes:
                                chosen = token
                                conn.execute(
                                    "UPDATE classes SET size=size+1, representative_task_id=CASE WHEN representative_task_id>? THEN ? ELSE representative_task_id END WHERE class_token=?",
                                    (tid, tid, token),
                                )
                                break
                        if chosen is None:
                            ci = max(int(x[1]) for x in classes)+1
                            chosen = f"{digest}:{ci}"
                            conn.execute("INSERT INTO classes(class_token,signature_sha256,class_index,representative_task_id,size) VALUES(?,?,?,?,1)", (chosen,digest,ci,tid))
                            wp = phase_root / "digest_collision_witnesses"; wp.mkdir(parents=True, exist_ok=True)
                            write_json_atomic(wp / f"{digest}-{ci}.json", {"signature_sha256":digest,"new_task_id":tid,"new_signature":sig})
                    locator = canonical_text({"path": str(sp), "row_index": idx})
                    conn.execute(
                        "INSERT INTO task_results(task_id,payload_sha256,signature_sha256,class_token,outcome_count,metrics_json,locator_json,committed_utc) VALUES(?,?,?,?,?,?,?,?)",
                        (tid, "0"*64, digest, chosen, 0, "{}", locator, utc_now()),
                    )
                    done.add(tid); processed += 1
                    if processed % 64 == 0:
                        conn.commit()
                        rss, evidence = self._enforce_execution_budgets(phase_root)
                        peak_reducer_rss = max(peak_reducer_rss, rss)
                        peak_evidence_bytes = max(peak_evidence_bytes, evidence)
                        self._status(
                            phase_root, status="IMPORTING_LEGACY", shard_count=len(files_list), completed=processed,
                            elapsed_seconds=time.monotonic()-started, reducer_rss_bytes=rss, evidence_bytes=evidence,
                        )
                conn.commit()
            def legacy_signature_bytes_for_task(tid: str) -> bytes:
                rr = conn.execute("SELECT locator_json FROM task_results WHERE task_id=?", (tid,)).fetchone()
                if rr is None or not rr[0]:
                    raise StageRuntimeError(f"legacy representative locator missing for {tid}")
                return structural_canonical_bytes(load_signature(rr[0]))

            deterministic_collision_class_remaps = self._finalize_collision_class_tokens(
                conn, legacy_signature_bytes_for_task
            )
            task_count = int(conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0])
            class_count = int(conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0])
            multi_count = int(conn.execute("SELECT COUNT(*) FROM classes WHERE size>1").fetchone()[0])
            max_class_size = int(conn.execute("SELECT COALESCE(MAX(size),0) FROM classes").fetchone()[0])
            first_multi_row = conn.execute(
                "SELECT class_token,size,representative_task_id FROM classes WHERE size>1 ORDER BY class_token LIMIT 1"
            ).fetchone()
            summary_base = {
                "schema_id": PARTITION_SUMMARY_SCHEMA,
                "chain_id": self.chain_id,
                "stage_id": self.stage_id,
                "phase_id": phase_id,
                "question_sha256": self.question_sha256,
                "task_count": task_count,
                "class_count": class_count,
                "multi_class_count": multi_count,
                "max_class_size": max_class_size,
                "first_multi_class": None if first_multi_row is None else {
                    "class_token":first_multi_row[0],"size":int(first_multi_row[1]),"representative_task_id":first_multi_row[2]
                },
                "partition_bindings_sha256": self._partition_bindings_sha256(conn),
                "partition_binding_hash_schema": "IG_DECODER_V05_PARTITION_BINDINGS_STREAM_V1",
                "structural_equality_used_for_equal_digest": True,
                "hash_equality_never_decides_scientific_equality": True,
                "deterministic_collision_class_ordering": "EXACT_CANONICAL_BYTES_ASCENDING_V1",
            }
            summary = dict(summary_base, summary_sha256=canonical_sha256(summary_base))
            rss, evidence = self._enforce_execution_budgets(phase_root)
            peak_reducer_rss = max(peak_reducer_rss, rss)
            peak_evidence_bytes = max(peak_evidence_bytes, evidence)
            exec_meta = {
                "backend":"LEGACY_STREAM_IMPORT", "workers":0, "wall_seconds":time.monotonic()-started,
                "streaming_results":True, "durable_store":"SQLITE_COMPACT_ENGINE_OWNED_LEGACY_IMPORT",
                "durable_scope_sha256":scope_sha, "legacy_shards_verified":len(files_list),
                "evidence_bytes":evidence, "evidence_bytes_peak":peak_evidence_bytes,
                "reducer_rss_bytes_final":rss, "reducer_rss_bytes_peak":peak_reducer_rss,
                "structural_signature_encoder_id": STRUCTURAL_ENCODER_ID,
                "workspace_accounting_mode": "CLOSED_LAYOUT_INCREMENTAL_FILE_SIZES_V1",
                "deterministic_collision_class_remaps": deterministic_collision_class_remaps,
            }
            write_json_atomic(phase_root / "SUMMARY.json", {"science": summary, "execution": exec_meta})
            self._status(phase_root,status="COMPLETE",completed=task_count,class_count=class_count,multi_class_count=multi_count,evidence_bytes=evidence,reducer_rss_bytes=rss)
            return StructuralPartitionResult(summary=summary, execution_metadata=exec_meta)
        finally:
            conn.close()
