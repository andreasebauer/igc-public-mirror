from __future__ import annotations

import contextlib
import dataclasses
from contextlib import contextmanager
import importlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import multiprocessing as mp

try:  # POSIX host-wide lease coordination (Linux/macOS)
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - Windows fallback
    fcntl = None

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .records import utc_now
from .runtime_telemetry import RuntimeTelemetry


class ExecutionError(RuntimeError):
    pass


class NestedParallelismError(ExecutionError):
    pass


class WorkerTaskError(ExecutionError):
    pass


class CpuLeaseError(ExecutionError):
    pass


class DuplicateTaskError(ExecutionError):
    pass


class MissingTaskError(ExecutionError):
    pass


# A process can acquire more than one Decoder CPU lease over its lifetime.  The
# on-disk lease file survives exceptions and test/session reuse, so PID liveness
# alone is insufficient: an orphaned lease written by *this same interpreter*
# would otherwise remain live forever while the PID remains alive.  Track the
# lease ids that are actually owned by live CpuLease objects in this process and
# use that registry to prune same-PID orphans fail-closed on the next lease-file
# reconciliation.  This is execution metadata only and never enters science
# hashes.
_LIVE_LOCAL_LEASE_IDS: set[str] = set()
_LIVE_LOCAL_LEASE_LOCK = threading.RLock()


def _local_lease_ids_snapshot() -> set[str]:
    with _LIVE_LOCAL_LEASE_LOCK:
        return set(_LIVE_LOCAL_LEASE_IDS)


def _register_local_lease(lease_id: str) -> None:
    with _LIVE_LOCAL_LEASE_LOCK:
        _LIVE_LOCAL_LEASE_IDS.add(str(lease_id))


def _discard_local_lease(lease_id: str) -> None:
    with _LIVE_LOCAL_LEASE_LOCK:
        _LIVE_LOCAL_LEASE_IDS.discard(str(lease_id))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_kind: str
    binding_sha256: str
    payload: Any
    cost_weight: float = 1.0
    checkpoint_policy: str = "NONE"

    def __post_init__(self):
        if not self.task_id or "/" in self.task_id or "\\" in self.task_id:
            raise ValueError("task_id must be a non-empty path-safe string")
        if not self.task_kind:
            raise ValueError("task_kind is required")
        if not isinstance(self.binding_sha256, str) or len(self.binding_sha256) != 64:
            raise ValueError("binding_sha256 must be a 64-hex SHA-256 string")
        try:
            int(self.binding_sha256, 16)
        except Exception as exc:
            raise ValueError("binding_sha256 must be hexadecimal") from exc
        if not math.isfinite(float(self.cost_weight)) or float(self.cost_weight) < 0:
            raise ValueError("cost_weight must be finite and non-negative")


@dataclass(frozen=True)
class ExecutionPolicy:
    backend: str = "AUTO"  # AUTO | SERIAL | LOCAL_PROCESS_POOL
    requested_workers: int | str | None = "AUTO"
    scheduler: str = "ORDERED_MAP"  # ORDERED_MAP | COST_WEIGHTED_SHARDS
    start_method: str = "AUTO"
    reserve_cores: int = 0
    lease_root: str | None = None
    wait_for_lease: bool = True
    lease_timeout_seconds: float = 600.0
    poll_seconds: float = 0.25
    target_shards_per_worker: int = 4
    stream_shard_task_limit: int = 24
    stream_inflight_shards_per_worker: int = 2
    telemetry_interval_seconds: float = 5.0
    owner: str = "decoder"


@dataclass
class ExecutionBatch:
    results: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def ordered_results(self) -> list[Any]:
        return [self.results[k] for k in sorted(self.results)]


def _set_single_thread_env() -> None:
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[k] = "1"
    os.environ["IG_DECODER_EXECUTION_WORKER"] = "1"


def _read_cgroup_quota() -> int | None:
    # cgroup v2
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip().split()
        if len(raw) >= 2 and raw[0] != "max":
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except Exception:
        pass
    # cgroup v1
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except Exception:
        pass
    return None


def detect_effective_cpu_count() -> int:
    """Return the effective local CPU budget, honoring explicit override, affinity and cgroup quota."""
    override = os.environ.get("IG_DECODER_CPU_BUDGET")
    if override:
        try:
            return max(1, int(override))
        except ValueError as exc:
            raise CpuLeaseError("IG_DECODER_CPU_BUDGET must be a positive integer") from exc
    vals: list[int] = []
    c = os.cpu_count()
    if c:
        vals.append(int(c))
    try:
        vals.append(len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        pass
    q = _read_cgroup_quota()
    if q:
        vals.append(q)
    return max(1, min(vals) if vals else 1)


def normalized_worker_request(requested: int | str | None, *, task_count: int | None = None, reserve_cores: int = 0) -> int:
    budget = max(1, detect_effective_cpu_count() - max(0, int(reserve_cores)))
    if requested is None or (isinstance(requested, str) and requested.upper() == "AUTO"):
        n = budget
    else:
        n = int(requested)
        if n <= 0:
            raise ValueError("requested_workers must be positive or AUTO")
        n = min(n, budget)
    if task_count is not None:
        n = min(n, max(1, int(task_count)))
    return max(1, n)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


class HostCpuLeaseManager:
    """Process-safe host CPU lease pool shared by all new Decoder executions.

    Lease coordination is execution-only metadata. It is never part of a scientific hash.
    Existing historical jobs that predate this runtime are intentionally not retrofitted.
    """

    def __init__(self, root: str | Path | None = None, *, budget: int | None = None):
        if root is None:
            root = os.environ.get("IG_DECODER_LEASE_ROOT") or "/tmp/infinity_grid_decoder_execution"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "cpu_leases.json"
        self.lock_path = self.root / "cpu_leases.lock"
        self.budget = max(1, int(budget if budget is not None else detect_effective_cpu_count()))

    @contextlib.contextmanager
    def _locked(self):
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("r+") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema": "IG_DECODER_CPU_LEASES_V1", "leases": []}
        try:
            obj = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CpuLeaseError(f"corrupt CPU lease state: {self.state_path}") from exc
        if obj.get("schema") != "IG_DECODER_CPU_LEASES_V1" or not isinstance(obj.get("leases"), list):
            raise CpuLeaseError("CPU lease state schema mismatch")
        return obj

    def _clean(self, obj: dict[str, Any]) -> dict[str, Any]:
        leases = []
        local_pid = os.getpid()
        local_live = _local_lease_ids_snapshot()
        for rec in obj.get("leases", []):
            try:
                pid = int(rec["pid"])
                workers = int(rec["workers"])
                lease_id = str(rec["lease_id"])
            except Exception:
                continue
            if workers <= 0:
                continue
            if pid == local_pid:
                # A same-PID record is live only when this interpreter still
                # owns the exact lease id.  This removes orphaned records left
                # by a failed/aborted context without stealing another active
                # lease legitimately held by the same process.
                if lease_id in local_live:
                    leases.append(rec)
                continue
            if _pid_alive(pid):
                leases.append(rec)
        obj = dict(obj)
        obj["leases"] = leases
        return obj

    def _save(self, obj: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, obj)

    def status(self) -> dict[str, Any]:
        with self._locked():
            obj = self._clean(self._load())
            self._save(obj)
            used = sum(int(x["workers"]) for x in obj["leases"])
            return {
                "status": "PASS",
                "budget": self.budget,
                "used": used,
                "available": max(0, self.budget - used),
                "leases": sorted(obj["leases"], key=lambda x: (int(x["pid"]), x["lease_id"])),
                "state_path": str(self.state_path),
            }

    def acquire(
        self,
        requested: int | str | None,
        *,
        owner: str,
        task_count: int | None = None,
        reserve_cores: int = 0,
        wait: bool = True,
        timeout_seconds: float = 600.0,
        poll_seconds: float = 0.25,
    ) -> "CpuLease":
        max_budget = max(1, self.budget - max(0, int(reserve_cores)))
        explicit_auto = requested is None or (isinstance(requested, str) and requested.upper() == "AUTO")
        wanted = max_budget if explicit_auto else min(max_budget, max(1, int(requested)))
        if task_count is not None:
            wanted = min(wanted, max(1, int(task_count)))
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            with self._locked():
                obj = self._clean(self._load())
                used = sum(int(x["workers"]) for x in obj["leases"])
                available = max(0, max_budget - used)
                # AUTO is elastic: take all currently available. Explicit request waits for full request.
                grant = min(wanted, available) if explicit_auto else (wanted if available >= wanted else 0)
                if grant > 0:
                    lease_id = uuid.uuid4().hex
                    rec = {
                        "lease_id": lease_id,
                        "pid": os.getpid(),
                        "workers": grant,
                        "owner": owner,
                        "host": socket.gethostname(),
                        "created_utc": utc_now(),
                    }
                    obj["leases"].append(rec)
                    # Register before publishing the on-disk record so another
                    # same-process reconciliation cannot mistake this new lease
                    # for an orphan.  Roll back the in-memory ownership marker if
                    # publication fails.
                    _register_local_lease(lease_id)
                    try:
                        self._save(obj)
                    except BaseException:
                        _discard_local_lease(lease_id)
                        raise
                    return CpuLease(self, rec)
                self._save(obj)
            if not wait or time.monotonic() >= deadline:
                raise CpuLeaseError(f"CPU lease unavailable: requested={wanted} budget={max_budget}")
            time.sleep(max(0.01, float(poll_seconds)))

    def release(self, lease_id: str) -> None:
        with self._locked():
            obj = self._clean(self._load())
            before = len(obj["leases"])
            obj["leases"] = [x for x in obj["leases"] if x.get("lease_id") != lease_id]
            # Save first.  If the durable lease-state update fails, retain the
            # local ownership marker so subsequent reconciliation does not
            # silently erase a lease that may still be published on disk.
            self._save(obj)
            _discard_local_lease(lease_id)
            if len(obj["leases"]) == before:
                # Idempotent release is intentionally allowed.
                return


class CpuLease:
    def __init__(self, manager: HostCpuLeaseManager, record: dict[str, Any]):
        self.manager = manager
        self.record = record
        self.released = False

    @property
    def workers(self) -> int:
        return int(self.record["workers"])

    def release(self):
        if not self.released:
            self.manager.release(self.record["lease_id"])
            self.released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def _resolve_ref(ref: str):
    mod, sep, name = ref.partition(":")
    if not sep or not mod or not name:
        raise ExecutionError(f"invalid callable ref {ref!r}; expected module:function")
    obj = importlib.import_module(mod)
    fn = getattr(obj, name)
    if not callable(fn):
        raise ExecutionError(f"callable ref is not callable: {ref}")
    return fn


_WORKER_FN = None
_WORKER_INIT_ERROR = None


def _pool_initializer(worker_ref: str, initializer_ref: str | None, initializer_payload: Any) -> None:
    from .v05_origin_guard import require_native_caller
    require_native_caller('multiprocessing.pool', {'worker'}, 'native-pool-initializer')
    global _WORKER_FN, _WORKER_INIT_ERROR
    _WORKER_INIT_ERROR = None
    try:
        _set_single_thread_env()
        _WORKER_FN = _resolve_ref(worker_ref)
        if initializer_ref:
            init = _resolve_ref(initializer_ref)
            init(initializer_payload)
    except BaseException as exc:
        # A failed initializer must not make Pool endlessly create new children.
        # Return one deterministic task failure with the original refusal data.
        _WORKER_INIT_ERROR = {'code':getattr(exc, 'code', 'WORKER_INITIALIZATION_FAILED'),
                              'checks':getattr(exc, 'checks', [str(exc)]), 'message':str(exc)}


def _run_one(task_wire: dict[str, Any]) -> tuple[str, Any]:
    if _WORKER_INIT_ERROR is not None:
        exc = WorkerTaskError(_WORKER_INIT_ERROR['message'])
        exc.code = _WORKER_INIT_ERROR['code']; exc.checks = _WORKER_INIT_ERROR['checks']
        raise exc
    if _WORKER_FN is None:
        raise ExecutionError("worker function not initialized")
    task = TaskSpec(**task_wire)
    try:
        return task.task_id, _WORKER_FN(task.payload)
    except BaseException as exc:
        # Exception crossing the process boundary preserves type poorly; include deterministic task identity.
        raise WorkerTaskError(f"task {task.task_id} ({task.task_kind}) failed: {type(exc).__name__}: {exc}") from exc


def _run_shard(shard: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [_run_one(t) for t in shard]


def _serial_initializer(worker_ref: str, initializer_ref: str | None, initializer_payload: Any):
    from .v05_origin_guard import require_controller_execution_origin, require_native_caller
    require_controller_execution_origin('native-serial-initializer')
    require_native_caller(__name__, {'execute_tasks', 'execute_tasks_stream'}, 'native-serial-initializer')
    global _WORKER_FN, _WORKER_INIT_ERROR
    _WORKER_INIT_ERROR = None
    # Serial execution runs in the caller process. Do not mark the caller as a
    # Decoder worker or mutate its thread environment; that would leak worker-only
    # state into later executions and can falsely trigger nested-parallelism guards.
    _WORKER_FN = _resolve_ref(worker_ref)
    if initializer_ref:
        _resolve_ref(initializer_ref)(initializer_payload)


def _forkserver_transport_available() -> bool:
    """Return whether this runtime permits the local transport forkserver needs."""
    if not hasattr(socket, "AF_UNIX"):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="ig-decoder-forkserver-probe-") as root:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(Path(root) / "listener.sock"))
                listener.listen(1)
        return True
    except Exception:
        return False


def _choose_start_method(policy: ExecutionPolicy, initializer_ref: str | None = None) -> str:
    requested = str(policy.start_method).lower()
    if requested == "auto":
        requested = os.environ.get("IG_DECODER_START_METHOD", "auto").lower()
    methods = mp.get_all_start_methods()
    if requested != "auto":
        if requested not in methods:
            raise ExecutionError(f"multiprocessing start method {requested!r} unavailable; have {methods}")
        return requested
    # A few legacy/native campaign handlers intentionally use a live read-only
    # parent context.  Make that contract explicit instead of accidentally
    # depending on forkserver snapshot timing.
    if initializer_ref and initializer_ref.endswith(":_init_worker_from_fork") and "fork" in methods:
        return "fork"
    if sys.platform == "darwin":
        return "spawn" if "spawn" in methods else methods[0]
    if "forkserver" in methods and _forkserver_transport_available():
        return "forkserver"
    if "spawn" in methods:
        return "spawn"
    return methods[0]


def cost_weighted_shards(tasks: Iterable[TaskSpec], shard_count: int) -> list[list[TaskSpec]]:
    """Deterministic greedy LPT assignment; output shard order and task order are canonical."""
    ordered = sorted(tasks, key=lambda t: (-float(t.cost_weight), t.task_id))
    if not ordered:
        return []
    n = max(1, min(int(shard_count), len(ordered)))
    shards: list[list[TaskSpec]] = [[] for _ in range(n)]
    weights = [0.0] * n
    for task in ordered:
        idx = min(range(n), key=lambda i: (weights[i], i))
        shards[idx].append(task)
        weights[idx] += float(task.cost_weight)
    for s in shards:
        s.sort(key=lambda t: t.task_id)
    return shards


def cost_weighted_bounded_shards(
    tasks: Iterable[TaskSpec], *, minimum_shards: int, max_tasks_per_shard: int
) -> list[list[TaskSpec]]:
    """Deterministic LPT shards with a hard logical-task bound per shard.

    Used by streaming execution so a worker can never return an unbounded list of
    compact task results.  Payload conversion is performed lazily by the caller.
    Completion order remains execution-only.
    """
    ordered = sorted(tasks, key=lambda t: (-float(t.cost_weight), t.task_id))
    if not ordered:
        return []
    limit = int(max_tasks_per_shard)
    if limit <= 0:
        raise ExecutionError("stream_shard_task_limit must be positive")
    required = math.ceil(len(ordered) / limit)
    n = max(1, min(len(ordered), max(int(minimum_shards), required)))
    shards: list[list[TaskSpec]] = [[] for _ in range(n)]
    weights = [0.0] * n
    for task in ordered:
        candidates = [i for i, sh in enumerate(shards) if len(sh) < limit]
        if not candidates:
            raise ExecutionError("bounded stream shard assignment exhausted capacity")
        idx = min(candidates, key=lambda i: (weights[i], len(shards[i]), i))
        shards[idx].append(task)
        weights[idx] += float(task.cost_weight)
    for sh in shards:
        sh.sort(key=lambda t: t.task_id)
    return shards


def _validate_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise DuplicateTaskError("duplicate deterministic task_id")
    return sorted(tasks, key=lambda t: t.task_id)


@contextmanager
def _suppress_main_reexecution_for_spawn_pool():
    """Prevent spawn/forkserver children from re-running an external launcher.

    Decoder workers use importable worker/initializer references, so they do not
    require the parent's ``__main__`` script.  Keeping ``__main__.__file__`` out
    of multiprocessing preparation data prevents a worker from re-executing the
    grandfathered controller bootstrap launcher.  The parent value is restored
    after the pool closes.  This does not grant controller context to workers.
    """
    import sys
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    sentinel = object()
    old_file = getattr(main, "__file__", sentinel)
    try:
        if old_file is not sentinel:
            main.__file__ = None
        yield
    finally:
        if old_file is sentinel:
            try:
                delattr(main, "__file__")
            except AttributeError:
                pass
        else:
            main.__file__ = old_file


def execute_tasks(
    tasks: Iterable[TaskSpec],
    *,
    worker_ref: str,
    policy: ExecutionPolicy | None = None,
    initializer_ref: str | None = None,
    initializer_payload: Any = None,
    telemetry: RuntimeTelemetry | None = None,
    scientific_task_units: Mapping[str, int] | None = None,
) -> ExecutionBatch:
    """Run deterministic independent tasks under the single Decoder-owned execution backend.

    Scientific modules supply TaskSpecs and a top-level worker callable reference. Scheduling,
    process pools, CPU leases, native thread suppression and completion-order erasure are owned here.
    """
    from .v05_origin_guard import require_controller_execution_origin, require_native_caller
    require_controller_execution_origin('shared-task-service')
    require_native_caller('infinity_grid.v05_stage_runtime', {'run_structural_partition', 'run_content_indexed_generation'}, 'shared-task-service')
    policy = policy or ExecutionPolicy()
    tasks = _validate_tasks(list(tasks))
    progress_units = {t.task_id: max(1, int((scientific_task_units or {}).get(t.task_id, 1))) for t in tasks}
    if not tasks:
        return ExecutionBatch({}, {
            "backend": "SERIAL", "workers": 0, "task_count": 0, "start_method": None,
            "scheduler": policy.scheduler, "wall_seconds": 0.0,
        })

    nested = os.environ.get("IG_DECODER_EXECUTION_WORKER") == "1"
    requested_pre = normalized_worker_request(policy.requested_workers, task_count=len(tasks), reserve_cores=policy.reserve_cores)
    if nested and requested_pre > 1:
        raise NestedParallelismError("Decoder workers may not create child process pools")

    backend = policy.backend.upper()
    if backend == "AUTO":
        backend = "SERIAL" if requested_pre <= 1 else "LOCAL_PROCESS_POOL"
    if backend not in {"SERIAL", "LOCAL_PROCESS_POOL"}:
        raise ExecutionError(f"unknown backend {policy.backend}")
    if backend == "SERIAL":
        requested_pre = 1

    lease_manager = HostCpuLeaseManager(policy.lease_root)
    t0 = time.monotonic()
    with lease_manager.acquire(
        requested_pre,
        owner=policy.owner,
        task_count=len(tasks),
        reserve_cores=policy.reserve_cores,
        wait=policy.wait_for_lease,
        timeout_seconds=policy.lease_timeout_seconds,
        poll_seconds=policy.poll_seconds,
    ) as lease:
        workers = 1 if backend == "SERIAL" else min(lease.workers, len(tasks))
        if workers <= 1:
            backend = "SERIAL"
        if telemetry is not None:
            telemetry.set_workers(leased=lease.workers, active=workers)
            telemetry.mark_started(task_units=sum(progress_units.values()), kernel_units=len(tasks))
        start_method = None
        rows: list[tuple[str, Any]] = []
        if backend == "SERIAL":
            _serial_initializer(worker_ref, initializer_ref, initializer_payload)
            for t in tasks:
                try:
                    row = _run_one(dataclasses.asdict(t))
                except BaseException as exc:
                    if telemetry is not None:
                        telemetry.mark_failed(exc, task_units=progress_units[t.task_id])
                    raise
                rows.append(row)
                if telemetry is not None:
                    telemetry.mark_completed(task_units=progress_units[t.task_id], kernel_units=1)
        else:
            start_method = _choose_start_method(policy, initializer_ref)
            ctx = mp.get_context(start_method)
            try:
                with _suppress_main_reexecution_for_spawn_pool():
                    with ctx.Pool(
                        processes=workers,
                        initializer=_pool_initializer,
                        initargs=(worker_ref, initializer_ref, initializer_payload),
                    ) as pool:
                        if policy.scheduler.upper() == "ORDERED_MAP":
                            wires = [dataclasses.asdict(t) for t in tasks]
                            for row in pool.imap_unordered(_run_one, wires, chunksize=1):
                                rows.append(row)
                                if telemetry is not None:
                                    telemetry.mark_completed(task_units=progress_units[row[0]], kernel_units=1)
                        elif policy.scheduler.upper() == "COST_WEIGHTED_SHARDS":
                            nshards = min(len(tasks), max(workers, workers * max(1, int(policy.target_shards_per_worker))))
                            shards = cost_weighted_shards(tasks, nshards)
                            shard_wires = [[dataclasses.asdict(t) for t in sh] for sh in shards]
                            for sr in pool.imap_unordered(_run_shard, shard_wires, chunksize=1):
                                rows.extend(sr)
                                if telemetry is not None:
                                    telemetry.mark_completed(
                                        task_units=sum(progress_units[tid] for tid, _ in sr),
                                        kernel_units=len(sr),
                                    )
                        else:
                            raise ExecutionError(f"unknown scheduler {policy.scheduler}")
            except (ExecutionError, WorkerTaskError) as exc:
                if telemetry is not None:
                    telemetry.mark_failed(exc, task_units=1)
                raise
            except BaseException as exc:
                if telemetry is not None:
                    telemetry.mark_failed(exc, task_units=1)
                raise WorkerTaskError(f"local process pool failed: {type(exc).__name__}: {exc}") from exc
        if telemetry is not None:
            telemetry.set_workers(leased=lease.workers, active=0)

    observed_ids = [r[0] for r in rows]
    expected_ids = [t.task_id for t in tasks]
    if len(observed_ids) != len(set(observed_ids)):
        raise DuplicateTaskError("duplicate worker result task_id")
    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected_ids))
        raise MissingTaskError(f"worker task result mismatch missing={missing} extra={extra}")
    rows.sort(key=lambda x: x[0])
    results = {tid: result for tid, result in rows}
    wall = time.monotonic() - t0
    metadata = {
        "backend": backend,
        "workers": workers,
        "task_count": len(tasks),
        "scientific_task_units": sum(progress_units.values()),
        "scheduler": policy.scheduler.upper(),
        "start_method": start_method,
        "wall_seconds": wall,
        "cpu_budget_detected": detect_effective_cpu_count(),
        "lease_id": lease.record["lease_id"],
        "lease_workers": lease.workers,
        "worker_count_not_science": True,
        "completion_order_not_science": True,
    }
    return ExecutionBatch(results, metadata)


def runtime_status(lease_root: str | Path | None = None) -> dict[str, Any]:
    mgr = HostCpuLeaseManager(lease_root)
    out = mgr.status()
    out.update({
        "effective_cpu_count": detect_effective_cpu_count(),
        "platform": platform.platform(),
        "start_methods": mp.get_all_start_methods(),
        "nested_worker": os.environ.get("IG_DECODER_EXECUTION_WORKER") == "1",
    })
    return out


def normalized_science_payload(obj: Any) -> Any:
    """Strip execution-only metadata recursively before a science-equivalence hash."""
    drop = {
        "workers", "worker_count", "worker_pids", "backend", "start_method",
        "wall_seconds", "cpu_seconds", "task_schedule", "parallel_execution",
        "execution_metadata", "lease_id", "lease_workers", "cpu_budget_detected",
        "runtime_status", "runtime_timings", "telemetry", "cache_stats",
        "scientific_task_units",
    }

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in sorted(x.items()) if k not in drop}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, tuple):
            return tuple(clean(v) for v in x)
        return x

    return clean(obj)


def science_payload_sha256(obj: Any) -> str:
    return canonical_sha256(normalized_science_payload(obj))


def load_unified_parallel_runtime_spec() -> dict[str, Any]:
    """Load the frozen v1 execution policy spec shipped with Decoder v0.27."""
    from importlib.resources import files
    return json.loads(files("infinity_grid").joinpath("resources/decoder/UNIFIED_PARALLEL_RUNTIME_SPEC_v1.json").read_text(encoding="utf-8"))


def execute_tasks_stream(
    tasks: Iterable[TaskSpec],
    *,
    worker_ref: str,
    on_result,
    policy: ExecutionPolicy | None = None,
    initializer_ref: str | None = None,
    initializer_payload: Any = None,
    telemetry: RuntimeTelemetry | None = None,
    scientific_task_units: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Decoder-owned streaming task execution.

    This is the durable-stage counterpart to :func:`execute_tasks`.  The common
    execution backend owns the process pool, CPU lease and scheduling, but does
    not accumulate scientific results in RAM.  Each completed logical task is
    delivered to ``on_result(task_id, result)`` immediately in the controller
    process.  The callback is expected to commit through an engine-owned store.

    Completion order is explicitly execution-only and must never enter a
    scientific hash.
    """
    from .v05_origin_guard import require_controller_execution_origin, require_native_caller
    require_controller_execution_origin('shared-task-service')
    require_native_caller('infinity_grid.v05_stage_runtime', {'run_structural_partition', 'run_content_indexed_generation'}, 'shared-task-service')
    policy = policy or ExecutionPolicy()
    tasks = _validate_tasks(list(tasks))
    progress_units = {t.task_id: max(1, int((scientific_task_units or {}).get(t.task_id, 1))) for t in tasks}
    if not tasks:
        return {
            "backend": "SERIAL", "workers": 0, "task_count": 0, "start_method": None,
            "scheduler": policy.scheduler, "wall_seconds": 0.0,
            "streaming_results": True,
        }

    nested = os.environ.get("IG_DECODER_EXECUTION_WORKER") == "1"
    requested_pre = normalized_worker_request(policy.requested_workers, task_count=len(tasks), reserve_cores=policy.reserve_cores)
    if nested and requested_pre > 1:
        raise NestedParallelismError("Decoder workers may not create child process pools")

    backend = policy.backend.upper()
    if backend == "AUTO":
        backend = "SERIAL" if requested_pre <= 1 else "LOCAL_PROCESS_POOL"
    if backend not in {"SERIAL", "LOCAL_PROCESS_POOL"}:
        raise ExecutionError(f"unknown backend {policy.backend}")
    if backend == "SERIAL":
        requested_pre = 1

    lease_manager = HostCpuLeaseManager(policy.lease_root)
    t0 = time.monotonic()
    completed = 0
    with lease_manager.acquire(
        requested_pre,
        owner=policy.owner,
        task_count=len(tasks),
        reserve_cores=policy.reserve_cores,
        wait=policy.wait_for_lease,
        timeout_seconds=policy.lease_timeout_seconds,
        poll_seconds=policy.poll_seconds,
    ) as lease:
        workers = 1 if backend == "SERIAL" else min(lease.workers, len(tasks))
        if workers <= 1:
            backend = "SERIAL"
        if telemetry is not None:
            telemetry.set_workers(leased=lease.workers, active=workers)
        start_method = None
        stream_shard_count = 0
        max_stream_shard_tasks = 1 if backend == "SERIAL" else 0
        max_inflight_shards = 1 if backend == "SERIAL" else 0
        max_observed_inflight_shards = 1 if backend == "SERIAL" else 0
        completed_stream_shards = 0
        seen: set[str] = set()

        def publish(row: tuple[str, Any]) -> None:
            nonlocal completed
            tid, result = row
            if tid in seen:
                raise DuplicateTaskError(f"duplicate worker result task_id {tid}")
            seen.add(tid)
            on_result(tid, result)
            completed += 1
            if telemetry is not None:
                telemetry.mark_completed(task_units=progress_units[tid], kernel_units=1)

        if backend == "SERIAL":
            _serial_initializer(worker_ref, initializer_ref, initializer_payload)
            for t in tasks:
                try:
                    if telemetry is not None:
                        telemetry.mark_started(task_units=progress_units[t.task_id], kernel_units=1)
                    publish(_run_one(dataclasses.asdict(t)))
                except BaseException as exc:
                    if telemetry is not None:
                        telemetry.mark_failed(exc, task_units=progress_units[t.task_id])
                    raise
        else:
            start_method = _choose_start_method(policy, initializer_ref)
            ctx = mp.get_context(start_method)
            try:
                with _suppress_main_reexecution_for_spawn_pool():
                    with ctx.Pool(
                        processes=workers,
                        initializer=_pool_initializer,
                        initargs=(worker_ref, initializer_ref, initializer_payload),
                    ) as pool:
                        if policy.scheduler.upper() == "ORDERED_MAP":
                            if telemetry is not None:
                                telemetry.mark_started(task_units=sum(progress_units.values()), kernel_units=len(tasks))
                            wires = [dataclasses.asdict(t) for t in tasks]
                            for row in pool.imap_unordered(_run_one, wires, chunksize=1):
                                publish(row)
                        elif policy.scheduler.upper() == "COST_WEIGHTED_SHARDS":
                            minimum_shards = min(len(tasks), max(workers, workers * max(1, int(policy.target_shards_per_worker))))
                            shards = cost_weighted_bounded_shards(
                                tasks, minimum_shards=minimum_shards,
                                max_tasks_per_shard=max(1, int(policy.stream_shard_task_limit)),
                            )
                            stream_shard_count = len(shards)
                            max_stream_shard_tasks = max((len(sh) for sh in shards), default=0)
                            max_inflight_shards = max(1, workers * max(1, int(policy.stream_inflight_shards_per_worker)))
                            # Keep a hard controller-owned dispatch window.  Only this many
                            # shard payload copies/results may be outstanding at once.
                            shard_iter = iter(shards)
                            inflight = []
                            exhausted = False
                            while inflight or not exhausted:
                                while not exhausted and len(inflight) < max_inflight_shards:
                                    try:
                                        sh = next(shard_iter)
                                    except StopIteration:
                                        exhausted = True
                                        break
                                    wire = [dataclasses.asdict(t) for t in sh]
                                    if telemetry is not None:
                                        telemetry.mark_started(
                                            task_units=sum(progress_units[t.task_id] for t in sh),
                                            kernel_units=len(sh),
                                        )
                                    inflight.append((pool.apply_async(_run_shard, (wire,)), len(sh)))
                                    max_observed_inflight_shards = max(max_observed_inflight_shards, len(inflight))
                                    if telemetry is not None:
                                        telemetry.set_detail(
                                            stream_inflight_shards=len(inflight),
                                            stream_inflight_shard_limit=max_inflight_shards,
                                            stream_shards_completed=completed_stream_shards,
                                            stream_shards_total=stream_shard_count,
                                        )
                                if not inflight:
                                    continue
                                ready_index = next((i for i,(ar,_n) in enumerate(inflight) if ar.ready()), None)
                                if ready_index is None:
                                    time.sleep(min(0.01, max(0.001, float(policy.poll_seconds))))
                                    continue
                                ar, declared_n = inflight.pop(ready_index)
                                sr = ar.get()
                                if len(sr) != declared_n or len(sr) > int(policy.stream_shard_task_limit):
                                    raise ExecutionError("worker returned invalid/over-limit stream shard")
                                for row in sr:
                                    publish(row)
                                completed_stream_shards += 1
                                if telemetry is not None:
                                    telemetry.set_detail(
                                        stream_inflight_shards=len(inflight),
                                        stream_inflight_shard_limit=max_inflight_shards,
                                        stream_shards_completed=completed_stream_shards,
                                        stream_shards_total=stream_shard_count,
                                        last_result_batch_tasks=len(sr),
                                        max_result_batch_tasks=max_stream_shard_tasks,
                                    )
                        else:
                            raise ExecutionError(f"unknown scheduler {policy.scheduler}")
            except (ExecutionError, WorkerTaskError):
                raise
            except BaseException as exc:
                if telemetry is not None:
                    telemetry.mark_failed(exc, task_units=1)
                raise WorkerTaskError(f"local process pool failed: {type(exc).__name__}: {exc}") from exc
        if telemetry is not None:
            telemetry.set_workers(leased=lease.workers, active=0)

    expected_ids = {t.task_id for t in tasks}
    if seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise MissingTaskError(f"worker task result mismatch missing={missing} extra={extra}")
    return {
        "backend": backend,
        "workers": workers,
        "task_count": len(tasks),
        "completed_task_count": completed,
        "scientific_task_units": sum(progress_units.values()),
        "scheduler": policy.scheduler.upper(),
        "start_method": start_method,
        "wall_seconds": time.monotonic() - t0,
        "cpu_budget_detected": detect_effective_cpu_count(),
        "lease_id": lease.record["lease_id"],
        "lease_workers": lease.workers,
        "streaming_results": True,
        "stream_shard_task_limit": int(policy.stream_shard_task_limit),
        "stream_shard_count": stream_shard_count,
        "max_stream_shard_tasks": max_stream_shard_tasks,
        "max_inflight_shards": max_inflight_shards,
        "max_observed_inflight_shards": max_observed_inflight_shards,
        "completed_stream_shards": completed_stream_shards,
        "worker_count_not_science": True,
        "completion_order_not_science": True,
    }
