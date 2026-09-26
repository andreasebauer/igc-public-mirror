from __future__ import annotations

"""Controller-event execution context for origin-exclusivity migration C2.

C5 final origin guard. Root controller contexts are minted only by the long-lived supervisor child; workers may not create or re-enter root execution.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import secrets
import hashlib
import json
import os
import sys
from types import MappingProxyType
from pathlib import Path
from typing import Iterator

from .v05_execution_authority import ExecutionAuthorityError
from .v05_process_identity import (
    ProcessIdentityError,
    current_process_identity,
    process_identity,
)

REJECT_EXTERNAL_EXECUTION_ORIGIN = "REJECT_EXTERNAL_EXECUTION_ORIGIN"
REJECT_WORKER_ROOT_REENTRY = "REJECT_WORKER_ROOT_REENTRY"
CONTROLLER_EVENT_ORIGIN = "CONTROLLER_EVENT_LOOP"
WORKER_EXECUTION_ORIGIN = "DECODER_WORKER"
FORKED_WORKER_INGRESS = "CONTROLLER_DESCENDED_FORK_WORKER"
SUPERVISOR_EVENT_INGRESS = "LONG_LIVED_CONTROLLER_EVENT_LOOP"
CONTROLLER_ROOT_ROLE = "CONTROLLER_ROOT"
WORKER_ROLE = "WORKER"

_CONTEXT_MINT_KEY = object()
_CONTEXT_SEAL = object()
_ACTIVE_CONTEXT: ContextVar[object | None] = ContextVar("ig_decoder_execution_context", default=None)


def _issue(code: str, detail: str | None = None) -> ExecutionAuthorityError:
    from .invocation import InvocationRefused
    ctx = _ACTIVE_CONTEXT.get()
    binding = ctx.binding if type(ctx) is _ExecutionContext else {}
    options = {}
    if code == 'OUTPUT_OUTSIDE_RECORDED_ATTEMPT':
        options = {'next_operation':'runtime.publish_json',
                   'required_arguments':{'logical_name':None, 'obj':None,
                       'correction':'Use the supplied runtime without replacing its output root.'}}
    return InvocationRefused(code, detail or "execution", workspace=binding.get('workspace'),
                             job_id=binding.get('job_id'), **options)


class _ExecutionContext:
    __slots__ = (
        "role", "controller_session_id", "root_execution_id", "context_id",
        "parent_context_id", "ingress", "active", "_seal", "owner_pid", "binding",
    )

    def __init__(self, *, role: str, controller_session_id: str,
                 root_execution_id: str, context_id: str,
                 parent_context_id: str | None, ingress: str, _key: object, binding=None) -> None:
        if _key is not _CONTEXT_MINT_KEY:
            raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "context-mint")
        if role not in {CONTROLLER_ROOT_ROLE, WORKER_ROLE}:
            raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "context-role")
        self.role = role
        self.controller_session_id = controller_session_id
        self.root_execution_id = root_execution_id
        self.context_id = context_id
        self.parent_context_id = parent_context_id
        self.ingress = ingress
        self.active = True
        self._seal = _CONTEXT_SEAL
        self.owner_pid = os.getpid()
        self.binding = MappingProxyType(dict(binding or {}))

    def snapshot(self) -> dict[str, str | None]:
        return {
            "role": self.role,
            "origin": CONTROLLER_EVENT_ORIGIN if self.role == CONTROLLER_ROOT_ROLE else WORKER_EXECUTION_ORIGIN,
            "controller_session_id": self.controller_session_id,
            "root_execution_id": self.root_execution_id,
            "context_id": self.context_id,
            "parent_context_id": self.parent_context_id,
            "ingress": self.ingress,
            "owner_pid": self.owner_pid,
            "attempt": dict(self.binding),
        }


def _live_context() -> _ExecutionContext | None:
    ctx = _ACTIVE_CONTEXT.get()
    if type(ctx) is not _ExecutionContext or ctx._seal is not _CONTEXT_SEAL or not ctx.active or ctx.owner_pid != os.getpid():
        return None
    if not _attempt_is_live(ctx.binding):
        return None
    return ctx


def require_controller_execution_origin(entrypoint: str | None = None) -> None:
    ctx = _live_context()
    if ctx is None:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, entrypoint)
    if ctx.role == WORKER_ROLE:
        raise _issue(REJECT_WORKER_ROOT_REENTRY, entrypoint)
    if ctx.role != CONTROLLER_ROOT_ROLE:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, entrypoint)


def require_worker_execution_origin(entrypoint: str | None = None) -> None:
    ctx = _live_context()
    if ctx is None:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, entrypoint)
    if ctx.role != WORKER_ROLE:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, entrypoint)


def current_execution_origin() -> str | None:
    ctx = _live_context()
    if ctx is None:
        return None
    return CONTROLLER_EVENT_ORIGIN if ctx.role == CONTROLLER_ROOT_ROLE else WORKER_EXECUTION_ORIGIN


def current_execution_context_snapshot() -> dict[str, str | None] | None:
    ctx = _live_context()
    return None if ctx is None else ctx.snapshot()


def _new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(16)


@contextmanager
def _controller_event_scope(ingress: str, *, _key: object, _binding=None) -> Iterator[_ExecutionContext]:
    if _key is not _CONTEXT_MINT_KEY:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "controller-context-mint")
    if not _attempt_is_live(_binding):
        raise _issue("RECORDED_ATTEMPT_REQUIRED", "controller-context-mint")
    previous = _live_context()
    if previous is not None:
        if previous.role == WORKER_ROLE:
            raise _issue(REJECT_WORKER_ROOT_REENTRY, "controller-context-mint")
        if previous.role == CONTROLLER_ROOT_ROLE:
            # Nested controller calls are part of the same root event.  Reuse
            # the exact context rather than minting a second root identity.
            yield previous
            return
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "controller-context-role")
    ctx = _ExecutionContext(
        role=CONTROLLER_ROOT_ROLE,
        controller_session_id=_new_id("session-"),
        root_execution_id=_new_id("root-"),
        context_id=_new_id("ctx-"),
        parent_context_id=None,
        ingress=str(ingress),
        binding=_binding,
        _key=_CONTEXT_MINT_KEY,
    )
    token = _ACTIVE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        ctx.active = False
        _ACTIVE_CONTEXT.reset(token)


@contextmanager
def _controller_worker_scope(work_unit_id: str, *, ingress: str = FORKED_WORKER_INGRESS, _key: object) -> Iterator[_ExecutionContext]:
    if _key is not _CONTEXT_MINT_KEY:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "worker-context-mint")
    root = _ACTIVE_CONTEXT.get()
    if (type(root) is not _ExecutionContext or root._seal is not _CONTEXT_SEAL
        or not root.active or root.owner_pid == os.getpid()
        or root.owner_pid != os.getppid() or not _attempt_is_live(root.binding)):
        root = None
    if root is None:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "worker-context-without-root")
    if root.role == WORKER_ROLE:
        raise _issue(REJECT_WORKER_ROOT_REENTRY, "worker-context-nesting")
    if root.role != CONTROLLER_ROOT_ROLE:
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "worker-context-parent-role")
    unit = str(work_unit_id)
    if not unit or len(unit) > 128 or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for ch in unit):
        raise _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, "worker-context-id")
    worker = _ExecutionContext(
        role=WORKER_ROLE,
        controller_session_id=root.controller_session_id,
        root_execution_id=root.root_execution_id,
        context_id=_new_id("worker-"),
        parent_context_id=root.context_id,
        ingress=str(ingress),
        binding=root.binding,
        _key=_CONTEXT_MINT_KEY,
    )
    token = _ACTIVE_CONTEXT.set(worker)
    try:
        yield worker
    finally:
        worker.active = False
        _ACTIVE_CONTEXT.reset(token)


@contextmanager
def _forked_worker_scope(work_unit_id: str) -> Iterator[_ExecutionContext]:
    """Internal C4 worker ingress; succeeds only in a fork inheriting a live root."""
    with _controller_worker_scope(work_unit_id, ingress=FORKED_WORKER_INGRESS, _key=_CONTEXT_MINT_KEY) as ctx:
        yield ctx


_C6_LEASE_SCHEMA = "IG_DECODER_C6_SUPERVISOR_LEASE_V3"
_C6_CAPSULE_SCHEMA = "IG_DECODER_C6_RECOVERY_CAPSULE_V2"
_C6_LEASE_FIELDS = frozenset({
    "schema_id", "supervisor_pid", "supervisor_proc_pid", "supervisor_nspid",
    "supervisor_start_time_ticks", "secret_sha256", "accepted_source_sha256",
    "capsule_sha256", "supervisor_entrypoint_sha256", "supervisor_module_sha256",
    "service_root_sha256",
})
_C6_CAPSULE_FIELDS = frozenset({
    "schema_id", "generation", "accepted_source_sha256", "accepted_package_sha256",
    "source_zip_sha256", "source_manifest_file_sha256", "c5_acceptance_file_sha256",
    "supervisor_entrypoint_sha256", "supervisor_module_sha256", "child_entrypoint_sha256",
    "controller_entrypoint_sha256", "service_root_sha256", "final_origin_exclusivity",
})
_C6_SHA_CHARS = frozenset("0123456789abcdef")


def _c6_is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _C6_SHA_CHARS


def _c6_sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _c6_reject(detail: str, exc: BaseException | None = None) -> ExecutionAuthorityError:
    issue = _issue(REJECT_EXTERNAL_EXECUTION_ORIGIN, detail)
    if exc is not None:
        issue.__cause__ = exc
    return issue


def _c6_read_proc_cmdline(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        parts = raw.split(b"\0")
        if parts and parts[-1] == b"":
            parts.pop()
        return [x.decode("utf-8", "strict") for x in parts]
    except Exception as exc:
        raise _c6_reject("supervisor-proc-cmdline", exc)


def _c6_read_proc_exe(pid: int) -> Path:
    try:
        return (Path("/proc") / str(pid) / "exe").resolve(strict=True)
    except Exception as exc:
        raise _c6_reject("supervisor-proc-exe", exc)


def _c6_parent_holds_service_lock(pid: int, lock_path: Path) -> bool:
    """Linux fail-closed proof that *pid* owns the fixed exclusive lifecycle flock."""
    try:
        st = lock_path.stat()
        dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
        target = f"{dev}:{st.st_ino}".lower()
        rows = Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise _c6_reject("supervisor-lock-observation", exc)
    for row in rows:
        parts = row.split()
        if len(parts) < 6:
            continue
        if parts[1] == "FLOCK" and parts[2] == "ADVISORY" and parts[3] == "WRITE":
            if parts[4] == str(pid) and parts[5].lower() == target:
                return True
    return False


def _c6_verify_supervisor_identity(
    *, supervisor_pid: int, supervisor_secret: str, lease_path: str | Path,
    expected_source_sha256: str,
) -> None:
    """Verify the exact integrated recovery supervisor before root-context minting.

    This check intentionally uses only fixed Linux process/service identity plus the V3 lease.
    A caller-controlled lease path, fake parent, non-isolated Python parent, wrong service root,
    wrong capsule, wrong accepted child source or process without the lifecycle lock is rejected.
    """
    if type(supervisor_pid) is not int or supervisor_pid <= 1 or supervisor_pid != os.getppid():
        raise _c6_reject("supervisor-parent")
    if not _c6_is_sha(expected_source_sha256):
        raise _c6_reject("supervisor-source-binding")
    if type(supervisor_secret) is not str or len(supervisor_secret) != 64 or any(c not in _C6_SHA_CHARS for c in supervisor_secret):
        raise _c6_reject("supervisor-secret-shape")

    try:
        child_identity = current_process_identity()
        proc_supervisor_pid = child_identity["proc_ppid"]
        if type(proc_supervisor_pid) is not int or proc_supervisor_pid <= 1:
            raise ProcessIdentityError("PROC_IDENTITY_PARENT_PID")
        supervisor_identity = process_identity(proc_supervisor_pid)
    except ProcessIdentityError as exc:
        raise _c6_reject("supervisor-namespace-identity", exc)
    if supervisor_identity["nspid"][-1] != supervisor_pid:
        raise _c6_reject("supervisor-namespace-map")

    cmd = _c6_read_proc_cmdline(proc_supervisor_pid)
    if len(cmd) != 5 or cmd[1:4] != ["-B", "-I", "-S"]:
        raise _c6_reject("supervisor-cmdline")
    try:
        parent_exe = _c6_read_proc_exe(proc_supervisor_pid)
        child_exe = Path(sys.executable).resolve(strict=True)
        argv_exe = Path(cmd[0]).resolve(strict=True)
        entrypoint = Path(cmd[4]).resolve(strict=True)
    except Exception as exc:
        raise _c6_reject("supervisor-proc-identity", exc)
    if parent_exe != child_exe or argv_exe != child_exe:
        raise _c6_reject("supervisor-interpreter")
    if entrypoint.name != "v05_c6_recovery_entrypoint.py" or entrypoint.parent.name != "infinity_grid":
        raise _c6_reject("supervisor-entrypoint")
    accepted_source = entrypoint.parent.parent
    if accepted_source.name != "accepted_source":
        raise _c6_reject("supervisor-canonical-install")
    service_root = accepted_source.parent.resolve(strict=True)
    expected_lease = (service_root / "runtime" / "supervisor" / "lease.json").resolve(strict=True)
    try:
        actual_lease_path = Path(lease_path).resolve(strict=True)
    except Exception as exc:
        raise _c6_reject("supervisor-lease-path", exc)
    if actual_lease_path != expected_lease:
        raise _c6_reject("supervisor-lease-path")

    try:
        lease = json.loads(expected_lease.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _c6_reject("supervisor-lease", exc)
    if type(lease) is not dict or set(lease) != _C6_LEASE_FIELDS or lease.get("schema_id") != _C6_LEASE_SCHEMA:
        raise _c6_reject("supervisor-lease-schema")
    for key in _C6_LEASE_FIELDS:
        if key.endswith("sha256") and not _c6_is_sha(lease.get(key)):
            raise _c6_reject("supervisor-lease-sha")
    if lease.get("supervisor_pid") != supervisor_pid:
        raise _c6_reject("supervisor-lease-pid")
    if lease.get("supervisor_proc_pid") != proc_supervisor_pid:
        raise _c6_reject("supervisor-lease-proc-pid")
    if lease.get("supervisor_nspid") != list(supervisor_identity["nspid"]):
        raise _c6_reject("supervisor-lease-nspid")
    if lease.get("supervisor_start_time_ticks") != supervisor_identity["start_time_ticks"]:
        raise _c6_reject("supervisor-lease-start-time")
    if lease.get("accepted_source_sha256") != expected_source_sha256:
        raise _c6_reject("supervisor-lease-source")
    if lease.get("secret_sha256") != hashlib.sha256(supervisor_secret.encode("ascii")).hexdigest():
        raise _c6_reject("supervisor-secret")

    module = (entrypoint.parent / "v05_c6_recovery.py").resolve(strict=True)
    service_hash = hashlib.sha256(str(service_root).encode("utf-8")).hexdigest()
    if lease.get("supervisor_entrypoint_sha256") != _c6_sha_file(entrypoint):
        raise _c6_reject("supervisor-entrypoint-hash")
    if lease.get("supervisor_module_sha256") != _c6_sha_file(module):
        raise _c6_reject("supervisor-module-hash")
    if lease.get("service_root_sha256") != service_hash:
        raise _c6_reject("supervisor-service-root")

    capsule_path = (service_root / "RECOVERY_CAPSULE.json").resolve(strict=True)
    try:
        capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _c6_reject("supervisor-capsule", exc)
    if type(capsule) is not dict or set(capsule) != _C6_CAPSULE_FIELDS or capsule.get("schema_id") != _C6_CAPSULE_SCHEMA:
        raise _c6_reject("supervisor-capsule-schema")
    if capsule.get("final_origin_exclusivity") is not True:
        raise _c6_reject("supervisor-capsule-origin")
    for key in _C6_CAPSULE_FIELDS:
        if key.endswith("sha256") and not _c6_is_sha(capsule.get(key)):
            raise _c6_reject("supervisor-capsule-sha")
    if lease.get("capsule_sha256") != _c6_sha_file(capsule_path):
        raise _c6_reject("supervisor-capsule-hash")
    if capsule.get("supervisor_entrypoint_sha256") != lease.get("supervisor_entrypoint_sha256"):
        raise _c6_reject("supervisor-capsule-entrypoint")
    if capsule.get("supervisor_module_sha256") != lease.get("supervisor_module_sha256"):
        raise _c6_reject("supervisor-capsule-module")
    if capsule.get("service_root_sha256") != service_hash:
        raise _c6_reject("supervisor-capsule-service-root")

    lock_path = (service_root / "supervisor.lock").resolve()
    if not lock_path.is_file() or not _c6_parent_holds_service_lock(proc_supervisor_pid, lock_path):
        raise _c6_reject("supervisor-lifecycle-lock")


@contextmanager
def _supervisor_controller_event_scope(
    *, supervisor_pid: int, supervisor_secret: str, lease_path: str | Path,
    expected_source_sha256: str, attempt_binding=None,
) -> Iterator[_ExecutionContext]:
    """Mint a root context only for a child of the exact integrated recovery supervisor."""
    _c6_verify_supervisor_identity(
        supervisor_pid=supervisor_pid, supervisor_secret=supervisor_secret,
        lease_path=lease_path, expected_source_sha256=expected_source_sha256,
    )
    with _controller_event_scope(
        SUPERVISOR_EVENT_INGRESS, _key=_CONTEXT_MINT_KEY, _binding=attempt_binding,
    ) as ctx:
        yield ctx



def _attempt_is_live(binding):
    if not binding:
        return False
    try:
        from .canon import canonical_sha256
        raw = json.loads(Path(binding['attempt_path']).read_text())
        return raw['status'] == 'RUNNING' and canonical_sha256(raw) == binding['attempt_sha256']
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextmanager
def registered_workspace_scope(workspace: str | Path, job_id: str):
    """Compatibility name: reading a registration never grants execution."""
    from .invocation import InvocationRefused
    raise InvocationRefused('RECORDED_ATTEMPT_REQUIRED', 'registered_workspace_scope',
                            workspace=workspace, job_id=job_id)
    yield  # preserve context-manager call shape for an actionable refusal


@contextmanager
def _registered_attempt_scope(admission, attempt_path, output_dir):
    """Native run entry only, after saving and persisting the RUNNING attempt.

    Caller/byte checks catch accidental API misuse, not same-user Python tampering.
    """
    from . import v05_controller_event_loop as loop
    from .canon import canonical_sha256
    # contextlib.__enter__ resumes this generator on behalf of the native runner.
    if sys._getframe(2).f_code is not loop._run_workspace_job.__code__:
        raise _issue('NATIVE_RUN_ENTRY_REQUIRED', '_registered_attempt_scope')
    record = json.loads(Path(attempt_path).read_text())
    job = admission['job']; req = admission['request']
    expected = {'status': 'RUNNING', 'job_id': job['job_id'], 'pid': os.getpid(),
                'source_sha256': admission['source_sha256'],
                'registration_sha256': job['registration_sha256'],
                'operation': req['requested_operation_id'],
                'output_root': str(Path(output_dir).resolve())}
    if any(record.get(k) != v for k, v in expected.items()):
        raise _issue('ATTEMPT_BINDING_MISMATCH', '_registered_attempt_scope')
    binding = dict(expected, workspace=str(admission['workspace']),
                   attempt_path=str(Path(attempt_path).resolve()),
                   attempt_sha256=canonical_sha256(record), attempt_id=record['attempt_id'],
                   request_id=req['request_id'])
    with _controller_event_scope('REGISTERED_ATTEMPT_V1', _key=_CONTEXT_MINT_KEY, _binding=binding):
        yield admission


def require_registered_output(path, operation='publication'):
    require_controller_execution_origin(operation)
    binding = _live_context().binding
    if not Path(path).resolve().is_relative_to(Path(binding['output_root'])):
        raise _issue('OUTPUT_OUTSIDE_RECORDED_ATTEMPT', operation)


def require_registered_dispatch(admission, output_dir, operation='dispatch'):
    require_registered_output(output_dir, operation)
    from .canon import canonical_sha256
    binding = _live_context().binding
    job = admission['job']
    if (str(admission['workspace']) != binding['workspace']
        or admission['source_sha256'] != binding['source_sha256']
        or job['job_id'] != binding['job_id']
        or job['registration_sha256'] != binding['registration_sha256']
        or canonical_sha256({k:v for k,v in job.items() if k != 'registration_sha256'}) != binding['registration_sha256']
        or admission['request']['requested_operation_id'] != binding['operation']):
        raise _issue('ATTEMPT_BINDING_MISMATCH', operation)


def require_native_caller(module, names, operation):
    """Reject supported helper calls made outside their native owner."""
    frame = sys._getframe(2)
    imported = sys.modules.get(module)
    if imported is None or frame.f_globals is not vars(imported) or frame.f_code.co_name not in names:
        raise _issue('NATIVE_RUN_ENTRY_REQUIRED', operation)
