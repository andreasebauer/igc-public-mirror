from __future__ import annotations

"""C6 persistent Decoder recovery supervisor.

This module is a lifecycle component only.  It accepts no Decoder request payload,
selects no operation, and never mints a controller execution context.  Its only
supported action is to make the exact hash-bound accepted controller available.
Substantive execution remains owned by ``controller_child_main`` and begins only
inside the controller event loop after passive-request admission.
"""

import hashlib
import errno
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .v05_engineering_worker import engineering_source_tree_digest
from .v05_execution_authority import source_tree_digest
from .v05_process_identity import ProcessIdentityError, current_process_identity

CAPSULE_SCHEMA = "IG_DECODER_C6_RECOVERY_CAPSULE_V2"
LEASE_SCHEMA = "IG_DECODER_C6_SUPERVISOR_LEASE_V3"
STATUS_SCHEMA = "IG_DECODER_C6_SUPERVISOR_STATUS_V1"
RESTART_TO_ACTIVE_SOURCE = 75
STAGED_INSTALL_SCHEMA = "IG_DECODER_C6_STAGED_FIXED_INSTALL_V1"

CAPSULE_FIELDS = frozenset({
    "schema_id", "generation", "accepted_source_sha256", "accepted_package_sha256",
    "source_zip_sha256", "source_manifest_file_sha256", "c5_acceptance_file_sha256",
    "supervisor_entrypoint_sha256", "supervisor_module_sha256", "child_entrypoint_sha256",
    "controller_entrypoint_sha256", "service_root_sha256", "final_origin_exclusivity",
})
FORBIDDEN_RECOVERY_SELECTORS = frozenset({
    "request_id", "job_id", "registered_job_id", "operation", "requested_operation_id",
    "command", "shell", "module", "function", "evaluator", "callback", "worker_count",
    "input_path", "output_path", "overlay", "patch", "resume_token", "root_capability",
    "worker_capability", "publication", "promotion", "source_path", "capsule_path",
})
_SHA = frozenset("0123456789abcdef")


class RecoveryError(RuntimeError):
    pass


def _sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _is_sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA


def _atomic_json(path: Path, obj: Mapping[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(obj), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass



def _atomic_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def stage_fixed_installation_after_acceptance(*, runtime_root: str | Path,
                                              transition: Mapping[str, Any],
                                              acceptance_path: str | Path) -> dict[str, Any]:
    """Controller-owned staging for the next canonical C6 fixed installation.

    This does not mutate the running fixed install.  It only binds already-C5-accepted
    transition evidence for the lifecycle supervisor.  The supervisor may persist it
    only after the controller child has exited with RESTART_TO_ACTIVE_SOURCE.
    """
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin("c6-stage-fixed-install")
    runtime = Path(runtime_root).resolve(strict=True)
    service = runtime.parent.resolve(strict=True)
    candidate = Path(str(transition.get("candidate_source_path", ""))).resolve(strict=True)
    transitions_root = (runtime / "source_transitions").resolve(strict=True)
    if not candidate.is_relative_to(transitions_root) or candidate.name != "candidate_source":
        raise RecoveryError("STAGED_INSTALL_CANDIDATE_PATH")
    source_sha = engineering_source_tree_digest(candidate)
    package_sha = source_tree_digest(candidate / "infinity_grid")
    if transition.get("status") != "PASS" or transition.get("candidate_source_sha256") != source_sha:
        raise RecoveryError("STAGED_INSTALL_SOURCE_BINDING")
    if transition.get("candidate_package_sha256") != package_sha:
        raise RecoveryError("STAGED_INSTALL_PACKAGE_BINDING")
    transition_root = candidate.parent
    source_zip = (transition_root / "source.zip").resolve(strict=True)
    manifest = (transition_root / "SOURCE_MANIFEST.json").resolve(strict=True)
    if _sha_file(source_zip) != transition.get("source_zip_sha256"):
        raise RecoveryError("STAGED_INSTALL_SOURCE_ZIP_BINDING")
    _manifest_replay(candidate, manifest)
    manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_obj.get("source_sha256") != source_sha:
        raise RecoveryError("STAGED_INSTALL_MANIFEST_SOURCE")
    acceptance = Path(acceptance_path).resolve(strict=True)
    acc = json.loads(acceptance.read_text(encoding="utf-8"))
    if acc.get("status") != "PASS" or acc.get("final_origin_exclusivity") is not True:
        raise RecoveryError("STAGED_INSTALL_C5_STATUS")
    if acc.get("final_source_sha256") != source_sha:
        raise RecoveryError("STAGED_INSTALL_C5_SOURCE")
    fixed_capsule = validate_capsule(json.loads((service / "RECOVERY_CAPSULE.json").read_text(encoding="utf-8")))
    receipt = {
        "schema_id": STAGED_INSTALL_SCHEMA,
        "candidate_source_path": str(candidate),
        "accepted_source_sha256": source_sha,
        "accepted_package_sha256": package_sha,
        "source_zip_path": str(source_zip),
        "source_zip_sha256": _sha_file(source_zip),
        "source_manifest_path": str(manifest),
        "source_manifest_file_sha256": _sha_file(manifest),
        "c5_acceptance_path": str(acceptance),
        "c5_acceptance_file_sha256": _sha_file(acceptance),
        "from_generation": fixed_capsule["generation"],
        "to_generation": fixed_capsule["generation"] + 1,
        "final_origin_exclusivity": True,
    }
    staged = runtime / "final" / "STAGED_FIXED_INSTALL.json"
    if staged.exists():
        old = json.loads(staged.read_text(encoding="utf-8"))
        if old != receipt:
            raise RecoveryError("STAGED_INSTALL_CONFLICT")
    else:
        _atomic_json(staged, receipt, 0o444)
    return receipt


def _load_staged_install(runtime: Path) -> dict[str, Any] | None:
    path = runtime / "final" / "STAGED_FIXED_INSTALL.json"
    if not path.is_file():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_id", "candidate_source_path", "accepted_source_sha256", "accepted_package_sha256",
        "source_zip_path", "source_zip_sha256", "source_manifest_path", "source_manifest_file_sha256",
        "c5_acceptance_path", "c5_acceptance_file_sha256", "from_generation", "to_generation",
        "final_origin_exclusivity",
    }
    if type(obj) is not dict or set(obj) != required or obj.get("schema_id") != STAGED_INSTALL_SCHEMA:
        raise RecoveryError("STAGED_INSTALL_SCHEMA")
    if obj.get("final_origin_exclusivity") is not True:
        raise RecoveryError("STAGED_INSTALL_ORIGIN_EXCLUSIVITY")
    for key in ("accepted_source_sha256", "accepted_package_sha256", "source_zip_sha256",
                "source_manifest_file_sha256", "c5_acceptance_file_sha256"):
        if not _is_sha(obj.get(key)):
            raise RecoveryError("STAGED_INSTALL_SHA:" + key)
    if type(obj.get("from_generation")) is not int or type(obj.get("to_generation")) is not int:
        raise RecoveryError("STAGED_INSTALL_GENERATION")
    if obj["to_generation"] != obj["from_generation"] + 1:
        raise RecoveryError("STAGED_INSTALL_GENERATION_STEP")
    return obj


def _archive_current_source_for_fixed_install(
    *, service: Path, history: Path, old_source: Path, current: Mapping[str, Any],
) -> tuple[Path, Path | None]:
    """Archive the verified current source without weakening the fixed-path boundary.

    A native rename remains preferred.  Some managed executors expose the service and
    runtime paths through mounts that reject the cross-directory rename even though
    both parents are writable.  In only that case, create and verify the history copy
    first, then move the live tree to a same-parent rollback path before activation.
    """
    expected_source = current["accepted_source_sha256"]
    expected_package = current["accepted_package_sha256"]
    if engineering_source_tree_digest(old_source) != expected_source:
        raise RecoveryError("STAGED_INSTALL_CURRENT_SOURCE_DIGEST")
    if source_tree_digest(old_source / "infinity_grid") != expected_package:
        raise RecoveryError("STAGED_INSTALL_CURRENT_PACKAGE_DIGEST")

    archived_source = history / "accepted_source"
    fallback_errnos = {errno.EXDEV, errno.EACCES, errno.EPERM}
    if not archived_source.exists():
        try:
            os.replace(old_source, archived_source)
            return archived_source, None
        except OSError as exc:
            if exc.errno not in fallback_errnos:
                raise

        temporary = history / (".accepted_source.copy-" + str(os.getpid()))
        if temporary.exists():
            raise RecoveryError("STAGED_INSTALL_HISTORY_COPY_TEMP_EXISTS")
        try:
            shutil.copytree(old_source, temporary)
            if engineering_source_tree_digest(temporary) != expected_source:
                raise RecoveryError("STAGED_INSTALL_HISTORY_COPY_SOURCE_DIGEST")
            if source_tree_digest(temporary / "infinity_grid") != expected_package:
                raise RecoveryError("STAGED_INSTALL_HISTORY_COPY_PACKAGE_DIGEST")
            _manifest_replay(temporary, history / "SOURCE_MANIFEST.json")
            os.replace(temporary, archived_source)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    else:
        if engineering_source_tree_digest(archived_source) != expected_source:
            raise RecoveryError("STAGED_INSTALL_HISTORY_SOURCE_EXISTS")
        if source_tree_digest(archived_source / "infinity_grid") != expected_package:
            raise RecoveryError("STAGED_INSTALL_HISTORY_PACKAGE_EXISTS")
        _manifest_replay(archived_source, history / "SOURCE_MANIFEST.json")

    rollback_source = service / (
        ".accepted_source.rollback-generation-%04d-%s"
        % (current["generation"], expected_source[:16])
    )
    if rollback_source.exists():
        raise RecoveryError("STAGED_INSTALL_ROOT_ROLLBACK_EXISTS")
    os.replace(old_source, rollback_source)
    return archived_source, rollback_source


def _persist_staged_fixed_installation(info: Mapping[str, Any], staged: Mapping[str, Any]) -> Path:
    """Persist a controller-accepted source only after its child has stopped.

    This is lifecycle persistence only.  No request/operation selector enters here.
    On success the fixed Recovery Capsule and canonical accepted_source agree exactly.
    """
    p = info["paths"]
    service = p["service"]
    runtime = p["runtime"]
    current = info["capsule"]
    if staged["from_generation"] != current["generation"] or staged["to_generation"] != current["generation"] + 1:
        raise RecoveryError("STAGED_INSTALL_GENERATION_BINDING")
    candidate = Path(staged["candidate_source_path"]).resolve(strict=True)
    transitions_root = (runtime / "source_transitions").resolve(strict=True)
    if not candidate.is_relative_to(transitions_root) or candidate.name != "candidate_source":
        raise RecoveryError("STAGED_INSTALL_CANDIDATE_PATH")
    if engineering_source_tree_digest(candidate) != staged["accepted_source_sha256"]:
        raise RecoveryError("STAGED_INSTALL_SOURCE_DIGEST")
    if source_tree_digest(candidate / "infinity_grid") != staged["accepted_package_sha256"]:
        raise RecoveryError("STAGED_INSTALL_PACKAGE_DIGEST")
    source_zip = Path(staged["source_zip_path"]).resolve(strict=True)
    manifest = Path(staged["source_manifest_path"]).resolve(strict=True)
    acceptance = Path(staged["c5_acceptance_path"]).resolve(strict=True)
    if _sha_file(source_zip) != staged["source_zip_sha256"]:
        raise RecoveryError("STAGED_INSTALL_SOURCE_ZIP_HASH")
    if _sha_file(manifest) != staged["source_manifest_file_sha256"]:
        raise RecoveryError("STAGED_INSTALL_MANIFEST_HASH")
    if _sha_file(acceptance) != staged["c5_acceptance_file_sha256"]:
        raise RecoveryError("STAGED_INSTALL_C5_HASH")
    _manifest_replay(candidate, manifest)
    acc = json.loads(acceptance.read_text(encoding="utf-8"))
    if acc.get("status") != "PASS" or acc.get("final_origin_exclusivity") is not True or acc.get("final_source_sha256") != staged["accepted_source_sha256"]:
        raise RecoveryError("STAGED_INSTALL_C5_BINDING")

    token = staged["accepted_source_sha256"][:16]
    next_source = service / (".accepted_source.next-" + token)
    if next_source.exists():
        shutil.rmtree(next_source)
    shutil.copytree(candidate, next_source)
    if engineering_source_tree_digest(next_source) != staged["accepted_source_sha256"]:
        raise RecoveryError("STAGED_INSTALL_COPY_SOURCE_DIGEST")
    if source_tree_digest(next_source / "infinity_grid") != staged["accepted_package_sha256"]:
        raise RecoveryError("STAGED_INSTALL_COPY_PACKAGE_DIGEST")

    next_zip = service / (".ACTIVE_SOURCE_SOURCE.zip.next-" + token)
    next_manifest = service / (".SOURCE_MANIFEST.json.next-" + token)
    next_c5 = service / (".C5_FINAL_ACCEPTANCE.json.next-" + token)
    _atomic_bytes(next_zip, source_zip.read_bytes(), 0o444)
    _atomic_bytes(next_manifest, manifest.read_bytes(), 0o444)
    _atomic_bytes(next_c5, acceptance.read_bytes(), 0o444)
    next_capsule = {
        "schema_id": CAPSULE_SCHEMA,
        "generation": staged["to_generation"],
        "accepted_source_sha256": staged["accepted_source_sha256"],
        "accepted_package_sha256": staged["accepted_package_sha256"],
        "source_zip_sha256": _sha_file(next_zip),
        "source_manifest_file_sha256": _sha_file(next_manifest),
        "c5_acceptance_file_sha256": _sha_file(next_c5),
        "supervisor_entrypoint_sha256": _sha_file(next_source / "infinity_grid" / "v05_c6_recovery_entrypoint.py"),
        "supervisor_module_sha256": _sha_file(next_source / "infinity_grid" / "v05_c6_recovery.py"),
        "child_entrypoint_sha256": _sha_file(next_source / "infinity_grid" / "v05_c6_recovery_child.py"),
        "controller_entrypoint_sha256": _sha_file(next_source / "infinity_grid" / "v05_controller_event_loop.py"),
        "service_root_sha256": hashlib.sha256(str(service).encode("utf-8")).hexdigest(),
        "final_origin_exclusivity": True,
    }
    next_capsule_path = service / (".RECOVERY_CAPSULE.json.next-" + token)
    _atomic_json(next_capsule_path, next_capsule, 0o444)

    history = runtime / "recovery_history" / ("generation-%04d-%s" % (current["generation"], current["accepted_source_sha256"][:12]))
    history.mkdir(parents=True, exist_ok=True)
    for name in ("RECOVERY_CAPSULE.json", "SOURCE_MANIFEST.json", "C5_FINAL_ACCEPTANCE.json", "ACTIVE_SOURCE_SOURCE.zip"):
        src = service / name
        dst = history / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    old_source = service / "accepted_source"
    archived_source, rollback_source = _archive_current_source_for_fixed_install(
        service=service, history=history, old_source=old_source, current=current,
    )
    try:
        os.replace(next_source, old_source)
        os.replace(next_zip, service / "ACTIVE_SOURCE_SOURCE.zip")
        os.replace(next_manifest, service / "SOURCE_MANIFEST.json")
        os.replace(next_c5, service / "C5_FINAL_ACCEPTANCE.json")
        os.replace(next_capsule_path, service / "RECOVERY_CAPSULE.json")
    except BaseException:
        if not old_source.exists():
            if rollback_source is not None and rollback_source.exists():
                os.replace(rollback_source, old_source)
            elif archived_source.exists():
                os.replace(archived_source, old_source)
        raise

    final = runtime / "final"
    _atomic_bytes(final / "ACTIVE_SOURCE_PATH.txt", (str(old_source.resolve()) + "\n").encode("utf-8"), 0o444)
    _atomic_bytes(final / "ACTIVE_SOURCE_SHA256.txt", (staged["accepted_source_sha256"] + "\n").encode("ascii"), 0o444)
    applied = dict(staged)
    applied["schema_id"] = "IG_DECODER_C6_APPLIED_FIXED_INSTALL_V1"
    _atomic_json(final / ("APPLIED_FIXED_INSTALL_" + token + ".json"), applied, 0o444)
    try:
        (final / "STAGED_FIXED_INSTALL.json").unlink()
    except FileNotFoundError:
        pass
    return old_source / "infinity_grid" / "v05_c6_recovery_entrypoint.py"


def _exec_persisted_supervisor(entrypoint: Path) -> None:
    from .isolated_runtime import subprocess_environment
    env = subprocess_environment()
    os.execve(sys.executable, [sys.executable, "-B", "-I", "-S", str(entrypoint.resolve(strict=True))], env)

def validate_recovery_argv(argv: list[str] | tuple[str, ...]) -> None:
    """The supported recovery CLI has zero semantic arguments."""
    if type(argv) not in (list, tuple) or len(argv) != 1:
        raise RecoveryError("REJECT_RECOVERY_PAYLOAD")


def validate_capsule(obj: Mapping[str, Any]) -> dict[str, Any]:
    if type(obj) is not dict or set(obj) != CAPSULE_FIELDS:
        raise RecoveryError("RECOVERY_CAPSULE_SCHEMA")
    if obj.get("schema_id") != CAPSULE_SCHEMA:
        raise RecoveryError("RECOVERY_CAPSULE_SCHEMA_ID")
    if type(obj.get("generation")) is not int or obj["generation"] < 1:
        raise RecoveryError("RECOVERY_CAPSULE_GENERATION")
    for key in CAPSULE_FIELDS:
        if key.endswith("sha256") and not _is_sha(obj.get(key)):
            raise RecoveryError("RECOVERY_CAPSULE_SHA256:" + key)
    if obj.get("final_origin_exclusivity") is not True:
        raise RecoveryError("RECOVERY_CAPSULE_ORIGIN_EXCLUSIVITY")
    if set(obj) & FORBIDDEN_RECOVERY_SELECTORS:
        raise RecoveryError("RECOVERY_CAPSULE_EXECUTION_SELECTOR")
    return dict(obj)


def service_root_from_module() -> Path:
    """Return the canonical service root; no caller-provided path is accepted."""
    source_root = Path(__file__).resolve().parent.parent
    if source_root.name != "accepted_source":
        raise RecoveryError("RECOVERY_CANONICAL_INSTALL_REQUIRED")
    return source_root.parent.resolve()


def fixed_paths() -> dict[str, Path]:
    service = service_root_from_module()
    source = (service / "accepted_source").resolve()
    paths = {
        "service": service,
        "source": source,
        "runtime": (service / "runtime").resolve(),
        "capsule": (service / "RECOVERY_CAPSULE.json").resolve(),
        "source_zip": (service / "ACTIVE_SOURCE_SOURCE.zip").resolve(),
        "source_manifest": (service / "SOURCE_MANIFEST.json").resolve(),
        "c5_acceptance": (service / "C5_FINAL_ACCEPTANCE.json").resolve(),
        "supervisor_entrypoint": (source / "infinity_grid" / "v05_c6_recovery_entrypoint.py").resolve(),
        "supervisor_module": (source / "infinity_grid" / "v05_c6_recovery.py").resolve(),
        "child_entrypoint": (source / "infinity_grid" / "v05_c6_recovery_child.py").resolve(),
        "controller_entrypoint": (source / "infinity_grid" / "v05_controller_event_loop.py").resolve(),
    }
    return paths


def _manifest_replay(source: Path, manifest_path: Path) -> None:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(obj) is not dict or set(obj) != {"schema_id", "source_sha256", "files"}:
        raise RecoveryError("RECOVERY_SOURCE_MANIFEST_SCHEMA")
    rows = obj.get("files")
    if type(rows) is not list:
        raise RecoveryError("RECOVERY_SOURCE_MANIFEST_FILES")
    declared: dict[str, str] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            raise RecoveryError("RECOVERY_SOURCE_MANIFEST_ROW")
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or "\\" in row["path"]:
            raise RecoveryError("RECOVERY_SOURCE_MANIFEST_PATH")
        if not _is_sha(row["sha256"]):
            raise RecoveryError("RECOVERY_SOURCE_MANIFEST_SHA")
        declared[rel.as_posix()] = row["sha256"]
    actual: dict[str, str] = {}
    for p in sorted(source.rglob("*")):
        if p.is_symlink():
            raise RecoveryError("RECOVERY_SOURCE_SYMLINK")
        if p.is_file():
            rel = p.relative_to(source).as_posix()
            if "__pycache__" in Path(rel).parts or p.suffix in {".pyc", ".pyo"}:
                continue
            actual[rel] = _sha_file(p)
    if set(actual) != set(declared):
        raise RecoveryError("RECOVERY_SOURCE_MANIFEST_SET")
    for rel, sha in declared.items():
        if actual[rel] != sha:
            raise RecoveryError("RECOVERY_SOURCE_MANIFEST_MISMATCH:" + rel)


def verify_fixed_installation() -> dict[str, Any]:
    p = fixed_paths()
    for key in ("source", "capsule", "source_zip", "source_manifest", "c5_acceptance",
                "supervisor_entrypoint", "supervisor_module", "child_entrypoint", "controller_entrypoint"):
        if not p[key].exists():
            raise RecoveryError("RECOVERY_ARTIFACT_MISSING:" + key)
    capsule = validate_capsule(json.loads(p["capsule"].read_text(encoding="utf-8")))
    if _sha_file(p["source_zip"]) != capsule["source_zip_sha256"]:
        raise RecoveryError("RECOVERY_SOURCE_ZIP_HASH")
    if _sha_file(p["source_manifest"]) != capsule["source_manifest_file_sha256"]:
        raise RecoveryError("RECOVERY_MANIFEST_FILE_HASH")
    if _sha_file(p["c5_acceptance"]) != capsule["c5_acceptance_file_sha256"]:
        raise RecoveryError("RECOVERY_C5_ACCEPTANCE_FILE_HASH")
    if _sha_file(p["supervisor_entrypoint"]) != capsule["supervisor_entrypoint_sha256"]:
        raise RecoveryError("RECOVERY_SUPERVISOR_ENTRYPOINT_HASH")
    if _sha_file(p["supervisor_module"]) != capsule["supervisor_module_sha256"]:
        raise RecoveryError("RECOVERY_SUPERVISOR_MODULE_HASH")
    if hashlib.sha256(str(p["service"]).encode("utf-8")).hexdigest() != capsule["service_root_sha256"]:
        raise RecoveryError("RECOVERY_SERVICE_ROOT_HASH")
    if _sha_file(p["child_entrypoint"]) != capsule["child_entrypoint_sha256"]:
        raise RecoveryError("RECOVERY_CHILD_ENTRYPOINT_HASH")
    if _sha_file(p["controller_entrypoint"]) != capsule["controller_entrypoint_sha256"]:
        raise RecoveryError("RECOVERY_CONTROLLER_ENTRYPOINT_HASH")
    _manifest_replay(p["source"], p["source_manifest"])
    source_sha = engineering_source_tree_digest(p["source"])
    package_sha = source_tree_digest(p["source"] / "infinity_grid")
    if source_sha != capsule["accepted_source_sha256"]:
        raise RecoveryError("RECOVERY_SOURCE_DIGEST")
    if package_sha != capsule["accepted_package_sha256"]:
        raise RecoveryError("RECOVERY_PACKAGE_DIGEST")
    c5 = json.loads(p["c5_acceptance"].read_text(encoding="utf-8"))
    if c5.get("status") != "PASS" or c5.get("final_origin_exclusivity") is not True:
        raise RecoveryError("RECOVERY_C5_ACCEPTANCE")
    if c5.get("final_source_sha256") != source_sha:
        raise RecoveryError("RECOVERY_C5_SOURCE_BINDING")
    return {"paths": p, "capsule": capsule, "source_sha256": source_sha, "package_sha256": package_sha}


def _write_lease(
    runtime: Path, source_sha: str, capsule_sha: str, *,
    supervisor_entrypoint: Path, supervisor_module: Path, service_root: Path,
) -> str:
    """Write a V3 lease binding both local and procfs supervisor identities.

    The accepted source may advance after a controller-owned source transition, but the
    supervisor identity remains the actual parent process that owns the lifecycle lock.
    """
    import secrets
    entrypoint = Path(supervisor_entrypoint).resolve(strict=True)
    module = Path(supervisor_module).resolve(strict=True)
    service = Path(service_root).resolve(strict=True)
    try:
        identity = current_process_identity()
    except ProcessIdentityError as exc:
        raise RecoveryError("RECOVERY_SUPERVISOR_PROC_IDENTITY") from exc
    if identity["local_pid"] != os.getpid():
        raise RecoveryError("RECOVERY_SUPERVISOR_LOCAL_IDENTITY")
    secret = secrets.token_hex(32)
    lease = {
        "schema_id": LEASE_SCHEMA,
        "supervisor_pid": identity["local_pid"],
        "supervisor_proc_pid": identity["proc_pid"],
        "supervisor_nspid": list(identity["nspid"]),
        "supervisor_start_time_ticks": identity["start_time_ticks"],
        "secret_sha256": hashlib.sha256(secret.encode("ascii")).hexdigest(),
        "accepted_source_sha256": source_sha,
        "capsule_sha256": capsule_sha,
        "supervisor_entrypoint_sha256": _sha_file(entrypoint),
        "supervisor_module_sha256": _sha_file(module),
        "service_root_sha256": hashlib.sha256(str(service).encode("utf-8")).hexdigest(),
    }
    _atomic_json(runtime / "supervisor" / "lease.json", lease, 0o600)
    return secret


def _restore_runtime_bindings(info: Mapping[str, Any]) -> None:
    p = info["paths"]
    runtime = p["runtime"]
    final = runtime / "final"
    final.mkdir(parents=True, exist_ok=True)
    for name in ("C5_FINAL_ACCEPTANCE.json",):
        dst = final / name
        if not dst.exists():
            dst.write_bytes(p["c5_acceptance"].read_bytes())
            os.chmod(dst, 0o444)
    _atomic_bytes(final / "ACTIVE_SOURCE_PATH.txt", (str(p["source"]) + "\n").encode("utf-8"), 0o444)
    _atomic_bytes(final / "ACTIVE_SOURCE_SHA256.txt", (info["source_sha256"] + "\n").encode("ascii"), 0o444)


def _spawn_child(source: Path, runtime: Path, secret: str, child_entrypoint: Path) -> subprocess.Popen[bytes]:
    from .isolated_runtime import subprocess_environment
    env = subprocess_environment()
    rfd, wfd = os.pipe()
    try:
        os.write(wfd, (secret + "\n").encode("ascii"))
    finally:
        os.close(wfd)
    env.update({
        "IG_C6_SECRET_FD": str(rfd),
        "IG_C6_RUNTIME_ROOT": str(runtime),
        "IG_C6_SOURCE_ROOT": str(source),
    })
    log = (runtime / "supervisor" / "controller-child.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    fp = log.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-B", "-I", "-S", str(child_entrypoint)],
            cwd=str(source), env=env, stdin=subprocess.DEVNULL, stdout=fp, stderr=fp,
            close_fds=True, pass_fds=(rfd,), start_new_session=True,
        )
    finally:
        os.close(rfd)
        fp.close()
    return proc


def _terminate_controller_process_group(pgid: int, sig: int = signal.SIGTERM) -> None:
    """Best-effort controller-tree cleanup; the controller is launched in its own session."""
    try:
        os.killpg(int(pgid), int(sig))
    except ProcessLookupError:
        return


def _verified_active_source(runtime: Path) -> Path:
    source_path = (runtime / "final" / "ACTIVE_SOURCE_PATH.txt").read_text(encoding="utf-8").strip()
    expected = (runtime / "final" / "ACTIVE_SOURCE_SHA256.txt").read_text(encoding="utf-8").strip()
    source = Path(source_path).resolve(strict=True)
    if engineering_source_tree_digest(source) != expected:
        raise RecoveryError("RECOVERY_ACTIVE_SOURCE_MISMATCH")
    acc = json.loads((runtime / "final" / "C5_FINAL_ACCEPTANCE.json").read_text(encoding="utf-8"))
    if acc.get("status") != "PASS" or acc.get("final_origin_exclusivity") is not True or acc.get("final_source_sha256") != expected:
        raise RecoveryError("RECOVERY_ACTIVE_ACCEPTANCE_MISMATCH")
    return source


def supervisor_main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    validate_recovery_argv(args)
    info = verify_fixed_installation()
    p = info["paths"]
    runtime = p["runtime"]
    runtime.mkdir(parents=True, exist_ok=True)
    lock_path = p["service"] / "supervisor.lock"
    import fcntl
    lockf = lock_path.open("a+b")
    try:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    _restore_runtime_bindings(info)
    capsule_sha = _sha_file(p["capsule"])
    source = p["source"]
    running_entrypoint = Path(sys.argv[0]).resolve(strict=True)
    running_module = Path(__file__).resolve(strict=True)
    if running_entrypoint != p["supervisor_entrypoint"]:
        raise RecoveryError("RECOVERY_RUNNING_ENTRYPOINT_BINDING")
    if running_module != p["supervisor_module"]:
        raise RecoveryError("RECOVERY_RUNNING_MODULE_BINDING")
    stopping = False
    child: subprocess.Popen[bytes] | None = None

    def _sig(_n: int, _f: Any) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            _terminate_controller_process_group(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    while not stopping:
        # Lease is recreated for each child lifetime; it conveys no job or operation selector.
        secret = _write_lease(
            runtime, engineering_source_tree_digest(source), capsule_sha,
            supervisor_entrypoint=running_entrypoint, supervisor_module=running_module,
            service_root=p["service"],
        )
        child_entry = source / "infinity_grid" / "v05_c6_recovery_child.py"
        if not child_entry.is_file():
            raise RecoveryError("RECOVERY_CHILD_ENTRYPOINT_MISSING")
        child = _spawn_child(source, runtime, secret, child_entry)
        _atomic_json(runtime / "supervisor" / "status.json", {
            "schema_id": STATUS_SCHEMA,
            "status": "RUNNING",
            "supervisor_pid": os.getpid(),
            "controller_pid": child.pid,
            "source_sha256": engineering_source_tree_digest(source),
            "capsule_generation": info["capsule"]["generation"],
            "scientific_effect": "NONE",
        }, 0o444)
        child_pgid = child.pid
        active_marker=runtime/'status'/'ACTIVE_REQUEST.json'
        try: active_marker.unlink()
        except FileNotFoundError: pass
        from .v05_cancel_control import process_next_cancel_request
        while child.poll() is None and not stopping:
            directive=process_next_cancel_request(runtime,expected_source_sha256=engineering_source_tree_digest(source))
            if directive is not None and directive.get('terminate_controller_child') is True:
                _terminate_controller_process_group(child_pgid, signal.SIGTERM)
                break
            time.sleep(0.10)
        rc = child.wait()
        # The multiprocessing forkserver/resource tracker/workers inherit the
        # controller session. Ensure no descendants survive a stop or restart.
        _terminate_controller_process_group(child_pgid, signal.SIGKILL)
        child = None
        if stopping:
            break
        if rc == RESTART_TO_ACTIVE_SOURCE:
            staged = _load_staged_install(runtime)
            if staged is not None:
                entrypoint = _persist_staged_fixed_installation(info, staged)
                _exec_persisted_supervisor(entrypoint)
            source = _verified_active_source(runtime)
            continue
        # Unexpected exit: restore availability from the currently accepted path only.
        source = _verified_active_source(runtime)
        time.sleep(0.5)
    _atomic_json(runtime / "supervisor" / "status.json", {
        "schema_id": STATUS_SCHEMA, "status": "STOPPED", "supervisor_pid": os.getpid(),
        "scientific_effect": "NONE",
    }, 0o444)
    return 0


__all__ = [
    "CAPSULE_SCHEMA", "RecoveryError", "validate_recovery_argv", "validate_capsule",
    "stage_fixed_installation_after_acceptance",
    "service_root_from_module", "fixed_paths", "verify_fixed_installation", "supervisor_main",
]
