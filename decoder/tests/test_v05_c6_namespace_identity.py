from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import pytest


def test_process_identity_parses_same_and_nested_namespace_status():
    from infinity_grid.v05_process_identity import _parse_status

    same = _parse_status("Pid:\t42\nPPid:\t41\nNSpid:\t42\n")
    assert same == {"proc_pid": 42, "proc_ppid": 41, "nspid": (42,)}
    nested = _parse_status("Pid:\t636537\nPPid:\t636534\nNSpid:\t636537\t5\n")
    assert nested == {
        "proc_pid": 636537,
        "proc_ppid": 636534,
        "nspid": (636537, 5),
    }


def test_process_identity_rejects_inconsistent_outer_nspid():
    from infinity_grid.v05_process_identity import ProcessIdentityError, _parse_status

    with pytest.raises(ProcessIdentityError, match="NSPID_OUTER"):
        _parse_status("Pid:\t42\nPPid:\t41\nNSpid:\t43\t5\n")


def test_current_process_identity_binds_local_and_proc_views():
    from infinity_grid.v05_process_identity import current_process_identity

    identity = current_process_identity()
    assert identity["local_pid"] == os.getpid()
    assert identity["nspid"][-1] == os.getpid()
    assert identity["nspid"][0] == identity["proc_pid"]
    assert identity["start_time_ticks"] > 0


def test_forked_child_observes_both_parent_identities():
    from infinity_grid.v05_process_identity import current_process_identity, process_identity

    parent = current_process_identity()
    rfd, wfd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(rfd)
        try:
            current = current_process_identity()
            visible_parent = process_identity(current["proc_ppid"])
            result = {
                "local_ppid": os.getppid(),
                "proc_ppid": current["proc_ppid"],
                "parent_nspid": list(visible_parent["nspid"]),
                "parent_start_time_ticks": visible_parent["start_time_ticks"],
            }
            os.write(wfd, (json.dumps(result, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(wfd)
        os._exit(0)
    os.close(wfd)
    raw = b""
    while True:
        block = os.read(rfd, 65536)
        if not block:
            break
        raw += block
    os.close(rfd)
    _, status = os.waitpid(child, 0)
    assert status == 0
    result = json.loads(raw)
    assert result["local_ppid"] == parent["local_pid"]
    assert result["proc_ppid"] == parent["proc_pid"]
    assert result["parent_nspid"] == list(parent["nspid"])
    assert result["parent_start_time_ticks"] == parent["start_time_ticks"]


def _guard_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import infinity_grid.v05_origin_guard as guard

    service = tmp_path / "service"
    package = service / "accepted_source" / "infinity_grid"
    package.mkdir(parents=True)
    entrypoint = package / "v05_c6_recovery_entrypoint.py"
    module = package / "v05_c6_recovery.py"
    entrypoint.write_text("entry\n", encoding="utf-8")
    module.write_text("module\n", encoding="utf-8")
    (service / "supervisor.lock").write_bytes(b"")
    lease_path = service / "runtime" / "supervisor" / "lease.json"
    lease_path.parent.mkdir(parents=True)

    local_parent = 22
    proc_parent = 222
    start_time = 987654
    nspid = (proc_parent, local_parent)
    source_sha = "a" * 64
    secret = "b" * 64
    service_sha = hashlib.sha256(str(service.resolve()).encode("utf-8")).hexdigest()
    entry_sha = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    module_sha = hashlib.sha256(module.read_bytes()).hexdigest()
    capsule = {
        "schema_id": "IG_DECODER_C6_RECOVERY_CAPSULE_V2",
        "generation": 24,
        "accepted_source_sha256": source_sha,
        "accepted_package_sha256": "c" * 64,
        "source_zip_sha256": "d" * 64,
        "source_manifest_file_sha256": "e" * 64,
        "c5_acceptance_file_sha256": "f" * 64,
        "supervisor_entrypoint_sha256": entry_sha,
        "supervisor_module_sha256": module_sha,
        "child_entrypoint_sha256": "1" * 64,
        "controller_entrypoint_sha256": "2" * 64,
        "service_root_sha256": service_sha,
        "final_origin_exclusivity": True,
    }
    capsule_path = service / "RECOVERY_CAPSULE.json"
    capsule_path.write_text(json.dumps(capsule), encoding="utf-8")
    lease = {
        "schema_id": "IG_DECODER_C6_SUPERVISOR_LEASE_V3",
        "supervisor_pid": local_parent,
        "supervisor_proc_pid": proc_parent,
        "supervisor_nspid": list(nspid),
        "supervisor_start_time_ticks": start_time,
        "secret_sha256": hashlib.sha256(secret.encode("ascii")).hexdigest(),
        "accepted_source_sha256": source_sha,
        "capsule_sha256": hashlib.sha256(capsule_path.read_bytes()).hexdigest(),
        "supervisor_entrypoint_sha256": entry_sha,
        "supervisor_module_sha256": module_sha,
        "service_root_sha256": service_sha,
    }
    lease_path.write_text(json.dumps(lease), encoding="utf-8")

    monkeypatch.setattr(guard.os, "getppid", lambda: local_parent)
    monkeypatch.setattr(
        guard,
        "current_process_identity",
        lambda: {
            "local_pid": 23,
            "proc_pid": 223,
            "proc_ppid": proc_parent,
            "nspid": (223, 23),
            "start_time_ticks": 123,
        },
    )
    monkeypatch.setattr(
        guard,
        "process_identity",
        lambda pid: {
            "proc_pid": pid,
            "proc_ppid": 111,
            "nspid": nspid,
            "start_time_ticks": start_time,
        },
    )
    monkeypatch.setattr(
        guard,
        "_c6_read_proc_cmdline",
        lambda pid: [str(Path(guard.sys.executable).resolve()), "-B", "-I", "-S", str(entrypoint)],
    )
    monkeypatch.setattr(guard, "_c6_read_proc_exe", lambda pid: Path(guard.sys.executable).resolve())
    monkeypatch.setattr(guard, "_c6_parent_holds_service_lock", lambda pid, path: True)
    return guard, lease_path, lease, local_parent, proc_parent, source_sha, secret


def test_origin_guard_accepts_exact_dual_namespace_parent(tmp_path, monkeypatch):
    guard, lease_path, _lease, local_parent, _proc_parent, source_sha, secret = _guard_fixture(
        tmp_path, monkeypatch
    )
    guard._c6_verify_supervisor_identity(
        supervisor_pid=local_parent,
        supervisor_secret=secret,
        lease_path=lease_path,
        expected_source_sha256=source_sha,
    )


@pytest.mark.parametrize(
    "mutation,detail",
    [
        ("proc_pid", "lease-proc-pid"),
        ("nspid", "lease-nspid"),
        ("start_time", "lease-start-time"),
        ("lock", "lifecycle-lock"),
        ("cmdline", "supervisor-cmdline"),
    ],
)
def test_origin_guard_rejects_each_namespace_identity_break(
    tmp_path, monkeypatch, mutation, detail
):
    guard, lease_path, lease, local_parent, proc_parent, source_sha, secret = _guard_fixture(
        tmp_path, monkeypatch
    )
    if mutation == "proc_pid":
        lease["supervisor_proc_pid"] = proc_parent + 1
    elif mutation == "nspid":
        lease["supervisor_nspid"] = [proc_parent, local_parent + 1]
    elif mutation == "start_time":
        lease["supervisor_start_time_ticks"] += 1
    elif mutation == "lock":
        monkeypatch.setattr(guard, "_c6_parent_holds_service_lock", lambda pid, path: False)
    elif mutation == "cmdline":
        monkeypatch.setattr(
            guard,
            "_c6_read_proc_cmdline",
            lambda pid: [str(Path(guard.sys.executable).resolve()), "wrapper.py"],
        )
    lease_path.write_text(json.dumps(lease), encoding="utf-8")
    with pytest.raises(Exception, match=detail):
        guard._c6_verify_supervisor_identity(
            supervisor_pid=local_parent,
            supervisor_secret=secret,
            lease_path=lease_path,
            expected_source_sha256=source_sha,
        )


def test_origin_guard_rejects_wrong_local_parent(tmp_path, monkeypatch):
    guard, lease_path, _lease, local_parent, _proc_parent, source_sha, secret = _guard_fixture(
        tmp_path, monkeypatch
    )
    with pytest.raises(Exception, match="supervisor-parent"):
        guard._c6_verify_supervisor_identity(
            supervisor_pid=local_parent + 1,
            supervisor_secret=secret,
            lease_path=lease_path,
            expected_source_sha256=source_sha,
        )


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_files(source: Path):
    skip = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".engineering_tmp"}
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in skip for part in rel.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            yield path, rel.as_posix()


def test_exact_supervisor_child_roundtrip_in_nested_pid_namespace(tmp_path):
    """Exercise the real zero-payload supervisor/child boundary in this executor."""
    from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
    from infinity_grid.v05_execution_authority import source_tree_digest

    repository_source = Path(__file__).resolve().parents[1]
    service = tmp_path / "service"
    source = service / "accepted_source"
    shutil.copytree(
        repository_source,
        source,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "*.pyo"),
    )
    runtime = service / "runtime"
    runtime.mkdir()

    source_sha = engineering_source_tree_digest(source)
    package_sha = source_tree_digest(source / "infinity_grid")
    rows = [
        {"path": rel, "sha256": _sha_file(path), "size_bytes": path.stat().st_size}
        for path, rel in _source_files(source)
    ]
    manifest = {
        "schema_id": "IG_DECODER_SOURCE_MANIFEST_V1",
        "source_sha256": source_sha,
        "files": rows,
    }
    manifest_path = service / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    source_zip = service / "ACTIVE_SOURCE_SOURCE.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, rel in _source_files(source):
            archive.write(path, rel)

    c5 = {
        "schema_id": "IG_DECODER_TEST_ONLY_C5_BINDING_V1",
        "status": "PASS",
        "final_origin_exclusivity": True,
        "final_source_sha256": source_sha,
        "scientific_effect": "NONE",
    }
    c5_path = service / "C5_FINAL_ACCEPTANCE.json"
    c5_path.write_text(json.dumps(c5, sort_keys=True), encoding="utf-8")

    entrypoint = source / "infinity_grid" / "v05_c6_recovery_entrypoint.py"
    module = source / "infinity_grid" / "v05_c6_recovery.py"
    child = source / "infinity_grid" / "v05_c6_recovery_child.py"
    controller = source / "infinity_grid" / "v05_controller_event_loop.py"
    capsule = {
        "schema_id": "IG_DECODER_C6_RECOVERY_CAPSULE_V2",
        "generation": 1,
        "accepted_source_sha256": source_sha,
        "accepted_package_sha256": package_sha,
        "source_zip_sha256": _sha_file(source_zip),
        "source_manifest_file_sha256": _sha_file(manifest_path),
        "c5_acceptance_file_sha256": _sha_file(c5_path),
        "supervisor_entrypoint_sha256": _sha_file(entrypoint),
        "supervisor_module_sha256": _sha_file(module),
        "child_entrypoint_sha256": _sha_file(child),
        "controller_entrypoint_sha256": _sha_file(controller),
        "service_root_sha256": hashlib.sha256(str(service.resolve()).encode("utf-8")).hexdigest(),
        "final_origin_exclusivity": True,
    }
    (service / "RECOVERY_CAPSULE.json").write_text(
        json.dumps(capsule, sort_keys=True), encoding="utf-8"
    )

    from infinity_grid.v05_passive_intake import submit_passive_request
    request_id = "c6-live-recorded-attempt-probe"
    submit_passive_request(runtime / "intake", {
        "schema_id": "IG_DECODER_PASSIVE_REQUEST_V1",
        "request_id": request_id,
        "registered_job_id": "DECODER.C6.RECOVERY",
        "requested_operation_id": "C6_CONTEXT_PROBE",
        "parent_source_sha256": source_sha,
        "input_artifacts": [],
    })

    log_path = service / "integration-supervisor.log"
    command = [sys.executable, "-B", "-I", "-S", str(entrypoint)]
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
    }
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
    try:
        deadline = time.monotonic() + 20
        status_path = runtime / "status" / "CONTROLLER_STATUS.json"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            completion_path = runtime / "intake" / "completed" / (request_id + ".json")
            if status_path.is_file() and completion_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") == "RUNNING":
                    break
            time.sleep(0.1)
        assert proc.poll() is None, log_path.read_text(encoding="utf-8", errors="replace")
        assert status_path.is_file(), log_path.read_text(encoding="utf-8", errors="replace")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["status"] == "RUNNING"
        assert status["final_origin_exclusivity"] is True
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        assert completion["status"] == "PASS"
        assert completion["result"]["status"] == "PASS"
        attempts = sorted((runtime / "attempts" / request_id).glob("*.json"))
        assert len(attempts) == 1
        assert json.loads(attempts[0].read_text(encoding="utf-8"))["status"] == "COMPLETED"

        lease = json.loads((runtime / "supervisor" / "lease.json").read_text(encoding="utf-8"))
        assert lease["schema_id"] == "IG_DECODER_C6_SUPERVISOR_LEASE_V3"
        assert lease["supervisor_pid"] == status["supervisor_pid"]
        assert lease["supervisor_nspid"][-1] == lease["supervisor_pid"]
        assert lease["supervisor_proc_pid"] == lease["supervisor_nspid"][0]
        assert lease["supervisor_start_time_ticks"] > 0
        assert (runtime / "supervisor" / "controller-child.log").read_bytes() == b""
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
