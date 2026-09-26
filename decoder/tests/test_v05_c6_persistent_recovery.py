from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
from pathlib import Path
import pytest


def _capsule(**changes):
    import infinity_grid.v05_c6_recovery as r
    base = {
        "schema_id": r.CAPSULE_SCHEMA,
        "generation": 1,
        "accepted_source_sha256": "a" * 64,
        "accepted_package_sha256": "b" * 64,
        "source_zip_sha256": "c" * 64,
        "source_manifest_file_sha256": "d" * 64,
        "c5_acceptance_file_sha256": "e" * 64,
        "supervisor_entrypoint_sha256": "f" * 64,
        "supervisor_module_sha256": "0" * 64,
        "child_entrypoint_sha256": "1" * 64,
        "controller_entrypoint_sha256": "2" * 64,
        "service_root_sha256": "3" * 64,
        "final_origin_exclusivity": True,
    }
    base.update(changes)
    return base


def test_c6_recovery_cli_is_zero_payload():
    from infinity_grid.v05_c6_recovery import validate_recovery_argv, RecoveryError
    validate_recovery_argv(["recovery"])
    for argv in (["recovery", "run"], ["recovery", "--source", "x"], []):
        with pytest.raises(RecoveryError, match="REJECT_RECOVERY_PAYLOAD"):
            validate_recovery_argv(argv)


def test_c6_capsule_exact_fields_and_no_execution_selectors():
    import infinity_grid.v05_c6_recovery as r
    assert r.validate_capsule(_capsule())["generation"] == 1
    bad = _capsule(); bad["operation"] = "run"
    with pytest.raises(r.RecoveryError, match="RECOVERY_CAPSULE_SCHEMA|EXECUTION_SELECTOR"):
        r.validate_capsule(bad)
    bad = _capsule(final_origin_exclusivity=False)
    with pytest.raises(r.RecoveryError, match="ORIGIN_EXCLUSIVITY"):
        r.validate_capsule(bad)


def test_c6_recovery_has_no_decoder_operation_selection_surface():
    import infinity_grid.v05_c6_recovery as r
    sig = inspect.signature(r.supervisor_main)
    assert list(sig.parameters) == ["argv"]
    src = inspect.getsource(r)
    forbidden = [
        "run_controller_registered_source_transition(",
        "run_observer_continuity_repair(",
        "run_wider_feature_search(",
        "RegisteredExecutionService(",
        "subprocess.run(",
        "shell=True",
    ]
    for token in forbidden:
        assert token not in src


def test_c6_service_paths_are_derived_not_caller_selected():
    import infinity_grid.v05_c6_recovery as r
    assert list(inspect.signature(r.fixed_paths).parameters) == []
    assert list(inspect.signature(r.service_root_from_module).parameters) == []


def test_c6_child_direct_invocation_requires_supervisor_secret(monkeypatch):
    import infinity_grid.v05_c6_recovery_child as c
    monkeypatch.delenv("IG_C6_SECRET_FD", raising=False)
    with pytest.raises(RuntimeError, match="REJECT_EXTERNAL_EXECUTION_ORIGIN:C6_SUPERVISOR_ATTESTATION"):
        c._read_secret()


def test_c6_c5_direct_bootstrap_remains_closed(tmp_path):
    from infinity_grid.v05_controller_event_loop import start_c5_migration_supervisor
    with pytest.raises(Exception, match="REJECT_DIRECT_EXECUTION_ROUTE:C5_BOOTSTRAP_CLOSED"):
        start_c5_migration_supervisor(tmp_path / "r", tmp_path)


def test_c6_resource_states_no_new_execution_surface():
    from importlib import resources
    obj = json.loads(resources.files("infinity_grid").joinpath("resources/v05/C6_PERSISTENT_RECOVERY_V1.json").read_text())
    assert obj["recovery_payload_fields"] == []
    assert obj["normal_external_request_surfaces"] == ["passive_request.submit", "passive_cancel.submit"]
    assert obj["direct_execution_routes"] == "REJECT"


def test_c6_recovery_child_launch_uses_isolated_no_site_startup():
    import infinity_grid.v05_c6_recovery as r
    src = inspect.getsource(r._spawn_child)
    assert '"-S"' in src
    assert '/opt/python-hooks' not in src


def test_c6_v3_lease_binds_exact_running_supervisor_identity(tmp_path, monkeypatch):
    import hashlib
    import json
    import infinity_grid.v05_c6_recovery as r
    from infinity_grid.v05_process_identity import current_process_identity
    entry = tmp_path / "v05_c6_recovery_entrypoint.py"
    module = tmp_path / "v05_c6_recovery.py"
    entry.write_text("entry\n", encoding="utf-8")
    module.write_text("module\n", encoding="utf-8")
    secret = r._write_lease(
        tmp_path / "runtime", "a" * 64, "b" * 64,
        supervisor_entrypoint=entry, supervisor_module=module, service_root=tmp_path,
    )
    lease = json.loads((tmp_path / "runtime/supervisor/lease.json").read_text())
    identity = current_process_identity()
    assert lease["schema_id"] == "IG_DECODER_C6_SUPERVISOR_LEASE_V3"
    assert set(lease) == {
        "schema_id", "supervisor_pid", "supervisor_proc_pid", "supervisor_nspid",
        "supervisor_start_time_ticks", "secret_sha256", "accepted_source_sha256",
        "capsule_sha256", "supervisor_entrypoint_sha256", "supervisor_module_sha256",
        "service_root_sha256",
    }
    assert lease["supervisor_pid"] == identity["local_pid"]
    assert lease["supervisor_proc_pid"] == identity["proc_pid"]
    assert lease["supervisor_nspid"] == list(identity["nspid"])
    assert lease["supervisor_start_time_ticks"] == identity["start_time_ticks"]
    assert lease["secret_sha256"] == hashlib.sha256(secret.encode("ascii")).hexdigest()
    assert lease["supervisor_entrypoint_sha256"] == hashlib.sha256(entry.read_bytes()).hexdigest()
    assert lease["supervisor_module_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert lease["service_root_sha256"] == hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()


def test_c6_resource_declares_v3_namespace_identity_binding():
    from importlib import resources
    obj = json.loads(resources.files("infinity_grid").joinpath("resources/v05/C6_PERSISTENT_RECOVERY_V1.json").read_text())
    assert obj["recovery_capsule_schema"] == "IG_DECODER_C6_RECOVERY_CAPSULE_V2"
    assert obj["supervisor_lease_schema"] == "IG_DECODER_C6_SUPERVISOR_LEASE_V3"
    assert obj["origin_guard_requires_v3_supervisor_identity"] is True
    assert obj["namespace_identity_binding"] == "LOCAL_PPID_PLUS_PROCFS_PPID_NSPID_AND_STARTTIME"



def test_c6_origin_guard_rejects_fake_wrapper_supervisor_lease(tmp_path):
    """Regression: the old guard accepted any parent PID + caller-created lease/secret."""
    import hashlib
    import json
    import os
    from infinity_grid.v05_execution_authority import ExecutionAuthorityError
    from infinity_grid.v05_origin_guard import _supervisor_controller_event_scope
    secret = "a" * 64
    fake = {
        "schema_id": "IG_DECODER_C6_SUPERVISOR_LEASE_V3",
        "supervisor_pid": os.getppid(),
        "supervisor_proc_pid": 123456,
        "supervisor_nspid": [123456, os.getppid()],
        "supervisor_start_time_ticks": 1,
        "secret_sha256": hashlib.sha256(secret.encode("ascii")).hexdigest(),
        "accepted_source_sha256": "b" * 64,
        "capsule_sha256": "c" * 64,
        "supervisor_entrypoint_sha256": "d" * 64,
        "supervisor_module_sha256": "e" * 64,
        "service_root_sha256": "f" * 64,
    }
    lease = tmp_path / "lease.json"
    lease.write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(ExecutionAuthorityError, match="REJECT_EXTERNAL_EXECUTION_ORIGIN"):
        with _supervisor_controller_event_scope(
            supervisor_pid=os.getppid(), supervisor_secret=secret, lease_path=lease,
            expected_source_sha256="b" * 64,
        ):
            raise AssertionError("fake wrapper supervisor minted root context")


def test_c6_origin_guard_requires_expected_source_binding():
    import inspect
    from infinity_grid.v05_origin_guard import _supervisor_controller_event_scope
    assert "expected_source_sha256" in inspect.signature(_supervisor_controller_event_scope).parameters
    assert "attempt_binding" in inspect.signature(_supervisor_controller_event_scope).parameters


def test_c6_supervisor_scope_passes_recorded_attempt_binding_to_context_mint():
    import inspect
    from infinity_grid import v05_origin_guard as guard
    source = inspect.getsource(guard._supervisor_controller_event_scope)
    assert "_binding=attempt_binding" in source


def test_c6_context_probe_is_passive_registered_and_direct_call_rejected(tmp_path):
    import infinity_grid.v05_controller_event_loop as loop
    reg = loop._registry(Path(loop.__file__).resolve().parent.parent)
    assert reg[loop.C6_RECOVERY_JOB]["allowed_operations"] == ["C6_CONTEXT_PROBE"]
    with pytest.raises(Exception, match="REJECT_EXTERNAL_EXECUTION_ORIGIN"):
        loop._handle_c6_context_probe(
            {"internal_execution_id":"intent-test"}, tmp_path,
            Path(loop.__file__).resolve().parent.parent, supervisor_pid=os.getppid(),
        )


def test_c6_proc_cmdline_parser_splits_nul_fields():
    from infinity_grid.v05_origin_guard import _c6_read_proc_cmdline
    from infinity_grid.v05_process_identity import current_process_identity
    cmd = _c6_read_proc_cmdline(current_process_identity()["proc_pid"])
    assert len(cmd) >= 1
    assert all("\x00" not in field for field in cmd)


def test_c6_controller_child_uses_separate_session_and_group_cleanup():
    import inspect
    import infinity_grid.v05_c6_recovery as r
    spawn=inspect.getsource(r._spawn_child)
    supervisor=inspect.getsource(r.supervisor_main)
    assert 'start_new_session=True' in spawn
    assert '_terminate_controller_process_group' in supervisor
    assert 'signal.SIGKILL' in supervisor



def test_c6_source_activation_stages_fixed_install_before_restart(monkeypatch, tmp_path):
    import hashlib
    import json
    import infinity_grid.v05_c6_recovery as r
    monkeypatch.setattr(r, "require_controller_execution_origin" if hasattr(r,"require_controller_execution_origin") else "_unused", lambda _label: None, raising=False)
    # Patch the local import target used by the helper.
    import infinity_grid.v05_origin_guard as guard
    monkeypatch.setattr(guard, "require_controller_execution_origin", lambda _label: None)
    runtime = tmp_path / "runtime"; runtime.mkdir()
    service = tmp_path
    candidate = runtime / "source_transitions" / "intent-x" / "candidate_source"
    (candidate / "infinity_grid").mkdir(parents=True)
    (candidate / "infinity_grid" / "x.py").write_text("x=1\n", encoding="utf-8")
    source_sha = "a" * 64; package_sha = "b" * 64
    monkeypatch.setattr(r, "engineering_source_tree_digest", lambda _p: source_sha)
    monkeypatch.setattr(r, "source_tree_digest", lambda _p: package_sha)
    trroot = candidate.parent
    import zipfile
    with zipfile.ZipFile(trroot / "source.zip", "w") as z: z.writestr("x", b"1")
    manifest = {"schema_id":"IG_DECODER_SOURCE_MANIFEST_V1","source_sha256":source_sha,
                "files":[{"path":"infinity_grid/x.py","sha256":hashlib.sha256((candidate/"infinity_grid/x.py").read_bytes()).hexdigest(),"size_bytes":4}]}
    (trroot / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Existing fixed capsule supplies only the generation binding for staging.
    (service / "RECOVERY_CAPSULE.json").write_text(json.dumps(_capsule(generation=7)), encoding="utf-8")
    acc={"status":"PASS","final_origin_exclusivity":True,"final_source_sha256":source_sha}
    accp=runtime/"final/C5_FINAL_ACCEPTANCE.json"; accp.parent.mkdir(); accp.write_text(json.dumps(acc),encoding="utf-8")
    transition={"status":"PASS","candidate_source_path":str(candidate),"candidate_source_sha256":source_sha,
                "candidate_package_sha256":package_sha,"source_zip_sha256":hashlib.sha256((trroot/"source.zip").read_bytes()).hexdigest()}
    receipt=r.stage_fixed_installation_after_acceptance(runtime_root=runtime,transition=transition,acceptance_path=accp)
    assert receipt["from_generation"]==7 and receipt["to_generation"]==8
    assert receipt["accepted_source_sha256"]==source_sha
    assert (runtime/"final/STAGED_FIXED_INSTALL.json").is_file()


def test_c6_restart_branch_persists_then_execs_canonical_supervisor():
    import inspect
    import infinity_grid.v05_c6_recovery as r
    src=inspect.getsource(r.supervisor_main)
    assert "_load_staged_install(runtime)" in src
    assert "_persist_staged_fixed_installation(info, staged)" in src
    assert "_exec_persisted_supervisor(entrypoint)" in src
    assert 'source = _verified_active_source(runtime)' in src


@pytest.mark.parametrize("blocked_errno", [errno.EXDEV, errno.EACCES, errno.EPERM])
def test_c6_fixed_install_archive_falls_back_to_verified_same_parent_rollback(
    monkeypatch, tmp_path, blocked_errno,
):
    import infinity_grid.v05_c6_recovery as r
    service = tmp_path / "service"
    history = service / "runtime" / "recovery_history" / "generation-0007-test"
    old_source = service / "accepted_source"
    (old_source / "infinity_grid").mkdir(parents=True)
    (old_source / "infinity_grid" / "x.py").write_text("x=1\n", encoding="utf-8")
    history.mkdir(parents=True)
    source_sha = r.engineering_source_tree_digest(old_source)
    package_sha = r.source_tree_digest(old_source / "infinity_grid")
    manifest = {
        "schema_id": "IG_DECODER_SOURCE_MANIFEST_V1",
        "source_sha256": source_sha,
        "files": [{
            "path": "infinity_grid/x.py",
            "sha256": hashlib.sha256(
                (old_source / "infinity_grid" / "x.py").read_bytes()
            ).hexdigest(),
            "size_bytes": 4,
        }],
    }
    (history / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    real_replace = r.os.replace

    def executor_replace(source, destination):
        if Path(source) == old_source and Path(destination) == history / "accepted_source":
            raise OSError(blocked_errno, "managed executor cross-directory rename denied")
        return real_replace(source, destination)

    monkeypatch.setattr(r.os, "replace", executor_replace)
    archived, rollback = r._archive_current_source_for_fixed_install(
        service=service,
        history=history,
        old_source=old_source,
        current={
            "generation": 7,
            "accepted_source_sha256": source_sha,
            "accepted_package_sha256": package_sha,
        },
    )
    assert archived == history / "accepted_source"
    assert rollback == service / (
        ".accepted_source.rollback-generation-0007-" + source_sha[:16]
    )
    assert not old_source.exists()
    assert r.engineering_source_tree_digest(archived) == source_sha
    assert r.engineering_source_tree_digest(rollback) == source_sha
    assert r.source_tree_digest(archived / "infinity_grid") == package_sha
    assert r.source_tree_digest(rollback / "infinity_grid") == package_sha


def test_c6_fixed_install_archive_does_not_mask_unexpected_rename_error(monkeypatch, tmp_path):
    import infinity_grid.v05_c6_recovery as r
    service = tmp_path / "service"
    history = service / "runtime" / "recovery_history" / "generation-0007-test"
    old_source = service / "accepted_source"
    (old_source / "infinity_grid").mkdir(parents=True)
    (old_source / "infinity_grid" / "x.py").write_text("x=1\n", encoding="utf-8")
    history.mkdir(parents=True)
    source_sha = r.engineering_source_tree_digest(old_source)
    package_sha = r.source_tree_digest(old_source / "infinity_grid")

    def unexpected_error(_source, _destination):
        raise OSError(errno.EIO, "unexpected storage failure")

    # Restore shared os.replace before the native recorder writes the call report.
    with monkeypatch.context() as scoped:
        scoped.setattr(r.os, "replace", unexpected_error)
        with pytest.raises(OSError) as caught:
            r._archive_current_source_for_fixed_install(
                service=service,
                history=history,
                old_source=old_source,
                current={
                    "generation": 7,
                    "accepted_source_sha256": source_sha,
                    "accepted_package_sha256": package_sha,
                },
            )
    assert caught.value.errno == errno.EIO
    assert old_source.is_dir()
    assert not (history / "accepted_source").exists()


def test_c6_restore_runtime_bindings_atomically_replaces_read_only_pointers(tmp_path):
    import infinity_grid.v05_c6_recovery as r
    source = tmp_path / "accepted_source"
    source.mkdir()
    c5 = tmp_path / "C5_FINAL_ACCEPTANCE.json"
    c5.write_text("{}\n", encoding="utf-8")
    final = tmp_path / "runtime" / "final"
    final.mkdir(parents=True)
    path_pointer = final / "ACTIVE_SOURCE_PATH.txt"
    sha_pointer = final / "ACTIVE_SOURCE_SHA256.txt"
    path_pointer.write_text("old\n", encoding="utf-8")
    sha_pointer.write_text("0" * 64 + "\n", encoding="ascii")
    path_pointer.chmod(0o444)
    sha_pointer.chmod(0o444)

    r._restore_runtime_bindings({
        "paths": {
            "runtime": tmp_path / "runtime",
            "source": source,
            "c5_acceptance": c5,
        },
        "source_sha256": "a" * 64,
    })

    assert path_pointer.read_text(encoding="utf-8") == str(source) + "\n"
    assert sha_pointer.read_text(encoding="ascii") == "a" * 64 + "\n"
    assert path_pointer.stat().st_mode & 0o777 == 0o444
    assert sha_pointer.stat().st_mode & 0o777 == 0o444


def test_controller_source_activation_stages_persistent_fixed_install():
    import inspect
    import infinity_grid.v05_controller_event_loop as loop
    src=inspect.getsource(loop._handle_source_change)
    assert "stage_fixed_installation_after_acceptance" in src
    assert "fixed_install_staging" in src


def test_controller_source_activation_atomically_replaces_runtime_pointers():
    import inspect
    import infinity_grid.v05_controller_event_loop as loop
    src = inspect.getsource(loop._handle_source_change)
    assert "_atomic_bytes(final/'ACTIVE_SOURCE_PATH.txt'" in src
    assert "_atomic_bytes(final/'ACTIVE_SOURCE_SHA256.txt'" in src
