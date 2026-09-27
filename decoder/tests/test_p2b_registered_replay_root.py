from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infinity_grid import __version__
from infinity_grid.canon import canonical_sha256
from infinity_grid.replay_dag_runner import ReplayDagRunnerError
from infinity_grid.replay_root_job import HANDLER_REF, replay_root_job_handler
from infinity_grid.v05_execution_authority import ExecutionAuthorityError
from infinity_grid.v05_stage_architecture import audit_module_source
from infinity_grid.v05_stage_runtime import ReplayRootPaused, StageRuntimeError, StageScienceRuntime


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json"
PROFILE = ROOT / "infinity_grid/resources/replay/P2B_REPLAY_ROOT_REGISTRATION_PROFILE_V1.json"
QUALIFICATION = ROOT / "infinity_grid/resources/replay/P2B_REGISTERED_REPLAY_ROOT_QUALIFICATION_V1.json"
ZERO = "0" * 64


def parameters():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "operation": "SCHEDULE_ONLY_P2B",
        "manifest_input": "compiled_replay_manifest",
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_dag_sha256": manifest["dag_sha256"],
        "dataset_root_sha256": ZERO,
        "root_run_id": "P2B-QUALIFICATION-ROOT",
        "authorized_through_layer": "G8",
        "next_wait_layer": "GLOBAL",
        "node_executor_bindings": [],
    }


def fake_runtime(tmp_path):
    runtime = object.__new__(StageScienceRuntime)
    runtime._root = tmp_path / "registered-output"
    runtime._root.mkdir()
    runtime._require_execution = lambda: None
    publications = []
    runtime.publish_json = lambda name, obj: publications.append((name, obj)) or {
        "logical_name": name, "path": str(runtime._root / f"{name}.json"),
        "sha256": canonical_sha256(obj), "size_bytes": 1,
    }
    return runtime, publications


def test_dev50_registers_exactly_one_controller_owned_replay_root_handler():
    from infinity_grid.v05_controller_event_loop import _CONTROLLER_PROCESS_HANDLERS
    # The current identity is tested centrally; this record keeps its historical pin.
    assert HANDLER_REF in _CONTROLLER_PROCESS_HANDLERS
    assert len([ref for ref in _CONTROLLER_PROCESS_HANDLERS if "replay_root" in ref]) == 1
    assert audit_module_source(ROOT / "infinity_grid/replay_root_job.py")["status"] == "PASS"


def test_handler_cannot_be_called_outside_recorded_controller_origin():
    stage = {"stage_id": "REPLAY:L0_TO_G8:ROOT", "execution": {"parameters": parameters()},
             "input_artifacts": {"compiled_replay_manifest": str(MANIFEST)}}
    with pytest.raises(ExecutionAuthorityError):
        replay_root_job_handler(stage, SimpleNamespace())


def test_runtime_initializes_one_empty_root_and_pauses_before_l0_science(tmp_path, monkeypatch):
    runtime, publications = fake_runtime(tmp_path)
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)
    with pytest.raises(ReplayRootPaused, match="WAITING_FOR_NODE_EXECUTOR_BINDING"):
        runtime.run_registered_replay_root(
            parameters(), {"compiled_replay_manifest": str(MANIFEST)})
    result = publications[0][1]
    assert result["outcome"] == "WAITING_FOR_NODE_EXECUTOR_BINDING"
    assert result["runner_action"]["action"] == "EXECUTE_SCIENTIFIC_NODE"
    assert result["runner_action"]["node_id"].startswith("IG/L0/S/")
    assert result["completed_node_ids"] == [] and result["science_executed"] is False
    assert publications[0][0] == "REPLAY_ROOT_STATUS"
    state = json.loads((runtime._root / "replay_runner/runner_state.json").read_text())
    assert state["dataset_root"] == {"state": "EMPTY", "sha256": ZERO}


def test_runtime_resumes_same_root_state_without_reinitializing(tmp_path, monkeypatch):
    runtime, publications = fake_runtime(tmp_path)
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)
    with pytest.raises(ReplayRootPaused):
        runtime.run_registered_replay_root(parameters(), {"compiled_replay_manifest": str(MANIFEST)})
    first = publications[-1][1]
    state_path = runtime._root / "replay_runner/runner_state.json"
    before = state_path.read_bytes()
    with pytest.raises(ReplayRootPaused):
        runtime.run_registered_replay_root(parameters(), {"compiled_replay_manifest": str(MANIFEST)})
    second = publications[-1][1]
    assert first["runner_action"] == second["runner_action"]
    assert state_path.read_bytes() == before


@pytest.mark.parametrize("field,value,error", [
    ("operation", "EXECUTE_ALL", "REPLAY_ROOT_OPERATION_NOT_REGISTERED"),
    ("authorized_through_layer", "GLOBAL", "REPLAY_ROOT_AUTHORITY_BOUNDARY"),
    ("node_executor_bindings", ["unsafe"], "P2B_NODE_EXECUTORS_MUST_REMAIN_UNBOUND"),
    ("manifest_file_sha256", ZERO, "REPLAY_ROOT_MANIFEST_FILE_HASH"),
])
def test_registration_parameter_tampering_fails_closed(tmp_path, monkeypatch, field, value, error):
    runtime, _ = fake_runtime(tmp_path)
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)
    params = parameters(); params[field] = value
    with pytest.raises(StageRuntimeError, match=error):
        runtime.run_registered_replay_root(params, {"compiled_replay_manifest": str(MANIFEST)})
    assert not (runtime._root / "replay_runner/runner_state.json").exists()


def test_existing_runner_tampering_fails_closed_on_registered_resume(tmp_path, monkeypatch):
    runtime, _ = fake_runtime(tmp_path)
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)
    with pytest.raises(ReplayRootPaused):
        runtime.run_registered_replay_root(parameters(), {"compiled_replay_manifest": str(MANIFEST)})
    state_path = runtime._root / "replay_runner/runner_state.json"
    state = json.loads(state_path.read_text()); state["root_run_id"] = "TAMPERED"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ReplayDagRunnerError, match="state hash mismatch"):
        runtime.run_registered_replay_root(parameters(), {"compiled_replay_manifest": str(MANIFEST)})


def test_registration_profile_is_one_stage_job_and_freezes_p2b_nonclaims():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["job_id"] == "DECODER.REPLAY.L0_TO_G8.ROOT"
    assert profile["execution"]["handler_ref"] == HANDLER_REF
    assert profile["execution"]["evaluator_refs"] == []
    assert profile["execution"]["parameters"] == parameters() | {"root_run_id": "L0-G8-REPLAY-ROOT-V1"}
    assert profile["question"]["outcomes"] == [
        "WAITING_FOR_NODE_EXECUTOR_BINDING", "WAITING_FOR_EXTERNAL_AUDIT", "COMPLETE"]
    assert profile["p2b_boundary"] == {
        "node_executors_bound": False, "science_executed": False,
        "full_r02_complete": False, "global_authorized": False,
    }


def test_profile_materializes_as_a_valid_registered_workspace_job(tmp_path, monkeypatch):
    from infinity_grid import v05_controller_event_loop as loop
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    source_sha = "a" * 64
    job = {
        "schema_id": loop.JOB_SCHEMA,
        "job_id": profile["job_id"],
        "source_sha256": source_sha,
        "question": profile["question"],
        "input_artifacts": [{
            "logical_name": profile["required_input"]["logical_name"],
            "sha256": profile["required_input"]["sha256"],
        }],
        "execution": profile["execution"],
        "resources": profile["resources"],
    }
    job["registration_sha256"] = canonical_sha256(job)
    (tmp_path / "registry").mkdir()
    (tmp_path / "runtime").mkdir()
    from infinity_grid.canon import write_json_atomic
    write_json_atomic(tmp_path / "registry" / f"{job['job_id']}.json", job)
    monkeypatch.setattr(loop, "_workspace_ids", lambda *a, **k: (ROOT, source_sha, "b" * 64))
    monkeypatch.setattr(loop, "_artifact", lambda *a, **k: MANIFEST)
    admitted = loop.validate_workspace_job(tmp_path, job["job_id"])
    assert admitted["job"]["registration_sha256"] == job["registration_sha256"]
    assert admitted["artifacts"] == {"compiled_replay_manifest": MANIFEST}


def test_machine_qualification_pins_controller_integration_and_nonclaims():
    # This is immutable dev50 evidence, not a qualification of current bytes.
    # Current controller/runner behavior is tested above and in the native gate.
    assert hashlib.sha256(QUALIFICATION.read_bytes()).hexdigest() == "743203958e5e7ba0ce3cdb6366a04b6bc11e85abb6783b29d8e71f506f963c09"
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    assert record["decoder_version"] == "0.8.0.dev50+lib"
    assert record["status"] == "PASS_P2B_REGISTERED_ROOT_INTEGRATION"
    assert record["p2_complete"] is True
    assert record["science_executed"] is False
    assert record["next_step"] == "P3_UNIFIED_REFERENCE_DATA_CONTRACT"
    assert record["r02_boundary"]["full_r02_complete"] is False
    assert record["authority_boundary"] == {
        "authorized_replay_through_layer": "G8", "next_wait_layer": "GLOBAL",
        "global_authorized": False,
    }
