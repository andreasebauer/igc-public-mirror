from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.replay_c0_historical_executor import C0_NODE_IDS, NODE_EXECUTOR_BINDINGS
from infinity_grid.replay_frontier_import import P4_COMPLETED_NODE_IDS, P4_STATE_OBJECT_SHA256
from infinity_grid.replay_reference_data import ReplayReferenceDataStore, empty_manifest
from infinity_grid.v05_stage_architecture import audit_module_source
from infinity_grid.v05_stage_runtime import ReplayRootPaused, StageRuntimeError, StageScienceRuntime


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json"
CATALOGUE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json"
FRONTIER = ROOT / "infinity_grid/resources/replay/P4_CAPTURED_STATE_OBJECT_84B659_V1.zip"
PROFILE = ROOT / "infinity_grid/resources/replay/P5_C0_HISTORICAL_REGISTRATION_PROFILE_V1.json"


def parameters(*, injection="NONE"):
    manifest = json.loads(MANIFEST.read_text())
    return {
        "operation": "EXECUTE_C0_HISTORICAL_P5",
        "manifest_input": "compiled_replay_manifest",
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_dag_sha256": manifest["dag_sha256"],
        "reference_catalogue_input": "replay_reference_catalogue",
        "reference_catalogue_file_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
        "p4_frontier_input": "p4_captured_state_object",
        "p4_frontier_file_sha256": hashlib.sha256(FRONTIER.read_bytes()).hexdigest(),
        "dataset_root_sha256": empty_manifest()["manifest_sha256"],
        "root_run_id": "L0-G8-REPLAY-ROOT-V1",
        "authorized_through_layer": "G8", "next_wait_layer": "GLOBAL",
        "node_executor_bindings": NODE_EXECUTOR_BINDINGS,
        "pilot_fault_injection": injection,
    }


def artifacts():
    return {"compiled_replay_manifest": str(MANIFEST),
            "replay_reference_catalogue": str(CATALOGUE),
            "p4_captured_state_object": str(FRONTIER)}


def fake_runtime(tmp_path):
    runtime = object.__new__(StageScienceRuntime)
    runtime._root = tmp_path / "registered-output"; runtime._root.mkdir(parents=True)
    runtime._require_execution = lambda: None
    publications = []
    runtime.publish_json = lambda name, obj: publications.append((name, obj)) or {
        "logical_name": name, "path": str(runtime._root / f"{name}.json"),
        "sha256": canonical_sha256(obj), "size_bytes": 1}
    return runtime, publications


def permit_outputs(monkeypatch):
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)


def test_binds_only_c0_historical_and_imports_exact_captured_p4_frontier():
    assert hashlib.sha256(FRONTIER.read_bytes()).hexdigest() == P4_STATE_OBJECT_SHA256
    assert tuple(row["node_id"] for row in NODE_EXECUTOR_BINDINGS) == C0_NODE_IDS
    assert P4_COMPLETED_NODE_IDS[-1] == "IG/L0/GATE/CERTIFY"
    assert audit_module_source(ROOT / "infinity_grid/replay_c0_historical_executor.py")["status"] == "PASS"
    assert audit_module_source(ROOT / "infinity_grid/replay_frontier_import.py")["status"] == "PASS"


def test_p5_continues_from_p4_runs_c0_and_stops_before_l2j3(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, publications = fake_runtime(tmp_path)
    result = runtime.run_registered_replay_root(parameters(), artifacts())
    assert result["outcome"] == "C0_HISTORICAL_PILOT_COMPLETE"
    assert result["historical_evidence_nodes_executed"] == 4
    assert result["fresh_empty_root_science_node_count"] == 0
    assert result["science_executed"] is False
    assert result["runner_action"]["contract"]["layer"] == "L2J3"
    assert tuple(result["completed_node_ids"][:5]) == P4_COMPLETED_NODE_IDS
    assert all(node_id in result["completed_node_ids"] for node_id in C0_NODE_IDS)
    assert "IG/C0_HISTORICAL/GATE/CERTIFY" in result["completed_node_ids"]
    store = ReplayReferenceDataStore(runtime._root / "replay_reference_data")
    assert len(store.manifest["record_ids"]) == 40
    assert publications[-1][1]["outcome"] == "C0_HISTORICAL_PILOT_COMPLETE"


def test_p5_cold_restore_is_byte_stable(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, _ = fake_runtime(tmp_path)
    first = runtime.run_registered_replay_root(parameters(), artifacts())
    state_before = (runtime._root / "replay_runner/runner_state.json").read_bytes()
    manifest_before = (runtime._root / "replay_reference_data/MANIFEST.json").read_bytes()
    second = runtime.run_registered_replay_root(parameters(), artifacts())
    assert second["runner_state_sha256"] == first["runner_state_sha256"]
    assert (runtime._root / "replay_runner/runner_state.json").read_bytes() == state_before
    assert (runtime._root / "replay_reference_data/MANIFEST.json").read_bytes() == manifest_before


def test_p5_mismatch_stops_at_first_c0_failure(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, publications = fake_runtime(tmp_path)
    with pytest.raises(ReplayRootPaused, match="WAITING_FOR_EXTERNAL_AUDIT"):
        runtime.run_registered_replay_root(parameters(injection=C0_NODE_IDS[1]), artifacts())
    status = publications[-1][1]
    assert status["runner_action"]["audit_capsule"]["stopped_node_id"] == C0_NODE_IDS[1]
    assert status["historical_evidence_nodes_executed"] == 1
    assert status["fresh_empty_root_science_node_count"] == 0


def test_p5_rejects_tampered_frontier_or_binding_before_execution(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, _ = fake_runtime(tmp_path)
    bad = parameters(); bad["node_executor_bindings"] = bad["node_executor_bindings"][:-1]
    with pytest.raises(StageRuntimeError, match="P5_C0_EXECUTOR_BINDINGS"):
        runtime.run_registered_replay_root(bad, artifacts())
    corrupt = tmp_path / "corrupt.zip"; corrupt.write_bytes(FRONTIER.read_bytes() + b"x")
    runtime2, _ = fake_runtime(tmp_path / "second")
    with pytest.raises(StageRuntimeError, match="P5_FRONTIER_IMPORT_FAILED"):
        runtime2.run_registered_replay_root(parameters(), artifacts() | {"p4_captured_state_object": str(corrupt)})


def test_p5_profile_pins_inputs_and_honest_boundary():
    profile = json.loads(PROFILE.read_text())
    assert profile["execution"]["parameters"] == parameters()
    assert profile["p5_boundary"]["fresh_science_executed"] is False
    assert profile["p5_boundary"]["execution_classes"] == ["SOURCE_INTEGRITY_ONLY", "HISTORICAL_RESULT_ONLY"]
    for row in profile["required_inputs"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
