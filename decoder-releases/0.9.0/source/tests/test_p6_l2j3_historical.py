from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.replay_l2j3_executor import L2J3_NODE_IDS, NODE_EXECUTOR_BINDINGS
from infinity_grid.replay_p5_frontier_import import P5_STATE_OBJECT_SHA256
from infinity_grid.replay_reference_data import empty_manifest
from infinity_grid.v05_stage_runtime import ReplayRootPaused, StageRuntimeError, StageScienceRuntime


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "infinity_grid/resources/replay"
MANIFEST = RES / "L0_UPWARD_SRCF_ASSERTION_DAG_V9.json"
CATALOGUE = RES / "L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json"
FRONTIER = RES / "P5_CAPTURED_STATE_OBJECT_V1.zip"


def parameters(injection="NONE"):
    return {
        "operation": "EXECUTE_L2J3_P6",
        "manifest_input": "compiled_replay_manifest",
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_dag_sha256": json.loads(MANIFEST.read_text())["dag_sha256"],
        "reference_catalogue_input": "replay_reference_catalogue",
        "reference_catalogue_file_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
        "p5_frontier_input": "p5_captured_state_object",
        "p5_frontier_file_sha256": P5_STATE_OBJECT_SHA256,
        "dataset_root_sha256": empty_manifest()["manifest_sha256"],
        "root_run_id": "L0-G8-REPLAY-ROOT-V1",
        "authorized_through_layer": "G8", "next_wait_layer": "GLOBAL",
        "node_executor_bindings": NODE_EXECUTOR_BINDINGS,
        "pilot_fault_injection": injection,
    }


def artifacts(frontier=FRONTIER):
    return {"compiled_replay_manifest": str(MANIFEST),
            "replay_reference_catalogue": str(CATALOGUE),
            "p5_captured_state_object": str(frontier)}


def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)
    candidate = object.__new__(StageScienceRuntime)
    candidate._root = tmp_path / "registered-output"
    candidate._root.mkdir()
    candidate._require_execution = lambda: None
    candidate.publish_json = lambda name, obj: {"logical_name": name,
                                                 "sha256": canonical_sha256(obj)}
    return candidate


def test_p6_exact_frontier_comparison_and_cold_resume(tmp_path, monkeypatch):
    candidate = runtime(tmp_path, monkeypatch)
    result = candidate.run_registered_replay_root(parameters(), artifacts())
    assert result["outcome"] == "L2J3_HISTORICAL_PILOT_COMPLETE"
    assert result["historical_evidence_nodes_executed"] == 4
    assert result["science_executed"] is False
    assert result["runner_action"]["node_id"] == "IG/NODE_IN/S/ASSERTION_MAPPED_STRUCTURE"
    assert all(node in result["completed_node_ids"] for node in L2J3_NODE_IDS)
    assert "IG/L2J3/GATE/CERTIFY" in result["completed_node_ids"]
    state_path = candidate._root / "replay_runner/runner_state.json"
    original = state_path.read_bytes()
    resumed = candidate.run_registered_replay_root(parameters(), artifacts())
    assert resumed["runner_state_sha256"] == result["runner_state_sha256"]
    assert state_path.read_bytes() == original


def test_p6_mismatch_stops_on_first_node(tmp_path, monkeypatch):
    candidate = runtime(tmp_path, monkeypatch)
    with pytest.raises(ReplayRootPaused):
        candidate.run_registered_replay_root(parameters(L2J3_NODE_IDS[0]), artifacts())
    state = json.loads((candidate._root / "replay_runner/runner_state.json").read_text())
    assert len(state["completed_node_ids"]) == 10
    assert state["active_audit_capsule_sha256"] is not None


def test_p6_rejects_binding_and_frontier_tamper(tmp_path, monkeypatch):
    candidate = runtime(tmp_path, monkeypatch)
    params = parameters()
    params["node_executor_bindings"] = params["node_executor_bindings"][:-1]
    with pytest.raises(StageRuntimeError, match="P6_L2J3_EXECUTOR_BINDINGS"):
        candidate.run_registered_replay_root(params, artifacts())
    bad = tmp_path / "tampered.zip"
    bad.write_bytes(FRONTIER.read_bytes() + b"tamper")
    with pytest.raises(StageRuntimeError, match="P6_FRONTIER_FILE_HASH"):
        candidate.run_registered_replay_root(parameters(), artifacts(bad))
