from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from infinity_grid import __version__
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.replay_dag_runner import ReplayDagRunner
from infinity_grid.replay_l0_executor import (
    L0_NODE_IDS, NODE_EXECUTOR_BINDINGS, L0ReplayExecutorError,
    execute_l0_source_integrity_node,
)
from infinity_grid.replay_reference_data import ReplayReferenceDataStore, empty_manifest
from infinity_grid.v05_stage_runtime import ReplayRootPaused, StageRuntimeError, StageScienceRuntime
from infinity_grid.v05_stage_architecture import audit_module_source


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json"
CATALOGUE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json"
PROFILE = ROOT / "infinity_grid/resources/replay/P4_L0_PILOT_REGISTRATION_PROFILE_V1.json"
HARNESS_RESULT = ROOT / "infinity_grid/resources/replay/P4_L0_PILOT_RUNTIME_HARNESS_RESULT_V1.json"
QUALIFICATION = ROOT / "infinity_grid/resources/replay/P4_L0_PILOT_QUALIFICATION_V1.json"


def parameters(*, injection="NONE"):
    manifest = json.loads(MANIFEST.read_text())
    return {
        "operation": "EXECUTE_L0_PILOT_P4",
        "manifest_input": "compiled_replay_manifest",
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_dag_sha256": manifest["dag_sha256"],
        "reference_catalogue_input": "replay_reference_catalogue",
        "reference_catalogue_file_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
        "dataset_root_sha256": empty_manifest()["manifest_sha256"],
        "root_run_id": "P4-L0-PILOT-ROOT",
        "authorized_through_layer": "G8",
        "next_wait_layer": "GLOBAL",
        "node_executor_bindings": NODE_EXECUTOR_BINDINGS,
        "pilot_fault_injection": injection,
    }


def artifacts():
    return {
        "compiled_replay_manifest": str(MANIFEST),
        "replay_reference_catalogue": str(CATALOGUE),
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


def permit_outputs(monkeypatch):
    monkeypatch.setattr("infinity_grid.v05_origin_guard.require_registered_output", lambda *a, **k: None)


def test_dev52_binds_exactly_the_four_frozen_l0_nodes():
    # The current identity is tested centrally; this record keeps its historical pin.
    assert tuple(row["node_id"] for row in NODE_EXECUTOR_BINDINGS) == L0_NODE_IDS
    assert len({row["handler_ref"] for row in NODE_EXECUTOR_BINDINGS}) == 1
    assert audit_module_source(ROOT / "infinity_grid/replay_l0_executor.py")["status"] == "PASS"


def test_automatic_pilot_replays_all_l0_lanes_and_stops_before_c0(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, publications = fake_runtime(tmp_path)
    result = runtime.run_registered_replay_root(parameters(), artifacts())
    assert result["outcome"] == "L0_PILOT_COMPLETE"
    assert result["source_integrity_nodes_executed"] == 4
    assert result["fresh_empty_root_science_node_count"] == 0
    assert result["science_executed"] is False
    assert result["runner_action"]["node_id"].startswith("IG/C0_HISTORICAL/")
    assert all(node_id in result["completed_node_ids"] for node_id in L0_NODE_IDS)
    assert "IG/L0/GATE/CERTIFY" in result["completed_node_ids"]
    assert publications[-1][1]["outcome"] == "L0_PILOT_COMPLETE"
    store = ReplayReferenceDataStore(runtime._root / "replay_reference_data")
    assert len(store.manifest["record_ids"]) == 20
    assert store.manifest["science_executed"] is False


def test_cold_restore_reuses_exact_records_and_frontier(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, _ = fake_runtime(tmp_path)
    first = runtime.run_registered_replay_root(parameters(), artifacts())
    manifest_before = (runtime._root / "replay_reference_data/MANIFEST.json").read_bytes()
    state_before = (runtime._root / "replay_runner/runner_state.json").read_bytes()
    restored = runtime.run_registered_replay_root(parameters(), artifacts())
    assert restored["reference_manifest_sha256"] == first["reference_manifest_sha256"]
    assert (runtime._root / "replay_reference_data/MANIFEST.json").read_bytes() == manifest_before
    assert (runtime._root / "replay_runner/runner_state.json").read_bytes() == state_before


def test_injected_l0_mismatch_stops_at_first_failure_and_restores_cleanly(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, publications = fake_runtime(tmp_path)
    params = parameters(injection=L0_NODE_IDS[1])
    with pytest.raises(ReplayRootPaused, match="WAITING_FOR_EXTERNAL_AUDIT"):
        runtime.run_registered_replay_root(params, artifacts())
    status = publications[-1][1]
    assert status["runner_action"]["audit_capsule"]["stopped_node_id"] == L0_NODE_IDS[1]
    assert status["runner_action"]["audit_capsule"]["stop_outcome"] == "RESULT_MISMATCH"
    assert status["source_integrity_nodes_executed"] == 1
    state_path = runtime._root / "replay_runner/runner_state.json"
    manifest_path = runtime._root / "replay_reference_data/MANIFEST.json"
    before = (state_path.read_bytes(), manifest_path.read_bytes())
    with pytest.raises(ReplayRootPaused, match="WAITING_FOR_EXTERNAL_AUDIT"):
        runtime.run_registered_replay_root(params, artifacts())
    assert (state_path.read_bytes(), manifest_path.read_bytes()) == before


def test_executor_regenerates_each_lane_from_pinned_source_and_reuses_shared_input(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    catalogue = json.loads(CATALOGUE.read_text())
    runner = ReplayDagRunner.create(
        manifest, tmp_path / "runner", root_run_id="P4-DIRECT",
        dataset_root={"state": "EMPTY", "sha256": empty_manifest()["manifest_sha256"]})
    store = ReplayReferenceDataStore.initialize(tmp_path / "reference")
    for node_id in L0_NODE_IDS:
        action = runner.next_action()
        assert action["node_id"] == node_id
        result = execute_l0_source_integrity_node(
            node=runner.nodes[node_id], catalogue=catalogue, repository_root=ROOT,
            attempt=action["attempt"])
        for record in result.pop("reference_records"):
            store.put(record)
        assert result["comparison_outcome"] == "EXACT_HISTORICAL_REPLAY_AUTHORIZED"
        assert result["counts_toward_empty_root_science_replay"] is False
        runner.record_node_result(result)
    assert store.manifest["record_ids"].count("IGRD/L0/SOURCE/FORMAL_MONOGRAPH") == 1
    assert store.manifest["record_ids"].count("IGRD/L0/AUDIT/P1C2_V2") == 1


def test_source_or_binding_tampering_fails_closed_before_false_completion(tmp_path, monkeypatch):
    permit_outputs(monkeypatch)
    runtime, _ = fake_runtime(tmp_path)
    bad = parameters(); bad["node_executor_bindings"] = bad["node_executor_bindings"][:-1]
    with pytest.raises(StageRuntimeError, match="P4_L0_EXECUTOR_BINDINGS"):
        runtime.run_registered_replay_root(bad, artifacts())
    assert not (runtime._root / "replay_runner").exists()

    manifest = json.loads(MANIFEST.read_text())
    catalogue = json.loads(CATALOGUE.read_text())
    copied = tmp_path / "copy"
    source_ref = catalogue["historical_assertion_mappings"][0]["source_hashes"][0]["ref"]
    source = copied / source_ref
    source.parent.mkdir(parents=True)
    source.write_text("tampered", encoding="utf-8")
    store = ReplayReferenceDataStore.initialize(tmp_path / "tampered-store")
    node = next(row for row in manifest["nodes"] if row["canonical_id"] == L0_NODE_IDS[0])
    with pytest.raises(L0ReplayExecutorError, match="hash mismatch"):
        execute_l0_source_integrity_node(
            node=node, catalogue=catalogue, repository_root=copied, attempt=1)
    assert store.manifest == empty_manifest()


def test_registration_profile_freezes_the_honest_p4_boundary():
    profile = json.loads(PROFILE.read_text())
    assert profile["execution"]["parameters"] == parameters() | {"root_run_id": "L0-G8-REPLAY-ROOT-V1"}
    assert profile["p4_boundary"] == {
        "bound_layer": "L0", "execution_class": "SOURCE_INTEGRITY_ONLY",
        "mapped_obligation_count": 4, "counts_toward_empty_root_science_replay": False,
        "fresh_science_executed": False, "next_layer_executed": False,
        "global_authorized": False,
    }
    for row in profile["required_inputs"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_p4_profile_materializes_as_the_same_registered_root_job(tmp_path, monkeypatch):
    from infinity_grid import v05_controller_event_loop as loop
    profile = json.loads(PROFILE.read_text())
    source_sha = "a" * 64
    job = {
        "schema_id": loop.JOB_SCHEMA,
        "job_id": profile["job_id"],
        "source_sha256": source_sha,
        "question": profile["question"],
        "input_artifacts": [
            {"logical_name": row["logical_name"], "sha256": row["sha256"]}
            for row in profile["required_inputs"]
        ],
        "execution": profile["execution"],
        "resources": profile["resources"],
    }
    job["registration_sha256"] = canonical_sha256(job)
    (tmp_path / "registry").mkdir(); (tmp_path / "runtime").mkdir()
    write_json_atomic(tmp_path / "registry" / f"{job['job_id']}.json", job)
    by_hash = {
        profile["required_inputs"][0]["sha256"]: MANIFEST,
        profile["required_inputs"][1]["sha256"]: CATALOGUE,
    }
    monkeypatch.setattr(loop, "_workspace_ids", lambda *a, **k: (ROOT, source_sha, "b" * 64))
    monkeypatch.setattr(loop, "_artifact", lambda _root, digest: by_hash[digest])
    admitted = loop.validate_workspace_job(tmp_path, job["job_id"])
    assert admitted["job"]["registration_sha256"] == job["registration_sha256"]
    assert admitted["artifacts"] == {
        "compiled_replay_manifest": MANIFEST,
        "replay_reference_catalogue": CATALOGUE,
    }


def test_frozen_harness_result_is_self_hashed_and_not_a_controller_capture():
    result = json.loads(HARNESS_RESULT.read_text())
    assert result["result_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"})
    assert result["automatic"]["outcome"] == "L0_PILOT_COMPLETE"
    assert result["injected_stop"]["outcome"] == "RESULT_MISMATCH"
    assert result["execution_boundary"] == "RUNTIME_QUALIFICATION_HARNESS_NOT_RECORDED_CONTROLLER_ATTEMPT"


def test_machine_qualification_keeps_p4_open_until_recorded_controller_attempt():
    record = json.loads(QUALIFICATION.read_text())
    assert record["status"] == "PASS_IMPLEMENTATION_AND_HARNESS__CAPTURE_BLOCKED"
    assert record["p4_implementation_complete"] is True
    assert record["p4_plan_acceptance_complete"] is False
    assert record["recorded_controller_attempt_executed"] is False
    assert record["blocker"] == "CAPTURE_REQUIRED"
    # This is the immutable pre-capture dev52 qualification record.  Later
    # source versions must preserve its pins as historical identities rather
    # than pretending they still identify the current runtime/test bytes.
    assert len({row["ref"] for row in record["implementation_pins"]}) == len(record["implementation_pins"])
    assert all(len(row["sha256"]) == 64 for row in record["implementation_pins"])
