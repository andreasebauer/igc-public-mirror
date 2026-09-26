"""Registered fault-injection regressions for deep audit A01-A03; no science."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid import replay_reference_data as reference
from infinity_grid import replay_dag_runner as dag
from infinity_grid.replay_dag_runner import ReplayDagRunner, ReplayDagRunnerError, seal_external_decision, DECISION_SCHEMA
from test_p2a_replay_dag_runner import manifest, result
from test_p3_replay_reference_data import fixtures

ROOT = Path(__file__).parents[1]
OLD_HASHES = {
    "replay_dag_runner": "07d46f5c568eaa148f9a72706ede3d04283e570423a9769ae05854c430b19103",
    "replay_reference_data": "95282a32fdc4afe06c546ca6a3ad939ef67809152f092054247393c31cea22a8",
}


def old_module(name):
    path = ROOT / "tests/fixtures/dev80_audit_reproduction" / f"{name}.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OLD_HASHES[name]
    spec = importlib.util.spec_from_file_location(f"infinity_grid._audit_dev80_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner_at(root, cls=ReplayDagRunner):
    return cls.create(manifest(), root, root_run_id="AUDIT-STOP-RECOVERY",
                      dataset_root={"state": "EMPTY", "sha256": "0" * 64})


def stop(runner):
    return runner.record_node_result(result(runner.next_action(), "RESULT_MISMATCH"))["audit_capsule"]


def decision(capsule, name="REPEAT"):
    return seal_external_decision({"schema_id": DECISION_SCHEMA, "decision": name,
        "audit_capsule_sha256": capsule["audit_capsule_sha256"],
        "root_run_id": capsule["root_run_id"], "stopped_node_id": capsule["stopped_node_id"],
        "bound_result_sha256": capsule["node_result"]["result_sha256"],
        "bound_evidence_sha256": capsule["node_result"]["evidence_sha256"]})


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.json")}


def test_dev80_reproduces_hidden_stop(tmp_path):
    cls = old_module("replay_dag_runner").ReplayDagRunner
    runner = runner_at(tmp_path, cls)
    old = deepcopy(runner.state)
    stop(runner)
    write_json_atomic(tmp_path / "runner_state.json", old)
    assert cls.resume(manifest(), tmp_path).next_action()["attempt"] == 1


def test_dev80_reproduces_missing_repeat_evidence(tmp_path):
    cls = old_module("replay_dag_runner").ReplayDagRunner
    runner = runner_at(tmp_path, cls)
    row = decision(stop(runner))
    runner.apply_external_decision(row)
    (tmp_path / "external_decisions" / f"{row['decision_sha256']}.json").unlink()
    assert cls.resume(manifest(), tmp_path).next_action()["attempt"] == 2


def crash_before_record(module, monkeypatch):
    original = module.write_json_atomic
    def interrupted(path, value):
        if Path(path).parent.name == "records":
            raise RuntimeError("CRASH_BEFORE_RECORD")
        return original(path, value)
    monkeypatch.setattr(module, "write_json_atomic", interrupted)


def test_dev80_reproduces_pre_payload_recovery_failure(tmp_path, monkeypatch):
    old = old_module("replay_reference_data")
    store = old.ReplayReferenceDataStore.initialize(tmp_path)
    with monkeypatch.context() as m:
        crash_before_record(old, m)
        with pytest.raises(RuntimeError, match="CRASH_BEFORE_RECORD"):
            store.put(fixtures()[0])
    with pytest.raises(FileNotFoundError):
        old.ReplayReferenceDataStore(tmp_path)


def test_old_index_cannot_hide_retained_stop(tmp_path):
    runner = runner_at(tmp_path)
    old = deepcopy(runner.state)
    stop(runner)
    write_json_atomic(tmp_path / "runner_state.json", old)
    before = snapshot(tmp_path)
    with pytest.raises(ReplayDagRunnerError, match="RUNNER_AUDIT_ROLLBACK"):
        ReplayDagRunner.resume(manifest(), tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["deleted", "corrupt", "unindexed", "missing_capsule"])
def test_repeat_evidence_is_required_on_resume(tmp_path, fault):
    runner = runner_at(tmp_path)
    capsule = stop(runner)
    row = decision(capsule)
    runner.apply_external_decision(row)
    path = tmp_path / "external_decisions" / f"{row['decision_sha256']}.json"
    if fault == "deleted":
        path.unlink()
    elif fault == "corrupt":
        write_json_atomic(path, {**row, "bound_result_sha256": "f" * 64})
    elif fault == "missing_capsule":
        (tmp_path / "audit_capsules" / f"{capsule['audit_capsule_sha256']}.json").unlink()
    else:
        state = deepcopy(runner.state)
        state["external_decision_sha256s"] = []
        state["state_sha256"] = canonical_sha256({k: v for k, v in state.items() if k != "state_sha256"})
        write_json_atomic(tmp_path / "runner_state.json", state)
    before = snapshot(tmp_path)
    with pytest.raises(ReplayDagRunnerError):
        ReplayDagRunner.resume(manifest(), tmp_path)
    assert snapshot(tmp_path) == before


def test_multiple_repeats_and_success_resume(tmp_path):
    runner = runner_at(tmp_path)
    for attempt in (1, 2, 3):
        capsule = stop(runner)
        assert capsule["attempt"] == attempt
        runner.apply_external_decision(decision(capsule))
        runner = ReplayDagRunner.resume(manifest(), tmp_path)
        assert runner.next_action()["attempt"] == attempt + 1
    runner.record_node_result(result(runner.next_action()))
    assert len(ReplayDagRunner.resume(manifest(), tmp_path).state["completed_node_ids"]) == 1


@pytest.mark.parametrize("name", ["PATCH_REQUIRED", "NEW_SCIENCE_REQUIRED", "CERTIFY_AND_ADVANCE", "CONTINUE"])
def test_other_decisions_remain_resumable(tmp_path, name):
    runner = runner_at(tmp_path)
    row = decision(stop(runner), name)
    runner.apply_external_decision(row)
    resumed = ReplayDagRunner.resume(manifest(), tmp_path)
    if name in {"CERTIFY_AND_ADVANCE", "CONTINUE"}:
        assert len(resumed.state["completed_node_ids"]) == 1
    else:
        assert resumed.next_action()["action"] == "WAIT_FOR_EXTERNAL_AUDIT"


def test_crash_after_capsule_before_state_refuses(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    original = dag.write_json_atomic
    def crash(path, value):
        if Path(path).name == "runner_state.json" and list((tmp_path / "audit_capsules").glob("*.json")):
            raise RuntimeError("STATE_CRASH")
        return original(path, value)
    with monkeypatch.context() as m:
        m.setattr(dag, "write_json_atomic", crash)
        with pytest.raises(RuntimeError, match="STATE_CRASH"):
            stop(runner)
    before = snapshot(tmp_path)
    with pytest.raises(ReplayDagRunnerError, match="RUNNER_AUDIT_ROLLBACK"):
        ReplayDagRunner.resume(manifest(), tmp_path)
    assert snapshot(tmp_path) == before


def test_crash_after_decision_before_state_refuses(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    row = decision(stop(runner))
    def crash():
        raise RuntimeError("STATE_CRASH")
    monkeypatch.setattr(runner, "_write_state", crash)
    with pytest.raises(RuntimeError, match="STATE_CRASH"):
        runner.apply_external_decision(row)
    before = snapshot(tmp_path)
    with pytest.raises(ReplayDagRunnerError, match="RUNNER_DECISION_INVENTORY_MISMATCH"):
        ReplayDagRunner.resume(manifest(), tmp_path)
    assert snapshot(tmp_path) == before


def test_reference_recovers_before_payload_and_commits_once(tmp_path, monkeypatch):
    store = reference.ReplayReferenceDataStore.initialize(tmp_path)
    row = fixtures()[0]
    with monkeypatch.context() as m:
        crash_before_record(reference, m)
        with pytest.raises(RuntimeError, match="CRASH_BEFORE_RECORD"):
            store.put(row)
    transactions = list((tmp_path / "transactions").glob("*.json"))
    assert len(transactions) == 1
    original = transactions[0].read_bytes()
    resumed = reference.ReplayReferenceDataStore(tmp_path)
    assert resumed.manifest["record_ids"] == [row["record_id"]]
    assert resumed.put(row) == row
    assert transactions[0].read_bytes() == original
    assert len(list((tmp_path / "commits").glob("*.json"))) == 1
    assert reference.ReplayReferenceDataStore(tmp_path).manifest == resumed.manifest


def test_legacy_missing_payload_refuses_then_exact_restoration_recovers(tmp_path, monkeypatch):
    old = old_module("replay_reference_data")
    store = old.ReplayReferenceDataStore.initialize(tmp_path)
    row = fixtures()[0]
    with monkeypatch.context() as m:
        crash_before_record(old, m)
        with pytest.raises(RuntimeError):
            store.put(row)
    before = snapshot(tmp_path)
    with pytest.raises(reference.ReferenceDataError, match="LEGACY_TRANSACTION_PAYLOAD_REQUIRED"):
        reference.ReplayReferenceDataStore(tmp_path)
    assert snapshot(tmp_path) == before
    write_json_atomic(tmp_path / "records" / f"{row['record_sha256']}.json", row)
    assert reference.ReplayReferenceDataStore(tmp_path).manifest["record_ids"] == [row["record_id"]]


def test_corrupt_embedded_record_does_not_publish(tmp_path, monkeypatch):
    store = reference.ReplayReferenceDataStore.initialize(tmp_path)
    with monkeypatch.context() as m:
        crash_before_record(reference, m)
        with pytest.raises(RuntimeError):
            store.put(fixtures()[0])
    path, = (tmp_path / "transactions").glob("*.json")
    tx = json.loads(path.read_text())
    tx["record"]["record_id"] = "IGRD/WRONG/OBJECT"
    write_json_atomic(path, tx)
    before = snapshot(tmp_path)
    with pytest.raises(reference.ReferenceDataError, match="transaction hash mismatch"):
        reference.ReplayReferenceDataStore(tmp_path)
    assert snapshot(tmp_path) == before
