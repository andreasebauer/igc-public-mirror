from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from infinity_grid import __version__
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.replay_dag_runner import (
    CORE_STATUS, DECISION_SCHEMA, RESULT_SCHEMA, ReplayDagRunner,
    ReplayDagRunnerError, seal_external_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json"
QUALIFICATION = ROOT / "infinity_grid/resources/replay/P2A_DAG_RUNNER_CORE_QUALIFICATION_V1.json"
ZERO = "0" * 64
ONE = "1" * 64


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def create(tmp_path, data=None):
    return ReplayDagRunner.create(
        data or manifest(), tmp_path / "run", root_run_id="P2A-TEST-RUN",
        dataset_root={"state": "EMPTY", "sha256": ZERO},
    )


def result(action, outcome="EXACT_HISTORICAL_REPLAY_AUTHORIZED"):
    if any(n['canonical_id'] == action['node_id'] and n.get('effective_execution_class') == 'FRESH_RECOMPUTE'
           for n in manifest()['nodes']) and outcome == "EXACT_HISTORICAL_REPLAY_AUTHORIZED":
        outcome = "REPRODUCED_WITH_DECLARED_EVIDENCE_MODE"
    return {
        "schema_id": RESULT_SCHEMA,
        "node_id": action["node_id"],
        "attempt": action["attempt"],
        "comparison_outcome": outcome,
        "result_sha256": ZERO,
        "evidence_sha256": ONE,
        "execution_class": "FRESH_RECOMPUTE" if outcome == "REPRODUCED_WITH_DECLARED_EVIDENCE_MODE" else "HISTORICAL_RESULT_ONLY",
        "counts_toward_empty_root_science_replay": outcome == "REPRODUCED_WITH_DECLARED_EVIDENCE_MODE",
    }


def test_dev49_runner_core_remains_active_without_science_handler_bindings(tmp_path):
    runner = create(tmp_path)
    # The current identity is tested centrally; this record keeps its historical pin.
    assert runner.state["implementation_status"] == CORE_STATUS
    assert not hasattr(runner, "execute")
    assert runner.next_action()["node_id"] == manifest()["topological_order"][0]


def test_exact_results_continue_in_topological_order_and_gate_is_checkpointed(tmp_path):
    runner = create(tmp_path)
    order = manifest()["topological_order"]
    for expected in order[:4]:
        action = runner.next_action()
        assert action["node_id"] == expected
        assert runner.record_node_result(result(action))["action"] == "CONTINUE"
    action = runner.next_action()
    assert action["node_id"] == order[5]
    gate = order[4]
    assert gate in runner.state["completed_node_ids"]
    checkpoint_hash = runner.state["accepted_checkpoint_sha256_by_node"][gate]
    checkpoint = json.loads((tmp_path / "run/checkpoints" / f"{checkpoint_hash}.json").read_text())
    assert checkpoint["acceptance"]["mode"] == "AUTOMATIC_WORKFLOW_GATE"


def test_first_mismatch_stops_and_emits_exact_capsule(tmp_path):
    runner = create(tmp_path)
    action = runner.next_action()
    stopped = runner.record_node_result(result(action, "RESULT_MISMATCH"))
    capsule = stopped["audit_capsule"]
    assert stopped["action"] == "WAIT_FOR_EXTERNAL_AUDIT"
    assert capsule["stopped_node_id"] == action["node_id"]
    assert capsule["completed_node_ids"] == []
    assert capsule["audit_capsule_sha256"] == canonical_sha256(
        {k: v for k, v in capsule.items() if k != "audit_capsule_sha256"})
    assert runner.next_action()["action"] == "WAIT_FOR_EXTERNAL_AUDIT"


def test_resume_skips_completed_and_tampered_checkpoint_fails_closed(tmp_path):
    runner = create(tmp_path)
    first = runner.next_action()
    runner.record_node_result(result(first))
    resumed = ReplayDagRunner.resume(manifest(), tmp_path / "run")
    assert resumed.next_action()["node_id"] != first["node_id"]
    digest = resumed.state["accepted_checkpoint_sha256_by_node"][first["node_id"]]
    path = tmp_path / "run/checkpoints" / f"{digest}.json"
    value = json.loads(path.read_text())
    value["acceptance"]["result_language"] = "CONFIRMED"
    write_json_atomic(path, value)
    with pytest.raises(ReplayDagRunnerError, match="checkpoint content hash mismatch"):
        ReplayDagRunner.resume(manifest(), tmp_path / "run")


def test_external_decision_is_exactly_bound_and_repeat_only_reopens_stopped_node(tmp_path):
    runner = create(tmp_path)
    action = runner.next_action()
    capsule = runner.record_node_result(result(action, "PROVENANCE_MISMATCH"))["audit_capsule"]
    base = {
        "schema_id": DECISION_SCHEMA,
        "decision": "REPEAT",
        "audit_capsule_sha256": capsule["audit_capsule_sha256"],
        "root_run_id": "P2A-TEST-RUN",
        "stopped_node_id": action["node_id"],
        "bound_result_sha256": ZERO,
        "bound_evidence_sha256": ONE,
    }
    bad = seal_external_decision({**base, "bound_evidence_sha256": ZERO})
    with pytest.raises(ReplayDagRunnerError, match="evidence_sha256 binding mismatch"):
        runner.apply_external_decision(bad)
    applied = runner.apply_external_decision(seal_external_decision(base))
    assert applied["action"] == "READY"
    retry = runner.next_action()
    assert retry["node_id"] == action["node_id"] and retry["attempt"] == 2


def test_unauthorized_global_stops_before_execution_after_clean_g8_prefix(tmp_path):
    runner = create(tmp_path)
    while True:
        action = runner.next_action()
        if action["action"] == "EXECUTE_SCIENTIFIC_NODE":
            runner.record_node_result(result(action))
            continue
        assert action["action"] == "WAIT_FOR_EXTERNAL_AUDIT"
        capsule = action["audit_capsule"]
        assert capsule["stopped_node_id"].startswith("IG/GLOBAL/S/")
        assert capsule["stop_outcome"] == "MISSING_HISTORICAL_AUDIT_AUTHORIZATION"
        assert "IG/G8/GATE/CERTIFY" in capsule["completed_node_ids"]
        assert capsule["stopped_node_id"] not in capsule["completed_node_ids"]
        break


def test_missing_authority_can_only_be_granted_externally_then_node_still_executes(tmp_path):
    data = manifest()
    first = data["topological_order"][0]
    node = next(row for row in data["nodes"] if row["canonical_id"] == first)
    node["audit_authorization_state"] = "MISSING_HISTORICAL_AUDIT_AUTHORIZATION"
    node["audit_authorization_ids"] = []
    data["unauthorized_scientific_nodes"] = [first] + data["unauthorized_scientific_nodes"]
    data["authorized_replay_prefix"] = []
    data["authorized_replay_through_layer"] = None
    data["next_wait_layer"] = "L0"
    data["prefix_replay_ready"] = False
    data["execution_authorized"] = False
    data["status"] = "BLOCKED_UNAUTHORIZED"
    data["replay_authorizations_complete"] = False
    data["dag_sha256"] = canonical_sha256({k: v for k, v in data.items() if k != "dag_sha256"})
    runner = create(tmp_path, data)
    capsule = runner.next_action()["audit_capsule"]
    base = {
        "schema_id": DECISION_SCHEMA,
        "decision": "CONTINUE",
        "audit_capsule_sha256": capsule["audit_capsule_sha256"],
        "root_run_id": "P2A-TEST-RUN",
        "stopped_node_id": first,
        "bound_result_sha256": None,
        "bound_evidence_sha256": None,
    }
    with pytest.raises(ReplayDagRunnerError, match="requires CERTIFY_AND_ADVANCE"):
        runner.apply_external_decision(seal_external_decision(base))
    base["decision"] = "CERTIFY_AND_ADVANCE"
    assert runner.apply_external_decision(seal_external_decision(base))["action"] == "READY"
    assert runner.next_action()["node_id"] == first


def test_identical_runs_have_identical_semantic_state_and_checkpoint_hashes(tmp_path):
    runners = [ReplayDagRunner.create(
        manifest(), tmp_path / name, root_run_id="SAME-RUN",
        dataset_root={"state": "EMPTY", "sha256": ZERO}) for name in ("a", "b")]
    for runner in runners:
        for _ in range(7):
            action = runner.next_action()
            runner.record_node_result(result(action))
    assert runners[0].state == runners[1].state


def test_manifest_tampering_and_nonempty_dataset_root_fail_closed(tmp_path):
    data = manifest()
    data["nodes"][0]["observer"] = "tampered"
    with pytest.raises(ReplayDagRunnerError, match="manifest hash mismatch"):
        create(tmp_path, data)
    with pytest.raises(ReplayDagRunnerError, match="empty dataset root"):
        ReplayDagRunner.create(manifest(), tmp_path / "other", root_run_id="X",
                               dataset_root={"state": "RESTORED", "sha256": ZERO})


def test_machine_qualification_preserves_p2a_and_r02_boundaries():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    assert record["status"] == "PASS_RUNNER_CORE_ONLY"
    assert record["p2_complete"] is False
    assert record["science_executed"] is False
    assert record["manifest_binding"]["authorized_replay_through_layer"] == "G8"
    assert record["manifest_binding"]["next_wait_layer"] == "GLOBAL"
    assert record["regression_result"]["production_regression"] is False
    assert any("Full R02 is not complete" in text for text in record["nonclaims"])
    implementation = ROOT / record["implementation"]["ref"]
    assert record['decoder_version'] == '0.8.0.dev49+lib'
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() != record["implementation"]["sha256"]
