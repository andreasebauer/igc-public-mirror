from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.replay_obligation_compiler import (
    ReplayObligationError,
    compile_obligations,
    verify_compiled_manifest,
)


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "infinity_grid/resources/replay/L0_UPWARD_SRCF_OBLIGATION_SOURCE_V2.json"
)


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def make_authorization(node_id: str, *, downstream=()):
    auth = {
        "authorization_id": "IGA/L0/S/PRIMITIVE_RELATION_GENERATION/V1",
        "canonical_obligation_ids": [node_id],
        "historical_aliases": ["L0 audit"],
        "historical_source_hashes": [{"ref": "historical-result.json", "sha256": "1" * 64}],
        "expected_result_identity_or_comparison_contract": {"mode": "EXACT", "sha256": "2" * 64},
        "carrier": "FINITE_PRIMITIVE_RELATION_FAMILY",
        "equality_or_observer": "CANONICAL_EXACT_EQUALITY",
        "scope_and_bounds": "Complete declared L0 finite scope",
        "earned_algebra_statement": "Historical L0 structure reproduced within declared scope.",
        "strength": "EXHAUSTIVE",
        "preserved_falsifications": [],
        "limitations_and_nonclaims": ["No claim above L0"],
        "accepted_equivalence_certificates": [],
        "authorized_downstream_nodes": list(downstream),
        "audit_provenance": [{"ref": "historical-audit.txt", "sha256": "3" * 64}],
        "decision": "CONTINUE",
    }
    auth["decision_hash"] = canonical_sha256(auth)
    return auth


def test_p1a2_compiles_all_layers_and_four_lanes_fail_closed():
    result = compile_obligations(load_source())
    verify_compiled_manifest(result)
    assert result["status"] == "BLOCKED_UNMAPPED_AND_UNAUTHORIZED"
    assert result["execution_authorized"] is False
    assert result["catalogue_complete"] is False
    assert result["replay_authorizations_complete"] is False
    assert result["counts"] == {
        "layers": 16,
        "nodes": 80,
        "scientific": 3,
        "catalogue_gaps": 61,
        "workflow_gates": 16,
        "historical_audit_authorizations": 0,
        "unmapped": 61,
        "unauthorized_scientific": 3,
    }
    for layer in (row["layer"] for row in result["layers"]):
        ids = {node["canonical_id"] for node in result["nodes"] if node["layer"] == layer}
        for series in ("S", "R", "C", "F"):
            assert any(f"IG/{layer}/{series}/" in node_id for node_id in ids)
        assert f"IG/{layer}/GATE/CERTIFY" in ids
        assert not any("/ADMIN/" in node_id for node_id in ids)


def test_p1a2_audit_is_external_and_ci_is_excluded_from_science():
    result = compile_obligations(load_source())
    assert result["audit_policy"]["owner"] == "EXTERNAL_CHAT_WITH_USER"
    assert result["audit_policy"]["decoder_may_interpret_earned_algebra"] is False
    assert result["audit_policy"]["waiting_state"] == "WAITING_FOR_EXTERNAL_AUDIT"
    assert "EXACT_HISTORICAL_REPLAY_AUTHORIZED" in result["audit_policy"]["automatic_continue_outcomes"]
    assert "MISSING_HISTORICAL_AUDIT_AUTHORIZATION" in result["audit_policy"]["stop_outcomes"]
    assert "CONTROLLER_TESTS" in result["excluded_from_scientific_catalogue"]
    assert "PACKAGING_TESTS" in result["excluded_from_scientific_catalogue"]


def test_p1a2_is_deterministic_and_preserves_historical_axis_warning():
    first = compile_obligations(load_source())
    second = compile_obligations(load_source())
    assert first == second
    assert first["dag_sha256"] == second["dag_sha256"]
    assert "candidate-stage alias" in first["historical_axis_warning"]["Gx:Cn"]
    assert "recursive-stage alias" in first["historical_axis_warning"]["Gx:Rn"]


def test_p1a2_rejects_missing_required_scientific_field():
    source = load_source()
    del source["obligations"][0]["formation_provenance"]
    with pytest.raises(ReplayObligationError, match="formation_provenance"):
        compile_obligations(source)


def test_p1a2_rejects_unknown_dependency():
    source = load_source()
    source["obligations"][0]["dependencies"] = ["IG/L0/S/DOES_NOT_EXIST"]
    with pytest.raises(ReplayObligationError, match="unknown dependency"):
        compile_obligations(source)


def test_p1a2_rejects_cycle():
    source = load_source()
    source["obligations"][0]["dependencies"] = ["IG/L0/C/PORT_TYPE_AND_INTERFACE"]
    with pytest.raises(ReplayObligationError, match="dependency cycle"):
        compile_obligations(source)


def test_p1a2_rejects_forward_layer_dependency():
    source = load_source()
    later = deepcopy(source["obligations"][0])
    later["canonical_id"] = "IG/G8/S/LATE_SOURCE"
    later["layer"] = "G8"
    source["obligations"].append(later)
    source["obligations"][0]["dependencies"] = ["IG/G8/S/LATE_SOURCE"]
    with pytest.raises(ReplayObligationError, match="points forward"):
        compile_obligations(source)


def test_p1a2_compiled_hash_tamper_is_rejected():
    result = compile_obligations(load_source())
    result["counts"]["nodes"] += 1
    with pytest.raises(ReplayObligationError, match="hash mismatch"):
        verify_compiled_manifest(result)


def test_p1a2_f_node_must_target_existing_src_obligation():
    source = load_source()
    f_node = deepcopy(source["obligations"][0])
    f_node.update(
        canonical_id="IG/L0/F/PRIMITIVE_RELATION_HOLDOUT",
        series="F",
        dependencies=["IG/L0/S/PRIMITIVE_RELATION_GENERATION"],
        f_targets=["IG/L0/S/PRIMITIVE_RELATION_GENERATION"],
        audit_authorization_ids=[],
    )
    source["obligations"].append(f_node)
    result = compile_obligations(source)
    assert result["counts"]["scientific"] == 4
    assert result["counts"]["catalogue_gaps"] == 60

    source["obligations"][-1]["f_targets"] = ["IG/L0/S/UNKNOWN"]
    with pytest.raises(ReplayObligationError, match="unknown F target"):
        compile_obligations(source)


def test_p1a2_non_f_node_cannot_claim_f_targets():
    source = load_source()
    source["obligations"][0]["f_targets"] = [source["obligations"][1]["canonical_id"]]
    with pytest.raises(ReplayObligationError, match="only F nodes"):
        compile_obligations(source)


def test_p1a2_valid_hash_bound_authorization_changes_only_bound_node_state():
    source = load_source()
    node_id = source["obligations"][0]["canonical_id"]
    auth = make_authorization(node_id, downstream=["IG/L0/R/BRIDGE_REFUSAL_AND_BOUNDARY_RELATIONS"])
    source["historical_audit_authorizations"] = [auth]
    source["obligations"][0]["audit_authorization_ids"] = [auth["authorization_id"]]
    result = compile_obligations(source)
    assert result["counts"]["historical_audit_authorizations"] == 1
    assert result["counts"]["unauthorized_scientific"] == 2
    nodes = {node["canonical_id"]: node for node in result["nodes"]}
    assert nodes[node_id]["audit_authorization_state"] == "HISTORICALLY_AUTHORIZED"


def test_p1a2_rejects_tampered_audit_decision():
    source = load_source()
    node_id = source["obligations"][0]["canonical_id"]
    auth = make_authorization(node_id)
    auth["earned_algebra_statement"] = "tampered after decision"
    source["historical_audit_authorizations"] = [auth]
    source["obligations"][0]["audit_authorization_ids"] = [auth["authorization_id"]]
    with pytest.raises(ReplayObligationError, match="decision hash mismatch"):
        compile_obligations(source)


def test_p1a2_rejects_nonreciprocal_or_unknown_audit_binding():
    source = load_source()
    node_id = source["obligations"][0]["canonical_id"]
    source["historical_audit_authorizations"] = [make_authorization(node_id)]
    with pytest.raises(ReplayObligationError, match="not reciprocal"):
        compile_obligations(source)


def test_p1a2_orders_each_later_layer_after_previous_gate():
    result = compile_obligations(load_source())
    nodes = {node["canonical_id"]: node for node in result["nodes"]}
    assert "IG/L0/GATE/CERTIFY" in nodes["IG/C0_HISTORICAL/S/UNMAPPED_CATALOGUE_GAP"]["dependencies"]
    assert "IG/G7/GATE/CERTIFY" in nodes["IG/G8/F/UNMAPPED_CATALOGUE_GAP"]["dependencies"]


def test_p1a2_alias_index_can_retain_one_historical_bundle_for_multiple_roles():
    source = load_source()
    source["obligations"][0]["historical_aliases"].append("Scout.bundle.example")
    source["obligations"][1]["historical_aliases"].append("Scout.bundle.example")
    result = compile_obligations(source)
    assert result["historical_alias_index"]["Scout.bundle.example"] == [
        "IG/L0/R/BRIDGE_REFUSAL_AND_BOUNDARY_RELATIONS",
        "IG/L0/S/PRIMITIVE_RELATION_GENERATION",
    ]
