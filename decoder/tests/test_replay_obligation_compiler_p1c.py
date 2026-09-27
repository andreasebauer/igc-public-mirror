from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid.replay_obligation_compiler import (
    ReplayObligationError,
    compile_obligations,
    verify_compiled_manifest,
    verify_local_source_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V5.json"
PREFIX = [
    "L0", "C0_HISTORICAL", "L2J3", "NODE_IN", "SCOUT_HISTORICAL",
    "O1_O3", "O4_O7", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
]


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_p1c_authorizes_contiguous_replay_through_qualified_g8_only():
    result = compile_obligations(load_source())
    verify_compiled_manifest(result)
    assert result["science_executed"] is False
    assert result["status"] == "BLOCKED_UNAUTHORIZED"
    assert result["authorized_replay_prefix"] == PREFIX
    assert result["authorized_replay_through_layer"] == "G8"
    assert result["next_wait_layer"] == "GLOBAL"
    assert result["prefix_replay_ready"] is True
    assert result["execution_authorized"] is False
    assert result["counts"]["historical_audit_authorizations"] == 15
    assert result["counts"]["unauthorized_scientific"] == 4
    assert result["unauthorized_scientific_nodes"] == [
        "IG/GLOBAL/C/ASSERTION_MAPPED_INTERFACES",
        "IG/GLOBAL/F/ASSERTION_MAPPED_FALSIFICATIONS",
        "IG/GLOBAL/R/ASSERTION_MAPPED_RELATIONS",
        "IG/GLOBAL/S/ASSERTION_MAPPED_STRUCTURE",
    ]


def test_p1c_all_mapping_and_audit_source_hashes_match_local_bytes():
    verify_local_source_bindings(load_source(), ROOT)


def test_p1c_authorizations_are_reciprocal_and_cover_exactly_four_lanes_per_layer():
    source = load_source()
    auths = {row["authorization_id"]: row for row in source["historical_audit_authorizations"]}
    nodes = {row["canonical_id"]: row for row in source["obligations"]}
    assert len(auths) == len(PREFIX)
    for layer in PREFIX:
        auth_id = f"IGA/{layer}/P1C/EXTERNAL_AUDIT_REPLAY/V1"
        auth = auths[auth_id]
        assert len(auth["canonical_obligation_ids"]) == 4
        assert {nodes[node_id]["series"] for node_id in auth["canonical_obligation_ids"]} == {
            "S", "R", "C", "F"
        }
        assert all(nodes[node_id]["audit_authorization_ids"] == [auth_id]
                   for node_id in auth["canonical_obligation_ids"])


def test_p1c_g8_is_conditional_and_preserves_zero_data_and_open_frontier():
    source = load_source()
    auth = next(
        row for row in source["historical_audit_authorizations"]
        if row["authorization_id"] == "IGA/G8/P1C/EXTERNAL_AUDIT_REPLAY/V1"
    )
    assert auth["strength"] == "CONDITIONAL"
    joined = " ".join(auth["limitations_and_nonclaims"]).lower()
    assert "zero-data" in joined
    assert "open-frontier" in joined
    assert auth["expected_result_identity_or_comparison_contract"]["g8_frontier_rule"] == (
        "UNAUDITED_G8_FRONTIER"
    )


def test_p1c_external_decision_does_not_authorize_global_components():
    source = load_source()
    global_nodes = [row for row in source["obligations"] if row["layer"] == "GLOBAL"]
    assert len(global_nodes) == 4
    assert all(row["audit_authorization_ids"] == [] for row in global_nodes)
    assert all(
        "GLOBAL" not in auth["authorization_id"]
        for auth in source["historical_audit_authorizations"]
    )


def test_p1c_tampered_external_decision_pin_fails_closed():
    source = load_source()
    source = deepcopy(source)
    source["historical_audit_authorizations"][0]["audit_provenance"][0]["sha256"] = "0" * 64
    with pytest.raises(ReplayObligationError, match="decision hash mismatch"):
        compile_obligations(source)

