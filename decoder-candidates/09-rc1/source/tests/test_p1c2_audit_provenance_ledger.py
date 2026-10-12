from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid.replay_obligation_compiler import (
    ReplayObligationError,
    compile_obligations,
    verify_local_source_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V6.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C2_LAYER_AUDIT_PROVENANCE_LEDGER_V1.json"
BLANKET_ONLY = {"C0_HISTORICAL", "G1", "G2", "G3", "G4", "G5", "G6"}
COMPLETE_RECORDED = {"L0", "NODE_IN", "SCOUT_HISTORICAL"}


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_p1c2_separates_replay_authority_from_recorded_layer_audits():
    source = load_source()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    by_layer = {row["layer"]: row for row in ledger["entries"]}
    assert len(by_layer) == 15
    assert {layer for layer, row in by_layer.items()
            if row["audit_record_status"] == "USER_BLANKET_REPLAY_AUTHORIZATION_ONLY"} == BLANKET_ONLY
    assert {layer for layer, row in by_layer.items()
            if row["complete_layer_audit_recorded"]} == COMPLETE_RECORDED
    assert all(row["replay_authority_basis"] == "USER_BLANKET_REPLAY_AUTHORIZATION"
               for row in by_layer.values())
    assert all(row["independent_replay_recorded"] is False for row in by_layer.values())


def test_p1c2_authorizations_preserve_honest_audit_status_and_prefix():
    source = load_source()
    result = compile_obligations(source)
    assert result["authorized_replay_through_layer"] == "G8"
    assert result["next_wait_layer"] == "GLOBAL"
    for auth in source["historical_audit_authorizations"]:
        layer = auth["authorization_id"].split("/")[1]
        assert auth["authority_basis"] == "USER_BLANKET_REPLAY_AUTHORIZATION"
        assert auth["independent_replay_recorded"] is False
        if layer in BLANKET_ONLY:
            assert auth["audit_record_status"] == "USER_BLANKET_REPLAY_AUTHORIZATION_ONLY"
            assert auth["audit_evidence"] == []
        else:
            assert auth["audit_evidence"]


def test_p1c2_all_source_audit_and_ledger_pins_match_local_bytes():
    verify_local_source_bindings(load_source(), ROOT)


def test_p1c2_tampered_audit_evidence_pin_fails_closed():
    source = deepcopy(load_source())
    auth = next(row for row in source["historical_audit_authorizations"] if row["audit_evidence"])
    auth["audit_evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ReplayObligationError, match="source hash mismatch"):
        verify_local_source_bindings(source, ROOT)
