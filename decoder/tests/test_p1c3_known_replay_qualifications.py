from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid import __version__
from infinity_grid.replay_obligation_compiler import (
    ReplayObligationError,
    compile_obligations,
    verify_local_source_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V7.json"
CERT = ROOT / "infinity_grid/resources/replay/P1D_AUTHORIZED_PREFIX_REPLAY_CERTIFICATE_V2.json"


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_p1c3_binds_exactly_three_active_known_qualifications():
    source = load_source()
    rows = {row["qualification_id"]: row for row in source["known_replay_qualifications"]}
    assert set(rows) == {
        "IGKQ/G6/FEX1_S7_A30_IN_SAMPLE/V1",
        "IGKQ/G7/RECEIPT_HASH_DISCREPANCY/V1",
        "IGKQ/G8/CAP37_ZERO_DATA_FRONTIER/V1",
    }
    assert all(row["status"] == "ACTIVE" for row in rows.values())
    assert all(row["required_result_language"].startswith(("QUALIFIED_REPRODUCTION", "REPRODUCED"))
               for row in rows.values())


def test_p1c3_a30_namespace_collision_is_explicitly_avoided():
    source = load_source()
    mappings = {row["assertion_id"]: row for row in source["historical_assertion_mappings"]}
    qid = "IGKQ/G6/FEX1_S7_A30_IN_SAMPLE/V1"
    assert qid in mappings["IGAM/G6/F/TEST_G6_S8_INTRINSIC_DESCRIPTOR_PY"]["known_qualification_ids"]
    assert qid not in mappings["IGAM/L2J3/V026_ORACLE/A30"]["known_qualification_ids"]


def test_p1c3_reproduction_language_replaces_confirmation_semantics():
    source = load_source()
    policy = source["result_language_policy"]
    assert policy["success_term"] == "REPRODUCED"
    assert "MATCH" in policy["forbidden_unqualified_terms"]
    assert all(
        row["expected_result_identity_or_comparison_contract"]["on_match"]
        == "REPRODUCED_WITH_DECLARED_EVIDENCE_MODE"
        for row in source["historical_audit_authorizations"]
    )
    compiled = compile_obligations(source)
    assert compiled["authorized_replay_through_layer"] == "G8"
    assert compiled["next_wait_layer"] == "GLOBAL"
    assert compiled["counts"]["known_replay_qualifications"] == 3


def test_p1c3_all_qualification_pins_match_local_bytes():
    verify_local_source_bindings(load_source(), ROOT)


def test_p1c3_tampered_qualification_record_hash_fails_closed():
    source = deepcopy(load_source())
    source["known_replay_qualifications"][0]["record_hash"] = "0" * 64
    with pytest.raises(ReplayObligationError, match="record hash mismatch"):
        compile_obligations(source)


def test_p1c3_corrected_certificate_is_qualified_not_confirmatory():
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    assert cert["decoder_version"] == "0.8.0.dev44+lib"
    # The current identity is tested centrally; this record keeps its historical pin.
    assert cert["status"] == "QUALIFIED_REPRODUCTION_WAITING_AT_GLOBAL"
    assert len(cert["active_known_qualifications"]) == 3
    assert cert["underlying_replay"]["g6_named_executors"]["independent_confirmation"] is False
    assert cert["next_wait_layer"] == "GLOBAL"
