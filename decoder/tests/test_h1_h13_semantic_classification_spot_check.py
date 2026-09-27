from __future__ import annotations

import hashlib
import json
from pathlib import Path

from infinity_grid import __version__


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "infinity_grid/resources/replay/H1_H13_SEMANTIC_CLASSIFICATION_SPOT_CHECK_V1.json"
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json"
THEOREM = ROOT / "p1c_historical_sources/GRRL_THEOREM_SPEC_v1.json"
INSTANTIATION = ROOT / "p1c_historical_sources/O7_GRRL_INSTANTIATION_RESULT.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_spot_check_remains_a_dev48_historical_non_authorizing_record():
    audit = load(AUDIT)
    # The current identity is tested centrally; this record keeps its historical pin.
    assert audit["decoder_version"] == "0.8.0.dev48+lib"
    assert audit["status"] == "PASS_NO_RECLASSIFICATION"
    assert audit["classification_changes"] == []
    assert audit["science_executed"] is False
    assert audit["science_authority_effect"] == "NONE"
    assert audit["shared_pointer_advanced"] is False


def test_frozen_theorem_and_instantiation_bytes_match_pins():
    audit = load(AUDIT)["source_identity"]
    assert sha256(THEOREM) == audit["theorem_spec"]["sha256"]
    assert sha256(INSTANTIATION) == audit["o7_instantiation"]["sha256"]
    assert audit["theorem_spec"]["matches_original_bundle"] is True
    assert audit["o7_instantiation"]["matches_original_bundle"] is True


def test_every_hypothesis_text_and_single_series_mapping_match_the_audit():
    audit = load(AUDIT)
    theorem = load(THEOREM)["hypotheses"]
    source = load(SOURCE)
    rows = {
        row["assertion_id"].rsplit("H", 1)[1]: row
        for row in source["historical_assertion_mappings"]
        if row["assertion_id"].startswith("IGAM/O4_O7/REGISTRY/O7_H")
    }
    assert set(rows) == {str(n) for n in range(1, 14)}
    for hypothesis, series in audit["expected_series_map"].items():
        row = rows[hypothesis[1:]]
        assert row["claim_summary"] == theorem[hypothesis]
        assert len(row["canonical_obligation_ids"]) == 1
        assert row["canonical_obligation_ids"][0].split("/")[2] == series


def test_all_thirteen_verdicts_are_explicit_and_borderlines_are_adjudicated():
    rows = {row["hypothesis"]: row for row in load(AUDIT)["classifications"]}
    assert set(rows) == {f"H{n}" for n in range(1, 14)}
    assert all(row["verdict"] == "CONFIRMED" for row in rows.values())
    assert all(rows[key].get("borderline_adjudication") for key in ("H2", "H7", "H11"))


def test_historical_o7_pass_is_not_relabelled_as_independent_confirmation():
    audit = load(AUDIT)
    instantiation = load(INSTANTIATION)
    assert instantiation["all_H1_H13_pass"] is True
    assert all(value == "PASS" for value in instantiation["hypotheses"].values())
    assert "independent confirmation of the historical PASS results" in audit["o7_result_boundary"]["nonclaims"]
    assert audit["o7_result_boundary"]["meaning"].endswith("classification semantics only.")
