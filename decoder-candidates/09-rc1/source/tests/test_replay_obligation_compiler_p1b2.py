from __future__ import annotations

import json
from pathlib import Path

from infinity_grid.replay_obligation_compiler import (
    compile_obligations,
    verify_compiled_manifest,
    verify_local_source_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V4.json"


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_p1b2_compiles_complete_catalogue_without_executing_science():
    result = compile_obligations(load_source())
    verify_compiled_manifest(result)
    assert result["science_executed"] is False
    assert result["status"] == "BLOCKED_UNAUTHORIZED"
    assert result["catalogue_complete"] is True
    assert result["execution_authorized"] is False
    assert result["counts"] == {
        "layers": 16,
        "nodes": 80,
        "scientific": 64,
        "catalogue_gaps": 0,
        "workflow_gates": 16,
        "historical_audit_authorizations": 0,
        "historical_assertion_mappings": 121,
        "mapped_assertions": 121,
        "ambiguous_assertions": 0,
        "excluded_assertions": 0,
        "unmapped": 0,
        "unauthorized_scientific": 64,
    }
    assert result["unresolved_assertion_mappings"] == []


def test_p1b2_all_local_source_hashes_match_candidate_bytes():
    verify_local_source_bindings(load_source(), ROOT)


def test_p1b2_grrl_hypotheses_have_exact_single_series_classifications():
    result = compile_obligations(load_source())
    rows = {row["assertion_id"]: row for row in result["historical_assertion_mappings"]}
    expected = {
        1: "C", 2: "S", 3: "S", 4: "R", 5: "S", 6: "C", 7: "R",
        8: "R", 9: "C", 10: "R", 11: "S", 12: "C", 13: "C",
    }
    for hypothesis, series in expected.items():
        row = rows[f"IGAM/O4_O7/REGISTRY/O7_H{hypothesis}"]
        assert row["mapping_status"] == "MAPPED"
        assert row["canonical_obligation_ids"] == [
            f"IG/O4_O7/{series}/ASSERTION_MAPPED_"
            + {"S": "STRUCTURE", "R": "RELATIONS", "C": "INTERFACES"}[series]
        ]


def test_p1b2_g1_crosswalk_does_not_relabel_historical_o7_gate_names():
    result = compile_obligations(load_source())
    rows = {row["assertion_id"]: row for row in result["historical_assertion_mappings"]}
    assert rows["IGAM/O4_O7/REGISTRY/O7_G1"]["canonical_obligation_ids"] == [
        "IG/O4_O7/F/ASSERTION_MAPPED_FALSIFICATIONS"
    ]
    crosswalk = rows["IGAM/G1/S/O7_TO_G1_EARNED_CARRIER"]
    assert crosswalk["canonical_obligation_ids"] == [
        "IG/G1/S/ASSERTION_MAPPED_STRUCTURE"
    ]
    assert crosswalk["locator"]["source_ref"].endswith(
        "O7_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATION_STATEMENT.txt"
    )


def test_p1b2_every_layer_has_all_four_scientific_series_and_no_authority():
    result = compile_obligations(load_source())
    scientific = [node for node in result["nodes"] if node["node_kind"] == "SCIENTIFIC"]
    by_layer = {}
    for node in scientific:
        by_layer.setdefault(node["layer"], set()).add(node["series"])
        assert node["audit_authorization_ids"] == []
    assert len(by_layer) == 16
    assert all(series == {"S", "R", "C", "F"} for series in by_layer.values())
