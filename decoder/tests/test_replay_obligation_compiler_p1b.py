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
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V3.json"


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_p1b_compiles_assertion_level_mappings_without_executing_science():
    result = compile_obligations(load_source())
    verify_compiled_manifest(result)
    assert result["science_executed"] is False
    assert result["status"] == "BLOCKED_MAPPING_CATALOGUE_OR_AUTHORIZATION"
    assert result["execution_authorized"] is False
    assert result["counts"] == {
        "layers": 16,
        "nodes": 80,
        "scientific": 41,
        "catalogue_gaps": 23,
        "workflow_gates": 16,
        "historical_audit_authorizations": 0,
        "historical_assertion_mappings": 98,
        "mapped_assertions": 85,
        "ambiguous_assertions": 13,
        "excluded_assertions": 0,
        "unmapped": 23,
        "unauthorized_scientific": 41,
    }


def test_p1b_all_local_source_hashes_match_candidate_bytes():
    verify_local_source_bindings(load_source(), ROOT)


def test_p1b_o7_historical_g_labels_are_not_relabelled_as_g_layers():
    result = compile_obligations(load_source())
    rows = {row["assertion_id"]: row for row in result["historical_assertion_mappings"]}
    assert rows["IGAM/O4_O7/REGISTRY/O7_G8"]["canonical_obligation_ids"] == [
        "IG/O4_O7/C/ASSERTION_MAPPED_INTERFACES"
    ]
    assert rows["IGAM/O4_O7/REGISTRY/O7_G3"]["canonical_obligation_ids"] == [
        "IG/O4_O7/S/ASSERTION_MAPPED_STRUCTURE"
    ]


def test_p1b_ambiguous_grrl_hypotheses_block_instead_of_guessing():
    result = compile_obligations(load_source())
    assert len(result["unresolved_assertion_mappings"]) == 13
    assert "IGAM/O4_O7/REGISTRY/O7_H1" in result["unresolved_assertion_mappings"]
    row = next(
        item for item in result["historical_assertion_mappings"]
        if item["assertion_id"] == "IGAM/O4_O7/REGISTRY/O7_H1"
    )
    assert row["mapping_status"] == "AMBIGUOUS"
    assert set(row["canonical_obligation_ids"]) == {
        "IG/O4_O7/S/ASSERTION_MAPPED_STRUCTURE",
        "IG/O4_O7/R/ASSERTION_MAPPED_RELATIONS",
        "IG/O4_O7/C/ASSERTION_MAPPED_INTERFACES",
    }


def test_p1b_rejects_nonreciprocal_assertion_binding():
    source = load_source()
    node = max(source["obligations"], key=lambda item: len(item["assertion_mapping_ids"]))
    node["assertion_mapping_ids"].pop()
    with pytest.raises(ReplayObligationError, match="not reciprocal"):
        compile_obligations(source)


def test_p1b_excluded_assertion_cannot_bind_science():
    source = load_source()
    row = deepcopy(source["historical_assertion_mappings"][0])
    row.update(
        assertion_id="IGAM/GLOBAL/EXCLUDED/CONTROLLER_ONLY",
        mapping_status="EXCLUDED",
        evidence_mode="EXCLUDED_NONSCIENTIFIC",
        exclusion_category="CONTROLLER_TESTS",
        canonical_obligation_ids=["IG/GLOBAL/F/ASSERTION_MAPPED_FALSIFICATIONS"],
    )
    source["historical_assertion_mappings"].append(row)
    with pytest.raises(ReplayObligationError, match="excluded assertion cannot bind science"):
        compile_obligations(source)


def test_p1b_preserves_g7_g8_scope_and_open_global_frontier():
    result = compile_obligations(load_source())
    rows = {row["assertion_id"]: row for row in result["historical_assertion_mappings"]}
    assert "conditional" in rows["IGAM/G8/C/BOUNDED_GRADUATION"]["scope_and_bounds"].lower()
    assert "unique organization" in rows["IGAM/G7/S/C9A_C9B_STATE"]["scope_and_bounds"]
    assert "frontier" in rows["IGAM/GLOBAL/R/CORRELATION_FRONTIER"]["claim_summary"].lower()
