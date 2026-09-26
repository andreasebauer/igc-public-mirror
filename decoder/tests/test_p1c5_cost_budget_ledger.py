from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid import __version__
from infinity_grid.replay_obligation_compiler import ReplayObligationError, compile_obligations


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C5_COST_AND_BUDGET_LEDGER_V1.json"
CERT = ROOT / "infinity_grid/resources/replay/P1D_AUTHORIZED_PREFIX_REPLAY_CERTIFICATE_V4.json"
CAP37 = "IG/G8/F/ASSERTION_MAPPED_FALSIFICATIONS"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_scientific_node_has_both_cost_scenarios():
    source = load(SOURCE)
    nodes = [row for row in source["obligations"] if row["node_kind"] == "SCIENTIFIC"]
    assert len(nodes) == 64
    assert all(set(row["cost_budget"]) == {
        "current_evidence_replay", "prospective_full_empty_root_recompute"
    } for row in nodes)


def test_cost_ledger_preserves_honest_pre_global_counts():
    ledger = load(LEDGER)
    assert ledger["counts"]["pre_global_scientific_nodes"] == 60
    assert ledger["counts"]["current_pre_global_outcomes"] == {
        "BUDGET_REQUIRED_BEFORE_RUN": 1,
        "LIKELY_WITHIN_BUDGET": 59,
    }
    assert ledger["counts"]["prospective_full_empty_root_pre_global_outcomes"] == {
        "BUDGET_REQUIRED_BEFORE_RUN": 57,
        "LIKELY_WITHIN_BUDGET": 2,
        "PERFORMANCE_BUDGET_EXCEEDED_EXPECTED": 1,
    }


def test_cap37_is_expected_to_exceed_current_performance_budget():
    ledger = load(LEDGER)
    row = next(item for item in ledger["node_records"] if item["canonical_id"] == CAP37)
    assert row["prospective_full_empty_root_recompute"]["expected_outcome"] == (
        "PERFORMANCE_BUDGET_EXCEEDED_EXPECTED"
    )


def test_g6_measurement_is_shared_and_not_a_full_g6_estimate():
    ledger = load(LEDGER)
    measured = [row for row in ledger["node_records"]
                if "measurement_group_id" in row["current_evidence_replay"]]
    assert len(measured) == 2
    assert {row["current_evidence_replay"]["measurement_group_id"] for row in measured} == {
        "G6_DEV45_TWO_MODULE_PROBE"
    }
    assert all("MAPPED_SUBSET_ONLY_NOT_FULL_HISTORICAL_G6_SCIENCE"
               in row["prospective_full_empty_root_recompute"]["basis"] for row in measured)


def test_compiler_rejects_missing_cost_budget_under_policy():
    source = deepcopy(load(SOURCE))
    del source["obligations"][0]["cost_budget"]
    with pytest.raises(ReplayObligationError, match="cost budget missing"):
        compile_obligations(source)


def test_certificate_blocks_full_empty_root_launch_without_fabricated_total():
    cert = load(CERT)
    assert cert["decoder_version"] == "0.8.0.dev46+lib"
    # The current identity is tested centrally; this record keeps its historical pin.
    assert cert["status"] == (
        "QUALIFIED_EVIDENCE_REPRODUCTION_BUDGET_INCOMPLETE_WAITING_AT_GLOBAL"
    )
    full = cert["prospective_full_empty_root_recompute_budget"]
    assert full["status"] == "DO_NOT_LAUNCH_COST_MODEL_INCOMPLETE"
    assert full["total_runtime_estimate"] == "UNAVAILABLE"
    assert full["expected_performance_budget_exceeded_nodes"] == [CAP37]
