from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid import __version__
from infinity_grid.replay_obligation_compiler import ReplayObligationError, compile_obligations


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V8.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C4_EXECUTION_CLASS_LEDGER_V1.json"
CERT = ROOT / "infinity_grid/resources/replay/P1D_AUTHORIZED_PREFIX_REPLAY_CERTIFICATE_V3.json"


def load_source():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_p1c4_classifies_every_assertion_and_scientific_node():
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger["counts"]["assertions"] == 121
    assert ledger["counts"]["scientific_nodes"] == 64
    assert ledger["counts"]["assertion_execution_classes"] == {
        "FRESH_RECOMPUTE": 12,
        "VERIFIED_RESTORED_BLOCK": 14,
        "SOURCE_INTEGRITY_ONLY": 55,
        "HISTORICAL_RESULT_ONLY": 40,
    }
    assert ledger["counts"]["effective_node_execution_classes"] == {
        "FRESH_RECOMPUTE": 3,
        "VERIFIED_RESTORED_BLOCK": 1,
        "SOURCE_INTEGRITY_ONLY": 43,
        "HISTORICAL_RESULT_ONLY": 17,
    }
    assert ledger["counts"]["mixed_class_nodes"] == 6


def test_p1c4_does_not_overstate_empty_root_coverage():
    source = load_source()
    coverage = source["empty_root_science_coverage"]
    assert coverage["status"] == "FULL_EMPTY_ROOT_REPLAY_NOT_ACHIEVED"
    assert coverage["pre_global_scientific_nodes"] == 60
    assert coverage["pre_global_empty_root_counting_nodes"] == 3
    assert coverage["pre_global_empty_root_fraction"] == "3/60"
    assert coverage["complete_pre_global_empty_root_layers"] == []


def test_p1c4_node_class_is_conservative_for_mixed_assertions():
    source = load_source()
    node = next(row for row in source["obligations"]
                if row["canonical_id"] == "IG/O1_O3/S/ASSERTION_MAPPED_STRUCTURE")
    assert node["execution_class_is_mixed"] is True
    assert node["execution_classes_present"] == ["FRESH_RECOMPUTE", "HISTORICAL_RESULT_ONLY"]
    assert node["effective_execution_class"] == "HISTORICAL_RESULT_ONLY"
    assert node["counts_toward_empty_root_science_replay"] is False


def test_p1c4_compiler_rejects_optimistic_node_reclassification():
    source = deepcopy(load_source())
    node = next(row for row in source["obligations"]
                if row["canonical_id"] == "IG/O1_O3/S/ASSERTION_MAPPED_STRUCTURE")
    node["effective_execution_class"] = "FRESH_RECOMPUTE"
    with pytest.raises(ReplayObligationError, match="not conservative"):
        compile_obligations(source)


def test_p1c4_compiled_dag_preserves_execution_counts_and_boundary():
    result = compile_obligations(load_source())
    assert result["counts"]["assertion_execution_classes"]["FRESH_RECOMPUTE"] == 12
    assert result["counts"]["effective_node_execution_classes"]["FRESH_RECOMPUTE"] == 3
    assert result["authorized_replay_through_layer"] == "G8"
    assert result["next_wait_layer"] == "GLOBAL"


def test_p1c4_certificate_states_evidence_replay_not_empty_root_completion():
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    assert cert["decoder_version"] == "0.8.0.dev45+lib"
    # The current identity is tested centrally; this record keeps its historical pin.
    assert cert["status"] == "QUALIFIED_EVIDENCE_REPRODUCTION_WAITING_AT_GLOBAL"
    assert cert["empty_root_science_coverage"]["status"] == "FULL_EMPTY_ROOT_REPLAY_NOT_ACHIEVED"
    assert cert["empty_root_science_coverage"]["pre_global_empty_root_fraction"] == "3/60"
