from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from infinity_grid import __version__
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.replay_reference_data import (
    EQUALITY_MODES, PROVENANCE_CLASSES, RECORD_SCHEMA, RECORD_TYPES,
    SCIENCE_EXECUTION_STATUSES, ReferenceDataError, ReplayReferenceDataStore,
    empty_manifest, seal_reference_record, verify_reference_record,
)


ZERO = "0" * 64
ONE = "1" * 64


def equality(mode="EXACT_BYTES", cardinality="ORDERED", compression="EXACT", certificate=None):
    return {
        "mode": mode,
        "cardinality_semantics": cardinality,
        "compression": compression,
        "observer": "DECLARED_TEST_OBSERVER",
        "certificate_sha256": certificate,
    }


def record(kind, ident, payload, dependencies=(), *, provenance="FINITE_COMPUTATIONAL_OBSERVATION",
           provenance_status="PINNED", epistemic="REPLAYED", science_execution="NONE",
           authority="NONE"):
    sources = [] if provenance == "UNEXPLAINED" else [{"ref": "fixture/source.json", "sha256": ONE}]
    return seal_reference_record({
        "schema_id": RECORD_SCHEMA,
        "record_id": ident,
        "record_type": kind,
        "layer": "L0",
        "payload": payload,
        "provenance": {
            "classification": provenance,
            "status": provenance_status,
            "source_hashes": sources,
            "explanation": "P3 contract qualification fixture",
        },
        "scope": {"fixture": "P3_ONLY"},
        "nonclaims": ["NO_SCIENCE_AUTHORITY_GRANTED"],
        "dependencies": list(dependencies),
        "epistemic_status": epistemic,
        "science_execution": science_execution,
        "authority_effect": authority,
    })


def fixtures():
    obj = record("CANONICAL_OBJECT", "IGRD/L0/OBJECT/A", {
        "object_identity": "L0-A", "carrier_schema": "IG_L0_TEST_CARRIER_V1",
        "canonical_bytes_sha256": ZERO, "formation_provenance": "EMPTY_ROOT_GENERATION",
    })
    evidence = record("SRCF_EVIDENCE", "IGRD/L0/EVIDENCE/HIST", {
        "series": "S", "obligation_id": "IG/L0/S/ASSERTION_MAPPED_STRUCTURE",
        "result_identity": ZERO, "equality_contract": equality(),
        "evidence_mode": "HISTORICAL_RESULT_ONLY", "outcome": "REPRODUCED",
    }, epistemic="HISTORICAL")
    replay = record("SRCF_EVIDENCE", "IGRD/L0/EVIDENCE/REPLAY", {
        "series": "S", "obligation_id": "IG/L0/S/ASSERTION_MAPPED_STRUCTURE",
        "result_identity": ZERO, "equality_contract": equality(),
        "evidence_mode": "FRESH_RECOMPUTE", "outcome": "REPRODUCED",
    })
    mechanism = record("MECHANISM", "IGRD/L0/MECHANISM/GEN", {
        "mechanism_id": "L0-GEN", "input_record_ids": [obj["record_id"]],
        "output_record_ids": ["IGRD/L0/OBJECT/B"], "determinism": "DETERMINISTIC",
        "implementation_hashes": [{"ref": "generator.py", "sha256": ZERO}],
    }, [obj["record_id"]])
    graduation = record("GRADUATION", "IGRD/L0/GRADUATION/A", {
        "decision": "CONDITIONAL", "graduated_scope": {"bounded": True},
        "authorizes": ["L0_DOWNSTREAM_FIXTURE"], "evidence_record_ids": [evidence["record_id"]],
    }, [evidence["record_id"]])
    negative = record("NEGATIVE_RESULT", "IGRD/L0/NEGATIVE/A", {
        "obligation_id": "IG/L0/F/ASSERTION_MAPPED_FALSIFICATIONS",
        "tested_scope": {"cases": 4}, "negative_statement": "No witness in bounded fixture",
        "witnesses": [], "does_not_establish": ["UNBOUNDED_ABSENCE", "GLOBAL_CLOSURE"],
    })
    audit = record("AUDIT_AUTHORIZATION", "IGRD/L0/AUDIT/A", {
        "authorization_id": "IGA/L0/TEST", "obligation_ids": ["IG/L0/S/ASSERTION_MAPPED_STRUCTURE"],
        "decision": "CONTINUE", "authorized_scope": {"fixture": True},
        "limitations": ["P3_FIXTURE_ONLY"], "evidence_hashes": [{"ref": "audit.txt", "sha256": ONE}],
    }, provenance="EXTERNAL_AUDIT_DECISION", epistemic="EXTERNAL_DECISION",
       authority="EXTERNAL_DECISION_RECORDED")
    earned = record("EARNED_ALGEBRA", "IGRD/L0/ALGEBRA/A", {
        "statement_id": "L0-ALG-A", "statement": "Fixture statement only",
        "strength": "CONDITIONAL", "statement_scope": {"fixture": True},
        "supporting_record_ids": [graduation["record_id"]],
        "nonclaims": ["NO_GLOBAL_THEOREM"],
    }, [graduation["record_id"]])
    recipe = record("GENERATION_RECIPE", "IGRD/L0/RECIPE/A", {
        "recipe_id": "L0-RECIPE-A", "implementation_ref": "infinity_grid.fixture:generate",
        "implementation_sha256": ZERO, "input_record_ids": [obj["record_id"]],
        "parameters": {"limit": 1}, "expected_record_types": ["CANONICAL_OBJECT"],
        "determinism_contract": "BYTE_IDENTICAL",
    }, [obj["record_id"]])
    comparison = record("COMPARISON", "IGRD/L0/COMPARISON/A", {
        "obligation_id": "IG/L0/S/ASSERTION_MAPPED_STRUCTURE",
        "historical_record_ids": [evidence["record_id"]],
        "replay_record_ids": [replay["record_id"]], "equality_contract": equality(),
        "outcome": "REPRODUCED", "qualification_ids": [],
    }, [evidence["record_id"], replay["record_id"]])
    link = record("DEPENDENCY_LINK", "IGRD/L0/LINK/A", {
        "from_record_id": obj["record_id"], "to_record_id": evidence["record_id"],
        "relation": "REQUIRES", "required": True,
    }, [obj["record_id"], evidence["record_id"]])
    frontier = record("RESUME_FRONTIER", "IGRD/L0/FRONTIER/A", {
        "root_run_id": "P3-FIXTURE", "manifest_dag_sha256": ZERO,
        "runner_state_sha256": ONE, "completed_node_ids": ["IG/L0/S/NODE"],
        "checkpoint_sha256_by_node": {"IG/L0/S/NODE": ZERO},
        "next_node_id": "IG/L0/R/NODE", "frontier_status": "READY",
    })
    return [obj, evidence, replay, mechanism, graduation, negative, audit, earned, recipe, comparison, link, frontier]


def test_dev51_contract_covers_every_p3_record_category():
    # The current identity is tested centrally; this record keeps its historical pin.
    rows = fixtures()
    assert {row["record_type"] for row in rows} == set(RECORD_TYPES)
    assert all(verify_reference_record(row) == row for row in rows)


def test_empty_root_is_deterministic_and_contains_no_invented_seed(tmp_path):
    a = ReplayReferenceDataStore.initialize(tmp_path / "a")
    b = ReplayReferenceDataStore.initialize(tmp_path / "b")
    assert a.manifest == b.manifest == empty_manifest()
    assert a.manifest["record_ids"] == []
    assert a.manifest["science_executed"] is False


def test_frozen_profile_and_empty_root_match_runtime_contract():
    resources = Path(__file__).parents[1] / "infinity_grid" / "resources" / "replay"
    profile = json.loads((resources / "P3_REFERENCE_DATA_CONTRACT_V1.json").read_text())
    frozen_empty = json.loads((resources / "P3_EMPTY_REFERENCE_DATA_ROOT_V1.json").read_text())
    assert frozen_empty == empty_manifest()
    assert set(profile["record_types"]) == set(RECORD_TYPES)
    assert set(profile["provenance_classes"]) == set(PROVENANCE_CLASSES)
    assert set(profile["equality_modes"]) == set(EQUALITY_MODES)
    assert set(profile["science_execution_states"]) == set(SCIENCE_EXECUTION_STATUSES)
    assert profile["empty_root"]["manifest_sha256"] == frozen_empty["manifest_sha256"]


def test_all_categories_store_in_dependency_order_and_manifest_counts_are_exact(tmp_path):
    store = ReplayReferenceDataStore.initialize(tmp_path / "store")
    rows = fixtures()
    for row in rows:
        store.put(row)
    assert store.manifest["record_ids"] == [row["record_id"] for row in rows]
    assert set(store.manifest["record_type_counts"].values()) == {1, 2}
    assert store.manifest["record_type_counts"]["SRCF_EVIDENCE"] == 2
    store.verify()


def test_science_execution_is_explicit_and_manifest_state_is_derived(tmp_path):
    store = ReplayReferenceDataStore.initialize(tmp_path / "store")
    base = fixtures()[2]
    raw = {k: deepcopy(v) for k, v in base.items() if k != "record_sha256"}
    raw["science_execution"] = "EXECUTED"
    executed = seal_reference_record(raw)
    store.put(executed)
    assert store.manifest["science_executed"] is True
    ReplayReferenceDataStore(store.root).verify()


def test_missing_or_forward_dependency_fails_closed(tmp_path):
    store = ReplayReferenceDataStore.initialize(tmp_path / "store")
    row = fixtures()[3]
    with pytest.raises(ReferenceDataError, match="missing reference dependency"):
        store.put(row)
    assert store.manifest["status"] == "EMPTY"


def test_record_identity_is_immutable_and_hash_tampering_is_rejected(tmp_path):
    store = ReplayReferenceDataStore.initialize(tmp_path / "store")
    row = fixtures()[0]; store.put(row)
    changed = deepcopy(row); changed["payload"]["object_identity"] = "OTHER"
    changed = seal_reference_record(changed)
    with pytest.raises(ReferenceDataError, match="record id collision"):
        store.put(changed)
    tampered = deepcopy(row); tampered["payload"]["object_identity"] = "TAMPERED"
    with pytest.raises(ReferenceDataError, match="record hash mismatch"):
        verify_reference_record(tampered)


def test_interrupted_commit_recovers_exactly_once(tmp_path):
    root = tmp_path / "store"; store = ReplayReferenceDataStore.initialize(root)
    row = fixtures()[0]
    with pytest.raises(ReferenceDataError, match="INJECTED_INTERRUPTION"):
        store.put(row, _interrupt_after_record=True)
    assert json.loads((root / "MANIFEST.json").read_text())["status"] == "EMPTY"
    resumed = ReplayReferenceDataStore(root)
    assert resumed.manifest["record_ids"] == [row["record_id"]]
    assert resumed.put(row) == row
    assert resumed.manifest["record_ids"] == [row["record_id"]]


def test_unexplained_provenance_must_remain_explicitly_open():
    base = fixtures()[0]
    raw = {k: deepcopy(v) for k, v in base.items() if k != "record_sha256"}
    raw["record_id"] = "IGRD/L0/OBJECT/UNEXPLAINED"
    raw["provenance"] = {"classification": "UNEXPLAINED", "status": "OPEN",
                         "source_hashes": [], "explanation": "No earned provenance yet"}
    raw["epistemic_status"] = "OPEN"
    assert seal_reference_record(raw)["provenance"]["classification"] == "UNEXPLAINED"
    raw["epistemic_status"] = "REPLAYED"
    with pytest.raises(ReferenceDataError, match="cannot be certified"):
        seal_reference_record(raw)


def test_negative_result_keeps_nonclaims_and_cannot_silently_become_graduation():
    negative = fixtures()[5]
    assert negative["payload"]["does_not_establish"] == ["UNBOUNDED_ABSENCE", "GLOBAL_CLOSURE"]
    raw = {k: deepcopy(v) for k, v in negative.items() if k != "record_sha256"}
    raw["record_type"] = "GRADUATION"
    with pytest.raises(ReferenceDataError, match="GRADUATION payload fields"):
        seal_reference_record(raw)


def test_set_multiset_and_compression_semantics_are_mandatory():
    evidence = fixtures()[1]
    raw = {k: deepcopy(v) for k, v in evidence.items() if k != "record_sha256"}
    raw["record_id"] = "IGRD/L0/EVIDENCE/MULTISET"
    raw["payload"]["equality_contract"] = equality("MULTISET", "MULTISET", "LOSSLESS_COMPRESSED")
    assert seal_reference_record(raw)["payload"]["equality_contract"]["compression"] == "LOSSLESS_COMPRESSED"
    raw["payload"]["equality_contract"]["cardinality_semantics"] = "SET"
    with pytest.raises(ReferenceDataError, match="cardinality disagrees"):
        seal_reference_record(raw)


def test_audit_authority_must_remain_an_external_record():
    audit = fixtures()[6]
    raw = {k: deepcopy(v) for k, v in audit.items() if k != "record_sha256"}
    raw["provenance"]["classification"] = "FINITE_COMPUTATIONAL_OBSERVATION"
    with pytest.raises(ReferenceDataError, match="external audit provenance"):
        seal_reference_record(raw)


def test_manifest_or_record_file_tampering_fails_on_cold_restore(tmp_path):
    root = tmp_path / "store"; store = ReplayReferenceDataStore.initialize(root)
    row = fixtures()[0]; store.put(row)
    manifest = json.loads((root / "MANIFEST.json").read_text()); manifest["status"] = "EMPTY"
    write_json_atomic(root / "MANIFEST.json", manifest)
    with pytest.raises(ReferenceDataError, match="status disagrees|hash mismatch"):
        ReplayReferenceDataStore(root)
