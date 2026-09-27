from pathlib import Path
import hashlib
import json
import sys

import pytest

from infinity_grid import certified_base as cb
from infinity_grid import semantic_inputs as si
from infinity_grid import submission as sub


def add(store, raw, *, category="core_objects", coverage=None, meaning="fixture.shared",
        domain="engineering.fixture", schema="IG_FIXTURE_V1", invariants=None,
        producer_version="0.6.0"):
    row = cb.add_block(
        store, category=category, content=raw, canonicalization="RAW_BYTES_V1",
        schema_id=schema, meaning=meaning, domain=domain,
        coverage=["fixture.base"] if coverage is None else coverage,
        invariants=["bytes.exact"] if invariants is None else invariants,
        producer={"decoder_version": producer_version, "test_version": "DIAGNOSTIC.ONLY"},
    )
    cb.certify_block(store, row["block_id"], authority="DECODER.V07.STEP3",
                     basis=["exact.bytes", "declared.semantics"])
    return row


def need(need_id, *, coverage=None, capabilities=None, role="DATA", required=True,
         meaning="fixture.shared", domain="engineering.fixture", schema="IG_FIXTURE_V1",
         invariants=None):
    return {
        "need_id": need_id, "schema_id": schema, "meaning": meaning, "domain": domain,
        "required_capabilities": capabilities or [], "required_coverage": coverage or [],
        "required_invariants": invariants or ["bytes.exact"],
        "control_role": role, "required": required,
    }


def base(store, rows):
    return cb.create_base(store, name="STEP3.FIXTURE.BASE", block_ids=[r["block_id"] for r in rows])


def test_versions_are_diagnostic_and_do_not_gate_selection(tmp_path):
    old = add(tmp_path, b"old-certified", coverage=["fixture.base"], producer_version="0.5.9")
    catalog = base(tmp_path, [old])
    needs = si.create_test_needs(test_id="STEP3.VERSION.INDEPENDENT",
                                 requirements=[need("base", capabilities=["fixture.base"])])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    assert [row["block_id"] for row in selected["selected_blocks"]] == [old["block_id"]]
    assert selected["producer_versions_used_for_admission"] is False


def test_smallest_cover_is_one_block_then_fewest_bytes(tmp_path):
    both_big = add(tmp_path, b"X" * 20, coverage=["cap.a", "cap.b"])
    both_small = add(tmp_path, b"Y" * 5, coverage=["cap.a", "cap.b"])
    only_a = add(tmp_path, b"a", coverage=["cap.a"])
    only_b = add(tmp_path, b"b", coverage=["cap.b"])
    catalog = base(tmp_path, [both_big, both_small, only_a, only_b])
    needs = si.create_test_needs(test_id="STEP3.MINIMUM", requirements=[
        need("a", capabilities=["cap.a"]), need("b", coverage=["cap.b"]),
    ])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    assert selected["selection_metrics"] == {"block_count": 1, "total_size_bytes": 5}
    assert selected["selected_blocks"][0]["block_id"] == both_small["block_id"]


def test_equal_size_tie_is_deterministic_by_block_id(tmp_path):
    one = add(tmp_path, b"1", coverage=["cap.a"])
    two = add(tmp_path, b"2", coverage=["cap.a"])
    catalog = base(tmp_path, [two, one])
    needs = si.create_test_needs(test_id="STEP3.TIE",
                                 requirements=[need("a", capabilities=["cap.a"])])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    assert selected["selected_blocks"][0]["block_id"] == min(one["block_id"], two["block_id"])


@pytest.mark.parametrize("change", ["schema", "meaning", "domain", "coverage", "invariant"])
def test_incompatible_or_incomplete_semantics_refuse(tmp_path, change):
    kwargs = {}
    if change == "schema": kwargs["schema"] = "IG_OTHER_V1"
    if change == "meaning": kwargs["meaning"] = "fixture.other"
    if change == "domain": kwargs["domain"] = "other.domain"
    row = add(tmp_path, b"x", coverage=["cap.other"] if change == "coverage" else ["cap.a"],
              invariants=["other.invariant"] if change == "invariant" else ["bytes.exact"], **kwargs)
    catalog = base(tmp_path, [row])
    needs = si.create_test_needs(test_id="STEP3.REFUSE", requirements=[
        need("a", capabilities=["cap.a"], invariants=["bytes.exact"]),
    ])
    with pytest.raises(si.SemanticInputError, match="TEST_INPUT_REQUIRED_NEED_UNSATISFIED"):
        si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)


def test_control_roles_select_only_declared_control_category(tmp_path):
    data = add(tmp_path, b"d", coverage=["common"])
    positive = add(tmp_path, b"p", category="positive_controls", coverage=["common"])
    negative = add(tmp_path, b"n", category="negative_controls", coverage=["common"])
    catalog = base(tmp_path, [data, positive, negative])
    needs = si.create_test_needs(test_id="STEP3.CONTROLS", requirements=[
        need("data", coverage=["common"]),
        need("positive", coverage=["common"], role="POSITIVE_CONTROL"),
        need("negative", coverage=["common"], role="NEGATIVE_CONTROL"),
    ])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    assert {r["category"] for r in selected["selected_blocks"]} == {
        "core_objects", "positive_controls", "negative_controls"}


def test_optional_unavailable_is_reported_but_not_selected(tmp_path):
    row = add(tmp_path, b"x", coverage=["cap.a"])
    catalog = base(tmp_path, [row])
    needs = si.create_test_needs(test_id="STEP3.OPTIONAL", requirements=[
        need("a", coverage=["cap.a"]),
        need("future", coverage=["cap.future"], required=False),
    ])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    assert selected["optional_available_not_selected"] == {"future": []}
    assert selected["selection_metrics"]["block_count"] == 1


def test_changed_bytes_corruption_and_revocation_refuse(tmp_path):
    row = add(tmp_path, b"trusted", coverage=["cap.a"])
    catalog = base(tmp_path, [row])
    needs = si.create_test_needs(test_id="STEP3.BYTES", requirements=[need("a", coverage=["cap.a"])])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    test_hash = hashlib.sha256(b"test").hexdigest(); decoder_hash = hashlib.sha256(b"decoder").hexdigest()
    receipt = si.record_run(tmp_path, selection_id=selected["record_sha256"],
                            test_code_sha256=test_hash, decoder_code_sha256=decoder_hash,
                            parameters={"n": 1}, result={"status": "PASS"})
    assert si.verify_run_receipt(tmp_path, receipt["record_sha256"])["result"]["status"] == "PASS"
    cb.revoke(tmp_path, row["block_id"], reason="fixture.defect", evidence=["TICKET.STEP3"])
    with pytest.raises(si.SemanticInputError, match="CERTIFIED_BASE_CONTAINS_REVOKED"):
        si.verify_run_receipt(tmp_path, receipt["record_sha256"])


def test_run_receipt_binds_selection_code_parameters_and_result(tmp_path):
    row = add(tmp_path, b"trusted", coverage=["cap.a"])
    catalog = base(tmp_path, [row])
    needs = si.create_test_needs(test_id="STEP3.RECEIPT", requirements=[need("a", coverage=["cap.a"])])
    selected = si.resolve_test_inputs(tmp_path, catalog["base_id"], needs)
    test_hash = hashlib.sha256(b"test-v1").hexdigest(); decoder_hash = hashlib.sha256(b"decoder-v1").hexdigest()
    receipt = si.record_run(tmp_path, selection_id=selected["record_sha256"],
                            test_code_sha256=test_hash, decoder_code_sha256=decoder_hash,
                            parameters={"limit": 7}, result={"status": "PASS", "count": 7})
    checked = si.verify_run_receipt(tmp_path, receipt["record_sha256"],
                                    expected_test_code_sha256=test_hash,
                                    expected_decoder_code_sha256=decoder_hash)
    assert checked["selected_blocks"][0]["content_sha256"] == row["content_sha256"]
    with pytest.raises(si.SemanticInputError, match="RUN_RECEIPT_TEST_CODE_MISMATCH"):
        si.verify_run_receipt(tmp_path, receipt["record_sha256"],
                              expected_test_code_sha256="0" * 64)
    path = tmp_path / "run_receipts" / (receipt["record_sha256"] + ".json")
    tampered = json.loads(path.read_text()); tampered["result"]["count"] = 8
    path.write_text(json.dumps(tampered))
    with pytest.raises(si.SemanticInputError, match="SEMANTIC_RECORD_HASH_INVALID"):
        si.verify_run_receipt(tmp_path, receipt["record_sha256"])


def test_input_classification_keeps_administration_out_of_project_requirements():
    capture = {
        "job": {"input_artifacts": [
            {"logical_name": "science_data", "sha256": "1" * 64},
            {"logical_name": "runtime_dependency_snapshot", "sha256": "2" * 64},
            {"logical_name": "submission_contract", "sha256": "3" * 64},
        ]},
        "environment": {"artifacts": [
            {"logical_name": "runtime_dependency_snapshot", "sha256": "2" * 64},
        ]},
    }
    classified = si.validate_project_inputs(capture, ["science_data"], reject_unexpected=True)
    assert [r["logical_name"] for r in classified["project_inputs"]] == ["science_data"]
    assert {r["logical_name"] for r in classified["administrative_inputs"]} == {
        "runtime_dependency_snapshot", "submission_contract"}
    with pytest.raises(si.SemanticInputError, match="PROJECT_INPUT_RESERVED_ADMINISTRATIVE"):
        si.validate_project_inputs(capture, ["runtime_dependency_snapshot"])
    with pytest.raises(si.SemanticInputError, match="PROJECT_INPUT_MISSING"):
        si.validate_project_inputs(capture, ["science_data", "missing"])


def _capture_spec(tmp_path):
    project = tmp_path / "project"; project.mkdir(); (project / "main.py").write_text("print(1)\n")
    return {
        "schema_id": sub.SPEC_SCHEMA, "job_id": "STEP3.INPUT.NAMESPACE",
        "engine_source": str(Path(sub.__file__).resolve().parents[1]),
        "project_source": str(project),
        "question": {"stage_id": "DECODER:STEP3", "description": "namespace fixture",
                     "outcomes": ["PROCESS_COMPLETED", "PROCESS_FAILED"],
                     "stopping_rule": "one fixture"},
        "execution": {"kind": "SCRIPT", "entrypoint": "project/main.py", "argv": [], "parameters": {}},
        "resources": {"workers": 1, "start_method": "fork", "memory_budget_bytes": 268435456,
                      "workspace_budget_bytes": 268435456, "wall_seconds_max": 30},
        "inputs": [],
        "environment": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                        "requirements": [], "artifacts": []},
        "output_contract": {"process": "exit code recorded", "scientific_acceptance": "NONE"},
    }


def test_capture_reserves_admin_names_for_environment(tmp_path):
    spec = _capture_spec(tmp_path)
    snapshot = tmp_path / "deps.json"; snapshot.write_text("{}")
    row = {"logical_name": "runtime_dependency_snapshot", "path": str(snapshot),
           "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    spec["inputs"] = [row]
    with pytest.raises(sub.SubmissionError, match="CAPTURE_INPUT_FIELDS"):
        sub.capture(tmp_path / "bad-store", spec)
    spec["inputs"] = []; spec["environment"]["artifacts"] = [row]
    state = sub.capture(tmp_path / "good-store", spec)
    capture = sub.capture_record(state["workspace"])
    classified = si.classify_capture_inputs(capture)
    assert classified["project_inputs"] == []
    assert {r["logical_name"] for r in classified["administrative_inputs"]} == {
        "runtime_dependency_snapshot", "submission_contract"}
