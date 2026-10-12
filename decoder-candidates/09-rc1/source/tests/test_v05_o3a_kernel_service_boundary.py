from __future__ import annotations

from pathlib import Path
import hashlib
import pytest

from infinity_grid.v05_kernel_services import (
    KERNEL_SERVICE_CATALOG, PUBLIC_KERNEL_SERVICES, REFERENCE_KERNEL_SERVICES,
    KernelServiceViolation, KernelView, EvaluatorSpec, PRIMARY, REFERENCE_ORACLE,
    audit_evaluator_semantics, bind_kernel_view, current_kernel_view,
)
from infinity_grid.v05_stage_registry import (
    CONTROLLER_ONLY_EVALUATOR_SPECS, CONTROLLER_ONLY_WORKER_EVALUATORS,
    EVALUATOR_SPEC_BY_REF, controller_only_registry_audit, get_evaluator_spec,
)


def test_o3a_registry_has_one_machine_readable_spec_per_production_evaluator():
    assert len(CONTROLLER_ONLY_EVALUATOR_SPECS) == 25
    assert CONTROLLER_ONLY_WORKER_EVALUATORS == tuple(s.ref for s in CONTROLLER_ONLY_EVALUATOR_SPECS)
    assert set(EVALUATOR_SPEC_BY_REF) == set(CONTROLLER_ONLY_WORKER_EVALUATORS)
    assert "infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator" in EVALUATOR_SPEC_BY_REF
    for spec in CONTROLLER_ONLY_EVALUATOR_SPECS:
        assert spec.allowed_kernel_services
        if spec.role == PRIMARY:
            assert set(spec.allowed_kernel_services) <= PUBLIC_KERNEL_SERVICES
        elif spec.role == REFERENCE_ORACLE:
            assert set(spec.reference_oracle_for) <= REFERENCE_KERNEL_SERVICES


def test_o3b_closes_o3a_legacy_bridge_for_all_production_evaluators():
    audit = controller_only_registry_audit()
    assert audit["status"] == "PASS"
    assert audit["legacy_frozen_evaluator_count"] == 0
    for row in audit["evaluators"]:
        sem = row["semantic_gate"]
        assert sem["status"] == "PASS"
        assert sem["migration_required"] is False
        assert sem["legacy_semantic_source_sha256"] is None
        assert sem["violation_count"] == 0


def test_o3a_modified_duplicate_semantics_fail_when_frozen_hash_no_longer_matches(tmp_path):
    p = tmp_path / "new_eval.py"
    p.write_text("from infinity_grid.exact_tree_relation_kernel import get_relation_kernel\n", encoding="utf-8")
    spec = EvaluatorSpec(
        "infinity_grid.example:new_eval", PRIMARY, ("EXACT_RELATION",),
        legacy_semantic_source_sha256="0" * 64,
    )
    out = audit_evaluator_semantics(p, spec)
    assert out["status"] == "FAIL"
    assert out["violations"][0]["covered_service"] == "EXACT_RELATION"
    assert "KernelView" in out["violations"][0]["guidance"]


def test_o3a_service_only_evaluator_source_passes_semantic_gate(tmp_path):
    p = tmp_path / "thin_eval.py"
    p.write_text("def evaluator(payload):\n    return payload\n", encoding="utf-8")
    spec = EvaluatorSpec("infinity_grid.example:evaluator", PRIMARY, ("EXACT_RELATION",))
    out = audit_evaluator_semantics(p, spec)
    assert out["status"] == "PASS"
    assert out["violation_count"] == 0


def test_o3a_kernel_view_fails_closed_on_undeclared_or_unbound_service():
    spec = EvaluatorSpec("infinity_grid.example:evaluator", PRIMARY, ("EXACT_RELATION",))
    view = KernelView(spec)
    assert view.allows("EXACT_RELATION")
    with pytest.raises(KernelServiceViolation, match="KERNEL_SERVICE_NOT_DECLARED"):
        view.require("OBSERVER_Q")
    with pytest.raises(KernelServiceViolation, match="KERNEL_SERVICE_PROVIDER_NOT_BOUND"):
        view.call("EXACT_RELATION", 1, 2)


def test_o3a_kernel_view_provider_binding_is_restricted_and_execution_local():
    spec = EvaluatorSpec("infinity_grid.example:evaluator", PRIMARY, ("EXACT_RELATION",))
    with pytest.raises(KernelServiceViolation, match="KERNEL_VIEW_PROVIDER_UNDECLARED"):
        KernelView(spec, {"OBSERVER_Q": lambda x: x})
    view = bind_kernel_view(spec, {"EXACT_RELATION": lambda a,b: (a,b)})
    assert current_kernel_view() is view
    assert view.call("EXACT_RELATION", "A", "B") == ("A", "B")


def test_o3a_reference_oracle_role_cannot_be_smuggled_into_primary_services():
    with pytest.raises(ValueError, match="PRIMARY_NONPUBLIC"):
        EvaluatorSpec("infinity_grid.example:evaluator", PRIMARY, ("LEGACY_OBSERVER_DECODE_ORACLE",))
    ref = EvaluatorSpec(
        "infinity_grid.example:oracle", REFERENCE_ORACLE, ("EXACT_RELATION",),
        reference_oracle_for=("LEGACY_OBSERVER_DECODE_ORACLE",),
    )
    assert ref.reference_oracle_for == ("LEGACY_OBSERVER_DECODE_ORACLE",)


def test_o3a_catalog_names_are_unique_and_visibility_is_complete():
    assert len(KERNEL_SERVICE_CATALOG) == len(set(KERNEL_SERVICE_CATALOG))
    assert {v["visibility"] for v in KERNEL_SERVICE_CATALOG.values()} == {"PUBLIC", "KERNEL_INTERNAL", "REFERENCE_ONLY"}
