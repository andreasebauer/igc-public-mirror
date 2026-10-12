from __future__ import annotations

from infinity_grid.g6_controller_evaluators import (
    g6_s3_axis_a_generation_evaluator, g6_s3_axis_b_generation_evaluator, g6_s3_l1_legacy_signature_evaluator,
)
from infinity_grid.v05_stage_registry import get_evaluator_spec
from infinity_grid.v05_kernel_services import bind_kernel_view, clear_kernel_view
from infinity_grid.v05_kernel_service_providers import build_kernel_service_providers
from infinity_grid.g6_s1_repair import _basis, _tree_to_record
from infinity_grid.g6_s2_repaired import _primary_sig_for_context
from infinity_grid.g6_s3_repaired import _axis_a_task, _axis_b_task
from infinity_grid.exact_tree_relation_kernel import get_relation_kernel, configure_relation_kernel


def _bind(ref,scope):
    configure_relation_kernel(scope_identity=scope); spec=get_evaluator_spec(ref); bind_kernel_view(spec,build_kernel_service_providers(spec))

def _sample_s1_parent():
    configure_relation_kernel(scope_identity='E3:TEST:S1PARENT')
    b = _basis()
    rel = get_relation_kernel().relation(b["D2_BROOM"], b["D2_PATH"], (0, 0))
    assert rel.children
    return rel.children[0]


def test_e3_axis_a_generation_matches_historical_structural_relation():
    _bind('infinity_grid.g6_controller_evaluators:g6_s3_axis_a_generation_evaluator','E3:TEST:AXISA')
    triples = [
        ("D2_BROOM", "D2_PATH", "D4_BROOM"),
        ("D4_PATH", "D4_BROOM", "D2_PATH"),
    ]
    for triple in triples:
        old = _axis_a_task(triple)
        new = g6_s3_axis_a_generation_evaluator({"triple": list(triple)})
        assert {r["exact_canon_repr"] for r in old["records"]} == {
            repr(r["identity"]) for r in new["states"]
        }


def test_e3_axis_b_generation_matches_historical_structural_relation():
    parent = _sample_s1_parent()
    rec = _tree_to_record(parent)
    old = _axis_b_task(("TEST_PARENT", rec))
    _bind('infinity_grid.g6_controller_evaluators:g6_s3_axis_b_generation_evaluator','E3:TEST:AXISB')
    new = g6_s3_axis_b_generation_evaluator({"parent_state_id": "TEST_PARENT", "state_tree": rec})
    assert {r["exact_canon_repr"] for r in old["records"]} == {
        repr(r["identity"]) for r in new["states"]
    }


def test_e3_l1_wrapper_preserves_historical_signature_shape_exactly():
    parent = _sample_s1_parent()
    rec = _tree_to_record(parent)
    ctx = ("D2_PATH", (0, 0), "LEFT")
    old = [{"context": ["D2_PATH", [0, 0], "LEFT"],
            "outcomes": [repr(x) for x in _primary_sig_for_context(parent, ctx)]}]
    _bind('infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator','E3:TEST:L1')
    new = g6_s3_l1_legacy_signature_evaluator({
        "state_tree": rec, "probe_ref": "D2_PATH", "operator": [0, 0], "position": "LEFT"
    })
    assert new["signature"] == old
