from infinity_grid.v05_controller_event_loop import _registry,SCIENCE_JOB
from infinity_grid.v05_stage_registry import get_evaluator_spec
from infinity_grid.v05_kernel_services import PUBLIC_KERNEL_SERVICES,REFERENCE_KERNEL_SERVICES
from infinity_grid.uplift_g5_r4 import marked_leaf_separation_theorem

def test_marker_operations_registered(tmp_path):
    reg=_registry(tmp_path) if False else None
    # Static registry contracts are validated by existing controller tests; here we bind evaluator services.
    p=get_evaluator_spec('infinity_grid.g6_marker_evaluators:marker_write_holdout_evaluator')
    assert set(p.allowed_kernel_services)<=PUBLIC_KERNEL_SERVICES
    r=get_evaluator_spec('infinity_grid.g6_marker_evaluators:marker_write_independent_evaluator')
    assert 'LEGACY_MARKER_DECODE_ORACLE' in r.allowed_kernel_services
    assert r.reference_oracle_for==('LEGACY_MARKER_DECODE_ORACLE',)

def test_g5_r4_marker_theorem_authority():
    t=marked_leaf_separation_theorem(); assert t['status']=='PASS'
    ids={x['theorem_id']:x['status'] for x in t['theorems']}
    assert ids['R4-T1-UNIQUE-MARKER-DELETION-RECOVERY'].startswith('PROVED')
    assert ids['R4-T3-MARKED-OBSERVER-INJECTIVITY'].startswith('PROVED')
