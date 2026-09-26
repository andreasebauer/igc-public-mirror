from pathlib import Path
from infinity_grid.v05_controller_event_loop import _registry, SCIENCE_JOB

def test_l2p_operation_registered():
    source=Path(__file__).resolve().parents[1]
    assert 'G6_S6R_L2_PROOF_REDUCTION' in _registry(source)[SCIENCE_JOB]['allowed_operations']

def test_l2p_evaluator_registered_and_scoped():
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    s=get_evaluator_spec('infinity_grid.g6_s6r_evaluators:l2_sibling_q_evaluator')
    assert set(s.allowed_kernel_services)=={'OBSERVER_Q'}

def test_l2p_module_is_fail_closed_and_nonpromoting():
    import infinity_grid.g6_s6r_l2_proof_reduction as m
    text=Path(m.__file__).read_text(encoding='utf-8')
    assert 'L2_PROOF_REDUCTION_SIBLING_COUNTEREXAMPLE_FOUND' in text
    assert 'L2_REDUCED_TO_SIBLING_SEPARATION_NO_COUNTEREXAMPLE_IN_REGISTERED_SCOPE' in text
    assert "'global_q_injectivity_earned':False" in text
    assert "'sibling_separation_theorem_earned':False" in text
