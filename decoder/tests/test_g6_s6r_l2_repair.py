from pathlib import Path
from infinity_grid.v05_controller_event_loop import _registry, SCIENCE_JOB

def test_l2_operation_registered():
    source=Path(__file__).resolve().parents[1]
    assert 'G6_S6R_L2_REPAIR' in _registry(source)[SCIENCE_JOB]['allowed_operations']

def test_l2_evaluator_registered_and_scoped():
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    s=get_evaluator_spec('infinity_grid.g6_s6r_evaluators:l2_common_parent_uniqueness_evaluator')
    assert set(s.allowed_kernel_services)=={'EXACT_IDENTITY','OBSERVER_Q','OBSERVER_DECODE'}

def test_l2_module_is_fail_closed():
    import infinity_grid.g6_s6r_l2_repair as m
    text=Path(m.__file__).read_text(encoding='utf-8')
    assert 'L2_NO_COUNTEREXAMPLE_IN_EXHAUSTIVE_P_ONLY_00_SCOPE_CONTINUE_PROOF' in text
    assert "'strong_l2_proved':False" in text
    assert 'L2_STRONG_REFUTED_WITH_COUNTEREXAMPLE' in text
