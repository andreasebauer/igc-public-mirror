from pathlib import Path
from infinity_grid.v05_controller_event_loop import _registry, SCIENCE_JOB
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest

def test_global_operation_registered():
    source=Path(__file__).resolve().parents[1]
    reg=_registry(source)
    assert 'G6_S6R_GLOBAL_OBSERVER_CONGRUENCE' in reg[SCIENCE_JOB]['allowed_operations']

def test_global_module_has_fail_closed_unresolved_path():
    import infinity_grid.g6_s6r_global_observer_congruence as m
    text=Path(m.__file__).read_text(encoding='utf-8')
    assert 'GLOBAL_Q_CONGRUENCE_THEOREM_UNRESOLVED' in text
    assert 'finite_holdouts_used_as_universal_proof' in text
    assert 'L2_COMMON_FALSE_PARENT_EXCLUSION' in text
