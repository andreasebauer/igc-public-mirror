from pathlib import Path
import pytest

def test_wider_operation_is_registered_without_external_execution():
    import infinity_grid.v05_controller_event_loop as m
    source=Path(m.__file__).resolve().parent.parent
    reg=m._registry(source)
    assert 'G6_S5R_WIDER_FEATURE_SEARCH' in reg[m.SCIENCE_JOB]['allowed_operations']
    assert len(reg[m.SCIENCE_JOB]['implementation_sha256'])==64

def test_controller_event_runtime_permit_cannot_be_minted_externally(tmp_path):
    from infinity_grid.v05_execution_authority import issue_controller_event_runtime_permit, ExecutionAuthorityError
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises((ExecutionAuthorityError,RuntimeError)):
        issue_controller_event_runtime_permit(chain_dir=tmp_path,chain_id='x',stage_id='G6:S5R-WIDER',question_sha256='0'*64,registration_sha256='1'*64,source_sha256='2'*64,handler_key='x',handler_ref='x:y',handler_source_sha256='3'*64,parameters_sha256='4'*64,authority_sha256='5'*64,dependencies_sha256='6'*64,run_id='7'*32,evaluator_refs=('infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator',))

def test_wider_module_has_fixed_evaluator_and_candidate_order():
    import infinity_grid.g6_s5r_wider_feature_search as m
    assert m.OBSERVER_EVALUATOR=='infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator'
    assert m.STAGE_ID=='G6:S5R-WIDER'


def test_event_loop_dispatches_wider_operation():
    import infinity_grid.v05_controller_event_loop as m
    text=Path(m.__file__).read_text()
    assert "record['requested_operation_id'] in ('G6_S5R_OBSERVER_CONTINUITY','G6_S5R_WIDER_FEATURE_SEARCH','G6_S5R_COMPOSITIONAL_READ_WRITE','G6_S5R_COMPOSITIONAL_READ_WRITE_CRW1','G6_S6R_RECURSIVE_CLOSURE','G6_S6R_GLOBAL_OBSERVER_CONGRUENCE','G6_S6R_L2_REPAIR','G6_S6R_L2_PROOF_REDUCTION','G6_S5R_FRESH_MARKER_OBSERVER','G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE','G6_R0_POST_GRADUATION_FIBER','G6_S7_ORDINARY_FUTURE_QUOTIENT','G6_S7_A6_REUSE_PROVENANCE_VERIFY','G6_S7_DEPTH2_COMPLETION','G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE','G6_S8_V3_TARGETED_PROBE')" in text


def test_spawn_pool_guard_removes_launcher_main_path_and_restores_it():
    import multiprocessing.spawn as spawn
    import sys
    from infinity_grid.execution import _suppress_main_reexecution_for_spawn_pool
    main=sys.modules['__main__']
    old=getattr(main,'__file__',None)
    main.__file__='/tmp/decoder-grandfathered-launcher.py'
    try:
        with _suppress_main_reexecution_for_spawn_pool():
            data=spawn.get_preparation_data('ig-probe')
            assert 'init_main_from_path' not in data
        assert main.__file__=='/tmp/decoder-grandfathered-launcher.py'
    finally:
        if old is None:
            try: delattr(main,'__file__')
            except AttributeError: pass
        else:
            main.__file__=old
