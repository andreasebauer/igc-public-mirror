from __future__ import annotations
import json, tempfile
from pathlib import Path
import pytest

def test_postc5_registry_has_passive_g6_science_and_no_c5_finalize():
    from infinity_grid.v05_controller_event_loop import _registry, SCIENCE_JOB, ENGINEERING_JOB
    src=Path(__file__).resolve().parent.parent
    reg=_registry(src)
    assert 'DECODER.C5.FINALIZE' not in reg
    assert reg[SCIENCE_JOB]['allowed_operations']==['G6_S5R_OBSERVER_CONTINUITY','G6_S5R_WIDER_FEATURE_SEARCH','G6_S5R_COMPOSITIONAL_READ_WRITE','G6_S5R_COMPOSITIONAL_READ_WRITE_CRW1','G6_S6R_RECURSIVE_CLOSURE','G6_S6R_GLOBAL_OBSERVER_CONGRUENCE','G6_S6R_L2_REPAIR','G6_S6R_L2_PROOF_REDUCTION','G6_S5R_FRESH_MARKER_OBSERVER','G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE','G6_R0_POST_GRADUATION_FIBER','G6_S7_ORDINARY_FUTURE_QUOTIENT','G6_S7_A6_REUSE_PROVENANCE_VERIFY','G6_S7_DEPTH2_COMPLETION','G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE','G6_S8_V3_TARGETED_PROBE']
    assert 'APPLY_SOURCE_CHANGE_AND_ACTIVATE' in reg[ENGINEERING_JOB]['allowed_operations']

def test_s5r_plan_requires_fixed_observer_and_factorization():
    import infinity_grid.g6_s5r_observer_continuity as m
    assert m.PLAN_SCHEMA=='IG_G6_S5R_OBSERVER_CONTINUITY_REPAIR_REGISTRATION_V1'
    assert m.EXPECTED_LOGICAL_NAMES=={'science_plan','master_prereg','g5_parent_review','s0_s1_closeout','s1r_closeout','s2r_closeout','s3r_closeout','s4r_closeout','s5_closeout','e3_recovered_fixture','audit'}

def test_direct_science_evaluator_call_is_rejected():
    from infinity_grid.g6_s5r_observer_continuity import run_observer_continuity_repair
    with pytest.raises(Exception) as ei:
        run_observer_continuity_repair(plan_path='/nope',artifacts={},output_dir=tempfile.mkdtemp(),accepted_source_sha256='0'*64,internal_execution_id='x')
    assert 'REJECT_EXTERNAL_EXECUTION_ORIGIN' in str(ei.value)

def test_final_source_direct_bootstrap_remains_closed():
    from infinity_grid.v05_controller_event_loop import start_c5_migration_supervisor
    with pytest.raises(Exception) as ei:
        start_c5_migration_supervisor('/tmp/nope','/tmp/nope')
    assert 'REJECT_DIRECT_EXECUTION_ROUTE:C5_BOOTSTRAP_CLOSED' in str(ei.value)
