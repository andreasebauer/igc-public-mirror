from __future__ import annotations
import copy, inspect
from pathlib import Path
import pytest
from infinity_grid import g6_s8_evaluators as ev
from infinity_grid import g6_s8_v3_targeted_probe as v3
from infinity_grid.canon import canonical_sha256
from infinity_grid import v05_controller_event_loop as loop

V3=Path(__file__).resolve().parent/'fixtures'/'G6_S8_PREREGISTRATION_V3.json'
CTX=Path(__file__).resolve().parent/'fixtures'/'G6_S7_CONTEXT_NORMALIZATION_V1.json'

def _plan():
    import json
    return json.loads(V3.read_text())

def test_v3_hash_split_and_scope_accepts_frozen_plan():
    p=_plan(); q=copy.deepcopy(p); q['execution_contract']['executor_authority']['accepted_source_sha256']='x'*64; q['execution_contract_sha256']=canonical_sha256(q['execution_contract']); tmp=dict(q); tmp.pop('full_preregistration_sha256'); q['full_preregistration_sha256']=canonical_sha256(tmp)
    got=v3._validate_plan(q,accepted_source_sha256='x'*64)
    assert got['scientific_core_sha256']==p['scientific_core_sha256']
    assert got['question_sha256']==got['scientific_core_sha256']
    assert got['scientific_core']['targeted_probe']['execution_context_count']==8
    assert got['execution_contract']['authorized_execution']['full_v2_depth3_to6_sweep_authorized'] is False

def test_v3_executor_is_registered_and_old_v2_remains_registered():
    reg=loop._registry(Path(inspect.getfile(loop)).resolve().parents[1])
    ops=reg[loop.SCIENCE_JOB]['allowed_operations']
    assert 'G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE' in ops
    assert 'G6_S8_V3_TARGETED_PROBE' in ops

def _ops():
    import json
    c=json.loads(CTX.read_text())
    return [r['operator'] for r in c['execution_coordinates'][:31]]

class View:
    def call(self,name,*args):
        if name=='AUTHORITY_BASIS': return {x:{'seed':x} for x in ['D2_BROOM','D2_PATH','D4_BROOM','D4_PATH']}
        if name=='PUBLIC_READ': return {'legal':True,'descriptor':'CAPS7_PLUS_H_CLASS_BAG','caps7':[2,1,0,0,0,0,0],'H_class_bag':{'H':1}}
        if name=='EXACT_RELATION_PROFILE': return {'exact_outcome_count':1}
        if name=='EXACT_RELATION_PROFILE_FAMILY': return {'exact_outcome_counts':[1]*124}
        if name=='EXACT_RELATION': return {'children':[args[0]]}
        raise RuntimeError(name)

def _payload(**kw):
    p={'future_depth':3,'outer_execution_context_index':0,'recursive_prefix_schedule':[1,8,32,124],'s7_class_id':'C','class_members':[{'state_token':'a','state_tree':{'n':1}},{'state_token':'b','state_tree':{'n':1}}],'basis_refs':['D2_BROOM','D2_PATH','D4_BROOM','D4_PATH'],'operator_basis':_ops(),'max_recursive_states':250000,'max_exact_relation_calls':500000,'max_profile_coordinate_equivalents':500000,'max_worker_rss_bytes':2**40,'v3_budget_semantics':True}
    p.update(kw); return p

def test_v3_profile_coordinate_budget_is_counted_by_coordinate(monkeypatch):
    monkeypatch.setattr(ev,'current_kernel_view',lambda:View())
    comp=ev._RecursiveComputer(_payload(max_profile_coordinate_equivalents=100))
    with pytest.raises(ev.S8ScientificBudgetExceeded,match='MAX_PROFILE_COORDINATE_EQUIVALENTS'):
        comp.profile_counts({'n':1})
    assert comp.metrics['exact_relation_profile_family_call_count']==0
    assert comp.metrics['logical_profile_coordinate_count']==0

def test_v3_e1_is_emitted_distinctly(monkeypatch):
    monkeypatch.setattr(ev,'current_kernel_view',lambda:View())
    got=ev.s8_s7_class_recursive_prefix_comparator_evaluator(_payload(max_profile_coordinate_equivalents=100))
    assert got.get('v3_outcome')=='E1_FROZEN_TASK_BUDGET_EXCEEDED'
    assert got['metrics']['pause_reason']=='MAX_PROFILE_COORDINATE_EQUIVALENTS'

def test_v3_module_contains_witness_replay_and_no_full_sweep():
    src=inspect.getsource(v3.run_g6_s8_v3_targeted_probe)
    assert "REPLAY_W1" in src and "REPLAY_W4" in src
    assert "range(8)" in src
    assert "T_NO_SPLIT_IN_FROZEN_SENTINEL_CONTEXTS" in src
    assert "depth_schedule" not in src


def test_v3_stage_runtime_error_filter_is_fail_closed():
    helper=inspect.getsource(v3._stage_pause_from_error)
    runner=inspect.getsource(v3._run_staged_context)
    assert "RESOURCE" in helper and "MEMORY" in helper and "TIMEOUT" in helper and "LEASE" in helper
    assert "return None" in helper
    assert "if pause is not None" in runner and "raise" in runner


def test_v3_gen38_staged_executor_declares_parallel_dedup_and_incremental_prefixes():
    src=inspect.getsource(v3._run_staged_context)
    assert 'run_content_indexed_generation' in src
    assert 'requested_workers=workers' in src
    assert 'grandchild_universe' in src
    assert "for prefix in (1,8,32,124)" in src
    assert 'start=len(previous)' in src
    assert 'prefix_coordinate_reuse=True' in src


def test_v3_profile_extension_uses_batch_and_only_delta(monkeypatch):
    class BatchView(View):
        def call(self,name,*args):
            if name=='EXACT_RELATION_PROFILE_BATCH':
                _tree,_seed,ops=args
                return {'exact_outcome_counts':[7]*len(ops),'metrics':{}}
            return super().call(name,*args)
    monkeypatch.setattr(ev,'current_kernel_view',lambda:BatchView())
    import json
    ident=json.dumps(['I'],separators=(',',':'))
    state=json.dumps({'tree':{'n':1}},separators=(',',':'))
    got=ev.s8_v3_profile_extension_evaluator({'grandchild_identity_json':ident,'grandchild_state_json':state,
        'start_context_index':1,'end_context_index':8,'previous_counts':[3],
        'basis_refs':['D2_BROOM','D2_PATH','D4_BROOM','D4_PATH'],'operator_basis':_ops(),'max_worker_rss_bytes':2**40})
    assert tuple(got['states'][0]['state']['profile_counts'])==(3,7,7,7,7,7,7,7)
    assert got['metrics']['logical_profile_coordinate_count']==7
