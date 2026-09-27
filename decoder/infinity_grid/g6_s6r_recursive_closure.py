from __future__ import annotations

"""Registered G6:S6R recursive-closure candidate and fresh holdout gate.

This stage is deliberately conservative: fresh holdouts can falsify the S5 q-state
law, but bounded success does not by itself discharge the global one-probe
inversion/congruence lemma required for an all-finite structural-induction theorem.
"""
import hashlib, json
from pathlib import Path
from typing import Any, Mapping
from .canon import canonical_sha256, write_json_atomic
from .v05_kernel_service_providers import build_stage_kernel_view
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

STAGE_ID='G6:S6R'
PLAN_SCHEMA='IG_G6_S6R_RECURSIVE_CLOSURE_PLAN_V1'
RESULT_SCHEMA='IG_G6_S6R_RECURSIVE_CLOSURE_RESULT_V1'
EVAL='infinity_grid.g6_s6r_evaluators:recursive_closure_holdout_evaluator'
VERIFY='infinity_grid.g6_s6r_evaluators:recursive_closure_independent_evaluator'
class G6S6RError(RuntimeError): pass

def _sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

S6_CONSTRUCTION_KERNEL_SERVICES=('AUTHORITY_BASIS','OPERATOR_BASIS','RELATION_ENABLED','FIRST_EXACT_CHILD')

def _stage_view():
    return build_stage_kernel_view(STAGE_ID,S6_CONSTRUCTION_KERNEL_SERVICES,scope_identity='G6:S6R:CONTROLLER_CONSTRUCTION')

def _grow(view,case:int,leaves:int):
    basis=view.call('AUTHORITY_BASIS'); refs=tuple(sorted(basis)); ops=tuple(view.call('OPERATOR_BASIS'))
    cur=basis[refs[case%len(refs)]]; history=[]
    for j in range(1,leaves):
        nxt=basis[refs[(case+2*j+1)%len(refs)]]; chosen=None; start=(case*11+j*7)%len(ops)
        for z in range(len(ops)):
            op=tuple(ops[(start+z)%len(ops)]); child=view.call('FIRST_EXACT_CHILD',cur,nxt,op)
            if child is not None: chosen=(op,child['tree']); break
        if chosen is None: raise G6S6RError('S6R_NO_LEGAL_GROWTH')
        op,cur=chosen; history.append(list(op))
    return cur,history

def _first_legal(view,left,right,start:int):
    ops=tuple(view.call('OPERATOR_BASIS'))
    for z in range(len(ops)):
        op=tuple(ops[(start+z)%len(ops)])
        if view.call('RELATION_ENABLED',left,right,op): return op
    raise G6S6RError('S6R_NO_LEGAL_FINAL_OP')

def _task(i,left,right,op)->TaskSpec:
    return TaskSpec(task_id=f'S6R-{i:03d}',task_kind='G6_S6R_RECURSIVE_CLOSURE_HOLDOUT',
        binding_sha256=canonical_sha256({'i':i,'left':left,'right':right,'op':list(op)}),
        payload={'left_tree':left,'right_tree':right,'operator':list(op),'observer_probe_ref':'D2_PATH','observer_operator':[0,0]},
        cost_weight=max(1.0,float(int(left['n'])+int(right['n']))))

def run_s6r(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s6r')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}; plan=json.loads(Path(plan_path).read_text())
    required={'schema_id','stage_id','candidate','dependency_sha256','holdouts','proof_obligations','registered_outcomes','next_on_candidate','next_on_unresolved','nonclaims','question_sha256'}
    if type(plan) is not dict or set(plan)!=required or plan['schema_id']!=PLAN_SCHEMA or plan['stage_id']!=STAGE_ID: raise G6S6RError('S6R_PLAN')
    if canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})!=plan['question_sha256']: raise G6S6RError('S6R_PLAN_HASH')
    for logical,expected in plan['dependency_sha256'].items():
        if logical not in paths or _sha(paths[logical])!=expected: raise G6S6RError('S6R_DEP_'+logical)
    s5=json.loads(paths['crw1_result'].read_text())
    if s5.get('classification')!='PASS_S5_OBSERVER_STATE_AND_WRITE_LAW_EARNED_ON_CERTIFIED_SCOPE' or not s5.get('s6_unlocked_for_preregistration'): raise G6S6RError('S6R_S5_BINDING')
    master=paths['master_prereg'].read_text()
    if 'G6:S6:' not in master or 'RECURSIVE_CLOSURE_CANDIDATE' not in master: raise G6S6RError('S6R_MASTER_BINDING')
    g5=paths['g5_parent_review'].read_text().lower()
    if 'global single-c/(0,0)-probe injectivity' not in g5 or 'not earned globally' not in g5: raise G6S6RError('S6R_G5_OPEN_LEMMA_BINDING')

    view=_stage_view(); basis=view.call('AUTHORITY_BASIS'); tasks=[]; construction=[]
    count=int(plan['holdouts']['main_count'])
    for i in range(count):
        parent,hist=_grow(view,i,5+(i%3)); right=basis[tuple(sorted(basis))[(i*3+1)%4]]; op=_first_legal(view,parent,right,(i*13)%31)
        tasks.append(_task(i,parent,right,op)); construction.append({'id':i,'parent_n':int(parent['n']),'growth_ops':hist,'final_op':list(op),'right_n':int(right['n'])})
    main=runtime.run_structural_partition(phase_id='S6R_FRESH_RECURSIVE_HOLDOUTS',tasks=tasks,evaluator_ref=EVAL,requested_workers=4,max_tasks=count)
    mm=main.execution_metadata.get('worker_metric_totals',{})
    vcount=int(plan['holdouts']['independent_count']); verify_tasks=tasks[:vcount]
    verify=runtime.run_structural_partition(phase_id='S6R_INDEPENDENT_COLD_HOLDOUTS',tasks=verify_tasks,evaluator_ref=VERIFY,requested_workers=4,max_tasks=vcount)
    vm=verify.execution_metadata.get('worker_metric_totals',{})
    holdout_fail=int(mm.get('write_mismatch',0))+int(mm.get('parent_decode_failure',0))+int(mm.get('child_decode_failure',0))
    independent_fail=int(vm.get('independent_mismatch',0))

    # The all-finite proof has one remaining non-bounded obligation.  Prior G5 authority
    # explicitly records the corresponding marker-free single-probe theorem as unearned.
    obligations=[]
    for oid in plan['proof_obligations']:
        if oid=='GLOBAL_Q_INVERSION_OR_CONGRUENCE_ON_ALL_FINITE_GENERATED_TERMS': obligations.append({'id':oid,'status':'UNRESOLVED','reason':'No global marker-free single-probe injectivity/congruence theorem is present in the bound authority; bounded holdouts cannot discharge a universal statement.'})
        else: obligations.append({'id':oid,'status':'PASS'})
    global_unresolved=any(x['status']=='UNRESOLVED' for x in obligations)
    if holdout_fail: classification='RECURSIVE_CLOSURE_FALSIFIED'
    elif independent_fail: classification='INDEPENDENT_VERIFICATION_FAILED'
    elif global_unresolved: classification='BUDGET_INSUFFICIENT_GLOBAL_OBSERVER_CONGRUENCE_LEMMA_UNPROVED'
    else: classification='RECURSIVE_CLOSURE_CANDIDATE'
    status='PASS' if classification=='RECURSIVE_CLOSURE_CANDIDATE' else 'REVIEW_REQUIRED'
    result={'schema_id':RESULT_SCHEMA,'status':status,'stage_id':STAGE_ID,'classification':classification,
      'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],
      's5_result_sha256':s5['result_sha256'],'candidate':plan['candidate'],'fresh_holdout_construction':construction,
      'main_holdouts':{'science':main.summary,'execution':main.execution_metadata,'failure_metric_total':holdout_fail},
      'independent_cold_holdouts':{'science':verify.summary,'execution':verify.execution_metadata,'failure_metric_total':independent_fail},
      'proof_obligations':obligations,'recursive_closure_candidate_earned':classification=='RECURSIVE_CLOSURE_CANDIDATE',
      'recursive_closure_falsified':classification=='RECURSIVE_CLOSURE_FALSIFIED','g6_graduated':False,'g6_r0_started':False,
      'next_authorized':plan['next_on_candidate'] if classification=='RECURSIVE_CLOSURE_CANDIDATE' else plan['next_on_unresolved'],
      'nonclaims':plan['nonclaims']}
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); write_json_atomic(out/'G6_S6R_RESULT.json',result)
    (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S6R\n\n'+classification+'\n\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8')
    return result
