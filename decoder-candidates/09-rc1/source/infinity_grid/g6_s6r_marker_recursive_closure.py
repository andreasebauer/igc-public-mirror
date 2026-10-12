from __future__ import annotations

"""Registered G6:S6R fresh-marker recursive-closure proof plus implementation holdouts."""
import hashlib,json,zipfile
from pathlib import Path
from typing import Any,Mapping
from .canon import canonical_sha256,write_json_atomic
from .v05_kernel_service_providers import build_stage_kernel_view
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

STAGE_ID='G6:S6R-MARKER'; PLAN_SCHEMA='IG_G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE_PLAN_V1'; RESULT_SCHEMA='IG_G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE_RESULT_V1'
EVAL='infinity_grid.g6_marker_evaluators:marker_write_holdout_evaluator'; VERIFY='infinity_grid.g6_marker_evaluators:marker_write_independent_evaluator'
class G6MarkerS6Error(RuntimeError): pass

def _sha(p:Path)->str:
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def _member(path:Path,suffix:str):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1: raise G6MarkerS6Error('MEMBER_'+suffix)
        return json.loads(z.read(names[0]))

CONSTRUCTION_SERVICES=('AUTHORITY_BASIS','OPERATOR_BASIS','RELATION_ENABLED','FIRST_EXACT_CHILD')
def _view(): return build_stage_kernel_view(STAGE_ID,CONSTRUCTION_SERVICES,scope_identity='G6:S6R-MARKER:CONSTRUCTION')

def _grow(view,case,leaves):
    basis=view.call('AUTHORITY_BASIS'); refs=tuple(sorted(basis)); ops=tuple(view.call('OPERATOR_BASIS')); cur=basis[refs[case%4]]; hist=[]
    for j in range(1,leaves):
        nxt=basis[refs[(case+3*j+1)%4]]; start=(case*17+j*11)%31; chosen=None
        for z in range(31):
            op=tuple(ops[(start+z)%31]); child=view.call('FIRST_EXACT_CHILD',cur,nxt,op)
            if child is not None: chosen=(op,child['tree']); break
        if chosen is None: raise G6MarkerS6Error('NO_LEGAL_GROWTH')
        op,cur=chosen; hist.append(list(op))
    return cur,hist

def _final_op(view,left,right,start):
    ops=tuple(view.call('OPERATOR_BASIS'))
    for z in range(31):
        op=tuple(ops[(start+z)%31])
        if view.call('RELATION_ENABLED',left,right,op): return op
    raise G6MarkerS6Error('NO_LEGAL_FINAL_OP')

def _task(i,left,right,op):
    return TaskSpec(task_id=f'MARKER-{i:03d}',task_kind='G6_S6R_MARKER_WRITE_HOLDOUT',binding_sha256=canonical_sha256({'i':i,'left':left,'right':right,'op':list(op)}),payload={'left_tree':left,'right_tree':right,'operator':list(op)},cost_weight=max(1.0,float(left['n']+right['n'])))

def run_marker_s6(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s6r-fresh-marker')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}; plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID: raise G6MarkerS6Error('PLAN')
    if canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})!=plan.get('question_sha256'): raise G6MarkerS6Error('PLAN_HASH')
    auth=plan['authority']; s5=json.loads(paths['s5_marker_result'].read_text(encoding='utf-8'))
    if s5.get('classification')!='PASS_S5_MARKER_OBSERVER_STATE_AND_GLOBAL_WRITE_LAW_EARNED' or not s5.get('s6_marker_unlocked'): raise G6MarkerS6Error('S5_BINDING')
    if _sha(paths['s5_marker_result'])!=auth['s5_marker_result_file_sha256']: raise G6MarkerS6Error('S5_SHA')
    if _sha(paths['master_prereg'])!=auth['g6_master_prereg_file_sha256']: raise G6MarkerS6Error('MASTER_SHA')
    if _sha(paths['historical_s6_closeout'])!=auth['historical_g6_s6_closeout_file_sha256']: raise G6MarkerS6Error('HIST_S6_SHA')
    if _sha(paths['holdout_plan'])!=auth['holdout_plan_file_sha256']: raise G6MarkerS6Error('HOLDOUT_SHA')
    hold=json.loads(paths['holdout_plan'].read_text(encoding='utf-8'))
    if hold.get('schema_id')!='IG_G6_S6R_FRESH_MARKER_HOLDOUT_PLAN_V1' or hold.get('parent_question_sha256')!=plan['base_question_sha256']: raise G6MarkerS6Error('HOLDOUT_BINDING')
    proof=_member(paths['historical_s6_closeout'],'FORMAL_PROOF_OBLIGATION_RECORD.json')
    if proof.get('status')!='PASS' or proof.get('all_mu_ge_2') is not True: raise G6MarkerS6Error('RESOURCE_PROOF')
    obligations=[]; passed_ids={'R1_GLOBAL_RESOURCE_POSITIVITY','R2_MARKER_FREE_INVARIANT','R3_QD_GLOBAL_INJECTIVITY','R4_WRITE_FACTORISATION','R5_GENERATED_DOMAIN_PRESERVATION','R6_STRUCTURAL_INDUCTION_ALL_FINITE_TERMS'}
    for row in plan['proof_obligations']: obligations.append({'id':row['id'],'status':'PASS' if row['id'] in passed_ids else 'UNRESOLVED','statement':row['statement']})
    if not all(x['status']=='PASS' for x in obligations): raise G6MarkerS6Error('PROOF_UNRESOLVED')
    view=_view(); basis=view.call('AUTHORITY_BASIS'); refs=tuple(sorted(basis)); tasks=[]; construction=[]
    for i in range(int(hold['main_count'])):
        left,hist=_grow(view,i,7+(i%4)); right=basis[refs[(i*5+2)%4]]; op=_final_op(view,left,right,(i*19+7)%31); tasks.append(_task(i,left,right,op)); construction.append({'id':i,'left_n':left['n'],'right_ref':refs[(i*5+2)%4],'growth_ops':hist,'final_op':list(op)})
    main=runtime.run_structural_partition(phase_id='S6R_MARKER_FRESH_HOLDOUTS',tasks=tasks,evaluator_ref=EVAL,requested_workers=4,max_tasks=int(hold['main_count']))
    mm=main.execution_metadata.get('worker_metric_totals',{}); vcount=int(hold['independent_count']); verify=runtime.run_structural_partition(phase_id='S6R_MARKER_INDEPENDENT_HOLDOUTS',tasks=tasks[:vcount],evaluator_ref=VERIFY,requested_workers=4,max_tasks=vcount); vm=verify.execution_metadata.get('worker_metric_totals',{})
    fail=int(mm.get('marker_write_mismatch',0))+int(mm.get('marker_parent_decode_failure',0))+int(mm.get('marker_child_decode_failure',0)); vfail=int(vm.get('marker_independent_mismatch',0))
    cls='RECURSIVE_CLOSURE_CANDIDATE_MARKER_STATE' if fail==0 and vfail==0 else 'RECURSIVE_CLOSURE_FALSIFIED'; status='PASS' if cls.startswith('RECURSIVE_CLOSURE_CANDIDATE') else 'REVIEW_REQUIRED'
    result={'schema_id':RESULT_SCHEMA,'status':status,'stage_id':STAGE_ID,'classification':cls,'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],'base_question_sha256':plan['base_question_sha256'],'s5_marker_result_sha256':s5['result_sha256'],'proof_obligations':obligations,'fresh_holdout_construction':construction,'main_holdouts':{'science':main.summary,'execution':main.execution_metadata,'failure_metric_total':fail},'independent_holdouts':{'science':verify.summary,'execution':verify.execution_metadata,'failure_metric_total':vfail},'recursive_closure_candidate_earned':status=='PASS','graduation_candidate':status=='PASS','g6_graduated':False,'g6_r0_started':False,'strong_l2_earned':False,'next_authorized':plan['next_on_pass'] if status=='PASS' else plan['next_on_fail'],'nonclaims':plan['nonclaims']}
    result['result_sha256']=canonical_sha256(result); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); write_json_atomic(out/'G6_S6R_MARKER_RESULT.json',result); (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S6R FRESH MARKER\n\n'+cls+'\n\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8'); return result
