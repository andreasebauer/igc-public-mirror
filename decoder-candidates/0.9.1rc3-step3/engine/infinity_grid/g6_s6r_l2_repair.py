from __future__ import annotations

"""Registered bounded falsification lane for G6:S6R L2 strong common-false-parent exclusion.

This operation is exact but deliberately bounded.  A counterexample refutes the
strong L2 lemma.  Absence of a counterexample in the frozen P-only/(0,0) family
is supporting evidence only and cannot promote a universal theorem.
"""
import hashlib, json
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_kernel_service_providers import build_stage_kernel_view
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

PLAN_SCHEMA='IG_G6_S6R_L2_REPAIR_PREREGISTRATION_V1'
RESULT_SCHEMA='IG_G6_S6R_L2_REPAIR_RESULT_V1'
STAGE_ID='G6:S6R-L2-REPAIR'
EVAL='infinity_grid.g6_s6r_evaluators:l2_common_parent_uniqueness_evaluator'
CONSTRUCTION_SERVICES=('AUTHORITY_BASIS','EXACT_IDENTITY','EXACT_RELATION','OBSERVER_Q','OBSERVER_DECODE')

class G6S6RL2RepairError(RuntimeError): pass

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _load_json(p:Path)->dict[str,Any]:
    obj=json.loads(p.read_text(encoding='utf-8'))
    if type(obj) is not dict: raise G6S6RL2RepairError('L2_JSON_OBJECT_REQUIRED')
    return obj

def _question_sha(plan:Mapping[str,Any])->str:
    return canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})

def _view():
    return build_stage_kernel_view(STAGE_ID,CONSTRUCTION_SERVICES,scope_identity='G6:S6R-L2-REPAIR:CONSTRUCTION')

def _dedup(view, trees):
    out={}
    for t in trees:
        can=view.call('EXACT_IDENTITY',t)
        out.setdefault(repr(can),t)
    return [out[k] for k in sorted(out)]

def _task(leaf_count:int, idx:int, state:dict[str,Any])->TaskSpec:
    return TaskSpec(task_id=f'L2-{leaf_count:02d}-{idx:06d}',task_kind='G6_S6R_L2_COMMON_PARENT_UNIQUENESS',
        binding_sha256=canonical_sha256({'leaf_count':leaf_count,'state':state}),
        payload={'state_tree':state,'leaf_count':leaf_count,'observer_probe_ref':'D2_PATH','observer_operator':[0,0]},
        cost_weight=max(1.0,float(int(state['n']))))

def _capture_counterexample(view,state,leaf_count:int)->dict[str,Any]|None:
    q=view.call('OBSERVER_Q',state,'D2_PATH',(0,0)); dec=view.call('OBSERVER_DECODE',q,'D2_PATH',(0,0)); ident=view.call('EXACT_IDENTITY',state)
    candidates=tuple(dec.get('candidates') or ())
    if int(dec.get('candidate_count',-1))==1 and candidates and candidates[0]==ident: return None
    return {'leaf_count':int(leaf_count),'exact_X':state,'q_X':q,'candidate_count':int(dec.get('candidate_count',-1)),'all_common_parent_candidates':candidates,'exact_X_identity':ident,'decode_status':dec.get('status')}

def run_l2_repair(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,
                  accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s6r-l2-repair')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=_load_json(Path(plan_path).resolve(strict=True))
    if plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID: raise G6S6RL2RepairError('L2_PLAN_SCHEMA_STAGE')
    if _question_sha(plan)!=plan.get('question_sha256'): raise G6S6RL2RepairError('L2_PLAN_HASH')
    auth=plan.get('parent_authority') or {}
    if accepted_source_sha256!=auth.get('accepted_decoder_source_sha256'): raise G6S6RL2RepairError('L2_SOURCE_AUTHORITY')
    if _sha_file(paths['global_theorem_result'])!=auth.get('global_theorem_result_file_sha256'): raise G6S6RL2RepairError('L2_GLOBAL_RESULT_BINDING')
    prior=_load_json(paths['global_theorem_result'])
    if prior.get('classification')!='GLOBAL_Q_CONGRUENCE_THEOREM_UNRESOLVED': raise G6S6RL2RepairError('L2_GLOBAL_RESULT_AUTHORITY')
    target=plan.get('target') or {}
    if target.get('id')!='L2_COMMON_FALSE_PARENT_EXCLUSION': raise G6S6RL2RepairError('L2_TARGET')
    lane=plan.get('falsification_lane') or {}
    if lane.get('family_id')!='EXHAUSTIVE_D2_PATH_ONLY_00_SEQUENTIAL_GENERATED_FAMILY' or lane.get('seed')!='D2_PATH' or lane.get('growth_operator')!=[0,0]: raise G6S6RL2RepairError('L2_LANE')
    max_leaf=int(lane.get('max_probe_leaf_blocks',0))
    if max_leaf<1: raise G6S6RL2RepairError('L2_DEPTH')

    view=_view(); basis=view.call('AUTHORITY_BASIS'); probe=basis['D2_PATH']; current=[probe]
    levels=[]; total_states=0; counterexample=None
    for leaf_count in range(1,max_leaf+1):
        current=_dedup(view,current)
        tasks=[_task(leaf_count,i,t) for i,t in enumerate(current)]
        phase=runtime.run_structural_partition(phase_id=f'L2_EXHAUSTIVE_LEVEL_{leaf_count}',tasks=tasks,evaluator_ref=EVAL,requested_workers=4,max_tasks=len(tasks))
        metrics=phase.execution_metadata.get('worker_metric_totals',{})
        bad=int(metrics.get('l2_bad_candidate_state',0))
        levels.append({'leaf_count':leaf_count,'state_count':len(current),'science':phase.summary,'execution':phase.execution_metadata,'bad_candidate_state_count':bad})
        total_states+=len(current)
        if bad:
            for t in current:
                counterexample=_capture_counterexample(view,t,leaf_count)
                if counterexample is not None: break
            if counterexample is None: raise G6S6RL2RepairError('L2_BAD_METRIC_WITHOUT_WITNESS')
            break
        if leaf_count<max_leaf:
            nxt=[]
            for t in current:
                rel=view.call('EXACT_RELATION',t,probe,(0,0))
                nxt.extend(rel['children'])
            current=nxt

    if counterexample is not None:
        classification='L2_STRONG_REFUTED_WITH_COUNTEREXAMPLE'; status='REVIEW_REQUIRED'; next_auth=plan['next_on_counterexample']
    else:
        classification='L2_NO_COUNTEREXAMPLE_IN_EXHAUSTIVE_P_ONLY_00_SCOPE_CONTINUE_PROOF'; status='REVIEW_REQUIRED'; next_auth=plan['next_on_no_counterexample']
    result={'schema_id':RESULT_SCHEMA,'status':status,'stage_id':STAGE_ID,'classification':classification,
      'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],
      'target':target,'falsification_lane':lane,'levels':levels,'total_distinct_states_checked':total_states,'counterexample':counterexample,
      'strong_l2_proved':False,'global_q_injectivity_earned':False,'g6_public_layer_graduated':False,'g6_r0_started':False,
      'finite_success_effect':plan['proof_discipline']['finite_success_effect'],'next_authorized':next_auth,'nonclaims':plan['nonclaims']}
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); write_json_atomic(out/'G6_S6R_L2_REPAIR_RESULT.json',result)
    (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S6R L2 REPAIR\n\n'+classification+'\n\nStrong L2 proved: false\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8')
    return result
