from __future__ import annotations

"""Registered G6:S6R L2 proof reduction and sibling-separation adversarial lane.

The mathematical reduction narrows any non-isomorphic global q collision to a
collision between two distinct one-probe children of a smaller common core.
The execution lane then attacks that reduced theorem on a frozen exhaustive
P-only/(0,0) family.  A bounded pass is supporting evidence only.
"""
import hashlib, json
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_kernel_service_providers import build_stage_kernel_view
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

PLAN_SCHEMA='IG_G6_S6R_L2_PROOF_REDUCTION_PREREGISTRATION_V1'
RESULT_SCHEMA='IG_G6_S6R_L2_PROOF_REDUCTION_RESULT_V1'
STAGE_ID='G6:S6R-L2-PROOF-REDUCTION'
EVAL='infinity_grid.g6_s6r_evaluators:l2_sibling_q_evaluator'
SERVICES=('AUTHORITY_BASIS','EXACT_IDENTITY','EXACT_RELATION','OBSERVER_Q')

class G6S6RL2ProofReductionError(RuntimeError): pass

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _load_json(p:Path)->dict[str,Any]:
    obj=json.loads(p.read_text(encoding='utf-8'))
    if type(obj) is not dict: raise G6S6RL2ProofReductionError('L2P_JSON_OBJECT_REQUIRED')
    return obj

def _question_sha(plan:Mapping[str,Any])->str:
    return canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})

def _view():
    return build_stage_kernel_view(STAGE_ID,SERVICES,scope_identity='G6:S6R-L2-PROOF-REDUCTION:CONSTRUCTION')

def _dedup(view, trees):
    out={}
    for t in trees:
        can=view.call('EXACT_IDENTITY',t)
        out.setdefault(repr(can),t)
    return [out[k] for k in sorted(out)]

def _task(level:int, idx:int, core_can, child:dict[str,Any])->TaskSpec:
    payload={'core_identity':core_can,'child_tree':child,'core_leaf_count':int(level),'observer_probe_ref':'D2_PATH','observer_operator':[0,0]}
    return TaskSpec(task_id=f'L2P-{level:02d}-{idx:08d}',task_kind='G6_S6R_L2_SIBLING_Q',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=max(1.0,float(int(child['n']))))

def _capture_collision(view, rows):
    seen={}
    for row in rows:
        core=row['core_identity']; child=row['child_tree']; q=view.call('OBSERVER_Q',child,'D2_PATH',(0,0)); ident=view.call('EXACT_IDENTITY',child)
        key=repr((core,q))
        old=seen.get(key)
        if old is not None and old['child_identity']!=ident:
            return {'core_identity':core,'child_x':old['child_tree'],'child_y':child,'child_x_identity':old['child_identity'],'child_y_identity':ident,'common_q':q}
        seen[key]={'child_tree':child,'child_identity':ident}
    return None

def run_l2_proof_reduction(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,
                           accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s6r-l2-proof-reduction')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=_load_json(Path(plan_path).resolve(strict=True))
    if plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID: raise G6S6RL2ProofReductionError('L2P_PLAN_SCHEMA_STAGE')
    if _question_sha(plan)!=plan.get('question_sha256'): raise G6S6RL2ProofReductionError('L2P_PLAN_HASH')
    auth=plan.get('authority') or {}
    if accepted_source_sha256!=auth.get('accepted_decoder_source_sha256'): raise G6S6RL2ProofReductionError('L2P_SOURCE_AUTHORITY')
    if _sha_file(paths['l2_result'])!=auth.get('l2_repair_result_file_sha256'): raise G6S6RL2ProofReductionError('L2P_L2_RESULT_BINDING')
    if _sha_file(paths['crw0_result'])!=auth.get('crw0_result_file_sha256'): raise G6S6RL2ProofReductionError('L2P_CRW0_RESULT_BINDING')
    l2=_load_json(paths['l2_result']); crw0=_load_json(paths['crw0_result'])
    if l2.get('classification')!=auth.get('l2_repair_classification') or l2.get('strong_l2_proved') is not False: raise G6S6RL2ProofReductionError('L2P_L2_AUTHORITY')
    if crw0.get('classification')!=auth.get('crw0_classification') or int(crw0.get('decode_failure_count',-1))!=0: raise G6S6RL2ProofReductionError('L2P_CRW0_AUTHORITY')
    red=plan.get('proof_reduction') or {}
    required={'R1_COMMON_CANDIDATE_IFF_Q_INCLUSION','R2_NONISOMORPHIC_Q_COLLISION_HAS_SMALLER_COMMON_CORE','R3_SIBLING_SEPARATION_SUFFICES','base_case'}
    if set(red)!=required: raise G6S6RL2ProofReductionError('L2P_REDUCTION_BINDING')
    if (plan.get('new_critical_theorem') or {}).get('id')!='L2P_SIBLING_SEPARATION': raise G6S6RL2ProofReductionError('L2P_TARGET')
    lane=plan.get('bounded_adversarial_lane') or {}
    if lane.get('family_id')!='P_ONLY_00_SIBLING_COLLISION_CENSUS' or lane.get('core_seed')!='D2_PATH' or lane.get('growth_operator')!=[0,0]: raise G6S6RL2ProofReductionError('L2P_LANE')
    max_core=int(lane.get('max_core_probe_leaf_blocks',0))
    if max_core<1: raise G6S6RL2ProofReductionError('L2P_DEPTH')

    view=_view(); basis=view.call('AUTHORITY_BASIS'); probe=basis['D2_PATH']; current=[probe]
    levels=[]; collision=None; total_cores=0; total_siblings=0
    for level in range(1,max_core+1):
        current=_dedup(view,current); total_cores+=len(current)
        rows=[]; tasks=[]; idx=0
        for core in current:
            core_can=view.call('EXACT_IDENTITY',core)
            rel=view.call('EXACT_RELATION',core,probe,(0,0))
            # exact relation already deduplicates child canons
            for child in rel['children']:
                row={'core_identity':core_can,'child_tree':child}
                rows.append(row); tasks.append(_task(level,idx,core_can,child)); idx+=1
        total_siblings+=len(tasks)
        phase=runtime.run_structural_partition(phase_id=f'L2P_SIBLING_LEVEL_{level}',tasks=tasks,evaluator_ref=EVAL,requested_workers=4,max_tasks=len(tasks))
        class_count=int(phase.summary.get('class_count',-1)); task_count=int(phase.summary.get('task_count',len(tasks)))
        has_collision=class_count<task_count
        levels.append({'core_leaf_count':level,'core_count':len(current),'sibling_child_count':len(tasks),'q_signature_class_count':class_count,'collision_detected':has_collision,'science':phase.summary,'execution':phase.execution_metadata})
        if has_collision:
            collision=_capture_collision(view,rows)
            if collision is None: raise G6S6RL2ProofReductionError('L2P_COLLISION_WITHOUT_WITNESS')
            break
        if level<max_core:
            nxt=[]
            for core in current:
                rel=view.call('EXACT_RELATION',core,probe,(0,0)); nxt.extend(rel['children'])
            current=nxt

    reduction=[
      {'id':'R1_COMMON_CANDIDATE_IFF_Q_INCLUSION','status':'PASS','basis':'EXACT_PROBE_DECOMPOSITION_DEFINITION'},
      {'id':'R2_NONISOMORPHIC_Q_COLLISION_HAS_SMALLER_COMMON_CORE','status':'PASS','basis':'TREE_EDGE_CUT_LAMINARITY_EQUAL_SIZE_5_AND_N_GT_5_PLUS_CRW0_BASE'},
      {'id':'R3_SIBLING_SEPARATION_SUFFICES','status':'PASS','basis':'MINIMAL_COUNTEREXAMPLE_REDUCTION_R2'},
    ]
    if collision is not None:
        classification='L2_PROOF_REDUCTION_SIBLING_COUNTEREXAMPLE_FOUND'; next_auth=plan['next_on_counterexample']
    else:
        classification='L2_REDUCED_TO_SIBLING_SEPARATION_NO_COUNTEREXAMPLE_IN_REGISTERED_SCOPE'; next_auth=plan['next_on_no_counterexample']
    result={'schema_id':RESULT_SCHEMA,'status':'REVIEW_REQUIRED','stage_id':STAGE_ID,'classification':classification,
      'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],
      'proof_reduction_established':True,'proof_reduction_obligations':reduction,'new_critical_theorem':plan['new_critical_theorem'],
      'bounded_adversarial_lane':lane,'levels':levels,'total_cores_checked':total_cores,'total_sibling_children_checked':total_siblings,
      'counterexample':collision,'global_q_injectivity_earned':False,'global_write_congruence_earned':False,
      'sibling_separation_theorem_earned':False,'g6_public_layer_graduated':False,'g6_r0_started':False,
      'next_authorized':next_auth,'nonclaims':plan['nonclaims']}
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out/'G6_S6R_L2_PROOF_REDUCTION_RESULT.json',result)
    (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S6R L2 PROOF REDUCTION\n\n'+classification+'\n\nUniversal sibling theorem earned: false\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8')
    return result
