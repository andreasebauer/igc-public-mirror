from __future__ import annotations

"""Long-lived Decoder controller supervisor/event loop for C5 and later.

External actors only write passive request data.  The supervisor owns controller
child lifecycle; substantive work begins only inside the child event loop.
"""

import hashlib, json, os, shutil, sys, time
from pathlib import Path
from typing import Any
from .canon import canonical_bytes, canonical_sha256, write_json_atomic
from .v05_execution_authority import source_tree_digest
from .v05_engineering_worker import engineering_source_tree_digest
from .v05_origin_guard import _supervisor_controller_event_scope, require_controller_execution_origin
from .v05_passive_intake import ingest_passive_request

RESTART_TO_ACTIVE_SOURCE=75
ENGINEERING_JOB='DECODER.ENGINEERING.SOURCE_CHANGE'
SCIENCE_JOB='DECODER.G6.SCIENCE'
C6_RECOVERY_JOB='DECODER.C6.RECOVERY'

class ControllerLoopError(RuntimeError): pass

def _identity_is_mapped(map_path:Path,identity:int)->bool:
    rows=map_path.read_text(encoding='ascii').splitlines()
    for row in rows:
        fields=row.split()
        if len(fields)!=3: raise ControllerLoopError('SOURCE_TRANSITION_ID_MAP')
        inside,_,length=(int(x) for x in fields)
        if inside<=int(identity)<inside+length:return True
    return False

def _source_transition_worker_identity()->tuple[int|None,int|None]:
    """Use the unprivileged identity only when this Linux namespace maps it."""
    if os.geteuid()!=0:return None,None
    if not sys.platform.startswith('linux'):return 65534,65534
    try:
        mapped=_identity_is_mapped(Path('/proc/self/uid_map'),65534) and _identity_is_mapped(Path('/proc/self/gid_map'),65534)
    except (OSError,ValueError,ControllerLoopError):
        # Preserve the established fail-closed choice when mapping evidence is unavailable.
        return 65534,65534
    return (65534,65534) if mapped else (None,None)

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def _source_ids(source:Path)->tuple[str,str]:
    return engineering_source_tree_digest(source),source_tree_digest(source/'infinity_grid')

def _artifact(root:Path,sha:str)->Path:
    found=[]
    for p in (root/'intake'/'artifacts').glob(sha+'*'):
        if p.is_file() and _sha_file(p)==sha: found.append(p)
    if len(found)!=1: raise ControllerLoopError('PASSIVE_ARTIFACT_RESOLUTION')
    return found[0]

def _registry(source:Path)->dict[str,dict[str,Any]]:
    impl=_sha_file(source/'infinity_grid/v05_controller_event_loop.py')
    obs_impl=_sha_file(source/'infinity_grid/g6_s5r_observer_continuity.py')
    wider_impl=_sha_file(source/'infinity_grid/g6_s5r_wider_feature_search.py')
    crw_impl=_sha_file(source/'infinity_grid/g6_s5r_compositional_read_write.py')
    crw_eval=_sha_file(source/'infinity_grid/g6_s5r_crw_evaluators.py')
    crw_kernel=_sha_file(source/'infinity_grid/g6_s5r_crw_kernel.py')
    crw1_impl=_sha_file(source/'infinity_grid/g6_s5r_crw1.py')
    s6r_impl=_sha_file(source/'infinity_grid/g6_s6r_recursive_closure.py')
    s6r_eval=_sha_file(source/'infinity_grid/g6_s6r_evaluators.py')
    global_impl=_sha_file(source/'infinity_grid/g6_s6r_global_observer_congruence.py')
    l2_impl=_sha_file(source/'infinity_grid/g6_s6r_l2_repair.py')
    l2p_impl=_sha_file(source/'infinity_grid/g6_s6r_l2_proof_reduction.py')
    marker_kernel=_sha_file(source/'infinity_grid/g6_fresh_marker_kernel.py')
    marker_eval=_sha_file(source/'infinity_grid/g6_marker_evaluators.py')
    marker_s5=_sha_file(source/'infinity_grid/g6_s5r_marker_observer.py')
    marker_s6=_sha_file(source/'infinity_grid/g6_s6r_marker_recursive_closure.py')
    g6_r0_impl=_sha_file(source/'infinity_grid/g6_r0_post_graduation_fiber.py')
    g6_s7_impl=_sha_file(source/'infinity_grid/g6_s7_ordinary_future_quotient.py')
    g6_s7_eval=_sha_file(source/'infinity_grid/g6_s7_evaluators.py')
    g6_s7_d2_impl=_sha_file(source/'infinity_grid/g6_s7_depth2_completion.py')
    g6_s7_a6_prov_impl=_sha_file(source/'infinity_grid/g6_s7_a6_reuse_provenance.py')
    g6_s8_impl=_sha_file(source/'infinity_grid/g6_s8_intrinsic_descriptor.py')
    g6_s8_eval=_sha_file(source/'infinity_grid/g6_s8_evaluators.py')
    g6_s8_v3_impl=_sha_file(source/'infinity_grid/g6_s8_v3_targeted_probe.py')
    science_impl=hashlib.sha256((obs_impl+'|'+wider_impl+'|'+crw_impl+'|'+crw_eval+'|'+crw_kernel+'|'+crw1_impl+'|'+s6r_impl+'|'+s6r_eval+'|'+global_impl+'|'+l2_impl+'|'+l2p_impl+'|'+marker_kernel+'|'+marker_eval+'|'+marker_s5+'|'+marker_s6+'|'+g6_r0_impl+'|'+g6_s7_impl+'|'+g6_s7_eval+'|'+g6_s7_d2_impl+'|'+g6_s7_a6_prov_impl+'|'+g6_s8_impl+'|'+g6_s8_eval+'|'+g6_s8_v3_impl).encode('ascii')).hexdigest()
    origin_impl=_sha_file(source/'infinity_grid/v05_origin_guard.py')
    c6_impl=hashlib.sha256((impl+'|'+origin_impl).encode('ascii')).hexdigest()
    return {
      ENGINEERING_JOB:{'allowed_operations':['APPLY_SOURCE_CHANGE','APPLY_SOURCE_CHANGE_AND_ACTIVATE'],'registration_sha256':hashlib.sha256(b'IG-CONTROLLER-SOURCE-CHANGE-BUILTIN-V2').hexdigest(),'implementation_sha256':impl},
      SCIENCE_JOB:{'allowed_operations':['G6_S5R_OBSERVER_CONTINUITY','G6_S5R_WIDER_FEATURE_SEARCH','G6_S5R_COMPOSITIONAL_READ_WRITE','G6_S5R_COMPOSITIONAL_READ_WRITE_CRW1','G6_S6R_RECURSIVE_CLOSURE','G6_S6R_GLOBAL_OBSERVER_CONGRUENCE','G6_S6R_L2_REPAIR','G6_S6R_L2_PROOF_REDUCTION','G6_S5R_FRESH_MARKER_OBSERVER','G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE','G6_R0_POST_GRADUATION_FIBER','G6_S7_ORDINARY_FUTURE_QUOTIENT','G6_S7_A6_REUSE_PROVENANCE_VERIFY','G6_S7_DEPTH2_COMPLETION','G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE','G6_S8_V3_TARGETED_PROBE'],'registration_sha256':hashlib.sha256(b'IG-G6-SCIENCE-V5-S8-V3').hexdigest(),'implementation_sha256':science_impl},
      C6_RECOVERY_JOB:{'allowed_operations':['C6_CONTEXT_PROBE'],'registration_sha256':hashlib.sha256(b'IG-C6-CONTEXT-PROBE-V1').hexdigest(),'implementation_sha256':c6_impl},
    }

def _snapshot(runtime:Path,obj:dict[str,Any],name:str='CONTROLLER_STATUS.json')->None:
    p=runtime/'status'/name; p.parent.mkdir(parents=True,exist_ok=True); write_json_atomic(p,obj); os.chmod(p,0o444)

def _start_controller_attempt(runtime:Path,request_id:str,request:dict[str,Any],source_sha256:str,registration_sha256:str)->tuple[Path,dict[str,Any]]:
    attempts=runtime/'attempts'/request_id; attempts.mkdir(parents=True,exist_ok=True)
    attempt_no=len(list(attempts.glob('*.json')))+1
    attempt_path=attempts/f'{attempt_no:06d}.json'
    running={'status':'RUNNING','request_id':request_id,'job_id':request['registered_job_id'],
             'pid':os.getpid(),'attempt_id':request_id+':'+f'{attempt_no:06d}',
             'registration_sha256':registration_sha256,
             'operation':request['requested_operation_id'],'output_root':str(runtime.resolve()),
             'source_sha256':source_sha256,'started_unix':time.time()}
    write_json_atomic(attempt_path,running)
    binding=dict(running,workspace=str(runtime.resolve()),attempt_path=str(attempt_path.resolve()),
                 attempt_sha256=canonical_sha256(running))
    return attempt_path,binding

def _reconcile_workspace_attempts(attempts:Path, request_id:str, job_id:str,
                                  source_sha256:str, registration_sha256:str)->None:
    """Close interrupted records only while the caller owns workspace/work locks.

    PID values are deliberately not liveness evidence across restored namespaces.
    Retain the entire original record and its digest in the recovery transition.
    Validate every pending transition before writing any of them.
    """
    pending=[]
    for path in sorted(attempts.glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('status')!='RUNNING': continue
        expected={'request_id':request_id,'job_id':job_id,
                  'source_sha256':source_sha256,'registration_sha256':registration_sha256,
                  'attempt_id':request_id+':'+path.stem}
        if any(record.get(k)!=v for k,v in expected.items()):
            raise ControllerLoopError('INTERRUPTED_ATTEMPT_BINDING_MISMATCH')
        pending.append((path,record))
    for path,record in pending:
        recovered=dict(record,status='INTERRUPTED',finished_unix=time.time(),
                       reason='New registered attempt acquired exclusive workspace and work locks',
                       recovery={'prior_record':record,'prior_record_sha256':canonical_sha256(record),
                                 'basis':'EXCLUSIVE_WORKSPACE_AND_WORK_CLAIM'})
        write_json_atomic(path,recovered)


def _finish_controller_attempt(attempt_path:Path,status:str,reason:str|None=None)->None:
    record=json.loads(attempt_path.read_text(encoding='utf-8'))
    if record.get('status')!='RUNNING': raise ControllerLoopError('CONTROLLER_ATTEMPT_NOT_RUNNING')
    record['status']=status; record['finished_unix']=time.time()
    if reason is not None: record['reason']=reason
    write_json_atomic(attempt_path,record)

def _load_plan(record:dict[str,Any],runtime:Path)->tuple[dict[str,Any],Path]:
    amap={x['logical_name']:x['sha256'] for x in record['input_artifacts']}
    if set(amap)!={'source_transition_plan','source_overlay'}: raise ControllerLoopError('SOURCE_TRANSITION_ARTIFACT_SET')
    plan_path=_artifact(runtime,amap['source_transition_plan']); overlay=_artifact(runtime,amap['source_overlay'])
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    required={'schema_id','job_id','operation','parent_source_sha256','parent_package_sha256','expected_candidate_source_sha256','expected_candidate_package_sha256','overlay_sha256','validation_groups'}
    if type(plan) is not dict or set(plan)!=required or plan['schema_id']!='IG_DECODER_CONTROLLER_SOURCE_TRANSITION_REGISTRATION_V1': raise ControllerLoopError('SOURCE_TRANSITION_PLAN')
    if plan['overlay_sha256']!=amap['source_overlay'] or _sha_file(overlay)!=plan['overlay_sha256']: raise ControllerLoopError('SOURCE_TRANSITION_OVERLAY_BINDING')
    return plan,overlay

def _handle_source_change(record:dict[str,Any],runtime:Path,source:Path,*,activate:bool)->dict[str,Any]:
    require_controller_execution_origin('controller-source-change')
    from .v05_engineering_jobs import run_controller_registered_source_transition
    plan,overlay=_load_plan(record,runtime)
    cur_source,cur_pkg=_source_ids(source)
    if plan['parent_source_sha256']!=cur_source or plan['parent_package_sha256']!=cur_pkg: raise ControllerLoopError('SOURCE_TRANSITION_PARENT_MISMATCH')
    worker_uid,worker_gid=_source_transition_worker_identity()
    transition=run_controller_registered_source_transition(runtime_root=runtime/'source_transitions'/record['internal_execution_id'],registration=plan,overlay_path=overlay,worker_uid=worker_uid,worker_gid=worker_gid,workers=max(1,min(4,os.cpu_count() or 1)))
    out={'schema_id':'IG_DECODER_CONTROLLER_SOURCE_CHANGE_RESULT_V1','status':'PASS','internal_execution_id':record['internal_execution_id'],'source_transition':transition,'g6_science_effect':'NONE'}
    if activate:
        from .v05_c5_acceptance import run_c5_acceptance
        acceptance=run_c5_acceptance(final_source_root=transition['candidate_source_path'],runtime_root=runtime,final_source_sha256=transition['candidate_source_sha256'],source_transition=transition)
        out['c5_acceptance']=acceptance
        final=runtime/'final'; final.mkdir(parents=True,exist_ok=True)
        write_json_atomic(final/'C5_LAST_ACCEPTANCE.json',acceptance); os.chmod(final/'C5_LAST_ACCEPTANCE.json',0o444)
        if acceptance['status']!='PASS': raise ControllerLoopError('C5_ACCEPTANCE_FAILED')
        write_json_atomic(final/'C5_FINAL_ACCEPTANCE.json',acceptance); os.chmod(final/'C5_FINAL_ACCEPTANCE.json',0o444)
        from .v05_c6_recovery import _atomic_bytes, stage_fixed_installation_after_acceptance
        fixed_install=stage_fixed_installation_after_acceptance(runtime_root=runtime,transition=transition,acceptance_path=final/'C5_FINAL_ACCEPTANCE.json')
        out['fixed_install_staging']=fixed_install
        _atomic_bytes(final/'ACTIVE_SOURCE_PATH.txt',(transition['candidate_source_path']+'\n').encode('utf-8'),0o444)
        _atomic_bytes(final/'ACTIVE_SOURCE_SHA256.txt',(transition['candidate_source_sha256']+'\n').encode('ascii'),0o444)
        out['restart_required']=True
    return out

def _handle_g6_science(record:dict[str,Any],runtime:Path,source:Path)->dict[str,Any]:
    require_controller_execution_origin('controller-g6-science')
    amap={x['logical_name']:x['sha256'] for x in record['input_artifacts']}
    op=record['requested_operation_id']
    if op=='G6_S5R_OBSERVER_CONTINUITY':
        expected={'science_plan','master_prereg','g5_parent_review','s0_s1_closeout','s1r_closeout','s2r_closeout','s3r_closeout','s4r_closeout','s5_closeout','e3_recovered_fixture','audit'}
        if set(amap)!=expected: raise ControllerLoopError('G6_SCIENCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        from .g6_s5r_observer_continuity import run_observer_continuity_repair
        outdir=runtime/'science'/record['internal_execution_id']
        result=run_observer_continuity_repair(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'])
    elif op=='G6_S5R_WIDER_FEATURE_SEARCH':
        expected={'science_plan','master_prereg','g5_parent_review','s0_s1_closeout','s1r_closeout','s2r_closeout','s3r_closeout','s4r_closeout','s5_closeout','e3_recovered_fixture','continuity_result','audit'}
        if set(amap)!=expected: raise ControllerLoopError('G6_WIDER_SCIENCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evaluator_ref='infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator'
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id=plan['search_id'],stage_id='G6:S5R-WIDER',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s5r.wider-feature-search',handler_ref='infinity_grid.g6_s5r_wider_feature_search:run_wider_feature_search',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s5r_wider_feature_search.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=(evaluator_ref,))
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id=plan['search_id'],stage_id='G6:S5R-WIDER',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s5r_wider_feature_search import run_wider_feature_search
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_wider_feature_search(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S5R_COMPOSITIONAL_READ_WRITE':
        expected={'science_plan','master_prereg','audit','s1r_closeout','s2r_closeout','s4r_closeout','e3_recovered_fixture','wider_result','amendment_spec'}
        if set(amap)!=expected: raise ControllerLoopError('G6_CRW_SCIENCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evaluator_ref='infinity_grid.g6_s5r_crw_evaluators:observer_inversion_descriptor_evaluator'
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S5R-CRW0',stage_id='G6:S5R-CRW0',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s5r.compositional-read-write',handler_ref='infinity_grid.g6_s5r_compositional_read_write:run_compositional_read_write_crw0',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s5r_compositional_read_write.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=(evaluator_ref,))
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S5R-CRW0',stage_id='G6:S5R-CRW0',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s5r_compositional_read_write import run_compositional_read_write_crw0
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_compositional_read_write_crw0(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S5R_COMPOSITIONAL_READ_WRITE_CRW1':
        expected={'science_plan','master_prereg','audit','g5_parent_review','s1r_closeout','s2r_closeout','s4r_closeout','e3_recovered_fixture','wider_result','amendment_registered_input','crw0_result'}
        if set(amap)!=expected: raise ControllerLoopError('G6_CRW1_SCIENCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evaluator_ref='infinity_grid.g6_s5r_crw_evaluators:observer_state_write_law_evaluator'
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S5R-CRW1',stage_id='G6:S5R-CRW1',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s5r.crw1-observer-state-write',handler_ref='infinity_grid.g6_s5r_crw1:run_crw1',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s5r_crw1.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=(evaluator_ref,))
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S5R-CRW1',stage_id='G6:S5R-CRW1',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s5r_crw1 import run_crw1
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_crw1(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S6R_RECURSIVE_CLOSURE':
        expected={'science_plan','master_prereg','g5_parent_review','crw0_result','crw1_result'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S6R_SCIENCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evals=('infinity_grid.g6_s6r_evaluators:recursive_closure_holdout_evaluator','infinity_grid.g6_s6r_evaluators:recursive_closure_independent_evaluator')
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S6R',stage_id='G6:S6R',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s6r.recursive-closure',handler_ref='infinity_grid.g6_s6r_recursive_closure:run_s6r',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s6r_recursive_closure.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=evals)
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S6R',stage_id='G6:S6R',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s6r_recursive_closure import run_s6r
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_s6r(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S6R_GLOBAL_OBSERVER_CONGRUENCE':
        expected={'science_plan','crw0_result','crw1_result','s6r_result'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S6R_GLOBAL_SCIENCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        from .g6_s6r_global_observer_congruence import run_global_observer_congruence_theorem
        outdir=runtime/'science'/record['internal_execution_id']
        result=run_global_observer_congruence_theorem(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'])
    elif op=='G6_S6R_L2_REPAIR':
        expected={'science_plan','global_theorem_result'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S6R_L2_REPAIR_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evaluator_ref='infinity_grid.g6_s6r_evaluators:l2_common_parent_uniqueness_evaluator'
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S6R-L2-REPAIR',stage_id='G6:S6R-L2-REPAIR',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s6r.l2-repair',handler_ref='infinity_grid.g6_s6r_l2_repair:run_l2_repair',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s6r_l2_repair.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=(evaluator_ref,))
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S6R-L2-REPAIR',stage_id='G6:S6R-L2-REPAIR',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s6r_l2_repair import run_l2_repair
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_l2_repair(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S6R_L2_PROOF_REDUCTION':
        expected={'science_plan','l2_result','crw0_result'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S6R_L2_PROOF_REDUCTION_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evaluator_ref='infinity_grid.g6_s6r_evaluators:l2_sibling_q_evaluator'
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S6R-L2-PROOF-REDUCTION',stage_id='G6:S6R-L2-PROOF-REDUCTION',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s6r.l2-proof-reduction',handler_ref='infinity_grid.g6_s6r_l2_proof_reduction:run_l2_proof_reduction',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s6r_l2_proof_reduction.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=(evaluator_ref,))
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S6R-L2-PROOF-REDUCTION',stage_id='G6:S6R-L2-PROOF-REDUCTION',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s6r_l2_proof_reduction import run_l2_proof_reduction
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_l2_proof_reduction(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S5R_FRESH_MARKER_OBSERVER':
        expected={'science_plan','master_prereg','g5_r4_read_first','historical_s6_closeout','l2_repair_closeout'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S5R_MARKER_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        from .g6_s5r_marker_observer import run_marker_s5
        outdir=runtime/'science'/record['internal_execution_id']
        result=run_marker_s5(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'])
    elif op=='G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE':
        expected={'science_plan','master_prereg','s5_marker_result','historical_s6_closeout','holdout_plan'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S6R_MARKER_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}; plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evals=('infinity_grid.g6_marker_evaluators:marker_write_holdout_evaluator','infinity_grid.g6_marker_evaluators:marker_write_independent_evaluator')
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S6R-MARKER',stage_id='G6:S6R-MARKER',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s6r.marker-recursive-closure',handler_ref='infinity_grid.g6_s6r_marker_recursive_closure:run_marker_s6',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s6r_marker_recursive_closure.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=evals)
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S6R-MARKER',stage_id='G6:S6R-MARKER',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=3*1024**3,workspace_budget_bytes=3*1024**3,_execution_permit=permit)
            from .g6_s6r_marker_recursive_closure import run_marker_s6
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_marker_s6(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S7_ORDINARY_FUTURE_QUOTIENT':
        expected={'science_plan','master_prereg','post_r0_handoff'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S7_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}; plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evals=(
          'infinity_grid.g6_controller_evaluators:g6_s1_universe_generation_evaluator',
          'infinity_grid.g6_controller_evaluators:g6_s3_axis_a_generation_evaluator',
          'infinity_grid.g6_s7_evaluators:s7_public_state_evaluator',
          'infinity_grid.g6_s7_evaluators:s7_ordinary_branch_count_component_evaluator',
          'infinity_grid.g6_s7_evaluators:s7_ordinary_branch_relation_evaluator',
        )
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S7-ORDINARY-FUTURE',stage_id='G6:S7',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s7.ordinary-future-quotient',handler_ref='infinity_grid.g6_s7_ordinary_future_quotient:run_g6_s7',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s7_ordinary_future_quotient.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=evals)
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S7-ORDINARY-FUTURE',stage_id='G6:S7',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=4*1024**3,workspace_budget_bytes=4*1024**3,_execution_permit=permit)
            from .g6_s7_ordinary_future_quotient import run_g6_s7
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_g6_s7(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S7_A6_REUSE_PROVENANCE_VERIFY':
        expected={'verification_plan','a6_completion','reuse_input'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S7_A6_REUSE_PROVENANCE_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        from .g6_s7_a6_reuse_provenance import run_g6_s7_a6_reuse_provenance_verify
        outdir=runtime/'validation'/record['internal_execution_id']
        result=run_g6_s7_a6_reuse_provenance_verify(plan_path=resolved['verification_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'])
    elif op=='G6_S7_DEPTH2_COMPLETION':
        expected={'science_plan','a6_completion','reuse_input'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S7D2_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}; plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evals=(
            'infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator',
            'infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator',
            'infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator',
            'infinity_grid.g6_s7_evaluators:s7_depth2_parent_profile_multiset_evaluator',
        )
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S7-DEPTH2-COMPLETION',stage_id='G6:S7',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s7.depth2-completion',handler_ref='infinity_grid.g6_s7_depth2_completion:run_g6_s7_depth2_completion',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s7_depth2_completion.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=evals)
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S7-DEPTH2-COMPLETION',stage_id='G6:S7',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=4*1024**3,workspace_budget_bytes=16*1024**3,_execution_permit=permit)
            from .g6_s7_depth2_completion import run_g6_s7_depth2_completion
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_g6_s7_depth2_completion(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE':
        expected={'science_plan','s7_result','s7_manifest','s7_memberships','s7_survivors','s7_context_normalization','reuse_input'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S8_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}; plan=json.loads(resolved['science_plan'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evals=('infinity_grid.g6_s8_evaluators:s8_recursive_outer_component_evaluator','infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator')
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S8-INTRINSIC-DESCRIPTOR',stage_id='G6:S8',question_sha256=plan['question_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s8.intrinsic-descriptor-and-deeper-congruence',handler_ref='infinity_grid.g6_s8_intrinsic_descriptor:run_g6_s8',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s8_intrinsic_descriptor.py'),parameters_sha256=canonical_sha256(plan),authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=evals)
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S8-INTRINSIC-DESCRIPTOR',stage_id='G6:S8',question_sha256=plan['question_sha256'],default_workers=4,memory_budget_bytes=4*1024**3,workspace_budget_bytes=24*1024**3,_execution_permit=permit)
            from .g6_s8_intrinsic_descriptor import run_g6_s8
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_g6_s8(plan_path=resolved['science_plan'],artifacts={k:v for k,v in resolved.items() if k!='science_plan'},output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_S8_V3_TARGETED_PROBE':
        expected={'v3_prereg','s7_result','s7_manifest','s7_memberships','s7_survivors','s7_context_normalization','reuse_input'}
        if set(amap)!=expected: raise ControllerLoopError('G6_S8_V3_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}; plan=json.loads(resolved['v3_prereg'].read_text(encoding='utf-8'))
        chain_dir=runtime/'science_chains'/record['internal_execution_id']; chain_dir.mkdir(parents=True,exist_ok=True)
        from .v05_execution_authority import issue_controller_event_runtime_permit
        from .v05_stage_runtime import StageScienceRuntime
        evals=('infinity_grid.g6_s8_evaluators:s8_v3_relation_generation_evaluator','infinity_grid.g6_s8_evaluators:s8_v3_profile_extension_evaluator','infinity_grid.g6_s7_evaluators:s7_public_state_evaluator')
        permit=issue_controller_event_runtime_permit(chain_dir=chain_dir,chain_id='G6-S8-V3-TARGETED-PROBE',stage_id='G6:S8',question_sha256=plan['scientific_core_sha256'],registration_sha256=record['registration_sha256'],source_sha256=engineering_source_tree_digest(source),handler_key='g6.s8.v3-targeted-probe',handler_ref='infinity_grid.g6_s8_v3_targeted_probe:run_g6_s8_v3_targeted_probe',handler_source_sha256=_sha_file(source/'infinity_grid/g6_s8_v3_targeted_probe.py'),parameters_sha256=plan['execution_contract_sha256'],authority_sha256=record['registration_sha256'],dependencies_sha256=canonical_sha256(amap),run_id=record['internal_execution_id'].split('-',1)[1],evaluator_refs=evals)
        try:
            stage_runtime=StageScienceRuntime(chain_dir=chain_dir,chain_id='G6-S8-V3-TARGETED-PROBE',stage_id='G6:S8',question_sha256=plan['scientific_core_sha256'],default_workers=4,memory_budget_bytes=4*1024**3,workspace_budget_bytes=8*1024**3,_execution_permit=permit)
            from .g6_s8_v3_targeted_probe import run_g6_s8_v3_targeted_probe
            outdir=runtime/'science'/record['internal_execution_id']
            result=run_g6_s8_v3_targeted_probe(plan_path=resolved['v3_prereg'],artifacts={k:v for k,v in resolved.items() if k!='v3_prereg'},output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'],runtime=stage_runtime)
        finally:
            permit.revoke()
    elif op=='G6_R0_POST_GRADUATION_FIBER':
        expected={'science_plan','master_prereg','graduation_certificate','s5_marker_closeout','s6_marker_closeout'}
        if set(amap)!=expected: raise ControllerLoopError('G6_R0_ARTIFACT_SET')
        resolved={k:_artifact(runtime,v) for k,v in amap.items()}
        from .g6_r0_post_graduation_fiber import run_g6_r0
        outdir=runtime/'science'/record['internal_execution_id']
        result=run_g6_r0(plan_path=resolved['science_plan'],artifacts=resolved,output_dir=outdir,accepted_source_sha256=engineering_source_tree_digest(source),internal_execution_id=record['internal_execution_id'])
    else:
        raise ControllerLoopError('G6_SCIENCE_OPERATION_NOT_REGISTERED')
    return {'schema_id':'IG_DECODER_CONTROLLER_G6_SCIENCE_RESULT_V1','status':'PASS','internal_execution_id':record['internal_execution_id'],'operation':op,'science_result':result,'output_dir':str(outdir),'restart_required':False}

def _handle_c6_context_probe(record:dict[str,Any],runtime:Path,source:Path,*,supervisor_pid:int)->dict[str,Any]:
    """Non-mutating passive proof that the request is inside a verified controller root."""
    require_controller_execution_origin('c6-context-probe')
    from .v05_origin_guard import current_execution_context_snapshot
    ctx=current_execution_context_snapshot()
    if type(ctx) is not dict or ctx.get('role')!='CONTROLLER_ROOT' or ctx.get('origin')!='CONTROLLER_EVENT_LOOP':
        raise ControllerLoopError('C6_CONTEXT_PROBE_ORIGIN')
    return {
      'schema_id':'IG_DECODER_C6_CONTEXT_PROBE_RESULT_V1','status':'PASS',
      'internal_execution_id':record['internal_execution_id'],
      'source_sha256':engineering_source_tree_digest(source),
      'supervisor_pid':int(supervisor_pid),
      'controller_session_id':ctx['controller_session_id'],
      'root_execution_id':ctx['root_execution_id'],
      'origin':ctx['origin'],'role':ctx['role'],
      'scientific_effect':'NONE','restart_required':False,
    }


def controller_child_main(runtime_root:str|Path,source_root:str|Path,*,supervisor_pid:int,supervisor_secret:str)->int:
    runtime=Path(runtime_root).resolve(); source=Path(source_root).resolve(strict=True)
    current_source,current_pkg=_source_ids(source)
    lease=runtime/'supervisor'/'lease.json'
    # Resource housekeeping only: release clean page-cache pages belonging to
    # large completed phase stores left by prior executions.  No evidence bytes
    # are deleted or modified.
    from .v05_stage_runtime import reclaim_completed_phase_file_cache, cgroup_memory_snapshot
    reclaim = reclaim_completed_phase_file_cache(runtime)
    memory_status = {
        'schema_id':'IG_DECODER_MEMORY_HOUSEKEEPING_STATUS_V1',
        'status':'PASS' if str(reclaim.get('status','')).startswith('PASS') else 'PAUSED',
        'source_sha256':current_source,
        'reclaim':reclaim,
        'memory_after_reclaim':cgroup_memory_snapshot(),
        'scientific_effect':'NONE',
    }
    _snapshot(runtime,memory_status,name='MEMORY_HOUSEKEEPING_STATUS.json')
    _snapshot(runtime,{'schema_id':'IG_DECODER_ACTIVE_REQUEST_V1','state':'IDLE','request_id':None,'registered_job_id':None,'requested_operation_id':None},name='ACTIVE_REQUEST.json')
    while True:
        pending=runtime/'intake'/'pending'; pending.mkdir(parents=True,exist_ok=True)
        completed=runtime/'intake'/'completed'; completed.mkdir(parents=True,exist_ok=True)
        did=False
        for p in sorted(pending.glob('*.json')):
            rid=p.stem
            if (completed/(rid+'.json')).is_file(): continue
            did=True
            attempt_path=None
            try:
                from .v05_passive_intake import read_passive_request
                request=read_passive_request(runtime/'intake',rid)
                registry=_registry(source)
                registered=registry.get(request['registered_job_id'])
                if type(registered) is not dict: raise ControllerLoopError('CONTROLLER_JOB_NOT_REGISTERED')
                attempt_path,attempt_binding=_start_controller_attempt(
                    runtime,rid,request,current_source,registered['registration_sha256'])
                with _supervisor_controller_event_scope(supervisor_pid=supervisor_pid,supervisor_secret=supervisor_secret,lease_path=lease,expected_source_sha256=current_source,attempt_binding=attempt_binding):
                    record=ingest_passive_request(runtime/'intake',rid,accepted_registry=registry,expected_parent_source_sha256=current_source,internal_root=runtime/'internal')
                    _snapshot(runtime,{'schema_id':'IG_DECODER_ACTIVE_REQUEST_V1','state':'ACTIVE','request_id':rid,'internal_execution_id':record['internal_execution_id'],'registered_job_id':record['registered_job_id'],'requested_operation_id':record['requested_operation_id'],'source_sha256':current_source},name='ACTIVE_REQUEST.json')
                    if record['registered_job_id']==ENGINEERING_JOB and record['requested_operation_id']=='APPLY_SOURCE_CHANGE':
                        result=_handle_source_change(record,runtime,source,activate=False)
                    elif record['registered_job_id']==ENGINEERING_JOB and record['requested_operation_id']=='APPLY_SOURCE_CHANGE_AND_ACTIVATE':
                        result=_handle_source_change(record,runtime,source,activate=True)
                    elif record['registered_job_id']==SCIENCE_JOB and record['requested_operation_id'] in ('G6_S5R_OBSERVER_CONTINUITY','G6_S5R_WIDER_FEATURE_SEARCH','G6_S5R_COMPOSITIONAL_READ_WRITE','G6_S5R_COMPOSITIONAL_READ_WRITE_CRW1','G6_S6R_RECURSIVE_CLOSURE','G6_S6R_GLOBAL_OBSERVER_CONGRUENCE','G6_S6R_L2_REPAIR','G6_S6R_L2_PROOF_REDUCTION','G6_S5R_FRESH_MARKER_OBSERVER','G6_S6R_FRESH_MARKER_RECURSIVE_CLOSURE','G6_R0_POST_GRADUATION_FIBER','G6_S7_ORDINARY_FUTURE_QUOTIENT','G6_S7_A6_REUSE_PROVENANCE_VERIFY','G6_S7_DEPTH2_COMPLETION','G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE','G6_S8_V3_TARGETED_PROBE'):
                        result=_handle_g6_science(record,runtime,source)
                    elif record['registered_job_id']==C6_RECOVERY_JOB and record['requested_operation_id']=='C6_CONTEXT_PROBE':
                        result=_handle_c6_context_probe(record,runtime,source,supervisor_pid=supervisor_pid)
                    else: raise ControllerLoopError('CONTROLLER_OPERATION_NOT_REGISTERED')
                done={'schema_id':'IG_DECODER_CONTROLLER_REQUEST_COMPLETION_V1','status':'PASS','request_id':rid,'accepted_source_sha256':current_source,'result':result,'completed_sha256':canonical_sha256(result)}
                write_json_atomic(completed/(rid+'.json'),done); os.chmod(completed/(rid+'.json'),0o444)
                _snapshot(runtime,{'schema_id':'IG_DECODER_ACTIVE_REQUEST_V1','state':'IDLE','request_id':None,'registered_job_id':None,'requested_operation_id':None,'source_sha256':current_source},name='ACTIVE_REQUEST.json')
                _snapshot(runtime,{'schema_id':'IG_DECODER_CONTROLLER_STATUS_V1','status':'RUNNING','source_sha256':current_source,'package_sha256':current_pkg,'last_request_id':rid,'last_request_status':'PASS','supervisor_pid':supervisor_pid,'controller_pid':os.getpid(),'final_origin_exclusivity':bool((runtime/'final/C5_FINAL_ACCEPTANCE.json').is_file())})
                _finish_controller_attempt(attempt_path,'COMPLETED')
                if result.get('restart_required'): return RESTART_TO_ACTIVE_SOURCE
            except BaseException as exc:
                if attempt_path is not None:
                    try: _finish_controller_attempt(attempt_path,'PAUSED',type(exc).__name__+':'+str(exc))
                    except Exception: pass
                fail={'schema_id':'IG_DECODER_CONTROLLER_REQUEST_COMPLETION_V1','status':'PAUSED','request_id':rid,'reason':type(exc).__name__+':'+str(exc),'accepted_source_sha256':current_source}
                write_json_atomic(completed/(rid+'.json'),fail); os.chmod(completed/(rid+'.json'),0o444)
                _snapshot(runtime,{'schema_id':'IG_DECODER_CONTROLLER_STATUS_V1','status':'PAUSED_PENDING_VERIFICATION','source_sha256':current_source,'last_request_id':rid,'reason':fail['reason'],'supervisor_pid':supervisor_pid,'controller_pid':os.getpid(),'final_origin_exclusivity':False})
        if not did:
            _snapshot(runtime,{'schema_id':'IG_DECODER_CONTROLLER_STATUS_V1','status':'RUNNING','source_sha256':current_source,'package_sha256':current_pkg,'supervisor_pid':supervisor_pid,'controller_pid':os.getpid(),'final_origin_exclusivity':bool((runtime/'final/C5_FINAL_ACCEPTANCE.json').is_file())})
        time.sleep(0.2)

def start_c5_migration_supervisor(runtime_root:str|Path,transition_source_root:str|Path)->int:
    """Final source: direct/bootstrap supervisor startup is permanently closed."""
    raise ControllerLoopError('REJECT_DIRECT_EXECUTION_ROUTE:C5_BOOTSTRAP_CLOSED')

# Cut-down registered workspace path (policy approved 11 September 2026).
# This is the default CLI in this existing controller module. The C5/C6 entry
# above remains available only for historical replay; no credentials are imitated.
from contextlib import contextmanager
import argparse
import fcntl
import importlib
import re
import tempfile
import zipfile
import signal

WORKSPACE_SCHEMA = 'IG_DECODER_WORKSPACE_V1'
JOB_SCHEMA = 'IG_DECODER_WORKSPACE_JOB_V1'
_JOB_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')
_REF = re.compile(r'^(?:infinity_grid|project)(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*$')
_SKIP_SOURCE = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.engineering_tmp'}


def _json_object(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ControllerLoopError('DUPLICATE_JSON_FIELD:' + key)
            out[key] = value
        return out
    obj = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs)
    if type(obj) is not dict: raise ControllerLoopError('JSON_OBJECT_REQUIRED')
    canonical_sha256(obj)  # Reject non-finite or non-JSON values.
    return obj


def _workspace_ids(workspace: Path, *, check_loaded: bool = True) -> tuple[Path, str, str]:
    info = _json_object(workspace / 'WORKSPACE.json')
    if set(info) != {'schema_id', 'source_sha256', 'package_sha256'} or info['schema_id'] != WORKSPACE_SCHEMA:
        raise ControllerLoopError('WORKSPACE_MANIFEST')
    source = (workspace / 'source').resolve(strict=True)
    if not source.is_relative_to(workspace): raise ControllerLoopError('WORKSPACE_SOURCE_PATH')
    sid, pid = _source_ids(source)
    if (sid, pid) != (info['source_sha256'], info['package_sha256']):
        raise ControllerLoopError('WORKSPACE_SOURCE_MISMATCH')
    loaded = Path(__file__).resolve().parents[1]
    if check_loaded and loaded != source and _source_ids(loaded) != (sid, pid):
        raise ControllerLoopError('WORKSPACE_LOADED_CODE_MISMATCH')
    return source, sid, pid


def validate_workspace_job(workspace: str | Path, job_id: str, *, check_loaded: bool = True) -> dict[str, Any]:
    """Admission uses saved bytes, not process ancestry or historical liveness."""
    root = Path(workspace).resolve(strict=True)
    if type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None:
        raise ControllerLoopError('JOB_IDENTIFIER')
    path = root / 'registry' / (job_id + '.json')
    if not path.is_file(): raise ControllerLoopError('JOB_NOT_REGISTERED:' + job_id)
    job = _json_object(path)
    fields = {'schema_id', 'job_id', 'source_sha256', 'question', 'input_artifacts',
              'execution', 'resources', 'registration_sha256'}
    if set(job) != fields or job['schema_id'] != JOB_SCHEMA or job['job_id'] != job_id:
        raise ControllerLoopError('JOB_REGISTRATION_FIELDS')
    body = {k:v for k,v in job.items() if k != 'registration_sha256'}
    if canonical_sha256(body) != job['registration_sha256']:
        raise ControllerLoopError('JOB_REGISTRATION_MISMATCH')
    source, sid, pid = _workspace_ids(root, check_loaded=check_loaded)
    if job['source_sha256'] != sid: raise ControllerLoopError('JOB_SOURCE_MISMATCH')
    question = job['question']
    if (type(question) is not dict or set(question) != {'stage_id','description','outcomes','stopping_rule'}
        or any(type(question[k]) is not str or not question[k] for k in ('stage_id','description','stopping_rule'))
        or type(question['outcomes']) is not list or not question['outcomes']
        or any(type(x) is not str or not x for x in question['outcomes'])):
        raise ControllerLoopError('JOB_QUESTION')
    resources = job['resources']
    legacy_fields = {'workers','start_method','memory_budget_bytes','workspace_budget_bytes','wall_seconds_max'}
    new_fields = {'workers','start_method','memory_budget_bytes','workspace_budget_bytes','execution_policy'}
    if type(resources) is not dict or set(resources) not in (legacy_fields, new_fields, legacy_fields|{'execution_policy'}):
        raise ControllerLoopError('JOB_RESOURCES')
    from .execution_policy import name as execution_policy_name, NO_DEADLINE
    try: policy_name = execution_policy_name(resources)
    except ValueError as exc: raise ControllerLoopError(str(exc)) from exc
    if type(resources['workers']) is not int or not 1 <= resources['workers'] <= 4:
        raise ControllerLoopError('JOB_WORKERS')
    if resources['start_method'] not in ('AUTO','spawn','forkserver','fork'):
        raise ControllerLoopError('JOB_START_METHOD')
    for name in ('memory_budget_bytes','workspace_budget_bytes'):
        if type(resources[name]) is not int or resources[name] <= 0:
            raise ControllerLoopError('JOB_RESOURCE_BUDGET:' + name)
    if policy_name != NO_DEADLINE and (type(resources.get('wall_seconds_max')) is not int or resources['wall_seconds_max'] <= 0):
        raise ControllerLoopError('JOB_RESOURCE_BUDGET:wall_seconds_max')
    execution = job['execution']
    if type(execution) is not dict: raise ControllerLoopError('JOB_EXECUTION')
    kind = execution.get('kind')
    if kind == 'VALIDATION':
        if set(execution) != {'kind','nodes'} or type(execution['nodes']) is not list or not execution['nodes']:
            raise ControllerLoopError('JOB_VALIDATION_NODES')
        if any(type(n) is not str for n in execution['nodes']) or len(execution['nodes']) != len(set(execution['nodes'])):
            raise ControllerLoopError('JOB_VALIDATION_NODES')
        for node in execution['nodes']:
            p = Path(node.split('::',1)[0])
            if p.is_absolute() or '..' in p.parts or not str(p).startswith('tests/') or p.suffix != '.py':
                raise ControllerLoopError('JOB_VALIDATION_PATH')
            if not (source / p).resolve(strict=True).is_relative_to(source):
                raise ControllerLoopError('JOB_VALIDATION_PATH')
    elif kind == 'SCRIPT':
        if set(execution) != {'kind','entrypoint','argv','parameters'} or type(execution['parameters']) is not dict:
            raise ControllerLoopError('JOB_SCRIPT_FIELDS')
        p = Path(execution['entrypoint'])
        if p.is_absolute() or '..' in p.parts or not p.as_posix().startswith('project/') or p.suffix != '.py':
            raise ControllerLoopError('JOB_SCRIPT_PATH')
        if not (source/p).resolve(strict=True).is_relative_to(source/'project'):
            raise ControllerLoopError('JOB_SCRIPT_PATH')
        if type(execution['argv']) is not list or any(type(x) is not str for x in execution['argv']):
            raise ControllerLoopError('JOB_SCRIPT_ARGUMENTS')
        if resources['workers'] != 1 or resources['start_method'] not in ('AUTO','fork'):
            raise ControllerLoopError('SCRIPT_RESOURCE_CONFIGURATION:serial SCRIPT requires workers=1 and AUTO/fork; use STAGE for scheduled parallel work')
        if not {'PROCESS_COMPLETED','PROCESS_FAILED'} <= set(question['outcomes']):
            raise ControllerLoopError('SCRIPT_PROCESS_OUTCOMES_REQUIRED')
    elif kind == 'STAGE':
        if set(execution) != {'kind','handler_ref','evaluator_refs','parameters'} or type(execution['parameters']) is not dict or type(execution['evaluator_refs']) is not list:
            raise ControllerLoopError('JOB_STAGE_FIELDS')
        for ref in [execution['handler_ref'], *execution['evaluator_refs']]:
            if type(ref) is not str or _REF.fullmatch(ref) is None:
                raise ControllerLoopError('JOB_HANDLER_REFERENCE')
            mod = ref.split(':',1)[0]
            if mod.startswith('project.'):
                from .project_stage import callable_path, ProjectStageError
                try: callable_path(source, ref)
                except (OSError, ProjectStageError) as exc:
                    raise ControllerLoopError('JOB_HANDLER_MISSING') from exc
            elif not (source / (mod.replace('.', '/') + '.py')).is_file():
                raise ControllerLoopError('JOB_HANDLER_MISSING')
        if 'workers' in execution['parameters'] and execution['parameters']['workers'] != resources['workers']:
            raise ControllerLoopError('JOB_WORKER_CONFIGURATION_CONFLICT')
    else:
        raise ControllerLoopError('JOB_KIND_NOT_REGISTERED')
    from .v05_passive_intake import validate_passive_request
    request = validate_passive_request({
        'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1',
        'request_id':'job-' + job['registration_sha256'][:32],
        'registered_job_id':job_id, 'requested_operation_id':kind,
        'parent_source_sha256':sid, 'input_artifacts':job['input_artifacts'],
    })
    artifacts = {row['logical_name']:_artifact(root/'runtime',row['sha256']) for row in request['input_artifacts']}
    return {'workspace':root,'source':source,'source_sha256':sid,'package_sha256':pid,
            'job':job,'request':request,'artifacts':artifacts}


@contextmanager
def _workspace_lock(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / '.runner.lock').open('a+b') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise ControllerLoopError('WORKSPACE_BUSY') from exc
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _workspace_environment(runtime: Path, resources: dict[str, Any]):
    from .execution_policy import name as execution_policy_name
    values = {'IG_DECODER_LEASE_ROOT':str(runtime/'execution_leases'),
              'IG_DECODER_CPU_BUDGET':str(resources['workers']),
              'IG_DECODER_START_METHOD':resources['start_method'],
              'IG_DECODER_EXECUTION_POLICY':execution_policy_name(resources),
              'PYTHONDONTWRITEBYTECODE':'1'}
    previous = {k:os.environ.get(k) for k in values}
    os.environ.update(values)
    try: yield
    finally:
        for key,value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def _evidence_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.exists(): return rows
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ControllerLoopError('EVIDENCE_SYMLINK')
        if p.is_file():
            rows.append({'path':p.relative_to(root).as_posix(),'sha256':_sha_file(p),'size_bytes':p.stat().st_size})
    return rows



@contextmanager
def _stage_deadline(seconds: int):
    """Bound a stage invocation; normal exception cleanup retains committed tasks."""
    def expired(_signum, _frame):
        raise ControllerLoopError('REGISTERED_WALL_BUDGET_EXCEEDED')
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] > 0:
        raise ControllerLoopError('NESTED_WALL_TIMER_NOT_SUPPORTED')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

_CONTROLLER_PROCESS_HANDLERS = frozenset({
    # Registered engineering assertions inspect/migrate durable test databases.
    'infinity_grid.representative_qualification:handler',
    'infinity_grid.change_validation:validate_revision',
    'infinity_grid.v05_engineering_stage:engineering_job_handler',
    # Replay DAG scheduling is controller orchestration, never scientific work.
    'infinity_grid.replay_root_job:replay_root_job_handler',
})


def _dispatch_workspace_job(admission: dict[str, Any], record: dict[str, Any], out: Path) -> dict[str, Any]:
    from .v05_origin_guard import require_registered_dispatch, require_native_caller
    require_native_caller(__name__, {'_run_workspace_job'}, 'registered-workspace-dispatch')
    require_registered_dispatch(admission, out, 'registered-workspace-dispatch')
    job = admission['job']; ex = job['execution']; resources = job['resources']
    from .execution_policy import automatic_deadlines
    timed = automatic_deadlines(resources)
    if ex['kind'] == 'SCRIPT':
        from .script_runtime import run_captured_script
        return run_captured_script(admission, out/'script')
    if ex['kind'] == 'VALIDATION':
        from .v05_validation_runtime import run_registered_validation_nodes
        return run_registered_validation_nodes(admission['source'], ex['nodes'],
            workers=resources['workers'], wall_seconds_max=resources.get('wall_seconds_max') if timed else None, output_dir=out/'logs')
    from .v05_execution_authority import issue_controller_event_runtime_permit
    from .v05_stage_runtime import StageScienceRuntime
    chain = out/'chain'; chain.mkdir(parents=True,exist_ok=True)
    ref = ex['handler_ref']; mod, name = ref.split(':',1)
    from .project_stage import bind_callables, is_project_ref, resolve_callable
    project_bindings = bind_callables(
        admission['source'], [ref, *ex['evaluator_refs']],
        source_sha256=admission['source_sha256'])
    handler_path = (admission['source'] / project_bindings[ref]['module_path']
                    if is_project_ref(ref)
                    else admission['source']/(mod.replace('.','/')+'.py'))
    qsha = canonical_sha256(job['question']); regsha = job['registration_sha256']
    permit = issue_controller_event_runtime_permit(
        chain_dir=chain, chain_id=job['job_id'], stage_id=job['question']['stage_id'],
        question_sha256=qsha, registration_sha256=regsha, source_sha256=admission['source_sha256'],
        handler_key=job['job_id'], handler_ref=ref,
        handler_source_sha256=_sha_file(handler_path),
        parameters_sha256=canonical_sha256(ex['parameters']), authority_sha256=regsha,
        dependencies_sha256=canonical_sha256(job['input_artifacts']),
        run_id=record['internal_execution_id'].split('-',1)[1], evaluator_refs=tuple(ex['evaluator_refs']))
    try:
        runtime = StageScienceRuntime(chain_dir=chain, chain_id=job['job_id'], stage_id=job['question']['stage_id'],
            question_sha256=qsha, default_workers=resources['workers'], memory_budget_bytes=resources['memory_budget_bytes'],
            workspace_budget_bytes=resources['workspace_budget_bytes'], _execution_permit=permit,
            _project_callable_bindings=project_bindings)
        # The existing DECODER_STAGE protocol; no new scientific execution wrapper.
        stage = {'stage_id':job['question']['stage_id'],'question_sha256':qsha,
                 'execution':{'handler_key':job['job_id'],'parameters':ex['parameters']},
                 'input_artifacts':{k:str(v) for k,v in admission['artifacts'].items()}}
        from .workflow_guard import scientific_call
        from contextlib import nullcontext
        controller_process_handler = ref in _CONTROLLER_PROCESS_HANDLERS
        with (_stage_deadline(resources['wall_seconds_max']) if timed else nullcontext()), (nullcontext() if controller_process_handler else scientific_call(admission['source'], allow_scheduler=True)):
            handler = (resolve_callable(ref, project_bindings.get(ref)) if is_project_ref(ref)
                       else getattr(importlib.import_module(mod),name))
            value = handler(stage,runtime)
        result = value.result
        if type(result) is not dict or result.get('outcome') not in job['question']['outcomes']:
            raise ControllerLoopError('UNREGISTERED_SCIENTIFIC_OUTCOME')
        return result
    finally:
        permit.revoke()


def verified_completion(admission, *, allow_pending_checkpoint=False):
    """Verify original registered completion without executing source or relabelling it."""
    from .v05_passive_intake import request_claim_binding
    job=admission['job'];req=admission['request'];root=admission['workspace']
    identity=request_claim_binding(req,job['registration_sha256'],admission['source_sha256'],admission['source_sha256'])
    out=root/'runtime/runs'/('intent-'+canonical_sha256(identity)[:32])
    completion=root/'runtime/intake/completed'/(req['request_id']+'.json')
    if not completion.exists() and allow_pending_checkpoint:
        completion=root/'runtime/intake/prepared_completions'/(req['request_id']+'.json')
    if completion.exists():
        done = _json_object(completion)
        from .completion_evidence import evidence_root
        out = evidence_root(root, out, done)
        if (done.get('schema_id') != 'IG_DECODER_WORKSPACE_COMPLETION_V1'
            or done.get('registration_sha256') != job['registration_sha256']
            or done.get('source_sha256') != admission['source_sha256']
            or done.get('request_sha256') != identity['request_sha256']
            or done.get('result_sha256') != canonical_sha256(done.get('result'))
            or done.get('evidence') != _evidence_rows(out)
            or done.get('completion_sha256') != canonical_sha256({k:v for k,v in done.items() if k != 'completion_sha256'})):
            raise ControllerLoopError('COMPLETION_EVIDENCE_MISMATCH')
        if not allow_pending_checkpoint:
            from .preservation import terminal_completion_proof
            if not terminal_completion_proof(root,done):
                raise ControllerLoopError('COMPLETION_PENDING_TERMINAL_CHECKPOINT')
        return done
    return None


def _publish_checkpointed_completion(admission, done, claim):
    from .preservation import ensure_terminal_completion, COMPLETION_PROTOCOL
    root=admission['workspace']
    preservation=ensure_terminal_completion(root,done)
    completion=root/'runtime/intake/completed'/(done['request_id']+'.json')
    if completion.exists() and canonical_bytes(_json_object(completion))!=canonical_bytes(done):
        raise ControllerLoopError('COMPLETION_PUBLICATION_COLLISION')
    if not completion.exists():write_json_atomic(completion,done)
    if done.get('publication_protocol')==COMPLETION_PROTOCOL:
        attempts=sorted((root/'runtime/attempts'/done['request_id']).glob('*.json'))
        if attempts:
            latest=_json_object(attempts[-1])
            terminal=dict(latest,status=done['status'],finished_unix=time.time())
            write_json_atomic(attempts[-1],terminal);_snapshot(root/'runtime',terminal)
    claim.complete(done)
    return preservation


def _verified_artifact_evidence(output, verification):
    """Bind contract checks to the bytes actually sealed as completion evidence."""
    rows=_evidence_rows(output)
    by_path={row['path']:row for row in rows}
    for artifact in verification['artifacts']:
        if by_path.get(artifact['path'])!=artifact:
            raise ControllerLoopError('RESULT_ARTIFACT_CHANGED_AFTER_VERIFICATION')
    return rows


def run_workspace_job(workspace: str | Path, job_id: str) -> dict[str, Any]:
    """Public request entry. All refusals are retained, never interpreted as success."""
    try:
        return _run_workspace_job(workspace, job_id)
    except BaseException as exc:
        from .invocation import retain_refusal
        try: exc.refusal = retain_refusal(workspace, job_id, exc)
        except Exception: pass  # preserve the original error if the workspace cannot be written
        raise


def _run_workspace_job(workspace: str | Path, job_id: str) -> dict[str, Any]:
    """Run/resume one frozen saved job, recording an attempt BEFORE authority."""
    from .v05_origin_guard import _registered_attempt_scope
    from .v05_passive_intake import submit_passive_request, read_passive_request, request_claim_binding
    from .submission import require_saved
    from .workflow_guard import preflight_job
    root = Path(workspace).resolve(strict=True)
    with _workspace_lock(root):
        # Administrative reuse verifies the job's own captured source and exact
        # historical producer, but it neither loads nor executes that source.
        # Defer execution-only loaded-code and environment checks until after a
        # completed capsule has been sought and fully verified.
        admission = validate_workspace_job(root, job_id, check_loaded=False)
        require_saved(root, job_id, check_environment=False)
        from .result_contracts import prerequisites
        prerequisites(admission)
        from .preservation import backlog
        backlog(root,reserve=True)
        # A valid saved capsule does not excuse changed evidence in this workspace.
        # A fresh cross-capture workspace has no local completion; reuse stays allowed.
        done = verified_completion(admission, allow_pending_checkpoint=True)
        from .preservation import terminal_completion_proof
        pending_checkpoint=done is not None and not terminal_completion_proof(root,done)
        from .portable_registry import RunClaim
        reuse = None if pending_checkpoint else RunClaim(admission, reuse_verification=True).completed_reuse()
        if reuse is not None:
            write_json_atomic(root/'runtime/reuse'/('reuse-'+admission['job']['registration_sha256']+'.json'), reuse)
            return reuse
        if done is None:
            # From this point onward a scientific execution may occur.  The
            # running Decoder must therefore be the captured source and the
            # complete captured environment must match exactly.
            admission = validate_workspace_job(root, job_id, check_loaded=True)
            require_saved(root, job_id, check_environment=True)
        with RunClaim(admission) as claim:
            if claim.reused is not None:
                write_json_atomic(root/'runtime/reuse'/('reuse-'+admission['job']['registration_sha256']+'.json'), claim.reused)
                return claim.reused
            job = admission['job']; req = admission['request']; rid = req['request_id']; runtime = root/'runtime'
            registry = {job_id:{'allowed_operations':[req['requested_operation_id']],
                        'registration_sha256':job['registration_sha256'], 'implementation_sha256':admission['source_sha256']}}
            identity = request_claim_binding(req, job['registration_sha256'], admission['source_sha256'], admission['source_sha256'])
            intent_id = 'intent-' + canonical_sha256(identity)[:32]
            out = runtime/'runs'/intent_id
            completion = runtime/'intake/completed'/f'{rid}.json'
            if done is not None:
                preservation=_publish_checkpointed_completion(admission,done,claim)
                return dict(done, reused=True, preservation=preservation)
            # Lint is read-only and runs before imports/effects, retaining captured source on refusal.
            gate = preflight_job(admission)
            write_json_atomic(runtime/'registrations'/f'{rid}.json',job)
            attempts = runtime/'attempts'/rid; attempts.mkdir(parents=True,exist_ok=True)
            _reconcile_workspace_attempts(attempts,rid,job_id,admission['source_sha256'],
                                          job['registration_sha256'])
            attempt_no = len(list(attempts.glob('*.json'))) + 1
            attempt_path = attempts/f'{attempt_no:06d}.json'
            running = {'status':'RUNNING','request_id':rid,'job_id':job_id,'pid':os.getpid(),
                       'attempt_id':rid+':'+f'{attempt_no:06d}',
                       'registration_sha256':job['registration_sha256'],
                       'operation':req['requested_operation_id'], 'output_root':str(out),
                       'source_sha256':admission['source_sha256'],'started_unix':time.time(),
                       'workers_requested':job['resources']['workers'],'mirror_status':'NOT_CONFIRMED'}
            write_json_atomic(attempt_path,running); _snapshot(runtime,running)
            try:
                with _registered_attempt_scope(admission, attempt_path, out):
                    pending = runtime/'intake/pending'/f'{rid}.json'
                    if pending.exists():
                        if read_passive_request(runtime/'intake',rid) != req: raise ControllerLoopError('SAVED_REQUEST_MISMATCH')
                    else: submit_passive_request(runtime/'intake',req)
                    record = ingest_passive_request(runtime/'intake',rid,accepted_registry=registry,
                                expected_parent_source_sha256=admission['source_sha256'],internal_root=runtime/'internal')
                    if record['internal_execution_id'] != intent_id: raise ControllerLoopError('ATTEMPT_IDENTITY_MISMATCH')
                    out.mkdir(parents=True,exist_ok=True)
                    write_json_atomic(out/'PREFLIGHT.json', gate)
                    from .preservation import session, safe_point, poll
                    with _workspace_environment(runtime,job['resources']), session(admission):
                        safe_point('BEFORE_EXECUTION',force=True)
                        result = _dispatch_workspace_job(admission,record,out)
                        poll()
                        validation=result if job['execution']['kind']=='VALIDATION' else result.get('validation',{})
                        worker_rows=validation.get('execution_metadata',{}).get('worker_logs',[])
                        if any(r.get('return_code')==124 for r in worker_rows):
                            raise ControllerLoopError('VALIDATION_INTERRUPTED_BEFORE_COMPLETION')
                    validate_workspace_job(root,job_id)
                from .preservation import require_quiescent_task_databases
                require_quiescent_task_databases(out)
                from .result_contracts import contract_for, verify
                from .completion_evidence import prepare_evidence, PROTOCOL
                sealed, verification = prepare_evidence(admission, out, result)
                done = {'schema_id':'IG_DECODER_WORKSPACE_COMPLETION_V1',
                        'status':'RESULT_REJECTED' if verification['status']=='REJECTED' else 'COMPLETED' if job['execution']['kind']=='STAGE' or result.get('status')=='PASS' else 'VALIDATION_FAILED',
                        'execution_status':'FINISHED','evidence_status':verification['status'],
                        'scientific_outcome':verification['scientific_outcome'],
                        'request_id':rid,'request_sha256':record['request_sha256'],
                        'registration_sha256':job['registration_sha256'],'source_sha256':admission['source_sha256'],
                        'result':result,'result_sha256':canonical_sha256(result),'evidence':_verified_artifact_evidence(sealed,verification),
                        'evidence_protocol':PROTOCOL,'evidence_root':sealed.relative_to(root).as_posix(),
                        'policy':'REGISTERED_ATTEMPT_V1','mirror_status':'NOT_CONFIRMED',
                        'publication_protocol':'CHECKPOINT_BEFORE_COMPLETION_V1'}
                done['completion_sha256'] = canonical_sha256(done)
                prepared=runtime/'intake/prepared_completions'/f'{rid}.json'
                if prepared.exists() and _json_object(prepared)!=done:
                    raise ControllerLoopError('PREPARED_COMPLETION_COLLISION')
                if not prepared.exists():write_json_atomic(prepared,done)
                terminal = dict(running,status='COMPLETION_PENDING_CHECKPOINT',finished_unix=time.time())
                write_json_atomic(attempt_path,terminal); _snapshot(runtime,terminal)
                preservation=_publish_checkpointed_completion(admission,done,claim)
                return dict(done,reused=False,preservation=preservation)
            except BaseException as exc:
                paused = dict(running,status='PAUSED',reason=type(exc).__name__+':'+str(exc),finished_unix=time.time())
                write_json_atomic(attempt_path,paused); _snapshot(runtime,paused)
                try:
                    from .preservation import make_checkpoint
                    make_checkpoint(root,'PAUSED:'+type(exc).__name__)
                except Exception as save_exc:
                    write_json_atomic(root/'durability/CHECKPOINT_FAILURE.json',{'reason':str(save_exc),'execution_reason':str(exc),'status':'CHECKPOINT_UNAVAILABLE'})
                raise


def _snapshot_files(root: Path):
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if any(x in _SKIP_SOURCE for x in rel.parts) or rel.as_posix().startswith('runtime/execution_leases/'):
            continue
        if rel.as_posix().startswith(('durability/outbox/','durability/base_objects/')): continue
        if p.name == '.runner.lock' or p.suffix in {'.pyc','.pyo'}: continue
        if p.is_symlink(): raise ControllerLoopError('WORKSPACE_SNAPSHOT_SYMLINK')
        if p.is_file(): yield p,rel.as_posix()


def export_workspace(workspace: str | Path, output: str | Path, *, slim: bool = False) -> dict[str, Any]:
    """Idle/paused snapshot under the same lock; never copy a running task database."""
    root = Path(workspace).resolve(strict=True); dest = Path(output).resolve()
    if dest.is_relative_to(root): raise ControllerLoopError('EXPORT_MUST_BE_OUTSIDE_WORKSPACE')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with _workspace_lock(root):
        _workspace_ids(root)
        from .submission import capture_record
        if capture_record(root).get('result_contract'):
            from .preservation import make_checkpoint, export_checkpoint
            make_checkpoint(root,'EXPLICIT_IDLE_EXPORT')
            return export_checkpoint(root,dest,slim=slim)
        if slim: raise ControllerLoopError('LEGACY_SLIM_EXPORT_NOT_SUPPORTED')
        files = [(p,rel) for p,rel in _snapshot_files(root) if not rel.startswith('coordination/') and rel != 'PROJECT_LOCATION.json']
        extra={}
        if (root/'PROJECT_BINDING.json').exists():
            from .portable_registry import locate, snapshot
            registry_root,_=locate(root)
            import io
            with zipfile.ZipFile(io.BytesIO(snapshot(registry_root))) as project_zip:
                extra={'coordination/'+n:project_zip.read(n) for n in project_zip.namelist()}
        manifest = {'schema_id':'IG_DECODER_WORKSPACE_SNAPSHOT_V1','files':[
            {'path':rel,'sha256':_sha_file(p),'size_bytes':p.stat().st_size} for p,rel in files]+[{'path':n,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)} for n,b in extra.items()]}
        fd,tmpname = tempfile.mkstemp(prefix='.decoder-snapshot-',dir=dest.parent); os.close(fd)
        try:
            with zipfile.ZipFile(tmpname,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for p,rel in files: z.write(p,rel)
                for n,b in extra.items(): z.writestr(n,b)
                z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
            with open(tmpname,'rb') as f: os.fsync(f.fileno())
            os.replace(tmpname,dest)
        finally:
            Path(tmpname).unlink(missing_ok=True)
    return {'status':'EXPORTED_LOCAL','path':str(dest),'sha256':_sha_file(dest),'drive_save_confirmed':False}


def restore_workspace(archive: str | Path, destination: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Verify snapshot bytes and restore files. Starting work is a separate command."""
    archive = Path(archive).resolve(strict=True); dest = Path(destination).resolve()
    if _sha_file(archive) != expected_sha256: raise ControllerLoopError('RESTORE_ARCHIVE_HASH')
    if dest.exists(): raise ControllerLoopError('RESTORE_DESTINATION_EXISTS')
    with zipfile.ZipFile(archive) as inspected:
        if 'CHECKPOINT.json' in inspected.namelist():
            from .preservation import restore_checkpoint
            return restore_checkpoint(archive,dest,expected_sha256)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.decoder-restore-',dir=dest.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)): raise ControllerLoopError('RESTORE_DUPLICATE_PATH')
            manifest = json.loads(z.read('SNAPSHOT_MANIFEST.json'))
            if manifest.get('schema_id') != 'IG_DECODER_WORKSPACE_SNAPSHOT_V1': raise ControllerLoopError('RESTORE_MANIFEST')
            rows = manifest['files']
            if len(rows) != len({r['path'] for r in rows}) or set(names) != {r['path'] for r in rows}|{'SNAPSHOT_MANIFEST.json'}:
                raise ControllerLoopError('RESTORE_FILE_SET')
            for row in rows:
                rel = Path(row['path'])
                if rel.is_absolute() or '..' in rel.parts or '\\' in row['path']:
                    raise ControllerLoopError('RESTORE_PATH')
                target = tmp/rel; target.parent.mkdir(parents=True,exist_ok=True)
                h = hashlib.sha256(); size = 0
                if z.getinfo(row['path']).file_size != row['size_bytes']:
                    raise ControllerLoopError('RESTORE_FILE_SIZE')
                with z.open(row['path']) as src, target.open('xb') as dst:
                    for block in iter(lambda: src.read(1024*1024), b''):
                        h.update(block); size += len(block); dst.write(block)
                    dst.flush(); os.fsync(dst.fileno())
                if size != row['size_bytes'] or h.hexdigest() != row['sha256']:
                    raise ControllerLoopError('RESTORE_FILE_HASH')
        info = _json_object(tmp/'WORKSPACE.json')
        if _source_ids(tmp/'source') != (info['source_sha256'],info['package_sha256']):
            raise ControllerLoopError('RESTORE_SOURCE_HASH')
        if (tmp/'coordination/PROJECT.json').exists():
            from .portable_registry import events
            events(tmp/'coordination')
        os.replace(tmp,dest)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)
    return {'status':'RESTORED_NOT_RUNNING','workspace':str(dest),'source_sha256':info['source_sha256']}


def main() -> int:
    from ._version import __version__
    if len(sys.argv)>1 and sys.argv[1]=='preserve':
        from .preservation import main as preservation_main
        return preservation_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='transport':
        from .save_transport import main as transport_main
        return transport_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'change':
        from .change_sessions import main as change_main
        return change_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='project':
        from .portable_registry import main as project_main
        return project_main(sys.argv[2:])
    from . import submission
    parser = argparse.ArgumentParser(description='Decoder registered capture and execution.')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('preserve', help='Checkpoints and save outbox; use preserve --help')
    sub.add_parser('transport', help='Explicit chunked or compressed save copies; use transport --help')
    sub.add_parser('project', help='Portable project history; use project --help')
    sub.add_parser('change', help='Recorded engine changes; use change --help')
    run = sub.add_parser('run'); run.add_argument('workspace'); run.add_argument('job_id')
    stat = sub.add_parser('status'); stat.add_argument('workspace')
    export = sub.add_parser('export'); export.add_argument('workspace'); export.add_argument('output'); export.add_argument('--slim',action='store_true')
    restore = sub.add_parser('restore'); restore.add_argument('archive'); restore.add_argument('destination'); restore.add_argument('sha256')
    cap = sub.add_parser('capture'); cap.add_argument('store'); cap.add_argument('specification')
    pending = sub.add_parser('pending-saves'); pending.add_argument('workspace')
    confirm = sub.add_parser('confirm-save'); confirm.add_argument('workspace'); confirm.add_argument('sha256'); confirm.add_argument('readback'); confirm.add_argument('drive_file_id'); confirm.add_argument('--role'); confirm.add_argument('--logical-name')
    transport = sub.add_parser('confirm-transport'); transport.add_argument('workspace'); transport.add_argument('sha256'); transport.add_argument('manifest'); transport.add_argument('parts'); transport.add_argument('drive_file_id')
    failed = sub.add_parser('save-failed'); failed.add_argument('workspace'); failed.add_argument('sha256'); failed.add_argument('reason')
    recover = sub.add_parser('restore-capture'); recover.add_argument('capture_file'); recover.add_argument('objects'); recover.add_argument('destination')
    args = parser.parse_args()
    try:
        if args.command == 'capture': result = submission.capture(args.store,args.specification)
        elif args.command == 'pending-saves': result = submission.save_status(args.workspace)
        elif args.command == 'confirm-save': result = submission.confirm_save(args.workspace,args.sha256,args.readback,args.drive_file_id,role=args.role,logical_name=args.logical_name)
        elif args.command == 'confirm-transport': result = submission.confirm_transport(args.workspace,args.sha256,args.manifest,args.parts,args.drive_file_id)
        elif args.command == 'save-failed': result = submission.record_save_failure(args.workspace,args.sha256,args.reason)
        elif args.command == 'restore-capture': result = submission.restore_capture(args.capture_file,args.objects,args.destination)
        elif args.command == 'run': result = run_workspace_job(args.workspace,args.job_id)
        elif args.command == 'export': result = export_workspace(args.workspace,args.output,slim=args.slim)
        elif args.command == 'restore': result = restore_workspace(args.archive,args.destination,args.sha256)
        else:
            root = Path(args.workspace).resolve(strict=True)
            try:
                with _workspace_lock(root): active = False
            except ControllerLoopError as exc:
                if str(exc) != 'WORKSPACE_BUSY': raise
                active = True
            status = root/'runtime/status/CONTROLLER_STATUS.json'
            result = {'workspace_locked_now':active,'last_record':_json_object(status) if status.exists() else None,
                      'running_observation':'BUSY' if active else 'IDLE_OR_INTERRUPTED','drive_save_confirmed':False}
            result['decoder_version'] = __version__
            try: result['capture'] = submission.save_status(root)
            except submission.SubmissionError as exc: result['capture'] = {'status':exc.code,'next_action':exc.next_action}
        print(json.dumps(result,sort_keys=True,indent=2))
        return 1 if result.get('status') == 'VALIDATION_FAILED' else 0
    except Exception as exc:
        from .invocation import refusal_details
        refusal = getattr(exc, 'refusal', None) or refusal_details(exc, args.command,
            getattr(args, 'workspace', None), getattr(args, 'job_id', None))
        if hasattr(exc, 'preservation'): refusal['preservation'] = exc.preservation
        print(json.dumps(refusal, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == '__main__':
    from .controller import main as _native_main
    raise SystemExit(_native_main())
