from __future__ import annotations
import json
from pathlib import Path
from .canon import canonical_sha256
from .hashing import sha256_file
from .store import ArtifactStore
from .datasets import DatasetStore
from .protocols import ProtocolRegistry
from .records import runtime_sha256, source_sha256
from .checkpoints import CheckpointManager
from .resources import schema
from .schema import validate
from .safety import validate_identifier


def immutable_run_core(*,plan:dict, descriptor:dict, input_refs:list, execution_mode:dict, code_sha:str, env_sha:str) -> dict:
    return {
      'schema_id':'IG_RUN_CORE_V0_17_1','run_id':plan['run_id'],
      'protocol':{'protocol_id':descriptor['protocol_id'],'version':descriptor['version'],'descriptor_sha256':descriptor['descriptor_sha256']},
      'subject':plan['subject'],'question_sha256':plan['question_sha256'],'plan_sha256':plan['plan_sha256'],
      'evidence':plan['evidence'],'input_artifacts':input_refs,
      'code_sha256':code_sha,'environment_sha256':env_sha,
      'execution_mode':execution_mode,
    }

def seal_run_core(core:dict) -> dict:
    return dict(core,run_core_sha256=canonical_sha256(core))

def verify_run(paths, run_id:str, *, require_live_code:bool=True):
    failures=[]
    try: validate_identifier(run_id,field='run_id')
    except Exception as e: return {'status':'FAIL','run_id':run_id,'failures':[{'reason':'unsafe_run_id','error':str(e)}]}
    rd=paths.runs/run_id; rp=rd/'run.json'; pp=rd/'plan.json'; qp=rd/'question.json'; cp=rd/'run_core.json'
    for p,label in [(rp,'run'),(pp,'plan'),(qp,'question'),(cp,'run_core')]:
        if not p.is_file(): failures.append({'reason':'missing_record','record':label})
    if failures: return {'status':'FAIL','run_id':run_id,'failures':failures}
    try:
        rr=json.loads(rp.read_text()); plan=json.loads(pp.read_text()); q=json.loads(qp.read_text()); core=json.loads(cp.read_text())
    except Exception as e: return {'status':'FAIL','run_id':run_id,'failures':[{'reason':'json','error':str(e)}]}
    errs=validate(schema('UNIVERSAL_RUN_RECORD_SCHEMA_v0.17.json'),rr,raise_on_error=False)
    if errs: failures.append({'reason':'schema','errors':errs})
    try:
        reg=ProtocolRegistry(); desc=reg.get(plan['protocol_id'],version=plan.get('protocol_version'),descriptor_sha=plan.get('descriptor_sha256')); reg.verify_plan(plan,paths=paths)
        reg.verify_installed(paths,desc['descriptor_sha256'])
    except Exception as e: failures.append({'reason':'protocol_or_plan','error':str(e)}); desc=None
    observed_plan=canonical_sha256({k:v for k,v in plan.items() if k!='plan_sha256'})
    if plan.get('plan_sha256')!=observed_plan: failures.append({'reason':'plan_hash'})
    if q!=plan.get('question') or canonical_sha256(q)!=plan.get('question_sha256'): failures.append({'reason':'question_binding'})
    if desc:
        expected_core=immutable_run_core(plan=plan,descriptor=desc,input_refs=rr.get('input_artifacts',[]),execution_mode=rr.get('execution_mode') or {},code_sha=rr.get('code_identity',{}).get('source_sha256'),env_sha=rr.get('environment_identity',{}).get('runtime_sha256'))
        if core.get('run_core_sha256')!=canonical_sha256({k:v for k,v in core.items() if k!='run_core_sha256'}): failures.append({'reason':'run_core_hash'})
        if {k:v for k,v in core.items() if k!='run_core_sha256'}!=expected_core: failures.append({'reason':'run_core_binding'})
    # run fields must mirror core, detecting editable run.json tampering.
    mirrors={'run_id':rr.get('run_id'),'protocol':rr.get('protocol'),'subject':rr.get('subject'),'question_sha256':rr.get('question_sha256'),'plan_sha256':rr.get('plan_sha256'),'evidence':rr.get('evidence'),'input_artifacts':rr.get('input_artifacts'),'execution_mode':rr.get('execution_mode')}
    for k,v in mirrors.items():
        if core.get(k)!=v: failures.append({'reason':'run_record_tamper','field':k})
    if rr.get('run_core_sha256')!=core.get('run_core_sha256'): failures.append({'reason':'run_core_pointer'})
    if require_live_code:
        try:
            if rr.get('code_identity',{}).get('source_sha256')!=source_sha256(): failures.append({'reason':'live_code_identity'})
            if rr.get('environment_identity',{}).get('runtime_sha256')!=runtime_sha256(): failures.append({'reason':'live_environment_identity'})
        except Exception as e: failures.append({'reason':'identity_exception','error':str(e)})
    store=ArtifactStore(paths.store); ds=DatasetStore(store)
    for d in rr.get('input_artifacts',[]):
        v=ds.verify(d['dataset_sha256'])
        if v.get('status')!='PASS': failures.append({'reason':'input_dataset','detail':v})
    for a in rr.get('result_artifacts',[]):
        v=store.verify(a['sha256'],a.get('size_bytes'))
        if v.get('status')!='PASS': failures.append({'reason':'result_artifact','detail':v})
    # Reconstruct each current checkpoint and its direct dependency bindings.
    cm=CheckpointManager(rd,store)
    stage_map={s['stage_id']:s for s in plan.get('stages',[])}
    for sid,st in stage_map.items():
        try: c=cm.current(sid)
        except Exception as e: failures.append({'reason':'checkpoint','stage_id':sid,'error':str(e)}); continue
        if not c: failures.append({'reason':'missing_checkpoint','stage_id':sid}); continue
        dep=cm.dependency_bindings(st.get('depends_on',[]))
        if c.get('dependency_bindings')!=dep: failures.append({'reason':'checkpoint_dependency_lineage','stage_id':sid})
        if c.get('plan_sha256')!=plan.get('plan_sha256'): failures.append({'reason':'checkpoint_plan','stage_id':sid})
        if c.get('run_core_sha256')!=core.get('run_core_sha256'): failures.append({'reason':'checkpoint_run_core','stage_id':sid})
        if c.get('stage_spec_sha256')!=canonical_sha256(st): failures.append({'reason':'checkpoint_stage_spec','stage_id':sid})
    # claims referenced by run must exist and reciprocally cite this run.
    for ref in rr.get('claim_refs',[]):
        if isinstance(ref,str): cid,ver=ref,None
        else: cid,ver=ref.get('claim_id'),ref.get('version')
        if not cid: failures.append({'reason':'claim_ref_malformed'}); continue
        d=paths.store/'claims'/cid
        matches=[d/f'v{int(ver):04d}.json'] if ver else sorted(d.glob('*.json'))
        if not matches or not all(x.is_file() for x in matches): failures.append({'reason':'claim_ref_missing','claim_id':cid})
        for x in matches:
            co=json.loads(x.read_text())
            if run_id not in co.get('evidence_refs',[]): failures.append({'reason':'claim_ref_not_reciprocal','claim_id':cid})
    if rr.get('lifecycle')!='COMPLETE_VALID': failures.append({'reason':'lifecycle','observed':rr.get('lifecycle')})
    return {'status':'PASS' if not failures else 'FAIL','run_id':run_id,'failures':failures,'lifecycle':rr.get('lifecycle'),'run_core_sha256':core.get('run_core_sha256')}
