from __future__ import annotations
import argparse,json,os,tempfile
from importlib.resources import files
from pathlib import Path
from .canon import canonical_sha256,write_json_atomic
from .checkpoints import CheckpointManager
from .datasets import DatasetStore
from .records import live_source_sha256,runtime_sha256
from .store import ArtifactStore
from .v05 import V05RegistrationStore
from .v05_evidence import ENVELOPE_SCHEMA
from .v05_verifier import VERIFICATION_SCHEMA
from .v05_workflow import ControlledWorkflowEngine,validate_workflow_registration

VERIFIER_ID='v05.g5-s5.independent'
ORACLE_RESOURCE='resources/v05/G5_S5_PREREGISTERED_ORACLE_V1.json'
SPEC_RESOURCE='resources/uplift/G5_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1.json'

def _check(checks,cid,ok,detail=None):
    r={'check_id':cid,'status':'PASS' if ok else 'FAIL'}
    if detail is not None:r['detail']=detail
    checks.append(r)

def _load_hashed_resource(name):
    o=json.loads(files('infinity_grid').joinpath(name).read_text()); d=o.get('science_sha256')
    if d!=canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'}): raise RuntimeError(f'resource hash mismatch: {name}')
    return o

def _load_single_registration(root):
    found=sorted(p for p in Path(root).rglob('*.json') if p.is_file())
    if len(found)!=1: raise RuntimeError(f'expected exactly one workflow registration JSON, found {len(found)}')
    return validate_workflow_registration(json.loads(found[0].read_text()))

def _science_predicate(result,wf_reg,oracle,spec):
    d={}; st=result.get('stage_results') or []; d['stage_count']=len(st)
    if len(st)<6 or [x.get('stage') for x in st[:6]]!=['S0','S1','S2','S3','S4','S5'] or any(x.get('outcome')!='PASS' for x in st[:5]):
        return False,dict(d,failure='S0_S5_RESULT_CHAIN_MISSING')
    s5=st[5]; sr=s5.get('result') or {}; d.update(s5_outcome=s5.get('outcome'),s5_alternative=s5.get('observed_alternative'))
    if s5.get('outcome') not in oracle['allowed_s5_outcomes'] or s5.get('observed_alternative') not in oracle['allowed_s5_alternatives']:
        return False,dict(d,failure='S5_OUTCOME_OUTSIDE_ORACLE')
    if result.get('authority_effect')!=oracle['required_authority_effect'] or bool(result.get('graduated')):
        return False,dict(d,failure='AUTHORITY_OR_GRADUATION_MISMATCH')
    q=wf_reg.get('question',{}).get('stage_rows',[])[5]; params=q.get('parameters',{})
    if params.get('spec_science_sha256')!=spec['science_sha256'] or oracle.get('spec_science_sha256')!=spec['science_sha256']:
        return False,dict(d,failure='SPEC_BINDING_MISMATCH')
    for k in ('required_s1_rows_sha256','required_s2_quotient_table_sha256','required_s3_stage_science_sha256','required_s4_stage_science_sha256'):
        if params.get(k)!=oracle.get(k): return False,dict(d,failure=f'{k.upper()}_BINDING_MISMATCH')
    if st[4].get('science_sha256')!=oracle['required_s4_stage_science_sha256'] or st[4].get('observed_alternative')!=oracle['required_s4_classification']:
        return False,dict(d,failure='S4_PARENT_AUTHORITY_MISMATCH')
    if s5.get('outcome')=='PASS':
        req=(len(sr.get('H_class_alphabet_rows') or [])==oracle['required_H_class_alphabet_count'] and sr.get('H_class_alphabet_failure_count')==oracle['required_failure_count'] and sr.get('directed_operator_count')==oracle['required_directed_operator_count'] and sr.get('s1_pair_write_check_count')==oracle['required_s1_pair_write_checks'] and sr.get('s1_pair_write_failure_count')==oracle['required_failure_count'] and sr.get('s2_public_pair_key_count')==oracle['required_s2_public_pair_keys'] and sr.get('s2_single_value_failure_count')==oracle['required_failure_count'] and sr.get('s4_recursive_write_check_count')==oracle['required_s4_recursive_write_checks'] and sr.get('s4_recursive_write_failure_count')==oracle['required_failure_count'] and sr.get('coverage_complete') is True and sr.get('g5_finite_state_descriptor_earned') is oracle['required_g5_finite_state_descriptor_earned'] and sr.get('g5_finite_read_write_law_earned') is oracle['required_g5_finite_read_write_law_earned'] and sr.get('earned_descriptor')==oracle['required_candidate_descriptor'])
        if not req:return False,dict(d,failure='S5_FINITE_STATE_CERTIFICATE_MISMATCH')
        surf=sr.get('implementation_read_surface_audit') or {}
        if any(bool(surf.get(k)) for k in ('hidden_topology_read','hidden_owner_identity_read','construction_identity_read','ancestry_read','exact_relation_cardinality_read','branch_multiplicity_read')):
            return False,dict(d,failure='FORBIDDEN_ABSTRACT_READ')
        if bool(sr.get('g5_public_descriptor_changed')) or bool(sr.get('hidden_topology_promoted')) or bool(sr.get('hidden_owner_identity_promoted')) or bool(sr.get('g4_changed')) or bool(sr.get('g5_global_composition_law_promoted')) or bool(sr.get('g5_graduated')):
            return False,dict(d,failure='FORBIDDEN_PROMOTION_FLAG')
        if len(st)!=7 or st[6].get('stage')!='S6' or st[6].get('outcome')!='REVIEW_REQUIRED' or st[6].get('observed_alternative')!=oracle['required_post_s5_stop']:
            return False,dict(d,failure='S6_LOCK_MISMATCH')
        stop=result.get('stop_record') or {}
        if stop.get('stage')!='S6' or stop.get('reason')!='REVIEW_REQUIRED': return False,dict(d,failure='POST_S5_STOP_MISMATCH')
        d['scientific_disposition']='S5_FINITE_READ_WRITE_STATE_PASS_S6_REVIEW_REQUIRED'
    else:
        stop=result.get('stop_record') or {}
        if stop.get('stage')!='S5' or stop.get('reason') not in {'REVIEW_REQUIRED','BLOCKED'}: return False,dict(d,failure='S5_NONPASS_STOP_MISMATCH')
        d['scientific_disposition']='S5_REVIEW_OR_BLOCKED'
    return True,d

def verify(paths,run_id):
    checks=[]; rd=Path(paths.runs)/run_id; reqp={n:rd/n for n in ('run.json','plan.json','run_core.json','v05_envelope.json')}; _check(checks,'records_present',all(p.is_file() for p in reqp.values()))
    if not all(p.is_file() for p in reqp.values()): return _finish(run_id,None,None,checks,None,None)
    try: run=json.loads(reqp['run.json'].read_text()); plan=json.loads(reqp['plan.json'].read_text()); core=json.loads(reqp['run_core.json'].read_text()); env=json.loads(reqp['v05_envelope.json'].read_text()); _check(checks,'records_json',True)
    except Exception as e: _check(checks,'records_json',False,str(e)); return _finish(run_id,None,None,checks,None,None)
    es=env.get('envelope_sha256'); _check(checks,'envelope_hash',env.get('schema_id')==ENVELOPE_SCHEMA and es==canonical_sha256({k:v for k,v in env.items() if k!='envelope_sha256'})); _check(checks,'execution_terminal',run.get('lifecycle')==env.get('execution_lifecycle')=='COMPLETE_VALID')
    reg=None
    try:
        rs=plan.get('v05',{}).get('registration_sha256'); reg=V05RegistrationStore(paths.store).get(rs); _check(checks,'registration_binding',env.get('registration_sha256')==rs and reg.get('runner')=='adapter.v05_workflow' and reg.get('verification_policy',{}).get('verifier_id')==VERIFIER_ID and reg.get('protocol',{}).get('protocol_id')=='G5_WORKFLOW')
    except Exception as e:_check(checks,'registration_binding',False,str(e))
    if reg:
        _check(checks,'source_identity',reg['source_sha256']==run.get('code_identity',{}).get('source_sha256')==env.get('source_sha256')==live_source_sha256()); _check(checks,'environment_identity',reg['environment_sha256']==run.get('environment_identity',{}).get('runtime_sha256')==env.get('environment_sha256')==runtime_sha256()); _check(checks,'input_binding',reg['input_datasets']==plan.get('input_datasets')==run.get('input_artifacts')==env.get('input_artifacts')); _check(checks,'subject_binding',reg.get('subject')==plan.get('subject')==run.get('subject') and reg.get('subject',{}).get('phase')=='G5')
    else:
        for c in ('source_identity','environment_identity','input_binding','subject_binding'):_check(checks,c,False,'registration unavailable')
    _check(checks,'plan_hash',plan.get('plan_sha256')==canonical_sha256({k:v for k,v in plan.items() if k!='plan_sha256'})==env.get('plan_sha256')); _check(checks,'run_core_hash',core.get('run_core_sha256')==canonical_sha256({k:v for k,v in core.items() if k!='run_core_sha256'})==run.get('run_core_sha256')==env.get('run_core_sha256'))
    store=ArtifactStore(paths.store); ds=DatasetStore(store); _check(checks,'input_dataset_bytes',all(ds.verify(d['dataset_sha256']).get('status')=='PASS' for d in run.get('input_artifacts',[])))
    result=None; refs=run.get('result_artifacts',[]); logical=reg.get('output_contract',{}).get('logical_outputs',[]) if reg else []; _check(checks,'logical_output_set',len(logical)==1 and {x.get('logical_name') for x in refs}==set(logical)); ref=next((x for x in refs if logical and x.get('logical_name')==logical[0]),None)
    if ref:
        try:_check(checks,'result_artifact_bytes',store.verify(ref['sha256'],ref.get('size_bytes')).get('status')=='PASS'); result=json.loads(store.blob_path(ref['sha256']).read_text()); _check(checks,'result_json',result.get('science_sha256')==canonical_sha256({k:v for k,v in result.items() if k!='science_sha256'}))
        except Exception as e:_check(checks,'result_artifact_bytes',False,str(e)); _check(checks,'result_json',False,str(e))
    else:_check(checks,'result_artifact_bytes',False,'missing'); _check(checks,'result_json',False,'missing')
    try:
        cm=CheckpointManager(rd,store); cp=cm.current(reg['stage_id']) if reg else None; ptr=cm.current_pointer(reg['stage_id']) if reg else None; hits=[x for x in env.get('checkpoint_bindings',[]) if reg and x.get('stage_id')==reg['stage_id']]; ok=bool(cp and ptr and cp.get('status')=='COMPLETE_VALID' and len(hits)==1 and hits[0]['checkpoint_sha256']==ptr['checkpoint_sha256'] and hits[0]['checkpoint_content_sha256']==cp['checkpoint_content_sha256']); _check(checks,'checkpoint_binding',ok); wb=(cp or {}).get('stage_result',{}).get('worker_boundary',{}); _check(checks,'worker_boundary',bool(reg and wb.get('kind')=='POSIX_DROP_PRIVILEGE' and wb.get('uid')==reg['worker_policy']['uid'] and wb.get('gid')==reg['worker_policy']['gid'] and wb.get('store_write_access')=='DENIED_BY_POSIX' and wb.get('publication_write_access')=='DENIED_BY_POSIX'))
    except Exception as e:_check(checks,'checkpoint_binding',False,str(e)); _check(checks,'worker_boundary',False,str(e))
    recomputed=None; pd=None
    extra=('cold_recompute_exact','preregistered_oracle_binding','science_predicate','stored_science_predicate','g4_authority_frozen','s4_evidence_binding','finite_state_write_law','coverage_exact','read_surface','post_s5_lock')
    if reg and len(reg.get('input_datasets',[]))==1:
        try:
            with tempfile.TemporaryDirectory(prefix='ig-g5-s5-verify-') as td: fixture=ds.materialize(reg['input_datasets'][0]['dataset_sha256'],Path(td)/'workflow'); wr=_load_single_registration(fixture); recomputed=ControlledWorkflowEngine().run(wr)
            _check(checks,'cold_recompute_exact',result is not None and canonical_sha256(result)==canonical_sha256(recomputed)); oracle=_load_hashed_resource(ORACLE_RESOURCE); spec=_load_hashed_resource(SPEC_RESOURCE); _check(checks,'preregistered_oracle_binding',reg.get('output_contract',{}).get('science_oracle')==oracle); ok,pd=_science_predicate(recomputed,wr,oracle,spec); _check(checks,'science_predicate',ok,pd); _check(checks,'stored_science_predicate',bool(result is not None and _science_predicate(result,wr,oracle,spec)[0])); _check(checks,'g4_authority_frozen',spec['authority']['g4_status']=='GRADUATED_AND_FROZEN' and spec['authority']['g4_public_descriptor']=='CAPS7_PLUS_H_CLASS_BAG'); _check(checks,'s4_evidence_binding',spec['authority']['g5_s4_stage_science_sha256']==oracle['required_s4_stage_science_sha256'] and spec['authority']['g5_s4_classification']==oracle['required_s4_classification']); s5=(recomputed.get('stage_results') or [None]*6)[5]; sr=(s5 or {}).get('result') or {}; _check(checks,'finite_state_write_law',sr.get('g5_finite_state_descriptor_earned') is True and sr.get('g5_finite_read_write_law_earned') is True and sr.get('earned_descriptor')=='CAPS7_PLUS_H_CLASS_BAG' and sr.get('s1_pair_write_failure_count')==0 and sr.get('s4_recursive_write_failure_count')==0); _check(checks,'coverage_exact',sr.get('s1_pair_write_check_count')==496 and sr.get('s2_public_pair_key_count')==124 and sr.get('s4_recursive_write_check_count')==7688 and sr.get('coverage_complete') is True); surf=sr.get('implementation_read_surface_audit') or {}; _check(checks,'read_surface',not any(bool(surf.get(k)) for k in ('hidden_topology_read','hidden_owner_identity_read','construction_identity_read','ancestry_read','exact_relation_cardinality_read','branch_multiplicity_read'))); _check(checks,'post_s5_lock',bool(ok and pd and pd.get('scientific_disposition')=='S5_FINITE_READ_WRITE_STATE_PASS_S6_REVIEW_REQUIRED'))
        except Exception as e:
            for c in extra:_check(checks,c,False,str(e))
    else:
        for c in extra:_check(checks,c,False,'input registration unavailable')
    req=reg.get('verification_policy',{}).get('required_checks',[]) if reg else []; seen={c['check_id'] for c in checks}; _check(checks,'required_check_coverage',bool(req) and all(x in seen for x in req),{'required':req,'seen':sorted(seen)}); return _finish(run_id,es,reg,checks,recomputed,pd)

def _finish(run_id,es,reg,checks,recomputed,pd):
    req=reg.get('verification_policy',{}).get('required_checks',[]) if reg else []; cmap={c['check_id']:c['status'] for c in checks}; status='PASS' if req and all(c['status']=='PASS' for c in checks) and all(cmap.get(x)=='PASS' for x in req) and cmap.get('required_check_coverage')=='PASS' else 'FAIL'; base={'schema_id':VERIFICATION_SCHEMA,'contract_version':reg.get('contract_version') if reg else None,'verifier_id':VERIFIER_ID,'run_id':run_id,'registration_sha256':reg.get('registration_sha256') if reg else None,'envelope_sha256':es,'status':status,'comparison_mode':'COLD_RECOMPUTE_EXACT_PLUS_PREREGISTERED_G5_S5_PREDICATE','checks':checks,'required_checks':req,'verifier_implementation':{'kind':'G5_S5_COLD_RECOMPUTE_PLUS_FINITE_READ_WRITE_PREDICATE','independent_process_required':True},'invocation':{'pid':os.getpid(),'uid':os.getuid() if hasattr(os,'getuid') else None,'gid':os.getgid() if hasattr(os,'getgid') else None,'live_source_sha256':live_source_sha256(),'runtime_sha256':runtime_sha256()},'cold_recompute_science_sha256':recomputed.get('science_sha256') if recomputed else None,'scientific_disposition':(pd or {}).get('scientific_disposition'),'authority_effect':'NONE','limitations':['G5_S5_ONLY','FROZEN_S1_S2_S4_EVIDENCE_DOMAIN_ONLY','NO_GLOBAL_G5_CONGRUENCE','NO_UNBOUNDED_RECURSIVE_CLOSURE','NO_GLOBAL_G5_COMPOSITION_LAW_PROMOTION','NO_G5_PUBLIC_DESCRIPTOR_CHANGE','NO_G5_GRADUATION','NO_G4_CHANGE']}; return dict(base,verification_sha256=canonical_sha256(base))

def _cli():
    from .paths import resolve_root
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--run-id',required=True); ap.add_argument('--report',required=True); ns=ap.parse_args(); r=verify(resolve_root(ns.root),ns.run_id); write_json_atomic(Path(ns.report),r); print(json.dumps({'status':r['status'],'verification_sha256':r['verification_sha256'],'scientific_disposition':r.get('scientific_disposition')},sort_keys=True)); return 0 if r['status']=='PASS' else 3
if __name__=='__main__': raise SystemExit(_cli())
