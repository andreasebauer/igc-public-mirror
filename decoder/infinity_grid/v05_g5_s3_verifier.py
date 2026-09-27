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
VERIFIER_ID='v05.g5-s3.independent'
ORACLE_RESOURCE='resources/v05/G5_S3_PREREGISTERED_ORACLE_V1.json'
SPEC_RESOURCE='resources/uplift/G5_S3_HIGHER_ORDER_RESIDUAL_SPEC_V1.json'

def _check(checks,cid,ok,detail=None):
 r={'check_id':cid,'status':'PASS' if ok else 'FAIL'}
 if detail is not None:r['detail']=detail
 checks.append(r)

def _load_hashed_resource(name):
 o=json.loads(files('infinity_grid').joinpath(name).read_text()); d=o.get('science_sha256'); obs=canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})
 if d!=obs: raise RuntimeError(f'resource hash mismatch: {name}')
 return o

def _load_single_registration(root):
 found=sorted(p for p in Path(root).rglob('*.json') if p.is_file())
 if len(found)!=1: raise RuntimeError(f'expected exactly one workflow registration JSON, found {len(found)}')
 return validate_workflow_registration(json.loads(found[0].read_text()))

def _science_predicate(result,wf_reg,oracle,spec):
 d={}; st=result.get('stage_results') or []; d['stage_count']=len(st)
 if len(st)<4 or [x.get('stage') for x in st[:4]]!=['S0','S1','S2','S3'] or any(x.get('outcome')!='PASS' for x in st[:3]):
  return False,dict(d,failure='S0_S1_S2_S3_RESULT_MISSING')
 s3=st[3]; d.update(s3_outcome=s3.get('outcome'),s3_alternative=s3.get('observed_alternative'))
 if s3.get('outcome') not in oracle['allowed_s3_outcomes'] or s3.get('observed_alternative') not in oracle['allowed_s3_alternatives']:
  return False,dict(d,failure='S3_OUTCOME_OUTSIDE_ORACLE')
 if result.get('authority_effect')!=oracle['required_authority_effect'] or bool(result.get('graduated')):
  return False,dict(d,failure='AUTHORITY_OR_GRADUATION_MISMATCH')
 q=wf_reg.get('question',{}).get('stage_rows',[])[3]
 if q.get('parameters',{}).get('spec_science_sha256')!=spec['science_sha256'] or oracle.get('spec_science_sha256')!=spec['science_sha256']:
  return False,dict(d,failure='SPEC_BINDING_MISMATCH')
 for k in ('required_s2_primary_science_sha256','required_s2_workflow_science_sha256','required_s2_quotient_table_sha256','required_s1_rows_sha256'):
  if q.get('parameters',{}).get(k)!=oracle.get(k): return False,dict(d,failure=f'{k.upper()}_BINDING_MISMATCH')
 sr=s3.get('result') or {}
 if s3.get('outcome')=='PASS':
  required=(sr.get('s2_quotient_table_sha256')==oracle['required_s2_quotient_table_sha256'] and sr.get('public_carrier_class_count')==oracle['required_public_class_count'] and sr.get('exact_representatives_per_public_carrier_class')==oracle['required_exact_representatives_per_class'] and sr.get('local_two_reservation_row_count')==oracle['required_local_row_count'] and sr.get('local_representative_conflict_count')==0 and sr.get('all_public_classes_share_one_complete_local_continuation_vector_across_exact_representatives') is True and sr.get('total_public_context_count')==oracle['required_public_context_count'] and sr.get('missing_s2_edge_key_count')==0 and sr.get('factorization_complete') is True and sr.get('irreducible_triple_residual_count')==0 and sr.get('minimal_added_read_on_frozen_triple_observer')=='NONE_BEYOND_INHERITED_G4_PUBLIC_PLUS_S2_EDGE_OBSERVER')
  if not required:return False,dict(d,failure='HIGHER_ORDER_CERTIFICATE_MISMATCH')
  if any(bool(sr.get(k)) for k in ('hidden_topology_promoted','hidden_owner_identity_promoted','g4_changed','g5_public_descriptor_promoted','g5_composition_law_earned','g5_graduated')):
   return False,dict(d,failure='FORBIDDEN_PROMOTION_FLAG')
  if len(st)!=5 or st[4].get('stage')!='S4' or st[4].get('outcome')!='REVIEW_REQUIRED' or st[4].get('observed_alternative')!=oracle['required_post_s3_stop']:
   return False,dict(d,failure='S4_LOCK_MISMATCH')
  stop=result.get('stop_record') or {}
  if stop.get('stage')!='S4' or stop.get('reason')!='REVIEW_REQUIRED': return False,dict(d,failure='POST_S3_STOP_MISMATCH')
  d['scientific_disposition']='S3_NO_HIGHER_RESIDUAL_S4_REVIEW_REQUIRED'
 else:
  stop=result.get('stop_record') or {}
  if stop.get('stage')!='S3' or stop.get('reason') not in {'REVIEW_REQUIRED','BLOCKED'}: return False,dict(d,failure='S3_NONPASS_STOP_MISMATCH')
  d['scientific_disposition']='S3_REVIEW_OR_BLOCKED'
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
 if reg and len(reg.get('input_datasets',[]))==1:
  try:
   with tempfile.TemporaryDirectory(prefix='ig-g5-s3-verify-') as td: fixture=ds.materialize(reg['input_datasets'][0]['dataset_sha256'],Path(td)/'workflow'); wr=_load_single_registration(fixture); recomputed=ControlledWorkflowEngine().run(wr)
   _check(checks,'cold_recompute_exact',result is not None and canonical_sha256(result)==canonical_sha256(recomputed)); oracle=_load_hashed_resource(ORACLE_RESOURCE); spec=_load_hashed_resource(SPEC_RESOURCE); _check(checks,'preregistered_oracle_binding',reg.get('output_contract',{}).get('science_oracle')==oracle); ok,pd=_science_predicate(recomputed,wr,oracle,spec); _check(checks,'science_predicate',ok,pd); _check(checks,'stored_science_predicate',bool(result is not None and _science_predicate(result,wr,oracle,spec)[0])); _check(checks,'g4_authority_frozen',spec['authority']['g4_status']=='GRADUATED_AND_FROZEN' and spec['authority']['g4_public_descriptor']=='CAPS7_PLUS_H_CLASS_BAG'); _check(checks,'s2_evidence_binding',spec['authority']['g5_s2_primary_science_sha256']==oracle['required_s2_primary_science_sha256'] and spec['authority']['g5_s2_workflow_science_sha256']==oracle['required_s2_workflow_science_sha256'] and spec['authority']['g5_s2_quotient_table_sha256']==oracle['required_s2_quotient_table_sha256']); s3=(recomputed.get('stage_results') or [None,None,None,None])[3]; sr=(s3 or {}).get('result') or {}; _check(checks,'local_continuation_congruence',sr.get('local_two_reservation_row_count')==196 and sr.get('local_representative_conflict_count')==0 and sr.get('factorization_complete') is True)
  except Exception as e:
   for c in ('cold_recompute_exact','preregistered_oracle_binding','science_predicate','stored_science_predicate','g4_authority_frozen','s2_evidence_binding','local_continuation_congruence'):_check(checks,c,False,str(e))
 else:
  for c in ('cold_recompute_exact','preregistered_oracle_binding','science_predicate','stored_science_predicate','g4_authority_frozen','s2_evidence_binding','local_continuation_congruence'):_check(checks,c,False,'input registration unavailable')
 req=reg.get('verification_policy',{}).get('required_checks',[]) if reg else []; seen={c['check_id'] for c in checks}; _check(checks,'required_check_coverage',bool(req) and all(x in seen for x in req),{'required':req,'seen':sorted(seen)}); return _finish(run_id,es,reg,checks,recomputed,pd)

def _finish(run_id,es,reg,checks,recomputed,pd):
 req=reg.get('verification_policy',{}).get('required_checks',[]) if reg else []; cmap={c['check_id']:c['status'] for c in checks}; status='PASS' if req and all(c['status']=='PASS' for c in checks) and all(cmap.get(x)=='PASS' for x in req) and cmap.get('required_check_coverage')=='PASS' else 'FAIL'; base={'schema_id':VERIFICATION_SCHEMA,'contract_version':reg.get('contract_version') if reg else None,'verifier_id':VERIFIER_ID,'run_id':run_id,'registration_sha256':reg.get('registration_sha256') if reg else None,'envelope_sha256':es,'status':status,'comparison_mode':'COLD_RECOMPUTE_EXACT_PLUS_PREREGISTERED_G5_S3_PREDICATE','checks':checks,'required_checks':req,'verifier_implementation':{'kind':'G5_S3_COLD_RECOMPUTE_AND_PREREGISTERED_PREDICATE','independent_process_required':True},'invocation':{'pid':os.getpid(),'uid':os.getuid() if hasattr(os,'getuid') else None,'gid':os.getgid() if hasattr(os,'getgid') else None,'live_source_sha256':live_source_sha256(),'runtime_sha256':runtime_sha256()},'cold_recompute_science_sha256':recomputed.get('science_sha256') if recomputed else None,'scientific_disposition':(pd or {}).get('scientific_disposition'),'authority_effect':'NONE','limitations':['G5_S3_ONLY','FROZEN_P3_K3_CHALLENGE_DOMAIN_ONLY','NO_GLOBAL_G5_CONGRUENCE','NO_G5_COMPOSITION_LAW','NO_G5_DESCRIPTOR_PROMOTION','NO_G5_GRADUATION','NO_G4_CHANGE','NO_S4_EXECUTION']}; return dict(base,verification_sha256=canonical_sha256(base))

def _cli():
 from .paths import resolve_root
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--run-id',required=True); ap.add_argument('--report',required=True); ns=ap.parse_args(); r=verify(resolve_root(ns.root),ns.run_id); write_json_atomic(Path(ns.report),r); print(json.dumps({'status':r['status'],'verification_sha256':r['verification_sha256'],'scientific_disposition':r.get('scientific_disposition')},sort_keys=True)); return 0 if r['status']=='PASS' else 3
if __name__=='__main__': raise SystemExit(_cli())
