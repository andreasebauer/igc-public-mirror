from pathlib import Path
import sys,json,hashlib,runpy
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');RT=Path('/tmp/ig_runtime_v55_fresh_20260930');JOB='RC.FIXTURE.REPRESENTATIVE.V76.DEV146'
assert not sys.flags.optimize and sys.flags.dont_write_bytecode
assert Path(sys.executable).resolve()==RT/'base/bin/python3.13'
def read(p):return json.loads(p.read_text())
def write(n,x):
 with (D/(n+'.json')).open('x') as f:json.dump(x,f,indent=2)
verify=runpy.run_path(str(R/'decoder-admin/decoder.py'))['verify_runtime'];mode=sys.argv[1]
if mode=='prepare':
 write('PRE_RUNTIME',verify(RT));runpy.run_path(str(R/'decoder-import/verify_source.py'))['verify']()
 sys.path.insert(0,str(R/'decoder'))
 from infinity_grid import submission as sub,portable_registry
 from infinity_grid.sqlite_attestation import require_wal_fix
 write('SQLITE_BINDING',{'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'REQUIRE_REVIEWED_WAL_BUILD','observation':require_wal_fix()})
 artifacts=[]
 for name,p in [('sqlite_runtime_binding',D/'SQLITE_BINDING.json'),('host_runtime_manifest',R/'decoder-admin/RUNTIME_MANIFEST.json'),('runtime_mode_policy',R/'decoder-admin/RUNTIME_POLICY.json'),('qualification_protocol',D/'PREREGISTRATION.txt'),('operator_script',Path(__file__))]:
  artifacts.append({'logical_name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 profile=read(R/'decoder/qualification/PROFILE.json')
 spec={'schema_id':sub.SPEC_SCHEMA,'job_id':JOB,'engine_source':str(R/'decoder'),'project_source':str(D/'project'),'question':{'stage_id':'ENG:REPRESENTATIVE_QUALIFICATION','description':'Dev146 four-case representative qualification fixture','outcomes':['PASS'],'stopping_rule':(D/'PREREGISTRATION.txt').read_text()},'execution':{'kind':'STAGE','handler_ref':'infinity_grid.representative_qualification:handler','evaluator_refs':['infinity_grid.controller_only_fixture:partition_evaluator'],'parameters':{}},'resources':{'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1','memory_budget_bytes':1073741824,'start_method':'fork','workers':4,'workspace_budget_bytes':2147483648},'inputs':[],'environment':{'python':'3.13','requirements':[{k:r[k] for k in ['distribution','version','imports']} for r in profile['environment']],'artifacts':artifacts},'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'VALIDATION','required_artifacts':[],'result_checks':[{'pointer':'/outcome','equals':'PASS'},{'pointer': '/cases', 'equals': {'test_257_distinct_then_duplicate_needs_no_controller_reopen': 'PASS', 'test_cold_multicore_over_256_classes_keeps_exact_grouping_and_durable_bytes': 'PASS', 'test_partial_resume_reuses_durable_rep_bytes_without_reopen': 'PASS', 'test_legacy_null_rep_binds_view_once_backfills_then_reuses': 'PASS'}}],'prerequisites':[],'preservation':{'interval_seconds':30}}}
 write('CAPTURE_SPEC',spec);store=Path('/tmp/ig_fixture_v76_20261001/store');assert not store.exists();portable_registry.initialize(store,'Dev146 V76 representative qualification fixture',R/'decoder');write('CAPTURE_SAVE_STATUS',sub.capture(store,spec));exit()
s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub,preservation as pr
from infinity_grid.v05_controller_event_loop import export_workspace,restore_workspace,validate_workspace_job,verified_completion,run_workspace_job
if mode=='export_prerun':
 assert not sub.save_status(w)['pending_objects']
 for base in ['runtime/attempts','runtime/intake/completed','runtime/intake/prepared_completions']:assert not list((w/base).rglob('*.json'))
 write('PRERUN_EXPORT',export_workspace(w,D/'saved_stage_four_prerun.zip'))
elif mode=='run':
 assert not sub.save_status(w)['pending_objects']
 assert not list((w/'runtime/attempts').rglob('*.json'))
 write('PREEXEC_RUNTIME',verify(RT))
 try:write('NATIVE_RESULT',run_workspace_job(w,JOB))
 except BaseException as exc:write('NATIVE_EXCEPTION',{'type':type(exc).__name__,'message':str(exc)});raise
 finally:write('POSTEXEC_RUNTIME',verify(RT))
elif mode=='export_completed':
 assert not pr.status(w)['pending_objects'];done=verified_completion(validate_workspace_job(w,JOB,check_loaded=False));assert done and pr.terminal_completion_proof(w,done);write('VERIFIED_COMPLETION',done);write('COMPLETED_EXPORT',export_workspace(w,D/'saved_representative_qualification.zip'))
elif mode=='restore_completed':
 e=read(D/'COMPLETED_EXPORT.json');p=D/'saved_representative_qualification.readback.zip';assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'];dest=D/'completed_restore';rest=restore_workspace(p,dest,e['sha256']);done=verified_completion(validate_workspace_job(dest,JOB,check_loaded=False));assert done==read(D/'VERIFIED_COMPLETION.json') and pr.terminal_completion_proof(dest,done);write('COMPLETED_RESTORE_PROOF',{'restore':rest,'exact_completion_equal':True,'workload_executions':0,'completion_sha256':done['completion_sha256']})
else:raise RuntimeError('UNKNOWN_MODE')
