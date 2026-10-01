from pathlib import Path
import sys,json,hashlib,runpy
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev147_20261001');RT=Path('/tmp/ig_runtime_v55_fresh_20260930');JOB='RC.FIXTURE.STAGE.OUTCOME.V82.DEV147'
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
 spec={'schema_id':sub.SPEC_SCHEMA,'job_id':JOB,'engine_source':str(R/'decoder'),'project_source':str(D/'project'),'question':{'stage_id':'ENG:FIXTURE:STAGE_OUTCOME','description':'Dev147 exact 48-value modulo-7 expected-outcome fixture','outcomes':['EXPECTED_OTHER'],'stopping_rule':(D/'PREREGISTRATION.txt').read_text()},'execution':{'kind':'STAGE','handler_ref':'infinity_grid.controller_only_fixture:stage_handler','evaluator_refs':['infinity_grid.controller_only_fixture:partition_evaluator'],'parameters':{'values':list(range(48)),'modulus':7,'workers':1,'phase_id':'PARTITION'}},'resources':{'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1','memory_budget_bytes':1073741824,'start_method':'fork','workers':1,'workspace_budget_bytes':2147483648},'inputs':[],'environment':{'python':'3.13','requirements':[{k:r[k] for k in ['distribution','version','imports']} for r in profile['environment']],'artifacts':artifacts},'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'VALIDATION','required_artifacts':[],'result_checks':[{'pointer':'/outcome','equals':'PASS'},{'pointer':'/partition/task_count','equals':48},{'pointer':'/partition/class_count','equals':7}],'prerequisites':[],'preservation':{}}}
 write('CAPTURE_SPEC',spec);store=Path('/tmp/ig_fixture_v82_20261001/store');assert not store.exists();portable_registry.initialize(store,'Dev147 V82 expected-outcome fixture',R/'decoder');write('CAPTURE_SAVE_STATUS',sub.capture(store,spec));exit()
s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub,preservation as pr
from infinity_grid.v05_controller_event_loop import export_workspace,restore_workspace,validate_workspace_job,verified_completion,run_workspace_job
def paused(root):
 attempts=list((root/'runtime/attempts').rglob('*.json'));assert len(attempts)==1
 a=read(attempts[0]);assert a['status']=='PAUSED',a
 assert 'UNREGISTERED_SCIENTIFIC_OUTCOME' in str(a),a
 for base in ['runtime/intake/completed','runtime/intake/prepared_completions']:assert not list((root/base).rglob('*.json'))
 assert verified_completion(validate_workspace_job(root,JOB,check_loaded=False),allow_pending_checkpoint=True) is None
 return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in attempts}
if mode=='run':
 assert not sub.save_status(w)['pending_objects'] and not list((w/'runtime/attempts').rglob('*.json'))
 write('PREEXEC_RUNTIME',verify(RT))
 try:
  result=run_workspace_job(w,JOB)
 except Exception as exc:
  write('NATIVE_EXCEPTION',{'type':type(exc).__name__,'message':str(exc)})
  assert type(exc).__name__=='ControllerLoopError' and str(exc)=='UNREGISTERED_SCIENTIFIC_OUTCOME'
 else:
  write('UNEXPECTED_NATIVE_RESULT',result);raise RuntimeError('EXPECTED_REFUSAL_ABSENT')
 finally:write('POSTEXEC_RUNTIME',verify(RT))
 write('PAUSED_PROOF',{'fixture_acceptance':'EXPECTED_REFUSAL_VERIFIED','native_status':'PAUSED','attempt_hashes':paused(w),'published_completions':0,'prepared_completions':0,'native_attempts':1})
elif mode=='export_paused':
 assert not pr.status(w)['pending_objects'];assert paused(w)==read(D/'PAUSED_PROOF.json')['attempt_hashes'];write('PAUSED_EXPORT',export_workspace(w,D/'saved_stage_outcome.zip'))
elif mode=='restore_paused':
 e=read(D/'PAUSED_EXPORT.json');p=D/'saved_stage_outcome.readback.zip';assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'];dest=D/'paused_restore';rest=restore_workspace(p,dest,e['sha256']);assert paused(dest)==read(D/'PAUSED_PROOF.json')['attempt_hashes'];write('PAUSED_RESTORE_PROOF',{'restore':rest,'attempt_bytes_identical':True,'native_status':'PAUSED','published_completions':0,'prepared_completions':0,'native_attempts':1,'workload_executions':0})
else:raise RuntimeError('UNKNOWN_MODE')
