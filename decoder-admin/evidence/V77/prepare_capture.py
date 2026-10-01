from pathlib import Path
import sys,json,hashlib,runpy
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');RT=Path('/tmp/ig_runtime_v55_fresh_20260930');GROUP=D.name;JOB='RC.'+GROUP.upper()+'.V77.DEV146'
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
 for name,p in [('sqlite_runtime_binding',D/'SQLITE_BINDING.json'),('host_runtime_manifest',R/'decoder-admin/RUNTIME_MANIFEST.json'),('runtime_mode_policy',R/'decoder-admin/RUNTIME_POLICY.json'),('qualification_protocol',D/'PREREGISTRATION.txt'),('operator_script',Path(__file__)),('input_bindings',D/'INPUT_BINDINGS.json'),('selector_inventory',D/'SELECTOR_INVENTORY.json'),('active_monitor',D/'monitor.py'),('native_runner',D/'run_native.py'),('pause_cli',R/'decoder-admin/request_pause.py')]:
  artifacts.append({'logical_name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 profile=read(R/'decoder/qualification/PROFILE.json')
 spec={'schema_id':sub.SPEC_SCHEMA,'job_id':JOB,'engine_source':str(R/'decoder'),'project_source':None,'question':{'stage_id':'RC:'+GROUP.upper()+':DEV146','description':'Dev146 full '+GROUP+' qualification','outcomes':['PASS'],'stopping_rule':(D/'PREREGISTRATION.txt').read_text()},'execution':{'kind':'VALIDATION','nodes':profile['groups'][GROUP]['selectors']},'resources':{'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1','memory_budget_bytes':2147483648,'start_method':'fork','workers':profile['groups'][GROUP]['workers'],'workspace_budget_bytes':8589934592},'inputs':[{k:r[k] for k in ['logical_name','path','sha256']} for r in read(D/'INPUT_BINDINGS.json')],'environment':{'python':'3.13','requirements':[{k:r[k] for k in ['distribution','version','imports']} for r in profile['environment']],'artifacts':artifacts},'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'VALIDATION','required_artifacts':[],'result_checks':[{'pointer':'/status','equals':'PASS'}],'prerequisites':[],'preservation':{}}}
 write('CAPTURE_SPEC',spec);store=Path('/tmp/ig_qualification_v77_20261001')/GROUP/'store';assert not store.exists();portable_registry.initialize(store,'Dev146 V77 '+GROUP+' qualification',R/'decoder');write('CAPTURE_SAVE_STATUS',sub.capture(store,spec));exit()
