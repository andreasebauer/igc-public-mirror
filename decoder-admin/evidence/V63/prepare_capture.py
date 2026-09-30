from pathlib import Path
import sys,json,hashlib,runpy
D=Path(__file__).resolve().parent
R=Path('/tmp/ig_runtime_v55_fresh_20260930');REPO=Path('/tmp/ig_decoder_dev144_20261001');source=REPO/'decoder'
if sys.flags.optimize or not sys.flags.dont_write_bytecode or Path(sys.executable).resolve()!=R/'base/bin/python3.13':raise RuntimeError('LAUNCH_IDENTITY')
verify=runpy.run_path(str(REPO/'decoder-admin/decoder.py'))['verify_runtime']
(D/'PREPARATION_RUNTIME_VERIFICATION.json').write_text(json.dumps(verify(R),indent=2))
runpy.run_path(str(REPO/'decoder-import/verify_source.py'))['verify']()
sys.path.insert(0,str(source))
from infinity_grid import submission as sub,portable_registry
from infinity_grid.sqlite_attestation import observe_runtime,require_wal_fix
binding={'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'REQUIRE_REVIEWED_WAL_BUILD','observation':require_wal_fix()}
(D/'SQLITE_BINDING.json').write_text(json.dumps(binding,sort_keys=True,indent=2)+'\n')
protocol=D/'PROTOCOL.txt'
artifacts=[]
for name,p in [('sqlite_runtime_binding',D/'SQLITE_BINDING.json'),('host_runtime_manifest',REPO/'decoder-admin/RUNTIME_MANIFEST.json'),('runtime_mode_policy',REPO/'decoder-admin/RUNTIME_POLICY.json'),('forensic_protocol',protocol),('operator_script_hashes',D/'OPERATOR_SCRIPT_HASHES.json')]:artifacts.append({'logical_name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
profile=json.loads((source/'qualification/PROFILE.json').read_text())
nodes=['tests/integration_probes/native_save_wave_probe.py::test_long_durable_progress_selector']
spec={'schema_id':sub.SPEC_SCHEMA,'job_id':'RC.CORRUPTION.V63.DEV144','engine_source':str(source),'project_source':None,'question':{'stage_id':'RC:DEV144:SAVE_WAVES','description':'V63 native isolated live-corruption observation gate','outcomes':['PASS','FAIL'],'stopping_rule':protocol.read_text()},'execution':{'kind':'VALIDATION','nodes':nodes},'resources':{'workers':1,'start_method':'fork','memory_budget_bytes':1073741824,'workspace_budget_bytes':2147483648,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},'inputs':[],'environment':{'python':'3.13','requirements':[{k:r[k] for k in ['distribution','version','imports']} for r in profile['environment']],'artifacts':artifacts},'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'VALIDATION','required_artifacts':[],'result_checks':[{'pointer':'/status','equals':'PASS'}],'prerequisites':[],'preservation':{'validation_wave_selectors':1}}}
(D/'CAPTURE_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
store=Path('/tmp/ig_gate_v63_20261001/store')
if store.exists():raise RuntimeError('NEW_STORE_REQUIRED')
portable_registry.initialize(store,'V63 dev144 save wave targeted gate',source)
r=sub.capture(store,spec);(D/'CAPTURE_SAVE_STATUS.json').write_text(json.dumps(r,indent=2)+'\n')
(D/'POSTPREPARATION_RUNTIME_VERIFICATION.json').write_text(json.dumps(verify(R),indent=2))
print(json.dumps({'capture_id':r['capture_id'],'workspace':r['workspace'],'pending':len(r['pending_objects'])}))
