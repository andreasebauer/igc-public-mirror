from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source'
sys.path.insert(0,str(SOURCE))
from infinity_grid import submission as sub,portable_registry as reg,sqlite_attestation as sqlite,v05_controller_event_loop as loop
from infinity_grid.canon import write_json_atomic
ROOT=Path('/tmp/ig_p0_closure181_r1')
ROOT.mkdir(exist_ok=False)
binding=ROOT/'sqlite-binding.json'
write_json_atomic(binding,{'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'REQUIRE_REVIEWED_WAL_BUILD','observation':sqlite.require_wal_fix()})
sample=HERE/'results/sample.json'
spec={'schema_id':sub.SPEC_SCHEMA,'job_id':'P0.CLOSURE181.HET4.BOUNDED320','engine_source':str(SOURCE),'project_source':str(HERE/'project'),
'question':{'stage_id':'ENG:HET4-CLOSURE','description':'Exact full-witness adapter parity and durable reuse, 320 frozen sample descriptors','outcomes':['PASS'],'stopping_rule':'Complete the 320 registered sample rows; no scientific adoption'},
'execution':{'kind':'STAGE','handler_ref':'project.bulk_adapter:handler','evaluator_refs':['project.bulk_adapter:evaluate'],'parameters':{'registered_rows':320,'batch_size':16,'workers':4}},
'resources':{'workers':4,'start_method':'spawn','memory_budget_bytes':8589934592,'workspace_budget_bytes':134217728,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},
'inputs':[],'environment':{'python':f'{sys.version_info.major}.{sys.version_info.minor}','requirements':[],'artifacts':[{'logical_name':'sqlite_runtime_binding','path':str(binding),'sha256':sub._sha(binding.read_bytes())},{'logical_name':'descriptor_sample','path':str(sample),'sha256':sub._sha(sample.read_bytes())}]},
'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'EXECUTION_ONLY','required_artifacts':[],'result_checks':[{'pointer':'/outcome','equals':'PASS'}],'prerequisites':[],'preservation':{'project_evidence_externalization':1,'max_project_raw_bytes':None,'completion_project_closure':1}}}
write_json_atomic(HERE/'results/SPECIFICATION.json',spec)
store=ROOT/'store';reg.initialize(store,'Decoder checkpoint closure181 candidate',SOURCE)
capture=sub.capture(store,spec)
work=Path(capture['workspace']);rows=[]
for obj in sub.required_objects(work):
 rows.append(dict(obj,path=str(store/'objects'/obj['object_name'])))
write_json_atomic(HERE/'results/CAPTURE_PENDING.json',{'capture':capture,'workspace':str(work),'root':str(ROOT),'obligations':rows})
print(json.dumps({'status':capture['status'],'objects':len(rows),'workspace':str(work),'bytes':sum(x['size_bytes'] for x in rows)}))
