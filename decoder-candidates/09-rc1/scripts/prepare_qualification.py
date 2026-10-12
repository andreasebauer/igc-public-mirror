from pathlib import Path
import sys,json
h=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(h/'source'))
from infinity_grid import submission as sub,portable_registry as reg,sqlite_attestation as sqlite
from infinity_grid.runtime_capacity import host_memory_ceiling_bytes
from infinity_grid.canon import write_json_atomic
root=h/'qualification_r3';root.mkdir(exist_ok=False)
binding=root/'sqlite-binding.json';write_json_atomic(binding,{'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'REQUIRE_REVIEWED_WAL_BUILD','observation':sqlite.require_wal_fix()})
arts=[('sqlite_runtime_binding',binding),('descriptor_sample',h/'inputs/qualification_sample.json'),('full_domain_plan',h/'inputs/full_domain_plan.json'),('raw_descriptor_archive',h/'inputs/raw_descriptor_archive.zip'),('preserved_prefix_archive',h/'inputs/preserved_prefix_archive.zip'),('runtime_manifest',h/'runtime/V5_MANIFEST.json')]
spec={'schema_id':sub.SPEC_SCHEMA,'job_id':'DECODER09.RC1.HET4.QUALIFICATION320','engine_source':str(h/'source'),'project_source':str(h/'project'),
'question':{'stage_id':'ENG:DECODER09-RC1','description':'Integrated candidate exact320-row qualification;no scientific census dispatch','outcomes':['PASS'],'stopping_rule':'Exactly320registered qualification rows;not data adoption'},
'execution':{'kind':'STAGE','handler_ref':'project.bulk_adapter:handler','evaluator_refs':['project.bulk_adapter:evaluate'],'parameters':{'registered_rows':320,'batch_size':4,'workers':4,'window_index':None,'window_rows':16384}},
'resources':{'workers':4,'start_method':'spawn','memory_budget_bytes':host_memory_ceiling_bytes(),'workspace_budget_bytes':None,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},'inputs':[],
'environment':{'python':f'{sys.version_info.major}.{sys.version_info.minor}','requirements':[],'artifacts':[{'logical_name':n,'path':str(p),'sha256':sub._sha(p.read_bytes())} for n,p in arts]},
'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'EXECUTION_ONLY','required_artifacts':[],'result_checks':[{'pointer':'/outcome','equals':'PASS'}],'prerequisites':[],'preservation':{'project_evidence_externalization':1,'max_project_raw_bytes':None,'completion_project_closure':1,'rolling_unbounded_execution':1}}}
write_json_atomic(root/'SPECIFICATION.json',spec);store=root/'store';reg.initialize(store,'Decoder0.9rc1 integrated candidate',h/'source');capture=sub.capture(store,spec);work=Path(capture['workspace']);objects=[dict(o,path=str(store/'objects'/o['object_name'])) for o in sub.required_objects(work)]
write_json_atomic(h/'results/CAPTURE_PENDING.json',{'capture':capture,'workspace':str(work),'root':str(root),'obligations':objects})
print(json.dumps({'status':capture['status'],'native_dispatch':False,'objects':len(objects),'bytes':sum(x['size_bytes'] for x in objects),'source_sha256':sub.capture_record(work)['workspace']['source_sha256'],'package_sha256':sub.capture_record(work)['workspace']['package_sha256']}))
