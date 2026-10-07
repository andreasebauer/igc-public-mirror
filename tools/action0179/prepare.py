"""Freeze a byte-only reader contract and native capture specification."""
from pathlib import Path
import hashlib,json,shutil
B=Path(__file__).resolve().parent;R=B.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
(B/'inputs').mkdir(exist_ok=True)
for n in ['G2_S0_INTERFACE_POPULATION.json','POST_RESERVATION_CONTINUATION_HASHES.json']:shutil.copy2(R/'g_sufficiency0178/inputs'/n,B/'inputs'/n)
shutil.copy2(R/'g_sufficiency0178/LOCAL_SUFFICIENCY_GATE.json',B/'inputs/LOCAL_SUFFICIENCY_GATE.json')
shutil.copy2(R/'o7_additional_integrate0174/CATALOG_0149.json',B/'inputs/CATALOG_0149.json')
pins={n:{'sha256':sha(B/'inputs'/f),'bytes':(B/'inputs'/f).stat().st_size} for n,f in [('population','G2_S0_INTERFACE_POPULATION.json'),('continuations','POST_RESERVATION_CONTINUATION_HASHES.json')]}
bindings={n:sha(p) for n,p in [('contract',B/'EXPORT_CONTRACT.json'),('gate',B/'inputs/LOCAL_SUFFICIENCY_GATE.json'),('catalog',B/'inputs/CATALOG_0149.json')]}
paths={'population':B/'inputs/G2_S0_INTERFACE_POPULATION.json','continuations':B/'inputs/POST_RESERVATION_CONTINUATION_HASHES.json','contract':B/'EXPORT_CONTRACT.json','gate':B/'inputs/LOCAL_SUFFICIENCY_GATE.json','catalog':B/'inputs/CATALOG_0149.json'}
s=json.loads((R/'o7_additional_export0173/SPEC.json').read_text())
for row in s['environment']['artifacts']:
 p=R/'runtime'/Path(row['path']).name
 row['path']=str(p);row['sha256']=sha(p)
s.update(job_id='MASTER.G1.SAVED.PUBLIC.PROJECTION.READER.V1',project_source=str(B/'project'))
s['question'].update(stage_id='MASTER:G1:SAVED:PUBLIC:PROJECTION:READER:V1',description='193 saved G1 public interfaces and1351 reservation/continuation projections only',stopping_rule='Any saved hash, semantic or route mismatch stops; no exact state reconstruction or master admission')
s['execution']['parameters']={'pins':pins,'bindings':bindings}
s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p)} for n,p in paths.items()]
s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS','interfaces':193,'interface_classes':192,'reservation_rows':1351,'continuation_hashes':1351,'authority':'SAVED_PUBLIC_PROJECTION_ONLY','generation_calls':0,'scientific_master_admission':False,'exact_parent_DAG_available':False,'Q2_payload_available':False}.items()]
dump(B/'SPEC.json',s)
dump(B/'INPUT_BINDINGS.json',{n:{'relative_path':str(p.relative_to(B)),'sha256':sha(p),'bytes':p.stat().st_size} for n,p in paths.items()})
dump(B/'PROJECT_REGISTRATION.json',{'schema':'IG_FINITE_EXPORT_PROJECT_REGISTRATION_V1','job_id':s['job_id'],'status':'CONTRACT_AND_HANDLER_FROZEN_NOT_YET_CAPTURED','handler_ref':s['execution']['handler_ref'],'spec_sha256':sha(B/'SPEC.json'),'project_code':{str(p.relative_to(B)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (B/'project').glob('*.py')},'capture_required':True,'native_execution_started':False,'master_admission':False,'generation_calls':0,'next_scope':'WP6_CAPTURE_SAVED_G1_PUBLIC_PROJECTION_READER'})
