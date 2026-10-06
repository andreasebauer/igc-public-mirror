from pathlib import Path
import json,hashlib,zipfile,shutil
B=Path(__file__).resolve().parent;R=B.parent;S=next((R/'o_readiness0158/sources').glob('*v2.4_PHASE1*/*'));sha=lambda b:hashlib.sha256(b).hexdigest();items=[];blobs={}
def add(n,p):
 b=p.read_bytes();h=sha(b);blobs[h]=b;items.append({'name':n,'sha256':h,'bytes':len(b)})
for n,p in [('states',S/'04_RESULTS/O4_PHASE1_GENERATED_SELECTED_STATES.json'),('prototypes',S/'07_INPUTS/SELECTED_O3_PROTOTYPES.json'),('pin',S/'07_INPUTS/PHASE0_PINNED_ACTUAL_O3_COMPONENT.json'),('spec',S/'01_SPEC/O4_PHASE1_FORMAL_SPEC.json'),('algebra',S/'07_INPUTS/MATURE_NODE_ALGEBRA_SPEC.json'),('contract',R/'o_readiness0158/EXPORT_CONTRACT.json'),('readiness',R/'o_readiness0158/READINESS_VALIDATION.json'),('payloads',R/'o_readiness0158/O4_PAYLOAD_SHA256.json'),('source_manifest',S/'SHA256_MANIFEST.txt'),('selector',S/'07_INPUTS/O3_PROTOTYPE_SELECTION.json'),('features',S/'07_INPUTS/O3_PHASE1_ELIGIBLE_COMPONENT_FEATURES.csv.gz'),('templates',S/'07_INPUTS/MATURE_TEMPLATE_PETRI_TRANSITIONS.csv')]:add(n,p)
shutil.copy2(S/'03_CODE/o4_phase1_bounded_generation_audit.py',B/'project/frozen_o4.py');shutil.copytree(R/'o3_export0156/project',B/'project/previous_o3',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True);(B/'project/__init__.py').write_text('')
root={'schema':'IG_SAVED_O4_TYPED_CARRIERS_EXPORT_V1','scope':'Saved354 O4 states; two rank0 seeds external; exact admitted O3 component references; inherited external anonymous Q-bank boundary','content':items,'frozen_producer_sha256':sha((B/'project/frozen_o4.py').read_bytes()),'catalog_sha256':sha((R/'o3_integrate0157/CATALOG_0144.json').read_bytes()),'O3_bindings':json.load(open(R/'o3_export0156/EXPORT_BINDINGS.json'))};raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,6,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'catalog_sha256':root['catalog_sha256']};(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2))
s=json.load(open(R/'o3_export0156/SPEC.json'));s.update(job_id='MASTER.O4.SAVED.TYPED.CARRIERS.EXPORT.READER.V1',project_source=str(B/'project'));s['question'].update(stage_id='MASTER:O4:SAVED:TYPED:CARRIERS:EXPORT:READER:V1',description=root['scope'],stopping_rule='First unexplained source hash, lane/rank identity, typed incidence, O3 component binding or route mismatch stops; no generation/admission.');s['execution']['parameters']=meta;s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p.read_bytes())} for n,p in [('archive',B/'SCIENTIFIC_EXPORT.zip'),('catalog',R/'o3_integrate0157/CATALOG_0144.json'),('o3_archive',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')]];s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS','carriers':352,'seed_carriers':2,'typed_edges':1152,'components':442,'O3_owner_occurrences':1802,'generation_calls':0}.items()];(B/'SPEC.json').write_text(json.dumps(s,indent=2))
for n in ['operate.py','cold_ops.py']:shutil.copy2(R/'o3_export0156'/n,B/n)
print(meta)
