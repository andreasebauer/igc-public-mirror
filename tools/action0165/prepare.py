from pathlib import Path
import json,hashlib,zipfile,shutil
from freeze import freeze
B=Path(__file__).resolve().parent;R=B.parent;S=next((R/'o6_readiness0164/sources').glob('*v2.6_PHASE1*/*'));sha=lambda b:hashlib.sha256(b).hexdigest();items=[];blobs={}
for n,p in [('states',S/'04_RESULTS/O6_PHASE1_GENERATED_SELECTED_STATES.json'),('prototypes',S/'08_INPUTS/O5_PHASE1_RECONSTRUCTED_PROTOTYPE_POOL.json'),('pin',S/'08_INPUTS/PINNED_ACTUAL_O5_COMPONENT.json'),('spec',S/'01_SPEC/O6_PHASE1_FORMAL_SPEC.json'),('algebra',S/'08_INPUTS/MATURE_NODE_ALGEBRA_SPEC.json'),('contract',R/'o6_readiness0164/EXPORT_CONTRACT.json'),('readiness',R/'o6_readiness0164/READINESS_VALIDATION.json'),('payloads',R/'o6_readiness0164/O6_PAYLOAD_SHA256.json'),('components',R/'o6_readiness0164/COMPONENTS.json'),('source_manifest',S/'07_PROVENANCE/SHA256_MANIFEST.txt'),('selector',S/'08_INPUTS/O6_PHASE1_PROTOTYPE_SELECTION.json'),('templates',S/'08_INPUTS/MATURE_TEMPLATE_PETRI_TRANSITIONS.csv')]:
 b=p.read_bytes();h=sha(b);blobs[h]=b;items.append({'name':n,'sha256':h,'bytes':len(b)})
original=S/'03_CODE/o6_phase1_bounded_generation_audit.py';pure=freeze(original,B/'project/frozen_o6.py');(B/'PURE_SOURCE_EXTRACTION.json').write_text(json.dumps(pure,indent=2));
for n,p in [('original_producer',original),('pure_extraction',B/'PURE_SOURCE_EXTRACTION.json')]:
 b=p.read_bytes();h=sha(b);blobs[h]=b;items.append({'name':n,'sha256':h,'bytes':len(b)})
shutil.copytree(R/'o5_export0162/project',B/'project/previous_o5',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True);(B/'project/__init__.py').write_text('')
root={'schema':'IG_SAVED_O6_TYPED_CARRIERS_EXPORT_V1','scope':'Saved134 O6 states;2 external rank0 seeds;exact O5 component/root links and nested typed incidence','content':items,'frozen_producer_sha256':sha((B/'project/frozen_o6.py').read_bytes()),'original_producer_sha256':sha(original.read_bytes()),'catalog_sha256':sha((R/'o5_integrate0163/CATALOG_0146.json').read_bytes()),'O5_bindings':json.load(open(R/'o5_export0162/EXPORT_BINDINGS.json'))};raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,6,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'catalog_sha256':root['catalog_sha256']};(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2))
s=json.load(open(R/'o5_export0162/SPEC.json'));s.update(job_id='MASTER.O6.SAVED.TYPED.CARRIERS.EXPORT.READER.V1',project_source=str(B/'project'));s['question'].update(stage_id='MASTER:O6:SAVED:TYPED:CARRIERS:EXPORT:READER:V1',description=root['scope'],stopping_rule='First unexplained source hash, identity, typed incidence, nested component binding or route mismatch stops;no generation/admission.');s['execution']['parameters']=meta;s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p.read_bytes())} for n,p in [('archive',B/'SCIENTIFIC_EXPORT.zip'),('catalog',R/'o5_integrate0163/CATALOG_0146.json'),('o5_archive',R/'o5_export0162/SCIENTIFIC_EXPORT.zip'),('o4_archive',R/'o4_export0159/SCIENTIFIC_EXPORT.zip'),('o3_archive',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')]];s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS','carriers':132,'seed_carriers':2,'typed_edges':432,'components':172,'O5_owner_occurrences':682,'generation_calls':0}.items()];(B/'SPEC.json').write_text(json.dumps(s,indent=2))
for n in ['operate.py','cold_ops.py']:shutil.copy2(R/'o4_export0159'/n,B/n)
print(meta)
