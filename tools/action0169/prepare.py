from pathlib import Path
import json,hashlib,zipfile,ast,shutil
B=Path(__file__).resolve().parent;R=B.parent;V=R/'o7_readiness0168';sha=lambda b:hashlib.sha256(b).hexdigest()
import sys
sys.path.insert(0,str(B));from project.previous_o6.reader import ast_bytes
original=next((R/'o7_readiness0167/sources').glob('*v0.2.4_C04*/*/02_CODE/o7_live_engine.py'));pure=B/'project/frozen_o7.py';f={n.name:n for n in ast.parse(pure.read_bytes()).body if isinstance(n,ast.FunctionDef)}
ex={'original_sha256':sha(original.read_bytes()),'pure_sha256':sha(pure.read_bytes()),'functions_ast_sha256':{n:sha(ast_bytes(v)) for n,v in f.items()}};(B/'PURE_EXTRACTION.json').write_text(json.dumps(ex,indent=2))
files={'states':R/'o7_readiness0167/SAVED_STATE_ROWS.json','occurrences':R/'o7_readiness0167/SAVED_STATE_OCCURRENCES.json','selected':V/'SELECTED.json','twins':V/'TWINS.json','pool':V/'POOL.json','selector':V/'SELECTOR.json','spec':V/'SPEC.json','readiness':V/'READINESS_VALIDATION.json','contract':V/'EXPORT_CONTRACT.json','bindings':V/'O6_BINDINGS.json','external_twins':V/'EXTERNAL_TWIN_BINDINGS.json','components':V/'COMPONENTS.json','resources':V/'RESOURCE_PAYLOAD_SHA256.json','original_producer':original,'pure_extraction':B/'PURE_EXTRACTION.json','source_refs':R/'o7_readiness0167/RECOVERED_SOURCE_REFS.json'}
blobs={};items=[]
for n,p in files.items():
 b=p.read_bytes();h=sha(b);items.append({'name':n,'sha256':h,'bytes':len(b)});blobs[h]=b
root={'schema':'IG_SAVED_O7_TYPED_CARRIERS_EXPORT_V1','scope':'74 saved O7 states,HET4 m7=0..5,HOM6 m7=0..1;external twins retained as source evidence','content':items,'O6_bindings':json.loads((R/'o6_export0165/EXPORT_BINDINGS.json').read_bytes()),'catalog_sha256':sha((R/'o6_integrate0166/CATALOG_0147.json').read_bytes())};raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,6,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'catalog_sha256':root['catalog_sha256']};(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2))
s=json.loads((R/'o6_export0165/SPEC.json').read_bytes());s.update(job_id='MASTER.O7.SAVED.SCOPED.TYPED.CARRIERS.EXPORT.READER.V1',project_source=str(B/'project'));s['question'].update(stage_id='MASTER:O7:SAVED:SCOPED:TYPED:CARRIERS:EXPORT:READER:V1',description=root['scope']);s['execution']['parameters']=meta;s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p.read_bytes())} for n,p in [('archive',B/'SCIENTIFIC_EXPORT.zip'),('catalog',R/'o6_integrate0166/CATALOG_0147.json'),('o6_archive',R/'o6_export0165/SCIENTIFIC_EXPORT.zip'),('o5_archive',R/'o5_export0162/SCIENTIFIC_EXPORT.zip'),('o4_archive',R/'o4_export0159/SCIENTIFIC_EXPORT.zip'),('o3_archive',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')]];s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS','carriers':72,'seed_carriers':2,'typed_edges':192,'components':82,'O6_owner_occurrences':322,'generation_calls':0}.items()];(B/'SPEC.json').write_text(json.dumps(s,indent=2))
for n in ['operate.py','cold_ops.py']:shutil.copy2(R/'o6_export0165'/n,B/n)
print(json.dumps(meta))
