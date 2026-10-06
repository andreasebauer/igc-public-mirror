from pathlib import Path
import json,hashlib,zipfile,shutil
B=Path(__file__).resolve().parent;R=B.parent;sha=lambda b:hashlib.sha256(b).hexdigest();items=[];blobs={}
S=next(next(p for p in (R/'inter_node0150/sources').iterdir() if 'IN11_' in p.name).iterdir())
def add(name,p):
 b=p.read_bytes();h=sha(b);blobs[h]=b;items.append({'name':name,'sha256':h,'bytes':len(b),'source_member':str(p.relative_to(R))})
for d in range(77,110):
 add('panel'+str(d),S/f'checkpoints/d{d:03d}_selected_carriers.json.gz');add('observer'+str(d),S/f'checkpoints/d{d:03d}_observer_v11.json')
for name,p in [('contract',R/'inter_node0150/EXPORT_CONTRACT.json'),('readiness',R/'inter_node0150/READINESS_VALIDATION.json'),('source_manifest',R/'inter_node0150/SOURCE_MANIFEST.json'),('historical_result',S/'results/IN11_RESULT.json'),('historical_candidate',S/'results/O2_MATURATION_CANDIDATE_V1_1.json')]:add(name,p)
shutil.copy2(S/'code/in9_relational_longitudinal_scout.py',B/'project/frozen_in9.py');(B/'project/__init__.py').write_text('')
root={'schema':'IG_O2_SELECTED_CARRIERS_EXPORT_V1','scope':'Saved selected carriers d78-d109; external d77 seed; original observer sidecars and metadata; no full census or fresh graduation','content':items,'canonicalizer_sha256':sha((B/'project/frozen_in9.py').read_bytes()),'catalog_sha256':sha((R/'node_integrate0149/CATALOG_0142.json').read_bytes())};raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,6,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'catalog_sha256':root['catalog_sha256']};(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2))
s=json.load(open(R/'node_export0148/SPEC.json'));s.update(job_id='MASTER.O2.SELECTED.CARRIERS.EXPORT.READER.V1',project_source=str(B/'project'));s['question'].update(stage_id='MASTER:O2:SELECTED:CARRIERS:EXPORT:READER:V1',description=root['scope'],stopping_rule='First unexplained source hash, identity, capacity, sidecar or route mismatch stops. No generation or admission.');s['execution']['parameters']=meta;s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p.read_bytes())} for n,p in [('archive',B/'SCIENTIFIC_EXPORT.zip'),('catalog',R/'node_integrate0149/CATALOG_0142.json')]];s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS','carriers':25474,'seed_carriers':793,'parent_links':63885,'generation_calls':0}.items()];(B/'SPEC.json').write_text(json.dumps(s,indent=2))
for n in ['operate.py','cold_ops.py']:shutil.copy2(R/'node_export0148'/n,B/n)
print(meta)
