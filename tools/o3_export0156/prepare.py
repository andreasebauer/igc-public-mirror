from pathlib import Path
import json,hashlib,zipfile,shutil,ast
B=Path(__file__).resolve().parent;R=B.parent;S=next((R/'o3_scope0155/sources').glob('*V1_3*/*'));sha=lambda b:hashlib.sha256(b).hexdigest();items=[];blobs={}
def add(n,p):
 b=p.read_bytes();h=sha(b);blobs[h]=b;items.append({'name':n,'sha256':h,'bytes':len(b)})
for r in range(65):add('panel'+str(r),S/'inputs/source_panels'/f'r{r:03d}_selected_carriers.json.gz')
for n,p in [('bank',S/'inputs/O2_SOURCE_BANK.json'),('spec',S/'inputs/MATURE_NODE_ALGEBRA_SPEC.json'),('primitives',S/'inputs/FROZEN_PRIMITIVES.json'),('historical_graduation',S/'results/O3_GRADUATION_AUDIT_RESULT.json'),('contract',R/'o3_scope0155/SCOPE_CONTRACT.json'),('readiness',R/'o3_scope0155/READINESS_VALIDATION.json'),('payloads',R/'o3_scope0155/PAYLOAD_SHA256.json'),('components',R/'o3_scope0155/COMPONENT_ROWS.json'),('manifest',S/'provenance/BUNDLE_MANIFEST.json')]:add(n,p)
shutil.copy2(S/'code/o3_graduation_classification_audit_v1_3.py',B/'project/frozen_audit.py')
orig=next((R/'o3_scope0155/sources').glob('*V1_1*/*/parent_v1_1_snapshot/code/scout3_o3_roadmap_v1_0.py'));names={'sha_repr','_site_maps_cached','_site_maps','_entity_maps','base_automorphism_search_space','exact_canonical_base_key'};tree=ast.parse(orig.read_text());pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]);(B/'project/frozen_identity.py').write_text('import collections,functools,itertools,math,hashlib\n'+ast.unparse(pure)+'\n');(B/'project/__init__.py').write_text('')
root={'schema':'IG_SAVED_O3_TYPED_CARRIERS_EXPORT_V1','scope':'Selected r1..r64, external r0 seed, frozen anonymous O2 Q bank; missing microscopic O2 provenance excluded','content':items,'reader_bindings':{n:sha((B/'project'/n).read_bytes()) for n in ['frozen_audit.py','frozen_identity.py']},'original_identity_producer_sha256':sha(orig.read_bytes()),'catalog_sha256':sha((R/'o2_integrate0152/CATALOG_0143.json').read_bytes())};raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,6,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'catalog_sha256':root['catalog_sha256']};(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2))
v=json.load(open(R/'o3_scope0155/READINESS_VALIDATION.json'));seed=v['panels'][0]['carriers'];s=json.load(open(R/'o2_export0151/SPEC.json'));s.update(job_id='MASTER.O3.SAVED.TYPED.CARRIERS.EXPORT.READER.V1',project_source=str(B/'project'));s['question'].update(stage_id='MASTER:O3:SAVED:TYPED:CARRIERS:EXPORT:READER:V1',description=root['scope'],stopping_rule='First source hash, typed incidence, Q bank, identity, component or route mismatch stops. No generation or admission.');s['execution']['parameters']=meta;s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p.read_bytes())} for n,p in [('archive',B/'SCIENTIFIC_EXPORT.zip'),('catalog',R/'o2_integrate0152/CATALOG_0143.json')]];s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':x} for k,x in {'outcome':'PASS','carriers':4867-seed,'seed_carriers':seed,'parent_links':6031,'generation_calls':0,'components':4927,'typed_edges':155458,'bank_entries':256}.items()];(B/'SPEC.json').write_text(json.dumps(s,indent=2))
for n in ['operate.py','cold_ops.py']:shutil.copy2(R/'o2_export0151'/n,B/n)
print(meta)
