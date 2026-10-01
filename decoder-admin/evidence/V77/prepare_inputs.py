from pathlib import Path
import json,hashlib,ast,shutil
B=Path('/workspace/scratch/a2e5e2576f17');D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');c=json.loads((R/'decoder-admin/CATALOG.json').read_text());p=json.loads((R/'decoder/qualification/PROFILE.json').read_text());rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths={'saved_stage_one_prerun':B/'v73_stage_one/saved_stage_one_prerun.readback.zip','saved_stage_one':B/'v73_stage_one/saved_stage_one_completed.readback.zip','saved_stage_four':B/'v74_stage_four/saved_stage_four_completed.readback.zip','saved_stage_outcome':B/'v75_outcome/saved_stage_outcome.readback.zip','saved_representative_qualification':B/'v76_representative/saved_representative_qualification.readback.zip'}
prior={r['sha256']:r for dirname in ['v73_stage_one','v74_stage_four','v75_outcome','v76_representative'] for r in json.loads((B/dirname/'CATALOG.json').read_text())}
for role,f in c['candidate_fixtures'].items():
 path=paths[role];assert path.stat().st_size==f['size_bytes'] and sha(path)==f['sha256'];assert f['source_sha256']==c['source']['source_sha256'];partsdir=D/'transport'/role;partsdir.mkdir(parents=True);parts=[]
 for part in f['parts']:
  rp=Path(prior[part['sha256']]['readback']);assert sha(rp)==part['sha256'] and rp.stat().st_size==part['size_bytes'];shutil.copyfile(rp,partsdir/part['sha256']);parts.append({'sha256':part['sha256'],'size_bytes':part['size_bytes'],'drive_file_id':part['drive_id']})
 m={'schema_id':'IG_DECODER_TRANSPORT_MANIFEST_V1','object':{'sha256':f['sha256'],'size_bytes':f['size_bytes']},'encoding':'RAW_CHUNKS','parts':parts};mf=partsdir/'MANIFEST.json';mf.write_text(json.dumps(m,indent=2)+'\n');rows.append({'logical_name':role,'path':str(path),'sha256':f['sha256'],'size_bytes':f['size_bytes'],'state':f['state'],'manifest':str(mf),'parts_directory':str(partsdir)})
for f in c['objects']:
 if f['role']!='historical_fixture':continue
 path=next((B/'attachments').rglob(f['filename']));assert sha(path)==f['sha256'] and path.stat().st_size==f['size_bytes'];rows.append({'logical_name':f['name'],'path':str(path),'sha256':f['sha256'],'size_bytes':f['size_bytes'],'drive_id':f['drive_id'],'state':'IMMUTABLE_HISTORICAL','visibility':f['visibility']})
assert {x['logical_name'] for x in rows}=={x['logical_name'] for x in p['required_fixtures']}
(D/'INPUT_BINDINGS.json').write_text(json.dumps(rows,indent=2)+'\n')
selectors=[s for g in p['group_order'] for s in p['groups'][g]['selectors']];active={x.relative_to(R/'decoder').as_posix() for x in (R/'decoder/tests').glob('test_*.py')};assert len(selectors)==len(set(selectors))==170 and set(selectors)==active
inventory=[]
for g in p['group_order']:
 for s in p['groups'][g]['selectors']:
  path=R/'decoder'/s;t=ast.parse(path.read_text());inventory.append({'group':g,'selector':s,'sha256':sha(path),'static_test_definitions':sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_') for n in ast.walk(t))})
(D/'SELECTOR_INVENTORY.json').write_text(json.dumps({'scope':'STATIC_AST_NOT_NATIVE_COLLECTION_OR_PASS','selectors':inventory,'group_counts':{g:len(p['groups'][g]['selectors']) for g in p['group_order']}},indent=2)+'\n');print('7 input hashes verified; 170 selectors exactly once; no tests executed')
