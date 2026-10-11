import json,zipfile,io,hashlib
from pathlib import Path
B=Path(__file__).parent
A=Path('/workspace/scratch/87702f15343f/canonical_audit/ANCESTRY_ASSESSMENT.json')
ancestry=json.loads(A.read_text());ids=set(ancestry['O7_source_ids'])
archive=Path('/tmp/ig_engine0237/infinity_grid/resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
def locate(data):
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  for n in z.namelist():
   if n.endswith('O7_IMMUTABLE_SURVIVORS.json'):return z.read(n)
   if n.endswith('.zip'):
    r=locate(z.read(n))
    if r:return r
raw=locate(archive.read_bytes());survivors=json.loads(raw)['records']
with zipfile.ZipFile(B/'dev148.zip') as z:
 manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))
 for n,m in manifest['files'].items():
  data=z.read(n);assert sha(data)==m['sha256'] and len(data)==m['bytes'],n
 validation=json.loads(z.read('evidence/NATIVE_IDENTITY_VALIDATION.json'))
 deltas=json.loads(z.read('evidence/HISTORICAL_PROFILE_DELTAS.json'))
 assert deltas['historical_archive_sha256']==sha(archive.read_bytes())
 assert validation['status']=='PASS'
 mapped=[]
 for v in validation['records']:
  if v['kind']!='O7_SURVIVOR' or v['legacy_digest'] not in ids:continue
  row=survivors[v['index']]
  assert row['state_digest']==v['legacy_digest']
  assert row['lane']+'_PROFILE'==v['lane']
  assert len(row['parent_ids'])==len(v['parent_exact_colors'])
  assert all(v[k] for k in ['legacy_materialization_preserved','new_materialization_preserved','witness_verified','idempotent','order_orientation_invariant'])
  mapped.append({'immutable_row':row,'immutable_row_sha256':sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode()),'saved_native_validation':v})
 assert len(mapped)==6 and {m['immutable_row']['state_digest'] for m in mapped}==ids
 assert len({m['saved_native_validation']['global_identity'] for m in mapped})==6
 assert not ids.intersection(c['historical_state_alias'] for c in deltas['cases'])
 for n in ['BUNDLE_MANIFEST.json','evidence/NATIVE_IDENTITY_VALIDATION.json','evidence/HISTORICAL_PROFILE_DELTAS.json']: (B/Path(n).name).write_bytes(z.read(n))
result={'status':'PASS_SAVED_SIX_SEED_VALIDATION_BOUND','original_bundle_manifest_verified_files':len(manifest['files']),'bundle_sha256':sha((B/'dev148.zip').read_bytes()),'historical_archive_sha256':sha(archive.read_bytes()),'immutable_survivor_bytes_sha256':sha(raw),'ancestry_assessment_sha256':sha(A.read_bytes()),'mapped_seeds':mapped,'saved_corrected_identities_distinct':6,'representative_bytes_changed':sum(m['saved_native_validation']['legacy_representative_changed'] for m in mapped),'saved_resource_preservation_pass':6,'saved_witness_verification_pass':6,'native_runs_this_step':0,'full_G2_identity_migration_complete':False,'witnesses_reverified_this_step':False}
(B/'SIX_SEED_BINDING.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='mapped_seeds'},indent=2))
