import json,hashlib,zipfile
from pathlib import Path
B=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')).encode()
prior=json.loads(Path('/workspace/scratch/87702f15343f/seed_binding/SIX_SEED_BINDING.json').read_text())
assert sha((B/'P0_HANDOFF.zip').read_bytes())=='6e10e8d965b835b5486011a085c9c40cb54410ef116a16530635b52db5e71ce1'
with zipfile.ZipFile(B/'P0_HANDOFF.zip') as z:
 manifest=json.loads(z.read('MANIFEST.json'))
 for n,m in manifest.items():
  data=z.read(n);assert sha(data)==m['sha256'] and len(data)==m['bytes'],n
 table=[json.loads(x) for x in z.read('results/IDENTITY_MAPPING.jsonl').splitlines()]
 rows=[]
 for seed in prior['mapped_seeds']:
  idx=seed['saved_native_validation']['index'];name=f'results/rows/{idx:04d}.json';saved=json.loads(z.read(name));row=saved['row'];source=seed['immutable_row'];old=seed['saved_native_validation']
  assert sha(canon(row))==saved['row_sha256']
  assert row['record']==f'O7_SURVIVOR:{idx:03d}' and row['role']=='retained'
  assert row['historical_state_alias']==source['state_digest']
  assert row['parent_ids']==source['parent_ids'] and row['raw_edges']==source['edges']
  assert row['parent_colors']==old['parent_exact_colors']
  assert row['old_context_identity']==old['global_identity']
  assert row['new_identity_schema']=='IG_E7_GLOBAL_INCIDENCE_IDENTITY_V3' and row['witness_verified']
  assert [r for r in table if r['record']==row['record'] and r['role']=='retained']==[row]
  (B/f'SEED_{idx:03d}_P0_ROW.json').write_bytes(z.read(name));rows.append(row)
 for n in ['SOURCE_POINTERS.json','TABLE_VERIFICATION.json','transition/TRANSITION_DECISION.json']: (B/Path(n).name).write_bytes(z.read(n))
 witness_files=[n for n in z.namelist() if 'witness' in n.lower()]
 assert len({r['new_global_identity'] for r in rows})==6
 result={'status':'PASS_SIX_EXACT_SEED_ROWS_BOUND_TO_SAVED_V3_MAPPINGS','P0_handoff_sha256':sha((B/'P0_HANDOFF.zip').read_bytes()),'original_manifest_files_verified':len(manifest),'retained_mapping_rows':len(table),'seeds':rows,'full_witness_objects_in_handoff':witness_files,'saved_witness_hashes_bound':6,'full_witness_objects_recovered':0,'witnesses_independently_reverified_this_step':0,'native_runs':0,'migration_activated':False,'limitation':'Rows store witness_sha256 and witness_verified, not relabeling objects; cannot transport reservation endpoints from hashes','next_scope':'Recover full six witness objects from exact source-bound previous computation if retained; otherwise narrowly regenerate only six witnesses under registered target runtime, not observer campaign'}
 (B/'V3_BINDING_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='seeds'},indent=2))
