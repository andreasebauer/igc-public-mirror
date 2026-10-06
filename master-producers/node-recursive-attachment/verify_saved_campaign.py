"""Compare saved native campaign rows, in frozen oracle order, without generation."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(B/'engine'))
from infinity_grid.canon import canonical_bytes,canonical_sha256
sha=lambda raw:hashlib.sha256(raw).hexdigest()
load=lambda p:json.loads(p.read_text())
D=B/'l13_lineage';dry=load(D/'DRY_VERIFICATION.json');recipe=load(D/'DRY_RECIPE.json')
assert dry['status']=='PASS_DRY_RECURSIVE_ATTACHMENT'
assert sha((D/'DRY_RECIPE.json').read_bytes())==dry['recipe_sha256']
for k,n in [('parent_packet_sha256','PARENT_PACKET.json'),('primitive_packet_sha256','PRIMITIVE_PACKET.json')]:assert sha((D/n).read_bytes())==recipe[k]
parents=load(D/'PARENT_PACKET.json')['rows'];events=load(D/'PRIMITIVE_PACKET.json')['events']
pi={p['object_id']:i for i,p in enumerate(parents)};ei={e['event_id']:i for i,e in enumerate(events)}
assert len(pi)==1391 and len(ei)==13
rows={};oids=set();fids=set();boundaries=set();covered=set();proofs=[];stop=0
for t in range(1,7):
 R=B/'recursive_native'/('tranche01_repack' if t==1 else f'tranche{t:02d}')
 q=load(R/'RECIPE_CONTRACT.json');s=q['scope'];v=load(R/'SAVED_OUTPUT_VERIFICATION.json');c=load(R/'COLD_REUSE_RESULT.json');result=load(R/'NATIVE_RESULT.json');mapping=load(R/'CHECKPOINT_READBACK_MAPPING.json')
 assert s['parent_start']==stop;stop=s['parent_stop_exclusive']
 assert q['identity_recipe_sha256']==dry['recipe_sha256']
 for k in ['parent_packet_sha256','primitive_packet_sha256']:assert q[k]==recipe[k]
 assert v['status']=='PASS_COMPLETE_SAVED_OUTPUT_COMPARISON' and c['status']=='PASS' and c['generation_reexecuted'] is False and c['pending_bytes']==0
 for k in ['completion_sha256','result_sha256']:assert v[k]==c[k]==result[k]
 count=0;local_digest=hashlib.sha256()
 for ref in result['result']['data_shards']:
  raw=Path(mapping[ref['sha256']]['path']).read_bytes();assert sha(raw)==ref['sha256'] and len(raw)==ref['size_bytes'];shard=json.loads(raw)
  assert shard['recipe_sha256']==sha((R/'RECIPE_CONTRACT.json').read_bytes())
  for row in shard['rows']:
   i=row['identity'];p=pi[i['parent_object_id']];e=ei[i['attachment_event_id']];a,b=i['bridge_slots'];assert s['parent_start']<=p<s['parent_stop_exclusive'] and 0<=a<4 and 0<=b<3
   key=(p,e,a,b);assert key not in rows and row['object_id'] not in oids and row['formation_id'] not in fids
   assert i['recipe_sha256']==dry['recipe_sha256'];assert row['object_id']==canonical_sha256(i)
   encoded=canonical_bytes(row)+b'\n';rows[key]=encoded;local_digest.update(encoded);oids.add(row['object_id']);fids.add(row['formation_id']);boundaries.add(canonical_sha256(row['projected_boundary']));covered.add(p);count+=1
 assert count==v['rows_checked']==s['expected_lawful_rooted_constructions'];assert local_digest.hexdigest()==v['saved_order_stream_sha256']
 proofs.append({'tranche':t,'rows':count,'completion_sha256':result['completion_sha256'],'result_sha256':result['result_sha256']})
assert stop==len(parents) and covered==set(range(len(parents)))
digest=hashlib.sha256()
for key in sorted(rows):digest.update(rows[key])
assert len(rows)==len(oids)==len(fids)==dry['lawful_rooted_constructions']==198188
assert len(boundaries)==dry['projected_boundaries']==802
assert digest.hexdigest()==dry['ordered_stream_sha256']
out={'status':'PASS_COMPLETE_SIX_TRANCHE_SAVED_UNION','read_source':'Exact saved Drive raw readback shards','parents':len(covered),'native_events':len(events),'registered_attempts':len(parents)*len(events)*12,'rooted_constructions':len(rows),'unique_object_ids':len(oids),'unique_formation_ids':len(fids),'projected_boundaries':len(boundaries),'ordered_stream_sha256':digest.hexdigest(),'full_dry_stream_match':True,'no_missing_or_duplicate_constructions':True,'generation_reexecuted':False,'scientific_master_admission':False,'historical_l13_reproduced':False,'tranches':proofs}
(B/'recursive_native/CAMPAIGN_SAVED_UNION_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
