"""Independently verify every saved runtime row and full endpoint coverage."""
from pathlib import Path
import sys,json,hashlib,ast
R=Path(__file__).resolve().parent;B=R.parents[1];sys.path.insert(0,str(B/'engine'))
from infinity_grid.canon import canonical_sha256,canonical_bytes
sha=lambda b:hashlib.sha256(b).hexdigest()
recipe=json.loads((R/'RECIPE_CONTRACT.json').read_text());rh=sha((R/'RECIPE_CONTRACT.json').read_bytes());scope=recipe['scope'];mapping=json.loads((R/'CHECKPOINT_READBACK_MAPPING.json').read_text())
raw=(R/'PARENT_PACKET.json').read_bytes();assert sha(raw)==recipe['parent_packet_sha256'];parents={p['object_id']:p for p in json.loads(raw)['rows'][scope['parent_start']:scope['parent_stop_exclusive']]}
raw=(R/'PRIMITIVE_PACKET.json').read_bytes();assert sha(raw)==recipe['primitive_packet_sha256'];events={e['event_id']:e for e in json.loads(raw)['events']}
source=(B/'l13_lineage/historical_run_level.py').read_bytes();dry=json.loads((R/'DRY_RECIPE.json').read_text());assert sha(source)==dry['historical_oracle_sha256'];tree=ast.parse(source.decode());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'dtup','canon','bridge','compose'}],type_ignores=[]),'historical_pure_oracle','exec'),ns)
def record(r):return (r[0],tuple(tuple(p) for p in r[1]),r[2],tuple(r[3]) if isinstance(r[3],list) else r[3],r[4])
expected=set()
for p in parents.values():
 for e in events.values():
  for sa in range(4):
   for sb in range(3):
    if ns['compose'](record(p['ordered_record']),sa,record(e['record']),sb) is not None:expected.add((p['object_id'],e['event_id'],sa,sb))
seen=set();oids=set();fids=set();boundaries=set();count=0;digest=hashlib.sha256()
result=json.loads((R/'NATIVE_RESULT.json').read_text());assert result['evidence_status']=='VERIFIED'
for ref in result['result']['data_shards']:
 m=mapping[ref['sha256']];raw=Path(m['path']).read_bytes();assert sha(raw)==ref['sha256'] and len(raw)==ref['size_bytes'];shard=json.loads(raw);assert shard['recipe_sha256']==rh and shard['parent_packet_sha256']==recipe['parent_packet_sha256']
 for row in shard['rows']:
  i=row['identity'];p=parents[i['parent_object_id']];e=events[i['attachment_event_id']];sa,sb=i['bridge_slots'];key=(p['object_id'],e['event_id'],sa,sb);assert key in expected and key not in seen;seen.add(key)
  identity={'schema_id':'IG_NODE_RECURSIVE_ROOTED_ATTACHMENT_V1','recipe_sha256':recipe['identity_recipe_sha256'],'parent_object_id':p['object_id'],'parent_formation_id':p['formation_id'],'attachment_j3_id':e['j3_id'],'attachment_event_id':e['event_id'],'historical_selector':e['selector'],'bridge_slots':[sa,sb]};assert i==identity
  a=p['ordered_record'];b=e['record'];ordered=[a[0]|b[0],[a[1][k] for k in range(4) if k!=sa]+[b[1][k] for k in range(3) if k!=sb],min(a[2],b[2]),a[3]+[b[3]],int(a[4] and b[4])]
  po=sorted(range(5),key=lambda k:(ordered[1][k],k));to=sorted(range(3),key=lambda k:(ordered[3][k],k))
  want={'identity':identity,'object_id':canonical_sha256(identity),'formation_id':canonical_sha256({'schema_id':'IG_NODE_RECURSIVE_ROOTED_ATTACHMENT_V1_FORMATION','identity':identity}),'parent_state_sha256':canonical_sha256(p),'ordered_record':ordered,'projected_boundary':[ordered[0],[ordered[1][k] for k in po],ordered[2],[ordered[3][k] for k in to],ordered[4]],'projected_port_to_ordered_port':po,'projected_target_to_ordered_target':to,'external_port_origins':[['parent',k] for k in range(4) if k!=sa]+[['attachment',k] for k in range(3) if k!=sb],'bridge':[['parent',sa],['attachment',sb]],'target_origins':[['parent',0],['parent',1],['attachment',0]],'microscopic_realization_product_count':p['microscopic_realization_product_count']*len(e['realizations'])}
  assert row==want
  oracle=ns['compose'](record(a),sa,record(b),sb);assert canonical_bytes(oracle)==canonical_bytes(row['projected_boundary'])
  assert row['object_id'] not in oids and row['formation_id'] not in fids;oids.add(row['object_id']);fids.add(row['formation_id']);boundaries.add(canonical_sha256(row['projected_boundary']));digest.update(canonical_bytes(row)+b'\n');count+=1
assert seen==expected and count==scope['expected_lawful_rooted_constructions']==36352
assert canonical_sha256(sorted(boundaries))==recipe['projected_set_sha256'] and len(boundaries)==763
report={'status':'PASS_COMPLETE_SAVED_OUTPUT_COMPARISON','read_source':'Exact Drive raw readback shards','rows_checked':count,'full_endpoint_coverage':True,'projected_boundaries':len(boundaries),'all_original_constructor_results_match':True,'all_native_parent_and_port_origin_fields_match':True,'unique_objects_and_formations':count,'saved_order_stream_sha256':digest.hexdigest(),'official_master_admission':False,'completion_sha256':result['completion_sha256'],'result_sha256':result['result_sha256']}
(R/'SAVED_OUTPUT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
