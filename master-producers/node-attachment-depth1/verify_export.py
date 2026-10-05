"""Independent checks of saved native rows; never calls the producer evaluator."""
from pathlib import Path
import sys,json,hashlib,itertools,ast
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'engine'))
from infinity_grid.canon import canonical_sha256,canonical_bytes
sha=lambda b:hashlib.sha256(b).hexdigest()
packet=json.loads((R/'PRIMITIVE_PACKET.json').read_text());recipe=json.loads((R/'RECIPE_CONTRACT.json').read_text());rh=sha((R/'RECIPE_CONTRACT.json').read_bytes())
mapping=json.loads((R/'CHECKPOINT_READBACK_MAPPING.json').read_text());native=json.loads((R/'NATIVE_RESULT.json').read_text());events={e['event_id']:e for e in packet['events']}
source=(B/'node_input_audit/sources/run_level.py').read_bytes();assert sha(source)==recipe['historical_constructor_sha256']
tree=ast.parse(source.decode());pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'dtup','canon','bridge','compose'}],type_ignores=[]);oracle={};exec(compile(pure,'original_pure_constructor','exec'),oracle)
def rec(e):
 r=e['record'];return (r[0],tuple(tuple(p) for p in r[1]),r[2],r[3],r[4])
expected=set()
for a,b,sa,sb in itertools.product(events.values(),events.values(),range(3),range(3)):
 if oracle['compose'](rec(a),sa,rec(b),sb) is not None:expected.add((a['event_id'],b['event_id'],sa,sb))
rows=[]
for ref in native['result']['data_shards']:
 raw=Path(mapping[ref['sha256']]['path']).read_bytes();assert sha(raw)==ref['sha256'] and len(raw)==ref['size_bytes'];shard=json.loads(raw)
 assert shard['recipe_sha256']==rh and shard['primitive_packet_sha256']==sha((R/'PRIMITIVE_PACKET.json').read_bytes());rows.extend(shard['rows'])
seen=set();oids=set();fids=set();boundaries=set()
def verify(row):
 a,b=[events[c['event_id']] for c in row['components']];sa,sb=row['identity']['bridge_slots']
 comps=[{'role':role,'j3_id':e['j3_id'],'event_id':e['event_id'],'historical_selector':e['selector'],'option_realizations_ref':e['event_id']} for role,e in [('seed',a),('attachment',b)]]
 identity={'schema_id':'IG_NODE_ATTACHMENT_ROOTED_CONSTRUCTION_V1','recipe_sha256':rh,'components':comps,'bridge_slots':[sa,sb]}
 assert row['identity']==identity and row['components']==comps
 assert row['object_id']==canonical_sha256(identity)
 assert row['formation_id']==canonical_sha256({'schema_id':'IG_NODE_ATTACHMENT_FORMATION_V1','identity':identity})
 want=oracle['compose'](rec(a),sa,rec(b),sb);assert want is not None and canonical_bytes(want)==canonical_bytes(row['projected_boundary'])
 ordered=[a['record'][0]|b['record'][0],[a['record'][1][i] for i in range(3) if i!=sa]+[b['record'][1][i] for i in range(3) if i!=sb],min(a['record'][2],b['record'][2]),[a['record'][3],b['record'][3]],int(a['record'][4] and b['record'][4])]
 assert row['ordered_record']==ordered
 assert row['external_port_origins']==[['seed',i] for i in range(3) if i!=sa]+[['attachment',i] for i in range(3) if i!=sb]
 assert row['bridge']==[['seed',sa],['attachment',sb]]
 assert row['projected_port_to_ordered_port']==sorted(range(4),key=lambda i:(ordered[1][i],i))
 assert row['projected_target_to_component']==sorted(range(2),key=lambda i:(ordered[3][i],i))
 assert row['projection_profile']=='SORT_PORT_PM_AND_DESTINATIONS_WITH_EXPLICIT_PREIMAGE_V1'
 assert row['microscopic_realization_product_count']==len(a['realizations'])*len(b['realizations'])
 return (a['event_id'],b['event_id'],sa,sb)
for row in rows:
 key=verify(row);assert key not in seen and row['object_id'] not in oids and row['formation_id'] not in fids;seen.add(key);oids.add(row['object_id']);fids.add(row['formation_id']);boundaries.add(canonical_sha256(row['projected_boundary']))
assert seen==expected and len(rows)==1391 and len(boundaries)==108
# A copied row with a changed endpoint, lost component or bad projection must fail.
import copy
mutations=[]
for field,value in [('bridge',[['seed',99],['attachment',0]]),('components',[]),('projected_port_to_ordered_port',[0,0,0,0])]:
 bad=copy.deepcopy(rows[0]);bad[field]=value
 try:verify(bad)
 except (AssertionError,ValueError,IndexError):mutations.append(field)
 else:raise AssertionError('CORRUPTION_ACCEPTED')
report={'status':'PASS','read_source':'Drive raw readback shards','all_rows_independently_checked':len(rows),'full_lawful_endpoint_coverage':True,'formations':len(fids),'projected_boundaries':len(boundaries),'component_event_and_realization_reference_closure':True,'corruptions_rejected':mutations,'qualification_only':True}
(R/'NATIVE_EXPORT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
