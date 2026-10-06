"""Dry exhaustive independent semantics and reference checks; stores only digest/counts."""
import ast,json,hashlib,copy
from pathlib import Path
from canonical_kernel import canonical_sha256,canonical_bytes
from recursive_attachment import extend
D=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
recipe=json.loads((D/'DRY_RECIPE.json').read_text());rh=sha((D/'DRY_RECIPE.json').read_bytes())
for k,n in [('parent_packet_sha256','PARENT_PACKET.json'),('primitive_packet_sha256','PRIMITIVE_PACKET.json'),('adapter_sha256','recursive_attachment.py'),('canonical_kernel_sha256','canonical_kernel.py'),('historical_oracle_sha256','historical_run_level.py')]:assert recipe[k]==sha((D/n).read_bytes())
parents=json.loads((D/'PARENT_PACKET.json').read_text())['rows'];events=json.loads((D/'PRIMITIVE_PACKET.json').read_text())['events'];assert len(parents)==1391 and len(events)==13
source=ast.parse((D/'historical_run_level.py').read_text());ns={};exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'dtup','canon','bridge','compose'}],type_ignores=[]),'historical_pure_constructor','exec'),ns)
def record(r):return (r[0],tuple(tuple(p) for p in r[1]),r[2],tuple(r[3]) if isinstance(r[3],list) else r[3],r[4])
oids=set();fids=set();boundaries=set();digest=hashlib.sha256();attempts=lawful=0;example=None
for p in parents:
 for e in events:
  for sa in range(4):
   for sb in range(3):
    attempts+=1;want=ns['compose'](record(p['ordered_record']),sa,record(e['record']),sb);row=extend(p,e,sa,sb,rh)
    assert (want is None)==(row is None)
    if row is None:continue
    lawful+=1
    assert canonical_bytes(want)==canonical_bytes(row['projected_boundary'])
    a=p['ordered_record'];b=e['record'];ordered=[a[0]|b[0],[v for i,v in enumerate(a[1]) if i!=sa]+[v for i,v in enumerate(b[1]) if i!=sb],min(a[2],b[2]),a[3]+[b[3]],int(a[4] and b[4])]
    assert row['ordered_record']==ordered
    assert row['parent_state_sha256']==canonical_sha256(p)
    assert row['identity']['parent_object_id']==p['object_id'] and row['identity']['parent_formation_id']==p['formation_id']
    assert row['external_port_origins']==[['parent',i] for i in range(4) if i!=sa]+[['attachment',i] for i in range(3) if i!=sb]
    assert row['bridge']==[['parent',sa],['attachment',sb]]
    assert row['target_origins']==[['parent',0],['parent',1],['attachment',0]]
    assert row['projected_port_to_ordered_port']==sorted(range(5),key=lambda i:(ordered[1][i],i))
    assert row['projected_target_to_ordered_target']==sorted(range(3),key=lambda i:(ordered[3][i],i))
    assert row['microscopic_realization_product_count']==p['microscopic_realization_product_count']*len(e['realizations'])
    assert row['object_id']==canonical_sha256(row['identity']) and row['object_id'] not in oids and row['formation_id'] not in fids
    oids.add(row['object_id']);fids.add(row['formation_id']);boundaries.add(canonical_sha256(row['projected_boundary']));digest.update(canonical_bytes(row)+b'\n');example=example or row
assert attempts==recipe['attempts']
rejected=[]
for key,value in [('object_id','bad'),('formation_id','bad'),('external_port_origins',[])]:
 bad=copy.deepcopy(parents[0]);bad[key]=value
 try:extend(bad,events[0],0,0,rh)
 except (ValueError,KeyError):rejected.append(key)
 else:raise AssertionError('BAD_PARENT_ACCEPTED:'+key)
for sa,sb in [(-1,0),(4,0),(0,3)]:
 try:extend(parents[0],events[0],sa,sb,rh)
 except ValueError:rejected.append('endpoint:'+str((sa,sb)))
 else:raise AssertionError('BAD_SLOT_ACCEPTED')
report={'status':'PASS_DRY_RECURSIVE_ATTACHMENT','recipe_sha256':rh,'parents_checked':len(parents),'native_events':len(events),'attempted_endpoint_pairs':attempts,'lawful_rooted_constructions':lawful,'formation_ids':len(fids),'projected_boundaries':len(boundaries),'ordered_stream_sha256':digest.hexdigest(),'all_lawful_rows_match_independent_original_constructor':True,'all_parent_and_port_origin_references_checked':True,'corruptions_rejected':rejected,'official_runtime_run':False,'master_admission':False,'historical_l13_lineage_recovered':False}
(D/'DRY_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');(D/'EXAMPLE_CHILD.json').write_text(json.dumps(example,indent=2)+'\n');print(json.dumps(report))
