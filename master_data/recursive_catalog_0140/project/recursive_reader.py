"""Pinned read-only recursive qualification view; no generation or admission."""
from pathlib import Path
import json,hashlib
from infinity_grid.canon import canonical_bytes,canonical_sha256

class IntegrityError(ValueError):pass
def check(ok,reason):
 if not ok:raise IntegrityError(reason)
def strict(raw):
 def pairs(items):
  out={}
  for k,v in items:
   check(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(IntegrityError('NONFINITE_JSON')))
def read(path,sha,size=None,limit=4194304):
 if isinstance(path,bytes):
  raw=path;check(len(raw)<=limit,'OBJECT_LIMIT')
 else:
  p=Path(path);check(p.is_file() and not p.is_symlink(),'MISSING_OR_UNSAFE_OBJECT')
  check(p.stat().st_size<=limit,'OBJECT_LIMIT');raw=p.read_bytes()
 check(hashlib.sha256(raw).hexdigest()==sha and (size is None or len(raw)==size),'OBJECT_HASH_OR_SIZE');return strict(raw)
def lawful(a,b,sa,sb):
 pa,ma=a[1][sa];pb,mb=b[1][sb];return (ma==0 or ma&pb) and (mb==0 or mb&pa)

class RecursiveReader:
 def __init__(self,root_path,root_sha256,locations):
  self.root=read(root_path,root_sha256,limit=1048576);q=self.root
  check(q['schema_id']=='IG_RECURSIVE_QUALIFIED_VIEW_V1' and q['master_admission'] is False,'VIEW_AUTHORITY')
  self.locations=dict(locations);self.rows={};self.formations={};self.preimages={};self.depths={};self.closed=False
  self.packet=read(self.locations[q['primitive_packet_sha256']],q['primitive_packet_sha256']);self.events={e['event_id']:e for e in self.packet['events']}
  check(len(self.events)==13 and self.packet['accepted_catalog_sha256']==q['catalog_sha256'],'PRIMITIVE_CLOSURE')
  for e in self.events.values():check(e['realization_count']==len(e['realizations'])>0,'REALIZATION_CLOSURE')
  self.parents=read(self.locations[q['parent_packet_sha256']],q['parent_packet_sha256'])['rows'];self.parent_order={p['object_id']:i for i,p in enumerate(self.parents)};self.event_order={e['event_id']:i for i,e in enumerate(self.packet['events'])}
  seen={1:set(),2:set()};counts={1:0,2:0};streams=[]
  for s in q['sources']:
   depth=s['depth'];recipe=s['recipe_sha256'];digest=hashlib.sha256();count=0
   for ref in s['shards']:
    shard=read(self.locations[ref['sha256']],ref['sha256'],ref['size_bytes']);check(shard['recipe_sha256']==recipe,'SHARD_RECIPE');check(shard[s['packet_field']]==s['packet_sha256'],'SHARD_PACKET')
    for row in shard['rows']:
     key=self.validate(row,depth);check(key not in seen[depth],'DUPLICATE_ENDPOINT');seen[depth].add(key)
     check(row['object_id'] not in self.rows and row['formation_id'] not in self.formations,'DUPLICATE_ID')
     if depth==2:check(s['parent_start']<=self.parent_order[key[0]]<s['parent_stop_exclusive'],'PARENT_INTERVAL')
     self.rows[row['object_id']]=row;self.formations[row['formation_id']]=row['object_id'];self.depths[row['object_id']]=depth;h=canonical_sha256(row['projected_boundary']);self.preimages.setdefault((depth,h),[]).append(row['object_id']);digest.update(canonical_bytes(row)+b'\n');count+=1
   check(count==s['rows'],'SOURCE_COUNT')
   if 'saved_stream_sha256' in s:check(digest.hexdigest()==s['saved_stream_sha256'],'SAVED_ROW_STREAM')
   counts[depth]+=count;streams.append(digest.hexdigest())
  check(counts=={1:1391,2:198188},'CAMPAIGN_COUNTS')
  check(len(self.parents)==1391 and all(self.rows[p['object_id']]==p for p in self.parents),'PARENT_PACKET_EXACT_CLOSURE')
  expected1={(a['event_id'],b['event_id'],sa,sb) for a in self.events.values() for b in self.events.values() for sa in range(3) for sb in range(3) if lawful(a['record'],b['record'],sa,sb)}
  expected2={(p['object_id'],e['event_id'],sa,sb) for p in self.parents for e in self.events.values() for sa in range(4) for sb in range(3) if lawful(p['ordered_record'],e['record'],sa,sb)}
  check(seen[1]==expected1 and seen[2]==expected2,'FULL_ENDPOINT_COVERAGE')
  ordered=sorted((r for oid,r in self.rows.items() if self.depths[oid]==2),key=lambda r:(self.parent_order[r['identity']['parent_object_id']],self.event_order[r['identity']['attachment_event_id']],*r['identity']['bridge_slots']))
  digest=hashlib.sha256()
  for row in ordered:digest.update(canonical_bytes(row)+b'\n')
  check(digest.hexdigest()==q['full_depth2_stream_sha256'],'FULL_DRY_STREAM')
  check(sum(d==1 for d,h in self.preimages)==108 and sum(d==2 for d,h in self.preimages)==802,'PROJECTION_COUNTS')
  self.report={'status':'PASS_FULL_QUALIFIED_VIEW','depth1':1391,'depth2':198188,'all_rows_and_dependency_fields_checked':199579,'depth1_boundaries':108,'depth2_boundaries':802,'full_depth2_stream_sha256':digest.hexdigest(),'generator_calls':0,'master_admission':False,'historical_selection_weights_qualified':False}
 def validate(self,row,depth):
  i=row['identity'];sa,sb=i['bridge_slots'];check(type(sa) is int and type(sb) is int,'ENDPOINT_TYPE')
  if depth==1:
   check(0<=sa<3 and 0<=sb<3,'ENDPOINT_RANGE');a,b=[self.events[c['event_id']] for c in row['components']]
   comps=[{'role':role,'j3_id':e['j3_id'],'event_id':e['event_id'],'historical_selector':e['selector'],'option_realizations_ref':e['event_id']} for role,e in [('seed',a),('attachment',b)]]
   identity={'schema_id':'IG_NODE_ATTACHMENT_ROOTED_CONSTRUCTION_V1','recipe_sha256':self.root['depth1_identity_recipe_sha256'],'components':comps,'bridge_slots':[sa,sb]};left=a['record'];right=b['record'];targets=[left[3],right[3]];roles=['seed','attachment'];product=a['realization_count']*b['realization_count'];fs='IG_NODE_ATTACHMENT_FORMATION_V1';key=(a['event_id'],b['event_id'],sa,sb)
  else:
   check(0<=sa<4 and 0<=sb<3,'ENDPOINT_RANGE');p=self.rows[i['parent_object_id']];e=self.events[i['attachment_event_id']];check(self.depths[p['object_id']]==1,'PARENT_DEPTH')
   identity={'schema_id':'IG_NODE_RECURSIVE_ROOTED_ATTACHMENT_V1','recipe_sha256':self.root['depth2_identity_recipe_sha256'],'parent_object_id':p['object_id'],'parent_formation_id':p['formation_id'],'attachment_j3_id':e['j3_id'],'attachment_event_id':e['event_id'],'historical_selector':e['selector'],'bridge_slots':[sa,sb]};left=p['ordered_record'];right=e['record'];targets=left[3]+[right[3]];roles=['parent','attachment'];product=p['microscopic_realization_product_count']*e['realization_count'];fs='IG_NODE_RECURSIVE_ROOTED_ATTACHMENT_V1_FORMATION';key=(p['object_id'],e['event_id'],sa,sb)
  check(lawful(left,right,sa,sb),'UNLAWFUL_BRIDGE')
  ordered=[left[0]|right[0],[x for k,x in enumerate(left[1]) if k!=sa]+[x for k,x in enumerate(right[1]) if k!=sb],min(left[2],right[2]),targets,int(left[4] and right[4])];po=sorted(range(len(ordered[1])),key=lambda k:(ordered[1][k],k));to=sorted(range(len(targets)),key=lambda k:(targets[k],k))
  want={'identity':identity,'object_id':canonical_sha256(identity),'formation_id':canonical_sha256({'schema_id':fs,'identity':identity}),'ordered_record':ordered,'projected_boundary':[ordered[0],[ordered[1][k] for k in po],ordered[2],[targets[k] for k in to],ordered[4]],'projected_port_to_ordered_port':po,'external_port_origins':[[roles[0],k] for k in range(len(left[1])) if k!=sa]+[[roles[1],k] for k in range(3) if k!=sb],'bridge':[[roles[0],sa],[roles[1],sb]],'microscopic_realization_product_count':product}
  if depth==1:want.update(components=comps,projection_profile='SORT_PORT_PM_AND_DESTINATIONS_WITH_EXPLICIT_PREIMAGE_V1',projected_target_to_component=to)
  else:want.update(parent_state_sha256=canonical_sha256(p),projected_target_to_ordered_target=to,target_origins=[['parent',0],['parent',1],['attachment',0]])
  check(row==want,'ROW_SEMANTICS_OR_REFERENCE');return key
 def lookup(self,object_id):
  check(not self.closed,'READER_CLOSED');check(object_id in self.rows,'OBJECT_NOT_IN_SCOPE');return strict(canonical_bytes(self.rows[object_id]))
 def lookup_formation(self,formation_id):
  check(not self.closed,'READER_CLOSED');check(formation_id in self.formations,'FORMATION_NOT_IN_SCOPE');return self.lookup(self.formations[formation_id])
 def projection_preimages(self,depth,boundary_sha256):
  check(not self.closed,'READER_CLOSED');check((depth,boundary_sha256) in self.preimages,'BOUNDARY_NOT_IN_SCOPE');return tuple(self.preimages[depth,boundary_sha256])
 def lineage(self,object_id):
  row=self.lookup(object_id)
  return {'construction':row,'parent':self.lookup(row['identity']['parent_object_id']) if self.depths[object_id]==2 else None,'primitive_events':[strict(canonical_bytes(self.events[c['event_id']])) for c in row['components']] if self.depths[object_id]==1 else [strict(canonical_bytes(self.events[row['identity']['attachment_event_id']]))]}
 def close(self):self.closed=True;self.rows.clear();self.formations.clear();self.preimages.clear()
