"""Independent export validator; imports no producer or Decoder engine."""
from itertools import product
from .support import canonical_bytes
from .prefix_contract import digest,event_id
CATALOG='86e1e4be57ca96f7b4e32650c59a5385df15974712931696fd8b2cfee8ad5e58'
ROOT='e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351'
def check(x,why):
 if not x:raise ValueError(why)
def integer(x,lo,hi):check(type(x) is int and lo<=x<hi,'INTEGER_SCOPE')
def roots(p):check(type(p) is list and len(p)==3 and all(type(i) is int for i in p) and sorted(p)==[0,1,2] and p!=[0,1,2],'ROOT_SCOPE')
def validate_row(parent,row,expected_roots):
 roots(expected_roots)
 check(set(row)=={'j3_id','exact_tid','ordered_roots','source_sids','source_occurrence_indices','candidate_class_id','primitive_components','internal_incidence','events'},'ROW_FIELDS')
 tid=row['exact_tid'];integer(tid,0,50116);p=row['ordered_roots'];roots(p);check(p==expected_roots,'EXPECTED_ROOTS')
 a=parent.foundation['arrays'];source=a['tri'][3*tid:3*tid+3];sids=[source[i] for i in p]
 check(row['source_sids']==sids and all(type(s) is int for s in row['source_sids']) and canonical_bytes(row['source_occurrence_indices'])==canonical_bytes(p),'ORDERED_SOURCE')
 integer(row['candidate_class_id'],0,len(a['cid']));check(row['candidate_class_id']==a['cid'][tid],'SOURCE_CLASS')
 identity={'schema':'IG_MASTER_EXACT_ROOTED_J3_CARRIER_V1','catalog_sha256':parent.root['catalog_sha256'],'exact_tid':tid,'ordered_roots':p}
 check(row['j3_id']==digest(identity),'CARRIER_ID')
 components=[{'role':i,'source_occurrence_index':p[i],'source_sid':s,'external_port':i} for i,s in enumerate(sids)]
 check(canonical_bytes(row['primitive_components'])==canonical_bytes(components) and canonical_bytes(row['internal_incidence'])==canonical_bytes([[0,1],[0,2],[1,2]]),'COMPONENT_INCIDENCE')
 check(type(row['events']) is list and 0<len(row['events'])<=1000,'EVENT_BOUND')
 seen=set();ids=[];count=0
 for e in row['events']:
  check(set(e)=={'event_id','record','realizations'} and e['event_id']==event_id(e['record']),'EVENT_ID')
  ids.append(e['event_id']);check(type(e['realizations']) is list and 0<len(e['realizations'])<=1000,'WITNESS_BOUND');local=[]
  for w in e['realizations']:
   check(set(w)=={'option_indices','target_sids','target_tid'},'WITNESS_FIELDS');choice=w['option_indices']
   check(type(choice) is list and len(choice)==3,'CHOICE_SHAPE')
   for s,i in zip(sids,choice):integer(i,0,a['oc'][s])
   key=tuple(choice);check(key not in seen,'DUPLICATE_CHOICE');seen.add(key);local.append(key)
   check(type(w['target_sids']) is list and len(w['target_sids'])==3,'TARGET_SHAPE')
   for s in w['target_sids']:integer(s,0,66)
   integer(w['target_tid'],0,50116)
   computed=parent._choice_record(sids,choice)
   check(computed is not None and canonical_bytes(computed[0])==canonical_bytes(e['record']) and computed[1]==w['target_sids'] and computed[2]==w['target_tid'],'ORDERED_WITNESS_BINDING');count+=1
  check(local==sorted(local),'WITNESS_ORDER')
 check(ids==sorted(set(ids)),'EVENT_ORDER')
 lawful={c for c in product(*(range(a['oc'][s]) for s in sids)) if parent._choice_record(sids,c) is not None}
 check(seen==lawful,'ALL_LAWFUL_CHOICES');return len(ids),count
class PermutationShardReader:
 """Reads verified proposed exports; grants no admission and never generates."""
 def __init__(self,parent,shards,expected_roots,first_tid,last_tid):
  roots(expected_roots);integer(first_tid,0,50116);integer(last_tid,first_tid,50116)
  self.rows={};self.events=self.witnesses=0
  for shard in shards:
   check(set(shard)=={'schema_id','parent_root','catalog_sha256','accepted_catalog_sha256','ordered_roots','rows'},'SHARD_FIELDS')
   roots(shard['ordered_roots'])
   check(shard['schema_id']=='IG_NATIVE_PERMUTATION_LABELLED_J3_ROWS_V1' and shard['parent_root']==ROOT and shard['catalog_sha256']==parent.root['catalog_sha256'] and shard['accepted_catalog_sha256']==CATALOG and shard['ordered_roots']==expected_roots,'SHARD_SCOPE')
   check(len(canonical_bytes(shard))<=4*1024**2,'SHARD_BYTES');tids=[]
   for row in shard['rows']:
    e,w=validate_row(parent,row,expected_roots);t=row['exact_tid'];check(first_tid<=t<=last_tid and t not in self.rows,'ROW_CENSUS');self.rows[t]=row;self.events+=e;self.witnesses+=w;tids.append(t)
   check(tids==sorted(tids),'ROW_ORDER')
  check(set(self.rows)==set(range(first_tid,last_tid+1)),'EXPORT_CLOSURE')
 def lookup(self,tid):
  import json
  integer(tid,0,50116);return json.loads(canonical_bytes(self.rows[tid]))
