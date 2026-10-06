"""Existing independent row checks, parameterized by admitted slice bounds.
No producer imports or invocation.
"""
from itertools import product
from .support import canonical_bytes
from .prefix_contract import event_id,digest as canonical_sha256
from .prefix_reader import check,fields,integer,_boundary
ROOT='e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351'
def validate_rows(reader,shard,seen,low,high):
 fields(shard,['schema_id','parent_root','catalog_sha256','rows'])
 check(shard['schema_id']=='IG_NATIVE_EXACT_J3_ROWS_V1' and shard['parent_root']==ROOT and shard['catalog_sha256']==reader.root['catalog_sha256'],'RETAINED_SHARD_SCOPE')
 check(type(shard['rows']) is list and 0<len(shard['rows'])<=8192,'RETAINED_SHARD_COUNT')
 a=reader.foundation['arrays'];events=realizations=0
 for row in shard['rows']:
  fields(row,['j3_id','exact_tid','source_sids','candidate_class_id','primitive_components','internal_incidence','events'])
  tid=row['exact_tid'];integer(tid,low,high);check(tid not in seen,'RETAINED_DUPLICATE_TID');seen.add(tid)
  sids=a['tri'][tid*3:tid*3+3];check(row['source_sids']==sids and row['candidate_class_id']==a['cid'][tid],'RETAINED_EXACT_SOURCE')
  identity={'schema':'IG_MASTER_EXACT_ROOTED_J3_CARRIER_V1','catalog_sha256':reader.root['catalog_sha256'],'exact_tid':tid,'ordered_roots':[0,1,2]}
  check(row['j3_id']==canonical_sha256(identity),'RETAINED_CARRIER_IDENTITY')
  check(row['primitive_components']==[{'role':i,'source_sid':s,'external_port':i} for i,s in enumerate(sids)] and row['internal_incidence']==[[0,1],[0,2],[1,2]],'RETAINED_COMPONENT_PORT_INCIDENCE')
  check(type(row['events']) is list and 0<len(row['events'])<=1000,'RETAINED_EVENTS_BOUND')
  choices_seen=set();eids=[]
  for event in row['events']:
   fields(event,['event_id','record','realizations']);_boundary(event['record'],3,1)
   check(event['event_id']==event_id(event['record']),'RETAINED_EVENT_IDENTITY');eids.append(event['event_id'])
   check(type(event['realizations']) is list and 0<len(event['realizations'])<=1000,'RETAINED_REALIZATION_BOUND')
   for occ in event['realizations']:
    fields(occ,['option_indices','target_sids','target_tid']);check(type(occ['option_indices']) is list and len(occ['option_indices'])==3,'RETAINED_CHOICE_SHAPE')
    choices=tuple(occ['option_indices']);check(choices not in choices_seen,'RETAINED_DUPLICATE_REALIZATION')
    for sid,i in zip(sids,choices):integer(i,0,a['oc'][sid])
    choices_seen.add(choices);computed=reader._choice_record(sids,choices)
    check(computed is not None and computed[0]==event['record'] and computed[1]==occ['target_sids'] and computed[2]==occ['target_tid'],'RETAINED_OPTION_TARGET_BINDING');realizations+=1
   events+=1
  check(eids==sorted(set(eids)),'RETAINED_EVENT_ORDER')
  allowed={x for x in product(*(range(a['oc'][s]) for s in sids)) if reader._choice_record(sids,x) is not None}
  check(choices_seen==allowed,'RETAINED_OPTION_COMPLETENESS')
 return events,realizations
