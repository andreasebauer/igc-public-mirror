from pathlib import Path
import hashlib,json
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import CarrierReader

def handler(stage,runtime):
 p=stage['execution']['parameters'];i=stage['input_artifacts']
 if hashlib.sha256(Path(i['catalog']).read_bytes()).hexdigest()!=p['catalog_sha256']:raise ValueError('CATALOG_HASH')
 cat=json.loads(Path(i['catalog']).read_bytes());r=CarrierReader(i['archive'],p['archive_sha256'],p['root_sha256'],i['o4_archive'],i['o3_archive'])
 if cat['slices'][-1]['scientific_root_sha256']!=r.root['O4_bindings']['root_sha256']:raise ValueError('ADMITTED_O4_ROOT')
 count=owners=comps=0
 for (lane,rank,digest),row in r.records.items():
  if r.lookup(lane,rank,digest)['record']!=row:raise ValueError('ROOT_ROUTE')
  for index,pid in enumerate(row['proto_ids']):
   o=r.owner(lane,rank,digest,index);b=r.bindings[pid]
   if o['prototype_id']!=pid or o['O4_root']['record']!=r.o4.records[(b['source_lane'],b['source_rank'],b['source_digest'])] or o['O4_component']!=r.base_components[pid]:raise ValueError('OWNER_ROUTE')
   
   for j,nested in enumerate(o['nested_O3_owners']):
    if nested!=r.o4.owner(b['source_lane'],b['source_rank'],b['source_digest'],b['source_owners'][j]):raise ValueError('NESTED_O3_ROUTE')
   owners+=1
  if r.components(lane,rank,digest)!=r.component_records[(lane,rank,digest)]:raise ValueError('COMPONENT_ROUTE')
  comps+=len(r.components(lane,rank,digest));count+=1
 key=next(k for k in r.records if k[1]>0);x=r.lookup(*key);x['record']['proto_ids'][0]='changed'
 if x==r.lookup(*key):raise ValueError('ROOT_COPY_ISOLATION')
 x=r.owner(*key,0);x['O4_root']['record']['proto_ids'][0]='changed'
 if x==r.owner(*key,0):raise ValueError('NESTED_COPY_ISOLATION')
 x=r.components(*key);x[0]['edges'][0][0]=999
 if x==r.components(*key):raise ValueError('COMPONENT_COPY_ISOLATION')
 for call in [lambda:r.lookup('missing',1,key[2]),lambda:r.lookup(key[0],99,key[2]),lambda:r.lookup(key[0],key[1],'missing'),lambda:r.owner(*key,-1),lambda:r.owner(*key,999),lambda:r.parents(*key)]:
  try:call()
  except (KeyError,ValueError):pass
  else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
 report=dict(r.report);report.update(outcome='PASS',all_occurrence_routes_checked=count,all_O4_owner_routes_checked=owners,all_component_routes_checked=comps,scientific_root_sha256=p['root_sha256'],archive_sha256=p['archive_sha256'],scientific_master_admission=False,negative_checks=['lane_rank_digest_scope','owner_index_bounds','unavailable_derivation_lineage','root_nested_component_copy_isolation','closed_reader']);r.close()
 try:r.lookup(*key)
 except ValueError:pass
 else:raise ValueError('CLOSED_READER')
 return ChainExecutionResult(result=report)
