from pathlib import Path
import hashlib,json
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import CarrierReader

def handler(stage,runtime):
 p=stage['execution']['parameters'];i=stage['input_artifacts']
 if hashlib.sha256(Path(i['catalog']).read_bytes()).hexdigest()!=p['catalog_sha256']:raise ValueError('CATALOG_HASH')
 cat=json.loads(Path(i['catalog']).read_bytes());r=CarrierReader(i['archive'],p['archive_sha256'],p['root_sha256'],i['o6_archive'],i['o5_archive'],i['o4_archive'],i['o3_archive'])
 if cat['slices'][-1]['scientific_root_sha256']!=r.root['O6_bindings']['root_sha256']:raise ValueError('ADMITTED_O6_ROOT')
 count=owners=comps=0
 for key,row in r.records.items():
  if r.lookup(*key)['record']!=row:raise ValueError('ROOT_ROUTE')
  for idx,parent in enumerate(r.contexts[key].parents):
   o=r.owner(*key,idx);b=r.bindings[parent.pid];k=tuple(b['source_key'])
   if o['O6_root']['record']!=r.o6.records[k] or o['O6_component']!=r.base_components[parent.pid] or o['nested_O5_owners']!=[r.o6.owner(*k,j) for j in b['source_owners']]:raise ValueError('NESTED_OWNER_ROUTE')
   owners+=1
  comps+=len(r.components(*key));count+=1
  for h6 in r.resources(*key):
   from .frozen_o7 import iter_sites
   for *_,capacity,free in iter_sites(h6):
    if any(not 0<=free[a]<=capacity[a] for a in range(7)):raise ValueError('RESOURCE_ROUTE')
 key=next(k for k in r.records if k[1]>0);x=r.lookup(*key);x['record']['edges'][0][0]=999
 if x==r.lookup(*key):raise ValueError('ROOT_COPY_ISOLATION')
 x=r.owner(*key,0);x['O6_root']['record']['proto_ids'][0]='changed'
 if x==r.owner(*key,0):raise ValueError('NESTED_COPY_ISOLATION')
 x=r.components(*key);x[0]['edges'][0][0]=999
 if x==r.components(*key):raise ValueError('COMPONENT_COPY_ISOLATION')
 for call in [lambda:r.lookup('TWIN4',1,key[2]),lambda:r.lookup('HOM6',2,key[2]),lambda:r.lookup(key[0],key[1],'missing'),lambda:r.owner(*key,-1),lambda:r.owner(*key,999),lambda:r.parents(*key)]:
  try:call()
  except (KeyError,ValueError):pass
  else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
 report=dict(r.report);report.update(outcome='PASS',all_occurrence_routes_checked=count,all_O6_owner_routes_checked=owners,all_component_routes_checked=comps,scientific_root_sha256=p['root_sha256'],archive_sha256=p['archive_sha256'],scientific_master_admission=False,negative_checks=['unrecovered_HOM6_and_TWIN4_scope','owner_index_bounds','unavailable_derivation_lineage','root_nested_component_copy_isolation','closed_reader']);r.close()
 try:r.lookup(*key)
 except ValueError:pass
 else:raise ValueError('CLOSED_READER')
 return ChainExecutionResult(result=report)
