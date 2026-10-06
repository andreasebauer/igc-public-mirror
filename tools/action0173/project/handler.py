from pathlib import Path
import json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import CarrierReader,shaj
from .previous_o7.frozen_o7 import iter_sites

def handler(stage,runtime):
 p=stage['execution']['parameters'];i=stage['input_artifacts']
 if hashlib.sha256(Path(i['catalog']).read_bytes()).hexdigest()!=p['catalog_sha256']:raise ValueError('CATALOG_HASH')
 r=CarrierReader(i['archive'],p['archive_sha256'],p['root_sha256'],i['o7_archive'],i['o6_archive'],i['o5_archive'],i['o4_archive'],i['o3_archive'])
 cat=json.loads(Path(i['catalog']).read_bytes())
 if cat['slices'][-1]['scientific_root_sha256']!=r.root['O7_bindings']['root_sha256'] or r.data['contract']['master_catalog_sha256']!=p['catalog_sha256']:raise ValueError('ADMITTED_O7_BASE')
 rootowners=compowners=resourceowners=0
 def check_owner(o,pid):
  if o['prototype_id']!=pid or shaj(o['resource_O6'])!=shaj(r.P[pid].h6):raise ValueError('OWNER_RESOURCE_ROUTE')
  if pid in r.base.bindings:
   b=r.base.bindings[pid];k=tuple(b['source_key'])
   if o['O6_root']!=r.base.o6.lookup(*k) or o['O6_component']!=r.base.base_components[pid] or o['nested_O5_owners']!=[r.base.o6.owner(*k,j) for j in b['source_owners']]:raise ValueError('ADMITTED_OWNER_ROUTE')
  else:
   if o['external_O6_root']!=r.twins[pid] or o['admitted_O6_root'] is not False:raise ValueError('EXTERNAL_TWIN_ROUTE')
   for x in o['nested_O5_owners']:
    b=r.base.o6.bindings[x['prototype_id']]
    if x['O5_root']!=r.base.o6.o5.lookup(b['source_lane'],b['source_rank'],b['source_digest']) or x['O5_component']!=r.base.o6.base_components[x['prototype_id']]:raise ValueError('EXTERNAL_O5_ROUTE')
 def check_resources(resources):
  for h6 in resources:
   for *_,capacity,free in iter_sites(h6):
    if any(not 0<=free[a]<=capacity[a] for a in range(7)):raise ValueError('RESOURCE_ROUTE')
  return len(resources)
 for key,row in r.records.items():
  if r.lookup(*key)!=row:raise ValueError('ROOT_ROUTE')
  for j,pid in enumerate(row['parent_ids']):check_owner(r.owner(*key,j),pid);rootowners+=1
  resourceowners+=check_resources(r.resources(*key))
 for key,row in r.occurrences.items():
  if r.component(*key)!=row or r.object(row['object_id'])!=r.objects[row['object_id']]:raise ValueError('COMPONENT_ROUTE')
  for j,pid in enumerate(row['record']['parent_ids']):check_owner(r.component_owner(*key,j),pid);compowners+=1
  resourceowners+=check_resources(r.component_resources(*key))
 key=next(iter(r.records));ckey=next(iter(r.occurrences));oid=r.occurrences[ckey]['object_id']
 for getter,mutator in [(lambda:r.lookup(*key),lambda x:x['record']['edges'][0].__setitem__(0,999)),(lambda:r.component(*ckey),lambda x:x['record']['edges'][0].__setitem__(0,999)),(lambda:r.object(oid),lambda x:x['payload']['edges'][0].__setitem__(0,999)),(lambda:r.owner(*key,0),lambda x:x['resource_O6'].clear())]:
  x=getter();mutator(x)
  if x==getter():raise ValueError('COPY_ISOLATION')
 for call in [lambda:r.lookup('HOM6',4,key[2]),lambda:r.lookup('TWIN4',1,key[2]),lambda:r.lookup('HOM6',2,'missing'),lambda:r.component('TWIN4',1,'missing',0),lambda:r.object('missing'),lambda:r.owner(*key,-1),lambda:r.owner(*key,True),lambda:r.component_owner(*ckey,999),lambda:r.parents(*key),lambda:r.source_owner_mapping(*ckey)]:
  try:call()
  except (ValueError,KeyError):pass
  else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
 try:CarrierReader(i['archive'],'0'*64,p['root_sha256'],i['o7_archive'],i['o6_archive'],i['o5_archive'],i['o4_archive'],i['o3_archive'])
 except ValueError:pass
 else:raise ValueError('BAD_ARCHIVE_HASH_ACCEPTED')
 report=dict(r.report,outcome='PASS',all_root_routes_checked=len(r.records),all_component_occurrence_routes_checked=len(r.occurrences),all_distinct_component_routes_checked=len(r.objects),all_root_owner_routes_checked=rootowners,all_component_owner_routes_checked=compowners,all_resource_owner_routes_checked=resourceowners,scientific_root_sha256=p['root_sha256'],archive_sha256=p['archive_sha256'],negative_checks=['absent_whole_roots','bad_owner_indices','unavailable_ancestry_and_source_owner_mapping','copy_isolation','archive_hash','closed_reader'])
 r.close()
 for call in [lambda:r.lookup(*key),lambda:r.component(*ckey),lambda:r.object(oid),lambda:r.resources(*key)]:
  try:call()
  except ValueError:pass
  else:raise ValueError('CLOSED_READER')
 return ChainExecutionResult(result=report)
