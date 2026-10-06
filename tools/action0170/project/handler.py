from pathlib import Path
import tempfile,json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import O7MasterReader
from .previous_o6.handler import handler as prior_handler
from .frozen_o7 import iter_sites

def handler(stage,runtime):
 i=stage['input_artifacts'];sources=json.loads(Path(i['predecessor_sources']).read_bytes());base=Path(__file__).parent/'previous_o6'
 if set(sources)!={str(p.relative_to(base)) for p in base.rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
 for n,h in sources.items():
  if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREDECESSOR_SOURCE')
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as f:
   for j in range(2):f.write(Path(i['scientific_part'+str(j)]).read_bytes())
  args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
  levels=[('node','0141'),('o2','0142'),('o3','0143'),('o4','0144'),('o5','0145'),('o6','0146')]
  for name,cat in levels:args={name+'_admission':i[name+'_admission'],name+'_archive':i[name+'_archive'],'previous_catalog':i['catalog'+cat],'previous_arguments':args}
  with O7MasterReader(i['new_catalog'],i['o7_admission'],i['o7_archive'],previous_catalog=i['catalog0147'],previous_arguments=args) as r:
   count=owners=comps=0
   for key,row in r.o7.records.items():
    if r.lookup_o7(*key)['record']!=row:raise ValueError('O7_ROUTE')
    for index,parent in enumerate(r.o7.contexts[key].parents):
     x=r.o7_owner(*key,index);b=r.o7.bindings[parent.pid];k=tuple(b['source_key'])
     if x['O6_root']!=r.lookup_o6(*k) or x['O6_component'] not in r.o6_components(*k):raise ValueError('O7_OWNER_MASTER_BINDING')
     for j,nested in enumerate(x['nested_O5_owners']):
      if nested!=r.o6_owner(*k,b['source_owners'][j]):raise ValueError('O7_NESTED_O5_MASTER_BINDING')
     owners+=1
    if r.o7_components(*key)!=r.o7.component_records[key]:raise ValueError('O7_COMPONENT_ROUTE')
    for h6 in r.o7_resources(*key):
     for *_,capacity,free in iter_sites(h6):
      if any(not 0<=free[a]<=capacity[a] for a in range(7)):raise ValueError('O7_RESOURCE_ROUTE')
    comps+=len(r.o7_components(*key));count+=1
   k=next(k for k in r.o7.records if k[1]>0);x=r.lookup_o7(*k);x['record']['edges'][0][0]=999
   if x==r.lookup_o7(*k):raise ValueError('O7_COPY_ISOLATION')
   for call in [lambda:r.lookup_o7('TWIN4',1,k[2]),lambda:r.lookup_o7('HOM6',2,k[2]),lambda:r.o7_owner(*k,-1),lambda:r.o7_parents(*k)]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('O7_EXCLUDED_ROUTE_ACCEPTED')
   coverage=r.coverage_report()
  try:r.lookup_o7(*k)
  except ValueError:pass
  else:raise ValueError('CLOSED_READER')
 # Execute the immutable predecessor's full route suite against its original catalog.
 prior=dict(stage);prior['input_artifacts']=dict(i,new_catalog=i['catalog0147'],predecessor_sources=i['prior_o6_sources']);result=prior_handler(prior,runtime).result
 result.update(coverage);result.update(outcome='PASS',all_o7_routes_checked=count,all_o7_owner_routes_checked=owners,all_o7_component_routes_checked=comps,prior_slices_unchanged=147,prior_reader_bytes_unchanged=True,prior_reader_modules_checked=len(sources),generation_calls=0)
 result['negative_checks']+=['unrecovered_O7_scope','O7_owner_bounds','O7_unavailable_ancestry','O7_copy_isolation'];return ChainExecutionResult(result=result)
