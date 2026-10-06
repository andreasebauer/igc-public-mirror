from pathlib import Path
import tempfile,json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import O7ExtendedMasterReader
from .previous_o7.handler import handler as prior_handler

def handler(stage,runtime):
 i=stage['input_artifacts'];sources=json.loads(Path(i['predecessor_sources']).read_bytes());base=Path(__file__).parent/'previous_o7'
 if set(sources)!={str(p.relative_to(base)) for p in base.rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
 for n,h in sources.items():
  if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREDECESSOR_SOURCE')
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as f:
   for j in range(2):f.write(Path(i['scientific_part'+str(j)]).read_bytes())
  args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
  for name,cat in [('node','0141'),('o2','0142'),('o3','0143'),('o4','0144'),('o5','0145'),('o6','0146'),('o7','0147')]:args={name+'_admission':i[name+'_admission'],name+'_archive':i[name+'_archive'],'previous_catalog':i['catalog'+cat],'previous_arguments':args}
  with O7ExtendedMasterReader(i['new_catalog'],i['additional_admission'],i['additional_archive'],previous_catalog=i['catalog0148'],previous_arguments=args) as r:
   roots=comps=objects=owners=0
   def check_owner(x):
    if x['role']=='admitted_O6_component_binding':
     b=x['binding'];k=tuple(b['source_key'])
     if x['O6_root']!=r.lookup_o6(*k) or x['O6_component'] not in r.o6_components(*k):raise ValueError('ADMITTED_O6_MASTER_BINDING')
     for j,nested in enumerate(x['nested_O5_owners']):
      if nested!=r.o6_owner(*k,b['source_owners'][j]):raise ValueError('NESTED_O5_MASTER_BINDING')
    else:
     if x['admitted_O6_root'] is not False:raise ValueError('EXTERNAL_O6_ADMISSION')
     for nested in x['nested_O5_owners']:
      b=nested['binding']
      if nested['O5_root']!=r.lookup_o5(b['source_lane'],b['source_rank'],b['source_digest']):raise ValueError('TWIN_ADMITTED_O5_MASTER_BINDING')
   for key,row in r.additional.records.items():
    if r.lookup_o7(*key)['record']!=row['record'] or len(r.o7_resources(*key))!=len(row['parent_ids']):raise ValueError('ADDITIONAL_ROOT_ROUTE')
    for j in range(len(row['parent_ids'])):check_owner(r.o7_owner(*key,j));owners+=1
    if r.o7_components(*key)!=r.root_components[key]:raise ValueError('ADDITIONAL_ROOT_COMPONENT_ROUTE')
    roots+=1
   for key,occ in r.additional.occurrences.items():
    if r.lookup_o7_component(*key)!=occ or len(r.o7_component_resources(*key))!=len(occ['record']['parent_ids']):raise ValueError('ADDITIONAL_COMPONENT_ROUTE')
    for j in range(len(occ['record']['parent_ids'])):check_owner(r.o7_component_owner(*key,j));owners+=1
    comps+=1
   for oid,obj in r.additional.objects.items():
    if r.o7_component_object(oid)!=obj:raise ValueError('EXACT_COMPONENT_OBJECT_ROUTE')
    objects+=1
   k=next(iter(r.additional.records));ck=next(iter(r.additional.occurrences));x=r.lookup_o7(*k);x['record']['edges'][0][0]=999
   if x==r.lookup_o7(*k):raise ValueError('ROOT_COPY_ISOLATION')
   x=r.lookup_o7_component(*ck);x['record']['edges'][0][0]=999
   if x==r.lookup_o7_component(*ck):raise ValueError('COMPONENT_COPY_ISOLATION')
   for call in [lambda:r.lookup_o7('HOM6',4,k[2]),lambda:r.lookup_o7('TWIN4',1,k[2]),lambda:r.o7_owner(*k,-1),lambda:r.o7_component_owner(*ck,999),lambda:r.o7_parents(*k),lambda:r.o7_component_source_owner_mapping(*ck)]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
   coverage=r.coverage_report()
  for call in [lambda:r.lookup_o7(*k),lambda:r.lookup_o7_component(*ck),lambda:r.lookup_o6('HOM6',0,'missing')]:
   try:call()
   except ValueError:pass
   else:raise ValueError('CLOSED_READER')
 prior=dict(stage);prior['input_artifacts']=dict(i,new_catalog=i['catalog0148'],predecessor_sources=i['prior_o7_sources']);result=prior_handler(prior,runtime).result
 result.update(coverage);result.update(outcome='PASS',all_additional_o7_root_routes_checked=roots,all_additional_o7_component_occurrence_routes_checked=comps,all_additional_o7_object_routes_checked=objects,all_additional_o7_owner_routes_checked=owners,prior_slices_unchanged=148,prior_reader_bytes_unchanged=True,prior_reader_modules_checked=len(sources),generation_calls=0)
 result['negative_checks']+=['absent_O7_whole_forests','component_owner_bounds','absent_source_owner_mapping','root_component_copy_isolation'];return ChainExecutionResult(result=result)
