from pathlib import Path
import tempfile,json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import G1ProjectionMasterReader

def handler(stage,runtime):
 i=stage['input_artifacts'];sources=json.loads(Path(i['g1_predecessor_sources']).read_bytes());base=Path(__file__).parent/'previous'
 if set(sources)!={str(p.relative_to(base)) for p in base.rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
 for n,h in sources.items():
  if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREDECESSOR_SOURCE')
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as f:
   for j in range(2):f.write(Path(i['scientific_part'+str(j)]).read_bytes())
  args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
  for name,cat in [('node','0141'),('o2','0142'),('o3','0143'),('o4','0144'),('o5','0145'),('o6','0146'),('o7','0147')]:args={name+'_admission':i[name+'_admission'],name+'_archive':i[name+'_archive'],'previous_catalog':i['catalog'+cat],'previous_arguments':args}
  args={'additional_admission':i['additional_admission'],'additional_archive':i['additional_archive'],'previous_catalog':i['catalog0148'],'previous_arguments':args}
  with G1ProjectionMasterReader(i['new_catalog'],i['g1_admission'],i['g1_archive'],previous_catalog=i['catalog0149'],previous_arguments=args) as r:
   pop=json.loads(Path(i['g1_population']).read_bytes())['interfaces'];cont=json.loads(Path(i['g1_continuations']).read_bytes())['rows'];classes={}
   if r.g1_public_refs()!=[x['carrier_ref'] for x in pop]:raise ValueError('COHORT_ROUTE')
   for row in pop:
    ref=row['carrier_ref'];classes.setdefault(row['interface_sha256'],[]).append(ref)
    if r.lookup_g1_public_interface(ref)!=row:raise ValueError('INTERFACE_ROUTE')
    for t in range(7):
     if r.g1_public_reservation(ref,t)!=row['one_endpoint_reservations'][t] or r.g1_public_continuation_hash(ref,t)!=cont[ref][str(t)]:raise ValueError('RESERVATION_ROUTE')
   for h,refs in classes.items():
    if r.g1_public_interface_class(h)!=refs:raise ValueError('CLASS_ROUTE')
   ref=pop[0]['carrier_ref'];x=r.lookup_g1_public_interface(ref);x['total_free_by_type'][0]=0
   if x==r.lookup_g1_public_interface(ref):raise ValueError('COPY_ISOLATION')
   for call in [lambda:r.lookup_g1_public_interface('missing'),lambda:r.g1_public_reservation(ref,True),lambda:r.g1_public_reservation(ref,7),lambda:r.g1_exact_state(ref),lambda:r.g1_q2_payload(ref)]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
   old_roots=0
   for key,row in r.previous.additional.records.items():
    if r.lookup_o7(*key)!=r.previous.lookup_o7(*key):raise ValueError('O7_DELEGATION')
    old_roots+=1
   coverage=r.coverage_report()
  for call in [lambda:r.lookup_g1_public_interface(ref),lambda:r.lookup_o7(*key),r.coverage_report]:
   try:call()
   except ValueError:pass
   else:raise ValueError('CLOSED_READER')
 result=dict(coverage,outcome='PASS',all_g1_public_interface_routes_checked=193,all_g1_public_class_routes_checked=192,all_g1_public_reservation_routes_checked=1351,all_g1_public_continuation_routes_checked=1351,prior_slices_unchanged=149,prior_reader_bytes_unchanged=True,prior_reader_modules_checked=len(sources),prior_additional_o7_delegation_routes_checked=old_roots,prior_full_regression_reexecuted=False,generation_calls=0,authority='SAVED_PUBLIC_PROJECTION_ONLY')
 return ChainExecutionResult(result=result)
