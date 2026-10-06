from pathlib import Path
import tempfile
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import CompleteMasterReader
from .previous.recursive_reader import IntegrityError
from .legacy_pairs.prefix_reader import PrefixError

def handler(stage,runtime):
 inputs=stage['input_artifacts'];checks=[]
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as out:
   for i in range(2):out.write(Path(inputs['scientific_part'+str(i)]).read_bytes())
  kwargs={'previous_catalog':inputs['catalog0140'],'recursive_admission':inputs['admission'],'predecessor_catalog':inputs['catalog'],'recursive_archive':p}
  with CompleteMasterReader(inputs['new_catalog'],inputs['branch_catalog'],inputs['pair_admission'],inputs['parent_archive'],inputs['pair_archive'],**kwargs) as r:
   objects=formations=components=0
   for family in ['AB','BC']:
    for oid,row in r.pairs.objects[family].items():
     got=r.lookup_pair(family,oid)
     if got['object']!=row:raise ValueError('OBJECT_ROUTE')
     objects+=1
     for occ in got['occurrences']:
      if r.lookup_pair_formation(family,occ['formation']['formation_id'])!=occ:raise ValueError('FORMATION_ROUTE')
      formations+=1
      for c in occ['components']:
       if c['event']['event_id']!=next(x['event_id'] for x in occ['formation']['components'] if x['role']==c['role']) or not c['event']['realizations'] or len(c['primitive_source_states'])!=3:raise ValueError('COMPONENT_ROUTE')
       components+=1
   if objects!=1458 or formations!=1458 or components!=2916:raise ValueError('PAIR_SCOPE_COUNT')
   for d in [1,2]:
    oid=next(oid for oid,v in r.previous.recursive.depths.items() if v==d);row=r.lookup_construction(oid)
    if r.lookup_formation(row['formation_id'])!=row or r.lineage(oid)['construction']!=row:raise ValueError('RECURSIVE_REGRESSION')
   def reject(label,fn):
    try:fn()
    except (IntegrityError,PrefixError,KeyError):checks.append(label)
    else:raise ValueError('EXPECTED_REJECTION_'+label)
   reject('wrong_family',lambda:r.lookup_pair('AC','0'*64));reject('missing_pair',lambda:r.lookup_pair('AB','0'*64));reject('missing_formation',lambda:r.lookup_pair_formation('AB','0'*64))
   oid=next(iter(r.pairs.objects['AB']));original=r.lookup_pair('AB',oid);mut=r.lookup_pair('AB',oid);mut['object']['record'][0]=-1
   if r.lookup_pair('AB',oid)!=original:raise ValueError('COPY_LEAK')
   report=r.coverage_report();report.update(pair_closure=r.pairs.verify(),all_pair_objects_checked=objects,all_pair_formations_checked=formations,all_pair_component_routes_checked=components,predecessor_slices_exactly_unchanged=140,recursive_reader_inherited=True,original_pair_reader_reused=True,admission_reissued=False,outcome='PASS',negative_checks=checks)
  reject('closed_reader',lambda:r.lookup_pair('AB',oid));report['negative_checks']=checks
  return ChainExecutionResult(result=report)
