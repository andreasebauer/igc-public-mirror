from pathlib import Path
import tempfile,json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import O2MasterReader

def handler(stage,runtime):
 i=stage['input_artifacts'];sources=json.loads(Path(i['predecessor_sources']).read_bytes());base=Path(__file__).parent/'previous_node'
 for n,h in sources.items():
  if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREDECESSOR_SOURCE')
 if set(sources)!={str(p.relative_to(base)) for p in base.rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as f:
   for j in range(2):f.write(Path(i['scientific_part'+str(j)]).read_bytes())
  args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
  nodeargs={'node_admission':i['node_admission'],'node_archive':i['node_archive'],'previous_catalog':i['catalog0141'],'previous_arguments':args}
  with O2MasterReader(i['new_catalog'],i['o2_admission'],i['o2_archive'],previous_catalog=i['catalog0142'],previous_arguments=nodeargs) as r:
   count=parents=0
   for d,rows in r.o2.records.items():
    if r.o2_observer(d)!=json.loads(r.o2.content_bytes('observer'+str(d))):raise ValueError('OBSERVER_ROUTE')
    for h,row in rows.items():
     if r.lookup_o2(d,h)['record']!=row:raise ValueError('O2_ROUTE')
     if d>77:
      got=r.o2_parents(d,h)
      if [x['exact_key'] for x in got]!=row['parents'] or any(x['depth']!=d-1 or x['record']!=r.o2.records[d-1][x['exact_key']] for x in got):raise ValueError('O2_PARENT_ROUTE')
      parents+=len(got)
     count+=1
   nodes=nodeparents=0
   for L,rows in r.previous.nodes.records.items():
    for h,row in rows.items():
     if r.lookup_node_panel(L,h)['record']!=row:raise ValueError('NODE_ROUTE')
     expected=r.lookup_node_panel(L,h)['parents'];got=r.node_parents(L,h)
     if [x['boundary_sha256'] for x in got]!=expected or any(x['record']!=r.previous.nodes.records[L-1][x['boundary_sha256']] for x in got):raise ValueError('NODE_PARENT_ROUTE')
     nodes+=1;nodeparents+=len(got)
   pairs=formations=0
   for family in ['AB','BC']:
    for h,row in r.previous.previous.pairs.objects[family].items():
     if r.lookup_pair(family,h)['object']!=row:raise ValueError('PAIR_ROUTE')
     pairs+=1
   for d in [1,2]:
    rec=r.previous.previous.previous.recursive;h=next(h for h,v in rec.depths.items() if v==d);row=r.lookup_construction(h)
    if r.lookup_formation(row['formation_id'])!=row or r.lineage(h)['construction']!=row:raise ValueError('RECURSIVE_ROUTE')
   report=r.coverage_report();report.update(outcome='PASS',all_o2_routes_checked=count,all_o2_parent_routes_checked=parents,all_node_routes_checked=nodes,all_node_parent_routes_checked=nodeparents,prior_slices_unchanged=142,prior_reader_bytes_unchanged=True,prior_reader_modules_checked=len(sources),all_pair_routes_checked=pairs,recursive_depth_route_samples=[1,2],generation_calls=0)
   h=next(iter(r.o2.records[78]));x=r.lookup_o2(78,h);x['record']['states'][0][0]+=1
   if x==r.lookup_o2(78,h):raise ValueError('COPY_ISOLATION')
   for call in [lambda:r.lookup_o2(76,h),lambda:r.lookup_o2(78,'0'*64),lambda:r.o2_parents(77,next(iter(r.o2.records[77])))]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
  try:r.lookup_o2(78,h)
  except ValueError:pass
  else:raise ValueError('CLOSED_READER')
  report['negative_checks']=['excluded_depth','missing_identity','external_seed_ancestry','copy_isolation','closed_reader'];return ChainExecutionResult(result=report)
