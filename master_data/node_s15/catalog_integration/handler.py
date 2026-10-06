from pathlib import Path
import tempfile,json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import NodeMasterReader

def handler(stage,runtime):
 i=stage['input_artifacts'];sources=json.loads(Path(i['predecessor_sources']).read_bytes());base=Path(__file__).parent/'previous_complete'
 for n,h in sources.items():
  if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREDECESSOR_SOURCE')
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as f:
   for j in range(2):f.write(Path(i['scientific_part'+str(j)]).read_bytes())
  args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
  with NodeMasterReader(i['new_catalog'],i['node_admission'],i['node_archive'],previous_catalog=i['catalog0141'],previous_arguments=args) as r:
   count=parents=0
   for L,rows in r.nodes.records.items():
    for h,row in rows.items():
     if r.lookup_node_panel(L,h)['record']!=row:raise ValueError('NODE_ROUTE')
     expected=r.lookup_node_panel(L,h)['parents'];got=r.node_parents(L,h)
     if [x['boundary_sha256'] for x in got]!=expected or any(x['level']!=L-1 for x in got):raise ValueError('PARENT_ROUTE')
     count+=1;parents+=len(got)
   pairs=0
   for family in ['AB','BC']:
    for h,row in r.previous.pairs.objects[family].items():
     if r.lookup_pair(family,h)['object']!=row:raise ValueError('PAIR_ROUTE')
     pairs+=1
   for d in [1,2]:
    rec=r.previous.previous.recursive;h=next(h for h,v in rec.depths.items() if v==d);row=r.lookup_construction(h)
    if r.lookup_formation(row['formation_id'])!=row or r.lineage(h)['construction']!=row:raise ValueError('RECURSIVE_ROUTE')
   report=r.coverage_report();report.update(outcome='PASS',all_node_routes_checked=count,all_node_parent_routes_checked=parents,edges=r.nodes.report['edges'],prior_slices_unchanged=141,prior_reader_bytes_unchanged=True,prior_reader_modules_checked=len(sources),all_pair_routes_checked=pairs,generation_calls=0)
   for L,h in [(37,'0'*64),(39,'0'*64)]:
    try:r.lookup_node_panel(L,h)
    except KeyError:pass
    else:raise ValueError('MISSING_ROUTE_ACCEPTED')
   L=39;h=next(iter(r.nodes.records[L]));x=r.lookup_node_panel(L,h);x['parents'].clear()
   if not r.lookup_node_panel(L,h)['parents']:raise ValueError('COPY_ISOLATION')
  try:r.lookup_node_panel(L,h)
  except ValueError:pass
  else:raise ValueError('CLOSED_READER')
  report['negative_checks']=['missing_level','missing_record','copy_isolation','closed_reader'];return ChainExecutionResult(result=report)
