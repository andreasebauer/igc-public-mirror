from pathlib import Path
import tempfile,json,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import O6MasterReader

def handler(stage,runtime):
 i=stage['input_artifacts'];sources=json.loads(Path(i['predecessor_sources']).read_bytes());base=Path(__file__).parent/'previous_o5'
 for n,h in sources.items():
  if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREDECESSOR_SOURCE')
 if set(sources)!={str(p.relative_to(base)) for p in base.rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'recursive.zip'
  with p.open('wb') as f:
   for j in range(2):f.write(Path(i['scientific_part'+str(j)]).read_bytes())
  args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
  nodeargs={'node_admission':i['node_admission'],'node_archive':i['node_archive'],'previous_catalog':i['catalog0141'],'previous_arguments':args}
  o2args={'o2_admission':i['o2_admission'],'o2_archive':i['o2_archive'],'previous_catalog':i['catalog0142'],'previous_arguments':nodeargs}
  o3args={'o3_admission':i['o3_admission'],'o3_archive':i['o3_archive'],'previous_catalog':i['catalog0143'],'previous_arguments':o2args}
  o4args={'o4_admission':i['o4_admission'],'o4_archive':i['o4_archive'],'previous_catalog':i['catalog0144'],'previous_arguments':o3args}
  o5args={'o5_admission':i['o5_admission'],'o5_archive':i['o5_archive'],'previous_catalog':i['catalog0145'],'previous_arguments':o4args}
  with O6MasterReader(i['new_catalog'],i['o6_admission'],i['o6_archive'],previous_catalog=i['catalog0146'],previous_arguments=o5args) as r:
   o6count=o6owners=o6comps=0
   for key,row in r.o6.records.items():
    if r.lookup_o6(*key)['record']!=row:raise ValueError('O6_ROUTE')
    for index,pid in enumerate(row['proto_ids']):
     x=r.o6_owner(*key,index);b=r.o6.bindings[pid]
     if x['O5_root']!=r.lookup_o5(b['source_lane'],b['source_rank'],b['source_digest']) or x['O5_component'] not in r.o5_components(b['source_lane'],b['source_rank'],b['source_digest']):raise ValueError('O6_OWNER_MASTER_BINDING')
     for j,nested in enumerate(x['nested_O4_owners']):
      if nested!=r.o5_owner(b['source_lane'],b['source_rank'],b['source_digest'],b['source_owners'][j]):raise ValueError('O6_NESTED_O4_MASTER_BINDING')
     o6owners+=1
    if r.o6_components(*key)!=r.o6.component_records[key]:raise ValueError('O6_COMPONENT_ROUTE')
    for o5 in r.o6_resources(*key):
     for o4 in o5:
      for o3 in o4:
       for block in o3:
        for capacity,free in block:
         if len(capacity)!=7 or len(free)!=7 or any(not 0<=free[a]<=capacity[a] for a in range(7)):raise ValueError('O6_RESOURCE_ROUTE')
    o6comps+=len(r.o6_components(*key));o6count+=1
   o5count=o5owners=o5comps=0
   for key,row in r.previous.o5.records.items():
    if r.lookup_o5(*key)['record']!=row:raise ValueError('O5_ROUTE')
    for index,pid in enumerate(row['proto_ids']):
     x=r.o5_owner(*key,index);b=r.previous.o5.bindings[pid]
     if x['O4_root']!=r.lookup_o4(b['source_lane'],b['source_rank'],b['source_digest']) or x['O4_component'] not in r.o4_components(b['source_lane'],b['source_rank'],b['source_digest']):raise ValueError('O5_OWNER_MASTER_BINDING')
     for j,nested in enumerate(x['nested_O3_owners']):
      if nested!=r.o4_owner(b['source_lane'],b['source_rank'],b['source_digest'],b['source_owners'][j]):raise ValueError('O5_NESTED_O3_MASTER_BINDING')
     o5owners+=1
    if r.o5_components(*key)!=r.previous.o5.component_records[key]:raise ValueError('O5_COMPONENT_ROUTE')
    resources=r.o5_resources(*key)
    for groups in resources:
     for blocks in groups:
      for block in blocks:
       for p,f in block:
        if len(p)!=7 or len(f)!=7 or any(not 0<=f[a]<=p[a] for a in range(7)):raise ValueError('O5_RESOURCE_ROUTE')
    o5comps+=len(r.o5_components(*key));o5count+=1
   o4count=o4owners=o4comps=0
   for key,row in r.previous.previous.o4.records.items():
    if r.lookup_o4(*key)['record']!=row:raise ValueError('O4_ROUTE')
    for index,pid in enumerate(row['proto_ids']):
     x=r.o4_owner(*key,index);b=r.previous.previous.o4.bindings[pid]
     if x['O3_root']!=r.lookup_o3(b['source_rank'],b['source_exact_key']) or x['O3_component'] not in r.o3_components(b['source_rank'],b['source_exact_key']):raise ValueError('O4_OWNER_MASTER_BINDING')
     o4owners+=1
    if r.o4_components(*key)!=r.previous.previous.o4.component_records[key]:raise ValueError('O4_COMPONENT_ROUTE')
    o4comps+=len(r.o4_components(*key));o4count+=1
   o3count=o3parents=nested=comps=0
   for rank,rows in r.previous.previous.previous.o3.records.items():
    for h,row in rows.items():
     if r.lookup_o3(rank,h)['record']!=row:raise ValueError('O3_ROUTE')
     if rank:
      got=r.o3_parents(rank,h)
      if [x['exact_key'] for x in got]!=row['parent_keys'] or any(x['record']!=r.previous.previous.previous.o3.records[rank-1][x['exact_key']] for x in got):raise ValueError('O3_PARENT_ROUTE')
      o3parents+=len(got)
     for q in row['entities']:
      if r.o3_qstate(q)['state']!=r.previous.previous.previous.o3.qstates[q]:raise ValueError('Q_ROUTE')
      nested+=1
     if r.o3_components(rank,h)!=r.previous.previous.previous.o3.components(rank,h):raise ValueError('COMPONENT_ROUTE')
     comps+=len(r.o3_components(rank,h));o3count+=1
   count=parents=0
   for d,rows in r.previous.previous.previous.previous.o2.records.items():
    if r.o2_observer(d)!=json.loads(r.previous.previous.previous.previous.o2.content_bytes('observer'+str(d))):raise ValueError('OBSERVER_ROUTE')
    for h,row in rows.items():
     if r.lookup_o2(d,h)['record']!=row:raise ValueError('O2_ROUTE')
     if d>77:
      got=r.o2_parents(d,h)
      if [x['exact_key'] for x in got]!=row['parents'] or any(x['depth']!=d-1 or x['record']!=r.previous.previous.previous.previous.o2.records[d-1][x['exact_key']] for x in got):raise ValueError('O2_PARENT_ROUTE')
      parents+=len(got)
     count+=1
   nodes=nodeparents=0
   for L,rows in r.previous.previous.previous.previous.previous.nodes.records.items():
    for h,row in rows.items():
     if r.lookup_node_panel(L,h)['record']!=row:raise ValueError('NODE_ROUTE')
     expected=r.lookup_node_panel(L,h)['parents'];got=r.node_parents(L,h)
     if [x['boundary_sha256'] for x in got]!=expected or any(x['record']!=r.previous.previous.previous.previous.previous.nodes.records[L-1][x['boundary_sha256']] for x in got):raise ValueError('NODE_PARENT_ROUTE')
     nodes+=1;nodeparents+=len(got)
   pairs=formations=0
   for family in ['AB','BC']:
    for h,row in r.previous.previous.previous.previous.previous.previous.pairs.objects[family].items():
     if r.lookup_pair(family,h)['object']!=row:raise ValueError('PAIR_ROUTE')
     pairs+=1
   for d in [1,2]:
    rec=r.previous.previous.previous.previous.previous.previous.previous.recursive;h=next(h for h,v in rec.depths.items() if v==d);row=r.lookup_construction(h)
    if r.lookup_formation(row['formation_id'])!=row or r.lineage(h)['construction']!=row:raise ValueError('RECURSIVE_ROUTE')
   report=r.coverage_report();report.update(outcome='PASS',all_o6_routes_checked=o6count,all_o6_owner_routes_checked=o6owners,all_o6_component_routes_checked=o6comps,all_o5_routes_checked=o5count,all_o5_owner_routes_checked=o5owners,all_o5_component_routes_checked=o5comps,all_o4_routes_checked=o4count,all_o4_owner_routes_checked=o4owners,all_o4_component_routes_checked=o4comps,all_o3_routes_checked=o3count,all_o3_parent_routes_checked=o3parents,all_nested_Q_routes_checked=nested,all_o3_component_routes_checked=comps,all_o2_routes_checked=count,all_o2_parent_routes_checked=parents,all_node_routes_checked=nodes,all_node_parent_routes_checked=nodeparents,prior_slices_unchanged=146,prior_reader_bytes_unchanged=True,prior_reader_modules_checked=len(sources),all_pair_routes_checked=pairs,recursive_depth_route_samples=[1,2],generation_calls=0)
   h=next(iter(r.previous.previous.previous.previous.o2.records[78]));x=r.lookup_o2(78,h);x['record']['states'][0][0]+=1
   if x==r.lookup_o2(78,h):raise ValueError('COPY_ISOLATION')
   for call in [lambda:r.lookup_o2(76,h),lambda:r.lookup_o2(78,'0'*64),lambda:r.o2_parents(77,next(iter(r.previous.previous.previous.previous.o2.records[77])))]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('EXCLUDED_ROUTE_ACCEPTED')
   oh=next(iter(r.previous.previous.previous.o3.records[1]));x=r.lookup_o3(1,oh);x['record']['entities'][0]='changed'
   if x==r.lookup_o3(1,oh):raise ValueError('O3_COPY_ISOLATION')
   for call in [lambda:r.lookup_o3(65,oh),lambda:r.lookup_o3(1,'missing'),lambda:r.o3_parents(0,next(iter(r.previous.previous.previous.o3.records[0]))),lambda:r.o3_qstate('missing')]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('O3_EXCLUDED_ROUTE_ACCEPTED')
   ok=next(k for k in r.previous.previous.o4.records if k[1]>0);x=r.lookup_o4(*ok);x['record']['proto_ids'][0]='changed'
   if x==r.lookup_o4(*ok):raise ValueError('O4_COPY_ISOLATION')
   for call in [lambda:r.lookup_o4('missing',ok[1],ok[2]),lambda:r.o4_owner(*ok,-1),lambda:r.o4_parents(*ok)]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('O4_EXCLUDED_ROUTE_ACCEPTED')
   k5=next(k for k in r.previous.o5.records if k[1]>0);x=r.lookup_o5(*k5);x['record']['proto_ids'][0]='changed'
   if x==r.lookup_o5(*k5):raise ValueError('O5_COPY_ISOLATION')
   for call in [lambda:r.lookup_o5('missing',k5[1],k5[2]),lambda:r.o5_owner(*k5,-1),lambda:r.o5_parents(*k5)]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('O5_EXCLUDED_ROUTE_ACCEPTED')
   k6=next(k for k in r.o6.records if k[1]>0);x=r.lookup_o6(*k6);x['record']['proto_ids'][0]='changed'
   if x==r.lookup_o6(*k6):raise ValueError('O6_COPY_ISOLATION')
   for call in [lambda:r.lookup_o6('missing',k6[1],k6[2]),lambda:r.o6_owner(*k6,-1),lambda:r.o6_parents(*k6)]:
    try:call()
    except (KeyError,ValueError):pass
    else:raise ValueError('O6_EXCLUDED_ROUTE_ACCEPTED')
  try:r.lookup_o4(*ok)
  except ValueError:pass
  else:raise ValueError('CLOSED_READER')
  report['negative_checks']=['excluded_depth','missing_identity','external_seed_ancestry','copy_isolation','closed_reader'];return ChainExecutionResult(result=report)
