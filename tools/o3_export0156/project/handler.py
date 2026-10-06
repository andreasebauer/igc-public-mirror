from pathlib import Path
import hashlib,json,gzip
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import CarrierReader

def handler(stage,runtime):
 p=stage['execution']['parameters'];i=stage['input_artifacts']
 if hashlib.sha256(Path(i['catalog']).read_bytes()).hexdigest()!=p['catalog_sha256']:raise ValueError('CATALOG_HASH')
 r=CarrierReader(i['archive'],p['archive_sha256'],p['root_sha256']);count=links=nested=comps=0
 for rank,rows in r.records.items():
  if json.loads(gzip.decompress(r.content_bytes('panel'+str(rank))))!=r.panels[rank]:raise ValueError('LOSSLESS_PANEL')
  for h,row in rows.items():
   if r.lookup(rank,h)['record']!=row:raise ValueError('LOSSLESS_ROUTE')
   if rank:
    got=r.parents(rank,h)
    if [x['exact_key'] for x in got]!=row['parent_keys'] or any(x['record']!=r.records[rank-1][x['exact_key']] for x in got):raise ValueError('PARENT_ROUTE')
    links+=len(got)
   for q in row['entities']:
    if r.qstate(q)['state']!=r.qstates[q]:raise ValueError('NESTED_Q_ROUTE')
    nested+=1
   for c in r.components(rank,h):
    if len(c['edges'])!=c['diagnostic']['m3'] or len(c['entities'])!=c['diagnostic']['n3']:raise ValueError('COMPONENT_ROUTE')
    comps+=1
   count+=1
 # Mutating returned roots, nested resources and components must not mutate saved data.
 h=next(iter(r.records[1]));x=r.lookup(1,h);x['record']['entities'][0]='mutated';assert r.lookup(1,h)!=x
 q=next(iter(r.qstates));x=r.qstate(q);x['state']['sites'][0][0][0]+=1;assert r.qstate(q)!=x
 x=r.components(1,h);x[0]['edges'][0][0]=999;assert r.components(1,h)!=x
 for call in [lambda:r.lookup(1,'missing'),lambda:r.lookup(65,h),lambda:r.parents(0,next(iter(r.records[0]))),lambda:r.qstate('missing')]:
  try:call()
  except (KeyError,ValueError):pass
  else:raise ValueError('NEGATIVE_ROUTE_ACCEPTED')
 report=dict(r.report);report.update(outcome='PASS',all_occurrence_routes_checked=count,all_parent_routes_checked=links,all_nested_Q_routes_checked=nested,all_component_routes_checked=comps,scientific_root_sha256=p['root_sha256'],archive_sha256=p['archive_sha256'],scientific_master_admission=False)
 r.close()
 try:r.lookup(1,h)
 except ValueError:pass
 else:raise ValueError('CLOSED_ROUTE')
 report['negative_checks']=['missing_identity','excluded_rank','external_seed_ancestry','missing_Q_state','root_nested_component_copy_isolation','closed_reader'];return ChainExecutionResult(result=report)
