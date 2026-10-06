from pathlib import Path
import hashlib,json,gzip
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import CarrierReader

def handler(stage,runtime):
 p=stage['execution']['parameters'];i=stage['input_artifacts']
 if hashlib.sha256(Path(i['catalog']).read_bytes()).hexdigest()!=p['catalog_sha256']:raise ValueError('CATALOG_HASH')
 r=CarrierReader(i['archive'],p['archive_sha256'],p['root_sha256']);count=links=0
 for d,rows in r.records.items():
  raw=json.loads(gzip.decompress(r.content_bytes('panel'+str(d))))
  if raw!=r.panels[d]:raise ValueError('LOSSLESS_PANEL')
  json.loads(r.content_bytes('observer'+str(d)))
  for h,row in rows.items():
   if r.lookup(d,h)['record']!=row:raise ValueError('LOSSLESS_ROUTE')
   if d>77:
    got=r.parents(d,h)
    if [x['exact_key'] for x in got]!=row['parents'] or any(x['record']!=r.records[d-1][x['exact_key']] for x in got):raise ValueError('PARENT_ROUTE')
    links+=len(got)
   count+=1
 report=dict(r.report);report.update(outcome='PASS',all_occurrence_routes_checked=count,all_parent_routes_checked=links,scientific_root_sha256=p['root_sha256'],archive_sha256=p['archive_sha256'],scientific_master_admission=False)
 h=next(iter(r.records[78]));mut=r.lookup(78,h);mut['record']['states'][0][0]+=1
 if mut==r.lookup(78,h):raise ValueError('COPY_ISOLATION')
 for call in [lambda:r.lookup(78,'0'*64),lambda:r.lookup(76,h),lambda:r.parents(77,next(iter(r.records[77])))]:
  try:call()
  except (KeyError,ValueError):pass
  else:raise ValueError('NEGATIVE_ROUTE_ACCEPTED')
 r.close()
 try:r.lookup(78,h)
 except ValueError:pass
 else:raise ValueError('CLOSED_ROUTE')
 report['negative_checks']=['missing_identity','excluded_depth','external_seed_ancestry','copy_isolation','closed_reader'];return ChainExecutionResult(result=report)
