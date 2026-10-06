from pathlib import Path
import hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import PanelReader

def handler(stage,runtime):
 p=stage['execution']['parameters'];i=stage['input_artifacts']
 if hashlib.sha256(Path(i['catalog']).read_bytes()).hexdigest()!=p['catalog_sha256']:raise ValueError('CATALOG_HASH')
 r=PanelReader(i['archive'],p['archive_sha256'],p['root_sha256']);count=0
 for L,rows in r.records.items():
  for h,row in rows.items():
   got=r.lookup(L,h)
   if got['record']!=row or (L>38 and any(ph not in r.records[L-1] for ph in got['parents'])):raise ValueError('ROUTE')
   count+=1
 report=dict(r.report);report.update(outcome='PASS',all_occurrence_routes_checked=count,scientific_root_sha256=p['root_sha256'],archive_sha256=p['archive_sha256'],scientific_master_admission=False)
 L=39;h=next(iter(r.records[L]));mut=r.lookup(L,h);mut['parents'].clear()
 if not r.lookup(L,h)['parents']:raise ValueError('COPY_ISOLATION')
 try:r.lookup(39,'0'*64)
 except KeyError:pass
 else:raise ValueError('MISSING_ROUTE_ACCEPTED')
 r.close()
 try:r.lookup(L,h)
 except ValueError:pass
 else:raise ValueError('CLOSED_ROUTE')
 report['negative_checks']=['missing_record','copy_isolation','closed_reader'];return ChainExecutionResult(result=report)
