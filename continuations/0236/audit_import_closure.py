"""Independent complete static native closure audit, without science execution."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;Q=B.parent/'continuation0235';J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);S=J/'source';sys.path.insert(0,str(S))
from infinity_grid.workflow_guard import _imports,_module_file,RUNTIME_MODULES
from infinity_grid.v05_stage_architecture import audit_module_source
pending=[(S/'project/handler.py',[]),(S/'project/worker.py',[])];seen=set();rows=[]
while pending:
 p,chain=pending.pop()
 if p in seen:continue
 seen.add(p);r=audit_module_source(p);r['path']=str(p.relative_to(S));r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['import_chain']=chain;rows.append(r)
 for n in _imports(p,S):
  if n in RUNTIME_MODULES:continue
  target=_module_file(S,n)
  if target:pending.append((target,chain+[r['path']]))
failed=[r for r in rows if r['status']!='PASS'];old=json.loads((Q/'PREFLIGHT.json').read_text())
pre=json.loads((Q/'PREREGISTRATION.json').read_text())
for n,h in pre['source_sha256'].items():
 path=S/'infinity_grid/uplift_structural.py' if n.startswith('engine/') else S/n
 assert hashlib.sha256(path.read_bytes()).hexdigest()==h
assert failed and any(r['path']=='infinity_grid/semantic_sentinel.py' for r in failed)
out=dict(status='PASS_INDEPENDENT_NATIVE_REFUSAL_DIAGNOSIS',scientific_source_pins_unchanged=True,official_cases_executed=0,native_modules_checked=len(rows),local_preflight_modules_checked=len(old['native_guard']['modules']),root_cause='Earlier local preflight used project parent as source and could not traverse engine imports; native captured source includes engine and traverses full modules including imports inside unused functions.',failed_modules=failed,modules=rows,master_slices=152,new_admissions=0,G2_promotion=False,next_scope='REPAIR_COMPLETE_NATIVE_IMPORT_PREFLIGHT_WITHOUT_CHANGING_SCIENTIFIC_RECIPE')
(B/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['modules','failed_modules']}));print(json.dumps([dict(path=r['path'],violations=r['violations'],import_chain=r['import_chain']) for r in failed]))
