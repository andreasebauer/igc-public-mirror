from pathlib import Path
import sys,json,hashlib,importlib,shutil
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204');sys.path.insert(0,str(B))
for p in (B/'project').rglob('*.py'):compile(p.read_text(),str(p),'exec')
rs=importlib.import_module('project.historical.regime_scanner');mp=importlib.import_module('project.historical.maturation_parallel');md=importlib.import_module('project.historical.materialized_discovery')
assert mp.rs is rs and md.rs is rs
assert md._PROCESS_O7_RUNTIME is None
from infinity_grid.workflow_guard import preflight
C=Path('/tmp/ig_preflight0212');shutil.copytree('/tmp/ig_engine0204/infinity_grid',C/'infinity_grid',dirs_exist_ok=True);shutil.copytree(B/'project',C/'project',dirs_exist_ok=True)
r=preflight(C,list((C/'project').rglob('*.py')))
(B/'PREFLIGHT.json').write_text(json.dumps(r,indent=2))
assert r['status']=='PASS'
(B/'IMPORT_GATE.json').write_text(json.dumps({'status':'PASS_ISOLATED_HISTORICAL_NAMESPACE_NO_PRODUCER_EXECUTED','historical_sibling_routes_verified':True,'historical_kernel_initialized':False,'generator_calls':0,'all_project_sources_compile':True,'sha256_by_file':{str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (B/'project').rglob('*.py')}},indent=2))
print('PASS compiled source/import namespace/native preflight')
