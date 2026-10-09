from pathlib import Path
import sys,json,shutil
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid.workflow_guard import preflight
C=Path('/tmp/ig_preflight0233');shutil.copytree('/tmp/ig_engine0204/infinity_grid',C/'infinity_grid',dirs_exist_ok=True);shutil.copytree(B/'project',C/'project',dirs_exist_ok=True)
for p in (B/'project').glob('*.py'):compile(p.read_text(),str(p),'exec')
r=preflight(C,list((C/'project').rglob('*.py')));(B/'PREFLIGHT.json').write_text(json.dumps(r,indent=2));assert r['status']=='PASS';print('PASS native source preflight')
