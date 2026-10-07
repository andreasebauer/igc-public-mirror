from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine')
from infinity_grid import submission as sub
J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);m=json.loads((B/'READBACKS.json').read_text())
for x in sub.save_status(J)['pending_objects']:
 r=m[x['sha256']];sub.confirm_save(J,x['sha256'],r['path'],r['id'],role=x['role'],logical_name=x['logical_name'])
s=sub.save_status(J);assert not s['pending_objects'];(B/'CAPTURE_PRESERVED.json').write_text(json.dumps(s,indent=2)+'\n');print('capture preserved')
