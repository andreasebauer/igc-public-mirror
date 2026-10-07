from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
s=p.status(J);(B/'PRESERVATION_STATUS.json').write_text(json.dumps(s,indent=2));d=p.drain_plan(J);(B/'DRAIN_PLAN.json').write_text(json.dumps(d,indent=2));m=json.loads((B/'READBACKS.json').read_text());new=[x for x in d['physical_objects'] if x['sha256'] not in m];(B/'NEW_CHECKPOINT_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'pending':len(s['pending_objects']),'new':len(new),'new_bytes':sum(x['size_bytes'] for x in new)}))
