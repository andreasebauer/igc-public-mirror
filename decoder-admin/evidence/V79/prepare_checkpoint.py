from pathlib import Path
import sys,json
D=Path(__file__).parent;s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr,submission as sub
assert not sub.save_status(w)['pending_objects']
assert not list((w/'runtime/attempts').rglob('*.json'))
s=pr.make_checkpoint(w,'V79_PREREGISTERED_PRE_RUN')
(D/'PRE_RUN_STATUS.json').write_text(json.dumps(s,indent=2))
print(json.dumps({'pending_roles':len(s['pending_objects']),'pending_checkpoints':len(s['pending_checkpoints'])}))
