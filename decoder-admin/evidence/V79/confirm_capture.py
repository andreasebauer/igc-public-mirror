from pathlib import Path
import sys,json,hashlib
D=Path(__file__).parent;s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub
cat=json.loads((D/'SAVE_CATALOG.json').read_text());by={r['sha256']:r for r in cat}
for r in cat:
 raw=Path(r['readback']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'] and len(raw)==r['size_bytes']
for o in s['pending_objects']:
 r=by[o['sha256']];sub.confirm_save(w,o['sha256'],r['readback'],r['drive_id'],role=o['role'],logical_name=o['logical_name'])
state=sub.save_status(w);assert not state['pending_objects'];(D/'CAPTURE_CONFIRMED.json').write_text(json.dumps(state,indent=2));print('10 capture roles acknowledged; operator bundle byte-equal.')
