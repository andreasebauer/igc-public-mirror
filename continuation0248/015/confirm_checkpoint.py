from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
r=p.confirm_batch(J,B/'ACK_BATCH.json');s=p.status(J);assert not s['pending_objects'];(B/'CHECKPOINT_PRESERVED.json').write_text(json.dumps(s,indent=2));print('PASS checkpoint pending0')
