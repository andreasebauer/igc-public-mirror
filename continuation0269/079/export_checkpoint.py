from pathlib import Path
import json,sys
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
r=p.export_checkpoint(J,B/'NATIVE_CHECKPOINT_SLIM.zip',slim=True);(B/'CHECKPOINT_EXPORT.json').write_text(json.dumps(r,indent=2));print('PASS native export')
