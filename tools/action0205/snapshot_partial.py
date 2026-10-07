from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as pr
assert (J/'runtime/PAUSE_ACK.json').exists()
result=pr.snapshot_workspace(J,'USER_REQUESTED_PARTIAL_HANDOFF_AFTER_GENERATION94')
(B/'PARTIAL_SNAPSHOT.json').write_text(json.dumps(result,indent=2));print('PASS paused workspace snapshot retained')
