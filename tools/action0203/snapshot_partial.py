from pathlib import Path
import json,sys
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
r=p.snapshot_workspace(J,'PARTIAL_DEPTH88_STREAM_RESULT_BYTES_LIMIT_AT89');(B/'PARTIAL_SNAPSHOT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
