"""Snapshot idle refused capture; does not execute scientific cases."""
from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
r=p.snapshot_workspace(J,reason='NATIVE_PREFLIGHT_REFUSAL_OFFICIAL_CASES_ZERO');(B/'STOP_SNAPSHOT.json').write_text(json.dumps(r,indent=2,default=str));print('PASS idle refusal snapshot')
