"""Ask the native controller to stop at its monitored preservation boundary."""
from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as pr
result=pr.request_pause(J,'USER_REQUESTED_NEW_CHAT_HANDOFF_2026-10-08')
(B/'PAUSE_REQUEST_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
