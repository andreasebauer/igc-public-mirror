from pathlib import Path
import sys,json,traceback
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import submission as sub
from infinity_grid.v05_controller_event_loop import run_workspace_job
if __name__=='__main__':
 try:
  out=run_workspace_job(J,sub.capture_record(J)['job']['job_id']);(B/'NATIVE_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
 except Exception:
  (B/'ERROR.txt').write_text(traceback.format_exc());raise
