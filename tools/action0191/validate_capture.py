from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import submission as sub
from infinity_grid.v05_controller_event_loop import validate_workspace_job
J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);job=sub.capture_record(J)['job']['job_id'];a=validate_workspace_job(J,job)
result={'status':'PASS_NATIVE_CAPTURE_ENVIRONMENT_ADMISSION','capture_id':J.name,'job_id':job,'handler_called':False,'evaluator_called':False,'generator_calls':0,'source_sha256':a.get('source_sha256'),'pending_capture_objects':len(sub.save_status(J)['pending_objects'])}
(B/'NATIVE_ADMISSION_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
