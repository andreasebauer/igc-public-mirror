from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import submission as s
from infinity_grid.v05_controller_event_loop import validate_workspace_job
m=json.loads((B/'READBACKS.json').read_text())
for x in s.save_status(J)['pending_objects']:
 r=m[x['sha256']];assert r['raw_readback_verified'];s.confirm_save(J,x['sha256'],r['path'],r['id'],role=x['role'],logical_name=x['logical_name'])
p=s.save_status(J);assert not p['pending_objects'];(B/'CAPTURE_PRESERVED.json').write_text(json.dumps(p,indent=2));a=validate_workspace_job(J,s.capture_record(J)['job']['job_id']);(B/'ADMISSION.json').write_text(json.dumps({'status':'PASS_NATIVE_ENVIRONMENT_ADMISSION','capture_id':J.name,'generator_calls':0},indent=2));print('PASS capture preserved and native admission')
