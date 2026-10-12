from pathlib import Path
import sys,json,resource,hashlib
h=Path(__file__).resolve().parent;p=json.loads((h/'results/CAPTURE_PENDING.json').read_text())
for name,digest in json.loads((h/'PREREGISTRATION.json').read_text())['scripts'].items():
 assert hashlib.sha256((h/name).read_bytes()).hexdigest()==digest,name
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
sys.path.insert(0,p['workspace']+'/source')
from infinity_grid import v05_controller_event_loop as loop
result=loop.run_workspace_job(p['workspace'],p['capture']['job_id'])
(h/'results/NATIVE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
