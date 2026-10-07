from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine')
from infinity_grid import portable_registry,submission
R=Path('/tmp/ig_native0191');store=R/'store'
if not (store/'coordination/PROJECT.json').exists():portable_registry.initialize(store,'Bound historical-seed G1 exact ancestry reconstruction',Path('/tmp/ig_admit0188/engine'))
out=submission.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(out,indent=2)+'\n');J=Path(out['workspace']);pending=submission.save_status(J);(B/'PENDING.json').write_text(json.dumps(pending,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k in ['capture_id','workspace','status','capture_status','job_id']}));print(json.dumps({'pending_bytes':sum(x['size_bytes'] for x in pending['pending_objects']),'pending_objects':len(pending['pending_objects'])}))
