from pathlib import Path
import json,sys,hashlib,shutil
D=Path(__file__).resolve().parent;P=D.parent/'v69b_preparation';read=lambda p:json.loads(p.read_text());s=read(P/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
from infinity_grid.save_transport import unpack
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
objects=D/'restore_objects';objects.mkdir()
for r in read(P/'READBACKS.json'):
 p=objects/(r['sha256']+'.bin')
 if r['kind']=='RAW':
  src=Path(r['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==r['sha256'];shutil.copyfile(src,p)
 else:unpack(r['manifest'],r['parts'],p)
e=read(D/'EXPORT.json');rb=read(D/'EXPORT_READBACK.json');assert hashlib.sha256(Path(rb['path']).read_bytes()).hexdigest()==e['sha256'];dest=Path('/tmp/ig_v70_failure_restored_20261001');result=pr.restore_checkpoint(Path(rb['path']),dest,e['sha256'],objects)
for name,r in read(D/'ATTEMPTS.json').items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==r['sha256']
assert verified_completion(validate_workspace_job(dest,s['job_id'],check_loaded=False),allow_pending_checkpoint=True) is None
report={'restore':result,'attempt_bytes_identical':True,'native_status':'RUNNING_STALE_RECORD_PRESERVED','completion':None,'workloads_dispatched':0,'export_sha256':e['sha256']};(D/'RESTORE_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
