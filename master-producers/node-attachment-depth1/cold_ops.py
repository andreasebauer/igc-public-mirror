from pathlib import Path
import sys,json,hashlib,shutil
R=Path(__file__).resolve().parent;mode=sys.argv[1]
sys.path.insert(0,str(R/'cold_restored/source') if mode=='reuse' else str(R.parent/'engine'))
from infinity_grid import preservation as pr,submission as sub,v05_controller_event_loop as loop
def load(n):return json.loads((R/n).read_text())
def write(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
if mode=='restore':
 saved=load('CHECKPOINT_ARCHIVE_SAVED.json');assert hashlib.sha256(Path(saved['path']).read_bytes()).hexdigest()==saved['sha256']
 export=load('CHECKPOINT_EXPORT.json');assert saved['sha256']==export['sha256'];mapping=load('CHECKPOINT_READBACK_MAPPING.json');O=R/'cold_restore_objects';O.mkdir(exist_ok=True)
 for dep in export['dependencies']:
  src=Path(mapping[dep['sha256']]['path']);assert src.stat().st_size==dep['size_bytes'] and hashlib.sha256(src.read_bytes()).hexdigest()==dep['sha256'];target=O/(dep['sha256']+'.bin')
  if not target.exists():shutil.copy2(src,target)
 out=pr.restore_checkpoint(saved['path'],R/'cold_restored',export['sha256'],objects=O);write('COLD_RESTORE.json',out);print(json.dumps({'status':out['status']}))
elif mode=='reuse':
 J=R/'cold_restored';out=loop.run_workspace_job(J,sub.capture_record(J)['job']['job_id']);old=load('NATIVE_RESULT.json')
 assert out['reused'] and out['completion_sha256']==old['completion_sha256'] and out['result_sha256']==old['result_sha256'] and out['evidence_status']=='VERIFIED'
 status=pr.status(J);assert status['pending_bytes']==0
 write('COLD_REUSE_RESULT.json',{'status':'PASS','reused':out['reused'],'evidence_status':out['evidence_status'],'completion_sha256':out['completion_sha256'],'result_sha256':out['result_sha256'],'pending_bytes':0,'generation_reexecuted':False})
 print(json.dumps(load('COLD_REUSE_RESULT.json')))
else:raise ValueError(mode)
