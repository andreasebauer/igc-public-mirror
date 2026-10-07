"""Registered native operations only; scientific handler is controller-owned."""
from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent
R=Path('/tmp/ig_admit0188');mode=sys.argv[1]
source=R/'engine' if mode!='run' else Path(json.loads((B/'POINTER.json').read_text())['workspace'])/'source'
sys.path.insert(0,str(source))
from infinity_grid import submission as sub,preservation as pr,portable_registry
from infinity_grid.v05_controller_event_loop import run_workspace_job
def write(n,x):(B/n).write_text(json.dumps(x,indent=2)+'\n')
if mode=='capture':
 if not (R/'store/coordination/PROJECT.json').exists():portable_registry.initialize(R/'store','Master151 saved G2 S1 admission and published-reader qualification',source)
 out=sub.capture(R/'store',R/'SPEC.json');write('POINTER.json',out)
else:
 J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
 if mode=='pending':out=sub.save_status(J)
 elif mode=='ack':
  mapping=json.loads((B/'READBACKS.json').read_text());large=json.loads((B/'TRANSPORTS.json').read_text())
  for row in sub.save_status(J)['pending_objects']:
   h=row['sha256']
   if h in large:
    x=large[h];sub.confirm_transport(J,h,x['manifest'],x['parts'],x['drive_file_id'])
   else:
    x=mapping[h];sub.confirm_save(J,h,x['path'],x['id'],role=row['role'],logical_name=row['logical_name'])
  out=sub.save_status(J)
 elif mode=='run':out=run_workspace_job(J,sub.capture_record(J)['job']['job_id']);write('NATIVE_RESULT.json',out)
 elif mode=='checkpoint_pending':out=pr.drain_plan(J)
 elif mode=='checkpoint_ack':
  plan=pr.drain_plan(J);mapping=json.loads((B/'READBACKS.json').read_text());large=json.loads((B/'TRANSPORTS.json').read_text());refs=[]
  for h in {r['sha256'] for r in plan['obligations']}:
   if h in large:
    x=large[h];refs.append({'sha256':h,'kind':'MULTIPART','manifest':x['manifest'],'parts':x['parts'],'drive_file_id':x['drive_file_id']})
   else:refs.append({'sha256':h,'kind':'RAW','path':mapping[h]['path'],'drive_file_id':mapping[h]['id']})
  req={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':J.name,'obligations':plan['obligations'],'readbacks':refs};write('CHECKPOINT_ACK_REQUEST.json',req)
  out=pr.confirm_batch(J,B/'CHECKPOINT_ACK_REQUEST.json')
 elif mode=='status':out=pr.status(J)
 elif mode=='export':out=pr.export_checkpoint(J,R/'FINAL_CHECKPOINT_SLIM.zip',slim=True)
 else:raise ValueError(mode)
write(mode.upper()+'.json',out)
print(json.dumps({k:v for k,v in out.items() if k in ['status','capture_status','capture_id','job_id','workspace','pending_bytes','reused','evidence_status','completion_sha256','result_sha256','path','sha256']}))
