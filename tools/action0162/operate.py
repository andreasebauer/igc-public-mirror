from pathlib import Path
import json,sys
B=Path(__file__).resolve().parent;mode=sys.argv[1]
source=Path(json.loads((B/'POINTER.json').read_text())['workspace'])/'source' if mode=='run' else B.parent/'engine'
sys.path.insert(0,str(source))
from infinity_grid import submission as sub,portable_registry,preservation as pr
from infinity_grid.v05_controller_event_loop import run_workspace_job
mode=sys.argv[1]
def write(n,r):(B/n).write_text(json.dumps(r,indent=2)+'\n')
if mode=='capture':
 if not (B/'store/coordination/PROJECT.json').exists():portable_registry.initialize(B/'store','Immutable case5 pair branch reconciliation',B.parent/'engine')
 out=sub.capture(B/'store',json.loads((B/'SPEC.json').read_text()));write('POINTER.json',out)
else:
 J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
 if mode=='pending':out=sub.save_status(J)
 elif mode=='ack':
  mapping=json.loads((B/'READBACKS.json').read_text())
  for row in sub.save_status(J)['pending_objects']:
   m=mapping[row['sha256']];sub.confirm_save(J,row['sha256'],m['path'],m['id'],role=row['role'],logical_name=row['logical_name'])
  out=sub.save_status(J)
 elif mode=='run':out=run_workspace_job(J,sub.capture_record(J)['job']['job_id']);write('NATIVE_RESULT.json',out)
 elif mode=='checkpoint_pending':out=pr.drain_plan(J)
 elif mode=='checkpoint_ack':
  plan=pr.drain_plan(J);mapping=json.loads((B/'CHECKPOINT_READBACKS.json').read_text())
  refs=[{'sha256':h,'kind':'RAW','path':mapping[h]['path'],'drive_file_id':mapping[h]['id']} for h in {r['sha256'] for r in plan['obligations']}]
  write('CHECKPOINT_ACK_REQUEST.json',{'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':J.name,'obligations':plan['obligations'],'readbacks':refs})
  out=pr.confirm_batch(J,B/'CHECKPOINT_ACK_REQUEST.json')
 elif mode=='export':out=pr.export_checkpoint(J,B/'FINAL_CHECKPOINT_SLIM.zip',slim=True)
 else:raise ValueError(mode)
write(mode.upper()+'.json',out)
print(json.dumps(out))
