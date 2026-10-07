"""Administrative registered native operations; never calls a project handler directly."""
from pathlib import Path
import json,sys
B=Path(__file__).resolve().parent;mode=sys.argv[1];R=Path('/tmp/ig_g2_s1_native0187');SPEC=Path('/tmp/ig_g2_s1_capture0187_prepared/SPEC.json')
if mode=='run':source=Path(json.loads((B/'POINTER.json').read_text())['workspace'])/'source'
else:source=Path('/tmp/ig_g2_s1_capture0187_prepared/engine')
sys.path.insert(0,str(source))
from infinity_grid import submission as sub,portable_registry,preservation as pr
from infinity_grid.v05_controller_event_loop import run_workspace_job
def write(n,r):(B/n).write_text(json.dumps(r,indent=2)+'\n')
if mode=='capture':
 if not (R/'store/coordination/PROJECT.json').exists():portable_registry.initialize(R/'store','Registered saved G2 S1 scoped integration; no generation or publication',source)
 out=sub.capture(R/'store',str(SPEC));write('POINTER.json',out)
else:
 J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
 if mode=='pending':out=sub.save_status(J)
 elif mode=='ack':
  mapping=json.loads((B/'READBACKS.json').read_text())
  large={x['original']['sha256']:x for x in json.loads((B/'TRANSPORT_OBJECTS.json').read_text())}
  for row in sub.save_status(J)['pending_objects']:
   if row['sha256'] in large:
    x=large[row['sha256']];sub.confirm_transport(J,row['sha256'],x['manifest_readback_path'],x['verified_parts'],x['manifest_drive_file_id'])
   else:
    m=mapping[row['sha256']];sub.confirm_save(J,row['sha256'],m['path'],m['id'],role=row['role'],logical_name=row['logical_name'])
  out=sub.save_status(J)
 elif mode=='run':out=run_workspace_job(J,sub.capture_record(J)['job']['job_id']);write('NATIVE_RESULT.json',out)
 elif mode=='checkpoint_pending':out=pr.drain_plan(J)
 elif mode=='checkpoint_ack':
  plan=pr.drain_plan(J);mapping=json.loads((B/'READBACKS.json').read_text())
  large={x['original']['sha256']:x for x in json.loads((B/'TRANSPORT_OBJECTS.json').read_text())};refs=[]
  for h in {r['sha256'] for r in plan['obligations']}:
   if h in large:
    x=large[h];refs.append({'sha256':h,'kind':'MULTIPART','manifest':x['manifest_readback_path'],'parts':x['verified_parts'],'drive_file_id':x['manifest_drive_file_id']})
   else:refs.append({'sha256':h,'kind':'RAW','path':mapping[h]['path'],'drive_file_id':mapping[h]['id']})
  req={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':J.name,'obligations':plan['obligations'],'readbacks':refs};write('CHECKPOINT_ACK_REQUEST.json',req)
  out=pr.confirm_batch(J,B/'CHECKPOINT_ACK_REQUEST.json')
 elif mode=='export':out=pr.export_checkpoint(J,R/'FINAL_CHECKPOINT_SLIM.zip',slim=True)
 else:raise ValueError(mode)
write(mode.upper()+'.json',out)
print(json.dumps({k:v for k,v in out.items() if k in ['status','capture_status','capture_id','job_id','workspace','pending_bytes','reused','evidence_status','completion_sha256','result_sha256','path','sha256']}))
