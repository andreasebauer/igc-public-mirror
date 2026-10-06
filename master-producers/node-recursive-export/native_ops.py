from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parent;B=R.parents[1]
mode=sys.argv[1]
engine=Path(json.loads((R/'CAPTURE_SAVE_STATUS.json').read_text())['workspace'])/'source' if mode=='run' else B/'engine'
sys.path.insert(0,str(engine))
from infinity_grid import submission as sub,portable_registry,preservation as pr,v05_controller_event_loop as loop
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
def write(n,obj):(R/n).write_text(json.dumps(obj,indent=2)+'\n')
if mode=='capture':
 spec=json.loads((R/'CAPTURE_SPEC.json').read_text());q=json.loads((R/'QUALIFICATION.json').read_text())
 assert q['status']=='PASS' and engineering_source_tree_digest(B/'engine')==q['engine_digest']
 assert hashlib.sha256((R/'project/reader.py').read_bytes()).hexdigest()==q['adapter_sha256']
 assert hashlib.sha256((R/'project/handler.py').read_bytes()).hexdigest()==q['handler_sha256']
 assert json.loads((R/'PREFLIGHT.json').read_text())['status']=='PASS'
 assert not (R/'CAPTURE_SAVE_STATUS.json').exists()
 store=B/'campaign/stores/node_recursive_scientific_export'
 if not (store/'coordination/PROJECT.json').exists():portable_registry.initialize(store,'Native recursive scientific export',B/'engine')
 out=sub.capture(store,spec);write('CAPTURE_SAVE_STATUS.json',out);J=Path(out['workspace']);p=sub.save_status(J);write('CAPTURE_PENDING.json',p);print(json.dumps(p))
else:
 J=Path(json.loads((R/'CAPTURE_SAVE_STATUS.json').read_text())['workspace'])
 if mode=='confirm_capture':
  mapping=json.loads((R/'CAPTURE_READBACK_MAPPING.json').read_text())
  for row in sub.save_status(J)['pending_objects']:
   m=mapping[row['sha256']];sub.confirm_save(J,row['sha256'],m['path'],m['drive_file_id'],role=row['role'],logical_name=row['logical_name'])
  out=sub.save_status(J);assert not out['pending_objects'];write('CAPTURE_SAVE_CONFIRMED.json',out);print(json.dumps({'status':out['status']}))
 elif mode=='run':
  out=loop.run_workspace_job(J,sub.capture_record(J)['job']['job_id']);write('NATIVE_RESULT.json',out);print(json.dumps(out))
 elif mode=='pending':
  out=pr.drain_plan(J);write('CHECKPOINT_PENDING.json',out);print(json.dumps({'objects':len(out['physical_objects']),'bytes':sum(x['size_bytes'] for x in out['physical_objects'])}))
 elif mode=='confirm_checkpoint':
  plan=pr.drain_plan(J);mapping=json.loads((R/'CHECKPOINT_READBACK_MAPPING.json').read_text());refs=[]
  for h in {r['sha256'] for r in plan['obligations']}:
   m=mapping[h];refs.append({'sha256':h,'kind':'RAW','path':m['path'],'drive_file_id':m['drive_file_id']})
  request={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':J.name,'obligations':plan['obligations'],'readbacks':refs};write('CHECKPOINT_ACK.request.json',request)
  out=pr.confirm_batch(J,R/'CHECKPOINT_ACK.request.json');write('CHECKPOINT_ACK.json',out);state=pr.status(J);assert state['pending_bytes']==0;write('PRESERVATION_STATUS.json',state);print(json.dumps({'status':state['status']}))
 elif mode=='export':
  out=pr.export_checkpoint(J,R/'FINAL_CHECKPOINT_SLIM.zip',slim=True);write('CHECKPOINT_EXPORT.json',out);print(json.dumps({'sha256':out['sha256'],'dependencies':len(out['dependencies'])}))
 else:raise ValueError(mode)
