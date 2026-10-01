from pathlib import Path
import json,sys,time,hashlib
D=Path(__file__).resolve().parent;state=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(state['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr,submission as sub
from infinity_grid.v05_controller_event_loop import ControllerLoopError
mode=sys.argv[1];tag=sys.argv[2];start=time.time()
read_counts={}
readback_paths={x['readback'] for x in json.loads((D/'CATALOG.json').read_text())}
def observe(frame,event,arg):
 if event=='return' and frame.f_code.co_name=='read_bytes' and isinstance(arg,bytes):
  path=str(frame.f_locals.get('self',''))
  if '/durability/outbox/objects/' in path or path in readback_paths:
   row=read_counts.setdefault(path,{'calls':0,'bytes':0});row['calls']+=1;row['bytes']+=len(arg)
sys.setprofile(observe)
def write(n,d):(D/(n+'.json')).write_text(json.dumps(d,indent=2)+'\n')
def summary(s):
 attempts=[json.loads(p.read_text()) for p in (w/'runtime/attempts').rglob('*.json')]
 return {'pending_roles':len(s['pending_objects']),'pending_bytes':s['pending_bytes'],'pending_commits':len(s['pending_checkpoints']),'oldest_age':s['oldest_pending_age_seconds'],'count_reserve':8-len(s['pending_checkpoints']),'byte_reserve_after_commit':536870912-s['pending_bytes']-201326592,'age_reserve':3600-s['oldest_pending_age_seconds'],'native_physical_objects_verified':s['physical_objects_verified'],'attempt_statuses':[a['status'] for a in attempts], 'selector_finished':any(json.loads(p.read_text()).get('finished',False) for p in (w/'runtime/runs').rglob('nodes/*.json'))}
try:
 if mode=='capture':
  for x in json.loads((D/'CATALOG.json').read_text()):sub.confirm_save(w,x['sha256'],x['readback'],x['drive_id'],role=x['role'],logical_name=x['logical_name'])
  assert not sub.save_status(w)['pending_objects']
 if mode=='ack':
  path=D/(tag+'_BATCH.json')
  if not path.exists():
   s=json.loads((D/(tag+'_STATUS.json')).read_text());rr=json.loads((D/'CATALOG.json').read_text());by={r['sha256']:r for r in rr};items=s['pending_objects']
   batch={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':state['capture_id'],'obligations':[{k:x[k] for k in ['obligation_id','obligation_scope','role','logical_name','sha256','size_bytes']} for x in items],'readbacks':[{'sha256':h,'kind':'RAW','drive_file_id':by[h]['drive_id'],'path':by[h]['readback']} for h in sorted({x['sha256'] for x in items})]}
   write(tag+'_BATCH',batch)
  result=pr.confirm_batch(w,path);write(tag+'_ACK',result)
 s=pr.status(w)
 write(tag+('_POST_STATUS' if mode=='ack' else '_STATUS'),s)
 info={'started':start,'finished':time.time(),'mode':mode,'tag':tag,**summary(s)}
 if mode=='ack':
  prior=json.loads((D/(tag+'_STATUS.json')).read_text());pending={x['checkpoint_sha256'] for x in s['pending_checkpoints']};info['closed_checkpoints']=[x for x in prior['pending_checkpoints'] if x['checkpoint_sha256'] not in pending]
 sys.setprofile(None)
 info['payload_read_bytes_calls']=read_counts
 write(tag+'_'+mode+'_METRICS',info)
 unique={x['sha256']:x for x in s['pending_objects']}
 print(json.dumps({'metrics':info,'objects':list(unique.values())}))
except ControllerLoopError as exc:
 if str(exc)!='WORKSPACE_BUSY':raise
 print(json.dumps({'busy':True,'mode':mode,'tag':tag,'time':time.time()}));sys.exit(75)
