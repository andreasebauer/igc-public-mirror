from pathlib import Path
import sys,json,time
D=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
mode,tag=sys.argv[1:3]
def write(n,x):(D/(n+'.json')).write_text(json.dumps(x,indent=2)+'\n')
if mode=='ack':
 state=read(D/(tag+'_STATUS.json'));refs=read(D/'READBACKS.json');by={r['sha256']:r for r in refs};items=state['pending_objects'];assert items
 hashes=sorted({r['sha256'] for r in items});batch={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':s['capture_id'],'obligations':[{k:r[k] for k in ['obligation_id','obligation_scope','role','logical_name','sha256','size_bytes']} for r in items],'readbacks':[by[h] for h in hashes]};write(tag+'_BATCH',batch);write(tag+'_ACK',pr.confirm_batch(w,D/(tag+'_BATCH.json')))
state=pr.status(w);write(tag+('_POST_STATUS' if mode=='ack' else '_STATUS'),state);print(json.dumps({'pending_roles':len(state['pending_objects']),'pending_bytes':state['pending_bytes'],'pending_checkpoints':len(state['pending_checkpoints']),'oldest_age':state['oldest_pending_age_seconds'],'objects':list({x['sha256']:x for x in state['pending_objects']}.values())}))
