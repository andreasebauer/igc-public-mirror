import sys,json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine')
from infinity_grid import submission as sub
J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);m=json.loads((B/'READBACKS.json').read_text())
def ack(root):
 for x in sub.save_status(root)['pending_objects']:
  r=m[x['sha256']];assert r['raw_readback_verified'];assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==x['sha256']
  sub.confirm_save(root,x['sha256'],r['path'],r['id'],role=x['role'],logical_name=x['logical_name'])
 return sub.save_status(root)
out=ack(J);assert not out['pending_objects'];(B/'CAPTURE_PRESERVED.json').write_text(json.dumps(out,indent=2)+'\n')
rec=sub.capture_record(J);cap=J/'CAPTURE.json';dest=Path('/tmp/ig_native0191/cold_capture')
if not dest.exists():sub.restore_capture(cap,Path('/tmp/ig_native0191/store/objects'),dest)
cold=ack(dest);assert not cold['pending_objects'];assert sub.capture_record(dest)==rec
out={'status':'PASS_EXACT_PREEXECUTION_CAPTURE_RESTORE','capture_id':rec['capture_id'],'cold_workspace':str(dest),'capture_save_pending_objects':0,'handler_called':False,'evaluator_called':False,'scientific_generation_calls':0,'meaning':'Source/input capture cold-restored; scientific state restore remains an execution gate'}
(B/'COLD_CAPTURE_RESTORE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
