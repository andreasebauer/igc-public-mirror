import sys,json
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine')
from infinity_grid import submission as sub
out=sub.capture('/tmp/ig_native0191/store',B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(out,indent=2)+'\n');pending=sub.save_status(out['workspace']);(B/'PENDING.json').write_text(json.dumps(pending,indent=2)+'\n');old=json.loads((B.parent/'native0191/READBACKS.json').read_text());new=[x for x in pending['pending_objects'] if x['sha256'] not in old];(B/'NEW_SAVE_OBJECTS.json').write_text(json.dumps(new,indent=2)+'\n');(B/'READBACKS.json').write_text(json.dumps(old,indent=2)+'\n');print(json.dumps({'capture_id':out['capture_id'],'new_save_objects':len(new),'new_bytes':sum(x['size_bytes'] for x in new)}))
