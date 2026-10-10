"""Native dry capture of frozen observer chunk000; zero earned coverage credit."""
from pathlib import Path
import json,hashlib,sys,shutil
B=Path(__file__).resolve().parent;Q=B.parent/'continuation0285';E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid import submission as sub,portable_registry
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reg=json.loads((Q/'PREREGISTRATION.json').read_text())
for n,h in reg['source_sha256'].items():assert sha(Q/n)==h
for n,h in reg['required_engine_files'].items():assert sha(E/'infinity_grid'/n.split('/',1)[1])==h
assert sha(Q/'specs/000.json')==reg['specifications'][0]['spec_sha256']
assert shutil.disk_usage(B).free>=reg['minimum_free_disk_bytes']
store=Path('/tmp/ig_native0286_dry/store');assert not (store/'coordination/PROJECT.json').exists()
portable_registry.initialize(store,'Non-promoting historical observer dry gate chunk000',E)
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2)+'\n')
s=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({'capture_id':r['capture_id'],'pending_objects':len(s['pending_objects']),'pending_bytes':sum(x['size_bytes'] for x in s['pending_objects'])}))
