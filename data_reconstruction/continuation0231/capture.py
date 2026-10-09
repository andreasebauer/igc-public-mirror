from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import submission as sub,portable_registry
assert json.loads((B/'SOURCE_ATTESTATION.json').read_text())['status']=='PASS_HISTORICAL_V1_RESTORATION_SOURCE_BINDING';assert json.loads((B/'PREFLIGHT.json').read_text())['status']=='PASS'
store=Path('/tmp/ig_native0231/store');
if not (store/'coordination/PROJECT.json').exists():portable_registry.initialize(store,'Historical scientific helper seed/depth14qualification',Path('/tmp/ig_engine0204'))
else:assert not any((store/'captures').glob('*'))
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));p=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(p,indent=2));m=json.loads((B/'READBACKS.json').read_text());(B/'READBACKS.json').write_text(json.dumps(m,indent=2));new=[x for x in p['pending_objects'] if x['sha256'] not in m];(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
