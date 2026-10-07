from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import submission as sub,portable_registry
store=Path('/tmp/ig_native0206/store')
assert json.loads((B/'GATE.json').read_text())['status']=='PASS_INDEPENDENT_SAVED94_STATE_RECONSTRUCTION_AND_EXACT_DAG_ROUNDTRIP'
assert json.loads((B/'PREFLIGHT.json').read_text())['status']=='PASS'
portable_registry.initialize(store,'Bounded G1 continuation95-96 from verified saved depth94',Path('/tmp/ig_engine0204'))
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));p=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(p,indent=2));m=json.loads((B.parent/'recovery/0205/partition0205/READBACKS.json').read_text());(B/'READBACKS.json').write_text(json.dumps(m,indent=2));new=[x for x in p['pending_objects'] if x['sha256'] not in m];(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
