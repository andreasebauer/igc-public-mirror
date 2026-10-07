import sys,json
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine')
from infinity_grid import submission as sub,portable_registry
store=Path('/tmp/ig_native0200/store')
if not (store/'coordination/PROJECT.json').exists():portable_registry.initialize(store,'Bounded G1 continuation67-72 from exact depth66',Path('/tmp/ig_admit0188/engine'))
assert json.loads((B/'TESTS.json').read_text())['status'].startswith('PASS')
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));p=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(p,indent=2));m=json.loads((B.parent/'partition0199/READBACKS.json').read_text());new=[x for x in p['pending_objects'] if x['sha256'] not in m];(B/'READBACKS.json').write_text(json.dumps(m,indent=2));(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
