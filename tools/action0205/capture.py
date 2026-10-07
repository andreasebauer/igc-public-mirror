import sys,json
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import submission as sub,portable_registry
store=Path('/tmp/ig_native0205/store')
if not (store/'coordination/PROJECT.json').exists():portable_registry.initialize(store,'Bounded G1 continuation91-96 from exact depth90',Path('/tmp/ig_engine0204'))
assert json.loads((B/'TESTS.json').read_text())['status'].startswith('PASS')
assert json.loads((B/'SPEC.json').read_text())['engine_source']=='/tmp/ig_engine0204'
assert '5 passed' in (B/'BUDGET_TESTS.log').read_text()
m=json.loads((B.parent/'partition0204/READBACKS.json').read_text())
for x in json.loads((B/'RECOVERY_READBACKS.json').read_text()):
 if x['kind']=='checkpoint':m[x['sha256']]={'id':x['id'],'path':x['path'],'bytes':x['size_bytes'],'raw_readback_verified':True}
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));p=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(p,indent=2));new=[x for x in p['pending_objects'] if x['sha256'] not in m];(B/'READBACKS.json').write_text(json.dumps(m,indent=2));(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
