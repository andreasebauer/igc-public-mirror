"""Capture unchanged scientific recipe after saved preregistration and dry gates."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;Q=B.parent/'continuation0235';sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import submission as sub,portable_registry
p=json.loads((Q/'PREREGISTRATION.json').read_text());a=json.loads((Q/'AUDIT.json').read_text());assert a['dry_status']=='PASS_ONE_CASE_DRY_PREFLIGHT' and a['official_cases_executed']==0
assert hashlib.sha256((Q/'PREREGISTRATION.json').read_bytes()).hexdigest()==a['preregistration_sha256']
for n,h in p['source_sha256'].items():
 path=Path('/tmp/ig_engine0204/infinity_grid/uplift_structural.py') if n.startswith('engine/') else Q/n
 assert hashlib.sha256(path.read_bytes()).hexdigest()==h
store=Path('/tmp/ig_native0236/store');assert not (store/'coordination/PROJECT.json').exists()
portable_registry.initialize(store,'Registered bounded historical S1 Q2 realization',Path('/tmp/ig_engine0204'))
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));s=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(s,indent=2));m=json.loads((B/'READBACKS.json').read_text());new=[x for x in s['pending_objects'] if x['sha256'] not in m];(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
