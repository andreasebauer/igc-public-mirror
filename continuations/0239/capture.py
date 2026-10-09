"""Capture exact registered complete-class pilot after declared dry gate."""
from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent;Q=B.parent/'continuation0238';sys.path.insert(0,'/tmp/ig_engine0237')
from infinity_grid import submission as sub,portable_registry
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((Q/'NEXT_PREREGISTRATION.json').read_text());a=json.loads((Q/'NEXT_AUDIT.json').read_text());d=json.loads((B/'DRY_RESULT.json').read_text())
assert d['status']=='PASS_FIRST_COMPLETE_CLASS_RESOURCE_DRY' and d['official_cases_executed']==0 and d['peak_RSS_bytes']<=p['memory_budget_bytes']
assert sha(B/'SPEC.json')==a['specification_sha256'] and sha(Q/'NEXT_PREREGISTRATION.json')==a['preregistration_sha256']
for n,h in p['source_sha256'].items():assert sha(Path('/tmp/ig_engine0237/infinity_grid')/n.split('/',1)[1] if n.startswith('engine/') else Q/n)==h
store=Path('/tmp/ig_native0239/store');assert not (store/'coordination/PROJECT.json').exists();portable_registry.initialize(store,'Fixed complete-class historical Q2 pilot',Path('/tmp/ig_engine0237'))
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));s=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(s,indent=2));m=json.loads((B/'READBACKS.json').read_text());new=[x for x in s['pending_objects'] if x['sha256'] not in m];(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
