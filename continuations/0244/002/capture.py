"""Capture unchanged registered remaining Q2 chunk002 after earned dry gate."""
from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent;Q=Path('/workspace/scratch/2a87972b51af/continuation0240');E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid import submission as sub,portable_registry
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pre=json.loads((Q/'PREREGISTRATION.json').read_text());audit=json.loads((Q/'AUDIT.json').read_text());dry=json.loads((Path('/workspace/scratch/2a87972b51af')/'continuation0241/COLD_AUDIT.json').read_text());saved=json.loads((Path('/workspace/scratch/2a87972b51af')/'continuation0241/SAVE_RECEIPT.json').read_text())
assert sha(Q/'PREREGISTRATION.json')==audit['preregistration_sha256']
assert sha(B/'SPEC.json')==pre['specifications'][2]['sha256']
assert dry['cases_exact']==2 and dry['peak_RSS_bytes']<=pre['memory_budget_bytes'] and dry['earned_coverage_credit']==0
assert all(v['drive_exact_readback_verified'] and v['local_metadata_applied'] for v in saved['artifacts'].values())
for n,h in pre['source_sha256'].items():assert sha(E/'infinity_grid'/n.split('/',1)[1] if n.startswith('engine/') else Q/n)==h
prior=json.loads((Path('/workspace/scratch/2a87972b51af')/'continuation0243/EARNED_LEDGER.json').read_text());assert prior['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED' and prior['next_chunk_index']==2 and prior['plan_sha256']==sha(Q/'PLAN.json')
store=Path('/tmp/ig_native0244_002/store');assert not (store/'coordination/PROJECT.json').exists();portable_registry.initialize(store,'Registered remaining whole-class Q2 chunk002',E)
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));s=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(s,indent=2));m=json.loads((B/'READBACKS.json').read_text());new=[x for x in s['pending_objects'] if x['sha256'] not in m];(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
