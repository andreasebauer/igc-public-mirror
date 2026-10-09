"""Capture the declared two-member native adapter/resource/resume validation."""
from pathlib import Path
import json,hashlib,sys,shutil,tempfile
B=Path(__file__).resolve().parent;E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid import submission as sub,portable_registry
from infinity_grid.workflow_guard import preflight
from infinity_grid.result_contracts import normalize
from infinity_grid.canon import canonical_sha256
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((B/'PREREGISTRATION.json').read_text());s=json.loads((B/'SPEC.json').read_text())
assert sha(B/'SPEC.json')==p['specification_sha256'] and p['source_binding']==canonical_sha256(p['source_sha256'])
for n,h in p['source_sha256'].items():assert sha(E/'infinity_grid'/n.split('/',1)[1] if n.startswith('engine/') else B/n)==h
assert normalize(s['output_contract'],s['execution'],s['question'])['claim']=='EXECUTION_ONLY'
with tempfile.TemporaryDirectory(prefix='ig_preflight0241_') as t:
 root=Path(t);shutil.copytree(E/'infinity_grid',root/'infinity_grid');shutil.copytree(B/'project',root/'project');g=preflight(root,[root/'project/handler.py',root/'project/worker.py'])
assert g['status']=='PASS' and len(g['modules'])==22;(B/'PREFLIGHT.json').write_text(json.dumps(g,indent=2))
store=Path('/tmp/ig_native0241/store');assert not (store/'coordination/PROJECT.json').exists();portable_registry.initialize(store,'Two-member Q2 resource and resume validation',E)
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2));status=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(status,indent=2));m=json.loads((B/'READBACKS.json').read_text());new=[x for x in status['pending_objects'] if x['sha256'] not in m];(B/'NEW_OBJECTS.json').write_text(json.dumps(new,indent=2));print(json.dumps({'capture_id':r['capture_id'],'new_objects':len(new),'bytes':sum(x['size_bytes'] for x in new)}))
