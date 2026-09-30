"""Resolve exact observed checkpoint to genuinely saved original bytes; no mutation."""
import json,sys,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent
if sys.flags.optimize or not sys.flags.dont_write_bytecode:raise RuntimeError('PYTHON_MODE')
if not (D/'BASELINE_READY.json').exists():sys.exit(76)
c=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(c['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
s=pr.status(w)
if s['pending_objects']:sys.exit(76)
b=json.loads((D/'BASELINE_READY.json').read_text());checkpoint=b['after_completed_checkpoint']['sha256']
commit=json.loads((pr._root(w)/'commits'/(checkpoint+'.json')).read_text());sha=commit['state']['sha256']
rows=[r for r in json.loads((D/'CATALOG.json').read_text()) if r['sha256']==sha]
if not rows:raise RuntimeError('ORIGINAL_READBACK_MISSING')
r=rows[0];assert hashlib.sha256(Path(r['readback']).read_bytes()).hexdigest()==sha
config={'armed':True,'job_id':c['job_id'],'workspace':str(w),'capture_id':c['capture_id'],'external_evidence':str(D),'observer_baseline_ready':str(D/'BASELINE_READY.json'),'observed_checkpoint_sha256':checkpoint,'checkpoint_state_sha256':sha,'original_remote_readback':r['readback'],'original_drive_id':r['drive_id'],'remote_readback_verified':True}
with (D/'INJECTION_CONFIG.json').open('x') as f:json.dump(config,f,indent=2)
print(json.dumps({'checkpoint':checkpoint,'state':sha}))
