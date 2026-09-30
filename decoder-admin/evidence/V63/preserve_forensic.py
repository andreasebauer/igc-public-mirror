from pathlib import Path
import json,time
from lifecycle_monitor import inventory,process,identity
D=Path(__file__).resolve().parent;c=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(c['workspace'])
l=json.loads((D/'LIFECYCLE_OBSERVATIONS.json').read_text())
for old in l['known']:
 p=process(old['pid'])
 if p and identity(p)==identity(old):raise RuntimeError('REGISTERED_PROCESS_STILL_EXISTS')
a=inventory(w);start=time.monotonic();time.sleep(3.1);b=inventory(w)
assert a==b
for old in l['known']:
 p=process(old['pid'])
 if p and identity(p)==identity(old):raise RuntimeError('REGISTERED_PROCESS_REAPPEARED')
q={'live_registered_processes':[],'stable_seconds':time.monotonic()-start,'inventory_before':a,'inventory_after':b,'scope':'full workspace after observed registered-process quiescence'}
with (D/'FORENSIC_QUIESCENCE.json').open('x') as f:json.dump(q,f,indent=2)
print(json.dumps({'files':len(b),'stable_seconds':q['stable_seconds']}))
