from pathlib import Path
import json,sys,hashlib
h=Path(__file__).resolve().parent; cold=json.loads((h/'results/FINAL_COLD.json').read_text());w=Path(cold['destination']);p=json.loads((h/'results/CAPTURE_PENDING.json').read_text())
sys.path.insert(0,str(w/'source'))
from infinity_grid import v05_controller_event_loop as loop
paths=list((w/'runtime/attempts').rglob('*.json'))+list((w/'runtime/intake/completed').glob('*.json'))
before={str(f.relative_to(w)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths};count=len(list((w/'runtime/attempts').rglob('*.json')))
r=loop.run_workspace_job(w,p['capture']['job_id'])
after={str(f.relative_to(w)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
assert r['reused'] and before==after and len(list((w/'runtime/attempts').rglob('*.json')))==count==1
(h/'results/COLD_REUSE.json').write_text(json.dumps({'status':'PASS','completion_reused':True,'attempt_count':count,'prior_attempt_completion_bytes_exact':True},indent=2))
print('COLD_COMPLETION_REUSE_PASS')
