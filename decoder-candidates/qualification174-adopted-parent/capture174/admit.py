from pathlib import Path
import sys,json
h=Path(__file__).resolve().parent;p=json.loads((h/'results/CAPTURE_PENDING.json').read_text());sys.path.insert(0,p['workspace']+'/source')
from infinity_grid import submission as sub
proofs={r['sha256']:r for r in json.loads((h/'results/CAPTURE_PROVIDER_READBACKS.json').read_text())}
for o in p['obligations']:
 r=proofs[o['sha256']];sub.confirm_save(p['workspace'],o['sha256'],r['path'],r['drive_file_id'],role=o['role'],logical_name=o['logical_name'])
status=sub.save_status(p['workspace']);assert status['status']=='SAVED';(h/'results/CAPTURE_SAVED.json').write_text(json.dumps(status,indent=2));print(status['status'])
