"""Read native committed phase counts without touching candidate generation."""
from pathlib import Path
import json,sqlite3
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);rows=[]
for f in sorted(J.glob('runtime/runs/*/chain/decoder_stage_runtime/*/phases/*/state_store.sqlite3')):
 with sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True,timeout=1) as c:
  tasks=c.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0];states=c.execute('SELECT COUNT(*) FROM states').fetchone()[0]
 rows.append({'phase':f.parent.name,'committed_tasks':tasks,'states':states})
print(json.dumps(rows))
