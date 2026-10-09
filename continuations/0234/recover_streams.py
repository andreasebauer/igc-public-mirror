"""Recover only the saved record/audit bytes needed for source binding review."""
from pathlib import Path
import json,hashlib,shutil
B=Path(__file__).resolve().parent;O=B.parent/'drive_objects';D=B/'data';D.mkdir(exist_ok=True)
for name,h in [('records.jsonl.gz','7d9334008219c159af40a08f661b57b5daea7e449305c92608d33cc9098863d7'),('audits.jsonl.gz','4e7b2c6ec0773b5ae9438c7f706867a4d34b92c5c79f9d419427eb06e842500e')]:
    m=json.loads((B/(h+'.json')).read_text());p=D/name
    with p.open('wb') as w:
        for x in m['parts']:
            q=O/(x['sha256']+'.bin')
            with q.open('rb') as f:assert q.stat().st_size==x['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
            with q.open('rb') as f:shutil.copyfileobj(f,w)
    with p.open('rb') as f:assert p.stat().st_size==m['object']['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==h
shutil.copyfile(O/'35bb6899ab4d1fae4da11f12393658a233f70f3ad7a3dfaad95e617424499d85.bin',B/'HISTORICAL_CONTINUATIONS.json')
print('PASS exact saved record/audit transports recovered')
