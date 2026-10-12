from pathlib import Path
import gzip,json,hashlib,zipfile
root=Path(__file__).resolve().parent
ledger=json.loads((root/'inputs/CHUNK_LEDGER.json').read_text())
rows=[]
for i in [0,10,20,30,39]:
 r=ledger[i];p=root/'inputs/chunks'/r['file'];raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==r['sha256']
 lines=gzip.decompress(raw).splitlines()
 assert hashlib.sha256(gzip.decompress(raw)).hexdigest()==r['raw_stream_sha256']
 for line in lines[:64]:
  v=json.loads(line);rows.append({'ordinal':v[0],'edge':v[4]})
(root/'results/sample.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
print(len(rows),rows[0],rows[-1])
