"""Recover recorded raw dependencies; fail closed on every digest and length."""
import json,hashlib,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parent
D=Path('/tmp/ig_verified0205/recovery')
manifest=D/'private_downloads.json'
rows=json.loads(manifest.read_text())
def get(row):
    dest=D/(row['sha256']+'.bin')
    h=hashlib.sha256();count=0
    with urllib.request.urlopen(urllib.request.Request(row['url'],headers={'User-Agent':'Mozilla/5.0','Accept':'*/*'})) as source,dest.open('wb') as target:
        while chunk:=source.read(1048576):
            target.write(chunk);h.update(chunk);count+=len(chunk)
    assert h.hexdigest()==row['sha256'] and count==row['size_bytes']
    return {k:v for k,v in dict(row,path=str(dest),raw_readback_verified=True).items() if k!='url'}
with ThreadPoolExecutor(max_workers=4) as pool:
    saved=list(pool.map(get,rows))
(B/'RECOVERY_READBACKS.json').write_text(json.dumps(saved,indent=2))
manifest.unlink()
print(json.dumps({'status':'PASS_EXACT_RAW_RECOVERY','objects':len(saved),'bytes':sum(x['size_bytes'] for x in saved)}))
