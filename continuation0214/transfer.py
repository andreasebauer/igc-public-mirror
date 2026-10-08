"""Operator raw-readback transport; every object is verified before native ack."""
from pathlib import Path
import json,urllib.request,hashlib,sys
B=Path(__file__).resolve().parent;mode=sys.argv[1];manifest=B/'private_downloads.json';m=json.loads((B/'READBACKS.json').read_text());dest=Path('/tmp/ig_verified0214');dest.mkdir(exist_ok=True)
for x in json.loads(manifest.read_text()):
 p=dest/(x['sha256']+'.bin');h=hashlib.sha256();n=0
 with urllib.request.urlopen(urllib.request.Request(x['url'],headers={'User-Agent':'Mozilla/5.0','Accept':'*/*'})) as r,p.open('wb') as w:
  while chunk:=r.read(1048576):w.write(chunk);h.update(chunk);n+=len(chunk)
 assert h.hexdigest()==x['sha256'] and n==x['size_bytes'];m[x['sha256']]={'id':x['id'],'path':str(p),'bytes':n,'raw_readback_verified':True}
(B/'READBACKS.json').write_text(json.dumps(m,indent=2));manifest.unlink()
if mode=='checkpoint':
 p=json.loads((B/'DRAIN_PLAN.json').read_text());q={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':p['capture_id'],'obligations':p['obligations'],'readbacks':[{'sha256':x['sha256'],'kind':'RAW','path':m[x['sha256']]['path'],'drive_file_id':m[x['sha256']]['id']} for x in p['physical_objects']]};(B/'ACK_BATCH.json').write_text(json.dumps(q,indent=2))
print('PASS exact raw readbacks',mode)
