from pathlib import Path
import json,zipfile,hashlib
D=Path(__file__).resolve().parent
w=Path('/tmp/ig_qualification_v84_20261001/functional/store/captures/853b210f67a06c50ec51c262a72eae13918090e9e1246463a15327b074e88731')
s=json.loads((D/'STATUS_BEFORE.json').read_text());e=json.loads((D/'EXPORT.json').read_text());present={o['sha256'] for o in e['dependencies']}
extra={o['sha256']:o for o in s['pending_objects'] if o['sha256'] not in present}
p=D/'V84_PENDING_AUDIT.zip'
with zipfile.ZipFile(p,'x',zipfile.ZIP_DEFLATED) as z:
 for f in (w/'durability/outbox').rglob('*.json'):z.write(f,str(f.relative_to(w)))
 for h,r in extra.items():
  f=Path(r['local_object_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;z.write(f,'extra_objects/'+h+'.bin')
 z.write(D/'STATUS_BEFORE.json','STATUS_BEFORE.json')
print(json.dumps({'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'extra_objects':len(extra)}))
