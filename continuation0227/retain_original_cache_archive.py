"""Losslessly archive original 0226 working bytes before releasing its cache."""
from pathlib import Path
import zipfile,hashlib,json
B=Path(__file__).resolve().parent;p=Path('/tmp/ig_native0226/retained_working_runs');assert p.is_dir()
out=Path('/tmp/ig_verified0227/ORIGINAL0226_WORKING_CACHE.zip');out.parent.mkdir(exist_ok=True)
m={}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(p.rglob('*')):
  if f.is_file():
   assert not f.is_symlink();n=str(f.relative_to(p));m[n]={'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest(),'bytes':f.stat().st_size};z.write(f,n)
 z.writestr('ORIGINAL_INVENTORY.json',json.dumps(m,indent=2))
with zipfile.ZipFile(out) as z:
 for n,x in m.items():
  with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
x={'status':'PASS_EXACT_ORIGINAL_WORKING_CACHE_ARCHIVED','path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'size_bytes':out.stat().st_size,'raw_original_bytes':sum(x['bytes'] for x in m.values()),'files':len(m),'source_path':str(p),'restores_original_sqlite_headers_and_rows':True};assert x['size_bytes']<40*1024*1024;(B/'ORIGINAL0226_CACHE_ARCHIVE.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
