"""Recover finite saved public evidence; no producer imports or constructions."""
from pathlib import Path
import ast,hashlib,io,json,zipfile
R=Path.cwd();B=R/'g_closure0176';I=B/'inputs';I.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
p=next((B/'recovered').rglob('*S0_S1*.zip'));assert sha(p.read_bytes())=='9e7eb190fe63336e46c69b9e076e859f1679f73aad18bad35c2b817b7e66ff84'
with zipfile.ZipFile(p) as z:
 for n in ['audit/AUDIT_RESULT.json','audit/AUDIT_REPORT.txt','audit/AUDIT_FINDINGS.json']:(I/Path(n).name).write_bytes(z.read(n))
 innername=next(n for n in z.namelist() if n.startswith('science_original/') and n.endswith('.zip'))
 raw=z.read(innername)
 with zipfile.ZipFile(io.BytesIO(raw)) as inner:
  m=json.loads(inner.read('science/MANIFEST_SHA256.json'))
  for f in m['files']:
   b=inner.read('science/'+f['path']);assert sha(b)==f['sha256'] and len(b)==f['bytes']
   if not f['path'].endswith('.gz'):(I/Path(f['path']).name).write_bytes(b)
  (I/'SAVED_SCIENCE_MANIFEST.json').write_bytes(inner.read('science/MANIFEST_SHA256.json'))
 q=next(n for n in z.namelist() if n.startswith('source/') and n.endswith('.zip'))
 with zipfile.ZipFile(io.BytesIO(z.read(q))) as source:
  for suffix in ['maturation_parallel.py','uplift_structural.py','O_REGIME_MOTIF_LIBRARY_v1.json']:
   n=next(n for n in source.namelist() if n.endswith('/'+suffix));(I/('historical_'+suffix)).write_bytes(source.read(n))
p=next((B/'recovered').rglob('*REALIZATION*.zip'));assert sha(p.read_bytes())=='0df5119b53880d800627b52904b3c6978392f799ab049e1f18f8d70d2f83f3b5'
with zipfile.ZipFile(p) as z:
 for n in ['RESULT.json','PREREGISTRATION.json','evidence/PROJECTED_COLLISION_INDEX.json']:(I/('REALIZATION_'+Path(n).name)).write_bytes(z.read(n))
for f in ['maturation_parallel.py','uplift_structural.py','regime_scanner.py']:(I/f).write_bytes((R/'engine/infinity_grid'/f).read_bytes())
(I/'O_REGIME_MOTIF_LIBRARY_v1.json').write_bytes((R/'engine/infinity_grid/resources/decoder/O_REGIME_MOTIF_LIBRARY_v1.json').read_bytes())
with zipfile.ZipFile(R/'g_readiness0175/inputs/G1_saved.zip') as z:(I/'R100_SOURCE_INPUT.json').write_bytes(z.read('run/checkpoints/O00100/SOURCE_INPUT.json'))
dump(B/'INPUT_PINS.json',{p.name:{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in I.iterdir() if p.is_file()})
print('Saved8 science manifest members checked;public interface and later failed realization evidence recovered')
