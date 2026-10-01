from pathlib import Path
import sys,json,hashlib
D=Path(__file__).resolve().parent
w=Path('/tmp/ig_qualification_v84_20261001/functional/store/captures/853b210f67a06c50ec51c262a72eae13918090e9e1246463a15327b074e88731')
sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
s=pr.status(w)
(D/'STATUS_BEFORE.json').write_text(json.dumps(s,indent=2))
e=pr.export_checkpoint(w,D/'V84_PAUSED_COMPLETE.zip',slim=False)
(D/'EXPORT.json').write_text(json.dumps(e,indent=2))
parts=[]
with (D/'V84_PAUSED_COMPLETE.zip').open('rb') as f:
 while raw:=f.read(24*1024*1024):
  p=D/f'V84_PAUSED_PART_{len(parts)+1:02d}.bin';p.write_bytes(raw)
  parts.append({'path':str(p),'name':p.name,'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)})
(D/'PARTS.json').write_text(json.dumps(parts,indent=2))
print(json.dumps({'export_sha256':e['sha256'],'size_bytes':(D/'V84_PAUSED_COMPLETE.zip').stat().st_size,'parts':len(parts)}))
