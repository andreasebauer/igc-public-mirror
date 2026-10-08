"""Release the completed live copy only after exact terminal preservation.
The independently restored checkpoint retains both working and sealed bytes.
"""
from pathlib import Path
import json,hashlib,shutil
B=Path(__file__).resolve().parent;d=json.loads((B/'NATIVE_RESULT.json').read_text());assert d['status']=='COMPLETED' and d['evidence_status']=='VERIFIED'
assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
e=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());receipt=json.loads((B/'NATIVE_EXPORT_SAVE.json').read_text());assert receipt['raw_drive_readback_verified'] and receipt['sha256']==e['sha256'];z=B/'NATIVE_CHECKPOINT_SLIM.zip';assert hashlib.file_digest(z.open('rb'),'sha256').hexdigest()==e['sha256']
m=json.loads((B/'READBACKS.json').read_text());j=Path(json.loads((B/'POINTER.json').read_text())['workspace']);assert j.is_relative_to(Path('/tmp/ig_native0227/store/captures'))
for r in e['dependencies']:
 x=m[r['sha256']];p=Path(x['path']);assert x['raw_readback_verified'] and not p.is_relative_to(j);assert p.stat().st_size==r['size_bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==r['sha256']
assert not Path('/tmp/ig_native0227/cold').exists()
n=sum(p.stat().st_size for p in j.rglob('*') if p.is_file());shutil.rmtree(j)
(B/'LIVE_CACHE_RELEASE.json').write_text(json.dumps({'status':'PASS_EXACT_SAVED_TERMINAL_LIVE_COPY_RELEASED_BEFORE_COLD','released_bytes':n,'completion_sha256':d['completion_sha256'],'checkpoint_sha256':e['sha256'],'dependencies_verified':len(e['dependencies']),'raw_readbacks_retained':True,'cold_restore_pending':True},indent=2));print('Released verified live duplicate; free bytes',shutil.disk_usage('/tmp').free)
