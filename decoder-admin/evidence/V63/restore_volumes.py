import json,hashlib,zipfile,sys
from pathlib import Path
from lifecycle_monitor import inventory
D=Path(__file__).resolve().parent;m=json.loads((D/'FORENSIC_VOLUMES.json').read_text());rows=json.loads((D/'FORENSIC_VOLUME_READBACKS.json').read_text());dest=Path('/tmp/ig_v63_forensic_restored_20261001')
if dest.exists():raise RuntimeError('NEW_DESTINATION_REQUIRED')
seen=set()
for r in rows:
 b=Path(r['readback']).read_bytes();assert len(b)==r['size_bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 with zipfile.ZipFile(r['readback']) as z:
  for n in z.namelist():
   p=Path(n);assert n not in seen and not p.is_absolute() and '..' not in p.parts and n in m['files'];seen.add(n)
   b=z.read(n);f=m['files'][n];assert len(b)==f['size'] and hashlib.sha256(b).hexdigest()==f['sha256']
assert seen==set(m['files']);dest.mkdir()
for r in rows:
 with zipfile.ZipFile(r['readback']) as z:
  for n in z.namelist():
   p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n));p.chmod(m['files'][n]['mode'])
assert inventory(dest)==m['files']
sys.path.insert(0,str(dest/'source'));from infinity_grid import preservation as pr,submission as sub
expected=json.loads((D/'INJECTION_DONE.json').read_text())['original_sha256']
try:pr.status(dest)
except sub.SubmissionError as e:
 assert e.code=='OUTBOX_OBJECT_MISMATCH' and expected in str(e);reason=str(e)
else:raise RuntimeError('EXPECTED_REFUSAL_MISSING')
assert inventory(dest)==m['files']
r={'exact_files_and_modes':len(seen),'remote_volumes_verified':len(rows),'native_refusal':reason,'workload_executions':0,'original_workspace_repaired':False}
(D/'FORENSIC_RESTORE_VERIFICATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
