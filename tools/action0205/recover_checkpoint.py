"""Restore the preserved predecessor; do not reexecute its handler."""
from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent;W=B.parent
sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as pr
export=json.loads((W/'partition0204/CHECKPOINT_EXPORT.json').read_text())
archive=W/'partition0204/NATIVE_CHECKPOINT_SLIM.zip'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==export['sha256']
objects=Path('/tmp/ig_verified0205/recovery')
for row in export['dependencies']:
    p=objects/(row['sha256']+'.bin')
    assert p.stat().st_size==row['size_bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==row['sha256']
destination=Path('/tmp/ig_native0204/cold_checkpoint')
result=pr.restore_checkpoint(archive,destination,export['sha256'],objects=objects)
(B/'PREDECESSOR_RESTORED.json').write_text(json.dumps({'native_restore':result,'generator_calls':0,'handler_executed':False},indent=2));print('PASS native checkpoint0204 restored; no generation')
