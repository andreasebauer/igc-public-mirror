"""Restore this handoff's native checkpoint from hash-verified raw objects."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent
if len(sys.argv)!=4:raise SystemExit('Usage: python-fixed-host restore_current.py ENGINE_ROOT RAW_OBJECTS_DIRECTORY FRESH_DESTINATION')
engine=Path(sys.argv[1]);objects=Path(sys.argv[2]);dest=Path(sys.argv[3])
assert dest.is_absolute() and dest.is_relative_to(Path('/tmp')) and not dest.exists()
sys.path.insert(0,str(engine))
from infinity_grid import preservation as pr
e=json.loads((B/'CHECKPOINT_EXPORT.json').read_text())
archive=B/'NATIVE_CHECKPOINT_SLIM.zip'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==e['sha256']
for row in e['dependencies']:
    raw=objects/(row['sha256']+'.bin')
    assert raw.stat().st_size==row['size_bytes'] and hashlib.file_digest(raw.open('rb'),'sha256').hexdigest()==row['sha256']
result=pr.restore_checkpoint(archive,dest,e['sha256'],objects=objects)
print(json.dumps({'status':'PASS_CURRENT_CHECKPOINT_RESTORED','native_restore':result,'handler_executed':False,'generator_calls':0}))
