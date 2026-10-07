"""Recover exactly pinned userspace runtime and captured engine, without science."""
from pathlib import Path
import json,zipfile,tarfile,hashlib,os
B=Path(__file__).resolve().parent;W=B.parent
D=Path('/tmp/ig_verified0205/recovery')
archive=D/'runtime_handoff.zip'
spec=json.loads((W/'handoff0188/runtime/RECOVERY_RUNTIME_PARTS.json').read_text())
assert archive.stat().st_size==spec['archive']['size_bytes']
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==spec['archive']['sha256']
runtime_tar=D/'runtime.tar.gz'
with zipfile.ZipFile(archive) as z:
    runtime_tar.write_bytes(z.read(spec['extract_runtime']))
assert hashlib.file_digest(runtime_tar.open('rb'),'sha256').hexdigest()=='25f76fa883c03b2ccc8a902d0aca799abd167279dbc6108ee2dad8095ac90222'
root=Path('/tmp/ig_admit0188/runtime_recovery');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(runtime_tar) as t:
    t.extractall(root,filter='data')
runtime=root/'ig_runtime_v55_fresh_20260930'
manifest=json.loads((W/'handoff0188/runtime/RUNTIME_MANIFEST.json').read_text())
checked=0
for name,row in manifest['runtime_files'].items():
    p=runtime/name
    assert p.stat().st_size==row['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==row['sha256']
    os.chmod(p,0o644);checked+=1
policy=json.loads((W/'handoff0188/runtime/RUNTIME_POLICY.json').read_text())
for name in policy['executables']:os.chmod(runtime/name,0o755)
rows=json.loads((B/'RECOVERY_READBACKS.json').read_text())
engine_blob=next(x for x in rows if x['sha256']=='60c729a8ee3ec81b0b367b44719e238271ad7d0ae9112cebe81253cf3f874c30')
engine=Path('/tmp/ig_engine0204');engine.mkdir()
with zipfile.ZipFile(engine_blob['path']) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(engine)
change=json.loads((W/'partition0204/ENGINE_DIFF.json').read_text())
assert hashlib.file_digest((engine/'infinity_grid/v05_stage_runtime.py').open('rb'),'sha256').hexdigest()==change['new_file_sha256']
result={'status':'PASS_PINNED_RUNTIME_AND_PATCHED_ENGINE_BYTE_RECOVERY','runtime_files_checked':checked,'engine_archive_sha256':engine_blob['sha256'],'runtime_archive_sha256':hashlib.file_digest(runtime_tar.open('rb'),'sha256').hexdigest(),'generator_calls':0,'runtime_root':str(runtime)}
(B/'RUNTIME_RECOVERY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
