"""Restore exact already-downloaded runtime/source dependencies; no scientific execution."""
from pathlib import Path
import hashlib,json,shutil,tarfile,zipfile
B=Path(__file__).resolve().parent;W=B.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    transport=json.loads((W/'continuation0281/runtime/RECOVERY_RUNTIME_PARTS.json').read_text())
    archive=B/'runtime_recovery.zip'
    with archive.open('wb') as out:
        for row in transport['parts']:
            p=B/(row['sha256']+'.bin');assert sha(p)==row['sha256'] and p.stat().st_size==row['size_bytes']
            with p.open('rb') as f:shutil.copyfileobj(f,out)
    assert sha(archive)==transport['archive']['sha256']
    with zipfile.ZipFile(archive) as z:
        raw=z.read(transport['extract_runtime']);assert hashlib.sha256(raw).hexdigest()=='25f76fa883c03b2ccc8a902d0aca799abd167279dbc6108ee2dad8095ac90222'
        tar=B/'runtime.tar.gz';tar.write_bytes(raw);dest=Path('/tmp/ig_admit0188/runtime_recovery');dest.mkdir(parents=True,exist_ok=True)
        with tarfile.open(tar) as t:t.extractall(dest,filter='data')
        root=dest/'ig_runtime_v55_fresh_20260930';manifest=json.loads((W/'continuation0231/RUNTIME_MANIFEST.json').read_text())
        for name,row in manifest['runtime_files'].items():
            p=root/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        for row in manifest['host'].values():
            p=root/'host_bindings'/row['bundled_name'];p.parent.mkdir(exist_ok=True);p.write_bytes(z.read('RECOVERY/bootstrap/runtime/host_bindings/'+row['bundled_name']));assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    policy=json.loads((W/'continuation0281/runtime/RUNTIME_POLICY.json').read_text())
    for p in root.rglob('*'):
        if p.is_file() and not p.is_symlink():p.chmod(0o755 if str(p.relative_to(root)) in policy['executables'] else 0o644)
    source=W/'drive_objects/0f8d237d598a7e751c16fb032324e467a983ea5a81657a77578370a161913e33.bin'
    assert sha(source)==source.stem;E=Path('/tmp/ig_engine0237');E.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            p=Path(name);assert not p.is_absolute() and '..' not in p.parts
            if p.parts and p.parts[0]!='project':z.extract(name,E)
    spec=json.loads((W/'continuation0285/specs/000.json').read_text())
    for row in spec['inputs']:
        p=Path(row['path'])
        if not p.exists():shutil.copyfile(W/'drive_objects'/(row['sha256']+'.bin'),p)
        assert sha(p)==row['sha256']
    for row in spec['environment']['artifacts']:
        p=Path(row['path']);p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/(row['sha256']+'.bin'),p);assert sha(p)==row['sha256']
    print('PASS exact runtime, seven host bindings, engine, inputs and environment objects')
if __name__=='__main__':main()
