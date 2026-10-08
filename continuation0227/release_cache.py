"""Release reproducible completed workspace copies after dependency verification."""
from pathlib import Path
import json,hashlib,shutil
B=Path(__file__).resolve().parent;W=B.parent;all_refs=set()
for p in W.glob('continuation*/READBACKS.json'):
 all_refs.update(Path(x['path']).resolve() for x in json.loads(p.read_text()).values())
released=json.loads((B/'CACHE_RELEASE.json').read_text())['released'] if (B/'CACHE_RELEASE.json').exists() else []
for n in (225,226):
 p=W/f'continuation0{n}';r=json.loads((p/'SAVE_RECEIPT.json').read_text());z=Path(r['path']);assert r['raw_drive_readback_verified'] and r['library_save_completed'];assert z.stat().st_size==r['bytes'] and hashlib.file_digest(z.open('rb'),'sha256').hexdigest()==r['sha256'];assert json.loads((p/'AUDIT.json').read_text())['native_registered_scope_completed']
 e=json.loads((p/'CHECKPOINT_EXPORT.json').read_text());m=json.loads((p/'READBACKS.json').read_text())
 for x in e['dependencies']:
  f=Path(m[x['sha256']]['path']);assert m[x['sha256']]['raw_readback_verified'] and f.stat().st_size==x['size_bytes'] and hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==x['sha256']
 j=Path(json.loads((p/'POINTER.json').read_text())['workspace']);assert j.is_relative_to(Path(f'/tmp/ig_native0{n}/store/captures')) and not any(f.is_relative_to(j) for f in all_refs)
 if not j.exists():continue
 size=sum(f.stat().st_size for f in j.rglob('*') if f.is_file());released.append({'path':str(j),'bytes':size,'saved_handoff_sha256':r['sha256'],'checkpoint_dependencies_verified':len(e['dependencies'])});shutil.rmtree(j)
for j in []: # Retain current 0227 preflight until capture admitted

 assert not any(f.is_relative_to(j) for f in all_refs);size=sum(f.stat().st_size for f in j.rglob('*') if f.is_file());released.append({'path':str(j),'bytes':size,'kind':'reproducible_preflight_copy'});shutil.rmtree(j)
(B/'CACHE_RELEASE.json').write_text(json.dumps({'status':'PASS_VERIFIED_RECOVERABLE_CACHE_RELEASE','released':released,'raw_readbacks_preserved':True,'failed_native_originals_preserved':True},indent=2));print('Released bytes',sum(x['bytes'] for x in released))

a=json.loads((B/'ORIGINAL0226_CACHE_ARCHIVE.json').read_text());assert a['raw_drive_readback_verified'];f=Path(a['raw_readback_path']);assert f.stat().st_size==a['size_bytes'] and hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==a['sha256'];j=Path(a['source_path']);assert j==Path('/tmp/ig_native0226/retained_working_runs') and not any(f.is_relative_to(j) for f in all_refs);shutil.rmtree(j);a['original_directory_released_after_exact_archive_readback']=True;(B/'ORIGINAL0226_CACHE_ARCHIVE.json').write_text(json.dumps(a,indent=2));print('Original working bytes preserved in verified lossless archive; free bytes',shutil.disk_usage('/tmp').free)
