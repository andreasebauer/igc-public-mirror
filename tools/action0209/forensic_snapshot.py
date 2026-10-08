"""Byte-preserving forensic archive after frozen native checkpoint resource refusal."""
from pathlib import Path
import json,hashlib,zipfile
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
assert (B/'ERROR.txt').exists() and 'CHECKPOINT_STATE_RAW_LIMIT' in (B/'ERROR.txt').read_text()
files={str(p.relative_to(J)):p for p in J.rglob('*') if p.is_file() and (p.is_relative_to(J/'runtime') or (p.parent==J and p.suffix in ('.json','.jsonl','.txt')))}
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()};out=Path('/tmp/ig_verified0209/FAILED_NATIVE0209_FORENSIC_RAW.zip');out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('FORENSIC_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:assert z.testzip() is None
meta={'schema_id':'IG_FORENSIC_NATIVE_RUNTIME_ARCHIVE_V1','path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'capture_id':J.name,'native_registered_scope_completed':False,'native_checkpoint_limit':'CHECKPOINT_STATE_RAW_LIMIT:1190082971','native_checkpoint_limits_unchanged':True,'scope':'Every runtime file plus capture-root metadata; engine/project/capture inputs separately raw-preserved','manifest':manifest};(B/'FORENSIC_SNAPSHOT.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='manifest'}))
