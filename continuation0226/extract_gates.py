"""Recover only archived comparison gates from the byte-verified historical archive."""
from pathlib import Path
import json,zipfile,hashlib
B=Path(__file__).resolve().parent;sha='a09e2610bd63af005861baf381d9fdbcaa9b1101633d29644ffeadb35796cc45';archive=Path('/tmp/ig_verified0226')/(sha+'.bin');assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==sha
rows=[]
with zipfile.ZipFile(archive) as z:
 for depth in (86,92,98,100):
  prefix=f'run/checkpoints/O{depth:05d}/';manifest_raw=z.read(prefix+'CHECKPOINT_MANIFEST.json');m=json.loads(manifest_raw)
  assert m['depth']==depth
  for ref in m['files']:
   raw=z.read(prefix+ref['path']);assert len(raw)==ref['size_bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
  raw=z.read(prefix+'SOURCE_INPUT.json');name=f'HISTORICAL_DEPTH{depth}_SOURCE_INPUT.json';target=B/name
  if target.exists():assert target.read_bytes()==raw
  else:target.write_bytes(raw)
  mf=B/f'HISTORICAL_DEPTH{depth}_CHECKPOINT_MANIFEST.json';mf.write_bytes(manifest_raw)
  rows.append({'depth':depth,'source_path':str(target),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'manifest_path':str(mf),'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'manifest_files_verified':len(m['files'])})
out={'status':'PASS_ARCHIVED_GATES_RECOVERED_EXACT','archive_sha256':sha,'archive_bytes':archive.stat().st_size,'archive_drive_id':'1TSBYd8pwkMzO1l4ZvUwUDdyfpWEuKxDs','raw_readback_verified':True,'gates':rows,'candidate_generation_calls':0};(B/'ARCHIVED_GATE_RECOVERY.json').write_text(json.dumps(out,indent=2));n=rows[1];(B/'NEXT_GATE_EVIDENCE.json').write_text(json.dumps({'scope':'HISTORICAL87_92','path':n['source_path'],'sha256':n['source_sha256'],'bytes':n['source_bytes'],'archived_checkpoint_manifest':n['manifest_path'],'archived_manifest_sha256':n['manifest_sha256'],'historical_archive_sha256':sha,'historical_archive_drive_id':out['archive_drive_id'],'next_capture_created':False,'next_generator_calls':0,'minimum_registered_workspace_budget_bytes':4294967296},indent=2));print('PASS archived86,92,98,100 gates; current86 reference unchanged')
