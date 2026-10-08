"""Package a cold-audited bounded historical native continuation."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
A=json.loads((B/'AUDIT.json').read_text());assert A['status']=='PASS_COLD_HISTORICAL_DEPTH56_CONTINUATION'
assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
report='CHECKPOINT0220: historical depths51-56 verified\n2026-10-08\n\nSix frozen native phases continued from verified historical bootstrap50, with 193 candidate builds per phase and 24 selected roots: 1,158 candidate builds. All 24 depth56 construction digests, resource skins and capacity vectors match the archived hash-bound reference.\nFresh independent cold native restoration verified all checkpoint dependency bytes, six exact native identities, 144 reconstructed roots and exact DAG roundtrips. No candidates generated during audit. Qualified historical helpers remain byte-identical to 0213.\nMaster remains151; zero admissions. Terminal100 comparison, Q2 and full L0-G8 remain incomplete.\nNext: fresh bounded57-62 native capture from exact BOOTSTRAP56_HISTORICAL.json, preserving 193 recipes,31 bridge pairs and24-root beam. Save capture before execution; drain preservation and independently cold-audit before advancing.\nRecovery: verify MANIFEST.json and CHECKPOINT_EXPORT.json dependencies; READBACKS.json identifies raw objects and Drive IDs. Use restore_current.py with pinned CPython3.13.5/SQLite3.51.3, engine /tmp/ig_engine0204. Included predecessor0219 retains prior recovery and runtime provenance. Do not regenerate committed phases.\n'
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'continuation0219/STATUS_CANDIDATE.json').read_text());s.pop('latest_handoff',None)
s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=220,latest_capture_id=json.loads((B/'POINTER.json').read_text())['capture_id'],current_phase='G1_HISTORICAL_DEPTH56_VERIFIED',in_flight_active='NONE',active_native_runtime=json.loads((B/'POINTER.json').read_text())['workspace'],completed_depth=56,historical_verified_depth=56,historical_reconstruction_acceptance='THROUGH56_QUALIFIED_TERMINAL_PENDING',candidate_build_calls=1158,generator_calls=6,generator_calls_scope='0220:six193candidate native phases51-56; cold audit generated none',pending_bytes=0,master_slices=151,new_admissions=0,terminal_comparison='NOT_RUN_AFTER_HISTORICAL_REPAIR',next_scope='G1_HISTORICAL_BOUNDED57_62',next_scope_status='FRESH_CAPTURE_REQUIRED_FROM_AUDITED_HISTORICAL_BOOTSTRAP56',latest_audit=A,checkpoint0220=A)
if (B/'CODE_MIRROR.json').exists():s['code_mirror']=json.loads((B/'CODE_MIRROR.json').read_text())
(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
files={str(p.relative_to(W)):p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt','.zip','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('MIRROR_ELEMENTS.json','DELIVERABLES.json','SAVE_RECEIPT.json')}
boot=Path(json.loads((B/'BOOTSTRAP56_META.json').read_text())['path']);files['continuation0220/BOOTSTRAP56_HISTORICAL.json']=boot
pred=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0219_HANDOFF_2026-10-08.zip';files['predecessor/'+pred.name]=pred
manifest={k:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for k,p in files.items()}
out=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0220_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r=dict(path=str(out),sha256=hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),bytes=out.stat().st_size,checkpoint=220,status=A['status'],historical_verified_depth=56,master_slices=151,new_admissions=0,native_pending_bytes=0,next_scope=s['next_scope'])
(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
