"""Package a cold-audited bounded historical native continuation."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
A=json.loads((B/'AUDIT.json').read_text());assert A['status']=='PASS_COLD_HISTORICAL_DEPTH50_CONTINUATION'
assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
report='CHECKPOINT0219: historical depths45-50 verified\n2026-10-08\n\nSix frozen native phases continued from verified historical bootstrap44, with 193 candidate builds per phase and 24 selected roots: 1,158 candidate builds. All 24 depth50 construction digests, resource skins and capacity vectors match the archived hash-bound reference.\nFresh independent cold native restoration verified all checkpoint dependency bytes, six exact native identities, 144 reconstructed roots and exact DAG roundtrips. No candidates generated during audit. Qualified historical helpers remain byte-identical to 0213.\nMaster remains151; zero admissions. Terminal100 comparison, Q2 and full L0-G8 remain incomplete.\nNext: fresh bounded51-56 native capture from exact BOOTSTRAP50_HISTORICAL.json, preserving 193 recipes,31 bridge pairs and24-root beam. Save capture before execution; drain preservation and independently cold-audit before advancing.\nRecovery: verify MANIFEST.json and CHECKPOINT_EXPORT.json dependencies; READBACKS.json identifies raw objects and Drive IDs. Use restore_current.py with pinned CPython3.13.5/SQLite3.51.3, engine /tmp/ig_engine0204. Included predecessor0218 retains prior recovery and runtime provenance. Do not regenerate committed phases.\n'
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'restore0218/CURRENT_STATUS.json').read_text());s.pop('latest_handoff',None)
s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=219,current_phase='G1_HISTORICAL_DEPTH50_VERIFIED',in_flight_active='NONE',active_native_runtime=json.loads((B/'POINTER.json').read_text())['workspace'],completed_depth=50,historical_verified_depth=50,historical_reconstruction_acceptance='THROUGH50_QUALIFIED_TERMINAL_PENDING',candidate_build_calls=1158,generator_calls=6,generator_calls_scope='0219:six193candidate native phases45-50; cold audit generated none',pending_bytes=0,master_slices=151,new_admissions=0,terminal_comparison='NOT_RUN_AFTER_HISTORICAL_REPAIR',next_scope='G1_HISTORICAL_BOUNDED51_56',next_scope_status='FRESH_CAPTURE_REQUIRED_FROM_AUDITED_HISTORICAL_BOOTSTRAP50',latest_audit=A,checkpoint0219=A)
if (B/'CODE_MIRROR.json').exists():s['code_mirror']=json.loads((B/'CODE_MIRROR.json').read_text())
(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
files={str(p.relative_to(W)):p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt','.zip','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('MIRROR_ELEMENTS.json','DELIVERABLES.json','SAVE_RECEIPT.json')}
boot=Path(json.loads((B/'BOOTSTRAP50_META.json').read_text())['path']);files['continuation0219/BOOTSTRAP50_HISTORICAL.json']=boot
pred=next((W/'restore0218/checkpoint').glob('*.zip'));files['predecessor/'+pred.name]=pred
manifest={k:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for k,p in files.items()}
out=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0219_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r=dict(path=str(out),sha256=hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),bytes=out.stat().st_size,checkpoint=219,status=A['status'],historical_verified_depth=50,master_slices=151,new_admissions=0,native_pending_bytes=0,next_scope=s['next_scope'])
(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
