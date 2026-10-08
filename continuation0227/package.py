"""Package cold-audited historical depths 87–92 with restoration provenance."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
A=json.loads((B/'AUDIT.json').read_text());assert A['status']=='PASS_COLD_HISTORICAL_DEPTH92_CONTINUATION'
assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
report='Checkpoint 0227: historical depths 87–92 verified and natively completed.\n2026-10-08\n\nThe frozen capture starts from the exact cold-audited bootstrap 86 earned in saved checkpoint 0226, with lossless gzip transport bound by both compressed and decoded hashes. Historical scientific helpers are byte-identical to 0226; recipes, selection and candidate census are unchanged. Registered workspace and snapshot budgets are each 4 GiB; checkpoint backlog allowance is 16 commits.\nSix native phases each commit 193 candidates and select 24 roots: 1,158 candidates in the committed census. All 24 depth 92 roots, resource skins and capacity vectors match the archived reference. Terminal completion is preserved and independently authenticated from its original capsule.\nA fresh cold restoration verifies every dependency byte, all six exact native identities, and 144 root DAG roundtrips. No candidates are generated during audit. Bootstrap 92 is exported for the next tranche. Master remains 151 with no new admissions. Terminal 100 comparison, Q2 and full L0–G8 replay remain incomplete.\nPREDECESSOR_HANDOFF_REFERENCE.json identifies saved checkpoint 0226, including its retained preservation refusals and recoveries. ORIGINAL0226_CACHE_ARCHIVE.json points to the exact lossless archive of its original working bytes. Completed runtime copies were released only after saved handoff and dependency verification.\nNext: depths 93–98, binding bootstrap 92 and the archived depth 98 reference in a fresh preserved capture; stop on the first mismatch and cold-audit before advancing.\nRecovery: verify MANIFEST.json and retrieve exact native checkpoint dependencies using READBACKS.json. restore_current.py restores the checkpoint with the pinned CPython 3.13.5 / SQLite 3.51.3 runtime and frozen engine. Resume only missing phases.\n'
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.pop('latest_handoff',None)
s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=227,latest_capture_id=json.loads((B/'POINTER.json').read_text())['capture_id'],current_phase='G1_HISTORICAL_DEPTH92_NATIVE_VERIFIED',in_flight_active=None,active_native_runtime=None,completed_depth=92,historical_verified_depth=92,historical_cold_audited_depth=92,historical_reconstructed_depth=92,historical_reconstruction_acceptance='THROUGH92_HISTORICAL_NATIVE_QUALIFIED',candidate_build_calls=1158,generator_calls=6,committed_generator_tasks=6,interrupted_generator_attempts=0,interrupted_candidate_work='NONE',native_verification_tasks=0,generator_calls_scope='0227:six committed phase tasks plus one interrupted depth92 attempt; 1158 committed candidate census; interrupted candidate work not measured; audit generated none',pending_bytes=0,master_slices=151,new_admissions=0,terminal_comparison='NOT_RUN_AFTER_HISTORICAL_REPAIR',next_scope='G1_HISTORICAL93_98',next_scope_status='READY_FOR_FRESH_CAPTURE',latest_audit=A,checkpoint0227=A)
s['checkpoint0224_original_native_status']='PAUSED_WORKSPACE_BUDGET_UNMODIFIED'
if (B/'CODE_MIRROR.json').exists():s['code_mirror']=json.loads((B/'CODE_MIRROR.json').read_text())
s['catalog_path']=str(B/'CATALOG_0151.json')
(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
files={str(p.relative_to(W)):p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt','.zip','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('MIRROR_ELEMENTS.json','DELIVERABLES.json','SAVE_RECEIPT.json')}
boot=Path(json.loads((B/'BOOTSTRAP92_META.json').read_text())['path']);files['continuation0227/BOOTSTRAP92_HISTORICAL.json']=boot
manifest={k:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for k,p in files.items()}
out=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0227_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':227,'status':A['status'],'historical_verified_depth':92,'native_completion_status':'VERIFIED' ,'prior0224_native_status':'PAUSED_WORKSPACE_BUDGET_UNMODIFIED','master_slices':151,'new_admissions':0,'native_pending_bytes':0,'next_scope':s['next_scope']}
(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
