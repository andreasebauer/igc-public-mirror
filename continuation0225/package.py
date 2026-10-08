"""Package separately registered native recovery with predecessor provenance."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
A=json.loads((B/'AUDIT.json').read_text());assert A['status']=='PASS_COLD_NATIVE_DEPTH80_COMPLETION_RECOVERY'
assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
report='CHECKPOINT0225: depth80 separate native completion recovery PASS\n2026-10-08\n\nThe original0224 capture remains PAUSED_WORKSPACE_BUDGET. Its immutable registration and committed six phases were not changed. A separate frozen0225 native recovery registered a4GiB workspace budget, bound saved phase exact identities, bootstrap74, archived80 anchor, prior cold audit and prior checkpoint export before execution.\nNative recovery verified six saved identities, one committed task per original phase,144 DAG root roundtrips, and all24 depth80 roots, skins and capacity vectors. No candidate generation occurred during recovery or independent cold audit. New0225 native completion is published and independently authenticated after cold checkpoint restoration. This does not relabel old0224 completion.\nThe original1,158 candidate builds remain original0224 science. Bootstrap80 was independently reproduced byte-exactly for the next tranche. Master151; no new admissions. Terminal100 comparison and Q2 remain incomplete.\nNext: depths81–86 under a fresh frozen capture with adequate registered budget; use BOOTSTRAP80_HISTORICAL.json and saved archived86 source reference. Save all raw capture objects with exact readback before execution. Preserve and independently cold-audit before advancing.\nRestore with restore_current.py and CHECKPOINT_EXPORT.json dependencies. PREDECESSOR_HANDOFF_REFERENCE.json identifies the exact saved0224 handoff and paused provenance.\n'
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.pop('latest_handoff',None)
s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=225,latest_capture_id=json.loads((B/'POINTER.json').read_text())['capture_id'],current_phase='G1_HISTORICAL_DEPTH80_NATIVE_RECOVERY_VERIFIED',in_flight_active=None,active_native_runtime=None,completed_depth=80,historical_verified_depth=80,historical_cold_audited_depth=80,historical_reconstructed_depth=80,historical_reconstruction_acceptance='THROUGH80_QUALIFIED_WITH_SEPARATE_NATIVE_RECOVERY_0225',candidate_build_calls=0,generator_calls=1,native_verification_tasks=1,generator_calls_scope='0225:one native verification task; zero candidate generation',pending_bytes=0,master_slices=151,new_admissions=0,terminal_comparison='NOT_RUN_AFTER_HISTORICAL_REPAIR',next_scope='G1_HISTORICAL81_86',next_scope_status='READY_FOR_FRESH_CAPTURE',latest_audit=A,checkpoint0225=A)
s['checkpoint0224_original_native_status']='PAUSED_WORKSPACE_BUDGET_UNMODIFIED'
if (B/'CODE_MIRROR.json').exists():s['code_mirror']=json.loads((B/'CODE_MIRROR.json').read_text())
(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
files={str(p.relative_to(W)):p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt','.zip','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('MIRROR_ELEMENTS.json','DELIVERABLES.json','SAVE_RECEIPT.json')}
boot=Path(json.loads((B/'BOOTSTRAP80_META.json').read_text())['path']);files['continuation0225/BOOTSTRAP80_HISTORICAL.json']=boot
manifest={k:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for k,p in files.items()}
out=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0225_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':225,'status':A['status'],'historical_verified_depth':80,'native_completion_status':'VERIFIED_SEPARATE_RECOVERY_0225','prior0224_native_status':'PAUSED_WORKSPACE_BUDGET_UNMODIFIED','master_slices':151,'new_admissions':0,'native_pending_bytes':0,'next_scope':s['next_scope']}
(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
