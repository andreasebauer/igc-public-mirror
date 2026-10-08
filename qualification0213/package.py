"""Package audited historical qualification and immutable failed-attempt provenance."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='PASS_COLD_HISTORICAL_SEED_AND_DEPTH14_QUALIFICATION'
assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
report='''CHECKPOINT0213 — HISTORICAL SEED AND DEPTH14 QUALIFIED
2026-10-08

The isolated archived v0.28.8 scientific helper closure is now hosted by the unchanged current native Decoder controller. Retained historical definitions are AST-exact. Shared canonicalizer is byte-equal; overlay and dynamic loader definitions are AST-equal. The extractor now retains cross-module sibling dependencies and compiles all project sources before capture. Current global O7 scientific identity remains unchanged.

Fresh native capture generated depth7 from the hash-bound205seed records, selected24 under the historical identity namespace, and recovered the previously missing historical first seed source. Depths8–14 each committed193candidate builds and retained24roots. All24depth14construction digests, resource skins and seven-component capacity vectors exactly match the hash-bound historical SOURCE_INPUT probes. Total intermediate candidate builds1351. Eight native phases completed. No wrong-namespace replay state was reused.

Independent cold native checkpoint restoration verified every dependency hash/length, native exact canonical state identities and one committed task per phase; reconstructed192roots across depths7–14 and passed exact original DAG roundtrips. The cold depth14comparison again matched all24historical roots/skins/capacity vectors. The audit generated no candidates and read no original-workspace scientific state. BOOTSTRAP14_HISTORICAL.json is compact exact depth14scientific state for the next fresh registration.

Two failed operational captures are retained unchanged:0211stopped on future-import placement,0212on an omitted scanner SHA-file helper dependency. Both stopped at seed7with zero committed tasks/states and zero committed candidate builds. Their native checkpoints are saved, raw-readback verified and drained. No frozen source/spec was edited and no committed scientific task rerun. Planned but unused cold-audit scripts in those directories are explicitly unexecuted.

All native pending obligations are drained. Raw Drive object readbacks are SHA/length verified; READBACKS and CHECKPOINT_EXPORT files locate all restoration dependencies. NATIVE_CHECKPOINT_SLIM.zip plus dependencies restores the actual0213checkpoint. Included0210diagnostic handoff retains original divergence source evidence. Python code is mirrored in the authorized public repository; scientific state/evidence remains in the saved handoff and private object saves.

Master151 remains unchanged, zero admissions, Q2 not generated, G2 not graduated, full L0–G8 incomplete. Historical repair is qualified through14only; terminal100is still pending. Earlier current-namespace8–100states remain diagnostic provenance and cannot be promoted as historical reconstruction.

NEXT: Prepare a fresh bounded native historical15–20capture from this exact bootstrap, retaining193recipes/31bridgepairs,24intermediate beam and193terminal selection. Reuse the attested isolated historical helpers and pinned current runtime; verify recipe equality and each exact DAG, then compare depth20against archived probes before long continuation. Never change scientific selection rules to fit anchors. The prior terminal pretty-JSON1GiB checkpoint failure remains a separate operational issue; future terminal publication must be compact/bounded inside its own frozen registration.

RESTORE: Use pinned python-fixed-host (CPython3.13.5, SQLite3.51.3, Decoder0.8.0.dev151+lib). Verify MANIFEST.json and CHECKPOINT_EXPORT dependencies before restore_current.py; require_saved and environment validation before any new generation. Result transport64MiB, work900seconds per task, workspace2GiB, state file1GiB. Frozen engine SHA60c729a8ee3ec81b0b367b44719e238271ad7d0ae9112cebe81253cf3f874c30.
'''
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latest_checkpoint':213,'current_phase':'G1_HISTORICAL_SEED_AND_DEPTH14_QUALIFIED','in_flight_active':'NONE','latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'completed_depth':14,'historical_verified_depth':14,'historical_reconstruction_acceptance':'EARLY14_QUALIFIED_TERMINAL_PENDING','candidate_build_calls':1351,'pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN_AFTER_HISTORICAL_REPAIR','Q2_payload_available':False,'next_scope':'G1_HISTORICAL_BOUNDED15_20','next_scope_status':'FRESH_CAPTURE_REQUIRED_FROM_AUDITED_HISTORICAL_BOOTSTRAP14','latest_audit':a,'checkpoint0213':a,'checkpoint0211':json.loads((W/'qualification0211/FAILURE_AUDIT.json').read_text()),'checkpoint0212':json.loads((W/'qualification0212/FAILURE_AUDIT.json').read_text()),'code_mirror':json.loads((B/'CODE_MIRROR.json').read_text())});(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
files={}
for D in (W/'qualification0211',W/'qualification0212',B):
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.suffix in ('.py','.json','.zip','.txt','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('MIRROR_ELEMENTS.json','DELIVERABLES.json','SAVE_RECEIPT.json'):
   prefix='unexecuted_failed_attempt_audit_plans/' if D!=B and p.name=='audit_cold.py' else '';files[prefix+str(p.relative_to(W))]=p
boot=Path(json.loads((B/'BOOTSTRAP14_META.json').read_text())['path']);files['qualification0213/BOOTSTRAP14_HISTORICAL.json']=boot
pred=W/'IG_MASTER151_G1_DIVERGENCE_DIAGNOSIS_0210_HANDOFF_2026-10-08.zip';files['predecessor/'+pred.name]=pred
manifest={k:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for k,p in files.items()};out=W/'IG_MASTER151_G1_HISTORICAL_QUALIFICATION_0213_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':213,'status':a['status'],'historical_verified_depth':14,'candidate_build_calls':1351,'master_slices':151,'new_admissions':0,'native_pending_bytes':0,'next_scope':s['next_scope']};(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
