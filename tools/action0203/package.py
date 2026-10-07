from pathlib import Path
import json,datetime,zipfile,hashlib
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());m=json.loads((B/'CODE_MIRROR.json').read_text());last=a['rows'][-1]
assert a['completed_depths']==[85,88] and len(a['rows'])==4 and last['level']==88 and a['pending_bytes']==0 and a['failure']=='STREAM_RESULT_BYTES_LIMIT'
report=f'''CHECKPOINT 0203 — PRESERVED PARTIAL G1 CONTINUATION, DEPTHS 85–88
2026-10-07

RESULT: BLOCKED_AT_DEPTH89_STREAM_RESULT_BYTES_LIMIT. Registered scope was 85–90; it was NOT completed. Native controller stopped on its frozen 8 MiB result envelope limit at depth 89. Depth 89 was attempted but not committed; depth 90 was not started. No science mismatch is claimed or inferred from this storage failure. Do not represent the failed native job as PASS.

DONE: Depths 85–88 committed, with 24 roots per depth. Four committed phases contain 772 candidate builds; this excludes the work attempted by the failed depth-89 task. Earlier depths 7–84 were reused, not regenerated. Master remains at 151 slices, zero admissions. Terminal G1 R100 interface comparison and Q2 payload remain pending.

START: Depth-84 bootstrap reconstructed solely from cold-restored checkpoint 0202, with raw partition hash verification, full-node conflict checks, exact science identity, independent reconstruction and exact DAG roundtrip. Scientific adapter was unchanged. Transitive preflight, native capture readbacks and native environment admission passed. Python 3.13.5, SQLite 3.51.3, Decoder 0.8.0.dev151+lib.

STOP: Preserve the original registered job and its refusal. Never silently raise the 8 MiB limit or mark depths 89–90 complete. Explicit idle snapshot records the stopped workspace after refusal; all native save obligations have real raw Drive readbacks and pending bytes are zero.

AUDIT: All four committed saved DAGs independently reconstructed and reserialized exactly from an isolated restored checkpoint. Original-workspace state was not used for this audit. No candidate generation during audit. This proves state restoration; a relocated controller rerun was not tested.
Depth 88 roots: 24
Depth 88 exact reachable nodes: {last['nodes']}
Depth 88 science SHA256: {last['science_sha256']}
Capture: {json.loads((B/'POINTER.json').read_text())['capture_id']}
Checkpoint: {json.loads((B/'PRESERVATION_FINAL.json').read_text())['latest_checkpoint']}
Code-only mirror: {m['repository']}, commit {m['commit_sha']}.

NEXT: Repair the exact delta transport envelope under a new registration, keeping the scientific recipe, 24-state beam, full-node collision checks, DAG science identities and limits unchanged. Validate deterministic lossless encoding and exact decode/roundtrip on saved deltas before native capture. Start from restored depth 88 and run only the missing depths 89–90; retain failed-attempt provenance. Do not regenerate 85–88. Continue toward 100 only after verified preservation and audit of the repaired tranche. Full terminal comparison of 193 interfaces is still required before admission.

RESTART: NATIVE_CHECKPOINT_SLIM.zip and CHECKPOINT_EXPORT.json define exact dependencies and saved IDs. READBACKS.json maps Drive IDs, hashes and verified byte paths. Fetch exact objects and verify raw bytes before restore. Resolve old partition transport locators to hash-identical files inside the restored tree, as audited. Bulk state is preserved separately; this small handoff includes scripts, registration, preflight, refusal, snapshot, audit and preservation metadata.
'''
p=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0203_START_2026-10-07.txt';p.write_text(report);(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(latest_checkpoint=203,status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),in_flight_active='NONE',pending_bytes=0,active_native_runtime=json.loads((B/'POINTER.json').read_text())['workspace'],next_scope='WP6_G1_EXACT_TRANSPORT_REPAIR_FROM88',next_scope_status='DEPTH89_STREAM_RESULT_BYTES_LIMIT_REQUIRES_NEW_REGISTRATION',generator_calls=5,generator_calls_scope='0203:four committed evaluator tasks/772 candidate builds plus one failed depth89 task; failed candidate count not asserted');s['code_mirror']=m;s['checkpoint0203']={k:v for k,v in a.items() if k!='rows'};s['checkpoint0203']['native_result']='FAILED_REGISTERED_SCOPE_NOT_COMPLETED';(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
z=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0203_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for q in sorted(B.rglob('*')):
  if q.is_file() and q.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in q.parts for v in ['readbacks','restore_objects','__pycache__']) and not q.name.startswith('private_'):f.write(q,'partition0203/'+str(q.relative_to(B)))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
meta=[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size} for q in [p,z]];(B/'DELIVERABLES.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
