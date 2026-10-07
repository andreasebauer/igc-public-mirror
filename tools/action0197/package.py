from pathlib import Path
import json,datetime,zipfile,hashlib
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());r=json.loads((B/'NATIVE_RESULT.json').read_text());m=json.loads((B/'CODE_MIRROR.json').read_text());last=a['rows'][-1]
assert r['status']=='COMPLETED' and r['result']['completed_depth']==54 and last['level']==54 and len(a['rows'])==6
report=f'''CHECKPOINT0197 — EXACT G1 CONTINUATION49–54
2026-10-07

DONE: Registered six-level tranche49–54 completed with verified native runtime evidence.24roots per depth;1158candidate builds. Earlier depths7–48 reused, not regenerated. Master151 unchanged; zero admissions. Terminal G1 R100 comparison and Q2 payload pending.

Depth48 bootstrap assembled solely from cold-restored0196files with raw partition hash verification, full-node conflict checks, exact DAG science identity, independent state reconstruction and exact DAG roundtrip. Native adapter source unchanged from0196; complete transitive preflight and native environment admission passed. New tranche uses an isolated portable registry. Scientific recipe,31bridgepairs,193templates and24-state beam frozen. One task/occurrence per depth;8MiB result limit and preservation limits unchanged.

Result: PASS_BOUNDED_CONTINUATION through54; not terminal campaign PASS.
Depth54 exact reachable nodes: {last['nodes']};roots24.
Depth54 science SHA256: {last['science_sha256']}
Capture: {json.loads((B/'POINTER.json').read_text())['capture_id']}
Checkpoint: {json.loads((B/'PRESERVATION_FINAL.json').read_text())['latest_checkpoint']}

Audit: PASS_COLD_NATIVE_CHECKPOINT_AND_PARTITION_DAG_RESTORE. All six saved DAGs independently reconstructed and reserialized exactly from the isolated restored tree. Original-workspace state was not used during audit. No generation during audit; no claim that a relocated controller rerun was tested. All native save obligations acknowledged with real raw Drive-object readbacks; pending bytes0.

Runtime CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib.
Code-only mirror: {m['repository']} commit {m['commit_sha']}.

NEXT: Register bounded55onward from preserved depth54. Initialize an isolated registry per tranche; reuse verified predecessor science and retain its provenance. Keep full-node collision checks, exact science hashes and current stopping rules. Continue toward100, independently compare193terminal interfaces with the bound saved reference, then review admission. No automatic changes to science or limits.

RESTART: NATIVE_CHECKPOINT_SLIM.zip and CHECKPOINT_EXPORT.json list exact raw dependencies and saved IDs. READBACKS.json maps IDs/hashes/verified bytes. Fetch exact objects, verify bytes before native restore. Partition paths are transport locators: after relocation resolve hash-identical files inside restored tree as audited. Handoff includes scripts, registration, tests, native result, audit and preservation metadata; bulk state is separately preserved.
'''
(B/'REPORT.txt').write_text(report);p=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0197_START_2026-10-07.txt';p.write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(latest_checkpoint=197,status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),in_flight_active='NONE',generator_calls=6,generator_calls_scope='0197:six evaluator tasks/1158candidate builds; historical cumulative count not asserted',pending_bytes=0,active_native_runtime=json.loads((B/'POINTER.json').read_text())['workspace'],next_scope='WP6_G1_EXACT_CONTINUATION_FROM54',next_scope_status='DEPTH54_COLD_PARTITION_RESTORE_VERIFIED');s['code_mirror']=m;s['checkpoint0197']={k:v for k,v in a.items() if k!='rows'};(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
z=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0197_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for q in sorted(B.rglob('*')):
  if q.is_file() and q.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in q.parts for v in ['readbacks','restore_objects','__pycache__']) and not q.name.startswith('private_'):f.write(q,'partition0197/'+str(q.relative_to(B)))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
meta=[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size} for q in [p,z]];(B/'DELIVERABLES.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
