from pathlib import Path
import json,datetime,zipfile,hashlib
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());r=json.loads((B/'NATIVE_RESULT.json').read_text());m=json.loads((B/'CODE_MIRROR.json').read_text());last=a['rows'][-1]
assert r['status']=='COMPLETED' and r['result']['completed_depth']==78 and last['level']==78 and len(a['rows'])==6
report=f'''CHECKPOINT0201 — EXACT G1 CONTINUATION73–78
2026-10-07

DONE: Registered six-level tranche73–78 completed with verified native runtime evidence.24roots per depth;1158candidate builds. Earlier depths7–72 reused, not regenerated. Master151 unchanged; zero admissions. Terminal G1 R100 comparison and Q2 payload pending.

Depth72 bootstrap assembled solely from cold-restored0200files with raw partition hash verification, full-node conflict checks, exact DAG science identity, independent state reconstruction and exact DAG roundtrip. Native adapter source unchanged from0200; complete transitive preflight and native environment admission passed. New tranche uses an isolated portable registry. Scientific recipe,31bridgepairs,193templates and24-state beam frozen. One task/occurrence per depth;8MiB result limit and preservation limits unchanged.

Result: PASS_BOUNDED_CONTINUATION through78; not terminal campaign PASS.
Depth78 exact reachable nodes: {last['nodes']};roots24.
Depth78 science SHA256: {last['science_sha256']}
Capture: {json.loads((B/'POINTER.json').read_text())['capture_id']}
Checkpoint: {json.loads((B/'PRESERVATION_FINAL.json').read_text())['latest_checkpoint']}

Audit: PASS_COLD_NATIVE_CHECKPOINT_AND_PARTITION_DAG_RESTORE. All six saved DAGs independently reconstructed and reserialized exactly from the isolated restored tree. Original-workspace state was not used during audit. No generation during audit; no claim that a relocated controller rerun was tested. All native save obligations acknowledged with real raw Drive-object readbacks; pending bytes0.

Runtime CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib.
Code-only mirror: {m['repository']} commit {m['commit_sha']}.

NEXT: Register bounded79onward from preserved depth78. Initialize an isolated registry per tranche; reuse verified predecessor science and retain its provenance. Keep full-node collision checks, exact science hashes and current stopping rules. Continue toward100, independently compare193terminal interfaces with the bound saved reference, then review admission. No automatic changes to science or limits.

RESTART: NATIVE_CHECKPOINT_SLIM.zip and CHECKPOINT_EXPORT.json list exact raw dependencies and saved IDs. READBACKS.json maps IDs/hashes/verified bytes. Fetch exact objects, verify bytes before native restore. Partition paths are transport locators: after relocation resolve hash-identical files inside restored tree as audited. Handoff includes scripts, registration, tests, native result, audit and preservation metadata; bulk state is separately preserved.
'''
(B/'REPORT.txt').write_text(report);p=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0201_START_2026-10-07.txt';p.write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(latest_checkpoint=201,status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),in_flight_active='NONE',generator_calls=6,generator_calls_scope='0201:six evaluator tasks/1158candidate builds; historical cumulative count not asserted',pending_bytes=0,active_native_runtime=json.loads((B/'POINTER.json').read_text())['workspace'],next_scope='WP6_G1_EXACT_CONTINUATION_FROM78',next_scope_status='DEPTH78_COLD_PARTITION_RESTORE_VERIFIED');s['code_mirror']=m;s['checkpoint0201']={k:v for k,v in a.items() if k!='rows'};(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
z=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0201_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for q in sorted(B.rglob('*')):
  if q.is_file() and q.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in q.parts for v in ['readbacks','restore_objects','__pycache__']) and not q.name.startswith('private_'):f.write(q,'partition0201/'+str(q.relative_to(B)))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
meta=[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size} for q in [p,z]];(B/'DELIVERABLES.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
