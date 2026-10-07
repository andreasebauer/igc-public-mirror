from pathlib import Path
import json,datetime,zipfile,hashlib
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());result=json.loads((B/'NATIVE_RESULT.json').read_text());m=json.loads((B/'CODE_MIRROR.json').read_text());last=a['rows'][-1]
assert result['status']=='COMPLETED' and result['result']['completed_depth']==36 and last['level']==36 and len(a['rows'])==6
report=f'''CHECKPOINT0194 — NATIVE PARTITIONED G1 CONTINUATION
2026-10-07

DONE: Registered six-level tranche31–36 executed and committed. 24 roots per depth. 1,158 candidate builds. No regeneration of depths7–30. Master151 remains unchanged; zero admissions. Terminal G1 R100 comparison and Q2 payload remain pending.

The depth25 blocker was resolved by returning only complete new DAG-node records plus exact roots and parent/child science bindings. Existing node records are compared in full; duplicate digests never authorize unequal node bytes. Each result still uses one native task/occurrence and the unchanged8MiB result limit. Scientific recipe,31bridgepairs,193templates and24-state beam remain frozen. Reachable ancestry is assembled from hash-verified partition files and pruned to the exact selected roots before checking the original DAG science hash.

Checks this tranche: exact depth30 bootstrap assembled exclusively from cold-restored files, independent state reconstruction and exact DAG roundtrip; complete transitive architecture preflight and native environment admission. Partition storage/evaluator code unchanged from0193, whose missing-node, changed-partition and full-node collision tests passed. No generator invoked during bootstrap checks.

Readback incident: capture acknowledgement refused an unexpectedly shortened local bootstrap copy (4MiB versus5277403bytes). Fresh raw bytes in a separate temporary location passed exact size and SHA256 verification; all capture roles and native admission then passed before execution. Cause undetermined; no identity checks bypassed.

Initial capture failed BEFORE_EXECUTION with CHECKPOINT_TREE_RAW_LIMIT because its inherited portable registry contained previous completed-work capsules. No phase or generator ran. Registered V2 uses a fresh bounded tranche registry, with exact checkpoint bootstrap linking prior science. Original failed capture is retained, limits unchanged.

Native execution: PASS_BOUNDED_CONTINUATION through36. This bounded outcome is distinct from the pending terminal campaign PASS. Depth36: {last['nodes']} exact reachable nodes,24roots. Science SHA256: {last['science_sha256']}

Preservation: all checkpoint role obligations acknowledged using real raw Drive-object readbacks; pending bytes0. Isolated native checkpoint restored and all six partition DAGs reconstructed and reserialized exactly. Audit resolved transport locators to files inside the restored tree and did not read original-workspace state. No generation during audit. This is state restoration proof, not a claim that the new capture executed all previously frozen depths or that a relocated full runtime rerun was tested.

Capture: {json.loads((B/'POINTER.json').read_text())['capture_id']}
Checkpoint: {json.loads((B/'PRESERVATION_FINAL.json').read_text())['latest_checkpoint']}
Runtime CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib.
Source-only mirror: {m['repository']} {m['commit_sha']}.

TO COME: Register the next bounded continuation from preserved depth36 (37onward), reusing its exact DAG. Maintain same full-node collision checks and native resource stopping rules; continue until100, independently compare all193terminal public interfaces against bound reference, then review admission. No automatic changes to science or limits.

RESTART: NATIVE_CHECKPOINT_SLIM.zip + CHECKPOINT_EXPORT.json receipt dependencies. Readback IDs/SHAs in READBACKS.json. Fetch exact raw objects, verify bytes, restore. Reference paths in partition manifests are transport locators; after relocation resolve verified hash-identical files inside the restored tree as audited. The bundle contains exact source, registrations, native result, tests, audit and preservation metadata; bulk state is separately preserved to keep handoff small.
'''
(B/'REPORT.txt').write_text(report);p=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0194_START_2026-10-07.txt';p.write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(latest_checkpoint=194,status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),in_flight_active='NONE',generator_calls=6,generator_calls_scope='0194: six level evaluator tasks / 1158 candidate builds; historical cumulative count not asserted',pending_bytes=0,active_native_runtime=json.loads((B/'POINTER.json').read_text())['workspace'],next_scope='WP6_G1_EXACT_CONTINUATION_FROM36',next_scope_status='DEPTH36_COLD_PARTITION_RESTORE_VERIFIED');s['code_mirror']=m;s['checkpoint0194']={k:v for k,v in a.items() if k!='rows'};(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
z=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0194_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for q in sorted(B.rglob('*')):
  if q.is_file() and q.suffix in ['.json','.txt','.zip','.log','.py'] and not any(x in q.parts for x in ['readbacks','restore_objects','__pycache__']) and not q.name.startswith('private_') and q.name!='BOOTSTRAP30.json':f.write(q,'partition0194/'+str(q.relative_to(B)))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
meta=[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size} for q in [p,z]];(B/'DELIVERABLES.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
