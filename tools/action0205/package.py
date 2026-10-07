from pathlib import Path
import json,datetime,zipfile,hashlib
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());m=json.loads((B/'CODE_MIRROR.json').read_text());pointer=json.loads((B/'POINTER.json').read_text());last=a['rows'][-1]
assert a['status']=='PASS_COLD_CHECKPOINT_EXACT_NATIVE_IDENTITIES_AND_DAG_CLOSURE' and a['generation_completed_depths']==[91,94] and a['native_published_manifest_depths']==[91,93] and a['pending_bytes']==0
report=f'''READ FIRST — CHECKPOINT0205 PARTIAL USER-PAUSED NEW-CHAT HANDOFF
2026-10-08

NOT A COMPLETED91–96 TRANCHE. User requested a new-chat handoff; native pause was requested and acknowledged before publishing94. ERROR.txt records REQUESTED_PAUSE, not a scientific mismatch. Original registration91–96 is frozen and incomplete. No terminal PASS claimed.

SAVED: native exact generation tasks91–94 completed and committed:772candidate builds,24roots per depth. Manifests91–93 published. Depth94 exact delta remains in its native state_store.sqlite3; its manifest was not published.95–96 were not started.7–90 reused, not regenerated. Master151unchanged; zero admissions and zero pending bytes. G1 terminal100 comparison/Q2 remain pending.

PROOF: all raw objects saved and downloaded with SHA/length verification before acknowledgement. Paused native checkpoint independently restored. All four full canonical identity byte strings compared with their stored state payloads; parent-science bindings, full-node conflicts, exact reachable DAG closure/science hashes verified. Published91–93 also match native stored states. No original-workspace state read during audit. No generation during audit.
BOUNDARY: independent state-object reconstruction/serialization for91–94 was NOT run; relocated controller resume NOT tested. Do not label this the earlier full independent-state restoration audit. The saved94 gate below is required before further generation.

Depth94 reachable nodes:{last['nodes']};roots24.
Depth94 science SHA256:{last['science_sha256']}
Capture:{pointer['capture_id']}
Recorded original workspace:{pointer['workspace']}
Code-only mirror:{m['repository']} commit {m['commit_sha']}.

NEXT CHAT:
1. Recover the exact pinned runtime. runtime/RECOVERY_RUNTIME_PARTS.json gives six raw IDs/hashes. Concatenate in part_number order, verify archive SHA, extract ONLY the specified runtime tar. runtime manifests/policy retain4721file hashes and executable modes. recover_runtime.py verifies these and the captured engine. CPython3.13.5/SQLite3.51.3/Decoder0.8.0.dev151+lib.
2. Fetch current CHECKPOINT_EXPORT.json dependencies via READBACKS.json; verify every raw hash/length. Restore NATIVE_CHECKPOINT_SLIM.zip with restore_current.py into a fresh /tmp tree. Exact captured engine archive SHA60c729a8ee3ec81b0b367b44719e238271ad7d0ae9112cebe81253cf3f874c30. No theory/concept files required.
3. Run reconstruct_saved94.py on that restored tree. It checks complete native identities/parent bindings/DAG closure, then independently reconstructs94state objects and requires exact DAG roundtrip. It writes canonical BOOTSTRAP94 JSON with zero candidate generation. Stop on any mismatch.
4. After this gate passes, register bounded95–96 from the saved94bootstrap in an isolated registry. DO NOT regenerate91–94. Keep193recipes,31bridgepairs,24-state beam, full-node equality and stopping rules. Engine and all four functional adapter files unchanged from0204; only the empty package marker differs by one newline, recorded in ADAPTER_BYTE_AUDIT.json. Keep the explicit64MiB result budget; unnecessary fixed8MiB ceiling was removed in0204.
5. Fully preserve/audit95–96 before97–99, then register100 and independently compare193terminal public interfaces before reviewing admission.

Original paused capture retained for provenance. Same-capture resume must keep all task/input bindings and reuse four committed tasks. Relocated controller resume has NOT been qualified; do not bypass a binding failure or rerun completed94. Separately registered95–96 after the exact94gate is the planned clean continuation.
audit_restore.py is the ORIGINAL planned six-depth audit, not valid yet for this partial scope. audit_handoff.py records the actual handoff audit. Self-contained runtime and predecessor recovery metadata included. Bulk raw checkpoint state is separately saved.
'''
(B/'REPORT.txt').write_text(report);start=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0205_START_2026-10-08.txt';start.write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(latest_checkpoint=205,status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),in_flight_active='NONE',generator_calls=4,generator_calls_scope='0205:four committed evaluator tasks/772candidate builds; paused before94publication',pending_bytes=0,active_native_runtime=pointer['workspace'],next_scope='WP6_G1_SAVED94_INDEPENDENT_RECONSTRUCTION_THEN95_96',next_scope_status='USER_PAUSED_HANDOFF_94_STATE_SAVED_NOT_PUBLISHED');s['code_mirror']=m;s['checkpoint0205']={k:v for k,v in a.items() if k!='rows'};(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
handoff=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0205_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(handoff,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(B.rglob('*')):
  if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(x in p.parts for x in ['restore_objects','__pycache__']) and not p.name.startswith('private_'):z.write(p,'partition0205/'+str(p.relative_to(B)))
with zipfile.ZipFile(handoff) as z:assert z.testzip() is None
meta=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [start,handoff]];(B/'DELIVERABLES.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
