from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='STOP_TERMINAL_PUBLIC_INTERFACE_EQUALITY_MISMATCH' and a['carrier_ref_overlap']==0 and a['native_pending_bytes']==0 and not a['restore0210_executed']
report=f'''CHECKPOINT0209 — TERMINAL100 SAVED; PUBLIC INTERFACE MISMATCH STOP
2026-10-08

Depth100 generation committed193candidate builds and193retained roots. The registered native restore scope stopped on CHECKPOINT_STATE_RAW_LIMIT:1190082971 because its1.19GB published terminal partition exceeded the frozen1GiB per-file checkpoint bound. No native restore task or result committed. Frozen registration/limits unchanged.

Every runtime file plus capture-root metadata was saved in a forensic ZIP, preserved in Drive and independently raw-readback verified. FORENSIC_SNAPSHOT.json contains all file hashes/lengths; FORENSIC_READBACK.json binds saved IDs and raw archive hash. The39MiB forensic archive is saved separately, not embedded here. Last valid native checkpoint obligations drained; native pending bytes0. The refused terminal snapshot has not been represented as a successful native checkpoint.

A fresh cold forensic audit verified native generation canonical identity and one committed task, reconstructed193roots and passed their exact DAG roundtrip. Full extracted public-interface equality then FAILED against the hash-bound reference. Direct cold-DAG diagnosis confirms0of193observed carrier refs match the193reference carrier refs. ROOT_REF_MISMATCH.json records both complete sets. Full boundary/reservation difference details were not retained from the failed assertion; no separate semantic-equivalence conclusion is claimed.

Terminal reconstructed DAG nodes:{a['nodes']}; roots193.
Science SHA256:{a['science_sha256']}
Native exact identity bytes:{a['native_exact_identity_bytes']}.
Audit generated no candidates and read no original-workspace scientific state.

No terminal PASS, master admission, Q2 generation, G2 graduation or full L0–G8 completion. Master151 unchanged; zero admissions. Depth99 remains the last completed, independently audited bounded continuation. No compact BOOTSTRAP100 was produced after the failed interface gate.

A separate restore-only0210template was prepared while the audit ran. It was NEVER captured or executed; its gate is blocked by this mismatch. It is retained only as unexecuted planning provenance. No subsequent native task or candidate generation occurred.

NEXT: Diagnose divergence between the frozen recipe replay and historical bound terminal population, starting with the disjoint carrier-ref sets and earliest common construction ancestry. Preserve inputs/rules and do not generate or admit until the mismatch has a source-grounded resolution. The operational1GiB publication/checkpoint issue remains a separate constraint for any future registration.

RESTORE: Verify FORENSIC_READBACK.json archive hash/length, then every FORENSIC_MANIFEST entry in a fresh directory. LAST_VALID_CHECKPOINT_EXPORT.json describes the earlier published native checkpoint, not terminal100. Captured engine/project/inputs remain mapped in READBACKS/CAPTURE_PENDING. Included predecessor0208handoff retains bootstrap99 and runtime/recovery provenance. Use python-fixed-host: CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib.193recipes/31bridgepairs, intermediate24beam, terminal193selection,64MiB result budget and engine unchanged. Terminal task budget1800seconds. Verify MANIFEST.json.
''';(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s['checkpoint0208']=s.get('latest_audit',{});s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latest_checkpoint':209,'current_phase':'G1_TERMINAL100_INTERFACE_MISMATCH_STOP','in_flight_active':'NONE','active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'completed_depth':100,'last_fully_verified_continuation_depth':99,'generator_calls':1,'generator_calls_scope':'0209:one committed193candidate generation task; native restore uncommitted; independent interface gate failed','candidate_build_calls':193,'pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'FAIL','interface_carrier_ref_overlap':0,'Q2_payload_available':False,'next_scope':'WP6_G1_TERMINAL_INTERFACE_MISMATCH_DIAGNOSIS_NO_GENERATION','next_scope_status':'STOP:FULL_INTERFACE_EQUALITY_FAILED;0OF193CARRIER_REFS_MATCH;NO_RESTORE0210CAPTURE','latest_audit':a,'checkpoint0209':a,'code_mirror':json.loads((B/'CODE_MIRROR.json').read_text())});(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2));(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2))
files={}
unused={'audit_restore.py','audit_planned_native.py','package.py','package_partial.py','package_planned_native.py'}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ('.json','.txt','.zip','.log','.py') and not any(v in p.parts for v in ('restore_objects','__pycache__')) and not p.name.startswith('private_') and p.name not in ('DELIVERABLES.json','SAVE_RECEIPT.json'):
  prefix='unexecuted_terminal_plans/' if p.name in unused else 'terminal0209/';files[prefix+str(p.relative_to(B))]=p
for p in sorted((W/'restore0210').rglob('*.py')):
 if '__pycache__' not in p.parts:files['unexecuted_restore0210/'+str(p.relative_to(W/'restore0210'))]=p
pred=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0208_HANDOFF_2026-10-08.zip';files['predecessor/'+pred.name]=pred
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()};out=W/'IG_MASTER151_G1_TERMINAL_0209_MISMATCH_STOP_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':209,'status':a['status'],'native_registered_scope_completed':False,'candidate_build_calls_committed':193,'roots':193,'native_pending_bytes':0,'next_scope':s['next_scope']};(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
