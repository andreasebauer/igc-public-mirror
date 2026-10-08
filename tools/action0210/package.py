from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='PASS_COLD_NATIVE193_INTERFACE_RESTORE_AND_PRIOR_INDEPENDENT_DAG_BINDING' and a['pending_bytes']==0 and a['native_registered_scope_completed']
report=f'''CHECKPOINT0210 — G1 TERMINAL193 INTERFACES NATIVE RESTORE VERIFIED
2026-10-08

Depth100 generated193candidates and retained193roots in frozen capture0209. Its native restore scope stopped on the1GiB per-file checkpoint bound after publishing a1.19GB terminal partition. That failure remains unchanged and preserved in a byte-verified forensic archive. Independent cold reconstruction of that archive passed exact193-root DAG roundtrip, native generation identity, and full193public-interface equality. Compact BOOTSTRAP100 was produced only after those checks.

Separate native capture0210 reused that exact audited BOOTSTRAP100 for a restore-only task. It reconstructed193roots and passed full interface equality with the bound reference. Native completion/evidence verified; no candidate generation and no earlier depth rerun. A fresh native checkpoint cold audit checked exact input-byte binding to the independently audited DAG, all193interface rows, native restore canonical identity bytes and its published comparison. All save dependencies raw-readback verified; pending bytes0.

Terminal science SHA256:{a['science_sha256']}
BOOTSTRAP100 SHA256:{a['bootstrap_sha256']}
Master151 unchanged; zero admissions; no Q2 payload generated.193recipes/31bridgepairs, intermediate24-state beam and terminal193selection unchanged. Same engine and checkpoint limits;64MiB result budget;1800-second task budget. Runtime CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib; use python-fixed-host.

NEXT: Review scoped admission of the reconstructed terminal ancestry and Q2 readiness. No master admission, G2 graduation or full L0–G8 completion claimed here. Existing lower-layer lineage limits remain.

RESTORE: CHECKPOINT_EXPORT.json and READBACKS.json list this successful native checkpoint's byte-verified dependencies. Included predecessor0209handoff contains BOOTSTRAP100, full forensic manifest/Drive IDs, independent DAG/interface audit, failed capture provenance and predecessor0208/runtime recovery. Raw forensic39MiB ZIP is separately saved as referenced there. Verify all hashes/lengths before fresh restoration. Absolute paths are transport locators. Verify MANIFEST.json.
''';(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latest_checkpoint':210,'in_flight_active':'NONE','current_phase':'G1_TERMINAL100_NATIVE193_INTERFACE_RESTORE_COLD_VERIFIED','active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'completed_depth':100,'generator_calls':1,'generator_calls_scope':'0210:one native restore-only task; zero candidate builds; depth100generation reused from0209','candidate_build_calls':0,'pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'PASS','interfaces_checked':193,'reconstructed_terminal_parent_DAG_available':True,'Q2_payload_available':False,'next_scope':'WP6_G1_TERMINAL_SCOPED_ADMISSION_AND_Q2_READINESS_REVIEW','next_scope_status':'TERMINAL193_INTERFACE_NATIVE_RESTORE_AND_INDEPENDENT_DAG_AUDIT_PASS;ADMISSION_PENDING','latest_audit':a,'checkpoint0210':a,'code_mirror':json.loads((B/'CODE_MIRROR.json').read_text())});(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2));(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2))
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ('.json','.txt','.zip','.log','.py') and not any(v in p.parts for v in ('restore_objects','__pycache__')) and not p.name.startswith('private_') and p.name not in ('DELIVERABLES.json','SAVE_RECEIPT.json'):files['restore0210/'+str(p.relative_to(B))]=p
pred=W/'IG_MASTER151_G1_TERMINAL_0209_PARTIAL_RESOURCE_HANDOFF_2026-10-08.zip';files['predecessor/'+pred.name]=pred
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()};out=W/'IG_MASTER151_G1_TERMINAL_RESTORE_0210_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,m in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==m['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':210,'terminal_comparison':'PASS','interfaces_checked':193,'pending_bytes':0,'next_scope':s['next_scope']};(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
