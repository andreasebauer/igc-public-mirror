from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='PASS_FORENSIC_COLD_TERMINAL100_DAG_AND193_INTERFACE_COMPARISON' and a['completed_depths']==[100] and a['native_registered_scope_completed'] is False and a['native_pending_bytes']==0
last=a['rows'][-1];meta=json.loads((B/'BOOTSTRAP100_META.json').read_text());boot=Path(meta['path']);assert hashlib.file_digest(boot.open('rb'),'sha256').hexdigest()==meta['sha256']
report=f'''CHECKPOINT0209 — TERMINAL100 SAVED; NATIVE SCOPE STOPPED
2026-10-08

Depth100 generation committed193candidate builds and193retained roots. The subsequent native restore task stopped at CHECKPOINT_STATE_RAW_LIMIT:1190082971 because the published terminal partition exceeded the frozen1GiB per-file checkpoint bound. No native restore task/state committed; no native terminal completion or terminal-scope PASS claimed. Frozen registration and limits retained unchanged.

Every runtime file was preserved in a forensic ZIP, saved to Drive and independently raw-readback verified. FORENSIC_SNAPSHOT.json contains all file hashes/lengths; FORENSIC_READBACK.json binds saved IDs. The39MiB archive is saved separately and not embedded here. Last valid published native checkpoint dependencies were acknowledged; native pending bytes0. This does not imply the refused terminal snapshot became a native checkpoint.

From that verified forensic archive, a fresh cold audit checked depth100 canonical identity bytes and one committed generation task, reconstructed all193roots, roundtripped the exact DAG, and independently compared all193public interfaces to the bound reference. The independent comparison passed. Audit candidate builds0; no original-workspace scientific state read.

Reachable nodes:{last['nodes']}; roots193.
Science SHA256:{last['science_sha256']}
BOOTSTRAP100 SHA256:{meta['sha256']} (included).
Master151 unchanged; zero admissions; no Q2 payload generated.193recipes/31bridgepairs, intermediate24beam, terminal193selection,64MiB result budget and engine unchanged.1800-second terminal task budget. CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib; use python-fixed-host.

NEXT: Register a separate RESTORE-only native task from byte-verified compact BOOTSTRAP100, with the same bound reference. Reuse completed depth100; generate no candidates. Preserve/audit that successful native result before scoped admission/Q2 readiness review. No G2 graduation or full L0–G8 completion claimed.

RESTORE: FORENSIC_READBACK.json locates the byte-identical forensic archive. Verify its hash/length and each FORENSIC_MANIFEST entry; restore into a fresh directory. LAST_VALID_CHECKPOINT_EXPORT.json describes the earlier native checkpoint, not the terminal100state. Included predecessor retains runtime/recovery provenance. Absolute paths are locators; resolve only hash-identical objects. Verify MANIFEST.json.
'''
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s['checkpoint0208']=s.get('latest_audit',{});s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'in_flight_active':'NONE','active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'next_scope_status':'TERMINAL100_AND_INDEPENDENT193_INTERFACE_COMPARISON_COLD_VERIFIED;NATIVE_RESTORE_TASK_PENDING','generator_calls_scope':'0209:one committed193candidate generation task; native restore uncommitted at checkpoint raw-file bound','latest_checkpoint':209,'current_phase':'G1_TERMINAL100_FORENSIC_COLD_AUDITED_NATIVE_RESTORE_PENDING','completed_depth':100,'master_slices':151,'new_admissions':0,'pending_bytes':0,'generator_calls':1,'candidate_build_calls':193,'latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'next_scope':'WP6_G1_NATIVE_RESTORE_ONLY_FROM100_AFTER_PRESERVATION_BOUND_STOP','latest_audit':a,'latest_bootstrap':meta});s['code_mirror']=json.loads((B/'CODE_MIRROR.json').read_text());s['checkpoint0208']=s.get('checkpoint0208',s.get('latest_audit',{}));s['checkpoint0209']=a;s['terminal_comparison']='INDEPENDENT_COLD_PASS_NATIVE_RESTORE_PENDING';s['interfaces_checked']=193;s['reconstructed_terminal_parent_DAG_available']=True;s['Q2_payload_available']=False;(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2));(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2))
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in p.parts for v in ['restore_objects','__pycache__']) and not p.name.startswith('private_') and p.name not in ['DELIVERABLES.json','SAVE_RECEIPT.json']:files['terminal0209/'+str(p.relative_to(B))]=p
files['BOOTSTRAP100.json']=boot
files['predecessor/IG_MASTER151_G1_PARTITION_CONTINUATION_0208_HANDOFF_2026-10-08.zip']=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0208_HANDOFF_2026-10-08.zip'
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()}
out=W/'IG_MASTER151_G1_TERMINAL_0209_PARTIAL_RESOURCE_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
result={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':209,'completed_depths':[100],'pending_bytes':0,'next_scope':'WP6_G1_NATIVE_RESTORE_ONLY_FROM100_AFTER_PRESERVATION_BOUND_STOP','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
