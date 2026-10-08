from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());r=json.loads((B/'NATIVE_RESULT.json').read_text());assert a['status']=='PASS_COLD_TERMINAL_DAG_AND193_INTERFACE_COMPARISON' and r['status']=='COMPLETED' and a['completed_depths']==[100] and a['pending_bytes']==0
last=a['rows'][-1];meta=json.loads((B/'BOOTSTRAP100_META.json').read_text());boot=Path(meta['path']);assert hashlib.file_digest(boot.open('rb'),'sha256').hexdigest()==meta['sha256']
report=f'''CHECKPOINT0209 — G1 TERMINAL100 AND193 INTERFACE COMPARISON
2026-10-08

Separate registered terminal run completed:193candidate builds,193retained roots and an independent native193root restore/interface comparison. Depth99 and all earlier completed depths reused. Master151 unchanged; zero admissions. No Q2 payload generated.

The native restore task compared all193public interfaces to the hash-bound reference. A separate fresh cold checkpoint audit reconstructed the exact terminal DAG, roundtripped all193roots exactly, compared stored native identity bytes for both tasks, and independently repeated full interface equality. Audit generated no candidates and read no original-workspace scientific state.

Depth100 reachable nodes:{last['nodes']}; roots193.
Science SHA256:{last['science_sha256']}
BOOTSTRAP100 SHA256:{meta['sha256']} (included).
All native save obligations acknowledged after real raw Drive readbacks; pending bytes0.

Frozen193recipes/31bridgepairs and the24-state intermediate beam unchanged. Terminal selection retains all193as prescribed by the bound plan. Explicit1800-second task budget covers the larger terminal/restore scope;64MiB result budget and engine unchanged. CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib; use python-fixed-host.

NEXT: Review scoped terminal admission and Q2 readiness using the exact reconstructed terminal DAG, full interface equality and preserved provenance. Master admission was not performed here. No G2 graduation or full L0–G8 completion claimed; inherited lower-layer lineage limits remain as recorded in CURRENT_STATUS.

RESTORE: CHECKPOINT_EXPORT.json lists raw dependencies; READBACKS.json maps hashes to Drive IDs. Included predecessor handoff retains runtime/recovery provenance. Verify raw hash/length before native restore_current.py. Absolute paths are locators: resolve only hash-identical files and prepare a fresh capture spec. Verify MANIFEST.json. CODE_MIRROR.json binds public Python source only.
'''
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s['checkpoint0208']=s.get('latest_audit',{});s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'in_flight_active':'NONE','active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'next_scope_status':'TERMINAL100_AND193_INTERFACE_COMPARISON_COLD_VERIFIED;SCOPED_ADMISSION_AND_Q2_READINESS_PENDING','generator_calls_scope':'0209:one193candidate generation task and one independent193root restore task; earlier depths reused','latest_checkpoint':209,'current_phase':'G1_TERMINAL100_AND193_INTERFACES_COLD_AUDITED','completed_depth':100,'master_slices':151,'new_admissions':0,'pending_bytes':0,'generator_calls':2,'candidate_build_calls':193,'latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'next_scope':'WP6_G1_TERMINAL_SCOPED_ADMISSION_AND_Q2_READINESS_REVIEW','latest_audit':a,'latest_bootstrap':meta});s['code_mirror']=json.loads((B/'CODE_MIRROR.json').read_text());s['checkpoint0208']=s.get('checkpoint0208',s.get('latest_audit',{}));s['checkpoint0209']=a;s['terminal_comparison']='PASS';s['interfaces_checked']=193;s['reconstructed_terminal_parent_DAG_available']=True;s['Q2_payload_available']=False;(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2));(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2))
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in p.parts for v in ['restore_objects','__pycache__']) and not p.name.startswith('private_') and p.name not in ['DELIVERABLES.json','SAVE_RECEIPT.json']:files['terminal0209/'+str(p.relative_to(B))]=p
files['BOOTSTRAP100.json']=boot
files['predecessor/IG_MASTER151_G1_PARTITION_CONTINUATION_0208_HANDOFF_2026-10-08.zip']=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0208_HANDOFF_2026-10-08.zip'
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()}
out=W/'IG_MASTER151_G1_TERMINAL_0209_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
result={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':209,'completed_depths':[100],'pending_bytes':0,'next_scope':'WP6_G1_TERMINAL_SCOPED_ADMISSION_AND_Q2_READINESS_REVIEW','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
