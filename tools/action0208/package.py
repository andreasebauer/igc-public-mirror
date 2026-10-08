from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());r=json.loads((B/'NATIVE_RESULT.json').read_text());assert a['status']=='PASS_COLD_NATIVE_CHECKPOINT_AND_PARTITION_DAG_RESTORE' and r['status']=='COMPLETED' and a['completed_depths']==[99] and a['pending_bytes']==0
last=a['rows'][-1];meta=json.loads((B/'BOOTSTRAP99_META.json').read_text());boot=Path(meta['path']);assert hashlib.file_digest(boot.open('rb'),'sha256').hexdigest()==meta['sha256']
report=f'''CHECKPOINT0208 — EXACT G1 DEPTH99 COMPLETE
2026-10-08

Registered native depth99 completed:193candidate builds,24roots. Depths97–98 reused from verified bootstrap98. Master151 unchanged; zero admissions. Terminal100 and independent193-interface comparison remain pending.

The frozen0207 capture ended at its300-second work bound after committing97–98, with no committed99. That failed registration remains preserved as provenance. New isolated0208 capture explicitly binds a900-second task budget. Only the operational handler budget binding changed; engine,193recipes,31bridgepairs,24-state beam and64MiB result budget unchanged. OPERATIONAL_AMENDMENT.json contains the exact diff.

Saved depth99 DAG independently reconstructed and reserialized exactly from a fresh cold checkpoint. Every raw dependency verified; native canonical identity bytes and stored payload compared. All native save obligations acknowledged after real raw Drive readbacks; pending bytes0. Audit generated no candidates and read no original-workspace scientific state.

Depth99 reachable nodes:{last['nodes']}; roots24.
Depth99 science SHA256:{last['science_sha256']}
BOOTSTRAP99 SHA256:{meta['sha256']} (included).
Runtime: CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib. Launch only with python-fixed-host.

NEXT: Separately register terminal100 from verified BOOTSTRAP99 and independently compare all193terminal public interfaces before any admission. No terminal campaign PASS, Q2 result, G2 graduation or full campaign completion claimed.

RESTORE: CHECKPOINT_EXPORT.json lists raw dependencies; READBACKS.json maps hashes to Drive IDs. Included predecessor handoff retains runtime/recovery provenance. Verify raw hash/length before native restore_current.py. Absolute paths are transport locators: resolve only hash-identical files and prepare a fresh capture spec. Verify MANIFEST.json. CODE_MIRROR.json binds public Python source only.
'''
(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'in_flight_active':'NONE','active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'next_scope_status':'DEPTH99_COLD_VERIFIED;TERMINAL100_AND193_INTERFACE_COMPARISON_PENDING','generator_calls_scope':'0208:one committed evaluator task/193candidate builds; earlier committed depths reused','latest_checkpoint':208,'current_phase':'G1_DEPTH99_COLD_AUDITED','completed_depth':99,'master_slices':151,'new_admissions':0,'pending_bytes':0,'generator_calls':1,'candidate_build_calls':193,'latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'next_scope':'WP6_G1_REGISTER_TERMINAL100_FROM99_AND193_INTERFACE_COMPARISON','latest_audit':a,'latest_bootstrap':meta});(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2));(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2))
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in p.parts for v in ['restore_objects','__pycache__']) and not p.name.startswith('private_') and p.name not in ['DELIVERABLES.json','SAVE_RECEIPT.json']:files['continuation0208/'+str(p.relative_to(B))]=p
files['BOOTSTRAP99.json']=boot
files['predecessor/IG_MASTER151_G1_PARTIAL_0207_TIME_BOUND_HANDOFF_2026-10-08.zip']=W/'IG_MASTER151_G1_PARTIAL_0207_TIME_BOUND_HANDOFF_2026-10-08.zip'
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()}
out=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0208_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
result={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':208,'completed_depths':[99],'pending_bytes':0,'next_depth':100,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
