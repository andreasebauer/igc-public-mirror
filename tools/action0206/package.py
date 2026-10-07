from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());r=json.loads((B/'NATIVE_RESULT.json').read_text());assert a['status']=='PASS_COLD_NATIVE_CHECKPOINT_AND_PARTITION_DAG_RESTORE' and r['status']=='COMPLETED' and a['completed_depths']==[95,96] and a['pending_bytes']==0
last=a['rows'][-1];meta=json.loads((B/'BOOTSTRAP96_META.json').read_text());boot=Path(meta['path']);assert hashlib.file_digest(boot.open('rb'),'sha256').hexdigest()==meta['sha256']
report=f"""CHECKPOINT0206 — EXACT G1 CONTINUATION95–96 COMPLETE
2026-10-08

Registered native continuation95–96 completed:386candidate builds,24roots per depth. Earlier depths reused, not regenerated. Master151unchanged; zero admissions. Terminal100comparison/Q2pending.

Bootstrap94 came from independent saved-state reconstruction and exact DAG roundtrip gate. Initial native launch refused before generation because the pinned Python binary loaded system SQLite; refusal preserved in FIRST_ATTEMPT_RUNTIME_ERROR.txt. Same captured job retried using python-fixed-host after explicit pinned runtime admission passed. No gate bypass, scientific input change, or completed-task rerun.

Both saved DAGs independently reconstructed and reserialized exactly from a fresh cold checkpoint. Every raw dependency verified; exact native canonical identity bytes and stored state payloads compared. Full-node equality checks retained. All native save obligations acknowledged after real raw Drive readbacks; pending bytes0. Audit generated no candidates and read no original-workspace scientific state.

Depth96 reachable nodes:{last['nodes']}; roots24.
Depth96 science SHA256:{last['science_sha256']}
BOOTSTRAP96 SHA256:{meta['sha256']} (included).
Runtime: CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib. Launch only with python-fixed-host.64MiB result budget preserved;193recipes/31bridgepairs/24-state beam unchanged.

NEXT: Register isolated bounded97–99 from verified BOOTSTRAP96. Fully preserve/audit before separately registering100and independently comparing193terminal public interfaces. No terminal campaign PASS or admission claimed here.

RESTORE: CHECKPOINT_EXPORT.json lists raw dependencies; READBACKS.json maps hashes to Drive IDs. Recover pinned runtime using included predecessor handoff metadata. Verify raw hash/length before native restore_current.py. Absolute paths are transport locators: resolve only hash-identical restored files and prepare a fresh capture spec. Original paused0205capture remains provenance. Verify MANIFEST.json.
"""
(B/'REPORT.txt').write_text(report)
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in p.parts for v in ['restore_objects','__pycache__']) and not p.name.startswith('private_'):files['continuation0206/'+str(p.relative_to(B))]=p
files['BOOTSTRAP96.json']=boot
files['predecessor/IG_MASTER151_G1_PARTITION_CONTINUATION_0205_HANDOFF_2026-10-08.zip']=W/'recovery/IG_MASTER151_G1_PARTITION_CONTINUATION_0205_HANDOFF_2026-10-08.zip'
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()}
out=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0206_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
result={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':206,'completed_depths':[95,96],'pending_bytes':0,'next_depths':[97,99],'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
