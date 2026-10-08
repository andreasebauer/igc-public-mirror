from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());r=json.loads((B/'NATIVE_RESULT.json').read_text());assert a['status']=='PASS_COLD_NATIVE_CHECKPOINT_AND_PARTITION_DAG_RESTORE' and r['status']=='COMPLETED' and a['completed_depths']==[97,99] and a['pending_bytes']==0
last=a['rows'][-1];meta=json.loads((B/'BOOTSTRAP99_META.json').read_text());boot=Path(meta['path']);assert hashlib.file_digest(boot.open('rb'),'sha256').hexdigest()==meta['sha256']
report=f"""CHECKPOINT0207 — EXACT G1 CONTINUATION97–99 COMPLETE
2026-10-08

Registered native continuation97–99 completed:579candidate builds,24roots per depth. Earlier depths reused, not regenerated. Master151unchanged; zero admissions. Terminal100comparison/Q2pending.

Bootstrap96 came from the preserved0206 cold native identity and exact DAG roundtrip audit. All functional adapter files unchanged. Pinned runtime admission passed using python-fixed-host before generation.

All three saved DAGs independently reconstructed and reserialized exactly from a fresh cold checkpoint. Every raw dependency verified; exact native canonical identity bytes and stored state payloads compared. Full-node equality checks retained. All native save obligations acknowledged after real raw Drive readbacks; pending bytes0. Audit generated no candidates and read no original-workspace scientific state.

Depth99 reachable nodes:{last['nodes']}; roots24.
Depth99 science SHA256:{last['science_sha256']}
BOOTSTRAP99 SHA256:{meta['sha256']} (included).
Runtime: CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib. Launch only with python-fixed-host.64MiB result budget preserved;193recipes/31bridgepairs/24-state beam unchanged.

NEXT: Separately register terminal100 from verified BOOTSTRAP99, preserving the bound193recipes and public reference. Independently compare193terminal public interfaces before any admission review. The current continuation handler stops at99; a separate frozen terminal route is required. No terminal campaign PASS or admission claimed here.

RESTORE: CHECKPOINT_EXPORT.json lists raw dependencies; READBACKS.json maps hashes to Drive IDs. Recover pinned runtime using included predecessor handoff metadata. Verify raw hash/length before native restore_current.py. Absolute paths are transport locators: resolve only hash-identical restored files and prepare a fresh capture spec. Original paused0205capture remains provenance. Verify MANIFEST.json.
"""
(B/'REPORT.txt').write_text(report)
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in p.parts for v in ['restore_objects','__pycache__']) and not p.name.startswith('private_') and p.name not in ['DELIVERABLES.json','SAVE_RECEIPT.json']:files['continuation0207/'+str(p.relative_to(B))]=p
files['BOOTSTRAP99.json']=boot
files['predecessor/IG_MASTER151_G1_PARTITION_CONTINUATION_0206_HANDOFF_2026-10-08.zip']=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0206_HANDOFF_2026-10-08.zip'
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()}
out=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0207_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
result={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':207,'completed_depths':[97,99],'pending_bytes':0,'next_depths':[100,100],'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
