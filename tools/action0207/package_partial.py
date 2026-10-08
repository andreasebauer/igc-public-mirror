"""Preserve an incomplete registration honestly after its explicit resource stop."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='PASS_COLD_PARTIAL97_98_NATIVE_IDENTITIES_AND_DAG_ROUNDTRIP' and a['completed_depths']==[97,98] and a['pending_bytes']==0 and not a['depth99_committed']
last=a['rows'][-1];meta=json.loads((B/'BOOTSTRAP98_META.json').read_text());boot=Path(meta['path']);assert hashlib.file_digest(boot.open('rb'),'sha256').hexdigest()==meta['sha256']
report=f'''CHECKPOINT0207 — PARTIAL97–98; REGISTERED TIME-BOUND STOP AT99
2026-10-08

NOT A COMPLETED97–99 TRANCHE. Native controller stopped on STREAM_WORK_BOUND_EXCEEDED:g1_partition_depth_99. The frozen handler declared max_work_seconds=300.0. Preserve that resource failure as recorded; it is not a terminal scientific comparison failure, and no campaign PASS is claimed.

COMMITTED: depths97–98,386candidate builds,24roots per depth; both native manifests published. Depth99started but has zero committed generation tasks and zero stored states; no result or manifest. Build count of its uncommitted attempt is UNKNOWN. Earlier depths reused; Master151unchanged; zero admissions. Terminal100comparison/Q2pending.

Capture raw objects saved and read back exactly before execution. Pinned runtime admission passed using python-fixed-host. Scientific adapter code byte-identical to0206. No resource budget or science changed inside the failed capture.

Failed-run checkpoint snapshot saved. Every raw dependency verified before native acknowledgement. Both committed DAGs independently reconstructed and reserialized exactly from a fresh cold checkpoint. Exact native identity bytes/state payloads, parent science bindings and full-node equality checked. Depth99absence of committed states/tasks confirmed again in cold tree. Audit produced no candidates and used no original-workspace scientific state. Pending bytes0.

Depth98 reachable nodes:{last['nodes']}; roots24.
Depth98 science SHA256:{last['science_sha256']}
BOOTSTRAP98 SHA256:{meta['sha256']} (included).
Runtime: CPython3.13.5 / SQLite3.51.3 / Decoder0.8.0.dev151+lib. Launch only using python-fixed-host.64MiB result budget unchanged;193recipes/31bridgepairs/24-state beam unchanged.

NEXT: review the operational300-second task budget and explicitly register missing99from verified BOOTSTRAP98under a revised budget. Retain failed0207capture and all input/task bindings as provenance. DO NOT regenerate97–98 or silently alter the frozen registration. After verified99, separately register100and compare193terminal public interfaces before any admission review. Current continuation handler stops at99; terminal route requires separate registration.

RESTORE: CHECKPOINT_EXPORT.json lists exact raw dependencies. READBACKS.json maps digests to saved Drive IDs; verify raw bytes/length before restore_current.py. Predecessor0206handoff contains pinned runtime recovery metadata. Resolve relocated paths only to hash-identical files and prepare a fresh capture specification. audit_restore.py and package.py describe the ORIGINAL planned97–99scope; audit_partial.py and this report describe the actual saved partial scope. Verify MANIFEST.json.
'''
(B/'REPORT_PARTIAL.txt').write_text(report)
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ['.json','.txt','.zip','.log','.py'] and not any(v in p.parts for v in ['restore_objects','__pycache__']) and not p.name.startswith('private_') and p.name not in ['DELIVERABLES.json','SAVE_RECEIPT.json']:files['continuation0207/'+str(p.relative_to(B))]=p
files['BOOTSTRAP98.json']=boot
files['predecessor/IG_MASTER151_G1_PARTITION_CONTINUATION_0206_HANDOFF_2026-10-08.zip']=W/'IG_MASTER151_G1_PARTITION_CONTINUATION_0206_HANDOFF_2026-10-08.zip'
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()}
out=W/'IG_MASTER151_G1_PARTIAL_0207_TIME_BOUND_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
result={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':207,'completed_depths':[97,98],'depth99_committed':False,'pending_bytes':0,'next_missing_depth':99,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
