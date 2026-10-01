from pathlib import Path
import json,shutil
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev145_20261001');A=R/'decoder-admin';read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);assert not list((w/'runtime/attempts').rglob('*.json'));assert not read(D/'CAPTURE_ACK.json')['pending_objects'];assert read(D/'POST_SOURCE.txt')['status']=='BYTE_EXACT_SOURCE_PASS';assert read(D/'POST_RUNTIME.txt')['status']=='BYTES_AND_MODES_VERIFIED'
result={'gate':'V69B_DEV145_FUNCTIONAL_PREPARATION','status':'SAVED_CAPTURE_PREPARED_NOT_EXECUTED','capture_id':s['capture_id'],'job_id':s['job_id'],'source_sha256':read(A/'CATALOG.json')['source']['source_sha256'],'input_roles_verified':7,'active_selectors':170,'functional_selectors':167,'capture_save_roles_confirmed':19,'raw_readbacks':14,'multipart_readbacks':5,'pending_capture_roles':0,'native_attempts':0,'tests_executed':0,'full_rc':'OPEN','rejected_preparation':'V69 protocol text said default300 seconds; actual default30. Capture preserved unused; corrected in fresh V69B.'};write(D/'V69B_RESULT.json',result)
report='''V69B DEV145 FUNCTIONAL CAPTURE — PREPARED, NOT EXECUTED — 2026-10-01

All seven input roles verified by exact archive size/hash:five earned dev145
fixtures and two immutable historical fixtures. Private failed_evidence bytes
are excluded from the public repository and this compact bundle.
170 distinct selectors exactly cover active test files:167 functional,1 workspace,
1 representative,1 timing. Native expanded node inventory is not yet earned.

Corrected functional capture is saved and acknowledged:19 obligations comprising
14 raw objects and five verified native multipart transports. Actual manifest
readbacks and previously saved part readbacks were checked by native APIs.
Zero pending capture obligations, zero attempts, zero tests executed.
Source1526 files and runtime4721 files/modes plus seven host files verified.
4workers,2GiB memory,8GiB workspace budget; no automatic runtime deadline.
Preservation default30seconds and other native limits unchanged.

RETAINED PREPARATION ERROR
Initial V69 protocol text incorrectly described the default interval as300s.
Preflight caught this before saving prerequisites or dispatch. Original capture
and script are retained unexecuted and unacknowledged; never dispatch it.
Fresh V69B corrected only preparation text/identity; frozen dev145 source unchanged.

NEXT — EXECUTE SAVED V69B FUNCTIONAL CAPTURE ONCE
Directory /workspace/scratch/a2e5e2576f17/v69b_preparation
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; PYTHONOPTIMIZE=0,-B.
Launch run_functional.py through a nonblocking exec session while continuously
servicing preservation with transport_admin.py status TAG / ack TAG.
READBACKS.json seeds RAW and MULTIPART objects already saved. Upload/read back
new pending objects, append exact native readback references, acknowledge by
role-aware batch. Do not use a raw receipt to mislabel a multipart archive.
Repeated status scans must not dispatch tests. Native limits remain fail-closed.
Observe progress/backlog and keep save waves running throughout the long gate.
Reconcile every native collection receipt and final phase outcome. Skips, errors,
interruptions, missing/unexecuted nodes are not PASS. Retain first failed group;
no automatic retry/resume or later group after unexplained failure.
Then workspace, representative, isolated one-worker timing last.
All23 RC rows remain OPEN. No science replay or release promotion.

RECOVERY
Clone published repository branch; verify source/runtime via decoder-admin.
All seven inputs and five native multipart manifests are pinned in this bundle.
CAPTURE_SAVE_STATUS and SAVE_CATALOG retain saved object identities. Restore
capture using original saved artifacts; never regenerate a different capture
and claim it is this prepared one. The current saved capture is local and intact.
'''
(D/'READ_FIRST_V69B_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
l=read(A/'RC_LEDGER.json');l['latest_preparation_gate']=result;write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV145 V69B FUNCTIONAL CAPTURE PREPARED — NOT EXECUTED\nAll five candidate fixtures earned; all seven inputs verified. 19 capture save obligations confirmed.\nNext: execute saved V69B 167-selector functional job once with continuous live save waves.\nInitial V69 protocol typo retained in unused capture; never dispatch V69.\nRead decoder-admin/evidence/V69B_REPORT.txt and V69B evidence. All23 RC rows OPEN.\n')
e=A/'evidence/V69B';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
old=D.parent/'v69_preparation';rej=e/'rejected_V69';rej.mkdir()
for n in ['PREREGISTRATION.txt','prepare_capture.py','CAPTURE_SPEC.json','CAPTURE_SAVE_STATUS.json','REJECTED_PREPARATION.json']:shutil.copy2(old/n,rej/n)
shutil.copy2(D/'READ_FIRST_V69B_2026-10-01.txt',A/'evidence/V69B_REPORT.txt');print(json.dumps(result))
