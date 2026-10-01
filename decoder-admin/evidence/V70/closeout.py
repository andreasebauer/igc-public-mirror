from pathlib import Path
import json,hashlib,shutil
D=Path(__file__).resolve().parent;P=D.parent/'v69b_preparation';R=Path('/tmp/ig_decoder_dev145_20261001');A=R/'decoder-admin';read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s=read(P/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);assert not read(P/'FINAL_006_POST_STATUS.json')['pending_objects'];assert read(D/'RESTORE_VERIFICATION.json')['attempt_bytes_identical']
collections=[read(p) for p in (w/'runtime/runs').rglob('collections/*.json')];nodes={n for r in collections for n in r['nodes']};selectors={n for r in collections for n in r['selectors']};reports=read(D/'NODE_REPORTS.json');reported={r['node'] for r in reports};result=read(D/'OBSERVED_RESULT.json');result.update(gate='V70_DEV145_FUNCTIONAL_INTERRUPTED_FAILURE',status='FAILED_UNQUALIFIED',capture_id=s['capture_id'],source_sha256=read(A/'CATALOG.json')['source']['source_sha256'],selected_files=167,collected_selectors=len(selectors),collected_nodes=len(nodes),collected_without_report=len(nodes-reported),uncollected_files=167-len(selectors),pending_checkpoint_roles=0,restore_workloads=0)
write(D/'V70_RESULT.json',result);write(D/'COLLECTION_RECEIPTS.json',collections)
rows=[]
for n,r in read(R/'decoder/tests/fixtures/preservation_rebind/DEV140_REVIEWED_CORE_CHANGES.json')['changes'].items():
 h=hashlib.sha256((R/'decoder'/n).read_bytes()).hexdigest()
 if h!=r['reviewed_sha256']:rows.append({'path':n,'reviewed_sha256':r['reviewed_sha256'],'actual_sha256':h})
write(D/'CORE_PIN_MISMATCHES.json',rows)
report='''V70 DEV145 FUNCTIONAL ATTEMPT — FAILED / INTERRUPTED / UNQUALIFIED — 2026-10-01

527 finished node reports have all three phases passed; one reported failure;
four node reports unfinished. These are incomplete observations, not a functional
PASS. Collection reconciliation below records missing/uncollected scope.
Failing test: test_change_preservation_rebind.py::
test_recovery_retains_reviewed_core_byte_boundaries.
Last dev140 reviewed pin record differs from actual controller, preservation
and validation-runtime files. Full source manifest still matches dev145 exactly.
The candidate itself is unchanged; the reviewed pin chain needs actual change
review before any new amendment. Never merely replace expected checksums.

STOP AND PRESERVATION
Operator stopped at the first observed unexplained failure. Namespace isolation
prevented direct PID signals; native session was ultimately stopped with Ctrl-C.
Tests continued during the stop attempts; final counts above include those reports.
Session exit130 preceded native finally/pause handling:attempt record remains
RUNNING although the process session ended and subsequent process observation
found no matching controller/validation workers. No native PAUSED state or
successful completion is claimed. Original stale record is intentionally retained.
No test rerun or automatic resume. Source/runtime post-check passed.
Checkpoint transport ran throughout. A temporary WORKSPACE_BUSY acknowledgment
was retried; no workload was retried. All final checkpoint roles acknowledged,
zero pending bytes/checkpoints. Native idle slim export saved/read back and
restored with identical attempt bytes, no completion and zero workload dispatch.
The saved recovery preserves interrupted evidence; it does not grant resume.

NEXT
Review diffs since dev140 for three mismatched core files and record why each
change is valid before a separate candidate amendment. Also correct the operator
launch/interrupt mechanism so a requested stop reaches native cleanup within
its process namespace. Preserve this failed capture and report every missing
node. Decide a separately preregistered qualification continuation only after
repair/review. Do not rerun V70 or advance to other groups automatically.
All23 RC rows remain OPEN. No science replay or release promotion.
Worktree /tmp/ig_decoder_dev145_20261001; runtime /tmp/ig_runtime_v55_fresh_20260930.
V69B capture c320632d07c8f0d6ad2631c44de777a16b32bcb17f7d50d9fffd18c8ef02ba39.
READBACKS.json includes native RAW/MULTIPART references for all preserved objects.
Use saved export plus verified objects with native restore_checkpoint; do not
repair the original record or dispatch its job. Private fixture bytes excluded
from public repository and compact bundle.
'''
(D/'READ_FIRST_V70_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
for name in ['READBACKS.json','STARTED.json','PREEXEC_RUNTIME.json','FINAL_006_POST_STATUS.json']:shutil.copy2(P/name,D/name)
for name in ['CAPTURE_SAVE_STATUS.json','INPUT_BINDINGS.json']:shutil.copy2(P/name,D/name)
l=read(A/'RC_LEDGER.json');l['latest_current_source_attempt']=result;write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV145 V70 FUNCTIONAL FAILED / INTERRUPTED — UNQUALIFIED\n527 passed node observations,1 failure,4 unfinished reports; incomplete suite.\nReviewed-core pin mismatch; original stale RUNNING record preserved, no completion.\nRecovery verified,0 workloads rerun,0 pending saves. Do not resume automatically.\nRead decoder-admin/evidence/V70_REPORT.txt and V70 evidence. All23 RC rows OPEN.\n')
e=A/'evidence/V70';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V70_2026-10-01.txt',A/'evidence/V70_REPORT.txt');print(json.dumps(result))
