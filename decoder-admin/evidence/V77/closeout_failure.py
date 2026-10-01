from pathlib import Path
import json,shutil,hashlib,subprocess
D=Path(__file__).resolve().parent;G=D/'functional';R=Path('/tmp/ig_decoder_dev146_20261001');A=R/'decoder-admin';read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not read(G/'EXPORT_013_POST_STATUS.json')['pending_objects'];rest=read(G/'RESTORE_VERIFICATION.json');assert rest['files_checked']==889 and rest['recovery_workloads_dispatched']==0
rec=read(G/'RECONCILIATION.json');assert rec['status']=='FAILED_INCOMPLETE' and rec['all_phases_passed']==583 and rec['nonpass_nodes']==0
p=subprocess.run(['python3','-B',str(R/'decoder-import/verify_source.py')],capture_output=True,text=True);assert p.returncode==0;write(G/'POST_SOURCE.json',json.loads(p.stdout))
ids=set()
for p in G.glob('*_ACK.json'):
 x=read(p)
 if x.get('schema_id')=='IG_CHECKPOINT_ACK_COMPLETE_V1':
  batch=read(G/(p.name.replace('_ACK.json','_BATCH.json')));ids.update(r['obligation_id'] for r in batch['obligations'])
result={'gate':'V77_DEV146_FULL_QUALIFICATION_ATTEMPT','status':'FAILED_INCOMPLETE_INFRASTRUCTURE_LOCK_CONTENTION','source_sha256':'930d6b4287782687891a57e034f7da69ab75a6b3b2426b20d85da36c64186137','capture_id':read(G/'CAPTURE_SAVE_STATUS.json')['capture_id'],'native_attempts':1,'native_status':'PAUSED','reason':'ControllerLoopError:WORKSPACE_BUSY','native_completions':0,'functional_registered_files':167,'functional_collected_files':65,'functional_collected_nodes':594,'fully_passed_nodes':583,'reported_test_failures_or_skips':0,'unfinished_node_reports':4,'collected_without_report':7,'uncollected_functional_files':102,'other_groups':'NOT_STARTED','v70_core_pin_regression':'PASS','checkpoint_roles_acknowledged':len(ids),'pending_checkpoint_roles':0,'restore_files_exact_bytes_modes':889,'recovery_workloads_dispatched':0,'export_sha256':read(G/'EXPORT.json')['sha256'],'export_drive_id':read(G/'EXPORT_READBACK.json')['drive_id'],'source_unchanged':True,'full_rc':'OPEN','next':'Review narrow durability-lock serialization repair on new candidate; deterministic contention regression and native live-save proof before another full qualification attempt.'};write(D/'V77_RESULT.json',result)
report='''V77 DEV146 FULL QUALIFICATION ATTEMPT — FAILED/INCOMPLETE — 2026-10-01

BIGGER STEP EXECUTED
All seven exact fixture inputs verified;170 files registered in four ordered
groups. Functional167 files dispatched once with4 workers. Other groups were
planned to run automatically only after prior group passed. No routine pause or
user confirmation stopped the campaign.

OBSERVATION
583 nodes fully passed all three phases;0 failed/skipped reports.65 functional
files collected594 nodes;587 have reports,4 unfinished and7 not yet reported.
102 functional files and all3 later groups remain unexecuted. V70's previously
failing reviewed-core-pin test passed under native dev146 execution.
The controller aborted on WORKSPACE_BUSY while making a progress checkpoint.
It retained PAUSED attempt/refusal. Its immediate paused-checkpoint attempt also
failed WORKSPACE_BUSY; CHECKPOINT_FAILURE.json is retained. No completion exists.
No pause was requested by the report monitor because no test non-pass appeared.
This is an incomplete failed campaign, not a successful583-test qualification.

LOCK TRIAGE
preservation.make_checkpoint and preservation_batch.confirm_batch both acquire
the same durability/outbox .runner.lock through the nonblocking _workspace_lock.
The native stack proves checkpoint lock acquisition failed. The54-role batch
journal has receipt timestamps overlapping the failure (LOCK_TRIAGE.json).
This supports save-acknowledgment/checkpoint contention; lock owner was not traced
independently. Top-level workload exclusion must remain fail-fast. The narrow
repair is to serialize controller checkpoint writes with legitimate acknowledgment
work, with deterministic cross-process tests and native live-save qualification.
Do not swallow every WORKSPACE_BUSY, relax backlog limits or retry the old job.

PRESERVATION
All22 capture roles saved/read back before dispatch (legacy helper prints19;
actual pending-role count was22 and confirmation covered every role).
All checkpoint roles drained after exit. Later idle export captures the unchanged
failed attempt and refusal; it does not retroactively make the immediate paused
checkpoint succeed. Saved export raw readback restored889 state/input files with
exact bytes and ordinary rwx modes; same PAUSED attempt/refusal, no completion,
zero recovery workloads. Source and reviewed runtime passed verification.
One administrative acknowledgment poll returned a still-running session; it was
waited to successful completion, not retried. No workload rerun or source change.

NEXT / RC POSITION
Five dev146 fixtures remain earned but dev146 full qualification is NOT passed.
Prepare a reviewed new candidate for the durability-lock repair, update exact
core-pin successor if applicable, save/read back candidate/protocol before tests,
prove contention handling and live saves, then reconcile exact-candidate fixture
requirements and retry full qualification on a fresh capture. Never resume V77.
All23 RC acceptance rows OPEN. V75 RC roadmap still applies:full profile, finding
dispositions, independent remote-only recovery, final matrix and explicit promotion.
No science replay or release claim. Private failed_evidence bytes are excluded
from public artifacts; raw node output is kept in the private saved bundle.
Worktree /tmp/ig_decoder_dev146_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host;PYTHONOPTIMIZE=0,-B.
Failure restore /tmp/ig_v77_failure_restored_20261001
'''
(D/'READ_FIRST_V77_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
l=read(A/'RC_LEDGER.json');assert len(l['rows'])==23 and all(r['rc_acceptance_status']=='OPEN' for r in l['rows']);l['latest_current_source_attempt']=result;l['latest_failure_triage']=read(G/'LOCK_TRIAGE.json');l['current_rc_roadmap'].update(as_of_gate='V77',full_candidate_qualification='FAILED_INCOMPLETE_LOCK_CONTENTION',next=result['next']);write(A/'RC_LEDGER.json',l)
c=read(A/'CATALOG.json');c.setdefault('qualification_attempts',[]).append(result);write(A/'CATALOG.json',c)
(R/'DECODER_READ_FIRST.txt').write_text('DEV146 V77 FULL QUALIFICATION FAILED/INCOMPLETE — WORKSPACE_BUSY\n583 fully passed nodes,0 reported test failures/skips;controller checkpoint lock contention stopped run.\nFailed PAUSED capture restored889 files exactly,zero recovery workloads. Do not resume V77.\nNext:narrow durability-lock repair and contention/native-save proof on new candidate.\nRead decoder-admin/evidence/V77_REPORT.txt.All23 RC rows OPEN;no release promotion.\n')
e=A/'evidence/V77';e.mkdir();(e/'functional').mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
for p in G.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'} and p.name not in {'NODE_REPORTS.json','NATIVE_CONSOLE.txt'}:shutil.copy2(p,e/'functional'/p.name)
shutil.copy2(D/'READ_FIRST_V77_2026-10-01.txt',A/'evidence/V77_REPORT.txt');print(json.dumps(result))
