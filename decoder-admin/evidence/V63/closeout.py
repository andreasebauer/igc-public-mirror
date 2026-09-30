from pathlib import Path
import json,hashlib,sys,shutil
D=Path(__file__).parent;R=Path('/tmp/ig_decoder_dev144_20261001');read=lambda p:json.loads(p.read_text())
cap=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(cap['workspace']);inj=read(D/'INJECTION_DONE.json');expected='OUTBOX_OBJECT_MISMATCH:'+inj['original_sha256'];life=read(D/'LIFECYCLE_RESULT.json');recovery=read(D/'FORENSIC_RESTORE_VERIFICATION.json')
assert read(D/'RUN_EXCEPTION.json')['reason']==expected and read(D/'FULL_STATUS_REFUSAL.json')['reason']==expected
assert life['lifecycle_pass'] and not life['survivors_at_controller_return'] and not life['late_runtime_changes']
assert recovery['native_refusal']==expected and recovery['workload_executions']==0
assert read(w/'durability/CHECKPOINT_FAILURE.json')['reason']==expected
attempts=[read(p) for p in (w/'runtime/attempts').rglob('*.json')];assert len(attempts)==1 and attempts[0]['status']=='PAUSED'
assert not list((w/'runtime/intake/completed').glob('*.json')) and not list((w/'runtime/intake/prepared_completions').glob('*.json'))
assert hashlib.sha256(Path(inj['target']).read_bytes()).hexdigest()==inj['damaged_sha256']
from lifecycle_monitor import inventory
assert inventory(w)==read(D/'FORENSIC_VOLUMES.json')['files']
for n,h in read(D/'OPERATOR_SCRIPT_HASHES.json').items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
assert read(D/'PRE_RUNTIME.json')==read(D/'POST_RUNTIME.json')
baseline=read(D/'BASELINE_READY.json');assert baseline['healthy_metadata_checks']>=10 and baseline['span_seconds']>=3
result={'gate':'V63','status':'PASS_BOUNDED_LIVE_CORRUPTION_CLEANUP_FORENSIC_RECOVERY','candidate':'0.8.0.dev144+lib','source_sha256':'2b99ebb20575b2b43043812fdb1d39a2c682b5ff0081d38af2ddab4e7e9b177e','capture_id':cap['capture_id'],'baseline':baseline,'injection':inj,'controller_reason':expected,'attempt':'PAUSED','accepted_completion':None,'checkpoint_save_status':'CHECKPOINT_UNAVAILABLE_EXPECTED_CORRUPT_OUTBOX','native_save_closure':'NOT_CLAIMED','pending_roles':'UNRESOLVED_NATIVE_STATUS_REFUSES','lifecycle_pass':True,'poll_interval_seconds':0.25,'late_runtime_changes':False,'forensic_recovery':recovery,'new_completed_test_passes':0,'full_RC':'OPEN','data_block_ready':False,'prior_V55_failure':'UNCHANGED','prior_V61_failure':'UNCHANGED','next':'Current-source fixture completion/rebinding and full RC matrix; remaining finding dispositions and independent-host recovery.'}
(D/'V63_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
report='''V63 DEV144 LIVE CORRUPTION RELIABILITY — BOUNDED PASS — 2026-10-01

Fresh native RC.CORRUPTION.V63.DEV144. Source unchanged from V62.
Ten capture prerequisites genuinely saved/read back before execution.
Three operator mock cases verified bounded admission retries, stop on unexpected
error, and one launch/registration. They are not decoder test results.

Healthy baseline: 13 metadata checks over 3.250 seconds following a completed
checkpoint; default budgets/reserve verified, zero observed payload reads in those
metadata checks. This is Python-level instrumentation, not physical disk measurement.
Original checkpoint-state object was genuinely saved/read back before injection.
Two live checkpoints were acknowledged while the selector remained unfinished.
One first-byte flip under the native outbox lock; length unchanged, no other target.
Full native status refused the exact object. The controller later refused the same
object at its payload-verifying safe-point boundary, paused, and retained no accepted
completion. Secondary paused checkpoint also refused the corrupt object and recorded
CHECKPOINT_UNAVAILABLE. We do not claim native save closure or zero pending roles.

Lifecycle: no sampled descendants at controller return; no late runtime changes;
quiescence and a further stable full-workspace observation verified. Polling interval
0.25 seconds can miss short-lived descendants; this is not an absolute census.
No external process killer or new supervisor was added.

Forensic recovery: five separately saved and raw-readback-verified ZIP volumes
reconstructed all 1617 files with exact bytes and ordinary permission bits. The
isolated restored copy reproduced the same OUTBOX_OBJECT_MISMATCH, with ZERO
workload executions. Original damaged object and workspace remain unchanged.
This is forensic recovery of deliberately damaged evidence, not a successful native
checkpoint export. V62 separately established healthy exact paused restoration.

Source/runtime verified before and after. Existing reviewed runtime reused;
independent-host qualification remains open. No science replay and no additional
completed-test PASS credit. V55 and V61 remain their original failed cases.
Dev144 remains UNQUALIFIED overall; all 23 RC acceptance rows OPEN.

NEXT
Current-source fixture completion/rebinding and full functional/workspace/
representative/timing RC matrix; remaining finding dispositions, dependency lock,
and independent-host recovery before source/runtime freeze and scientific inputs.
Do not rerun V63 or repair its damaged original. Use new captures for later gates.

'''
report+='Capture: '+cap['capture_id']+'\nSource: '+result['source_sha256']+'\nExpected refusal: '+expected+'\nForensic restored path: /tmp/ig_v63_forensic_restored_20261001\nRepository: /tmp/ig_decoder_dev144_20261001\nEvidence: '+str(D)+'\n'
(D/'READ_FIRST_V63_RELIABILITY_PASS_2026-10-01.txt').write_text(report)
e=R/'decoder-admin/evidence/V63';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in ('.txt','.json','.py','.js','.jsonl'):shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V63_RELIABILITY_PASS_2026-10-01.txt',R/'decoder-admin/evidence/V63_REPORT.txt')
ledger=read(R/'decoder-admin/RC_LEDGER.json');assert len(ledger['rows'])==23 and all(x['rc_acceptance_status']=='OPEN' for x in ledger['rows'])
ledger['latest_live_fault_gate_v63']=result
ledger['latest_candidate'].update(status='UNQUALIFIED_BOUNDED_V62_V63_PASS',reason='Current-source fixtures, full RC, finding dispositions and independent-host recovery remain OPEN')
(R/'decoder-admin/RC_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
(R/'DECODER_READ_FIRST.txt').write_text('DEV144 V62 + V63 BOUNDED PASS; OVERALL UNQUALIFIED\n75 focused PASS; V62 native interrupt/exact paused recovery; V63 live-corruption refusal/cleanup/exact forensic recovery.\nAll 23 RC rows OPEN. Next: current-source fixtures and full RC matrix. No science replay.\nRead decoder-admin/evidence/V63_REPORT.txt and RC_LEDGER.json.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
print(json.dumps({'status':result['status'],'files':recovery['exact_files_and_modes'],'recovery_workloads':0}))
