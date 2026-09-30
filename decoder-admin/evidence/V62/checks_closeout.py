from pathlib import Path
import json,hashlib,shutil,sys
D=Path(__file__).parent;N=D.parent/'v62_interrupt';R=Path('/tmp/ig_decoder_dev144_20261001')
read=lambda p:json.loads(p.read_text())
identity=read(D/'SOURCE_IDENTITY.json');restore=read(N/'RESTORE_VERIFICATION.json');ack=read(N/'ACK_RESULT.json');life=read(N/'LIFECYCLE_RESULT.json');cap=read(N/'CAPTURE_SAVE_STATUS.json');export=read(N/'EXPORT_READBACK.json')
export['sha256']=hashlib.sha256(Path(export['readback']).read_bytes()).hexdigest();assert export['sha256']==read(N/'NATIVE_EXPORT.json')['sha256'];export['raw_readback_verified']=True;(N/'EXPORT_READBACK.json').write_text(json.dumps(export,indent=2))
assert read(D/'RESULT.json')['returncode']==0 and restore['runtime_bytes_and_modes_equal'] and life['lifecycle_pass']
sys.path.insert(0,str(N));from lifecycle_monitor import inventory
old=Path('/workspace/scratch/6d6f5c6d37c8/v61_interrupt');oldcap=read(old/'CAPTURE_SAVE_STATUS.json');assert inventory(Path(oldcap['workspace'])/'runtime')==read(old/'PAUSE_VERIFICATION.json')['runtime_inventory']
assert (Path('/tmp/ig_gate_v61_restored_20261001')/'runtime').is_dir()
summary={'gate':'V62','candidate':identity,'status':'PASS_BOUNDED_INTERRUPT_AND_EXACT_PAUSED_RECOVERY','standalone_passes':75,'standalone_failures':0,'new_regression_cases':9,'native_workload_outcome':'INTERRUPTED_NOT_TEST_PASS','capture_id':cap['capture_id'],'native_attempt':'PAUSED','accepted_completion':None,'lifecycle_observation_pass':True,'polling_interval_seconds':0.1,'late_write_observation_seconds':3,'exact_restore':restore,'save_roles_acknowledged':ack['obligations_acknowledged'],'unique_saved_readbacks':ack['readbacks_verified'],'pending_roles':len(ack['outbox']['pending_objects']),'export':export,'V61_original_runtime_unchanged':True,'V61_failure_preserved':True,'V55_failure_preserved':True,'full_RC':'OPEN','data_block_ready':False,'next':'Fresh dev144 corruption/controller-error lifecycle gate, then current-source fixtures/full RC/independent-host recovery.'}
(D/'V62_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
report='''V62 DEV144 RECOVERY REPAIR — BOUNDED PASS — 2026-10-01

Implemented checkpoint-bound file permission metadata for state and captured input
files, exact restoration with complete metadata validation, and refusal retention
before the paused checkpoint. A permission-only change creates a new checkpoint.
Legacy checkpoints restore without an invented original-permission guarantee.
Original execution errors survive refusal/checkpoint-save errors. V61 unchanged.

Focused verification: 75 PASS / 0 FAIL in 122.02 seconds, including nine new cases.
Covers mode roundtrip and mode-only changes; malformed/incomplete metadata refusal;
legacy limitation; real controller pause path with unit-only injected interruption;
refusal-write failure; existing preservation, publication, save-wave and cancellation.
These are standalone checks, not 75 new native-controller completions.

Fresh native case RC.INTERRUPT.V62.DEV144:
Ten capture prerequisites genuinely saved/read back before dispatch. One registered
long durable-progress selector, one worker, default preservation policy, no engine
deadline. One SIGINT sent through the owned controller Popen after live tick 1.
Controller returned KeyboardInterrupt, code -2, PAUSED, no accepted completion.
No descendant observed at return; no runtime change in the following three seconds.
Polling at 0.1 seconds can miss short-lived descendants; no absolute census claim.
Thirty checkpoint roles / fifteen unique objects saved with verified raw readbacks.
Zero pending roles/bytes/checkpoints. Saved pause export read back and hash-verified.
Fresh destination restored exactly: all 18 runtime file contents and rwx permissions,
including refusal record, identical PAUSED attempt, no accepted completion.
Restoration dispatched ZERO tests. The interrupted selector is not counted as PASS.

Source and runtime verified before/after. Reused reviewed Python 3.13 runtime;
not independent-host qualification. Permission guarantee covers recorded ordinary
rwx bits, not ownership, timestamps or extended ACLs. Checkpoint mode validation
rejects special-mode metadata. No science replay.

V61 remains FAILED_EXACT_PAUSED_RESTORATION; its original runtime is unchanged and
its failed restoration remains retained. V55 FAILED_LIFECYCLE_CLEANUP is unchanged.
Dev144 remains UNQUALIFIED overall. All 23 RC rows OPEN. Data blocks NOT ready.

NEXT
Fresh dev144 corruption/controller-error lifecycle case (V55 scenario successor),
with genuine live evidence, refusal, no survivor/late writes and exact forensic
recovery. Then current-source fixtures, full RC, findings dispositions and
independent-host qualification before source/runtime freeze and science replay.

'''
report+='SOURCE\n'+json.dumps(identity,indent=2)+'\nSource archive Drive ID '+read(D/'SOURCE_READBACK.json')['drive_id']+'\nNative capture '+cap['capture_id']+'\nPaused export SHA256 '+export['sha256']+'\nPaused export Drive ID '+export['drive_id']+'\n'
(D/'READ_FIRST_V62_DEV144_2026-10-01.txt').write_text(report)
e=R/'decoder-admin/evidence/V62';e.mkdir()
for prefix,base in [('checks',D),('native',N)]:
 for p in base.iterdir():
  if p.is_file() and p.suffix in ('.json','.txt','.py','.xml','.jsonl'):shutil.copy2(p,e/(prefix+'_'+p.name))
shutil.copy2(D/'READ_FIRST_V62_DEV144_2026-10-01.txt',R/'decoder-admin/evidence/V62_REPORT.txt')
ledger=read(R/'decoder-admin/RC_LEDGER.json');assert len(ledger['rows'])==23 and all(x['rc_acceptance_status']=='OPEN' for x in ledger['rows'])
ledger['latest_interrupt_recovery_gate']=summary
ledger['latest_candidate'].update(tests_executed=75,status='UNQUALIFIED_BOUNDED_V62_PASS',drive_id=read(D/'SOURCE_READBACK.json')['drive_id'],reason='Fresh corruption scenario, fixtures, full RC and independent-host recovery remain OPEN')
ledger['latest_source_repair'].update(scope='checkpoint state file modes and paused refusal ordering',status='BOUNDED_V62_PASS',native_tests_executed=0,native_interrupt_scenarios=1)
(R/'decoder-admin/RC_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
catalog=read(R/'decoder-admin/CATALOG.json');catalog['source']={**identity,'drive_id':read(D/'SOURCE_READBACK.json')['drive_id']};(R/'decoder-admin/CATALOG.json').write_text(json.dumps(catalog,indent=2)+'\n')
(R/'DECODER_READ_FIRST.txt').write_text('DEV144 V62 INTERRUPT/EXACT RECOVERY BOUNDED PASS; OVERALL UNQUALIFIED\n75 focused PASS; fresh native interrupt and exact saved paused restoration; zero recovery dispatch.\nAll 23 RC rows OPEN. Fresh corruption scenario next. No science replay.\nRead decoder-admin/evidence/V62_REPORT.txt and RC_LEDGER.json.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
(R/'decoder-import/VERIFICATION.txt').write_text('V62 source bytes verified; 75 standalone PASS and fresh native interrupt/exact paused recovery.\nOverall qualification not granted.\n'+json.dumps(identity,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'runtime_files':len(read(N/'PAUSE_VERIFICATION.json')['runtime_inventory']),'standalone':75,'pending_roles':summary['pending_roles']}))
