from pathlib import Path
import json,hashlib,shutil
D=Path(__file__).parent;N=D.parent/'v64_native';R=Path('/tmp/ig_decoder_dev145_20261001');read=lambda p:json.loads(p.read_text())
id=read(D/'SOURCE_IDENTITY.json');done=read(N/'VERIFIED_COMPLETION.json');restore=read(N/'RESTORE_VERIFICATION.json');status=read(N/'TERMINAL_STATUS.json');nodes=done['result']['result']['nodes'];assert len(nodes)==18 and all(x['status']=='PASS' for x in nodes)
assert restore['exact_completion_equal'] and restore['tests_dispatched']==0 and not status['pending_objects']
export=read(N/'EXPORT_READBACK.json');export['sha256']=hashlib.sha256(Path(export['readback']).read_bytes()).hexdigest();assert export['sha256']==read(N/'NATIVE_EXPORT.json')['sha256'];export['raw_readback_verified']=True;(N/'EXPORT_READBACK.json').write_text(json.dumps(export,indent=2))
acks=[read(p) for p in N.glob('LIVE_*_ACK.json')]
result={'gate':'V64','status':'PASS_BOUNDED_NATIVE_PROFILE_AND_RECOVERY','candidate':id,'native_passes':18,'native_failures':0,'native_skips':0,'standalone_passes':4,'distinct_count_note':'Four standalone qualification cases overlap the 18 native cases; do not sum to 22.','active_test_files':170,'registered_test_files':170,'engine_implementation_changed':False,'test_bodies_changed':False,'capture_id':read(N/'CAPTURE_SAVE_STATUS.json')['capture_id'],'completion_sha256':done['completion_sha256'],'checkpoint_roles_acknowledged':sum(x['obligations_acknowledged'] for x in acks),'unique_saved_objects':len(read(N/'CATALOG.json')),'pending_roles':0,'exact_recovery':restore,'export':export,'full_RC':'OPEN','candidate_fixtures_remaining':5,'data_block_ready':False}
(D/'V64_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
report='''V64 DEV145 PROFILE COVERAGE — BOUNDED NATIVE PASS — 2026-10-01

Preflight found a real qualification defect in dev144: 170 active test files but
only 168 profile entries. The new cancellation and paused-recovery files were
omitted. Fixed in a separately versioned dev145 candidate. Exactly three files
changed: version, build metadata and PROFILE. Engine implementation and all test
bodies unchanged. Prior dev144 captures/source and V62/V63 evidence remain intact.
All 170 active test files are now registered exactly once; this is coverage of
file selectors, not evidence that all 170 files have executed successfully.

Standalone qualification contract: 4 PASS / 0 FAIL.
Fresh saved native job: 18 PASS / 0 FAIL / 0 SKIP, one worker, 72.053 seconds.
Suites: qualification contract (4), cancellation (5), paused checkpoint recovery (9).
The four standalone cases overlap the native 18; they are not 22 distinct cases.
Ten capture prerequisites saved/read back before execution. Live save/ack loop
operated through the run with default preservation and no engine deadline.
Zero pending checkpoint roles, bytes or checkpoints at closeout. Native published
completion and terminal proof verified. Saved export fetched and hash-verified;
isolated restoration recovered the identical completion and terminal proof with
ZERO tests dispatched. As designed, the terminal checkpoint may hold the prepared
completion before publication; this is not a claim of identical post-publication
workspace state. Source and reviewed runtime verified before/after.

Reclaimed 3.9 GiB of disposable successful V62 standalone pytest fixture copies;
retained reports, test logs, scripts and every native capture/failed restoration.

LIMITS / NEXT
Dev145 remains UNQUALIFIED overall; all 23 RC acceptance rows OPEN.
Five exact-candidate fixture roles remain: stage-one pre-run, stage-one completed,
stage-four, expected-outcome refusal and representative qualification. Generate
these on dev145 next; retain genuine pre-run export before stage-one execution.
Then full functional/workspace/representative/timing RC, remaining finding
dispositions, dependency lock and independent-host recovery. No science replay.
V62/V63 are separately identified dev144 bounded results, not full dev145 RC.
Do not rerun V64 or rewrite its completed source/capture.

'''
report+='Source identity:\n'+json.dumps(id,indent=2)+'\nCapture: '+result['capture_id']+'\nCompletion: '+done['completion_sha256']+'\nExport SHA256: '+export['sha256']+'\nExport Drive ID: '+export['drive_id']+'\nSource Drive ID: '+read(D/'SOURCE_READBACK.json')['drive_id']+'\nCheckpoint roles acknowledged: '+str(result['checkpoint_roles_acknowledged'])+'\n'
(D/'READ_FIRST_V64_DEV145_2026-10-01.txt').write_text(report)
e=R/'decoder-admin/evidence/V64';e.mkdir()
for prefix,base in [('checks',D),('native',N)]:
 for p in base.iterdir():
  if p.is_file() and p.suffix in ('.json','.txt','.py','.xml'):shutil.copy2(p,e/(prefix+'_'+p.name))
shutil.copy2(D/'READ_FIRST_V64_DEV145_2026-10-01.txt',R/'decoder-admin/evidence/V64_REPORT.txt')
ledger=read(R/'decoder-admin/RC_LEDGER.json');assert len(ledger['rows'])==23 and all(x['rc_acceptance_status']=='OPEN' for x in ledger['rows'])
ledger['historical_candidates']['dev144']=ledger['latest_candidate']
ledger['latest_candidate']={**id,'tests_executed':18,'status':'UNQUALIFIED_BOUNDED_V64_PASS','drive_id':read(D/'SOURCE_READBACK.json')['drive_id'],'reason':'Exact-candidate fixtures/full RC/findings/independent-host recovery pending'}
ledger['latest_profile_recovery_gate']=result
(R/'decoder-admin/RC_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
catalog=read(R/'decoder-admin/CATALOG.json');catalog['historical_sources']['dev144']=catalog['source'];catalog['source']={**id,'drive_id':read(D/'SOURCE_READBACK.json')['drive_id']};catalog['scope']='dev145 unqualified source/runtime locators; five candidate fixtures pending; prior candidate results historical.';(R/'decoder-admin/CATALOG.json').write_text(json.dumps(catalog,indent=2)+'\n')
(R/'DECODER_READ_FIRST.txt').write_text('DEV145 V64 BOUNDED NATIVE PASS — OVERALL UNQUALIFIED\n170 active files registered; 18 native cases PASS, exact saved completion recovery, zero restoration dispatch.\nFive current-source fixtures next; all 23 RC rows OPEN. No science replay.\nRead decoder-admin/evidence/V64_REPORT.txt, RC_LEDGER.json and CATALOG.json.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
(R/'decoder-import/README.txt').write_text('CURRENT SOURCE dev145 UNQUALIFIED. Read ../DECODER_READ_FIRST.txt.\n')
(R/'decoder-import/VERIFICATION.txt').write_text('V64 dev145 byte-exact source; 18 native PASS and exact terminal recovery. Full RC OPEN.\n'+json.dumps(id,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','native_passes','checkpoint_roles_acknowledged','unique_saved_objects']}))
