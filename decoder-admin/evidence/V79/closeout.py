from pathlib import Path
import json,shutil,hashlib,subprocess
D=Path(__file__).parent;R=Path('/tmp/ig_decoder_dev147_20261001');read=lambda p:json.loads(p.read_text())
rest=read(D/'RESTORE_VERIFICATION.json');assert rest['status']=='EXACT_PAUSED_RESTORATION_PASS'
s=read(D/'CAPTURE_SAVE_STATUS.json');post=read(D/'POSTPAUSE_POST_STATUS.json');assert not post['pending_objects']
lock=read(D/'ACK_LOCKED.json')['unix'];release=read(D/'ACK_RELEASE.json')['unix'];done=read(D/'GATED_ACK.json')['unix'];events=[json.loads(x) for x in (D/'LOCK_EVENTS.jsonl').read_text().splitlines()]
assert lock<events[0]['unix']<release<=done<=events[1]['unix'];assert release-events[0]['unix']>=.5
summary={'gate':'V79_DEV147_NATIVE_STARTUP_CONTENTION','status':'BOUNDED_NATIVE_CONTENTION_PAUSE_AND_EXACT_RESTORE_PASS','source_sha256':s['source_sha256'] if 'source_sha256' in s else rest['restore']['source_sha256'],'capture_id':s['capture_id'],'native_attempts':1,'attempt_status':'PAUSED','native_completions':0,'completed_test_passes':0,'startup_lock_wait_seconds':events[1]['unix']-events[0]['unix'],'pending_roles':0,'pending_bytes':0,'pending_checkpoints':0,'checkpoint_roles_acknowledged':read(D/'GATED_ACK.json')['result']['obligations_acknowledged']+read(D/'POSTPAUSE_ACK.json')['obligations_acknowledged'],'restore_files_bytes_and_modes':rest['files_checked'],'recovery_workloads_dispatched':0,'export_sha256':rest['export_sha256'],'export_drive_id':read(D/'EXPORT_READBACK.json')['drive_file_id'],'full_rc':'OPEN','limitations':['Induced startup checkpoint contention; not incidental acknowledgment overlap during an active worker.','Cleanup inferred from unchanged reviewed fail-closed control flow; raw worker cleanup pipe acknowledgment not persisted.','Same-host recovery; independent remote-only recovery remains open.']}
(D/'GATE_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
report='''V79 / DEV147 NATIVE CONTENTION AND EXACT PAUSED RECOVERY — 2026-10-01

BOUNDED GATE PASS. Native startup checkpoint publication waited for genuine readback acknowledgment to release the outbox lock. The worker then started; the requested native pause was retained with its matching acknowledgment and refusal. There is no completed test pass or accepted completion from this intentionally paused probe.

All 45 checkpoint-role acknowledgments across three checkpoints are complete (15 during startup plus 30 during closeout). Zero pending roles, bytes or checkpoints. Nineteen downloaded dependency objects were hash-verified. The saved slim export was downloaded and restored; all 34 checkpoint state/input files matched bytes and ordinary permission modes. Recovery dispatched zero workloads. No rerun of V79 or V77.

Exact source: 098d505dfba5575aec3efc9f2ecf3614a28be3b7caeda41482a94dda2e5a1ea8
Export: 3b8849b05deec8ee8bd2c50f3ed42c1f6b2d5ec3fd596f2c84821cd1424aeb39
Export Drive ID: 15PAz5a50sMMkhBuqsoZSfZG3nIj1c5v-
Source verification: 1,528 files byte-exact. Recorded pre/post runtime verification: 4,721 runtime files and seven host files. Engine source unchanged by this closeout.

LIMITS
This deliberately induced startup contention, not incidental connector overlap during an active worker. Worker cleanup is inferred from reviewed fail-closed control flow; raw pipe acknowledgment was not persisted. Recovery uses downloaded files on the same host, not an independent-host qualification. V77 remains failed/incomplete with its original 583 passed nodes; those passes are not dev147 qualification.

CURRENT POSITION AND NEXT
Dev147 remains UNQUALIFIED. V78's 30 focused passes remain valid bounded evidence. All 23 RC acceptance rows remain OPEN. Five exact-dev147 fixture roles remain to be generated; historical dev146 fixtures cannot be relabeled. Next: prepare the one-worker stage fixture, preserving both pre-run and completed states, then four-worker, expected-outcome refusal, and representative fixtures. After fixtures: all four qualification groups / 170 test files, finding dispositions, independent remote-only recovery, explicit RC acceptance/promotion. Canonical scientific L0-to-G8 replay has not started.

RECOVERY
Use READBACKS.json for exact saved dependency IDs/hashes, EXPORT_READBACK.json for the export, and restore_exact.py as the reconstruction reference. Its existing destination and restore_objects directory are exclusive; do not blindly rerun it. Existing source: /tmp/ig_decoder_dev147_20261001. Existing runtime: /tmp/ig_runtime_v55_fresh_20260930. Existing paused capture and restored copy remain intact. No workload is running for V79.
'''
(D/'READ_FIRST_V79_2026-10-01.txt').write_text(report)
for filename in ['CATALOG.json','RC_LEDGER.json']:
 p=R/'decoder-admin'/filename;d=read(p)
 if filename=='CATALOG.json':d.update(native_contention_gate=summary,scope='Dev147 V78 focused repair and V79 bounded native startup contention/recovery passed; full RC OPEN; exact-candidate fixtures pending.')
 else:
  assert len(d['rows'])==23 and all(x['rc_acceptance_status']=='OPEN' for x in d['rows'])
  d['latest_native_contention_gate']=summary;d['last_updated']='2026-10-01';d['prior_rc_roadmap_v77']=d['current_rc_roadmap'];d['current_rc_roadmap']={'as_of_gate':'V79','candidate':'dev147','fixture_roles_complete':0,'fixture_roles_total':5,'full_candidate_qualification':'NOT_RUN','profile_files':170,'open_acceptance_rows':23,'independent_remote_recovery':'OPEN','promotion':'NOT_GRANTED','next':'Generate five exact-dev147 fixture roles, then full qualification and acceptance review.'}
  d['prior_latest_candidate_v71']=d['latest_candidate'];d['latest_candidate']={**read(R/'decoder-admin/CATALOG.json')['source'],'qualification':'UNQUALIFIED','status':'V78_AND_V79_BOUNDED_PASS_FULL_RC_OPEN'}
 p.write_text(json.dumps(d,indent=2)+'\n')
(R/'DECODER_READ_FIRST.txt').write_text(report)
E=R/'decoder-admin/evidence/V79';E.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.py') and p.name not in ['SAVE_CATALOG.json']:
  shutil.copyfile(p,E/p.name)
subprocess.run(['git','-C',str(R),'diff','--check'],check=True)
subprocess.run(['git','-C',str(R),'add','DECODER_READ_FIRST.txt','decoder-admin/CATALOG.json','decoder-admin/RC_LEDGER.json','decoder-admin/evidence/V79'],check=True)
subprocess.run(['git','-C',str(R),'-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Record V79 native startup contention and exact paused recovery'],check=True)
files=subprocess.check_output(['git','-C',str(R),'diff','--name-only','HEAD^','HEAD'],text=True).splitlines()
payload=[{'path':n,'mode':'100644','type':'blob','content':(R/n).read_text()} for n in files]
(D/'TREE_PAYLOAD.json').write_text(json.dumps(payload));print(json.dumps(summary))
