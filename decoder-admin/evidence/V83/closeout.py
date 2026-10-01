from pathlib import Path
import json,shutil
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev147_20261001');A=R/'decoder-admin'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not read(D/'FINAL_POST_STATUS.json')['pending_objects'];assert read(D/'COMPLETED_RESTORE_PROOF.json')['exact_completion_equal']
cat={r['sha256']:r for r in read(D/'CATALOG.json')};source=read(A/'CATALOG.json')['source']['source_sha256'];done=read(D/'VERIFIED_COMPLETION.json');e=read(D/'COMPLETED_EXPORT.json');parts=[]
for p in read(D/'COMPLETED_PARTS.json'):
 row=cat[p['sha256']];parts.append({'filename':'V83_'+p['sha256']+'.bin','sha256':p['sha256'],'size_bytes':p['size_bytes'],'drive_id':row['drive_id']})
fixtures={'saved_representative_qualification':{'sha256':e['sha256'],'size_bytes':sum(p['size_bytes'] for p in parts),'transport':'CONCATENATE_PARTS_IN_ORDER_THEN_VERIFY_ARCHIVE_HASH','parts':parts,'source_sha256':source,'state':'PUBLISHED_COMPLETION','completion_sha256':done['completion_sha256'],'restored_from_drive_readbacks':True,'workload_executions_on_restore':0}};write(D/'FIXTURE_CATALOG.json',fixtures)
cases=read(D/'NATIVE_RESULT.json')['result']['cases'];assert len(cases)==4 and set(cases.values())=={'PASS'}
result={'gate':'V83_DEV147_REPRESENTATIVE_FIXTURE','status':'PASS','source_sha256':source,'native_jobs_executed':1,'cases_passed':4,'cases_failed':0,'cases':cases,'fixture_roles_earned':['saved_representative_qualification'],'all_five_candidate_fixtures_earned':True,'pending_checkpoint_roles':0,'workload_executions_on_restore':0,'completion_sha256':done['completion_sha256'],'full_rc':'OPEN','remaining_fixtures':[]};write(D/'V83_RESULT.json',result)
c=read(A/'CATALOG.json');c['candidate_fixtures'].update(fixtures);c['unresolved_fixtures']=[x for x in c['unresolved_fixtures'] if x not in fixtures];assert not c['unresolved_fixtures'] and len(c['candidate_fixtures'])==5;assert all(x['source_sha256']==source for x in c['candidate_fixtures'].values());write(A/'CATALOG.json',c);write(D/'ALL_FIVE_FIXTURE_CATALOG.json',c['candidate_fixtures'])
assert read(D/'POST_SOURCE.json')['source_sha256']==source
obligations={x['obligation_id'] for p in D.glob('*_BATCH.json') for x in read(p)['obligations']};assert len(obligations)==45
report='''V83 DEV147 REPRESENTATIVE FIXTURE — PASS — 2026-10-01

Four representative-byte cases PASS, zero FAIL. Cases:257 distinct then duplicate;
cold multicore grouping and durable bytes; partial resume without representative
reopen; legacy NULL backfill then reuse. Unchanged handler and assertions.
Internal interruption/resume is part of the declared scenario. Standalone
schema-migration test remains for full qualification.

One fresh registered job completed. Source unchanged; 1528 files verified.
Reviewed runtime verified before/after:4721 files/modes and seven host files.
Ten capture prerequisites saved/read back before dispatch. interval_seconds=30,
other preservation limits default. Completed fixture saved in three ordered
parts, reconstructed from actual Drive readbacks, restored to identical published
completion and terminal proof. Zero restoration workloads. All45 checkpoint
role obligations acknowledged; zero pending. No producer rerun.

FIVE-FIXTURE PREPARATION GATE COMPLETE
All five exact-dev147 roles earned:stage-one pre-run, stage-one completed,
stage-four completed, expected-outcome refusal (native PAUSED), representative
qualification completed. ALL_FIVE_FIXTURE_CATALOG.json pins every archive and
ordered Drive part. Two historical profile inputs remain separately pinned in
repository CATALOG:step2_parent_snapshot and private failed_evidence.
This is fixture readiness, not full source qualification or RC acceptance.

NEXT
Preregister full exact-source qualification and bind all seven fixture roles.
Verify historical inputs from saved bytes and capture actual node inventory.
Functional group first; workspace and representative next; isolated one-worker
timing last. Skips/errors/unexecuted cases never count as PASS. Preserve source
identity; do not edit frozen PROFILE just to update progress. All23 RC rows OPEN.
Install lock/native dependency and independent-host closure remain open.
No science replay or release promotion. Do not rerun V80/V81/V82/V83 producers.
Worktree /tmp/ig_decoder_dev147_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; PYTHONOPTIMIZE=0, -B.
After reset clone public branch, verify source/tree, import runtime objects from
repository CATALOG, then restore with decoder-admin/decoder.py. Fixture recovery:
concatenate listed parts, verify every part and archive, native restore_workspace.
'''
(D/'READ_FIRST_V83_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
l=read(A/'RC_LEDGER.json');l['fixture_gate_history'].append(l['latest_fixture_gate']);l['latest_fixture_gate']=result;l['current_rc_roadmap'].update(as_of_gate='V83',fixture_roles_complete=5,next='Preregister full exact-dev147 qualification, bind all seven fixture roles; functional group first.');write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV147 V83 FIVE-FIXTURE PREPARATION COMPLETE — OVERALL UNQUALIFIED\nAll five candidate fixture roles earned and saved/readback/restoration verified.\nV83 four representative cases PASS; identical completion restored without rerun.\nNext: preregister full qualification, bind all seven fixture roles, functional first.\nRead decoder-admin/evidence/V83_REPORT.txt, CATALOG.json and RC_LEDGER.json. All23 RC rows OPEN.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
e=A/'evidence/V83';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V83_2026-10-01.txt',A/'evidence/V83_REPORT.txt');print(json.dumps(result))
