from pathlib import Path
import json,shutil
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev147_20261001');A=R/'decoder-admin'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not read(D/'FINAL_POST_STATUS.json')['pending_objects'];assert read(D/'PAUSED_RESTORE_PROOF.json')['attempt_bytes_identical']
cat={r['sha256']:r for r in read(D/'CATALOG.json')};source=read(A/'CATALOG.json')['source']['source_sha256'];e=read(D/'PAUSED_EXPORT.json');parts=[]
for p in read(D/'PAUSED_PARTS.json'):
 row=cat[p['sha256']];parts.append({'filename':'V82_'+p['sha256']+'.bin','sha256':p['sha256'],'size_bytes':p['size_bytes'],'drive_id':row['drive_id']})
fixtures={'saved_stage_outcome':{'sha256':e['sha256'],'size_bytes':sum(p['size_bytes'] for p in parts),'transport':'CONCATENATE_PARTS_IN_ORDER_THEN_VERIFY_ARCHIVE_HASH','parts':parts,'source_sha256':source,'state':'PAUSED_EXPECTED_REFUSAL','completion_sha256':None,'restored_from_drive_readbacks':True,'workload_executions_on_restore':0}}
write(D/'FIXTURE_CATALOG.json',fixtures)
result={'gate':'V82_DEV147_EXPECTED_OUTCOME_REFUSAL','fixture_acceptance':'EXPECTED_REFUSAL_VERIFIED','native_status':'PAUSED','reason':'ControllerLoopError:UNREGISTERED_SCIENTIFIC_OUTCOME','source_sha256':source,'native_attempts':1,'published_completions':0,'prepared_completions':0,'fixture_roles_earned':['saved_stage_outcome'],'pending_checkpoint_roles':0,'attempt_bytes_identical_on_restore':True,'workload_executions_on_restore':0,'full_rc':'OPEN','remaining_fixtures':['saved_representative_qualification']};write(D/'V82_RESULT.json',result)
assert read(D/'POST_SOURCE.json')['source_sha256']==source
obligations={x['obligation_id'] for p in D.glob('*_BATCH.json') for x in read(p)['obligations']};assert len(obligations)==45
report='''V82 DEV147 EXPECTED-OUTCOME REFUSAL FIXTURE — ACCEPTED — 2026-10-01

One registered attempt returned exactly the expected controller refusal:
ControllerLoopError:UNREGISTERED_SCIENTIFIC_OUTCOME. Native status remains PAUSED.
Unchanged handler returns PASS; registration permits only EXPECTED_OTHER.
One attempt, zero prepared completions, zero published completions.
No native success or test-suite PASS is claimed.

Source unchanged; all 1528 files verified. Reviewed runtime verified before and
after execution: 4721 files/modes and seven exact host dependencies.
All ten capture prerequisites saved/read back before dispatch. Complete paused
workspace saved in two ordered parts, reconstructed from actual Drive readbacks,
and restored with identical attempt bytes, PAUSED status and no completion.
Zero workloads dispatched on restoration. All 45 checkpoint role obligations
acknowledged, zero pending at closeout. No native rerun or unexplained failure.

Four of five current-dev147 fixture roles are earned. Next: representative
qualification via representative_qualification:handler, stage_id
ENG:REPRESENTATIVE_QUALIFICATION. Then bind all seven profile fixture roles
(including two historical inputs) and execute full RC. All 23 acceptance rows
remain OPEN. No science replay. Do not rerun V80, V81 or V82.
Read FIXTURE_CATALOG.json for ordered Drive parts and exact hashes.
Worktree /tmp/ig_decoder_dev147_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; PYTHONOPTIMIZE=0, -B.
After reset clone public branch, verify source/tree, import runtime objects from
decoder-admin/CATALOG.json, restore with decoder-admin/decoder.py.
'''
(D/'READ_FIRST_V82_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
c=read(A/'CATALOG.json');c['candidate_fixtures'].update(fixtures);c['unresolved_fixtures']=[x for x in c['unresolved_fixtures'] if x not in fixtures];write(A/'CATALOG.json',c)
l=read(A/'RC_LEDGER.json');l['fixture_gate_history'].append(l['latest_fixture_gate']);l['latest_fixture_gate']=result;l['current_rc_roadmap'].update(as_of_gate='V82',fixture_roles_complete=4,next='Generate representative dev147 fixture, then full qualification.');write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV147 V82 EXPECTED-REFUSAL FIXTURE ACCEPTED — OVERALL UNQUALIFIED\nFour of five fixture roles earned; representative qualification remains.\nOne PAUSED attempt, no completion; saved attempt bytes restored identically without rerun.\nRead decoder-admin/evidence/V82_REPORT.txt, CATALOG.json and RC_LEDGER.json. All 23 RC rows OPEN.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
e=A/'evidence/V82';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V82_2026-10-01.txt',A/'evidence/V82_REPORT.txt');print(json.dumps(result))
