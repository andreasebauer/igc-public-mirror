from pathlib import Path
import json,shutil
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev145_20261001');A=R/'decoder-admin'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not read(D/'FINAL_POST_STATUS.json')['pending_objects'];assert read(D/'COMPLETED_RESTORE_PROOF.json')['exact_completion_equal'];assert read(D/'EXACT_RELATION_COMPARISON.json')['exact_relation_equal']
cat={r['sha256']:r for r in read(D/'CATALOG.json')};source=read(A/'CATALOG.json')['source']['source_sha256'];done=read(D/'VERIFIED_COMPLETION.json');e=read(D/'COMPLETED_EXPORT.json');parts=[]
for p in read(D/'COMPLETED_PARTS.json'):
 row=cat[p['sha256']];parts.append({'filename':'V66_'+p['sha256']+'.bin','sha256':p['sha256'],'size_bytes':p['size_bytes'],'drive_id':row['drive_id']})
fixtures={'saved_stage_four':{'sha256':e['sha256'],'size_bytes':sum(p['size_bytes'] for p in parts),'transport':'CONCATENATE_PARTS_IN_ORDER_THEN_VERIFY_ARCHIVE_HASH','parts':parts,'source_sha256':source,'state':'PUBLISHED_COMPLETION','completion_sha256':done['completion_sha256'],'restored_from_drive_readbacks':True,'workload_executions_on_restore':0}}
write(D/'FIXTURE_CATALOG.json',fixtures)
result={'gate':'V66_DEV145_FOUR_WORKER_FIXTURE','status':'PASS','source_sha256':source,'native_jobs_executed':1,'task_count':48,'class_count':7,'observed_workers':4,'backend':'LOCAL_PROCESS_POOL','exact_relation_matches_V65':True,'compared_rows':48,'V65_rerun':False,'fixture_roles_earned':['saved_stage_four'],'pending_checkpoint_roles':0,'workload_executions_on_restore':0,'completion_sha256':done['completion_sha256'],'full_rc':'OPEN','remaining_fixtures':['saved_stage_outcome','saved_representative_qualification']};write(D/'V66_RESULT.json',result)
report='''V66 DEV145 FOUR-WORKER FIXTURE — PASS — 2026-10-01

One fresh registered native job completed: PASS, 48 task results, seven classes.
Observed LOCAL_PROCESS_POOL with four workers. Exact task_id and representative
signature bytes match all 48 rows of saved V65 one-worker result. V65 was not
rerun. These are fixture counts, not 48 test-suite passes.

Source unchanged; all 1526 files verified. Reviewed runtime verified before and
after execution: 4721 files/modes and seven exact host dependencies.
All ten capture prerequisites saved/read back before dispatch. Completed native
export saved in three ordered parts, reconstructed from actual Drive readbacks,
restored to identical published completion and verified terminal proof. Zero
workloads dispatched on restoration. All 45 checkpoint role obligations
acknowledged, zero pending at closeout. No native reruns or unexplained failures.

Three of five current-dev145 fixture roles are earned. Next: expected-outcome
refusal and representative qualification. Then full RC; all 23 acceptance rows
remain OPEN. No science replay. Do not rerun V65 or V66.
Read FIXTURE_CATALOG.json for ordered Drive parts and exact hashes.
Worktree /tmp/ig_decoder_dev145_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; PYTHONOPTIMIZE=0, -B.
After reset clone public branch, verify source/tree, import runtime objects from
decoder-admin/CATALOG.json, restore with decoder-admin/decoder.py.
'''
(D/'READ_FIRST_V66_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
c=read(A/'CATALOG.json');c['candidate_fixtures'].update(fixtures);c['unresolved_fixtures']=[x for x in c['unresolved_fixtures'] if x not in fixtures];write(A/'CATALOG.json',c)
l=read(A/'RC_LEDGER.json');l['fixture_gate_history'].append(l['latest_fixture_gate']);l['latest_fixture_gate']=result;write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV145 V66 FOUR-WORKER FIXTURE PASS — OVERALL UNQUALIFIED\nThree of five fixture roles earned; expected-outcome refusal and representative qualification remain.\n48 exact rows equal saved V65; four workers observed; saved identical completion restored without rerun.\nRead decoder-admin/evidence/V66_REPORT.txt, CATALOG.json and RC_LEDGER.json. All 23 RC rows OPEN.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
e=A/'evidence/V66';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V66_2026-10-01.txt',A/'evidence/V66_REPORT.txt');print(json.dumps(result))
