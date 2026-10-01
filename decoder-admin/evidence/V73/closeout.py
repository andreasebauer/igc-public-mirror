from pathlib import Path
import json,hashlib,shutil,subprocess
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');A=R/'decoder-admin'
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not read(D/'EXPORT_FINAL_POST_STATUS.json')['pending_objects']
assert read(D/'COMPLETED_RESTORE_PROOF.json')['exact_completion_equal']
cat={r['sha256']:r for r in read(D/'CATALOG.json')};fixtures={};source=read(A/'CATALOG.json')['source']['source_sha256'];done=read(D/'VERIFIED_COMPLETION.json')
assert done['result']['outcome']=='PASS' and done['result']['partition']['task_count']==48 and done['result']['partition']['class_count']==7
for role,prefix,state in [('saved_stage_one_prerun','PRERUN','GENUINE_PRERUN'),('saved_stage_one','COMPLETED','PUBLISHED_COMPLETION')]:
 e=read(D/(prefix+'_EXPORT.json'));parts=[]
 for p in read(D/(prefix+'_PARTS.json')):
  row=cat[p['sha256']];parts.append({'filename':'V73_'+p['sha256']+'.bin','sha256':p['sha256'],'size_bytes':p['size_bytes'],'drive_id':row['drive_id']})
 fixtures[role]={'sha256':e['sha256'],'size_bytes':sum(p['size_bytes'] for p in parts),'transport':'CONCATENATE_PARTS_IN_ORDER_THEN_VERIFY_ARCHIVE_HASH','parts':parts,'source_sha256':source,'state':state,'completion_sha256':done['completion_sha256'] if prefix=='COMPLETED' else None,'restored_from_drive_readbacks':True,'workload_executions_on_restore':0}
write(D/'FIXTURE_CATALOG.json',fixtures)
obligations=set()
for p in D.glob('*_BATCH.json'):
 obligations.update(x['obligation_id'] for x in read(p)['obligations'])
assert len(obligations)==60
p=subprocess.run(['python3','-B',str(R/'decoder-import/verify_source.py')],capture_output=True,text=True);assert p.returncode==0;write(D/'POST_SOURCE.json',json.loads(p.stdout))
result={'gate':'V73_DEV146_STAGE_ONE_FIXTURES','status':'PASS','source_sha256':source,'native_jobs_executed':1,'task_count':48,'class_count':7,'workers':1,'fixture_roles_earned':list(fixtures),'checkpoint_roles_acknowledged':len(obligations),'pending_checkpoint_roles':0,'workload_executions_on_restore':0,'completion_sha256':done['completion_sha256'],'full_rc':'OPEN','remaining_fixtures':['saved_stage_four','saved_stage_outcome','saved_representative_qualification']};write(D/'V73_RESULT.json',result)
report='''V73 DEV146 STAGE-ONE FIXTURES — PASS — 2026-10-01

One fresh registered stage-one job completed PASS:48 task results,seven classes,
one worker. These are fixture counts, not 48 test-suite passes. Exact frozen
dev146 source and reviewed runtime passed byte/mode verification.

Ten capture prerequisites saved and read back before dispatch. Genuine complete
pre-run export saved in two parts and full hash verified; restored with zero
attempts, prepared completions and published completions. Complete post-publication
export saved in three parts, full hash verified, restored with identical published
completion and terminal proof. Zero workloads dispatched on either restore.
All60 checkpoint role obligations acknowledged; zero pending at closeout.
No workload failures/retries; source unchanged. V70 and V72 captures untouched.

Two of five current-dev146 fixture roles earned. NEXT: fresh four-worker stage
fixture using the same48 values modulo7; use V66 protocol adapted to dev146 and
compare exact partition relation with V73. Then expected-outcome refusal and
representative qualification fixtures, followed by fresh full RC. All23 acceptance
rows remain OPEN. No science replay or release qualification claim.
Do not rerun V73. FIXTURE_CATALOG.json contains ordered Drive part IDs and hashes.
Worktree /tmp/ig_decoder_dev146_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; PYTHONOPTIMIZE=0,-B.
Producer and both restored fixtures remain available under this evidence folder
and /tmp/ig_fixture_v73_20261001. After reset use saved readbacks; no recovery rerun.
'''
(D/'READ_FIRST_V73_2026-10-01.txt').write_text(report+'\n'+json.dumps(result,indent=2)+'\n')
c=read(A/'CATALOG.json');assert not c['candidate_fixtures'];c['candidate_fixtures'].update(fixtures);c['unresolved_fixtures']=[x for x in c['unresolved_fixtures'] if x not in fixtures];write(A/'CATALOG.json',c)
l=read(A/'RC_LEDGER.json');assert len(l['rows'])==23 and all(x['rc_acceptance_status']=='OPEN' for x in l['rows']);l['fixture_gate_history'].append(l['latest_fixture_gate']);l['latest_fixture_gate']=result;write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV146 V73 STAGE-ONE FIXTURES PASS — OVERALL UNQUALIFIED\nTwo of five current-candidate fixture roles earned;three remain.\n48 task results,seven classes;genuine pre-run and identical completion recovery,zero restore dispatch.\nRead decoder-admin/evidence/V73_REPORT.txt,CATALOG.json and RC_LEDGER.json.All23 RC rows OPEN.\n')
e=A/'evidence/V73';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V73_2026-10-01.txt',A/'evidence/V73_REPORT.txt');print(json.dumps(result))
