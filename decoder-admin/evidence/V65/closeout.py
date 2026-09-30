from pathlib import Path
import json,hashlib,shutil
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev145_20261001');A=R/'decoder-admin'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not read(D/'FINAL_POST_STATUS.json')['pending_objects']
assert read(D/'COMPLETED_RESTORE_PROOF.json')['exact_completion_equal']
cat={r['sha256']:r for r in read(D/'CATALOG.json')};fixtures={};source=read(A/'CATALOG.json')['source']['source_sha256'];done=read(D/'VERIFIED_COMPLETION.json')
for role,prefix,state in [('saved_stage_one_prerun','PRERUN','GENUINE_PRERUN'),('saved_stage_one','COMPLETED','PUBLISHED_COMPLETION')]:
 e=read(D/(prefix+'_EXPORT.json'));parts=[]
 for p in read(D/(prefix+'_PARTS.json')):
  row=cat[p['sha256']];parts.append({'filename':'V65_'+p['sha256']+'.bin','sha256':p['sha256'],'size_bytes':p['size_bytes'],'drive_id':row['drive_id']})
 fixtures[role]={'sha256':e['sha256'],'size_bytes':sum(p['size_bytes'] for p in parts),'transport':'CONCATENATE_PARTS_IN_ORDER_THEN_VERIFY_ARCHIVE_HASH','parts':parts,'source_sha256':source,'state':state,'completion_sha256':done['completion_sha256'] if prefix=='COMPLETED' else None,'restored_from_drive_readbacks':True,'workload_executions_on_restore':0}
write(D/'FIXTURE_CATALOG.json',fixtures)
result={'gate':'V65_DEV145_STAGE_ONE_FIXTURES','status':'PASS','source_sha256':source,'native_jobs_executed':1,'task_count':48,'class_count':7,'workers':1,'fixture_roles_earned':list(fixtures),'pending_checkpoint_roles':0,'workload_executions_on_restore':0,'completion_sha256':done['completion_sha256'],'full_rc':'OPEN','remaining_fixtures':['saved_stage_four','saved_stage_outcome','saved_representative_qualification']}
write(D/'V65_RESULT.json',result)
report='''V65 DEV145 STAGE-ONE FIXTURES — PASS — 2026-10-01

Recovered after environment reset: V64 bundle and dev145 archive hashes matched.
Public Git tree 21143fbde596c8407140716f364fd742c95d3a40 matched; 1526 source
files verified. Runtime reconstructed: 4721 exact files/modes plus seven exact
host files. No V64 tests rerun.

One fresh registered stage-one job completed: PASS, 48 task results, seven
classes, one worker. These are fixture counts, not 48 test-suite passes.
Ten capture prerequisites saved and read back before dispatch. Genuine complete
pre-run export saved and reconstructed from Drive parts, restored with zero
attempts/prepared/published completions. Complete post-publication export saved,
read back, and restored to identical published completion and terminal proof.
Zero workloads dispatched on either restore. All 60 checkpoint role obligations
acknowledged, zero pending at closeout. Source unchanged.

Two of five current-dev145 fixture roles are earned. Next: stage-four (four
workers, same 48 values modulo seven), expected-outcome refusal, representative
qualification. Then full RC; all 23 acceptance rows remain OPEN. No science replay.
Do not rerun V65. Read FIXTURE_CATALOG.json for ordered Drive parts and hashes.
Worktree /tmp/ig_decoder_dev145_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; PYTHONOPTIMIZE=0, -B.
After reset clone published branch, verify tree/source, import runtime objects
from decoder-admin/CATALOG.json and restore using decoder-admin/decoder.py.
Frozen producer capture and both restored fixtures are retained locally.

Administrative transport scripting hit a missing JavaScript btoa helper after
three successful saves. Retained those saved readbacks and resumed only missing
objects using plain JSON persistence; no native execution failed or was retried.
'''
report+='\n'+json.dumps(result,indent=2)+'\n';(D/'READ_FIRST_V65_2026-10-01.txt').write_text(report)
c=read(A/'CATALOG.json');c['candidate_fixtures'].update(fixtures);c['unresolved_fixtures']=[x for x in c['unresolved_fixtures'] if x not in fixtures];write(A/'CATALOG.json',c)
l=read(A/'RC_LEDGER.json');l['fixture_gate_history'].append(l['latest_fixture_gate']);l['latest_fixture_gate']=result;write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV145 V65 STAGE-ONE FIXTURES PASS — OVERALL UNQUALIFIED\nTwo of five current-candidate fixture roles earned; three remain.\n48 task results, seven classes; genuine pre-run and identical completion recovery, zero restoration dispatch.\nRead decoder-admin/evidence/V65_REPORT.txt, CATALOG.json and RC_LEDGER.json. All 23 RC rows OPEN.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
e=A/'evidence/V65';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py'}:shutil.copy2(p,e/p.name)
for p in (D.parent/'v65_recovery').iterdir():shutil.copy2(p,e/('recovery_'+p.name))
shutil.copy2(D/'READ_FIRST_V65_2026-10-01.txt',A/'evidence/V65_REPORT.txt')
print(json.dumps(result))
