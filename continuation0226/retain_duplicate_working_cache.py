"""Retain duplicate working cache outside capture only after terminal raw-limit refusal.
Authenticates sealed completion, compares all scientific files and SQLite rows,
then renames the original working tree. Frozen policy and evidence stay untouched.
"""
from pathlib import Path
import sys,json,hashlib,sqlite3,itertools,os
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
assert 'CHECKPOINT_STATE_RAW_LIMIT' in (B/'ERROR.txt').read_text()
assert not (B/'NATIVE_RESULT.json').exists()
sys.path.insert(0,str(J/'source'))
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion,_evidence_rows
from infinity_grid import submission as sub
ad=validate_workspace_job(J,sub.capture_record(J)['job']['job_id']);done=verified_completion(ad,allow_pending_checkpoint=True)
assert done is not None and done['status']=='COMPLETED' and done['evidence_status']=='VERIFIED'
assert done['evidence_protocol']=='SEPARATE_COMPLETION_EVIDENCE_V1'
sealed=J/done['evidence_root'];working=J/'runtime/runs'/sealed.name
retained=J.parents[2]/'retained_working_runs';assert not retained.exists()
assert len(list((J/'runtime/runs').iterdir()))==1
original=_evidence_rows(working);by={x['path']:x for x in original};rows=[]
for x in done['evidence']:
 name=x['path'];a=working/name;b=sealed/name
 if name=='RESULT_VERIFICATION.json' and not a.exists():continue
 assert name in by
 if a.name in ('state_store.sqlite3','partition.sqlite3') and name.startswith('chain/decoder_stage_runtime/'):
  with sqlite3.connect(a.resolve().as_uri()+'?mode=ro',uri=True) as ca,sqlite3.connect(b.resolve().as_uri()+'?mode=ro',uri=True) as cb:
   schema="SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
   assert ca.execute(schema).fetchall()==cb.execute(schema).fetchall()
   for (table,) in ca.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
    quoted='"'+table.replace('"','""')+'"';query='SELECT * FROM '+quoted+' ORDER BY rowid';sentinel=object()
    for ra,rb in itertools.zip_longest(ca.execute(query),cb.execute(query),fillvalue=sentinel):assert ra==rb
   if 'historical_g1_depth_' in name:
    assert ca.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1
    assert ca.execute('SELECT COUNT(*) FROM states').fetchone()[0]==1
  rows.append({'path':name,'comparison':'EXACT_SQLITE_SCHEMA_AND_ROWS'})
 else:
  assert by[name]['sha256']==x['sha256'] and by[name]['size_bytes']==x['size_bytes']
  rows.append({'path':name,'comparison':'EXACT_RAW_BYTES'})
assert len([x for x in rows if x['path'].endswith('/state_store.sqlite3') and 'historical_g1_depth_' in x['path']])==6
assert _evidence_rows(working)==original
os.rename(J/'runtime/runs',retained)
assert verified_completion(ad,allow_pending_checkpoint=True)['completion_sha256']==done['completion_sha256']
r={'status':'PASS_DUPLICATE_WORKING_CACHE_RETAINED','capture_id':sub.capture_record(J)['capture_id'],'completion_sha256':done['completion_sha256'],'retained_original_path':str(retained),'original_working_inventory':original,'comparisons':rows,'frozen_capture_modified':False,'resource_policy_modified':False,'sealed_evidence_modified':False,'candidate_generation_calls':0,'reason':'Original terminal checkpoint refused duplicate working plus sealed state at registered raw snapshot limit; sealed completion authority retained unchanged.'}
(B/'CACHE_RETENTION_RECOVERY.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('original_working_inventory','comparisons')}))
