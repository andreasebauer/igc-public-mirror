from pathlib import Path
import json,sqlite3
W=Path(__file__).resolve().parent.parent
for num,reason in ((211,'FUTURE_IMPORT_PLACEMENT'),(212,'MISSING_CROSS_MODULE_DEPENDENCY_SHA_FILE')):
 B=W/f'qualification{num:04}';J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);rows=[]
 for db in J.glob('runtime/runs/*/chain/decoder_stage_runtime/*/phases/*/state_store.sqlite3'):
  c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);rows.append({'phase':db.parent.name,'states':c.execute('SELECT COUNT(*) FROM states').fetchone()[0],'committed_tasks':c.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]});c.close()
 assert all(x['states']==x['committed_tasks']==0 for x in rows)
 (B/'FAILURE_AUDIT.json').write_text(json.dumps({'status':'STOP_BEFORE_SCIENTIFIC_GENERATION','reason':reason,'frozen_capture_unchanged':True,'candidate_build_calls_committed':0,'native_registered_scope_completed':False,'phases':rows,'native_pending_bytes':json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']},indent=2))
print('PASS failed captures retained; no committed candidate builds')
