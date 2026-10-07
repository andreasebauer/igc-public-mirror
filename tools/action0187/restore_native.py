"""Verify and cold-reuse this completed native job without invoking its handler."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent
if len(sys.argv)!=4:raise SystemExit('Usage: python-fixed-host restore_native.py ENGINE_ROOT OBJECTS_DIRECTORY FRESH_TMP_DESTINATION')
engine=Path(sys.argv[1]).resolve();objects=Path(sys.argv[2]).resolve();dest=Path(sys.argv[3]).resolve()
if not dest.is_relative_to(Path('/tmp')) or dest.exists():raise ValueError('FRESH_ISOLATED_DESTINATION_REQUIRED')
sys.path.insert(0,str(engine))
from infinity_grid import preservation as pr,submission as sub
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion,run_workspace_job
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
assert engineering_source_tree_digest(engine)=='bb65b91085c86d6aee1deab5a43300dfec6e8b225370b4df8f11f016e8c11952'
old=json.loads((B/'NATIVE_RESULT.json').read_bytes());export=json.loads((B/'EXPORT.json').read_bytes());archive=B/'FINAL_CHECKPOINT_SLIM.zip'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==export['sha256']
restored=pr.restore_checkpoint(archive,dest,export['sha256'],objects=objects)
admission=validate_workspace_job(dest,sub.capture_record(dest)['job']['job_id'],check_loaded=False)
done=verified_completion(admission,allow_pending_checkpoint=True)
assert done and done['completion_sha256']==old['completion_sha256'] and done['result_sha256']==old['result_sha256']
reused=run_workspace_job(dest,sub.capture_record(dest)['job']['job_id'])
assert reused['reused'] and reused['evidence_status']=='VERIFIED'
assert reused['completion_sha256']==old['completion_sha256'] and reused['result_sha256']==old['result_sha256']
assert pr.status(dest)['pending_bytes']==0
(B/'COLD_RESTORE.json').write_text(json.dumps(restored,indent=2)+'\n')
(B/'COLD_REUSE_RESULT.json').write_text(json.dumps(reused,indent=2)+'\n')
print(json.dumps({'status':'PASS_ISOLATED_COLD_RESTORE_AND_EXACT_NATIVE_REUSE','reused':True,'completion_sha256':reused['completion_sha256'],'result_sha256':reused['result_sha256'],'pending_bytes':0,'handler_called':False}))
