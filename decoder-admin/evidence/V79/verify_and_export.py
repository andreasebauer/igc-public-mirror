from pathlib import Path
import json,sys,runpy,hashlib,subprocess
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev147_20261001');read=lambda p:json.loads(p.read_text());s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
assert not pr.status(w)['pending_objects']
attempts=[read(p) for p in (w/'runtime/attempts').rglob('*.json')];assert len(attempts)==1 and attempts[0]['status']=='PAUSED'
ack=read(w/'runtime/PAUSE_ACK.json');req=read(w/'runtime/PAUSE_REQUEST.json');assert ack==req and ack['capture_id']==s['capture_id']
exc=read(D/'NATIVE_EXCEPTION.json');assert exc['type']=='SubmissionError' and exc['message']=='REQUESTED_PAUSE:V79_PREREGISTERED_ACTIVE_PAUSE'
ref=[read(p) for p in (w/'runtime/refusals').glob('*.json')];assert len(ref)==1 and 'REQUESTED_PAUSE' in json.dumps(ref)
assert verified_completion(validate_workspace_job(w,s['job_id'],check_loaded=False),allow_pending_checkpoint=True) is None
progress=list((w/'runtime/runs').rglob('long_durable_progress.log'));assert len(progress)==1
rows=[json.loads(x) for x in progress[0].read_text().splitlines()];assert rows[0]['event']=='START' and not any(x['event']=='END' for x in rows)
source=runpy.run_path(str(R/'decoder-import/verify_source.py'))['verify']()
(D/'POST_SOURCE.json').write_text(json.dumps(source,indent=2))
report={'status':'NATIVE_PAUSE_OBSERVATIONS_PASS','attempts':attempts,'pause_ack':ack,'refusals':ref,'progress':rows,'cleanup_evidence':'Inference from unchanged reviewed control flow: cancel_workers must reap worker and validate cleanup_quiescent reply before REQUESTED_PAUSE propagates. Raw pipe reply is not persisted. No cleanup error observed.','native_completion':None,'native_attempts':1,'workload_reruns':0}
(D/'LIVE_VERIFICATION.json').write_text(json.dumps(report,indent=2))
e=pr.export_checkpoint(w,D/'V79_PAUSED_CHECKPOINT.zip',slim=True);(D/'EXPORT.json').write_text(json.dumps(e,indent=2));print(json.dumps({'status':report['status'],'export_sha256':e['sha256']}))
