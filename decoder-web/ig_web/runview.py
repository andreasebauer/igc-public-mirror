"""Run-screen observations and guarded transport requests; native admission is final."""
from contextlib import closing
import json
from pathlib import Path
import time

from .models import AcceptedRequest
from .native import AdapterError, PIN_SOURCE, canonical_hash, contained, read_json
from .results import results_observation
from .worker import command_digest

ACTIVE = {'queued','dispatching','running','needs_reconciliation','interrupted'}


def project(job_id, status, preservation, requests, result):
    """Pure projection. Unknown, busy-other-job and audit stops never enable Start."""
    out={'execution':'Unknown','backup':'Unknown','action':None,'reason':'NATIVE_OBSERVATION_UNAVAILABLE'}
    if status.get('classification')!='RESPONSE': return out
    native=status.get('native') or {};record=native.get('last_record');busy=native.get('workspace_locked_now')
    if type(busy) is not bool: return out
    capture=preservation.get('capture',{});outbox=preservation.get('outbox',{})
    if capture.get('classification')=='RESPONSE' and outbox.get('classification')=='RESPONSE':
        c=capture.get('native') or {};o=outbox.get('native') or {}
        if all(isinstance(v,list) for v in (c.get('pending_objects'),o.get('pending_objects'),o.get('pending_checkpoints'))):
            if c['pending_objects'] or o['pending_objects'] or o['pending_checkpoints']:out['backup']='Awaiting backup'
            elif c.get('status')=='SAVED' and o.get('status')=='PRESERVED':out['backup']='Native reports preserved'
    run=next((r for r in requests if r['operation']=='run'),None)
    pause=next((r for r in requests if r['operation']=='pause'),None)
    matching=isinstance(record,dict) and record.get('job_id')==job_id
    if busy:
        out.update(execution='Running reported' if matching and record.get('status')=='RUNNING' else 'Workspace busy · job unconfirmed',reason='WORKSPACE_BUSY')
        if matching and record.get('status')=='RUNNING':
            # Finished control process means request delivered, not pause acknowledged.
            pending=pause and pause['status'] in ACTIVE|{'finished'} and (not run or pause['created']>=run['created'])
            if pending:out.update(execution='Pause requested',reason='WAIT_FOR_NATIVE_PAUSE')
            else:out.update(action='pause',reason='COOPERATIVE_PAUSE_AVAILABLE')
        return out
    if any(r['status'] in ACTIVE for r in requests):
        active=next(r for r in requests if r['status'] in ACTIVE)
        out.update(execution='Needs reconciliation' if active['status'] in {'needs_reconciliation','interrupted','running'} else 'Queued' if active['status']=='queued' else 'Dispatching',reason='WEB_REQUEST_UNRESOLVED');return out
    if result.get('record_status')=='PENDING_CHECKPOINT_RECORD':
        out.update(execution='Checkpoint pending',reason='CHECKPOINT_PUBLICATION_PENDING');return out
    if result.get('record_status')=='PUBLISHED_RECORD':
        out.update(execution='Finished',reason='VIEW_RESULTS');return out
    if result.get('record_status')!='ABSENT':return out
    if matching and record.get('status')=='RUNNING':
        out.update(execution='Needs reconciliation',reason='IDLE_WITH_RUNNING_RECORD');return out
    if record is not None and not matching:
        out.update(reason='NATIVE_JOB_UNCONFIRMED');return out
    if matching:
        if record.get('status')!='PAUSED' or not str(record.get('reason','')).startswith('SubmissionError:REQUESTED_PAUSE'):
            out.update(execution='Needs review',reason='NATIVE_STOP_REQUIRES_REVIEW');return out
        out['execution']='Paused'
    else:out['execution']='Prepared'
    if run and run['status'] in {'refused','finished'} and not matching:
        out.update(execution='Request refused',reason='RETAINED_REFUSAL_REQUIRES_REVIEW');return out
    if out['backup']!='Native reports preserved':
        out.update(reason='SAVE_REQUIRED' if out['backup']=='Awaiting backup' else 'PRESERVATION_UNCONFIRMED');return out
    out.update(action='resume' if matching else 'start',reason='NATIVE_ADMISSION_STILL_REQUIRED')
    return out


def recent_requests(queue,job):
    if not hasattr(queue,'connect'):return []
    native_key=canonical_hash([str(Path(job.workspace).resolve()),job.native_job_id])
    with closing(queue.connect()) as db:
        # Include pause aliases via their resolved workspace, not just web identity.
        rows=[]
        for operation in ('run','pause'):
            rows.extend(db.execute("SELECT * FROM requests WHERE operation=? AND (target=? OR run_key=? OR (operation='pause' AND json_extract(command,'$.argv[#-2]')=?)) ORDER BY created DESC LIMIT 1",(operation,job.id,native_key,str(Path(job.workspace).resolve()))).fetchall())
        rows.sort(key=lambda r:r['created'],reverse=True)
    selected=[]
    for row in rows:
        command=json.loads(row['command']);argv=command['argv']
        matches=row['target']==job.id or row['run_key']==native_key or (row['operation']=='pause' and len(argv)>=2 and argv[-2]==str(Path(job.workspace).resolve()))
        if matches and not any(r['operation']==row['operation'] for r in selected):
            selected.append({k:row[k] for k in ('id','operation','status','created','updated','error')})
        if len(selected)==2:break
    return selected


def observe(settings,queue,adapter,job):
    root=contained(Path(settings.workspace_root),job.workspace,directory=True)
    reg=read_json(contained(root,'registry/'+job.native_job_id+'.json',directory=False))
    if reg.get('schema_id')!='IG_DECODER_WORKSPACE_JOB_V1' or reg.get('job_id')!=job.native_job_id or reg.get('source_sha256')!=PIN_SOURCE or reg.get('registration_sha256')!=canonical_hash({k:v for k,v in reg.items() if k!='registration_sha256'}):raise AdapterError('REGISTRATION_MISMATCH',409)
    def read(operation):
        try:
            value=adapter.observe(operation,job)
            return value.model_dump() if hasattr(value,'model_dump') else value
        except AdapterError as exc:return {'classification':'UNAVAILABLE','error':exc.code,'native':None}
    status=read('status');preservation={'capture':read('pending-saves'),'outbox':read('preservation')}
    try:result=results_observation(settings,job)
    except AdapterError as exc:result={'record_status':'UNAVAILABLE','error':exc.code}
    record=(status.get('native') or {}).get('last_record')
    if isinstance(record,dict) and record.get('job_id')==job.native_job_id and (record.get('registration_sha256')!=reg['registration_sha256'] or record.get('source_sha256')!=PIN_SOURCE):
        status=dict(status,classification='REFUSED',error='STATUS_IDENTITY_MISMATCH')
    requests=recent_requests(queue,job);state=project(job.native_job_id,status,preservation,requests,result)
    runtime='NOT_CHECKED'
    if state['action'] in ('start','resume'):
        try:
            adapter.verify_source(contained(root,'source',directory=True));adapter.verify_runtime();runtime='PROFILE_MATCH'
        except AdapterError as exc:state.update(action=None,reason=exc.code);runtime='UNAVAILABLE'
    if not hasattr(queue,'submit') or not hasattr(queue,'connect'):state.update(action=None,reason='BACKGROUND_WORKER_NOT_CONFIGURED')
    identity={'job_id':job.id,'registration':reg['registration_sha256'],'native':status.get('native'),
              'capture':preservation['capture'].get('native'),'outbox':preservation['outbox'].get('native'),
              'requests':requests,'result_status':result['record_status'],'state':state}
    # Outbox age is wall-clock telemetry, not a change to the observation's identity.
    if isinstance(identity['outbox'],dict):identity['outbox']={k:v for k,v in identity['outbox'].items() if k!='oldest_pending_age_seconds'}
    return {'job':{'id':job.id,'name':job.name,'native_job_id':job.native_job_id,'question':reg.get('question',{}).get('description','')},
            'observed_at':time.time(),'view_token':canonical_hash(identity),'state':state,'runtime':runtime,
            'native_status':status,'preservation':preservation,'requests':requests,'result':result,
            'admission':'Native source, environment, save and execution gates remain authoritative.'}


def submit(settings,queue,adapter,job,body):
    if not hasattr(queue,'connect'):raise AdapterError('BACKGROUND_WORKER_NOT_CONFIGURED')
    operation='pause' if body.action=='pause' else 'run'
    key='run-view:'+canonical_hash({'job':job.id,'view':body.view_token,'action':body.action})
    command=adapter.command(operation,job,reason=body.reason if operation=='pause' else None)
    digest=command_digest(command)
    with closing(queue.connect()) as db:old=db.execute('SELECT id,digest,status FROM requests WHERE key=?',(key,)).fetchone()
    if old:
        if old['digest']!=digest:raise AdapterError('IDEMPOTENCY_CONFLICT',409)
        return AcceptedRequest(request_id=old['id'],status=old['status'])
    view=observe(settings,queue,adapter,job)
    if view['view_token']!=body.view_token:raise AdapterError('OBSERVATION_CHANGED_REFRESH_REQUIRED',409)
    if view['state']['action']!=body.action:raise AdapterError('ACTION_UNAVAILABLE',409)
    return queue.submit(command,key,digest)
