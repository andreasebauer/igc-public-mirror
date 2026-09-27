"""Three independent result facts. Read-only; no execution or preservation mutation."""
from pathlib import Path
import time

from .native import AdapterError, canonical_hash, contained, read_json
from .results import results_observation


def save_summary(capture, outbox):
    summary={'status':'UNKNOWN','capture_pending':None,'checkpoint_objects_pending':None,
             'checkpoints_pending':None,'scope':'WORKSPACE_NATIVE_OBSERVATION'}
    if capture.get('classification')!='RESPONSE' or outbox.get('classification')!='RESPONSE':return summary
    c=capture.get('native');o=outbox.get('native')
    if not isinstance(c,dict) or not isinstance(o,dict):return summary
    lists=(c.get('pending_objects'),o.get('pending_objects'),o.get('pending_checkpoints'))
    if not all(isinstance(x,list) for x in lists):return summary
    summary.update(capture_pending=len(lists[0]),checkpoint_objects_pending=len(lists[1]),checkpoints_pending=len(lists[2]))
    if any(lists):summary['status']='PENDING'
    elif c.get('status')=='SAVED' and o.get('status')=='PRESERVED':summary['status']='NATIVE_REPORTS_PRESERVED'
    elif c.get('status')=='SAVED' and o.get('status')=='NO_CHECKPOINT':summary['status']='NO_CHECKPOINT'
    return summary


def result_facts(result):
    state=result.get('record_status','UNAVAILABLE')
    known=state in {'PUBLISHED_RECORD','PENDING_CHECKPOINT_RECORD'}
    reported=result.get('reported') if known else None
    if not isinstance(reported,dict):reported={}
    evidence=reported.get('evidence_status')
    return {'record_status':state,
            'scientific_outcome':{'value':reported.get('scientific_outcome'),'scope':'NATIVE_REPORTED' if known else 'UNKNOWN'},
            'evidence':{'reported_status':evidence if evidence in {'VERIFIED','REJECTED','PENDING','UNKNOWN'} else 'UNKNOWN',
                        'artifact_reverification':'NOT_RUN','scope':'NATIVE_COMPLETION_REPORT' if known else 'UNKNOWN'},
            'execution_reported':reported.get('execution_status'),
            'completion_reported':reported.get('status'),
            'publication':'PUBLISHED' if state=='PUBLISHED_RECORD' else 'PENDING_CHECKPOINT' if state=='PENDING_CHECKPOINT_RECORD' else 'NONE',
            'record_integrity':result.get('record_integrity','NOT_CONFIRMED'),
            'reusable':False}


def observe(settings,adapter,job):
    try:result=results_observation(settings,job)
    except AdapterError as exc:result={'job_id':job.id,'record_status':'UNAVAILABLE','error':exc.code}
    def read(operation):
        try:
            value=adapter.observe(operation,job)
            return value.model_dump() if hasattr(value,'model_dump') else value
        except AdapterError as exc:return {'classification':'UNAVAILABLE','native':None,'error':exc.code}
    capture=read('pending-saves');outbox=read('preservation')
    # A returned capture observation must identify the selected native job.
    if capture.get('classification')=='RESPONSE' and (capture.get('native') or {}).get('job_id')!=job.native_job_id:
        capture={'classification':'UNAVAILABLE','native':None,'error':'CAPTURE_JOB_MISMATCH'}
    question=None
    try:
        root=contained(Path(settings.workspace_root),job.workspace,directory=True)
        reg=read_json(contained(root,'registry/'+job.native_job_id+'.json',directory=False))
        if reg.get('job_id')==job.native_job_id and reg.get('registration_sha256')==canonical_hash({k:v for k,v in reg.items() if k!='registration_sha256'}):
            candidate=reg.get('question',{}).get('description')
            if isinstance(candidate,str):question=candidate
    except (AdapterError,TypeError,AttributeError):pass
    return {'job':{'id':job.id,'name':job.name,'native_job_id':job.native_job_id,'question':question},
            'observed_at':time.time(),'facts':result_facts(result),'save':save_summary(capture,outbox),
            'result':result,'preservation':{'capture':capture,'outbox':outbox},
            'scope':'Recorded scientific and evidence reports, plus current native workspace save observation. No independent artifact re-verification or reuse authorization.'}
