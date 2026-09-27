"""Persistent web index and transport reconciliation; no native execution authority."""
from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from .models import Catalog, Job
from .native import AdapterError, NativeAdapter, PIN_SOURCE, canonical_hash, contained, read_json


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_record(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('x') as out:
        json.dump(value,out,sort_keys=True,allow_nan=False)
        out.flush(); os.fsync(out.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def catalogue(settings, queue=None):
    catalog=Catalog.model_validate(read_json(Path(settings.catalog)))
    if queue:
        with closing(queue.connect()) as db:
            catalog.jobs.extend(Job.model_validate(json.loads(r[0])) for r in db.execute('SELECT job FROM jobs ORDER BY id'))
    for group in (catalog.jobs,catalog.tasks,catalog.inputs):
        ids={}
        for item in group:
            if item.id in ids:
                raise AdapterError('CATALOG_DUPLICATE_ID')
            ids[item.id]=item
    return catalog


def capture_job(settings, row, directory, native):
    """Bind native capture identity to frozen submitted intent before indexing."""
    spec_path=contained(directory,'specification.json',directory=False)
    command=json.loads(row['command'])
    if file_hash(spec_path)!=command['specification_sha256']:
        raise AdapterError('FROZEN_SPECIFICATION_MISMATCH')
    spec=read_json(spec_path)
    workspace=contained(Path(settings.capture_store)/'captures',native['workspace'],directory=True)
    contained(Path(settings.workspace_root),str(workspace),directory=True)
    def record(name): return read_json(contained(workspace,name,directory=False))
    rec=record('CAPTURE.json')
    cid=rec.get('capture_id')
    if (rec.get('schema_id')!='IG_DECODER_CAPTURE_V1' or
        cid!=canonical_hash({k:v for k,v in rec.items() if k!='capture_id'}) or
        cid!=native.get('capture_id') or workspace.name!=cid):
        raise AdapterError('CAPTURE_IDENTITY_MISMATCH')
    job=rec['job']; jid=job['job_id']
    # Validate identifier before using it as a relative filename.
    indexed=Job(id='capture-'+cid,name=jid,native_job_id=jid,workspace=str(workspace))
    if (job!=record('registry/'+jid+'.json') or rec['workspace']!=record('WORKSPACE.json') or
        job.get('registration_sha256')!=canonical_hash({k:v for k,v in job.items() if k!='registration_sha256'}) or
        job.get('source_sha256')!=PIN_SOURCE or rec['workspace'].get('source_sha256')!=PIN_SOURCE or
        jid!=spec['job_id'] or jid!=native.get('job_id') or
        any(job[k]!=spec[k] for k in ('execution','resources','question')) or
        rec['output_contract']!=spec['output_contract'] or rec.get('repeat_id')!=spec.get('repeat_id')):
        raise AdapterError('CAPTURE_INTENT_MISMATCH')
    env=dict(spec['environment'],artifacts=[{'logical_name':x['logical_name'],'sha256':x['sha256']} for x in spec['environment']['artifacts']])
    expected=[{'logical_name':x['logical_name'],'sha256':x['sha256']} for x in spec['inputs']+spec['environment']['artifacts']]
    actual=[x for x in job['input_artifacts'] if x['logical_name']!='submission_contract']
    if env!=rec['environment'] or sorted(expected,key=lambda x:x['logical_name'])!=sorted(actual,key=lambda x:x['logical_name']):
        raise AdapterError('CAPTURE_DEPENDENCY_MISMATCH')
    NativeAdapter(settings).verify_source(contained(workspace,'source',directory=True))
    return indexed,cid,job['registration_sha256']


def finalize(settings, queue, row):
    directory=contained(queue.root,row['id'],directory=True)
    receipt=read_json(contained(directory,'exit.json',directory=False))
    if (receipt.get('schema')!='IG_WEB_EXIT_V1' or receipt.get('request_id')!=row['id'] or
        receipt.get('digest')!=row['digest'] or type(receipt.get('exit_code')) is not int):
        raise AdapterError('EXIT_RECEIPT_MISMATCH')
    for name in ('stdout','stderr'):
        if file_hash(contained(directory,name,directory=False))!=receipt.get(name+'_sha256'):
            raise AdapterError('EXIT_LOG_MISMATCH')
    rc=receipt['exit_code']; native=None
    for name in (('stderr','stdout') if rc else ('stdout','stderr')):
        try:
            value=read_json(directory/name)
            if isinstance(value,dict): native=value; break
        except AdapterError: pass
    state='needs_reconciliation' if native is None else 'refused' if rc else 'finished'
    indexed=None
    if row['operation']=='capture' and rc==0 and native is not None:
        indexed=capture_job(settings,row,directory,native)
    # Index and terminal request transition are one durable transaction.
    with queue.transaction() as db:
        current=db.execute('SELECT status FROM requests WHERE id=?',(row['id'],)).fetchone()
        if current is None: raise AdapterError('REQUEST_NOT_FOUND',404)
        if indexed:
            job,cid,registration=indexed
            existing=db.execute('SELECT job FROM jobs WHERE id=?',(job.id,)).fetchone()
            if existing and json.loads(existing[0])!=job.model_dump():
                raise AdapterError('JOB_INDEX_CONFLICT')
            db.execute('INSERT OR IGNORE INTO jobs(id,capture_id,registration,job,created) VALUES (?,?,?,?,?)',
                       (job.id,cid,registration,json.dumps(job.model_dump()),time.time()))
        db.execute('UPDATE requests SET status=?,exit_code=?,classification=?,native=?,error=NULL,updated=? WHERE id=?',
                   (state,rc,'UNPARSEABLE' if native is None else 'REFUSED' if rc else 'RESPONSE',
                    json.dumps(native) if native is not None else None,time.time(),row['id']))
        queue.event(db,row['id'],'EXIT_RECONCILED',{'status':state,'job_id':indexed[0].id if indexed else None})
    return queue.get(row['id'])


@contextmanager
def lane_lock(queue,lane):
    with (queue.root/(lane+'.lock')).open('a+b') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise AdapterError('WORKER_LANE_ACTIVE',409) from exc
        yield lock


def reconcile(settings,queue,request_id):
    with closing(queue.connect()) as db:
        raw=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
    if raw is None: raise AdapterError('REQUEST_NOT_FOUND',404)
    row=dict(raw)
    # Same lock as dispatch, including inherited native-child descriptor.
    with lane_lock(queue,row['lane']):
        if row['status'] in ('queued','finished','refused','interrupted'):
            return queue.get(request_id)
        path=queue.root/request_id/'exit.json'
        if path.exists() or path.is_symlink():
            try: return finalize(settings,queue,row)
            except Exception as exc:
                queue.update(request_id,status='needs_reconciliation',error=exc.code if isinstance(exc,AdapterError) else type(exc).__name__)
                raise AdapterError('RECONCILIATION_REFUSED',409) from exc
        observation={'status':'DISPATCH_UNRESOLVED','automatic_replay':False,'observed_at':time.time()}
        if row['operation']=='run':
            # Read native records for diagnosis only: an old completion does not
            # prove that this particular web request finished or even started.
            from .results import results_observation
            job=next((j for j in catalogue(settings,queue).jobs if j.id==row['target']),None)
            if job:
                try: observation['native_result']=results_observation(settings,job)
                except AdapterError as exc: observation['native_error']=exc.code
        queue.update(request_id,status='needs_reconciliation',error='EXIT_RECEIPT_ABSENT')
        with queue.transaction() as db: queue.event(db,request_id,'RECONCILIATION_BLOCKED',observation)
        return queue.get(request_id)
