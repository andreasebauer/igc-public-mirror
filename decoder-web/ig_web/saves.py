"""Durable Drive saves. Only native confirmations can satisfy save obligations."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .models import Settings
from .native import AdapterError,NativeAdapter,PIN_SOURCE,canonical_hash,contained,read_json
from .tracking import catalogue,file_hash,lane_lock
from .worker import Queue


class Saves:
    def __init__(self,settings,queue):
        if not hasattr(queue,'connect'):raise AdapterError('BACKGROUND_WORKER_NOT_CONFIGURED')
        self.settings,self.queue=settings,queue
        with queue.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS saves (id TEXT PRIMARY KEY,key TEXT UNIQUE,binding TEXT,native_key TEXT,target TEXT,status TEXT,detail TEXT,created REAL,updated REAL)')
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS active_save ON saves(native_key) WHERE status!='finished'")

    def get(self,rid,private=False):
        with closing(self.queue.connect()) as db:row=db.execute('SELECT * FROM saves WHERE id=?',(rid,)).fetchone()
        if row is None:raise AdapterError('SAVE_REQUEST_NOT_FOUND',404)
        row=dict(row);row['detail']=json.loads(row['detail'])
        if private:return row
        d=row['detail']
        return {k:row[k] for k in ('id','target','status','created','updated')} | {
            'error':d.get('error'),'native_remaining':d.get('remaining'),
            'objects':[{'sha256':o['sha256'],'size_bytes':o['size_bytes'],'phase':o.get('phase','pending'),
                        'drive_id':o.get('drive_id')} for o in d.get('objects',[])],
            'confirmed_obligations':len(d.get('confirmed',[])),
            'save_authority':'NATIVE_OBSERVATION_ONLY'}

    def history(self,job):
        with closing(self.queue.connect()) as db:rows=db.execute('SELECT id FROM saves WHERE target=? ORDER BY created DESC LIMIT 50',(job.id,)).fetchall()
        return {'transport':'CONFIGURED_NOT_QUALIFIED' if self.settings.drive_token_file else 'UNCONFIGURED',
                'items':[self.get(r['id']) for r in rows]}

    def submit(self,job,key):
        binding=canonical_hash(job.model_dump());native_key=canonical_hash([job.workspace,job.native_job_id])
        with self.queue.transaction() as db:
            existing=db.execute('SELECT * FROM saves WHERE key=?',(key,)).fetchone()
            if existing:
                if existing['binding']!=binding:raise AdapterError('IDEMPOTENCY_CONFLICT',409)
                return {'request_id':existing['id'],'status':existing['status']}
            if not self.settings.drive_token_file:raise AdapterError('DRIVE_NOT_CONFIGURED')
            if db.execute("SELECT 1 FROM saves WHERE native_key=? AND status!='finished'",(native_key,)).fetchone():
                raise AdapterError('SAVE_ALREADY_ACTIVE',409)
            rid='save-'+uuid.uuid4().hex;now=time.time()
            db.execute('INSERT INTO saves VALUES (?,?,?,?,?,?,?,?,?)',(rid,key,binding,native_key,job.id,'queued','{}',now,now))
        return {'request_id':rid,'status':'queued'}

    def write(self,row,status):
        with self.queue.transaction() as db:db.execute('UPDATE saves SET status=?,detail=?,updated=? WHERE id=?',
            (status,json.dumps(row['detail']),time.time(),row['id']))

    def retry(self,rid):
        # Must take the same lock as transfer and native confirmation. An active
        # worker cannot be restarted by a second HTTP request.
        with lane_lock(self.queue,'save'):
            row=self.get(rid,True)
            if row['status'] not in ('needs_reconciliation','failed','running'):
                return self.get(rid)
            row['detail'].pop('error',None);self.write(row,'queued')
        return self.get(rid)

    def claim(self):
        with self.queue.transaction() as db:
            db.execute("UPDATE saves SET status='needs_reconciliation',updated=? WHERE status='running'",(time.time(),))
            if db.execute("SELECT 1 FROM saves WHERE status='needs_reconciliation'").fetchone():return None
            row=db.execute("SELECT id FROM saves WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row:db.execute("UPDATE saves SET status='running',updated=? WHERE id=?",(time.time(),row['id']))
        return self.get(row['id'],True) if row else None


class NativeSave:
    def __init__(self,settings,job):
        self.settings,self.job=settings,job;self.adapter=NativeAdapter(settings)
        self.adapter.verify_source();self.adapter.verify_runtime()
        self.workspace=contained(Path(settings.workspace_root),job.workspace,directory=True)
        self.source=contained(self.workspace,'source',directory=True);self.adapter.verify_source(self.source)
        if read_json(self.workspace/'WORKSPACE.json').get('source_sha256')!=PIN_SOURCE:
            raise AdapterError('CAPTURE_SOURCE_NOT_SUPPORTED')

    def pending(self):
        result=[]
        for scope,operation in [('capture','pending-saves'),('outbox','preservation')]:
            observation=self.adapter.observe(operation,self.job)
            if observation.classification!='RESPONSE' or not isinstance(observation.native,dict):raise AdapterError('SAVE_OBSERVATION_UNAVAILABLE')
            native=observation.native
            if scope=='capture' and native.get('job_id')!=self.job.native_job_id:raise AdapterError('CAPTURE_JOB_MISMATCH')
            pending=native.get('pending_objects')
            if not isinstance(pending,list):raise AdapterError('SAVE_OBSERVATION_INVALID')
            for row in pending:
                if (not isinstance(row,dict) or not re.fullmatch('[0-9a-f]{64}',row.get('sha256',''))
                        or type(row.get('size_bytes')) is not int or row['size_bytes']<0
                        or not all(isinstance(row.get(k),str) and row[k] and '\x00' not in row[k]
                                   for k in ('role','logical_name','obligation_id','local_object_path'))):
                    raise AdapterError('SAVE_OBLIGATION_INVALID')
                result.append(dict(row,scope=scope))
        return result

    def path(self,obj):
        for root in (self.settings.workspace_root,self.settings.capture_store):
            try:return contained(Path(root),obj['local_object_path'],directory=False)
            except AdapterError:pass
        raise AdapterError('SAVE_OBJECT_PATH_NOT_ALLOWED')

    def confirm(self,obj,readback,drive_id,directory):
        args=['confirm-save',str(self.workspace)] if obj['scope']=='capture' else ['preserve','confirm',str(self.workspace)]
        args += [obj['sha256'],str(readback),drive_id,'--role',obj['role'],'--logical-name',obj['logical_name']]
        if obj['scope']=='outbox':args+=['--obligation-id',obj['obligation_id']]
        prefix='confirm-'+uuid.uuid4().hex
        with (directory/(prefix+'.stdout')).open('xb') as out,(directory/(prefix+'.stderr')).open('xb') as err:
            result=subprocess.run([self.settings.engine_python,'-B','-m','infinity_grid.controller',*args],
                cwd=self.source,env=self.adapter.environment(self.source),stdin=subprocess.DEVNULL,stdout=out,stderr=err,shell=False)
            out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
        if result.returncode:raise AdapterError('NATIVE_SAVE_REFUSED',409)


def identity(obj):return (obj['scope'],obj['obligation_id'],obj['sha256'],obj['role'],obj['logical_name'])


def process(service,row,transport,native):
    directory=service.queue.root/row['id'];directory.mkdir(mode=0o700,exist_ok=True)
    contained(service.queue.root,str(directory),directory=True)
    d=row['detail']
    def persist():service.write(row,'running')
    try:
        pending=native.pending()
        if 'obligations' not in d:
            d.update(obligations=pending,objects=[],confirmed=[])
            by_sha={}
            for obj in pending:
                if obj['sha256'] in by_sha:
                    if by_sha[obj['sha256']]['size_bytes']!=obj['size_bytes']:raise AdapterError('SAVE_SIZE_CONFLICT')
                else:
                    by_sha[obj['sha256']]=dict(obj,phase='pending');d['objects'].append(by_sha[obj['sha256']])
            persist()
        current={identity(o) for o in pending}
        for obj in d['objects']:
            obligations=[o for o in d['obligations'] if o['sha256']==obj['sha256'] and identity(o) in current]
            if not obligations:continue
            path=native.path(obj)
            if path.stat().st_size!=obj['size_bytes'] or file_hash(path)!=obj['sha256']:raise AdapterError('SAVE_SOURCE_HASH_MISMATCH',409)
            if not obj.get('drive_id'):
                obj['drive_id']=transport.reserve();persist()
            from .drive import file_id
            file_id(obj['drive_id'])
            if obj['phase'] not in ('uploaded','readback_verified','confirmed'):
                obj['phase']='uploading';persist();transport.upload(path,obj,persist)
                obj.pop('session',None);obj['phase']='uploaded';persist()
            readback=directory/(obj['sha256']+'.readback')
            if readback.is_symlink():raise AdapterError('PATH_NOT_ALLOWED')
            if not readback.exists() or file_hash(readback)!=obj['sha256'] or readback.stat().st_size!=obj['size_bytes']:
                # Retain incomplete downloads as evidence; never use their bytes.
                attempt=directory/(obj['sha256']+'.part-'+uuid.uuid4().hex)
                transport.download(obj['drive_id'],attempt,obj['size_bytes'])
                if file_hash(attempt)!=obj['sha256'] or attempt.stat().st_size!=obj['size_bytes']:raise AdapterError('SAVE_READBACK_MISMATCH',409)
                os.replace(attempt,readback)
            obj['phase']='readback_verified';persist()
            for obligation in obligations:
                # Reobserve exact roles before each mutation, including recovery
                # after native confirmation succeeded but the worker crashed.
                if identity(obligation) in {identity(o) for o in native.pending()}:
                    native.confirm(obligation,readback,obj['drive_id'],directory)
                    if identity(obligation) in {identity(o) for o in native.pending()}:raise AdapterError('NATIVE_SAVE_STILL_PENDING',409)
                key=list(identity(obligation))
                if key not in d['confirmed']:d['confirmed'].append(key)
                persist()
            obj['phase']='confirmed';persist()
        remaining=native.pending();d['remaining']={'capture':sum(o['scope']=='capture' for o in remaining),'outbox':sum(o['scope']=='outbox' for o in remaining)}
        if {identity(o) for o in remaining}&{identity(o) for o in d['obligations']}:raise AdapterError('SAVE_STILL_PENDING',409)
        d.pop('error',None);service.write(row,'finished')
    except Exception as exc:
        d['error']=exc.code if isinstance(exc,AdapterError) else type(exc).__name__
        service.write(row,'failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    settings=Settings.model_validate(read_json(Path(args.config)));os.umask(0o077)
    queue=Queue(settings.worker_state);service=Saves(settings,queue)
    from .drive import Drive
    while True:
        with lane_lock(queue,'save'):
            row=service.claim()
            if row:
                transport=None
                try:
                    job=next((j for j in catalogue(settings,queue).jobs if j.id==row['target']),None)
                    if job is None or canonical_hash(job.model_dump())!=row['binding']:raise AdapterError('SAVE_JOB_CHANGED',409)
                    transport=Drive(settings)
                    process(service,row,transport,NativeSave(settings,job))
                except Exception as exc:
                    row['detail']['error']=exc.code if isinstance(exc,AdapterError) else type(exc).__name__;service.write(row,'failed')
                finally:
                    if transport:transport.client.close()
        if args.once:return
        if row is None:time.sleep(0.5)


if __name__=='__main__':main()
