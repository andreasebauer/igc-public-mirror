"""Explicit connector/operator queue: acknowledgments require exact readback bytes.

No credentials, implied upload success or self-issued provider IDs. The operator
services each request through their connected provider and supplies real readbacks.
"""
from pathlib import Path
import json,os,re,time,uuid
from .canon import canonical_bytes,write_json_atomic
from . import submission as sub,preservation as pr


class OperatorQueueTransport:
    def __init__(self,queue):
        self.queue=Path(queue).resolve();self.queue.mkdir(parents=True,exist_ok=True)
    def _request(self,workspace,scope,obligations,physical):
        if scope not in ('CAPTURE_SAVE','CHECKPOINT_OUTBOX'):
            raise sub.SubmissionError('OPERATOR_REQUEST_SCOPE')
        request={'schema_id':'IG_OPERATOR_SAVE_REQUEST_V1',
                 'workspace':str(Path(workspace).resolve()),'scope':scope,
                 'capture_id':sub.capture_record(workspace)['capture_id'],
                 'obligations':obligations,'physical_objects':physical}
        # Retain an identical outstanding request across a stopped operator call.
        # This reuses passive transport data, never a completion/save authority.
        request_id=None
        request_bytes=canonical_bytes(request)
        for path in sorted(self.queue.glob('*.request.json')):
            # History is discovery data. An unrelated torn, malformed, or
            # symlinked candidate cannot deny a new native save request.
            try:
                if path.is_symlink() or not path.is_file():continue
                prior=json.loads(path.read_bytes())
                if (type(prior) is not dict or set(prior)!=set(request)|{'request_id'}
                        or type(prior['request_id']) is not str
                        or re.fullmatch('[0-9a-f]{32}',prior['request_id']) is None
                        or path.name!=prior['request_id']+'.request.json'):
                    continue
                matches=canonical_bytes({k:v for k,v in prior.items() if k!='request_id'})==request_bytes
            except (OSError,ValueError,TypeError):
                continue
            if matches:
                # A .confirmed marker is observational only. Reuse still reads
                # and verifies the exact response and invokes native confirmation.
                request_id=prior['request_id'];break
        if request_id is None:
            request_id=uuid.uuid4().hex
            write_json_atomic(self.queue/(request_id+'.request.json'),dict(request,request_id=request_id))
        started=time.monotonic();last_write=0.0
        def publish(status, *, force=False, exc=None):
            nonlocal last_write
            now=time.monotonic()
            if not force and now-last_write<5:return
            row={'schema_id':'IG_OPERATOR_TRANSPORT_STATUS_V1','request_id':request_id,
                 'capture_id':request['capture_id'],'scope':scope,'status':status,
                 'pid':os.getpid(),'heartbeat_unix':time.time(),'elapsed_seconds':now-started,
                 'obligations':len(obligations),'physical_objects':len(physical),
                 'physical_bytes':sum(item['size_bytes'] for item in physical),
                 'completion_authority':'NONE; OBSERVATION_ONLY'}
            if exc is not None:
                row['termination']={'exception_type':type(exc).__name__,'reason':str(exc)[:1000],
                    'code':getattr(exc,'code',None)}
            try:write_json_atomic(self.queue/'OPERATOR_TRANSPORT_STATUS.json',row)
            except OSError:pass
            last_write=now
        try:
            response=self.queue/(request_id+'.response.json')
            publish('WAITING_FOR_OPERATOR',force=True)
            while not response.exists():
                if response.is_symlink():raise sub.SubmissionError('OPERATOR_RESPONSE_SYMLINK')
                publish('WAITING_FOR_OPERATOR');time.sleep(0.25)
            if response.is_symlink():raise sub.SubmissionError('OPERATOR_RESPONSE_SYMLINK')
            publish('VERIFYING_READBACKS',force=True)
            reply=json.loads(response.read_bytes())
            if type(reply) is not dict or set(reply)!={'schema_id','request_id','objects'} or reply['schema_id']!='IG_OPERATOR_SAVE_RESPONSE_V1' or reply['request_id']!=request_id:
                raise sub.SubmissionError('OPERATOR_RESPONSE_BINDING')
            rows=reply['objects']
            if type(rows) is not list or len(rows)!=len(physical):raise sub.SubmissionError('OPERATOR_RESPONSE_COVERAGE')
            if any(type(row) is not dict or set(row)!={'sha256','readback_path','drive_file_id'}
                   or any(type(value) is not str for value in row.values()) for row in rows):
                raise sub.SubmissionError('OPERATOR_RESPONSE_FIELDS')
            by_sha={x['sha256']:x for x in rows}
            if len(by_sha)!=len(rows) or set(by_sha)!={x['sha256'] for x in physical}:
                raise sub.SubmissionError('OPERATOR_RESPONSE_COVERAGE')
            from .checkpoint_files import verify_file
            for item in physical:
                path=Path(by_sha[item['sha256']]['readback_path'])
                if path.is_symlink() or not path.is_file():raise sub.SubmissionError('OPERATOR_READBACK_REQUIRED')
                verify_file(path,item,'SAVE_READBACK_MISMATCH')
                publish('VERIFYING_READBACKS')
            publish('CONFIRMING_NATIVE_SAVE',force=True)
            if scope=='CAPTURE_SAVE':
                for item in obligations:
                    row=by_sha[item['sha256']]
                    sub.confirm_save(workspace,item['sha256'],row['readback_path'],row['drive_file_id'],role=item['role'],logical_name=item['logical_name'])
                    publish('CONFIRMING_NATIVE_SAVE')
            elif scope=='CHECKPOINT_OUTBOX':
                # Use the established streaming, journalled batch confirmer. Each
                # logical role remains bound; repeated object reads are bounded.
                batch=self.queue/(request_id+'.ack-batch.json')
                write_json_atomic(batch,{'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1',
                    'capture_id':request['capture_id'],'obligations':obligations,
                    'readbacks':[{'sha256':row['sha256'],'kind':'RAW','path':row['readback_path'],
                        'drive_file_id':row['drive_file_id']} for row in rows]})
                pr.confirm_batch(workspace,batch)
            write_json_atomic(self.queue/(request_id+'.confirmed.json'),{'request_id':request_id,'status':'CONFIRMED_EXACT_BYTES','scope':scope})
            publish('CONFIRMED_EXACT_BYTES',force=True)
        except BaseException as exc:
            publish('STOPPED',force=True,exc=exc);raise
    def save_capture(self,workspace):
        pending=sub.save_status(workspace)['pending_objects']
        if not pending:return
        physical={}
        for x in pending:
            physical[x['sha256']]={'sha256':x['sha256'],'size_bytes':x['size_bytes'],
                'path':x['local_object_path']}
        self._request(workspace,'CAPTURE_SAVE',pending,list(physical.values()))
        if sub.save_status(workspace)['status']!='SAVED':raise sub.SubmissionError('CAPTURE_OPERATOR_SAVE_INCOMPLETE')
    def preserve_workspace(self,workspace):
        while pr.status(workspace)['pending_objects']:
            plan=pr.drain_plan(workspace)
            self._request(workspace,'CHECKPOINT_OUTBOX',plan['obligations'],plan['physical_objects'])
        if pr.status(workspace)['status']!='PRESERVED':raise sub.SubmissionError('CHECKPOINT_OPERATOR_SAVE_INCOMPLETE')
