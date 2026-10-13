"""Explicit connector/operator queue: acknowledgments require exact readback bytes.

No credentials, implied upload success or self-issued provider IDs. The operator
services each request through their connected provider and supplies real readbacks.
"""
from pathlib import Path
import json,time,uuid
from .canon import write_json_atomic
from . import submission as sub,preservation as pr


class OperatorQueueTransport:
    def __init__(self,queue):
        self.queue=Path(queue).resolve();self.queue.mkdir(parents=True,exist_ok=True)
    def _request(self,workspace,scope,obligations,physical):
        request_id=uuid.uuid4().hex
        request={'schema_id':'IG_OPERATOR_SAVE_REQUEST_V1','request_id':request_id,
                 'workspace':str(Path(workspace).resolve()),'scope':scope,
                 'capture_id':sub.capture_record(workspace)['capture_id'],
                 'obligations':obligations,'physical_objects':physical}
        write_json_atomic(self.queue/(request_id+'.request.json'),request)
        response=self.queue/(request_id+'.response.json')
        while not response.exists():time.sleep(0.25)
        if response.is_symlink():raise sub.SubmissionError('OPERATOR_RESPONSE_SYMLINK')
        reply=json.loads(response.read_bytes())
        if set(reply)!={'schema_id','request_id','objects'} or reply['schema_id']!='IG_OPERATOR_SAVE_RESPONSE_V1' or reply['request_id']!=request_id:
            raise sub.SubmissionError('OPERATOR_RESPONSE_BINDING')
        rows=reply['objects']
        if type(rows) is not list or len(rows)!=len(physical):raise sub.SubmissionError('OPERATOR_RESPONSE_COVERAGE')
        by_sha={x['sha256']:x for x in rows}
        if len(by_sha)!=len(rows) or set(by_sha)!={x['sha256'] for x in physical}:
            raise sub.SubmissionError('OPERATOR_RESPONSE_COVERAGE')
        for item in physical:
            row=by_sha[item['sha256']]
            if set(row)!={'sha256','readback_path','drive_file_id'}:raise sub.SubmissionError('OPERATOR_RESPONSE_FIELDS')
            path=Path(row['readback_path'])
            if path.is_symlink() or not path.is_file():raise sub.SubmissionError('OPERATOR_READBACK_REQUIRED')
            raw=path.read_bytes()
            if sub._sha(raw)!=item['sha256'] or len(raw)!=item['size_bytes']:raise sub.SubmissionError('SAVE_READBACK_MISMATCH')
        for item in obligations:
            row=by_sha[item['sha256']]
            if scope=='CAPTURE_SAVE':sub.confirm_save(workspace,item['sha256'],row['readback_path'],row['drive_file_id'],role=item['role'],logical_name=item['logical_name'])
            else:pr.confirm(workspace,item['sha256'],row['readback_path'],row['drive_file_id'],obligation_id=item['obligation_id'])
        write_json_atomic(self.queue/(request_id+'.confirmed.json'),{'request_id':request_id,'status':'CONFIRMED_EXACT_BYTES','scope':scope})
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
