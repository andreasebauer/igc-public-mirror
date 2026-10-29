"""Native resume outcomes and passive connector waits; no remote save claims."""
import json
from pathlib import Path

import pytest

from infinity_grid import operator_transport as ot, rolling_operator as op
from infinity_grid import bulk_submission as bulk, v05_controller_event_loop as loop
from infinity_grid.canon import write_json_atomic


@pytest.mark.parametrize('status', ['RESULT_REJECTED', 'VALIDATION_FAILED', None])
def test_resume_rejects_non_successful_native_completion(tmp_path, monkeypatch, status):
    workspace=tmp_path/'capture';(workspace/'runtime').mkdir(parents=True)
    monkeypatch.setattr(op.sub, 'capture_record', lambda p:{'capture_id':'c'*64,'job':{'job_id':'J'}})
    monkeypatch.setattr(loop, 'validate_workspace_job', lambda *a,**kw:{})
    monkeypatch.setattr(loop, 'verified_completion', lambda a:None if status is None else {'status':status})
    monkeypatch.setattr(bulk, '_run_saved_capture', lambda *a:{'status':status})
    with pytest.raises(ValueError, match='REGISTERED_COMPLETION'):
        op.resume_registered_capture(workspace, tmp_path/'queue')
    row=json.loads((tmp_path/'queue/OPERATING_STATUS.json').read_bytes())
    assert row['operation']=='saved_completion_verification' and row['status']=='STOPPED'
    assert row['termination']['exception_type']=='ValueError'
    assert row['updated_unix']>=row['operation_started_unix'] and row['pid']>0


def test_resume_retains_typed_native_failure_without_diagnostic_dependency(tmp_path, monkeypatch):
    workspace=tmp_path/'capture';(workspace/'runtime').mkdir(parents=True)
    monkeypatch.setattr(op.sub, 'capture_record', lambda p:{'capture_id':'c'*64,'job':{'job_id':'J'}})
    monkeypatch.setattr(loop, 'validate_workspace_job', lambda *a,**kw:{})
    error=ot.sub.SubmissionError('SAVE_READBACK_MISMATCH')
    def run(workspace, transport, operation):
        with operation('native_execution'):raise error
    monkeypatch.setattr(bulk, '_run_saved_capture', run)
    with pytest.raises(ot.sub.SubmissionError) as caught:
        op.resume_registered_capture(workspace, tmp_path/'queue')
    assert caught.value is error
    row=json.loads((tmp_path/'queue/OPERATING_STATUS.json').read_bytes())
    assert row['termination']['code']=='SAVE_READBACK_MISMATCH'
    monkeypatch.setattr(op, 'write_json_atomic', lambda *a,**kw:(_ for _ in ()).throw(OSError('diagnostic unavailable')))
    with pytest.raises(ot.sub.SubmissionError) as caught:
        op.resume_registered_capture(workspace, tmp_path/'queue')
    assert caught.value is error


def request_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(ot.sub, 'capture_record', lambda p:{'capture_id':'c'*64})
    readback=tmp_path/'readback';readback.write_bytes(b'actual retained bytes')
    sha=ot.sub._sha(readback.read_bytes())
    physical=[{'sha256':sha,'size_bytes':readback.stat().st_size,'path':str(readback)}]
    obligations=[{'sha256':sha,'role':'engine_source','logical_name':'engine'}]
    transport=ot.OperatorQueueTransport(tmp_path/'queue')
    return transport,readback,sha,physical,obligations


def test_wait_heartbeats_then_resumes_same_passive_request(tmp_path, monkeypatch):
    transport,readback,sha,physical,obligations=request_fixture(tmp_path,monkeypatch)
    clock=[0.0];observed=[]
    monkeypatch.setattr(ot.time, 'monotonic', lambda:clock[0])
    def wait(seconds):
        observed.append(json.loads((transport.queue/'OPERATOR_TRANSPORT_STATUS.json').read_bytes()))
        clock[0]+=6
        if len(observed)==2:raise KeyboardInterrupt()
    monkeypatch.setattr(ot.time, 'sleep', wait)
    with pytest.raises(KeyboardInterrupt):transport._request(tmp_path,'CAPTURE_SAVE',obligations,physical)
    assert [row['status'] for row in observed]==['WAITING_FOR_OPERATOR']*2
    assert observed[1]['elapsed_seconds']>observed[0]['elapsed_seconds']
    stopped=json.loads((transport.queue/'OPERATOR_TRANSPORT_STATUS.json').read_bytes())
    assert stopped['status']=='STOPPED' and stopped['termination']['exception_type']=='KeyboardInterrupt'
    request=json.loads(next(transport.queue.glob('*.request.json')).read_bytes())
    write_json_atomic(transport.queue/(request['request_id']+'.response.json'),
        {'schema_id':'IG_OPERATOR_SAVE_RESPONSE_V1','request_id':request['request_id'],
         'objects':[{'sha256':sha,'readback_path':str(readback),'drive_file_id':'UNIT_FIXTURE_NO_REMOTE_CREDIT'}]})
    confirms=[];monkeypatch.setattr(ot.sub, 'confirm_save', lambda *a,**kw:confirms.append(kw))
    transport._request(tmp_path,'CAPTURE_SAVE',obligations,physical)
    assert len(list(transport.queue.glob('*.request.json')))==1 and len(confirms)==1
    assert json.loads((transport.queue/'OPERATOR_TRANSPORT_STATUS.json').read_bytes())['status']=='CONFIRMED_EXACT_BYTES'


def test_changed_logical_obligation_cannot_reuse_pending_request(tmp_path, monkeypatch):
    transport,readback,sha,physical,obligations=request_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(ot.time, 'sleep', lambda seconds:(_ for _ in ()).throw(KeyboardInterrupt()))
    for role in ['engine_source','project_baseline']:
        with pytest.raises(KeyboardInterrupt):
            transport._request(tmp_path,'CAPTURE_SAVE',[dict(obligations[0],role=role)],physical)
    assert len(list(transport.queue.glob('*.request.json')))==2


def test_malformed_unrelated_history_cannot_block_new_save_request(tmp_path, monkeypatch):
    transport,readback,sha,physical,obligations=request_fixture(tmp_path,monkeypatch)
    (transport.queue/'broken.request.json').write_text('{broken historical JSON')
    (transport.queue/'nonfinite.request.json').write_text('{"old":NaN}')
    (transport.queue/'not-an-object.request.json').write_text('null')
    (transport.queue/'symlink.request.json').symlink_to(tmp_path/'missing')
    monkeypatch.setattr(ot.time,'sleep',lambda seconds:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):transport._request(tmp_path,'CAPTURE_SAVE',obligations,physical)
    row=json.loads((transport.queue/'OPERATOR_TRANSPORT_STATUS.json').read_bytes())
    request=json.loads((transport.queue/(row['request_id']+'.request.json')).read_bytes())
    assert request['capture_id']=='c'*64 and request['obligations']==obligations
    assert row['termination']['exception_type']=='KeyboardInterrupt'


def test_confirmed_marker_cannot_bypass_exact_response_verification(tmp_path, monkeypatch):
    transport,readback,sha,physical,obligations=request_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(ot.time,'sleep',lambda seconds:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):transport._request(tmp_path,'CAPTURE_SAVE',obligations,physical)
    request=json.loads(next(transport.queue.glob('*.request.json')).read_bytes());rid=request['request_id']
    write_json_atomic(transport.queue/(rid+'.confirmed.json'),{'status':'CONFIRMED_EXACT_BYTES'})
    write_json_atomic(transport.queue/(rid+'.response.json'),
        {'schema_id':'IG_OPERATOR_SAVE_RESPONSE_V1','request_id':'a-different-request',
         'objects':[{'sha256':sha,'readback_path':str(readback),'drive_file_id':'UNIT_FIXTURE_NO_REMOTE_CREDIT'}]})
    monkeypatch.setattr(ot.sub,'confirm_save',lambda *a,**kw:pytest.fail('marker granted save authority'))
    with pytest.raises(ot.sub.SubmissionError) as caught:
        transport._request(tmp_path,'CAPTURE_SAVE',obligations,physical)
    assert caught.value.code=='OPERATOR_RESPONSE_BINDING'
    assert len(list(transport.queue.glob('*.request.json')))==1


def test_checkpoint_transport_uses_existing_native_batch_confirmer(tmp_path, monkeypatch):
    transport,readback,sha,physical,obligations=request_fixture(tmp_path,monkeypatch)
    obligations=[dict(obligations[0],obligation_id='o'*64,obligation_scope='CHECKPOINT_OUTBOX',size_bytes=physical[0]['size_bytes'])]
    def reply(seconds):
        request=json.loads(next(transport.queue.glob('*.request.json')).read_bytes())
        write_json_atomic(transport.queue/(request['request_id']+'.response.json'),
            {'schema_id':'IG_OPERATOR_SAVE_RESPONSE_V1','request_id':request['request_id'],
             'objects':[{'sha256':sha,'readback_path':str(readback),'drive_file_id':'UNIT_FIXTURE_NO_REMOTE_CREDIT'}]})
    monkeypatch.setattr(ot.time,'sleep',reply)
    batches=[]
    monkeypatch.setattr(ot.pr,'confirm_batch',lambda workspace,path:batches.append(json.loads(path.read_bytes())))
    monkeypatch.setattr(ot.pr,'confirm',lambda *a,**kw:pytest.fail('per-role confirmer bypassed native batch route'))
    original=Path.read_bytes
    def bounded_read(path):
        assert path!=readback,'readback must be streamed'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',bounded_read)
    transport._request(tmp_path,'CHECKPOINT_OUTBOX',obligations,physical)
    assert len(batches)==1 and batches[0]['obligations']==obligations
    assert batches[0]['readbacks']==[{'sha256':sha,'kind':'RAW','path':str(readback),'drive_file_id':'UNIT_FIXTURE_NO_REMOTE_CREDIT'}]


def test_transport_diagnostics_do_not_mask_original_refusal(tmp_path, monkeypatch):
    transport,readback,sha,physical,obligations=request_fixture(tmp_path,monkeypatch)
    original=ot.write_json_atomic
    def write(path,value):
        if path.name=='OPERATOR_TRANSPORT_STATUS.json':raise OSError('diagnostic unavailable')
        return original(path,value)
    monkeypatch.setattr(ot,'write_json_atomic',write)
    monkeypatch.setattr(ot.time,'sleep',lambda seconds:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):transport._request(tmp_path,'CAPTURE_SAVE',obligations,physical)
    assert len(list(transport.queue.glob('*.request.json')))==1
