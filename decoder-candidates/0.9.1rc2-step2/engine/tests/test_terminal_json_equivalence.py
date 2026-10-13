"""Focused registered publication regression; synthetic fixture grants no authority."""
import copy
import json
from types import SimpleNamespace
import pytest
from infinity_grid import preservation as pr, submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.canon import canonical_bytes, write_json_atomic


def fixture(tmp_path,monkeypatch,mutate=None,completed=False):
    done={'request_id':'UNIT_JSON','source_sha256':'1'*64,'completion_sha256':'2'*64,
          'status':'COMPLETED','publication_protocol':pr.COMPLETION_PROTOCOL,
          'result':{'subjects':[(7,8,9,(2,0,1)),()], 'flags':{'ok':True,'count':1,'rate':1.0}},
          'evidence':[{'path':'answer.json','sha256':sub._sha(b'{}'),'size_bytes':2}]}
    saved=json.loads(canonical_bytes(done))
    if mutate:mutate(saved)
    name='runtime/intake/'+('completed' if completed else 'prepared_completions')+'/UNIT_JSON.json'
    files={name:canonical_bytes(saved),'runtime/runs/UNIT_JSON/answer.json':b'{}'}
    row={'schema_id':pr.SCHEMA,'terminal':True,'capture_id':'UNIT_CAPTURE','job_id':'UNIT.JOB',
         'source_sha256':done['source_sha256'],'terminal_completions':{'UNIT_JSON':done['completion_sha256']},
         'state':{'sha256':'3'*64}}
    digest=sub._sha(sub._json_bytes(row));write_json_atomic(tmp_path/'durability/outbox/commits'/f'{digest}.json',row)
    monkeypatch.setattr(sub,'capture_record',lambda root:{'capture_id':'UNIT_CAPTURE','job':{'job_id':'UNIT.JOB'}})
    monkeypatch.setattr(pr,'make_checkpoint',lambda *a,**kw:{'latest_checkpoint':digest,'status':'UNIT_FIXTURE'})
    monkeypatch.setattr(pr,'_bytes',lambda *a:sub._archive(files))
    return done,files


@pytest.mark.parametrize('completed',[False,True])
def test_nested_tuple_publication_and_no_rewrite(tmp_path,monkeypatch,completed):
    done,files=fixture(tmp_path,monkeypatch,completed=completed);before=copy.deepcopy(done);raw=dict(files)
    saved=json.loads(next(v for k,v in files.items() if 'intake' in k))
    assert saved!=done and canonical_bytes(saved)==canonical_bytes(done)
    assert pr.ensure_terminal_completion(tmp_path,done)['status']=='UNIT_FIXTURE'
    assert pr.terminal_completion_proof(tmp_path,done)
    assert done==before and files==raw


@pytest.mark.parametrize('field,value',[('ok',1),('count',True),('count',1.0),('rate',1),('count',2)])
def test_scalar_differences_refuse_before_publication(tmp_path,monkeypatch,field,value):
    done,files=fixture(tmp_path,monkeypatch,lambda d:d['result']['flags'].__setitem__(field,value))
    with pytest.raises(sub.SubmissionError,match='TERMINAL_COMPLETION_CHECKPOINT_MISMATCH'):
        pr.ensure_terminal_completion(tmp_path,done)
    assert not (tmp_path/'runtime/terminal_checkpoints/UNIT_JSON.json').exists()


def test_list_order_change_refuses(tmp_path,monkeypatch):
    done,files=fixture(tmp_path,monkeypatch,lambda d:d['result']['subjects'][0][3].reverse())
    with pytest.raises(sub.SubmissionError,match='TERMINAL_COMPLETION_CHECKPOINT_MISMATCH'):pr.ensure_terminal_completion(tmp_path,done)


def test_evidence_still_checked_after_equivalence(tmp_path,monkeypatch):
    done,files=fixture(tmp_path,monkeypatch);files['runtime/runs/UNIT_JSON/answer.json']=b'changed'
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):pr.ensure_terminal_completion(tmp_path,done)
    assert not (tmp_path/'runtime/terminal_checkpoints/UNIT_JSON.json').exists()


def test_real_publication_path_preserves_canonical_completion(tmp_path,monkeypatch):
    done,files=fixture(tmp_path,monkeypatch);calls=[]
    claim=SimpleNamespace(complete=lambda value:calls.append(canonical_bytes(value)))
    loop._publish_checkpointed_completion({'workspace':tmp_path},done,claim)
    assert calls==[canonical_bytes(done)]
    assert canonical_bytes(json.loads((tmp_path/'runtime/intake/completed/UNIT_JSON.json').read_text()))==canonical_bytes(done)
    monkeypatch.setattr(pr,'status',lambda root:{'status':'UNIT_REUSED'})
    monkeypatch.setattr(pr,'make_checkpoint',lambda *a,**kw:pytest.fail('checkpoint repeated'))
    assert pr.ensure_terminal_completion(tmp_path,done)['status']=='UNIT_REUSED'


def test_existing_publication_accepts_tuple_json_equivalence(tmp_path,monkeypatch):
    done,files=fixture(tmp_path,monkeypatch,completed=True)
    write_json_atomic(tmp_path/'runtime/intake/completed/UNIT_JSON.json',done)
    original=(tmp_path/'runtime/intake/completed/UNIT_JSON.json').read_bytes()
    calls=[]
    loop._publish_checkpointed_completion({'workspace':tmp_path},done,SimpleNamespace(complete=lambda d:calls.append(d)))
    assert len(calls)==1 and (tmp_path/'runtime/intake/completed/UNIT_JSON.json').read_bytes()==original


def test_existing_publication_changed_scalar_refuses(tmp_path,monkeypatch):
    done,files=fixture(tmp_path,monkeypatch)
    changed=json.loads(canonical_bytes(done));changed['result']['flags']['ok']=1
    write_json_atomic(tmp_path/'runtime/intake/completed/UNIT_JSON.json',changed)
    with pytest.raises(loop.ControllerLoopError,match='COMPLETION_PUBLICATION_COLLISION'):
        loop._publish_checkpointed_completion({'workspace':tmp_path},done,SimpleNamespace(complete=lambda d:pytest.fail('published')))
