"""Data-only policy/publication fault checks; no synthetic save or science claim."""
from types import SimpleNamespace
import pytest
from infinity_grid import result_contracts as rc, project_checkpoint as pc
from infinity_grid import preservation as pr, submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.checkpoint_stream import bounded_tree


def test_new_capture_removes_project_cap_and_legacy_remains_frozen(tmp_path):
    new=rc.policy({'preservation':{'max_project_raw_bytes':None,'completion_project_closure':1,'project_evidence_externalization':1}})
    assert new['max_project_raw_bytes'] is None
    assert rc.policy({})['max_project_raw_bytes']==33554432
    assert rc.policy({})['completion_project_closure']==0
    (tmp_path/'large').write_bytes(b'x'*(33*1024**2))
    assert bounded_tree(tmp_path,None)==33*1024**2
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_TREE_RAW_LIMIT'):
        bounded_tree(tmp_path,33554432)
    pc.require_metadata_bound(33*1024**2,None)
    with pytest.raises(sub.SubmissionError,match='PROJECT_METADATA_RAW_LIMIT'):
        pc.require_metadata_bound(33*1024**2,33554432)


@pytest.mark.parametrize('bad',[True,-1,0,'none'])
def test_project_bound_is_positive_integer_or_explicit_none(bad):
    with pytest.raises(sub.SubmissionError):rc.policy({'preservation':{'max_project_raw_bytes':bad}})


def test_publication_closes_after_claim_with_precheckpoint_still_first(tmp_path,monkeypatch):
    events=[]
    monkeypatch.setattr(pr,'ensure_terminal_completion',lambda *a:events.append('pre') or {'status':'PRE'})
    def close(root,reason,**kw):
        assert reason=='COMPLETION_PROJECT_CLOSURE' and kw=={'terminal':True}
        events.append('post');return {'status':'POST'}
    monkeypatch.setattr(pr,'make_checkpoint',close)
    done={'request_id':'UNIT','publication_protocol':pr.COMPLETION_PROTOCOL,'status':'COMPLETED'}
    claim=SimpleNamespace(complete=lambda done:events.append('claim'))
    admission={'workspace':tmp_path,'result_contract':{'preservation':{'completion_project_closure':1}}}
    assert loop._publish_checkpointed_completion(admission,done,claim)=={'status':'POST'}
    assert events==['pre','claim','post']


def test_postpublication_checkpoint_failure_is_not_reported_as_success(tmp_path,monkeypatch):
    monkeypatch.setattr(pr,'ensure_terminal_completion',lambda *a:{'status':'PRE'})
    monkeypatch.setattr(pr,'make_checkpoint',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('POST_CLOSURE_FAILURE')))
    done={'request_id':'UNIT','publication_protocol':pr.COMPLETION_PROTOCOL,'status':'COMPLETED'}
    with pytest.raises(RuntimeError,match='POST_CLOSURE_FAILURE'):
        loop._publish_checkpointed_completion({'workspace':tmp_path,'result_contract':{'preservation':{'completion_project_closure':1}}},done,SimpleNamespace(complete=lambda d:None))


def test_none_metadata_cap_roundtrip_preserves_unreferenced_objects(tmp_path):
    from infinity_grid import portable_registry as reg
    from infinity_grid.canon import write_json_atomic
    root=tmp_path/'project';write_json_atomic(root/'PROJECT.json',{'schema_id':'IG_PORTABLE_PROJECT_V1','project_id':'UNIT','name':'UNIT'})
    baseline=reg.snapshot(root);reg.put(root,b'x'*(33*1024**2))
    limits=rc.policy({'preservation':{'project_evidence_externalization':1,'max_project_raw_bytes':None}})
    prepared=pc.prepare(root,baseline,limits,tmp_path)
    assert prepared['metadata_raw_bytes']>33554432
    projected=pc.metadata_baseline(baseline,prepared['evidence_path'].read_bytes(),limits)
    assert pr._restore_project(projected,pr._project_delta(projected,prepared['metadata']),prepared['metadata_sha256'])==prepared['metadata']
    assert pc.hydrate(prepared['metadata'],prepared['evidence_path'].read_bytes(),dict(prepared,schema_id=pc.SCHEMA),limits)==reg.snapshot(root)
