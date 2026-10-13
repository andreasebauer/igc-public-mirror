"""Local coordination checks; run only through a captured VALIDATION job."""
import copy
import pytest
from infinity_grid.shared_authority import Authority, CoordinationRefused, work_identity

A='a'*64
B='b'*64
C='c'*64
D='d'*64

def spec():
    return dict(engine=A, implementation=B, inputs={'data':C}, parameters={'seed':7},
                contract=D, environment={'python':'3.12','platform':'linux-x86_64'},
                resources={'workers':1}, repeat=None)

def clients(tmp_path):
    a=Authority(tmp_path/'authority.sqlite');a.initialize(A)
    return a,Authority(tmp_path/'authority.sqlite')

def test_labels_discover_and_completed_work_reuses(tmp_path):
    a,b=clients(tmp_path);claim=a.claim(spec(),'chat-a','first')
    other=b.claim(spec(),'chat-b','totally different label')
    assert other['status']=='RUNNING' and 'token' not in other
    a.finish(claim['work_id'],'chat-a',claim['token'],{'completion':C,'snapshot':D})
    assert b.claim(spec(),'chat-b','third')['status']=='REUSE'

def test_repeat_and_unrelated_work_can_coexist(tmp_path):
    a,b=clients(tmp_path);x=a.claim(spec(),'a','x')
    repeat=spec();repeat['repeat']='intentional-repeat-1'
    y=b.claim(repeat,'b','x');other=spec();other['parameters']['seed']=8
    z=b.claim(other,'b','z')
    assert len({x['work_id'],y['work_id'],z['work_id']})==3
    assert all(r['status']=='CLAIMED' for r in (x,y,z))

def test_stale_activation_and_aba_are_refused(tmp_path):
    a,b=clients(tmp_path);parent=a.current();a.activate(parent,B,C,D)
    with pytest.raises(CoordinationRefused,match='STALE_PARENT'): b.activate(parent,C,C,D)
    a.activate(a.current(),A,C,D)
    with pytest.raises(CoordinationRefused,match='STALE_PARENT'): b.activate(parent,B,C,D)

def test_running_work_keeps_original_engine_after_activation(tmp_path):
    a,b=clients(tmp_path);x=a.claim(spec(),'a','x');b.activate(b.current(),B,C,D)
    a.finish(x['work_id'],'a',x['token'],{'completion':C,'snapshot':D})
    assert a.inspect_work(x['work_id'])['state']=='COMPLETED'
    assert b.claim(spec(),'b','old exact result')['status']=='REUSE'
    changed=spec();changed['parameters']['seed']=8
    with pytest.raises(CoordinationRefused,match='STALE_ENGINE'): b.claim(changed,'b','old new work')

def test_provider_replacement_detects_conflict(tmp_path):
    a,b=clients(tmp_path);checks={'correctness':C,'performance':D}
    a.replace_provider('decode',None,A,checks)
    with pytest.raises(CoordinationRefused,match='PROVIDER_CONFLICT'): b.replace_provider('decode',None,B,checks)
    b.replace_provider('decode',A,B,checks)
    with pytest.raises(CoordinationRefused,match='PROVIDER_CONFLICT'): a.replace_provider('decode',A,C,checks)

def test_identity_binds_every_semantic_field():
    original=spec();wid=work_identity(original)
    for key in original:
        changed=copy.deepcopy(original)
        if key in ('engine','implementation','contract'): changed[key]='e'*64
        elif key=='repeat': changed[key]='repeat'
        elif key=='inputs': changed[key]['data']='e'*64
        else: changed[key]['extra']='different'
        assert work_identity(changed)!=wid,key
    changed=spec();changed['label']='ignored?'
    with pytest.raises(CoordinationRefused): work_identity(changed)

def test_wrong_owner_cannot_complete_and_completion_is_immutable(tmp_path):
    a,b=clients(tmp_path);x=a.claim(spec(),'a','x');ev={'completion':C,'snapshot':D}
    with pytest.raises(CoordinationRefused): b.finish(x['work_id'],'b',x['token'],ev)
    with pytest.raises(CoordinationRefused): b.finish(x['work_id'],'a','bad',ev)
    a.finish(x['work_id'],'a',x['token'],ev)
    assert a.finish(x['work_id'],'a',x['token'],ev)['status']=='COMPLETED'
    with pytest.raises(CoordinationRefused): a.finish(x['work_id'],'a',x['token'],{'completion':B,'snapshot':D})

def test_reopen_retains_claim_release_and_provider(tmp_path):
    a,b=clients(tmp_path);x=a.claim(spec(),'a','x');a.activate(a.current(),B,C,D)
    del a,b
    c=Authority(tmp_path/'authority.sqlite')
    assert c.current()=={'generation':1,'engine':B}
    assert c.inspect_work(x['work_id'])['state']=='RUNNING'
    c.finish(x['work_id'],'a',x['token'],{'completion':C,'snapshot':D})
