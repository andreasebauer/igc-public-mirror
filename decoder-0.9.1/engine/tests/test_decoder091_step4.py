"""Operator/release admission checks; unit cases do not claim remote save authority."""
from pathlib import Path
import fcntl,json,threading,time
import pytest
from infinity_grid import operator_transport as ot,rolling_operator as rolling
from infinity_grid import bulk_submission as bulk
from infinity_grid.canon import write_json_atomic

@pytest.mark.parametrize('reply_kind',['wrong_scope','missing','duplicate','bad_bytes','extra_field'])
def test_operator_queue_refuses_bad_reply_before_confirm(tmp_path,monkeypatch,reply_kind):
    monkeypatch.setattr(ot.sub,'capture_record',lambda p:{'capture_id':'c'*64})
    confirms=[];monkeypatch.setattr(ot.sub,'confirm_save',lambda *a,**k:confirms.append(a))
    queue=tmp_path/'queue';transport=ot.OperatorQueueTransport(queue);raw=tmp_path/'readback';raw.write_bytes(b'actual')
    digest=ot.sub._sha(b'actual');errors=[]
    def run():
        try:transport._request(tmp_path,'CAPTURE_SAVE',[{'sha256':digest,'role':'engine_source','logical_name':'engine'}],
              [{'sha256':digest,'size_bytes':6,'path':str(raw)}])
        except BaseException as exc:errors.append(exc)
    worker=threading.Thread(target=run);worker.start()
    deadline=time.monotonic()+5
    while not list(queue.glob('*.request.json')):
        if time.monotonic()>deadline:raise AssertionError('request not published')
        time.sleep(.01)
    req=json.loads(next(queue.glob('*.request.json')).read_bytes())
    row={'sha256':digest,'readback_path':str(raw),'drive_file_id':'NOT_A_PROVIDER_ACK_UNIT_FIXTURE'}
    objects=[row]
    if reply_kind=='missing':objects=[]
    if reply_kind=='duplicate':objects=[row,row]
    if reply_kind=='bad_bytes':raw.write_bytes(b'wrong')
    if reply_kind=='extra_field':row['unsaved_override']=True
    reply={'schema_id':'IG_OPERATOR_SAVE_RESPONSE_V1','request_id':req['request_id'],'objects':objects}
    if reply_kind=='wrong_scope':reply['request_id']='wrong'
    write_json_atomic(queue/(req['request_id']+'.response.json'),reply);worker.join(timeout=5)
    assert not worker.is_alive() and errors and confirms==[]
    assert not list(queue.glob('*.confirmed.json'))


def test_competing_rolling_writer_refuses_before_capture(tmp_path):
    writer=(tmp_path/'WRITER.lock').open('a+b');fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError,match='COMPETING_WINDOW_WRITER'):
            bulk._coordinate_registered_windows(tmp_path,{'windows':[],'plan_sha256':'engineering'},'s','p',None,
                lambda *a:pytest.fail('capture called'),lambda *a:None,lambda *a:None,lambda *a:None)
    finally:writer.close()


def test_engineering_route_cannot_accept_data_owner_adoption(tmp_path):
    with pytest.raises(ValueError,match='ENGINEERING_REGISTRATION_REQUIRED'):
        rolling.run_engineering(tmp_path,tmp_path,tmp_path,{},
          {'schema_id':'IG_DATA_OWNER_DECODER_ADOPTION_V1','status':'ADOPTED'},None)


def test_namespace_handler_evaluator_reuse_ignores_ambient_project_roots(tmp_path,monkeypatch):
    import sys
    from infinity_grid.project_stage import bind_callables,resolve_callable
    from infinity_grid.v05_controller_event_loop import _source_ids
    source=tmp_path/'captured';ambient=tmp_path/'ambient'
    (source/'project').mkdir(parents=True);(ambient/'project').mkdir(parents=True)
    (source/'infinity_grid').mkdir();(source/'infinity_grid/_version.py').write_text('__version__="0.9.1"\n')
    (source/'project/module.py').write_text('def handler():return "PINNED"\ndef evaluate():return "PINNED_EVALUATOR"\n')
    (ambient/'project/module.py').write_text('def handler():return "UNPINNED"\ndef evaluate():return "UNPINNED_EVALUATOR"\n')
    for name in list(sys.modules):
        if name=='project' or name.startswith('project.'):monkeypatch.delitem(sys.modules,name)
    monkeypatch.syspath_prepend(str(ambient))
    bindings=bind_callables(source,['project.module:handler','project.module:evaluate'],source_sha256=_source_ids(source)[0])
    assert resolve_callable('project.module:handler',bindings['project.module:handler'])()=='PINNED'
    assert resolve_callable('project.module:evaluate',bindings['project.module:evaluate'])()=='PINNED_EVALUATOR'
    assert list(sys.modules['project'].__path__)==[str(source/'project')]


@pytest.mark.parametrize('variant',['identical','different_source','unbound_sibling'])
def test_sequential_capture_reload_requires_entire_known_identical_closure(tmp_path,monkeypatch,variant):
    import sys,shutil,types
    from infinity_grid.project_stage import bind_callables,resolve_callable,ProjectStageError
    from infinity_grid.v05_controller_event_loop import _source_ids
    first=tmp_path/'first';second=tmp_path/'second'
    (first/'project').mkdir(parents=True);(first/'infinity_grid').mkdir()
    (first/'infinity_grid/_version.py').write_text('__version__="0.9.1"\n')
    (first/'project/helper.py').write_text('VALUE="PINNED"\n')
    (first/'project/module.py').write_text('from .helper import VALUE\ndef handler():return VALUE\ndef evaluate():return VALUE\n')
    shutil.copytree(first,second)
    for name in list(sys.modules):
        if name=='project' or name.startswith('project.'):monkeypatch.delitem(sys.modules,name)
    refs=['project.module:handler','project.module:evaluate']
    one=bind_callables(first,refs,source_sha256=_source_ids(first)[0])
    original=resolve_callable(refs[0],one[refs[0]]);assert original()=='PINNED'
    if variant=='different_source':(second/'project/helper.py').write_text('VALUE="CHANGED"\n')
    if variant=='unbound_sibling':monkeypatch.setitem(sys.modules,'project.unknown',types.ModuleType('project.unknown'))
    two=bind_callables(second,refs,source_sha256=_source_ids(second)[0])
    if variant!='identical':
        with pytest.raises(ProjectStageError,match='CACHED_MODULE_UNBOUND'):resolve_callable(refs[0],two[refs[0]])
        assert sys.modules['project.module'].handler is original
    else:
        current=resolve_callable(refs[0],two[refs[0]])
        assert current()=='PINNED' and current is not original
        assert current.__code__.co_filename==str(second/'project/module.py')
        assert Path(sys.modules['project.helper'].__file__)==second/'project/helper.py'
        assert resolve_callable(refs[1],two[refs[1]])()=='PINNED'
        assert list(sys.modules['project'].__path__)==[str(second/'project')]
