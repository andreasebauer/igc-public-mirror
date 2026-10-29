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

@pytest.mark.parametrize('failure',['yield','mismatch','false_save'])
def test_existing_rolling_loop_validation_pause(tmp_path,monkeypatch,failure):
    from infinity_grid import submission as sub,preservation as pr,v05_controller_event_loop as loop
    from infinity_grid.validation_frontier import ValidationYield
    events=[];calls=[];workspace=tmp_path/'capture'
    monkeypatch.setattr(sub,'capture_record',lambda p:{'workspace':{'source_sha256':'s','package_sha256':'p'},'job':{'job_id':'j'}})
    monkeypatch.setattr(sub,'save_status',lambda p:{'status':'SAVED'})
    monkeypatch.setattr(pr,'status',lambda p:{'status':'PENDING' if failure=='false_save' else 'PRESERVED'})
    def run(*args):
        calls.append(1);events.append('run')
        if len(calls)==1:
            if failure=='mismatch':raise ValueError('scientific mismatch')
            raise ValidationYield('VALIDATION_UNIT_BUDGET_REACHED')
    monkeypatch.setattr(loop,'run_workspace_job',run)
    class Transport:
        def save_capture(self,p):events.append('save_capture')
        def preserve_workspace(self,p):events.append('preserve')
    invoke=lambda:bulk._coordinate_registered_windows(tmp_path,{'windows':[{}],'plan_sha256':'plan'},'s','p',Transport(),
        lambda *a:{'capture':{'workspace':str(workspace)}},lambda *a:{'index':0},lambda *a:None,
        lambda p,r:{'completed_windows':[r]})
    if failure=='yield':
        assert invoke()['completed_windows']==[{'index':0}]
        assert events==['save_capture','run','preserve','run','preserve']
    else:
        with pytest.raises(ValueError,match='scientific mismatch' if failure=='mismatch' else 'BACKLOG_TRANSPORT_NOT_ACKNOWLEDGED'):invoke()
        assert len(calls)==(0 if failure=='false_save' else 1)


@pytest.mark.parametrize('kind', ['backlog', 'acceptance', 'native_source', 'mismatch', 'fake_boundary'])
def test_shared_capture_reducer_save_boundaries(tmp_path, monkeypatch, kind):
    from contextlib import nullcontext
    from infinity_grid import submission as sub, preservation as pr, v05_controller_event_loop as loop
    from infinity_grid.accepted_result_ledger import AcceptanceStopped
    events=[]
    monkeypatch.setattr(sub, 'capture_record', lambda p: {'job':{'job_id':'j'}})
    monkeypatch.setattr(sub, 'save_status', lambda p: {'status':'SAVED'})
    monkeypatch.setattr(pr, 'status', lambda p: {'status':'PRESERVED'})
    failures={'backlog':sub.SubmissionError('SAVE_BACKLOG_PAUSE','detail'),
              'acceptance':AcceptanceStopped('ACCEPTANCE_SAVE_DRAIN_REQUIRED:sha'),
              'native_source':AcceptanceStopped('NATIVE_SOURCE_SAVE_DRAIN_REQUIRED:sha'),
              'mismatch':AcceptanceStopped('HISTORICAL_MISMATCH'),
              'fake_boundary':sub.SubmissionError('OTHER','SAVE_BACKLOG_PAUSE')}
    def run(*a):
        events.append('run')
        if events.count('run')==1: raise failures[kind]
        return {'status':'COMPLETED'}
    monkeypatch.setattr(loop, 'run_workspace_job', run)
    class Transport:
        def save_capture(self,p):events.append('capture')
        def preserve_workspace(self,p):events.append('preserve')
    call=lambda:bulk._run_saved_capture(tmp_path, Transport(), lambda name:nullcontext())
    if kind in ('backlog','acceptance','native_source'):
        assert call()['status']=='COMPLETED'
        assert events==['capture','run','preserve','run','preserve']
    else:
        with pytest.raises(type(failures[kind])):call()
        assert events==['capture','run']


def test_resume_capture_refuses_changed_source_before_transport(tmp_path, monkeypatch):
    from infinity_grid import rolling_operator as op, v05_controller_event_loop as loop
    w=tmp_path/'capture';w.mkdir()
    monkeypatch.setattr(op.sub,'capture_record',lambda p:{'job':{'job_id':'j'}})
    def refuse(*a,**kw):
        assert kw['check_loaded'] is True
        raise ValueError('WORKSPACE_LOADED_CODE_MISMATCH')
    monkeypatch.setattr(loop,'validate_workspace_job',refuse)
    with pytest.raises(ValueError,match='WORKSPACE_LOADED_CODE_MISMATCH'):
        op.resume_registered_capture(w,tmp_path/'queue')
    assert not (tmp_path/'queue').exists()


def test_fresh_capture_needs_no_pre_execution_checkpoint(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from infinity_grid import submission as sub, preservation as pr, v05_controller_event_loop as loop
    events=[]
    monkeypatch.setattr(sub,'capture_record',lambda p:{'job':{'job_id':'j'}})
    monkeypatch.setattr(sub,'save_status',lambda p:{'status':'SAVED'})
    monkeypatch.setattr(pr,'status',lambda p:{'status':'PRESERVED' if 'preserve' in events else 'NO_CHECKPOINT'})
    monkeypatch.setattr(loop,'run_workspace_job',lambda *a:events.append('run'))
    class Transport:
        def save_capture(self,p):events.append('capture')
        def preserve_workspace(self,p):
            assert 'run' in events
            events.append('preserve')
    bulk._run_saved_capture(tmp_path,Transport(),lambda name:nullcontext())
    assert events==['capture','run','preserve']
