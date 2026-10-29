"""Operational freshness does not invent work or cap native pool history."""
import json
import pytest
from infinity_grid import acceptance_observer as observer
from infinity_grid.full_block_runner import recover_workspace,workspace_bytes


def test_heartbeats_context_and_unchanged_operations_do_not_refresh_progress(tmp_path,monkeypatch):
    clock=[100.0]
    monkeypatch.setattr(observer.time,'time',lambda:clock[0])
    monkeypatch.setattr(observer.time,'monotonic',lambda:clock[0])
    with observer.session(tmp_path):
        active=observer._ACTIVE.get()
        assert json.loads(active.path.read_text())['last_progress_unix'] is None
        observer.operation('bytes',0,10,bytes_done=0)
        clock[0]=110.0
        observer.operation('bytes',1,10,bytes_done=100)
        active.write(True)
        assert json.loads(active.path.read_text())['last_progress_unix']==110.0
        clock[0]=130.0
        with observer.phase('waiting'):
            observer.detail(wait_reason='native verifier')
        observer.operation('bytes',1,10,bytes_done=100)
        active.write(True)
        data=json.loads(active.path.read_text())
        assert data['heartbeat_unix']==130.0
        assert data['last_progress_unix']==110.0
        clock[0]=140.0
        observer.operation('bytes',1,10,bytes_done=101)
        active.write(True)
        assert json.loads(active.path.read_text())['last_progress_unix']==140.0
        clock[0]=150.0
        observer.tick('verified',1)
        active.write(True)
        assert json.loads(active.path.read_text())['last_progress_unix']==150.0


def test_registered_progress_refreshes_only_when_done_advances(tmp_path,monkeypatch):
    clock=[100.0]
    monkeypatch.setattr(observer.time,'time',lambda:clock[0])
    monkeypatch.setattr(observer.time,'monotonic',lambda:clock[0])
    with observer.session(tmp_path):
        active=observer._ACTIVE.get()
        observer.register_plan({'plan_sha256':'p','manifest_sha256':'m',
            'stages':[{'stage_id':'comparison'}],'whole_run_eta_seconds':None,'eta_reason':'fixture'})
        observer.progress('comparison',1,10,5.0)
        assert active.last_progress==100.0
        clock[0]=125.0
        observer.progress('comparison',1,10,30.0)
        active.write(True)
        assert json.loads(active.path.read_text())['last_progress_unix']==100.0
        observer.progress('comparison',2,10,30.0)
        assert active.last_progress==125.0


def test_worker_history_above_old_cap_is_accounted_and_validated(tmp_path):
    root=tmp_path/'full_generation';workers=root/'telemetry/workers';workers.mkdir(parents=True)
    raw=b'{"status":"COMPLETE"}\n'
    for pid in range(300):
        worker=workers/str(pid);worker.mkdir();(worker/'ACCEPTANCE_STATUS.json').write_bytes(raw)
    assert workspace_bytes(root)==len(raw)*300
    recover_workspace(root)
    (workers/'299/unknown.json').write_text('{}')
    with pytest.raises(ValueError,match='ACCEPTANCE_TELEMETRY_FILE'):
        recover_workspace(root)


@pytest.mark.parametrize('scope',['controller','worker'])
def test_interrupted_atomic_status_history_is_preserved_and_accounted(tmp_path,scope):
    root=tmp_path/'full_generation'
    status=root/'telemetry'
    if scope=='worker':status=status/'workers/123'
    status.mkdir(parents=True)
    raw=b'{"status":"RUNNING"}\n'
    names=['ACCEPTANCE_STATUS.json']+[
        '.ACCEPTANCE_STATUS.json.tmp-123-'+format(index,'032x') for index in range(9)]
    for name in names:(status/name).write_bytes(raw)
    assert workspace_bytes(root)==len(raw)*len(names)
    recover_workspace(root)
    assert {path.name for path in status.iterdir()}==set(names)
    assert all((status/name).read_bytes()==raw for name in names)
