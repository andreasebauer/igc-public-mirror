"""Registered checks of completed worker output and its parent acknowledgement."""
import hashlib
import os
import sys
import time
from pathlib import Path
import pytest
from infinity_grid import v05_validation_runtime as runtime


@pytest.mark.parametrize('raw',[b'',b'progress\nsummary\n',b'line\r\n\xff\x00'])
def test_closed_log_publication_preserves_exact_bytes(tmp_path,raw):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(raw)
    record=runtime._publish_worker_log(active,final)
    assert not active.exists() and final.read_bytes()==raw
    assert record=={'log_sha256':hashlib.sha256(raw).hexdigest(),'log_size_bytes':len(raw)}
    runtime._verify_worker_log(final,record)


def test_existing_completed_log_is_not_overwritten(tmp_path):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(b'new');final.write_bytes(b'prior')
    with pytest.raises(FileExistsError):runtime._publish_worker_log(active,final)
    assert active.read_bytes()==b'new' and final.read_bytes()==b'prior'


@pytest.mark.parametrize('mutation',['shortened','same_size','missing'])
def test_parent_rejects_changed_or_missing_published_log(tmp_path,mutation):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(b'original');record=runtime._publish_worker_log(active,final)
    if mutation=='missing':final.unlink()
    else:final.write_bytes(b'x' if mutation=='shortened' else b'changed!')
    with pytest.raises((runtime.ValidationRuntimeError,FileNotFoundError)):
        runtime._verify_worker_log(final,record)


@pytest.mark.parametrize('field',['log_sha256','log_size_bytes'])
def test_parent_requires_both_exact_child_bindings(tmp_path,field):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(b'original');record=runtime._publish_worker_log(active,final)
    del record[field]
    with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_WORKER_LOG_MISMATCH'):
        runtime._verify_worker_log(final,record)


def test_publication_readback_failure_retains_active_evidence(tmp_path,monkeypatch):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(b'complete')
    original=Path.read_bytes
    def stale(path):return b'stale' if path==final else original(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'read_bytes',stale)
        with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_LOG_PUBLICATION_READBACK'):
            runtime._publish_worker_log(active,final)
    assert active.read_bytes()==final.read_bytes()==b'complete'


def test_worker_waits_for_forked_descendant_before_publication(tmp_path):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(b'initial\n')
    pid=os.fork()
    if pid==0:
        time.sleep(0.05)
        with active.open('ab') as stream:stream.write(b'late\n')
        os._exit(0)
    runtime._wait_for_validation_descendants(timeout_seconds=1.0)
    record=runtime._publish_worker_log(active,final)
    assert final.read_bytes()==b'initial\nlate\n'
    runtime._verify_worker_log(final,record)
    with pytest.raises(ChildProcessError):os.waitpid(pid,os.WNOHANG)


def test_worker_reaps_reparented_double_fork_before_publication(tmp_path):
    active=tmp_path/'worker.active';final=tmp_path/'worker.log'
    active.write_bytes(b'initial\n')
    runtime._enable_validation_subreaper()
    pid=os.fork()
    if pid==0:
        grandchild=os.fork()
        if grandchild==0:
            time.sleep(0.10)
            with active.open('ab') as stream:stream.write(b'reparented-late\n')
            os._exit(0)
        os._exit(0)
    runtime._wait_for_validation_descendants(timeout_seconds=2.0)
    record=runtime._publish_worker_log(active,final)
    assert final.read_bytes()==b'initial\nreparented-late\n'
    runtime._verify_worker_log(final,record)
    with pytest.raises(ChildProcessError):os.waitpid(-1,os.WNOHANG)


def test_controller_boundary_rejects_any_active_worker_evidence(tmp_path):
    final=tmp_path/'worker-0-1.log';final.write_bytes(b'complete')
    result={'log_path':final.name}
    runtime._verify_worker_evidence_set(tmp_path,[result])
    (tmp_path/'worker-0-1.active').write_bytes(b'late')
    with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_WORKER_EVIDENCE_SET_MISMATCH'):
        runtime._verify_worker_evidence_set(tmp_path,[result])


def test_controller_boundary_rejects_unbound_published_log(tmp_path):
    (tmp_path/'worker-0-1.log').write_bytes(b'complete')
    (tmp_path/'worker-1-2.log').write_bytes(b'unbound')
    with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_WORKER_EVIDENCE_SET_MISMATCH'):
        runtime._verify_worker_evidence_set(tmp_path,[{'log_path':'worker-0-1.log'}])


def test_resumed_worker_evidence_retains_exact_prior_logs(tmp_path):
    prior=tmp_path/'worker-1-prior.log';prior.write_bytes(b'prior')
    baseline=runtime._evidence_baseline(tmp_path,'worker-*.log')
    current=tmp_path/'worker-0-current.log';current.write_bytes(b'current')
    runtime._verify_worker_evidence_set(
        tmp_path,[{'log_path':current.name}],prior=baseline)
    prior.write_bytes(b'alter')
    with pytest.raises(runtime.ValidationRuntimeError,
                       match='VALIDATION_PRIOR_EVIDENCE_CHANGED'):
        runtime._verify_worker_evidence_set(
            tmp_path,[{'log_path':current.name}],prior=baseline)


def test_controller_boundary_binds_complete_node_evidence_set(tmp_path):
    node='tests/test_example.py::test_one'
    root=tmp_path/'node_process_logs';root.mkdir()
    expected=f'000000-{runtime.canonical_sha256(node)}.log'
    (root/expected).write_bytes(b'complete')
    results=[{'nodes':[node]}]
    runtime._verify_node_evidence_set(tmp_path,results)
    (root/(expected.removesuffix('.log')+'.active')).write_bytes(b'.')
    with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_NODE_EVIDENCE_SET_MISMATCH'):
        runtime._verify_node_evidence_set(tmp_path,results)


def test_controller_boundary_rejects_missing_or_unbound_node_log(tmp_path):
    node='tests/test_example.py::test_one'
    root=tmp_path/'node_process_logs';root.mkdir()
    results=[{'nodes':[node]}]
    with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_NODE_EVIDENCE_SET_MISMATCH'):
        runtime._verify_node_evidence_set(tmp_path,results)
    expected=f'000000-{runtime.canonical_sha256(node)}.log'
    (root/expected).write_bytes(b'complete')
    (root/'999999-unbound.log').write_bytes(b'unbound')
    with pytest.raises(runtime.ValidationRuntimeError,match='VALIDATION_NODE_EVIDENCE_SET_MISMATCH'):
        runtime._verify_node_evidence_set(tmp_path,results)


def test_resumed_node_evidence_retains_exact_prior_logs(tmp_path):
    prior_node='tests/test_prior.py::test_prior'
    current_node='tests/test_current.py::test_current'
    root=tmp_path/'node_process_logs';root.mkdir()
    prior=root/f'000000-{runtime.canonical_sha256(prior_node)}.log'
    prior.write_bytes(b'prior')
    baseline=runtime._evidence_baseline(root,'*.log')
    current=root/f'000000-{runtime.canonical_sha256(current_node)}.log'
    current.write_bytes(b'current')
    runtime._verify_node_evidence_set(
        tmp_path,[{'nodes':[current_node]}],prior=baseline)
    prior.write_bytes(b'alter')
    with pytest.raises(runtime.ValidationRuntimeError,
                       match='VALIDATION_PRIOR_EVIDENCE_CHANGED'):
        runtime._verify_node_evidence_set(
            tmp_path,[{'nodes':[current_node]}],prior=baseline)


def test_node_pipe_waits_for_descendant_eof_and_has_no_active_path(tmp_path):
    final=tmp_path/'selector.log'
    script=("import os,time; os.write(1,b'parent\\n'); pid=os.fork(); "
            "(time.sleep(.08),os.write(1,b'late\\n'),os._exit(0)) if pid==0 else None")
    proc,raw=runtime._run_node_process(
        [sys.executable,'-c',script],cwd=tmp_path,env=dict(os.environ),final_path=final)
    assert proc.returncode==0
    assert raw==b'parent\nlate\n' and final.read_bytes()==raw
    assert not list(tmp_path.glob('*.active'))


def test_node_publication_refuses_existing_final(tmp_path):
    final=tmp_path/'selector.log';final.write_bytes(b'prior')
    with pytest.raises(FileExistsError):runtime._publish_node_output(final,b'new')
    assert final.read_bytes()==b'prior'
