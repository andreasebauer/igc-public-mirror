import pickle
import pytest
from infinity_grid import execution as ex
from infinity_grid.v05_stage_runtime import StageScienceRuntime, StageRuntimeError

def test_large_result_allowed_and_budget_still_enforced(monkeypatch):
    rows=[('task', {'payload':b'x'*(9*1024*1024)})]
    monkeypatch.setattr(ex,'_run_shard',lambda shard:rows)
    with pytest.raises(ex.WorkerTaskError,match='STREAM_RESULT_BYTES_LIMIT'):
        ex._run_bounded_stream_shard([],8*1024*1024)
    assert ex._run_bounded_stream_shard([],64*1024*1024)==rows
    size=len(pickle.dumps(rows,protocol=5))
    assert ex._run_bounded_stream_shard([],size)==rows
    with pytest.raises(ex.WorkerTaskError,match='STREAM_RESULT_BYTES_LIMIT'):
        ex._run_bounded_stream_shard([],size-1)

def test_api_accepts_registered_budget_and_rejects_invalid_values(monkeypatch):
    # Stop at the execution permit: validation must accept >8MiB before any work.
    runtime=object.__new__(StageScienceRuntime)
    def reached():raise RuntimeError('EXECUTION_PERMIT_REACHED')
    monkeypatch.setattr(runtime,'_require_execution',reached)
    kwargs=dict(phase_id='engineering',tasks=[],evaluator_ref='project.worker:evaluate')
    for value in [0,-1,True,None,1.5]:
        with pytest.raises(StageRuntimeError,match='GENERATION_TASK_LIMIT_POLICY'):
            runtime.run_content_indexed_generation(**kwargs,max_result_bytes=value)
    with pytest.raises(RuntimeError,match='EXECUTION_PERMIT_REACHED'):
        runtime.run_content_indexed_generation(**kwargs,max_result_bytes=64*1024*1024)
