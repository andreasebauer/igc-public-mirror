from pathlib import Path
from unittest.mock import patch
import pytest
from infinity_grid.execution import ExecutionPolicy,TaskSpec,_run_bounded_stream_shard
from infinity_grid.v05_stage_runtime import _bounded_task_list,StageRuntimeError
from infinity_grid.runtime_capacity import host_memory_ceiling_bytes


def test_host_ceiling_uses_actual_cgroup_limit():
    raw=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if raw!='max':assert 0<host_memory_ceiling_bytes()<=int(raw)


def test_unmeasured_time_and_result_defaults_are_absent():
    policy=ExecutionPolicy()
    assert policy.lease_timeout_seconds is None
    assert policy.stream_work_seconds_max is None
    assert policy.stream_result_bytes_max is None


def test_stream_result_above_old_cap_is_allowed_unless_explicitly_bounded():
    rows=[('t',b'x'*(9*1024**2))]
    with patch('infinity_grid.execution._run_shard',return_value=rows):
        assert _run_bounded_stream_shard([],None) is rows
        with pytest.raises(Exception,match='STREAM_RESULT_BYTES_LIMIT'):
            _run_bounded_stream_shard([],8*1024**2)


def test_large_phase_payload_is_not_rejected_by_old_fixed_caps():
    tasks=[TaskSpec('t','POLICY','a'*64,{'payload':'x'*(5*1024**2)})]
    assert _bounded_task_list(tasks,None)==tasks
    with pytest.raises(StageRuntimeError,match='PHASE_TASK_COUNT_LIMIT'):
        _bounded_task_list(tasks+tasks,1)
