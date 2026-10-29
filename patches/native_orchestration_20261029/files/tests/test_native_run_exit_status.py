"""The public native command must preserve unsuccessful completion outcomes."""
import json
import pytest
from infinity_grid import v05_controller_event_loop as loop


@pytest.mark.parametrize('status,expected', [
    ('COMPLETED', 0), ('RESULT_REJECTED', 1),
    ('VALIDATION_FAILED', 1), ('UNKNOWN_OUTCOME', 1),
])
def test_native_run_exit_status(monkeypatch, capsys, status, expected):
    monkeypatch.setattr(loop.sys, 'argv', ['ig-decoder', 'run', '/unused', 'JOB'])
    monkeypatch.setattr(loop, 'run_workspace_job', lambda *args: {'status': status})
    assert loop.main() == expected
    assert json.loads(capsys.readouterr().out)['status'] == status
