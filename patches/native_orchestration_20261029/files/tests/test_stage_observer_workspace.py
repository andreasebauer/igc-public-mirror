"""Native stage observers must fit the runtime's closed workspace layout."""
import pytest
from infinity_grid.acceptance_observer import session
from infinity_grid.v05_stage_runtime import _stage_workspace_bytes


def test_native_stage_observer_is_accounted_during_execution_and_recovery(tmp_path):
    with session(tmp_path/'telemetry'):
        assert _stage_workspace_bytes(tmp_path) > 0
    expected=sum(p.stat().st_size for p in (tmp_path/'telemetry').rglob('*') if p.is_file())
    assert _stage_workspace_bytes(tmp_path)==expected


@pytest.mark.parametrize('kind',['unknown','symlink','oversize','nested'])
def test_stage_observer_keeps_exact_layout_checks(tmp_path,kind):
    root=tmp_path/'telemetry'
    with session(root):pass
    if kind=='unknown':(root/'arbitrary.json').write_text('{}')
    elif kind=='symlink':
        (root/'ACCEPTANCE_STATUS.json').unlink()
        (root/'ACCEPTANCE_STATUS.json').symlink_to(tmp_path/'missing')
    elif kind=='oversize':(root/'ACCEPTANCE_STATUS.json').write_bytes(b'x'*65537)
    else:(root/'unknown').mkdir()
    with pytest.raises(ValueError):_stage_workspace_bytes(tmp_path)
