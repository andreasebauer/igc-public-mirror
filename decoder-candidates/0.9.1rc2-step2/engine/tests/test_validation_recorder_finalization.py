"""Native registered regressions for final validation phase persistence."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from infinity_grid import validation_reports as vr
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid import submission as sub

NODE='tests/fixture.py::test_case'

def phase(when, outcome='passed'):
    return SimpleNamespace(nodeid=NODE,when=when,outcome=outcome,duration=0.01,
                           capstdout='',capstderr='',failed=outcome=='failed',longrepr='failure' if outcome=='failed' else '')

def record(root, phases=('setup','call','teardown'), fail=None):
    rec=vr.Recorder(root,'bound',['tests/fixture.py']);rec.pytest_runtest_logstart(NODE,None)
    for name in phases:rec.pytest_runtest_logreport(phase(name,'failed' if name==fail else 'passed'))
    return rec,root/'nodes'/(canonical_sha256(NODE)+'.json')

@pytest.mark.parametrize('damage', ['setup_only','missing','call_only'])
def test_finalization_recovers_observed_phases_only(tmp_path,damage):
    rec,p=record(tmp_path);expected=deepcopy(sub._read(p))
    if damage=='missing':p.unlink()
    else:
        stale=deepcopy(expected);stale['phases']={k:v for k,v in stale['phases'].items() if k==('setup' if damage=='setup_only' else 'call')};stale['finished']=False;write_json_atomic(p,stale)
        assert vr.node_status(sub._read(p))=='INTERRUPTED'
    rec.finalize();assert sub._read(p)==expected;assert vr.node_status(sub._read(p))=='PASS'

def test_observation_does_not_depend_on_stale_summary_read(tmp_path):
    rec,p=record(tmp_path,phases=('setup',));stale=sub._read(p);stale['phases']={};write_json_atomic(p,stale)
    rec.pytest_runtest_logreport(phase('call'));rec.pytest_runtest_logreport(phase('teardown'));rec.finalize()
    assert vr.node_status(sub._read(p))=='PASS'

def test_finalization_preserves_missing_call(tmp_path):
    rec,p=record(tmp_path,phases=('setup','teardown'));rec.finalize()
    assert 'call' not in sub._read(p)['phases'];assert vr.node_status(sub._read(p))=='INCOMPLETE'

def test_finalization_preserves_failure(tmp_path):
    rec,p=record(tmp_path,fail='call');rec.finalize();assert vr.node_status(sub._read(p))=='FAIL'

def test_finalization_detects_stale_readback(tmp_path,monkeypatch):
    rec,p=record(tmp_path);stale=sub._read(p);stale['phases'].pop('teardown');stale['finished']=False;write_json_atomic(p,stale)
    with monkeypatch.context() as m:
        m.setattr(vr,'write_json_atomic',lambda *a,**k:None)
        with pytest.raises(RuntimeError,match='VALIDATION_REPORT_FINAL_READBACK'):rec.finalize()

def test_new_attempt_does_not_reuse_observed_prior_phases(tmp_path):
    rec,p=record(tmp_path);rec.pytest_runtest_logstart(NODE,None);rec.pytest_runtest_logreport(phase('setup'));rec.finalize()
    row=sub._read(p);assert row['phases'].keys()=={'setup'};assert vr.node_status(row)=='INTERRUPTED'
    old=list((tmp_path/'prior_reports').glob('*.json'));assert len(old)==1;assert vr.node_status(sub._read(old[0]))=='PASS'
