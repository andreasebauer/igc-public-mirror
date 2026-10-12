"""Child-to-parent phase evidence checks; native registered validation only."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
from infinity_grid import validation_reports as vr
from infinity_grid import submission as sub
from infinity_grid.canon import canonical_sha256,write_json_atomic
NODE='tests/example.py::test_one'
def recorder(root,phases=('setup','call','teardown'),failure=False):
    r=vr.Recorder(root,'bound',['tests/example.py'])
    r.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(nodeid=NODE)]))
    r.pytest_runtest_logstart(NODE,None)
    for when in phases:
        fail=failure and when=='call'
        r.pytest_runtest_logreport(SimpleNamespace(nodeid=NODE,when=when,outcome='failed' if fail else 'passed',duration=.01,capstdout='',capstderr='',failed=fail,longrepr='failure' if fail else ''))
    return r

def test_complete_child_packet_verified(tmp_path):
    receipt=recorder(tmp_path).publish_finalization()
    assert vr.node_status(vr.verify_finalization(tmp_path,'bound',receipt)[0])=='PASS'

@pytest.mark.parametrize('damage',['setup_only','missing','changed_outcome'])
def test_parent_rejects_report_drift_after_child_finalization(tmp_path,damage):
    receipt=recorder(tmp_path).publish_finalization();p=tmp_path/'nodes'/(canonical_sha256(NODE)+'.json')
    row=sub._read(p)
    if damage=='missing':p.unlink()
    else:
        if damage=='setup_only':row['phases']={'setup':row['phases']['setup']};row['finished']=False
        else:row['phases']['call']['outcome']='failed'
        write_json_atomic(p,row)
    with pytest.raises(RuntimeError,match='VALIDATION_FINAL_REPORT_CHANGED'):vr.verify_finalization(tmp_path,'bound',receipt)

@pytest.mark.parametrize('damage',['missing','altered'])
def test_parent_rejects_missing_or_changed_final_packet(tmp_path,damage):
    receipt=recorder(tmp_path).publish_finalization();p=tmp_path/'finalizations'/(receipt['sha256']+'.json')
    if damage=='missing':p.unlink()
    else:p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(RuntimeError,match='VALIDATION_FINAL_'):vr.verify_finalization(tmp_path,'bound',receipt)

def test_parent_rejects_foreign_binding(tmp_path):
    receipt=recorder(tmp_path).publish_finalization()
    with pytest.raises(RuntimeError,match='VALIDATION_FINAL_BINDING'):vr.verify_finalization(tmp_path,'other',receipt)

def test_packet_never_fabricates_missing_phases(tmp_path):
    receipt=recorder(tmp_path,phases=('setup',)).publish_finalization()
    row=vr.verify_finalization(tmp_path,'bound',receipt)[0]
    assert vr.node_status(row)=='INTERRUPTED' and set(row['phases'])=={'setup'}

def test_packet_preserves_actual_failure(tmp_path):
    receipt=recorder(tmp_path,failure=True).publish_finalization()
    assert vr.node_status(vr.verify_finalization(tmp_path,'bound',receipt)[0])=='FAIL'

def test_missing_collected_node_refused(tmp_path):
    r=recorder(tmp_path);r.observed.clear()
    with pytest.raises(RuntimeError,match='VALIDATION_FINAL_COLLECTION_MISMATCH'):r.publish_finalization()
