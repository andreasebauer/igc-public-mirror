from pathlib import Path
from infinity_grid import v05_validation_runtime as runtime
from infinity_grid.validation_reports import report_rows

def test_first_nonzero_selector_stops_one_worker_and_retains_exact_evidence(tmp_path):
    candidate=Path(__file__).resolve().parents[1]
    selectors=['tests/_validation_stop_fixture.py::test_failure',
               'tests/_validation_stop_fixture.py::test_later_must_not_run']
    result=runtime._run_group(candidate,'NEGATIVE_STOP',workers=1,uid=None,gid=None,
        wall_seconds_max=None,registered_nodes=selectors,log_root=tmp_path)
    assert result['status']=='FAIL'
    rows={r['node']:r['status'] for r in result['result']['nodes']}
    assert rows=={selectors[0]:'FAIL',selectors[1]:'NOT_COMPLETED'}
    logs=list((tmp_path/'node_process_logs').glob('*.log'))
    assert len(logs)==1
    assert 'REGISTERED_NEGATIVE_CONTROL' in logs[0].read_text()
    assert 'LATER_SELECTOR_EXECUTED' not in logs[0].read_text()

def test_log_inventory_requires_only_verified_executed_prefix(tmp_path):
    root=tmp_path/'node_process_logs';root.mkdir()
    first='tests/a.py::test_first';later='tests/b.py::test_later'
    (root/('000000-'+runtime.canonical_sha256(first)+'.log')).write_bytes(b'failure')
    runtime._verify_node_evidence_set(tmp_path,[{'nodes':[first,later],'executed_nodes':[first]}])
