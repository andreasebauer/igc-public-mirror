from __future__ import annotations
import inspect, json, tempfile
from pathlib import Path
import pytest

def test_c5_parallel_probe_1(): assert sum([1,2,3])==6
def test_c5_parallel_probe_2(): assert ''.join(sorted('cba'))=='abc'
def test_c5_parallel_probe_3(): assert len({1,1,2})==2
def test_c5_parallel_probe_4(): assert (17*19)==323

def test_c5_ar19_enters_worker_scope_only_after_fork():
    from infinity_grid import v05_c5_acceptance as c5
    source=inspect.getsource(c5.run_c5_acceptance)
    ar19=source.split('# AR-19',1)[1].split('# AR-20',1)[0]
    assert ar19.index('pid=os.fork()') < ar19.index("with _forked_worker_scope('c5-ar19')")

def test_c5_ar07_enters_worker_scope_only_after_fork():
    from infinity_grid import v05_c5_acceptance as c5
    source=inspect.getsource(c5.run_c5_acceptance)
    ar07=source.split('# AR-07',1)[1].split('# AR-08',1)[0]
    assert ar07.index('pid=os.fork()') < ar07.index("with _forked_worker_scope('c5-ar07')")

def test_c5_route_inventory_final_and_passive_only():
    from infinity_grid.v05_route_closure import route_inventory
    x=route_inventory(); assert x['final_origin_exclusivity'] is True; assert x['temporary_c5_exceptions']==[]; assert x['normal_external_request_surfaces']==['passive_request.submit','passive_cancel.submit']; assert x['default_policy']=='DENY'

def test_c5_bootstrap_and_test_root_scopes_absent():
    import infinity_grid.v05_origin_guard as g
    assert not hasattr(g,'_bootstrap_controller_event_scope'); assert not hasattr(g,'_test_controller_event_scope'); assert not hasattr(g,'_test_worker_scope')

def test_c5_registered_service_direct_dispatch_rejected():
    import infinity_grid.v05_registered_service as s
    obj=object.__new__(s.RegisteredExecutionService); obj.session=None
    with pytest.raises(Exception,match='REJECT_DIRECT_EXECUTION_ROUTE|REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        obj.dispatch({'schema_id':s.REQUEST_SCHEMA,'operation':'run','registration_sha256':'0'*64})

def test_c5_engineering_worker_has_no_validation_engine():
    import infinity_grid.v05_engineering_worker as w
    src=inspect.getsource(w.worker_main)
    assert 'pytest.main' not in src and 'subprocess.run' not in src and '_run_group(' not in src and 'validation_results' not in src

def test_c5_validation_runtime_owns_group_topology():
    import infinity_grid.v05_validation_runtime as v
    assert 'full_regression' in v.REGISTERED_VALIDATION_GROUPS and len(v.REGISTERED_VALIDATION_GROUPS['c5_parallel_probe'])==4
    src=inspect.getsource(v.run_registered_validation_groups); assert 'require_controller_execution_origin' in src

def test_c5_direct_source_transition_requires_controller_root(tmp_path):
    from infinity_grid.v05_engineering_jobs import run_controller_registered_source_transition
    with pytest.raises(Exception,match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        run_controller_registered_source_transition(runtime_root=tmp_path,registration={},overlay_path=tmp_path/'x',worker_uid=None,worker_gid=None,workers=1)

def test_c5_passive_submission_still_data_only(tmp_path):
    from infinity_grid.v05_passive_intake import submit_passive_request
    from infinity_grid.v05_origin_guard import current_execution_context_snapshot
    req={'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1','request_id':'c5-passive','registered_job_id':'DECODER.ENGINEERING.SOURCE_CHANGE','requested_operation_id':'APPLY_SOURCE_CHANGE','parent_source_sha256':'a'*64,'input_artifacts':[],'human_note':'data only'}
    assert current_execution_context_snapshot() is None; r=submit_passive_request(tmp_path,req); assert r['state']=='PENDING_PASSIVE'; assert current_execution_context_snapshot() is None; assert len(list(tmp_path.rglob('*.json')))==1

def test_c5_final_resource():
    from importlib import resources
    obj=json.loads(resources.files('infinity_grid').joinpath('resources/v05/C5_FINAL_ORIGIN_EXCLUSIVITY_V1.json').read_text())
    assert obj['final_origin_exclusivity'] is True and obj['normal_external_request_surfaces']==['passive_request.submit','passive_cancel.submit'] and obj['temporary_exceptions']==[]


def test_c5_direct_bootstrap_start_is_closed(tmp_path):
    from infinity_grid.v05_controller_event_loop import start_c5_migration_supervisor
    with pytest.raises(Exception, match='REJECT_DIRECT_EXECUTION_ROUTE:C5_BOOTSTRAP_CLOSED'):
        start_c5_migration_supervisor(tmp_path/'fresh-runtime', tmp_path)
