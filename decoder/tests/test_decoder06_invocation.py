"""Run only as frozen registered Decoder validation nodes.

Real saved SCRIPT/STAGE jobs supply positive execution and repair evidence.
These tests do not fabricate positive save receipts or controller authority.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import pytest

from infinity_grid import v05_controller_event_loop as loop
from infinity_grid import v05_origin_guard as origin
from infinity_grid.invocation import InvocationRefused, refusal_details
from infinity_grid.workflow_guard import preflight


def test_public_scope_refuses_and_has_working_run_instruction(tmp_path):
    with pytest.raises(InvocationRefused) as issue:
        with origin.registered_workspace_scope(tmp_path, 'SAVED.JOB'):
            pytest.fail('scope granted authority')
    row = issue.value.as_dict()
    assert row['reason_code'] == 'RECORDED_ATTEMPT_REQUIRED'
    assert row['next_supported_operation'] == 'run'
    assert row['required_arguments'] == {'workspace':str(tmp_path), 'job_id':'SAVED.JOB'}
    assert origin.current_execution_context_snapshot() is None


def test_private_scope_refuses_even_before_reading_attempt(tmp_path):
    with pytest.raises(InvocationRefused, match='NATIVE_RUN_ENTRY_REQUIRED'):
        with origin._registered_attempt_scope({}, tmp_path/'absent', tmp_path/'out'):
            pytest.fail('private scope granted authority')
    assert not (tmp_path/'out').exists()


def test_legacy_context_key_cannot_create_unbound_root():
    with pytest.raises(InvocationRefused, match='RECORDED_ATTEMPT_REQUIRED'):
        with origin._controller_event_scope('old-client', _key=origin._CONTEXT_MINT_KEY):
            pytest.fail('unbound context granted authority')


def test_direct_internal_dispatch_has_no_effect(tmp_path):
    with pytest.raises(InvocationRefused): loop._dispatch_workspace_job({}, {}, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_direct_script_runtime_has_no_effect(tmp_path):
    from infinity_grid.script_runtime import run_captured_script
    with pytest.raises(InvocationRefused): run_captured_script({}, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_direct_task_service_does_not_consume_generator():
    from infinity_grid.execution import execute_tasks, execute_tasks_stream
    def forbidden():
        pytest.fail('task generator evaluated before admission')
        yield
    for run, args in ((execute_tasks, {}), (execute_tasks_stream, {'on_result':lambda *a:None})):
        with pytest.raises(InvocationRefused): run(forbidden(), worker_ref='missing:worker', **args)


def test_direct_validation_refuses_before_output(tmp_path):
    from infinity_grid.v05_validation_runtime import run_registered_validation_nodes
    with pytest.raises(InvocationRefused):
        run_registered_validation_nodes(tmp_path, ['tests/x.py'], workers=1, wall_seconds_max=1, output_dir=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_direct_publication_refuses_before_path_access(tmp_path):
    from infinity_grid.v05_publication import publish_verified_execution, publish_registered_chain_execution
    with pytest.raises(InvocationRefused): publish_verified_execution(None, 'job', tmp_path/'x')
    with pytest.raises(InvocationRefused): publish_registered_chain_execution(None, 'job')
    assert not list(tmp_path.iterdir())


def test_direct_worker_scope_refuses():
    with pytest.raises(InvocationRefused):
        with origin._forked_worker_scope('direct'):
            pytest.fail('worker scope opened')


def test_native_worker_initialization_cannot_be_called_directly():
    from infinity_grid.execution import _pool_initializer, _serial_initializer
    from infinity_grid.v05_stage_runtime import _init_partition_worker
    with pytest.raises(InvocationRefused): _pool_initializer('missing:fn', None, None)
    with pytest.raises(InvocationRefused): _serial_initializer('missing:fn', None, None)
    with pytest.raises(InvocationRefused): _init_partition_worker({})


def test_direct_handler_pool_is_rejected_before_import(tmp_path):
    path = tmp_path/'handler.py'
    path.write_text('from multiprocessing import Pool as NativePool\ndef handler(stage, runtime):\n    return NativePool(2)\n')
    with pytest.raises(InvocationRefused, match='PRIVATE_EXECUTION_PREFLIGHT'):
        preflight(tmp_path, [path])


def test_helper_pool_rejected_before_import_side_effect(tmp_path):
    project = tmp_path/'project'; project.mkdir()
    entry = project/'entry.py'; entry.write_text('import helper\n')
    helper = project/'helper.py'
    helper.write_text('from concurrent.futures import ProcessPoolExecutor as PoolAlias\n'
                      'from pathlib import Path\nPath("IMPORTED").write_text("bad")\n'
                      'def parallel():\n    return PoolAlias(2)\n')
    before = helper.read_bytes()
    with pytest.raises(InvocationRefused, match='PRIVATE_EXECUTION_PREFLIGHT') as issue:
        preflight(tmp_path, [entry])
    assert issue.value.checks[0]['path'] == 'project/helper.py'
    assert helper.read_bytes() == before and not Path('IMPORTED').exists()


def test_version_style_helper_name_is_not_exempt(tmp_path):
    package = tmp_path/'infinity_grid'; package.mkdir()
    (package/'handler.py').write_text('from . import v05_invented_helper\n')
    (package/'v05_invented_helper.py').write_text('import multiprocessing as mp\ndef work():\n    return mp.Pool(2)\n')
    with pytest.raises(InvocationRefused, match='PRIVATE_EXECUTION_PREFLIGHT'):
        preflight(tmp_path, [package/'handler.py'])


def test_pure_cache_tempfiles_shutil_and_main_are_allowed(tmp_path):
    path = tmp_path/'ordinary.py'
    path.write_text('from functools import lru_cache\nimport tempfile, shutil\n'
                    '@lru_cache(None)\ndef square(n): return n*n\n'
                    'def main():\n    return square(3)\n'
                    'if __name__ == "__main__": main()\n')
    assert preflight(tmp_path, [path])['status'] == 'PASS'


def test_refusal_contract_does_not_claim_missing_artifacts_saved(tmp_path):
    row = refusal_details(ValueError('JOB_NOT_REGISTERED'), 'run', tmp_path, 'unknown')
    required = {'status','reason_code','operation_attempted','preserved_artifact_ids','unmet_checks',
                'next_supported_operation','required_arguments','retry_is_safe'}
    assert required <= row.keys()
    assert row['preserved_artifact_ids'] == [] and row['retry_is_safe'] is False


def test_neutral_entrypoints_and_legacy_aliases_share_native_controller():
    import tomllib
    from infinity_grid import controller, cli
    source = Path(loop.__file__).resolve().parents[1]
    scripts = tomllib.loads((source/'pyproject.toml').read_text())['project']['scripts']
    assert all(scripts[x] == 'infinity_grid.controller:main' for x in ('ig', 'igref', 'ig-decoder','ig-registered-service'))
    with pytest.raises(InvocationRefused): cli._historical_main([])
    assert 'from .controller import main' in (source/'infinity_grid/__main__.py').read_text()


def test_saved_current_validation_is_attempt_bound():
    # Source is executed in a native validation child, which has no root authority.
    root = Path(loop.__file__).resolve().parents[2]
    from infinity_grid import submission
    record = submission.capture_record(root)
    running = [json.loads(p.read_text()) for p in (root/'runtime/attempts').rglob('*.json')]
    active = [x for x in running if x['status'] == 'RUNNING']
    assert len(active) == 1
    assert active[0]['source_sha256'] == record['job']['source_sha256']
    assert active[0]['registration_sha256'] == record['job']['registration_sha256']
    assert active[0]['operation'] and active[0]['attempt_id'] and active[0]['output_root']
    assert origin.current_execution_context_snapshot() is None


@pytest.mark.parametrize('allow_scheduler', [False, True])
def test_private_process_refused_with_separate_admission_root(tmp_path, allow_scheduler):
    # Real audit event: a refusal must happen before the subprocess can run.
    import subprocess
    from infinity_grid.workflow_guard import scientific_call
    marker = tmp_path/'PROCESS_STARTED'
    with scientific_call(tmp_path, allow_scheduler=allow_scheduler):
        with pytest.raises(InvocationRefused, match='PRIVATE_EXECUTION_RUNTIME'):
            subprocess.run([sys.executable, '-c',
                'from pathlib import Path; Path('+repr(str(marker))+').write_text("bad")'], check=True)
    assert not marker.exists()


def test_interrupted_attempt_reconciliation_retains_original(tmp_path):
    from infinity_grid.canon import canonical_sha256
    original={'status':'RUNNING','request_id':'r','job_id':'j','attempt_id':'r:000001',
              'source_sha256':'s','registration_sha256':'g','pid':os.getpid(),
              'output_root':'/previous/environment','started_unix':1}
    p=tmp_path/'000001.json';p.write_text(json.dumps(original))
    done=tmp_path/'000002.json';done.write_text('{"status":"COMPLETED"}')
    unchanged=done.read_bytes()
    loop._reconcile_workspace_attempts(tmp_path,'r','j','s','g')
    result=json.loads(p.read_text())
    assert result['status']=='INTERRUPTED'
    assert result['recovery']['prior_record']==original
    assert result['recovery']['prior_record_sha256']==canonical_sha256(original)
    assert done.read_bytes()==unchanged
    first=p.read_bytes()
    loop._reconcile_workspace_attempts(tmp_path,'r','j','s','g')
    assert p.read_bytes()==first


def test_interrupted_attempt_binding_mismatch_is_atomic(tmp_path):
    original={'status':'RUNNING','request_id':'r','job_id':'j','attempt_id':'r:000001',
              'source_sha256':'s','registration_sha256':'g'}
    a=tmp_path/'000001.json';a.write_text(json.dumps(original))
    b=tmp_path/'000002.json';b.write_text(json.dumps(dict(original,attempt_id='r:000002',source_sha256='other')))
    before={p.name:p.read_bytes() for p in (a,b)}
    with pytest.raises(loop.ControllerLoopError,match='INTERRUPTED_ATTEMPT_BINDING_MISMATCH'):
        loop._reconcile_workspace_attempts(tmp_path,'r','j','s','g')
    assert {p.name:p.read_bytes() for p in (a,b)}==before
