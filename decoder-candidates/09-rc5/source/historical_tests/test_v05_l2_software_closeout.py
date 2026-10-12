from pathlib import Path
import json
import pytest
import infinity_grid
from infinity_grid.v05_execution_authority import ExecutionAuthorityError, OFFICIAL_ROLE
from infinity_grid.v05_registered_service import RegisteredServiceSession, POLICY_SCHEMA
from infinity_grid.v05_chain import ScientificChainController

def test_build_meta_declares_fail_closed_l2_software_closeout():
    m=infinity_grid.build_meta()
    assert m["L2_complete"] is True
    assert m["status"]=="SOFTWARE_RELEASE_ACCEPTED_FAIL_CLOSED"
    assert m["official_execution"]=="PAUSED_PENDING_PROTECTED_DEPLOYMENT"
    assert m["production_key_created"] is False and m["production_service_deployed"] is False
    assert m["remaining_release_gates"]==[]
    assert len(m["deployment_prerequisites"])==3

def test_closeout_resource_keeps_g6_unchanged_and_official_disabled():
    p=Path(infinity_grid.__file__).resolve().parent/'resources/v05/L2_SOFTWARE_CLOSEOUT_V1.json'
    x=json.loads(p.read_text())
    assert x["software_l2_complete"] is True
    assert x["official_science_enabled"] is False
    assert x["g6_science_changed"] is False
    assert x["accepted_evidence"]["historical_private_executor"]=="DISABLED_INSPECT_ONLY"

def test_default_controller_still_has_no_local_official_fallback(tmp_path):
    c=ScientificChainController(tmp_path/'chain')
    with pytest.raises(ExecutionAuthorityError,match='OFFICIAL_AUTHORITY_NOT_CONFIGURED'):
        c._authority.require_run({})

def test_official_service_policy_still_waits_before_store_creation(tmp_path):
    cfg=tmp_path/'cfg';cfg.mkdir(mode=0o700)
    key=cfg/'key.bin';key.write_bytes(b'0'*32);key.chmod(0o600)
    policy={'schema_id':POLICY_SCHEMA,'role':OFFICIAL_ROLE,'source_sha256':'0'*64,'service_root':str(tmp_path/'service'),'signing_key_path':str(key),'public_key_hex':'00'*32,'registrations':[]}
    pp=cfg/'policy.json';pp.write_text(json.dumps(policy,sort_keys=True));pp.chmod(0o600)
    with pytest.raises(ExecutionAuthorityError,match='OFFICIAL_SERVICE_ACCEPTANCE_PENDING'):
        RegisteredServiceSession(pp)
    assert not (tmp_path/'service').exists()


def test_engineering_worker_launches_from_controller_descended_child_copy():
    import inspect
    import infinity_grid.v05_engineering_jobs as engineering_jobs
    src=inspect.getsource(engineering_jobs._forked_engineering_worker)
    assert 'os.chdir(candidate)' in src
    assert '_forked_worker_scope' in src


def test_c1_direct_chain_run_rejected_before_registration_io(tmp_path):
    c=ScientificChainController(tmp_path/'direct-run', engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        c.run('NO-CHAIN')
    assert not (tmp_path/'direct-run'/'NO-CHAIN').exists()


def test_c1_direct_resume_rejected_before_chain_lookup(tmp_path):
    c=ScientificChainController(tmp_path/'direct-resume', engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        c.resume('NO-CHAIN')
    assert not (tmp_path/'direct-resume'/'NO-CHAIN').exists()


def test_c1_direct_stage_runtime_rejected_before_runtime_io(tmp_path):
    from infinity_grid.v05_stage_runtime import StageScienceRuntime
    root=tmp_path/'runtime-not-created'
    with pytest.raises(ExecutionAuthorityError, match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        StageScienceRuntime(chain_dir=root,chain_id='ENG',stage_id='ENG:S0',question_sha256='2'*64)
    assert not root.exists()


def test_c1_direct_registered_publication_rejected_by_origin_first(tmp_path):
    from infinity_grid.v05_publication import publish_registered_chain_execution
    c=ScientificChainController(tmp_path/'publish', engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        publish_registered_chain_execution(c,'NO-CHAIN')


def test_c1_guard_is_central_and_bootstrap_exception_is_explicit():
    import inspect
    import infinity_grid.v05_chain as chain
    import infinity_grid.v05_registered_service as service
    import infinity_grid.v05_stage_runtime as runtime
    import infinity_grid.v05_publication as publication
    from infinity_grid.v05_origin_guard import REJECT_EXTERNAL_EXECUTION_ORIGIN
    assert REJECT_EXTERNAL_EXECUTION_ORIGIN == 'REJECT_EXTERNAL_EXECUTION_ORIGIN'
    assert 'require_controller_execution_origin' in inspect.getsource(chain.ScientificChainController.run)
    assert 'require_controller_execution_origin' in inspect.getsource(chain.ScientificChainController.resume)
    assert 'require_controller_execution_origin' in inspect.getsource(runtime.StageScienceRuntime._require_execution)
    assert 'require_controller_execution_origin' in inspect.getsource(publication.publish_registered_chain_execution)
    # C1-C5 migration exception only; C5 must remove this service bootstrap scope.
    assert '_bootstrap_controller_event_scope' in inspect.getsource(service.RegisteredExecutionService.dispatch)
    assert '_bootstrap_registered_service_scope' not in inspect.getsource(service.RegisteredExecutionService.dispatch)



def test_c2_controller_context_identity_and_nested_reuse():
    from infinity_grid.v05_origin_guard import (
        CONTROLLER_EVENT_ORIGIN, CONTROLLER_ROOT_ROLE,
        _test_controller_event_scope, current_execution_context_snapshot,
        current_execution_origin,
    )
    assert current_execution_context_snapshot() is None
    with _test_controller_event_scope():
        root = current_execution_context_snapshot()
        assert root is not None
        assert root['role'] == CONTROLLER_ROOT_ROLE
        assert root['origin'] == CONTROLLER_EVENT_ORIGIN
        assert root['parent_context_id'] is None
        assert root['controller_session_id'].startswith('session-')
        assert root['root_execution_id'].startswith('root-')
        assert root['context_id'].startswith('ctx-')
        assert current_execution_origin() == CONTROLLER_EVENT_ORIGIN
        with _test_controller_event_scope():
            nested = current_execution_context_snapshot()
            assert nested == root
    assert current_execution_context_snapshot() is None


def test_c2_worker_context_is_derived_and_cannot_reenter_root():
    from infinity_grid.v05_origin_guard import (
        REJECT_WORKER_ROOT_REENTRY, WORKER_EXECUTION_ORIGIN, WORKER_ROLE,
        _test_controller_event_scope, _test_worker_scope,
        current_execution_context_snapshot, require_controller_execution_origin,
    )
    with _test_controller_event_scope():
        root = current_execution_context_snapshot()
        with _test_worker_scope('c2-worker'):
            worker = current_execution_context_snapshot()
            assert worker['role'] == WORKER_ROLE
            assert worker['origin'] == WORKER_EXECUTION_ORIGIN
            assert worker['controller_session_id'] == root['controller_session_id']
            assert worker['root_execution_id'] == root['root_execution_id']
            assert worker['parent_context_id'] == root['context_id']
            assert worker['context_id'] != root['context_id']
            with pytest.raises(ExecutionAuthorityError, match=REJECT_WORKER_ROOT_REENTRY):
                require_controller_execution_origin('worker-root-attempt')
            with pytest.raises(ExecutionAuthorityError, match=REJECT_WORKER_ROOT_REENTRY):
                with _test_controller_event_scope():
                    pass
        assert current_execution_context_snapshot() == root


def test_c2_worker_scope_requires_live_controller_root():
    from infinity_grid.v05_origin_guard import (
        REJECT_EXTERNAL_EXECUTION_ORIGIN, _test_worker_scope,
    )
    with pytest.raises(ExecutionAuthorityError, match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        with _test_worker_scope('orphan-worker'):
            pass


def test_c2_context_lifetime_is_fail_closed_after_scope():
    from infinity_grid.v05_origin_guard import (
        REJECT_EXTERNAL_EXECUTION_ORIGIN, _test_controller_event_scope,
        current_execution_context_snapshot, require_controller_execution_origin,
    )
    with _test_controller_event_scope():
        assert current_execution_context_snapshot() is not None
    assert current_execution_context_snapshot() is None
    with pytest.raises(ExecutionAuthorityError, match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        require_controller_execution_origin('expired-context')


def test_c2_external_environment_cannot_mint_context(monkeypatch):
    from infinity_grid.v05_origin_guard import (
        REJECT_EXTERNAL_EXECUTION_ORIGIN, current_execution_context_snapshot,
        require_controller_execution_origin,
    )
    monkeypatch.setenv('IG_DECODER_EXECUTION_ORIGIN', 'CONTROLLER_EVENT_LOOP')
    monkeypatch.setenv('IG_DECODER_CONTROLLER_SESSION_ID', 'forged')
    monkeypatch.setenv('IG_DECODER_ROOT_EXECUTION_ID', 'forged')
    assert current_execution_context_snapshot() is None
    with pytest.raises(ExecutionAuthorityError, match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        require_controller_execution_origin('forged-environment')


def test_c2_registered_service_bootstrap_mints_controller_root_not_second_origin():
    import inspect
    import infinity_grid.v05_registered_service as service
    import infinity_grid.v05_origin_guard as guard
    src = inspect.getsource(service.RegisteredExecutionService.dispatch)
    assert '_bootstrap_controller_event_scope' in src
    assert '_bootstrap_registered_service_scope' not in src
    assert guard.BOOTSTRAP_SERVICE_INGRESS == 'BOOTSTRAP_REGISTERED_SERVICE'
    assert guard.CONTROLLER_ROOT_ROLE == 'CONTROLLER_ROOT'



def _c3_request(**updates):
    base = {
        'schema_id': 'IG_DECODER_PASSIVE_REQUEST_V1',
        'request_id': 'req-c3-001',
        'registered_job_id': 'JOB.C3.TEST',
        'requested_operation_id': 'RUN_REGISTERED',
        'parent_source_sha256': 'a' * 64,
        'input_artifacts': [{'logical_name': 'fixture', 'sha256': 'b' * 64}],
        'human_note': 'informational only',
    }
    base.update(updates)
    return base


def test_c3_passive_submission_is_data_only_and_creates_no_execution_context(tmp_path):
    import inspect
    import infinity_grid.v05_passive_intake as intake
    from infinity_grid.v05_origin_guard import current_execution_context_snapshot
    assert current_execution_context_snapshot() is None
    receipt = intake.submit_passive_request(tmp_path / 'intake', _c3_request())
    assert receipt['state'] == 'PENDING_PASSIVE'
    assert current_execution_context_snapshot() is None
    pending = tmp_path / 'intake' / 'pending' / 'req-c3-001.json'
    assert pending.is_file()
    assert len(list((tmp_path / 'intake').rglob('*.json'))) == 1
    src = inspect.getsource(intake.submit_passive_request)
    forbidden = ['ScientificChainController', 'RegisteredExecutionService', 'StageScienceRuntime',
                 'subprocess', 'os.system', 'exec(', 'eval(', 'publish_registered_chain_execution']
    for token in forbidden:
        assert token not in src


def test_c3_passive_request_rejects_execution_selection_and_identity_fields(tmp_path):
    import infinity_grid.v05_passive_intake as intake
    forbidden = [
        'command', 'code', 'module', 'function', 'handler', 'evaluator', 'python',
        'subprocess', 'environment', 'workers', 'worker_count', 'origin',
        'controller_session_id', 'root_execution_id', 'context_id', 'signing_key',
        'publication_target', 'path', 'url',
    ]
    for field in forbidden:
        req = _c3_request(request_id='req-' + field.replace('_', '-'))
        req[field] = 'x'
        with pytest.raises(intake.PassiveRequestError, match='PASSIVE_REQUEST_FIELD_FORBIDDEN'):
            intake.submit_passive_request(tmp_path / 'intake', req)


def test_c3_artifact_reference_is_hash_only_no_path_or_url(tmp_path):
    import infinity_grid.v05_passive_intake as intake
    for field in ['path', 'url', 'module', 'handler']:
        req = _c3_request(request_id='req-art-' + field)
        req['input_artifacts'] = [{'logical_name': 'fixture', 'sha256': 'b' * 64, field: 'x'}]
        with pytest.raises(intake.PassiveRequestError, match='PASSIVE_REQUEST_ARTIFACT_FIELD_FORBIDDEN'):
            intake.submit_passive_request(tmp_path / 'intake', req)


def test_c3_duplicate_request_is_not_overwritten(tmp_path):
    import infinity_grid.v05_passive_intake as intake
    root = tmp_path / 'intake'
    first = intake.submit_passive_request(root, _c3_request())
    path = root / 'pending' / 'req-c3-001.json'
    before = path.read_bytes()
    with pytest.raises(intake.PassiveRequestError, match='PASSIVE_REQUEST_DUPLICATE'):
        intake.submit_passive_request(root, _c3_request(human_note='changed'))
    assert path.read_bytes() == before
    assert first['request_sha256'] == intake.submit_passive_request(
        tmp_path / 'other', _c3_request()
    )['request_sha256']


def test_c3_direct_external_ingestion_is_rejected_by_c2_origin_guard(tmp_path):
    import infinity_grid.v05_passive_intake as intake
    from infinity_grid.v05_execution_authority import ExecutionAuthorityError
    from infinity_grid.v05_origin_guard import REJECT_EXTERNAL_EXECUTION_ORIGIN
    root = tmp_path / 'intake'
    intake.submit_passive_request(root, _c3_request())
    registry = {'JOB.C3.TEST': {
        'allowed_operations': ['RUN_REGISTERED'],
        'registration_sha256': 'c' * 64,
        'implementation_sha256': 'd' * 64,
    }}
    with pytest.raises(ExecutionAuthorityError, match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        intake.ingest_passive_request(
            root, 'req-c3-001', accepted_registry=registry,
            expected_parent_source_sha256='a' * 64, internal_root=tmp_path / 'internal')


def test_c3_controller_ingestion_mints_new_internal_identity_and_does_not_execute(tmp_path):
    import inspect
    import infinity_grid.v05_passive_intake as intake
    from infinity_grid.v05_origin_guard import _test_controller_event_scope, current_execution_context_snapshot
    root = tmp_path / 'intake'
    intake.submit_passive_request(root, _c3_request())
    registry = {'JOB.C3.TEST': {
        'allowed_operations': ['RUN_REGISTERED'],
        'registration_sha256': 'c' * 64,
        'implementation_sha256': 'd' * 64,
    }}
    with _test_controller_event_scope():
        ctx = current_execution_context_snapshot()
        record = intake.ingest_passive_request(
            root, 'req-c3-001', accepted_registry=registry,
            expected_parent_source_sha256='a' * 64, internal_root=tmp_path / 'internal')
        assert record['internal_execution_id'].startswith('intent-')
        assert record['internal_execution_id'] != record['external_request_id']
        assert record['registration_sha256'] == 'c' * 64
        assert record['implementation_sha256'] == 'd' * 64
        assert record['controller_root_execution_id'] == ctx['root_execution_id']
        assert record['controller_context_id'] == ctx['context_id']
        assert record['state'] == 'INGESTED_NOT_EXECUTED'
    internal_files = list((tmp_path / 'internal' / 'ingested').glob('*.json'))
    assert len(internal_files) == 1
    src = inspect.getsource(intake.ingest_passive_request)
    forbidden = ['ScientificChainController', 'RegisteredExecutionService', 'StageScienceRuntime',
                 'subprocess', 'os.system', 'publish_registered_chain_execution']
    for token in forbidden:
        assert token not in src


def test_c3_ingestion_uses_controller_registry_and_parent_not_request_supplied_code(tmp_path):
    import infinity_grid.v05_passive_intake as intake
    from infinity_grid.v05_origin_guard import _test_controller_event_scope
    root = tmp_path / 'intake'
    intake.submit_passive_request(root, _c3_request())
    bad_registry = {'JOB.C3.TEST': {
        'allowed_operations': ['OTHER'],
        'registration_sha256': 'c' * 64,
        'implementation_sha256': 'd' * 64,
    }}
    with _test_controller_event_scope():
        with pytest.raises(intake.PassiveRequestError, match='PASSIVE_INGEST_OPERATION_NOT_REGISTERED'):
            intake.ingest_passive_request(root, 'req-c3-001', accepted_registry=bad_registry,
                                         expected_parent_source_sha256='a' * 64,
                                         internal_root=tmp_path / 'internal-a')
        good_registry = {'JOB.C3.TEST': {
            'allowed_operations': ['RUN_REGISTERED'],
            'registration_sha256': 'c' * 64,
            'implementation_sha256': 'd' * 64,
        }}
        with pytest.raises(intake.PassiveRequestError, match='PASSIVE_INGEST_PARENT_MISMATCH'):
            intake.ingest_passive_request(root, 'req-c3-001', accepted_registry=good_registry,
                                         expected_parent_source_sha256='e' * 64,
                                         internal_root=tmp_path / 'internal-b')


def test_c3_policy_resource_matches_data_only_contract():
    import json
    from importlib import resources
    raw = resources.files('infinity_grid').joinpath('resources/v05/C3_PASSIVE_INTAKE_V1.json').read_text(encoding='utf-8')
    obj = json.loads(raw)
    assert obj['submission_effect'] == 'PASSIVE_RECORD_ONLY'
    assert obj['controller_ingestion_required'] is True
    assert obj['ingestion_executes_job'] is False
    assert obj['final_origin_exclusivity'] is False



def test_c4_route_inventory_is_default_deny_with_passive_submit_only():
    from infinity_grid.v05_route_closure import route_inventory
    x=route_inventory()
    assert x['normal_external_request_surfaces']==['passive_request.submit']
    assert x['default_policy']=='DENY'
    assert x['final_origin_exclusivity'] is False
    assert x['temporary_c5_exceptions']==[
        'bootstrap_registered_service_controller_event_ingress',
        'non_authoritative_test_controller_scope',
        'non_authoritative_test_worker_scope',
    ]


def test_c4_cli_classifier_default_denies_execution_and_unknown_routes():
    from types import SimpleNamespace
    from infinity_grid.v05_route_closure import classify_external_cli, REJECT, READ_ONLY, PASSIVE_SUBMIT
    assert classify_external_cli(SimpleNamespace(cmd='run'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='resume'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='release',release_cmd='create'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='uplift',uplift_cmd='native-run'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='jumpstart',jumpstart_cmd='launch'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='science',science_cmd='run'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='future-new-route'))==REJECT
    assert classify_external_cli(SimpleNamespace(cmd='status'))==READ_ONLY
    assert classify_external_cli(SimpleNamespace(cmd='request',request_cmd='submit'))==PASSIVE_SUBMIT


def test_c4_cli_passive_submit_creates_record_only(tmp_path):
    import json
    from infinity_grid.cli import main
    from infinity_grid.v05_origin_guard import current_execution_context_snapshot
    req={
        'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1','request_id':'c4-cli-passive',
        'registered_job_id':'JOB.C4.TEST','requested_operation_id':'RUN_REGISTERED',
        'parent_source_sha256':'a'*64,'input_artifacts':[],'human_note':'data only',
    }
    rp=tmp_path/'request.json'; rp.write_text(json.dumps(req),encoding='utf-8')
    assert current_execution_context_snapshot() is None
    rc=main(['request','submit','--intake-root',str(tmp_path/'intake'),'--request',str(rp)])
    assert rc==0
    assert (tmp_path/'intake'/'pending'/'c4-cli-passive.json').is_file()
    assert current_execution_context_snapshot() is None
    assert len(list((tmp_path/'intake').rglob('*.json')))==1


def test_c4_direct_worker_entrypoints_reject_before_job_io(tmp_path):
    from infinity_grid.v05_execution_authority import ExecutionAuthorityError
    from infinity_grid.v05_origin_guard import REJECT_EXTERNAL_EXECUTION_ORIGIN
    from infinity_grid.v05_worker import worker_main as science_worker
    from infinity_grid.v05_engineering_worker import worker_main as engineering_worker
    missing=tmp_path/'does-not-exist.json'
    with pytest.raises(ExecutionAuthorityError,match=REJECT_EXTERNAL_EXECUTION_ORIGIN): science_worker(missing)
    with pytest.raises(ExecutionAuthorityError,match=REJECT_EXTERNAL_EXECUTION_ORIGIN): engineering_worker(missing)


def test_c4_direct_legacy_launcher_is_rejected_without_parsing_work():
    from infinity_grid.v05_launcher import main
    assert main([])==4


def test_c4_controller_descended_fork_mints_worker_child_context():
    from infinity_grid.v05_origin_guard import _test_controller_event_scope, current_execution_context_snapshot
    from infinity_grid.v05_route_closure import controller_descended_worker_probe
    with _test_controller_event_scope():
        root=current_execution_context_snapshot()
        child=controller_descended_worker_probe()
        assert child['role']=='WORKER'
        assert child['root_execution_id']==root['root_execution_id']
        assert child['controller_session_id']==root['controller_session_id']
        assert child['parent_context_id']==root['context_id']


def test_c4_registration_install_freeze_mirror_and_protected_publish_require_root(tmp_path):
    from infinity_grid.v05 import V05RegistrationStore
    from infinity_grid.v05_chain import ScientificChainController
    from infinity_grid.v05_publication import publish_verified_execution
    from infinity_grid.v05_execution_authority import ExecutionAuthorityError
    from infinity_grid.v05_origin_guard import REJECT_EXTERNAL_EXECUTION_ORIGIN
    with pytest.raises(ExecutionAuthorityError,match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        V05RegistrationStore(tmp_path/'store').install({})
    c=ScientificChainController(tmp_path/'chain',engineering_only=True)
    with pytest.raises(ExecutionAuthorityError,match=REJECT_EXTERNAL_EXECUTION_ORIGIN): c.freeze({})
    with pytest.raises(ExecutionAuthorityError,match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        c.acknowledge_external_mirror('NOCHAIN','S0',commit_sha256='a'*64,mirror_uri='x',mirror_sha256='b'*64)
    class P: store=tmp_path/'store2'
    with pytest.raises(ExecutionAuthorityError,match=REJECT_EXTERNAL_EXECUTION_ORIGIN):
        publish_verified_execution(P(),'NO-RUN',tmp_path/'no-report')


def test_c4_launcher_source_has_no_internal_worker_dispatch_path():
    import inspect
    import infinity_grid.v05_launcher as launcher
    src=inspect.getsource(launcher.main)
    assert 'REJECT_DIRECT_EXECUTION_ROUTE' in src
    assert '_run_worker(ns)' not in src
    assert '_run_engineering_worker(ns)' not in src


def test_c4_resource_matches_route_closure_contract():
    import json
    from importlib import resources
    obj=json.loads(resources.files('infinity_grid').joinpath('resources/v05/C4_DIRECT_ROUTE_CLOSURE_V1.json').read_text(encoding='utf-8'))
    assert obj['normal_external_request_surfaces']==['passive_request.submit']
    assert obj['external_cli_default_policy']=='DENY'
    assert obj['legacy_launcher_direct_invocation']=='REJECTED'
    assert obj['worker_transport']=='CONTROLLER_DESCENDED_FORK_CONTEXT'
    assert obj['g6_science_effect']=='NONE'
    assert obj['final_origin_exclusivity'] is False
