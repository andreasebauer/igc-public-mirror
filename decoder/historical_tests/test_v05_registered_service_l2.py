from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from infinity_grid.canon import canonical_sha256
from infinity_grid.v05_chain import ScientificChainController, seal_chain_registration
from infinity_grid.controller_only_fixture import stage_handler
from infinity_grid.v05_execution_authority import (
    ENGINEERING_ROLE, OFFICIAL_ROLE, ExecutionAuthorityError, digest, source_tree_digest,
    verify_execution_receipt,
)
from infinity_grid.v05_publication import publish_registered_chain_execution
from infinity_grid.v05_registered_service import (
    RegisteredExecutionService, RegisteredServiceSession, REQUEST_SCHEMA, POLICY_SCHEMA,
    strict_json, validate_request, verify_service_publication,
)
from test_v05_execution_authority_v03088 import registration

PACKAGE = Path(__import__('infinity_grid').__file__).resolve().parent


def write_private(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(0o600)


def setup_service(tmp_path: Path, *, workers=1, mirror=False, regs=None, evaluators=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / 'configuration' / 'engineering-key.bin'
    write_private(key_path, key.private_bytes_raw())
    reg = registration(chain_id='L2-ENG', workers=workers, mirror=mirror)
    regs = regs or [reg]
    policy = {
        'schema_id': POLICY_SCHEMA, 'role': ENGINEERING_ROLE,
        'source_sha256': source_tree_digest(PACKAGE),
        'service_root': str(tmp_path / 'service'), 'signing_key_path': str(key_path),
        'public_key_hex': key.public_key().public_bytes_raw().hex(),
        'registrations': [{'registration': r, 'evaluator_refs': evaluators or [
            'infinity_grid.controller_only_fixture:partition_evaluator']} for r in regs],
    }
    path = tmp_path / 'configuration' / 'policy.json'
    write_private(path, json.dumps(policy, sort_keys=True).encode())
    return path, policy, regs[0]


def req(reg, operation='run'):
    return {'schema_id': REQUEST_SCHEMA, 'operation': operation,
            'registration_sha256': reg['registration_sha256']}


def cli(policy, request):
    proc = subprocess.run([sys.executable, '-m', 'infinity_grid.v05_registered_service',
                           '--policy', str(policy)], input=json.dumps(request),
                          text=True, capture_output=True, timeout=30, check=False)
    return proc, json.loads(proc.stdout)


def chain_commit(policy, reg, stage='ENG:S0'):
    path = Path(policy['service_root']) / 'chains' / reg['chain_id'] / 'stage_commits' / (stage.replace(':','__')+'.json')
    return path, json.loads(path.read_text())


def check_publication(published, service, reg):
    return verify_service_publication(published, expected_policy_sha256=service.session.policy_sha256,
        expected_source_sha256=service.session._policy['source_sha256'],
        expected_registration_sha256=reg['registration_sha256'], trusted_public_keys=service.session.public_keys)


def test_real_service_subprocess_and_fresh_process_reentry(tmp_path):
    path, policy, reg = setup_service(tmp_path)
    first, a = cli(path, req(reg))
    assert first.returncode == 0, first.stdout + first.stderr
    assert a['result']['status'] == 'COMPLETE' and a['authoritative'] is False
    cp, before = chain_commit(policy, reg)
    original = cp.read_bytes()
    second, b = cli(path, req(reg, 'status'))
    assert second.returncode == 0, second.stdout + second.stderr
    assert b['result']['stage_execution_count'] == 1
    third, c = cli(path, req(reg, 'run'))
    assert third.returncode == 0 and c['result']['stage_execution_count'] == 1
    assert cp.read_bytes() == original
    assert before['result']['partition']['class_count'] == 3
    assert before['execution_receipt']['key_id'] == hashlib.sha256(bytes.fromhex(policy['public_key_hex'])).hexdigest()


def test_service_publish_through_existing_publication_module(tmp_path):
    path, policy, reg = setup_service(tmp_path)
    service = RegisteredExecutionService(path)
    service.dispatch(req(reg))
    pub = service.dispatch(req(reg, 'publish'))['result']
    assert check_publication(pub, service, reg)['status'] == 'VERIFIED'
    assert pub['record']['science_authority_effect'] == 'NONE'
    assert pub['record']['authoritative'] is False
    assert len(pub['record']['commits']) == 1
    assert service.dispatch(req(reg, 'publish'))['result'] == pub
    assert len(list((Path(policy['service_root'])/'registered_publications').glob('*.json'))) == 1
    assert not list((Path(policy['service_root'])/'registered_publications').glob('.registered-publication-*'))


def test_subprocess_publication_reentry_uses_same_service_key(tmp_path):
    path, policy, reg = setup_service(tmp_path)
    assert cli(path, req(reg))[0].returncode == 0
    first, a = cli(path, req(reg, 'publish'))
    second, b = cli(path, req(reg, 'publish'))
    assert first.returncode == second.returncode == 0
    assert a['result'] == b['result']


def test_one_four_workers_use_shared_runtime_and_equal_science(tmp_path):
    outcomes=[]
    for n in (1, 4):
        path, policy, reg = setup_service(tmp_path/str(n), workers=n)
        p, data = cli(path, req(reg))
        assert p.returncode == 0, p.stdout+p.stderr
        _, c = chain_commit(policy, reg)
        outcomes.append((c['result'], c['science_sha256']))
        assert list((Path(policy['service_root'])/'chains').rglob('partition.sqlite3'))
        summary_path = next((Path(policy['service_root'])/'chains').rglob('SUMMARY.json'))
        execution = json.loads(summary_path.read_text())['execution']
        from infinity_grid.execution import detect_effective_cpu_count
        assert execution['workers'] == min(n, detect_effective_cpu_count())
        assert execution['backend'] == ('SERIAL' if execution['workers'] == 1 else 'LOCAL_PROCESS_POOL')
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('extra', ['result', 'receipt', 'signature', 'key_id', 'public_key',
                                   'policy', 'handler_ref', 'source_sha256', 'complete'])
def test_request_carries_only_registration_identifier(tmp_path, extra):
    path, policy, reg = setup_service(tmp_path)
    service=RegisteredExecutionService(path)
    request=req(reg);request[extra]='not-a-request-field'
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_REQUEST_FIELDS'):
        service.dispatch(request)
    assert not list((Path(policy['service_root'])/'chains').rglob('chain_registration.json'))


@pytest.mark.parametrize('operation', ['sign','commit','register','mirror','set_status'])
def test_only_declared_service_operations(tmp_path, operation):
    path, policy, reg=setup_service(tmp_path)
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_OPERATION_NOT_REGISTERED'):
        RegisteredExecutionService(path).dispatch(req(reg, operation))


def test_unknown_registration_waits_before_chain_work(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    request=req(reg);request['registration_sha256']='f'*64
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_REGISTRATION_NOT_ADMITTED'):
        RegisteredExecutionService(path).dispatch(request)
    assert not list((Path(policy['service_root'])/'chains').rglob('partition.sqlite3'))


@pytest.mark.parametrize('operation', ['status','resume','publish'])
def test_existing_chain_is_required(tmp_path, operation):
    path, policy, reg=setup_service(tmp_path)
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_CHAIN_NOT_STARTED'):
        RegisteredExecutionService(path).dispatch(req(reg, operation))


def test_official_policy_awaits_acceptance_before_key_or_store(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    policy['role']=OFFICIAL_ROLE
    Path(policy['signing_key_path']).unlink()
    write_private(path,json.dumps(policy).encode())
    with pytest.raises(ExecutionAuthorityError, match='OFFICIAL_SERVICE_ACCEPTANCE_PENDING'):
        RegisteredExecutionService(path)
    assert not Path(policy['service_root']).exists()


def test_default_controller_official_entry_remains_inactive(tmp_path):
    c=ScientificChainController(tmp_path)
    with pytest.raises(ExecutionAuthorityError, match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        c.run(registration(hierarchy='G'))
    with pytest.raises(ExecutionAuthorityError, match='OFFICIAL_AUTHORITY_NOT_CONFIGURED'):
        c._authority.require_run({})


def test_engineering_policy_does_not_admit_g_registration(tmp_path):
    path, policy, reg=setup_service(tmp_path,regs=[registration(hierarchy='G')])
    with pytest.raises(ExecutionAuthorityError, match='ENGINEERING_MODE_CANNOT_RUN_SCIENCE'):
        RegisteredExecutionService(path)
    assert not Path(policy['service_root']).exists()


def test_policy_requires_exact_source_identity(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    policy['source_sha256']='f'*64;write_private(path,json.dumps(policy).encode())
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_SOURCE_POLICY_MISMATCH'):
        RegisteredExecutionService(path)


def test_policy_change_after_loading_is_visible(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    service=RegisteredExecutionService(path)
    write_private(path,path.read_bytes()+b'\n')
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_POLICY_CHANGED'):
        service.dispatch(req(reg))


def test_key_must_match_installed_public_key(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    write_private(Path(policy['signing_key_path']),Ed25519PrivateKey.generate().private_bytes_raw())
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_KEY_POLICY_MISMATCH'):
        RegisteredExecutionService(path)


def test_service_uses_installed_key_not_chain_local_key_list(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    service=RegisteredExecutionService(path);service.dispatch(req(reg))
    key_file=Path(policy['service_root'])/'chains'/reg['chain_id']/'ENGINEERING_PUBLIC_KEYS.json'
    assert not key_file.exists()
    key_file.write_text(json.dumps({'role':ENGINEERING_ROLE,'keys':{'0'*64:'00'*32}}))
    assert RegisteredExecutionService(path).dispatch(req(reg,'status'))['result']['status']=='COMPLETE'


def test_local_engineering_receipt_does_not_enter_service_chain(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    # Two ordinary fixture executions under different engineering receipt keys.
    service=RegisteredExecutionService(path);service.dispatch(req(reg))
    cp,c=chain_commit(policy,reg)
    from infinity_grid.v05_execution_authority import EngineeringAuthority
    other=EngineeringAuthority(True)
    c['execution_receipt']=other.receipt(c['execution_receipt']['payload']['bindings'],c['result'])
    c['commit_sha256']=canonical_sha256({k:v for k,v in c.items() if k!='commit_sha256'})
    cp.write_text(json.dumps(c))
    with pytest.raises(ExecutionAuthorityError, match='UNTRUSTED_RECEIPT_KEY'):
        RegisteredExecutionService(path).dispatch(req(reg,'status'))


def test_evaluator_inventory_applies_before_partition_work(tmp_path):
    path, policy, reg=setup_service(tmp_path,evaluators=['infinity_grid.controller_only_fixture:generation_evaluator'])
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_EVALUATOR_NOT_ADMITTED'):
        RegisteredExecutionService(path).dispatch(req(reg))
    assert not list((Path(policy['service_root'])/'chains').rglob('partition.sqlite3'))


def test_catalog_has_only_installed_engineering_evaluators(tmp_path):
    path, policy, reg=setup_service(tmp_path,evaluators=['infinity_grid.g6_controller_evaluators:exact_one_step_relation_evaluator'])
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_EVALUATOR_INVENTORY'):
        RegisteredExecutionService(path)


def test_catalog_handler_must_be_in_engineering_inventory(tmp_path):
    r=registration();r['stages'][0]['execution']['handler_key']='other'
    r=seal_chain_registration(r)
    path, policy, reg=setup_service(tmp_path,regs=[r])
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_HANDLER_INVENTORY'):
        RegisteredExecutionService(path)


def test_duplicate_catalog_waits_before_store_creation(tmp_path):
    r=registration()
    path, policy, reg=setup_service(tmp_path,regs=[r,r])
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_CATALOG_DUPLICATE'):
        RegisteredExecutionService(path)
    assert not Path(policy['service_root']).exists()


def test_registration_content_cannot_change_after_policy(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    session=RegisteredServiceSession(path)
    r=copy.deepcopy(reg);r['stages'][0]['execution']['parameters']['modulus']=7
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_REGISTRATION_CONTENT_MISMATCH'):
        session.require_registration(r)


def test_service_session_requires_its_owned_chain_root(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    session=RegisteredServiceSession(path)
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_CHAIN_ROOT_MISMATCH'):
        ScientificChainController(tmp_path/'other',engineering_only=True,_service_session=session)
    assert not (tmp_path/'other').exists()


def test_service_session_cannot_select_official_controller_mode(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    session=RegisteredServiceSession(path)
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_SESSION_MODE'):
        ScientificChainController(session.chain_root,_service_session=session)


def test_missing_mirror_keeps_publication_pending(tmp_path):
    path, policy, reg=setup_service(tmp_path,mirror=True)
    service=RegisteredExecutionService(path)
    state=service.dispatch(req(reg))['result']
    assert state['status']=='PAUSED' and state['external_mirror_pending']
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_PUBLICATION_AWAITS_COMPLETE_CHAIN'):
        service.dispatch(req(reg,'publish'))
    assert service.dispatch(req(reg,'resume'))['result']['status']=='PAUSED'


def test_two_stage_pause_resume_and_dependency_receipts(tmp_path):
    r=registration(); second=copy.deepcopy(r['stages'][0])
    second['stage_id']='ENG:S1';second['depends_on']=['ENG:S0'];second['auto_run']=False
    r['stages'][0]['transitions']['PASS']={'action':'NEXT','next_stage':'ENG:S1','reason':'NEXT_REGISTERED_STAGE'}
    r['stages'].append(second);r=seal_chain_registration(r)
    path, policy, reg=setup_service(tmp_path,regs=[r])
    first,a=cli(path,req(reg));assert first.returncode==0
    assert a['result']['status']=='PAUSED' and a['result']['current_stage']=='ENG:S1'
    second,b=cli(path,req(reg,'resume'));assert second.returncode==0,second.stdout+second.stderr
    assert b['result']['status']=='COMPLETE' and b['result']['stage_execution_count']==2
    service=RegisteredExecutionService(path)
    pub=service.dispatch(req(reg,'publish'))['result']
    assert len(pub['record']['commits'])==2
    _,c0=chain_commit(policy,reg);_,c1=chain_commit(policy,reg,'ENG:S1')
    assert c1['execution_receipt']['payload']['bindings']['dependencies_sha256']==digest([['ENG:S0',c0['commit_sha256']]])
    assert check_publication(pub,service,reg)['status']=='VERIFIED'


def test_service_root_lock_has_bounded_wait_result(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    service=RegisteredExecutionService(path)
    with service.session.request_slot():
        with pytest.raises(ExecutionAuthorityError, match='SERVICE_BUSY'):
            RegisteredExecutionService(path).dispatch(req(reg))
    assert service.dispatch(req(reg))['result']['status']=='COMPLETE'


@pytest.mark.parametrize('which', ['policy','key','root'])
def test_service_configuration_permissions(tmp_path, which):
    path, policy, reg=setup_service(tmp_path)
    if which=='policy':path.chmod(0o644)
    elif which=='key':Path(policy['signing_key_path']).chmod(0o644)
    else:Path(policy['service_root']).mkdir(mode=0o755)
    with pytest.raises(ExecutionAuthorityError, match='PERMISSIONS'):
        RegisteredExecutionService(path)


def test_standard_configuration_paths_required(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    link=tmp_path/'policy-link';link.symlink_to(path)
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_STANDARD_PATH_REQUIRED'):
        RegisteredExecutionService(link)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}',b'{"x":{"a":1,"a":2}}',b'{"x":NaN}',b'[]',b'{'])
def test_strict_json_is_unambiguous(raw):
    with pytest.raises(ExecutionAuthorityError):strict_json(raw)


def test_bounded_cli_request(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    request=req(reg);request['result']='x'*5000
    p,out=cli(path,request)
    assert p.returncode==4 and out['reason']=='SERVICE_REQUEST_SIZE'
    assert not Path(policy['service_root']).exists()


def test_publication_requires_registered_service(tmp_path):
    c=ScientificChainController(tmp_path,engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match='REJECT_EXTERNAL_EXECUTION_ORIGIN'):
        publish_registered_chain_execution(c,'L2-ENG')
    from infinity_grid.v05_origin_guard import _test_controller_event_scope
    with _test_controller_event_scope():
        with pytest.raises(ExecutionAuthorityError, match='REGISTERED_SERVICE_PUBLICATION_REQUIRED'):
            publish_registered_chain_execution(c,'L2-ENG')


@pytest.mark.parametrize('field', ['signature','record','key_id','role'])
def test_publication_changes_are_detected(tmp_path, field):
    path, policy, reg=setup_service(tmp_path);service=RegisteredExecutionService(path)
    service.dispatch(req(reg));pub=service.dispatch(req(reg,'publish'))['result']
    altered=copy.deepcopy(pub)
    if field=='signature':altered[field]=base64.b64encode(b'\x00'*64).decode()
    elif field=='record':
        altered['record']['commits'][0]['result']['partition']['class_count']=7
        altered['record_sha256']=digest(altered['record'])
    elif field=='key_id':altered[field]='f'*64
    else:altered[field]=OFFICIAL_ROLE
    with pytest.raises(ExecutionAuthorityError):check_publication(altered,service,reg)


def test_existing_publication_is_verified_not_overwritten(tmp_path):
    path, policy, reg=setup_service(tmp_path);service=RegisteredExecutionService(path)
    service.dispatch(req(reg));pub=service.dispatch(req(reg,'publish'))['result']
    p=Path(policy['service_root'])/'registered_publications'/(pub['record_sha256']+'.json')
    p.chmod(0o600);altered=copy.deepcopy(pub);altered['signature']=base64.b64encode(b'\x00'*64).decode()
    p.write_text(json.dumps(altered));before=p.read_bytes()
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_PUBLICATION_SIGNATURE'):
        service.dispatch(req(reg,'publish'))
    assert p.read_bytes()==before


def test_engineering_service_receipt_has_no_official_role(tmp_path):
    path, policy, reg=setup_service(tmp_path);service=RegisteredExecutionService(path)
    service.dispatch(req(reg));_,commit=chain_commit(policy,reg)
    receipt=commit['execution_receipt']
    with pytest.raises(ExecutionAuthorityError, match='RECEIPT_ROLE_NOT_AUTHORIZED'):
        verify_execution_receipt(receipt,expected_bindings=receipt['payload']['bindings'],
            result=commit['result'],trusted_public_keys=service.session.public_keys,required_role=OFFICIAL_ROLE)


def test_source_change_before_dispatch_waits(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    service=RegisteredExecutionService(path)
    alternate=tmp_path/'source-example';alternate.mkdir();(alternate/'example.py').write_text('VALUE = 1\n')
    service.session.package_root=alternate
    with pytest.raises(ExecutionAuthorityError, match='SERVICE_SOURCE_POLICY_MISMATCH'):
        service.dispatch(req(reg))
    assert not list((Path(policy['service_root'])/'chains').rglob('partition.sqlite3'))


def test_public_key_view_does_not_modify_installed_policy(tmp_path):
    path, policy, reg=setup_service(tmp_path)
    session=RegisteredServiceSession(path)
    keys=session.public_keys;keys.clear()
    assert len(session.public_keys)==1
