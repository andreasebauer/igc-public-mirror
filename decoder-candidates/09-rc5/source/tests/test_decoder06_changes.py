"""Focused recorded-change checks. Execute only through saved Decoder validation.

Preparation fixtures use an actual saved Step2 completion, never fabricated
positive save or test receipts. The real failure/repair/activation cycle is a
separate registered job, saved before execution.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest
from infinity_grid import __version__
from infinity_grid import change_sessions as changes
from infinity_grid import submission as sub
from infinity_grid import v05_controller_event_loop as loop


def _engine():
    return Path(loop.__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def parent(tmp_path_factory):
    root = _engine().parent
    record = sub.capture_record(root)
    row = next(r for r in record['job']['input_artifacts'] if r['logical_name'] == 'step2_parent_snapshot')
    archive = root/'runtime/intake/artifacts'/(row['sha256']+'.bin')
    dest = tmp_path_factory.mktemp('saved-step2')/'restored'
    loop.restore_workspace(archive, dest, row['sha256'])
    return dest


def _open(tmp_path, parent, cid='FIXTURE.CHANGE'):
    req = {'schema_id': 'IG_DECODER_CHANGE_OPEN_V1', 'change_id': cid,
           'parent_workspace': str(parent), 'purpose': 'Preparation fixture, no candidate body executed',
           'requirements': {'statement': 'Existing source check remains required',
                            'nodes': ['tests/test_decoder06_capture.py::test_active_version_has_one_source']},
           'environment': {'python': f'{sys.version_info.major}.{sys.version_info.minor}', 'requirements': [], 'artifacts': []},
           'resources': {'workers': 1, 'start_method': 'fork', 'wall_seconds_max': 30,
                         'memory_budget_bytes': 268435456, 'workspace_budget_bytes': 268435456}}
    store = tmp_path/'store'
    state = changes.open_change(store, req)
    return store, req, state


def test_open_binds_verified_parent_without_importing_it(tmp_path, parent):
    store, req, state = _open(tmp_path, parent)
    assert state['session']['parent']['version'] == '0.6.0.dev2'
    assert state['session']['parent']['source_sha256'] == '37253ad418261f712b4ff84b552cff26da9e45f77312aa87f6481102584b8c2a'
    assert state['session']['purpose'] == req['purpose']
    with pytest.raises(sub.SubmissionError, match='CHANGE_ALREADY_EXISTS'):
        changes.open_change(store, req)


def test_deep_controller_edit_captured_and_unchanged_revision_reused(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    candidate = tmp_path/'candidate'; shutil.copytree(parent/'source', candidate)
    control = candidate/'infinity_grid/v05_controller_event_loop.py'
    control.write_text('def broken(:\n')
    first = changes.capture_revision(store, req['change_id'], candidate)
    again = changes.capture_revision(store, req['change_id'], candidate)
    assert first['revision_id'] == again['revision_id']
    assert again['reused_revision'] is True
    assert first['capture']['status'] == 'SAVE_REQUIRED'
    diff = changes.diff_revision(store, req['change_id'], first['revision_id'])
    assert diff['changed_files'][0]['path'] == 'infinity_grid/v05_controller_event_loop.py'
    assert '+def broken(:' in diff['unified_diff']
    assert not list((Path(first['capture']['workspace'])/'runtime/attempts').glob('*'))


def test_requirement_change_needs_reason_and_retains_both_forms(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    first = changes.capture_revision(store, req['change_id'], parent/'source')
    requirement = dict(req['requirements'], statement='Corrected interpretation of the same check')
    with pytest.raises(sub.SubmissionError, match='REQUIREMENT_CHANGE_REASON_REQUIRED'):
        changes.capture_revision(store, req['change_id'], parent/'source', requirement)
    second = changes.capture_revision(store, req['change_id'], parent/'source', requirement, 'Clarify scope without changing expected outcome')
    assert first['revision_id'] != second['revision_id']
    revision = changes.status(store, req['change_id'])['revisions']
    assert len(revision) == 2
    changed = next(r for r in revision if r['record_sha256'] == second['revision_id'])
    assert changed['requirements_change']['before'] == req['requirements']
    assert changed['requirements_change']['after'] == requirement


def test_changed_test_bytes_need_reason(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    candidate = tmp_path/'candidate'; shutil.copytree(parent/'source', candidate)
    path = candidate/'tests/test_decoder06_capture.py'; path.write_text(path.read_text()+'\n# Fixture check clarification\n')
    with pytest.raises(sub.SubmissionError, match='TEST_CODE_CHANGE_REASON_REQUIRED'):
        changes.capture_revision(store, req['change_id'], candidate)
    state = changes.capture_revision(store, req['change_id'], candidate, reason='Added a documented fixture note; assertions unchanged')
    record = changes.status(store, req['change_id'])['revisions'][0]
    assert record['test_code_changes'][0]['path'] == 'tests/test_decoder06_capture.py'
    assert record['test_code_change_reason']
    assert state['status'] == 'SAVE_REQUIRED'


def test_candidate_status_flag_cannot_prepare_activation(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    candidate = tmp_path/'candidate'; shutil.copytree(parent/'source', candidate)
    (candidate/'ACCEPTED.json').write_text('{"status":"PASS","accepted":true}')
    result = changes.capture_revision(store, req['change_id'], candidate)
    with pytest.raises(sub.SubmissionError, match='CHANGE_TEST_RECEIPT_REQUIRED'):
        changes.prepare_activation(store, req['change_id'], result['revision_id'])
    assert changes.status(store, req['change_id'])['current']['epoch'] == 0


def test_unsaved_change_test_refuses_and_keeps_event(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    capture = changes.capture_revision(store, req['change_id'], parent/'source')
    with pytest.raises(sub.SubmissionError, match='SAVE_REQUIRED'):
        changes.test_revision(store, req['change_id'], capture['revision_id'])
    assert any(json.loads(p.read_text())['operation'] == 'TEST_PAUSED' for p in (store/'change_events').glob('*.json'))
    assert not (Path(capture['capture']['workspace'])/'runtime/attempts').exists()


def test_modified_revision_and_wrong_save_bytes_are_rejected(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    captured = changes.capture_revision(store, req['change_id'], parent/'source')
    folder = store/'changes'/req['change_id']; rid = captured['revision_id']
    path = folder/'revisions'/(rid+'.json'); record = json.loads(path.read_text())
    record['requirements']['statement'] = 'Changed silently'
    path.write_text(json.dumps(record))
    with pytest.raises(sub.SubmissionError, match='CHANGE_RECORD_MISMATCH'):
        changes.diff_revision(store, req['change_id'], rid)
    obj = next((store/'objects').glob('*.zip')); wrong = tmp_path/'wrong'; wrong.write_bytes(b'wrong')
    with pytest.raises(sub.SubmissionError, match='SAVE_READBACK_MISMATCH'):
        changes.confirm_object_save(store, obj.name, wrong, 'fixture_negative_only')


def test_rollback_requires_exact_current_and_previous(tmp_path, parent):
    store, req, state = _open(tmp_path, parent)
    pointer = changes.status(store, req['change_id'])['current']['record_sha256']
    with pytest.raises(sub.SubmissionError, match='ROLLBACK_CURRENT_MOVED'):
        changes.rollback(store, '0'*64, 'stale fixture')
    with pytest.raises(sub.SubmissionError, match='ROLLBACK_NO_PREVIOUS_VERSION'):
        changes.rollback(store, pointer, 'initial parent fixture')


def test_recovery_offers_engineering_only():
    result = subprocess.run([sys.executable, '-m', 'infinity_grid.recovery', '--help'], capture_output=True, text=True, check=True)
    assert 'capture-revision' in result.stdout and 'prepare-activation' in result.stdout
    assert 'engineering validation only' in result.stdout
    refused = subprocess.run([sys.executable, '-m', 'infinity_grid.recovery', 'run'], capture_output=True, text=True)
    assert refused.returncode == 2 and 'invalid choice' in refused.stderr


def test_built_and_installed_version_matches_runtime_and_cli(tmp_path):
    source = tmp_path/'build_source'; shutil.copytree(_engine(), source)
    wheelhouse = tmp_path/'wheels'; wheelhouse.mkdir()
    # Backend invocation is part of this saved Decoder validation, not a shell wrapper.
    import setuptools.build_meta
    old = Path.cwd()
    try:
        os.chdir(source)
        wheel = setuptools.build_meta.build_wheel(str(wheelhouse))
    finally:
        os.chdir(old)
    prefix = tmp_path/'installed'; prefix.mkdir()
    with zipfile.ZipFile(wheelhouse/wheel) as z:
        metadata = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        assert 'Version: '+__version__+'\n' in metadata
    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                                '--no-compile', '--target', str(prefix), str(wheelhouse/wheel)],
                               capture_output=True, text=True, check=True)
    assert 'Successfully installed infinity-grid-algebra-decoder-'+__version__ in installed.stdout
    env = dict(os.environ, PYTHONPATH=str(prefix), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-c', 'import importlib.metadata, infinity_grid; print(importlib.metadata.version("infinity-grid-algebra-decoder")); print(infinity_grid.__version__); print(infinity_grid.build_meta()["version"])']
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [__version__]*3
    cli = subprocess.run([sys.executable, '-m', 'infinity_grid.v05_controller_event_loop', '--version'], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert cli.stdout.strip() == changes.source_version(_engine()) == __version__
    entry = subprocess.run([str(prefix/'bin/ig-decoder'), '--version'], cwd=tmp_path, env=env,
                           capture_output=True, text=True, check=True)
    recover = subprocess.run([str(prefix/'bin/ig-decoder-recover'), '--version'], cwd=tmp_path, env=env,
                             capture_output=True, text=True, check=True)
    assert entry.stdout.strip() == recover.stdout.strip() == __version__


def test_receipt_fields_cannot_overwrite_event_identity(tmp_path):
    receipt = {'schema_id': 'IG_DECODER_CHANGE_TEST_RECEIPT_V1', 'record_sha256': 'original-receipt-hash', 'outcome': 'FAIL'}
    event = changes._event(tmp_path, 'TEST_FINISHED', receipt)
    path = tmp_path/'change_events'/(event['record_sha256']+'.json')
    assert changes._verified(path) == event
    assert event['schema_id'] == 'IG_DECODER_CHANGE_EVENT_V1'
    assert event['details'] == receipt and event['operation'] == 'TEST_FINISHED'


def test_fabricated_pass_receipt_requires_real_controller_completion(tmp_path, parent):
    store, req, _ = _open(tmp_path, parent)
    captured = changes.capture_revision(store, req['change_id'], parent/'source')
    rid = captured['revision_id']; folder = store/'changes'/req['change_id']
    receipt = changes._sealed({'revision_id': rid, 'outcome': 'PASS', 'completion_sha256': '0'*64})
    changes._immutable(folder/'test_receipts'/(rid+'.json'), receipt)
    with pytest.raises(sub.SubmissionError, match='SAVE_REQUIRED'):
        changes.prepare_activation(store, req['change_id'], rid)
    assert changes.status(store, req['change_id'])['current']['epoch'] == 0


def test_validation_runner_project_does_not_change_parent_release(tmp_path, parent):
    from infinity_grid import portable_registry as project
    store, req, _ = _open(tmp_path, parent)
    before = project.current(store/'coordination', 'release')
    captured = changes.capture_revision(store, req['change_id'], parent/'source')
    workspace = Path(captured['capture']['workspace'])
    runner_project, _ = project.locate(workspace)
    _, runner_release = project.current(runner_project, 'release')
    assert runner_release['engine'] == sub._sha(project.engine_object(_engine()))
    assert runner_release['engine'] != before[1]['engine']
    assert project.current(store/'coordination', 'release') == before
    assert sub.save_status(workspace)['status'] == 'SAVE_REQUIRED'
    # The registered validation retains the exact historical parent and candidate.
    names = {r['logical_name'] for r in sub.capture_record(workspace)['job']['input_artifacts']}
    assert {'candidate_source', 'change_session', 'change_revision', 'change_parent_source'} <= names
