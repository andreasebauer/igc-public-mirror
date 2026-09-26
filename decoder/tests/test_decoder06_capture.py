"""Bounded Step2 checks, run only by the saved Decoder validation registration.

Temporary project files exercise preparation/refusals; no unsaved fixture body
is executed. The positive script execution is a separate real Drive-saved job.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tomllib

import pytest
from infinity_grid import __version__, build_meta
from infinity_grid import resources
from infinity_grid import submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.script_runtime import _FrozenProjectFinder


def _engine():
    return Path(loop.__file__).resolve().parents[1]


def _spec(tmp_path, *, project=True):
    folder = tmp_path / 'working_project'; folder.mkdir(exist_ok=True)
    if project:
        (folder / 'main.py').write_text('from helper import value\nprint(value)\n')
        (folder / 'helper.py').write_text('value = 7\n')
    return {'schema_id': sub.SPEC_SCHEMA, 'job_id': 'STEP2.PREPARATION.FIXTURE',
            'engine_source': str(_engine()), 'project_source': str(folder) if project else None,
            'question': {'stage_id': 'DECODER:STEP2', 'description': 'Captured fixture only',
                         'outcomes': ['PROCESS_COMPLETED', 'PROCESS_FAILED'], 'stopping_rule': 'One fixture; no scientific claim'},
            'execution': {'kind': 'SCRIPT', 'entrypoint': 'project/main.py', 'argv': [], 'parameters': {'value': 7}} if project else
                         {'kind': 'VALIDATION', 'nodes': ['tests/test_decoder06_capture.py::test_active_version_has_one_source']},
            'resources': {'workers': 1, 'start_method': 'fork', 'memory_budget_bytes': 268435456,
                          'workspace_budget_bytes': 268435456, 'wall_seconds_max': 30},
            'inputs': [], 'environment': {'python': f'{sys.version_info.major}.{sys.version_info.minor}', 'requirements': [], 'artifacts': []},
            'output_contract': {'process': 'exit code recorded', 'stdout': 'one integer', 'scientific_acceptance': 'NONE'}}


def test_active_version_has_one_source():
    config = tomllib.loads((_engine() / 'pyproject.toml').read_text())
    import ast
    version_tree = ast.parse((_engine() / 'infinity_grid/_version.py').read_text())
    canonical = next(n.value.value for n in version_tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    assert __version__ == canonical
    from packaging.version import Version
    identity = Version(canonical)
    assert identity.release == (0, 8, 0) and identity.local == "lib"
    assert 'version' not in config['project'] and 'version' in config['project']['dynamic']
    assert config['tool']['setuptools']['dynamic']['version']['attr'] == 'infinity_grid._version.__version__'
    assert build_meta()['version'] == resources.build_meta()['version'] == __version__
    raw = json.loads((_engine() / 'infinity_grid/_build_meta.json').read_text())
    assert 'version' not in raw and raw['version_source'] == 'infinity_grid._version.__version__'
    old = json.loads((_engine() / 'infinity_grid/resources/history/BUILD_META_BEFORE_06.json').read_text())
    assert old['version'] == '0.30.88.dev6'


def test_unchanged_capture_is_reused_and_helper_change_is_new(tmp_path):
    spec = _spec(tmp_path); store = tmp_path / 'store'
    first = sub.capture(store, spec); again = sub.capture(store, spec)
    assert first['capture_id'] == again['capture_id']
    p = Path(spec['project_source']) / 'helper.py'; p.write_text('value = 8\n')
    changed = sub.capture(store, spec)
    assert first['capture_id'] != changed['capture_id']
    a = sub.capture_record(first['workspace']); b = sub.capture_record(changed['workspace'])
    assert a['job']['registration_sha256'] != b['job']['registration_sha256']
    assert a['workspace']['source_sha256'] != b['workspace']['source_sha256']
    assert a['objects'][0] == b['objects'][0]
    assert (Path(first['workspace']) / 'source/project/helper.py').read_text() == 'value = 7\n'


def test_unsaved_or_failed_save_cannot_start_a_job(tmp_path):
    state = sub.capture(tmp_path / 'store', _spec(tmp_path, project=False)); root = Path(state['workspace'])
    with pytest.raises(sub.SubmissionError, match='SAVE_REQUIRED'):
        loop.run_workspace_job(root, state['job_id'])
    digest = state['pending_objects'][0]['sha256']
    failed = sub.record_save_failure(root, digest, 'Fixture transport unavailable')
    assert failed['status'] == 'SAVE_REQUIRED'
    with pytest.raises(sub.SubmissionError, match='SAVE_REQUIRED'):
        loop.run_workspace_job(root, state['job_id'])
    assert not (root / 'runtime/attempts').exists()
    assert not (root / 'runtime/runs').exists()
    assert list((root / 'durability/events').glob('*.json'))


def test_wrong_readback_is_preserved_and_cannot_confirm(tmp_path):
    state = sub.capture(tmp_path / 'store', _spec(tmp_path)); root = Path(state['workspace'])
    wrong = tmp_path / 'wrong.bin'; wrong.write_bytes(b'not the saved object')
    with pytest.raises(sub.SubmissionError, match='SAVE_READBACK_MISMATCH'):
        sub.confirm_save(root, state['pending_objects'][0]['sha256'], wrong, 'fixture_failure_only')
    assert sub.save_status(root)['status'] == 'SAVE_REQUIRED'
    assert not list((root / 'durability/receipts').rglob('*.json'))
    assert list((root / 'durability/events').glob('*.json'))


def test_capture_objects_restore_after_working_project_deletion(tmp_path):
    spec = _spec(tmp_path); store = tmp_path / 'store'; state = sub.capture(store, spec)
    root = Path(state['workspace']); before = (root / 'source/project/helper.py').read_bytes()
    record = tmp_path / 'capture.json'; shutil.copy2(root / 'CAPTURE.json', record)
    shutil.rmtree(spec['project_source']); shutil.rmtree(root)
    restored = sub.restore_capture(record, store / 'objects', tmp_path / 'restored')
    assert (Path(restored['workspace']) / 'source/project/helper.py').read_bytes() == before
    assert restored['status'] == 'SAVE_REQUIRED'  # Restore does not invent a Drive attestation.


def test_unresolved_helper_and_generated_code_are_actionable(tmp_path):
    spec = _spec(tmp_path); main = Path(spec['project_source']) / 'main.py'
    main.write_text('import missing_generated_helper\n')
    with pytest.raises(sub.SubmissionError, match='UNRESOLVED_EXECUTABLE_DEPENDENCY') as caught:
        sub.capture(tmp_path / 'store', spec)
    assert 'helper' in caught.value.next_action
    main.write_text('exec("print(1)")\n')
    with pytest.raises(sub.SubmissionError, match='GENERATED_CODE_REQUIRES_CAPTURE') as caught:
        sub.capture(tmp_path / 'store', spec)
    assert 'saved file' in caught.value.next_action
    assert len(list((tmp_path / 'store/events').glob('*.json'))) == 2


def test_input_hash_and_symlink_mismatches_are_refused(tmp_path):
    spec = _spec(tmp_path); data = tmp_path / 'input.json'; data.write_text('{}')
    spec['inputs'] = [{'logical_name': 'data', 'path': str(data), 'sha256': '0' * 64}]
    with pytest.raises(sub.SubmissionError, match='INPUT_HASH_MISMATCH'):
        sub.capture(tmp_path / 'store', spec)
    spec['inputs'] = []
    (Path(spec['project_source']) / 'outside.py').symlink_to(data)
    with pytest.raises(sub.SubmissionError, match='CAPTURE_SYMLINK'):
        sub.capture(tmp_path / 'store', spec)


def test_contract_and_environment_are_bound_to_revision(tmp_path):
    spec = _spec(tmp_path); store = tmp_path / 'store'; first = sub.capture(store, spec)
    spec['output_contract']['stdout'] = 'one labelled integer'
    second = sub.capture(store, spec)
    assert first['capture_id'] != second['capture_id']
    a, b = sub.capture_record(first['workspace']), sub.capture_record(second['workspace'])
    assert a['objects'][:2] == b['objects'][:2]
    assert a['job']['registration_sha256'] != b['job']['registration_sha256']
    assert a['output_contract'] != b['output_contract']
    assert a['environment'] == b['environment']


def test_project_import_guard_checks_helper_bytes(tmp_path):
    helper = tmp_path / 'helper.py'; helper.write_text('value = 7\n')
    finder = _FrozenProjectFinder(tmp_path)
    assert finder.find_spec('helper', [str(tmp_path)]) is None
    helper.write_text('value = 8\n')
    with pytest.raises(ImportError, match='UNREGISTERED_PROJECT_HELPER'):
        finder.find_spec('helper', [str(tmp_path)])


def test_current_validation_job_has_real_saved_source_records():
    root = _engine().parent
    state = sub.save_status(root)
    assert state['status'] == 'SAVED' and state['pending_objects'] == []
    rec = sub.require_saved(root, sub.capture_record(root)['job']['job_id'])
    assert rec['decoder_version'] == __version__
    assert len(sub.required_objects(root)) >= 3


def test_capture_metadata_change_cannot_rebind_frozen_job(tmp_path):
    state = sub.capture(tmp_path / 'store', _spec(tmp_path)); root = Path(state['workspace'])
    saved = sub.capture_record(root); saved['output_contract']['tampered'] = True
    (root / 'CAPTURE.json').write_text(json.dumps(saved))
    with pytest.raises(sub.SubmissionError, match='CAPTURE_RECORD_MISMATCH'):
        sub.save_status(root)


def test_rejected_capture_preserves_original_script_and_intent(tmp_path):
    import zipfile
    spec = _spec(tmp_path); project = Path(spec['project_source'])
    original = b'exec("print(9)")\n'
    (project/'main.py').write_bytes(original)
    store = tmp_path/'store'
    with pytest.raises(sub.SubmissionError, match='GENERATED_CODE_REQUIRES_CAPTURE') as caught:
        sub.capture(store, spec)
    shutil.rmtree(project)
    evidence = caught.value.preservation
    assert evidence['execution_authorized'] is False
    rows = {o['role']:o for o in evidence['objects']}
    with zipfile.ZipFile(store/'objects'/rows['project_source']['object_name']) as z:
        assert z.read('main.py') == original
    saved_spec = json.loads((store/'objects'/rows['submitted_specification']['object_name']).read_text())
    assert saved_spec == spec
    events = [json.loads(p.read_text()) for p in (store/'events').glob('*.json')]
    assert any(e.get('preservation') == evidence for e in events)
    assert not (store/'captures').exists()
