"""Fault probes on temporary state and mock transport; no scientific credit."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

OPS = Path(__file__).resolve().parents[1] / 'qualification' / 'p0_ops'
sys.path.insert(0, str(OPS))
import p0_control as control


class Provider:
    def __init__(self):
        self.objects = {}; self.uploads = 0; self.fail_readback = False; self.lose_response = False
    def upload(self, path, name):
        self.uploads += 1; key = str(self.uploads)
        self.objects[key] = (name, Path(path).read_bytes())
        if self.lose_response:
            self.lose_response = False
            raise OSError('provider committed but response lost')
        return key
    def download(self, key):
        if self.fail_readback:
            raise OSError('readback unavailable')
        return self.objects[key][1]
    def reconcile(self, name):
        return [key for key, value in self.objects.items() if value[0] == name]
    def journal(self, path):
        return control.SaveJournal(path, self.upload, self.download, self.reconcile)


def test_save_restart_after_provider_id_before_readback_never_uploads_twice(tmp_path):
    provider = Provider(); payload = tmp_path / 'data'; payload.write_bytes(b'exact')
    provider.fail_readback = True
    with pytest.raises(OSError):
        provider.journal(tmp_path / 'journal.json').save(payload)
    provider.fail_readback = False
    receipt = provider.journal(tmp_path / 'journal.json').save(payload)
    assert receipt['drive_exact_readback_verified'] and provider.uploads == 1


def test_unknown_committed_upload_is_reconciled_and_not_repeated(tmp_path):
    provider = Provider(); payload = tmp_path / 'data'; payload.write_bytes(b'exact')
    provider.lose_response = True
    with pytest.raises(OSError):
        provider.journal(tmp_path / 'journal.json').save(payload)
    assert provider.journal(tmp_path / 'journal.json').save(payload)['drive_file_id'] == '1'
    assert provider.uploads == 1


def test_unknown_invisible_upload_is_blocked_without_retry(tmp_path):
    provider = Provider(); payload = tmp_path / 'data'; payload.write_bytes(b'exact')
    provider.lose_response = True
    with pytest.raises(OSError):
        provider.journal(tmp_path / 'journal.json').save(payload)
    provider.reconcile = lambda name: []
    with pytest.raises(control.ControlError, match='UNKNOWN_NO_RETRY'):
        provider.journal(tmp_path / 'journal.json').save(payload)
    assert provider.uploads == 1


@pytest.mark.parametrize('optimize', [0, 1, 2])
def test_corrupt_readback_is_refused_by_real_save_function_in_all_compiler_modes(tmp_path, optimize):
    tree = ast.parse((OPS / 'continuous_worker.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'saved_file')
    namespace = {'O': tmp_path, 'SaveJournal': control.SaveJournal,
                 'upload_object': lambda p, name: 'provider-id',
                 'download_object': lambda key: b'wrong', 'reconcile_object': lambda name: []}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-saved-file>', 'exec', optimize=optimize), namespace)
    payload = tmp_path / 'data'; payload.write_bytes(b'exact')
    with pytest.raises(control.ControlError, match='READBACK_MISMATCH'):
        namespace['saved_file'](payload)


@pytest.mark.parametrize('fault_at', ['after_commit', 'after_ledger', 'after_status'])
def test_activation_faults_recover_one_committed_checkpoint(tmp_path, fault_at):
    work = tmp_path / 'work'; batch = work / 'continuation0290'; batch.mkdir(parents=True)
    control.write(work / 'CURRENT_STATUS.json', {'latest_checkpoint': 289})
    receipt = batch / 'verified-receipt'; receipt.write_bytes(b'verified')
    ledger = {'checkpoint': 290, 'status': 'EARNED_NATIVE_COLD_AND_EXACT_SAVED',
              'save_receipts': [{'path': str(receipt), 'sha256': control.sha(receipt)}]}
    status = {'latest_checkpoint': 290, 'official_observer_next_chunk': 12}
    def fault(phase):
        if phase == fault_at:
            raise OSError('injected loss')
    with pytest.raises(OSError):
        control.publish_activation(work, batch, ledger, status, fault=fault)
    control.recover_next_activation(work)
    assert control.read(batch / 'EARNED_LEDGER.json') == ledger
    assert control.read(work / 'CURRENT_STATUS.json') == status
    before = (batch / 'ACTIVATION_COMMIT.json').read_bytes()
    control.recover_next_activation(work)
    assert (batch / 'ACTIVATION_COMMIT.json').read_bytes() == before
    # A later in-flight annotation must not be overwritten by replaying an
    # already applied old activation.
    annotated = dict(status, observer_batch_in_flight={'checkpoint': 291})
    control.write(work / 'CURRENT_STATUS.json', annotated)
    control.recover_next_activation(work)
    assert control.read(work / 'CURRENT_STATUS.json') == annotated


def test_completed_stage_is_reused_and_input_or_output_change_is_refused(tmp_path):
    output = tmp_path / 'result'; calls = []
    def action():
        calls.append(1); output.write_bytes(b'earned')
    journal = tmp_path / 'stages.json'; binding = {'spec': 'fixed', 'operator': 'fixed'}
    assert control.stage(journal, 'native', action, [output], binding)
    assert not control.stage(journal, 'native', action, [output], binding)
    assert calls == [1]
    with pytest.raises(control.ControlError, match='SOURCE_OR_INPUT'):
        control.stage(journal, 'native', action, [output], {'spec': 'different'})
    output.write_bytes(b'altered')
    with pytest.raises(control.ControlError, match='OUTPUT_CHANGED'):
        control.stage(journal, 'native', action, [output], binding)
    assert calls == [1]


@pytest.mark.parametrize('optimize', [0, 1, 2])
def test_real_reader_rejects_invalid_append_only_scope_without_opening_seal(tmp_path, optimize):
    raw = (OPS / 'exact_parent_reader.py').read_text()
    namespace = {}; exec(compile(raw, '<real-reader>', 'exec', optimize=optimize), namespace)
    namespace['read_bound'] = lambda path, digest: {'release_id': 'WRONG', 'slices': []}
    namespace['ExactParentReader'] = lambda *args: pytest.fail('Invalid admission opened seal')
    names = ('previous_catalog', 'candidate_catalog', 'scoped_admission', 'proposal', 'admission_gate', 'seal')
    with pytest.raises(AssertionError):
        namespace['qualify']({n: n for n in names}, {n: n for n in names})


def test_transformed_template_preserves_lazy_message_and_future_import_under_optimization():
    transformed = control.explicit_assertions('"""template"""\nfrom __future__ import annotations\ncount=[]\nassert True, count.append(1)\nassert False, "refused"\n')
    namespace = {}
    with pytest.raises(AssertionError, match='refused'):
        exec(compile(transformed, '<template>', 'exec', optimize=2), namespace)
    assert namespace['count'] == []


def test_optimized_worker_is_refused_before_client_import_or_external_actions():
    done = subprocess.run([sys.executable, '-B', '-O', str(OPS / 'continuous_worker.py')], capture_output=True, text=True)
    assert done.returncode != 0 and 'OPTIMIZED_CONTROLLER_RUNTIME_REFUSED' in done.stderr


def test_atomic_failed_replace_leaves_old_valid_json(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'; control.write(path, {'checkpoint': 289})
    def fail(*args):
        raise OSError('rename interrupted')
    monkeypatch.setattr(control.os, 'replace', fail)
    with pytest.raises(OSError):
        control.write(path, {'checkpoint': 290})
    assert control.read(path) == {'checkpoint': 289}


def observer_setup_fixture(tmp_path, monkeypatch):
    work = tmp_path / 'work'; output = tmp_path / 'output'
    q = work / 'continuation0285'; templates = work / 'continuation0288'
    (q / 'specs').mkdir(parents=True); (templates / '003').mkdir(parents=True); output.mkdir()
    control.write(q / 'PREREGISTRATION.json', {})
    for i in range(9, 12):
        control.write(q / 'specs' / f'{i:03d}.json', {'chunk': i})
    capture = "from pathlib import Path\nimport hashlib\nsha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()\nstore=Path(f'/tmp/ig_native0288_{idx:03d}/store');assert not (store/'coordination/PROJECT.json').exists()\nportable_registry.initialize(store,f'Official historical observer chunk{idx:03d}',E)\n"
    (templates / 'capture.py').write_text(capture)
    (templates / 'audit_cold.py').write_text('assert True\n')
    for name in ['ack.py','run_native.py','prepare_drain.py','confirm_checkpoint.py','export_checkpoint.py']:
        (templates / '003' / name).write_text('pass\n')
    prior = work / 'continuation0500'; prior.mkdir()
    control.write(prior / 'READBACKS.json', {'old': {'id': 'saved'}})
    control.write(prior / 'EARNED_LEDGER.json', {'status': 'EARNED_NATIVE_COLD_AND_EXACT_SAVED', 'next_chunk_index': 9, 'save_receipts': [], 'completed_chunks': [{'unit_directory': str(prior)}]})
    control.write(work / 'CURRENT_STATUS.json', {'latest_checkpoint': 500, 'official_observer_next_chunk': 9})
    source = (OPS / 'continuous_observer.py').read_text()
    tree = ast.parse(source)
    # Use the real CLI body, changing only its workspace constants.
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('W', 'O'):
                node.value = ast.Call(func=ast.Attribute(value=ast.Name(id='pathlib',ctx=ast.Load()),attr='Path',ctx=ast.Load()), args=[ast.Constant(str(work if node.targets[0].id == 'W' else output))],keywords=[])
    ast.fix_missing_locations(tree)
    monkeypatch.setattr(sys, 'argv', ['continuous_observer.py', 'setup'])
    def setup():
        exec(compile(tree, '<real-observer-setup>', 'exec'), {'__file__': str(OPS / 'continuous_observer.py')})
    return work, templates, setup


def test_real_setup_resumes_partial_directory_and_preserves_existing_readbacks(tmp_path, monkeypatch):
    work, templates, setup = observer_setup_fixture(tmp_path, monkeypatch)
    batch = work / 'continuation0501'; (batch / '009').mkdir(parents=True)
    setup()
    readbacks = batch / '009' / 'READBACKS.json'
    control.write(readbacks, {'earned_after_first_setup': {'id': 'preserved'}})
    capture = (batch / 'capture.py').read_bytes()
    setup()
    assert control.read(batch / 'BATCH_PLAN.json')['scope'] == [9, 10, 11]
    assert control.read(readbacks) == {'earned_after_first_setup': {'id': 'preserved'}}
    assert (batch / 'capture.py').read_bytes() == capture
    assert b'CAPTURE_PROJECT_SOURCE_CHANGED' in capture
    assert b'assert not' not in capture
    (templates / '003' / 'ack.py').write_text('changed\n')
    with pytest.raises(control.ControlError, match='RESUME_FILE_IDENTITY'):
        setup()


def test_real_setup_refuses_unrecognized_capture_template(tmp_path, monkeypatch):
    work, templates, setup = observer_setup_fixture(tmp_path, monkeypatch)
    (templates / 'capture.py').write_text('pass\n')
    with pytest.raises(AssertionError, match='CAPTURE_TEMPLATE_CONTRACT_CHANGED'):
        setup()
    assert control.read(work / 'CURRENT_STATUS.json')['latest_checkpoint'] == 500


def test_capture_restart_after_publication_reuses_same_native_capture(tmp_path):
    from infinity_grid import submission, portable_registry
    from tests.test_decoder06_capture import _spec
    spec = _spec(tmp_path, project=False)
    store = tmp_path / 'store'
    portable_registry.initialize(store, 'P0 ENGINEERING CAPTURE ONLY', spec['engine_source'])
    pointer = tmp_path / 'pointer.json'; journal = tmp_path / 'stages.json'; captures = []
    def first():
        result = submission.capture(store, spec)
        captures.append(result['capture_id'])
        control.write(pointer, result)
        raise OSError('after capture publication before stage completion')
    with pytest.raises(OSError):
        control.stage(journal, 'capture', first, [pointer], {'spec': submission._sha(submission._json_bytes(spec))})
    def resumed():
        result = submission.capture(store, spec)
        captures.append(result['capture_id'])
        control.write(pointer, result)
    control.stage(journal, 'capture', resumed, [pointer], {'spec': submission._sha(submission._json_bytes(spec))})
    assert captures[0] == captures[1]
    assert len(list((store / 'captures').iterdir())) == 1
    assert control.read(journal)['stages']['capture']['phase'] == 'COMPLETED'
