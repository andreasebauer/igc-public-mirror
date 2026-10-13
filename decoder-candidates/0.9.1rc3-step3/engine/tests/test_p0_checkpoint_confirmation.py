"""Completed/pending checkpoint retries must respect native save authority."""
import ast
from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1] / 'qualification/p0_ops'
sys.path.insert(0, str(OPS))
import p0_control as control


class NativeStatus:
    def __init__(self, pending=False):
        self.pending = pending; self.acks = 0
    def status(self, workspace):
        return {'status': 'SAVE_REQUIRED' if self.pending else 'PRESERVED',
                'pending_objects': [{'sha256':'bound'}] if self.pending else [],
                'pending_bytes': 1 if self.pending else 0,
                'pending_checkpoints': ['bound'] if self.pending else []}
    def confirm_batch(self, workspace, path):
        self.acks += 1
        if Path(path).read_bytes() != b'valid-bound-batch':
            raise RuntimeError('ACK_BATCH_OBLIGATION_BINDING')
        self.pending = False


def test_preserved_checkpoint_ignores_stale_batch_without_new_ack_or_receipt(tmp_path):
    native = NativeStatus(); batch = tmp_path / 'stale'; batch.write_bytes(b'invalid-stale-batch')
    output = tmp_path / 'verified.json'
    first = control.confirm_checkpoint(tmp_path, batch, output, native)
    before = output.read_bytes()
    second = control.confirm_checkpoint(tmp_path, batch, output, native)
    assert first == second and native.acks == 0 and output.read_bytes() == before
    assert first['action'] == 'REUSED_NATIVE_VERIFIED_PRESERVATION'


def test_pending_roles_require_native_valid_batch_and_cannot_be_skipped(tmp_path):
    native = NativeStatus(pending=True); batch = tmp_path / 'batch'; output = tmp_path / 'verified.json'
    batch.write_bytes(b'invalid-stale-batch')
    with pytest.raises(RuntimeError, match='OBLIGATION_BINDING'):
        control.confirm_checkpoint(tmp_path, batch, output, native)
    assert not output.exists() and native.pending
    batch.write_bytes(b'valid-bound-batch')
    result = control.confirm_checkpoint(tmp_path, batch, output, native)
    assert result['action'] == 'CONFIRMED_PENDING_ROLES' and native.acks == 2
    assert control.read(output)['status'] == 'PRESERVED'


def test_inconsistent_empty_native_status_does_not_publish_preservation(tmp_path):
    native = NativeStatus()
    native.status = lambda work: {'status':'PRESERVED','pending_objects':[], 'pending_bytes':1,'pending_checkpoints':[]}
    with pytest.raises(control.ControlError, match='NOT_EXACTLY_PRESERVED'):
        control.confirm_checkpoint(tmp_path, tmp_path/'batch', tmp_path/'verified.json', native)
    assert not (tmp_path/'verified.json').exists()


def test_real_worker_empty_drain_does_not_create_an_invalid_ack_batch(tmp_path):
    tree = ast.parse((OPS/'continuous_worker.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transport')
    control.write(tmp_path/'DRAIN_PLAN.json', {'capture_id':'frozen','physical_objects':[], 'obligations':[]})
    control.write(tmp_path/'READBACKS.json', {})
    def require(condition):
        if not condition:raise AssertionError()
    namespace = {'read':control.read,'write':control.write,'pathlib':Path.__module__ and __import__('pathlib'),
                 '_require_integrity':require}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-empty-drain>', 'exec'), namespace)
    namespace['transport'](tmp_path,'checkpoint',98765,0)
    assert not (tmp_path/'ACK_BATCH.json').exists()


def test_no_checkpoint_is_not_treated_as_preserved(tmp_path):
    native = NativeStatus()
    native.status = lambda work: {'status':'NO_CHECKPOINT','pending_objects':[], 'pending_bytes':0,'pending_checkpoints':[]}
    with pytest.raises(control.ControlError, match='NOT_EXACTLY_PRESERVED'):
        control.confirm_checkpoint(tmp_path,tmp_path/'batch',tmp_path/'verified.json',native)
    assert not (tmp_path/'verified.json').exists()


def test_restore_requires_new_native_snapshot_before_save_drain(tmp_path):
    class Restored:
        created = False
        snapshots = 0
        def status(self, work):
            return {'status':'SAVE_REQUIRED' if self.created else 'NO_CHECKPOINT',
                    'latest_checkpoint':'new-bound-native-checkpoint' if self.created else None}
        def snapshot_workspace(self, work, reason):
            self.created=True;self.snapshots+=1
        def drain_plan(self, work):
            return {'physical_objects':[{'sha256':'new-state'}], 'obligations':['bound-role']}
    service = Restored();work=tmp_path/'work';unit=tmp_path/'unit';unit.mkdir();(work/'durability').mkdir(parents=True)
    control.write(work/'durability/RESTORED_CHECKPOINT_PROVENANCE.json', {'registered_fixture':True})
    control.write(unit/'NATIVE_RESULT.json', {'status':'COMPLETED','evidence_status':'VERIFIED'})
    control.write(unit/'READBACKS.json',{})
    first=control.prepare_checkpoint(work,unit,service);second=control.prepare_checkpoint(work,unit,service)
    assert first==second and service.snapshots==1
    assert control.read(unit/'NEW_CHECKPOINT_OBJECTS.json')==[{'sha256':'new-state'}]
    assert control.read(unit/'PRESERVATION_STATUS.json')['status']=='SAVE_REQUIRED'
