"""Preregistered completed-unit recovery probe; no scientific dispatch allowed."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PR = json.loads((HERE / 'PREREGISTRATION_REVISION2.json').read_text())
sys.path.insert(0, str(BASE / 'p0_candidate/source'))
sys.path.insert(0, str(BASE / 'p0_candidate/operational'))
import p0_control as control
from infinity_grid import preservation as preservation
from infinity_grid import submission
from infinity_grid import v05_controller_event_loop as loop

ROOT = Path(PR['disposable_root'])
RESULT = HERE / 'results/NATIVE_MATRIX.json'
rows = []


def record(name, **fields):
    rows.append({'operation': name, 'status': 'PASS', **fields})
    control.write(RESULT, {'status': 'RUNNING', 'rows': rows})


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def receipt_inventory(work):
    return {p.relative_to(work).as_posix(): control.sha(p) for directory in ('durability/receipts', 'durability/outbox/receipts')
            for p in (work / directory).rglob('*') if p.is_file()}


def no_dispatch(*args, **kwargs):
    raise RuntimeError('P0_PROBE_NEW_SCIENTIFIC_DISPATCH_REFUSED')


def main():
    if ROOT.exists():
        raise RuntimeError('DISPOSABLE_ROOT_EXISTS_RECONCILE_BEFORE_RETRY')
    ROOT.mkdir()
    objects = ROOT / 'objects'; objects.mkdir()
    for item in PR['dependencies']:
        source = Path(item['path']); target = objects / (item['sha256'] + '.bin')
        shutil.copyfile(source, target)
        require(control.sha(target) == item['sha256'], 'COPIED_DEPENDENCY_HASH')
    archive = Path(PR['saved_unit']) / 'NATIVE_CHECKPOINT_SLIM.zip'
    work = ROOT / 'restored'
    restored = preservation.restore_checkpoint(archive, work, PR['archive_sha256'], objects=objects)
    require(restored['checkpoint_sha256'] == PR['checkpoint_sha256'], 'RESTORED_CHECKPOINT_IDENTITY')
    require(submission.capture_record(work)['capture_id'] == PR['capture_id'], 'RESTORED_CAPTURE_IDENTITY')
    require(not preservation.status(work)['pending_objects'], 'RESTORED_PENDING_SAVES')
    record('EXACT_NATIVE_CHECKPOINT_RESTORE', **restored)
    receipts = receipt_inventory(work)
    before_attempts = {p.relative_to(work).as_posix(): control.sha(p) for p in (work / 'runtime/attempts').rglob('*.json')}
    loop._dispatch_workspace_job = no_dispatch
    result_path = ROOT / 'NATIVE_RESULT.json'
    calls = []
    attempt_snapshots = []
    def actual_reuse(fault):
        outcome = loop.run_workspace_job(work, PR['native_completed_job'])
        require(outcome['status'] == 'COMPLETED' and outcome['evidence_status'] == 'VERIFIED', 'NATIVE_REUSE_RESULT')
        control.write(result_path, outcome); calls.append(outcome)
        attempt_snapshots.append({p.relative_to(work).as_posix(): p.read_bytes() for p in (work / 'runtime/attempts').rglob('*.json')})
        if fault:
            raise InterruptedError('P0_INJECT_AFTER_NATIVE_RESULT_BEFORE_STAGE_COMPLETE')
    binding = {'candidate': PR['candidate_inventory_sha256'], 'checkpoint': PR['checkpoint_sha256'], 'capture_id': PR['capture_id']}
    try:
        control.stage(ROOT / 'STAGES.json', 'native', lambda: actual_reuse(True), [result_path], binding)
    except InterruptedError:
        pass
    else:
        raise RuntimeError('EXPECTED_NATIVE_PUBLICATION_FAULT_NOT_OBSERVED')
    control.stage(ROOT / 'STAGES.json', 'native', lambda: actual_reuse(False), [result_path], binding)
    require(len(calls) == 2, 'NATIVE_REUSE_RETRY_COUNT')
    require(not control.stage(ROOT / 'STAGES.json', 'native', lambda: actual_reuse(False), [result_path], binding), 'COMPLETED_STAGE_NOT_REUSED')
    after_attempts = {p.relative_to(work).as_posix(): control.sha(p) for p in (work / 'runtime/attempts').rglob('*.json')}
    require(set(before_attempts) == set(after_attempts), 'NATIVE_REUSE_CREATED_EXECUTION_ATTEMPT')
    require(attempt_snapshots[0] == attempt_snapshots[1], 'SECOND_REUSE_CHANGED_TERMINAL_ATTEMPT')
    import zipfile, io
    with zipfile.ZipFile(archive) as original_zip:
        packet = json.loads(original_zip.read('CHECKPOINT.json'))
    original_state = objects / (packet['checkpoint']['state']['sha256'] + '.bin')
    with zipfile.ZipFile(io.BytesIO(original_state.read_bytes())) as state_zip:
        original_attempts = {n: state_zip.read(n) for n in state_zip.namelist() if n in before_attempts}
    finalizations = []
    for name, original_bytes in original_attempts.items():
        original = json.loads(original_bytes); recovered = json.loads(attempt_snapshots[0][name])
        if original != recovered:
            require(original['status'] == 'COMPLETION_PENDING_CHECKPOINT' and recovered['status'] == 'COMPLETED', 'UNEXPECTED_ATTEMPT_STATUS_TRANSITION')
            require({k:v for k,v in original.items() if k not in ['status','finished_unix']} == {k:v for k,v in recovered.items() if k not in ['status','finished_unix']}, 'NATIVE_EXECUTION_IDENTITY_CHANGED')
            finalizations.append(name)
    require(receipt_inventory(work) == receipts, 'NATIVE_REUSE_CHANGED_SAVE_RECEIPTS')
    record('INTERRUPTED_NATIVE_RESULT_PUBLICATION_REUSES_COMPLETED_WORK', native_calls=2, scientific_dispatches=0,
           attempt_ids_unchanged=True, terminal_attempt_finalizations=finalizations, second_retry_attempt_bytes_unchanged=True, receipt_files_unchanged=True, capture_id=PR['capture_id'])
    # Candidate controller invokes acknowledgement even on completed stages.
    # The existing native receipt batch is used as exact historical evidence.
    ack = Path(PR['saved_unit']) / 'ACK_BATCH.json'
    first = preservation.confirm_batch(work, ack)
    second = preservation.confirm_batch(work, ack)
    require(first['batch_id'] == second['batch_id'], 'ACK_RETRY_BATCH_IDENTITY')
    require(receipt_inventory(work) == receipts, 'ACK_RETRY_CHANGED_RECEIPTS')
    record('EXACT_NATIVE_ACK_BATCH_RETRY', batch_id=first['batch_id'], receipt_files_unchanged=True)
    drain_one = preservation.drain_plan(work)
    drain_two = preservation.drain_plan(work)
    require(drain_one == drain_two, 'DRAIN_PLAN_RETRY_CHANGED')
    record('DRAIN_PLAN_RETRY', capture_id=PR['capture_id'], obligations=len(drain_one['obligations']))
    exports = []
    for name in ['export_first.zip', 'export_retry.zip']:
        output = ROOT / name
        exported = preservation.export_checkpoint(work, output, slim=True)
        exports.append(exported)
    require(exports[0]['sha256'] == exports[1]['sha256'], 'EXPORT_RETRY_BYTES_CHANGED')
    require(receipt_inventory(work) == receipts, 'EXPORT_RETRY_CHANGED_RECEIPTS')
    record('EXACT_NATIVE_EXPORT_RETRY', sha256=exports[0]['sha256'])
    control.write(RESULT, {'status': 'PASS_COMPLETED_UNIT_NATIVE_RECOVERY_PROBE', 'rows': rows,
                          'scientific_dispatches': 0, 'full_native_fault_matrix_qualified': False,
                          'paused_incomplete_execution_qualified': False})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        control.write(RESULT, {'status': 'STOPPED_AT_FIRST_NATIVE_MISMATCH', 'rows': rows,
                              'error': str(exc), 'traceback': traceback.format_exc(), 'scientific_credit': 0,
                              'full_native_fault_matrix_qualified': False})
        raise
