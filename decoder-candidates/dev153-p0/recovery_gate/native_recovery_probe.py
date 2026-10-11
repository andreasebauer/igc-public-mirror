"""Preregistered completed-unit recovery probe; no scientific dispatch allowed."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PR = json.loads((HERE / 'PREREGISTRATION_REVISION4.json').read_text())
sys.path.insert(0, str(HERE / 'candidate/source'))
sys.path.insert(0, str(HERE / 'candidate/operational'))
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
    # Restored restart closure intentionally has no active outbox. Native
    # snapshot must run under the captured source before pending roles exist.
    import subprocess
    unit = ROOT / 'unit'; unit.mkdir()
    control.write(unit / 'NATIVE_RESULT.json', calls[-1])
    shutil.copyfile(Path(PR['saved_unit']) / 'READBACKS.json', unit / 'READBACKS.json')
    child = "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from p0_control import prepare_checkpoint;sys.path.insert(0,str(Path(sys.argv[2])/'source'));from infinity_grid import preservation as p;prepare_checkpoint(Path(sys.argv[2]),Path(sys.argv[3]),p)"
    captured_python = '/tmp/ig_admit0188/runtime_recovery/ig_runtime_v55_fresh_20260930/python-fixed-host'
    subprocess.run([captured_python, '-B', '-c', child, str(HERE/'candidate/operational'), str(work), str(unit)], check=True)
    current = preservation.status(work)
    require(current['status'] == 'SAVE_REQUIRED' and current['latest_checkpoint'], 'RESTORED_OUTBOX_NOT_REBOUND')
    plan = control.read(unit / 'DRAIN_PLAN.json')
    require(plan == preservation.drain_plan(work), 'NATIVE_DRAIN_PLAN_CHANGED')
    mapping = control.read(unit / 'READBACKS.json')
    new = []
    for row in plan['physical_objects']:
        if row['sha256'] not in mapping:
            new.append(row)
        else:
            raw = Path(mapping[row['sha256']]['path']).read_bytes()
            require(len(raw) == row['size_bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'REUSED_RAW_OBJECT_MISMATCH')
    control.write(HERE/'results/LIVE_PENDING_OBJECTS.json', {'workspace':str(work),'unit':str(unit),'new_objects':new,'native_plan':plan,'checkpoint':current['latest_checkpoint']})
    record('RESTORED_NATIVE_OUTBOX_REBOUND_BEFORE_SAVE_DRAIN', checkpoint=current['latest_checkpoint'],
           pending_roles=len(plan['obligations']), physical_objects=len(plan['physical_objects']), new_objects=len(new), new_bytes=sum(x['size_bytes'] for x in new))
    control.write(RESULT, {'status':'AWAITING_EXACT_PROVIDER_READBACKS', 'rows':rows,'scientific_dispatches':0,
                          'full_native_fault_matrix_qualified':False})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        control.write(RESULT, {'status': 'STOPPED_AT_FIRST_NATIVE_MISMATCH', 'rows': rows,
                              'error': str(exc), 'traceback': traceback.format_exc(), 'scientific_credit': 0,
                              'full_native_fault_matrix_qualified': False})
        raise
