"""Boundary faults only; native admission, dispatch, and save gates remain real."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import sys
import traceback

HERE = Path(__file__).resolve().parent
pending = json.loads((HERE / 'results/CAPTURE_PENDING.json').read_text())
WORK = Path(pending.get('resume_workspace', pending['workspace']))
sys.path.insert(0, str(WORK / 'source'))
from infinity_grid import v05_controller_event_loop as loop, preservation as pr
from infinity_grid import submission as sub
from infinity_grid.canon import write_json_atomic


def db_rows():
    paths = list((WORK / 'runtime/runs').rglob('partition.sqlite3'))
    if len(paths) != 1:
        raise RuntimeError('PARTITION_DB_COUNT:' + str(len(paths)))
    with sqlite3.connect(f'file:{paths[0]}?mode=ro', uri=True) as db:
        return db.execute('SELECT * FROM task_results ORDER BY task_id').fetchall()


def main(mode):
    prereg = json.loads((HERE / 'PREREGISTRATION.json').read_text())
    for name, digest in prereg['scripts'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('PROBE_SOURCE_PIN:' + name)
    if mode == 'pause':
        os.environ['IG_V05_ENGINEERING_ABORT_AFTER_COMMITS'] = '2'
        os.environ['IG_V05_ENGINEERING_FAULT_INJECTION_ACK'] = 'STAGE_RUNTIME_ABORT_TEST'
    elif mode == 'prepared':
        def interrupted(*args, **kwargs):
            raise InterruptedError('PREREG_BEFORE_TERMINAL_CHECKPOINT')
        loop._publish_checkpointed_completion = interrupted
    elif mode == 'published':
        original = loop.write_json_atomic
        def publish_then_interrupt(path, value):
            original(path, value)
            if Path(path).parent.name == 'completed':
                raise InterruptedError('PREREG_AFTER_COMPLETED_FILE_BEFORE_CLAIM')
        loop.write_json_atomic = publish_then_interrupt
    elif mode not in {'finish', 'reuse'}:
        raise RuntimeError('UNKNOWN_MODE')
    result = None
    issue = None
    try:
        result = loop.run_workspace_job(WORK, pending['capture']['job_id'])
    except BaseException as exc:
        issue = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
    expected = {'pause': 'ENGINEERING_FAULT_INJECTION_ABORT_AFTER_2_COMMITS',
        'prepared': 'PREREG_BEFORE_TERMINAL_CHECKPOINT',
        'published': 'PREREG_AFTER_COMPLETED_FILE_BEFORE_CLAIM'}
    if mode in expected:
        if issue is None or expected[mode] not in issue['error']:
            raise RuntimeError('EXPECTED_NATIVE_BOUNDARY_NOT_REACHED:' + json.dumps(issue))
    elif issue is not None:
        raise RuntimeError(json.dumps(issue))
    rows = db_rows()
    attempts = {str(p.relative_to(WORK)): json.loads(p.read_text())
        for p in sorted((WORK / 'runtime/attempts').rglob('*.json'))}
    completions = {str(p.relative_to(WORK)): sub._sha(p.read_bytes())
        for p in (WORK / 'runtime/intake/completed').glob('*.json')}
    prepared = {str(p.relative_to(WORK)): json.loads(p.read_text())
        for p in (WORK / 'runtime/intake/prepared_completions').glob('*.json')}
    if len(rows) != (2 if mode == 'pause' else 8):
        raise RuntimeError('TASK_COVERAGE_MISMATCH')
    if mode in {'pause', 'prepared'} and completions:
        raise RuntimeError('PREMATURE_COMPLETION_PUBLICATION')
    if mode != 'pause':
        old = json.loads((HERE / 'results/PAUSE.json').read_text())['task_rows']
        current = {r[0]: list(r) for r in rows}
        if any(current[r[0]] != r for r in old):
            raise RuntimeError('PRIOR_COMMITTED_TASK_CHANGED')
        if not prepared or len(prepared) != 1:
            raise RuntimeError('PREPARED_COMPLETION_COUNT')
        done = next(iter(prepared.values()))
        if done['result']['execution']['task_count'] != 6:
            raise RuntimeError('RESTART_NOT_MISSING_ONLY')
        if mode not in {'prepared'}:
            old_done = next(iter(json.loads((HERE / 'results/PREPARED.json').read_text())['prepared'].values()))
            if done != old_done:
                raise RuntimeError('PREPARED_COMPLETION_CHANGED_ON_RETRY')
    write_json_atomic(HERE / 'results' / (mode.upper() + '.json'),
        {'mode': mode, 'status': 'PASS', 'issue': issue, 'result': result,
            'task_rows': rows, 'attempts': attempts, 'completions': completions,
            'prepared': prepared, 'native_preservation': pr.status(WORK),
            'scientific_acceptance': 'NONE_ENGINEERING_FIXTURE'})
    print(json.dumps({'mode': mode, 'status': 'PASS', 'task_count': len(rows),
        'completed_files': len(completions), 'reused': None if result is None else result.get('reused')}))


if __name__ == '__main__':
    main(sys.argv[1])
