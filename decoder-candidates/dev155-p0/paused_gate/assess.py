"""Assess persisted native gate evidence; no execution or promotion."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def main():
    read = lambda name: json.loads((HERE / 'results' / name).read_text())
    phases = {mode: read(mode + '.json') for mode in ['PAUSE', 'PREPARED', 'PUBLISHED', 'FINISH', 'REUSE']}
    pause, prepared = phases['PAUSE'], phases['PREPARED']
    if len(pause['attempts']) != 1 or len(prepared['attempts']) != 2:
        raise RuntimeError('EXPECTED_EXECUTION_ATTEMPTS')
    done = next(iter(prepared['prepared'].values()))
    execution = done['result']['execution']
    if execution['task_count'] != 6 or execution['completed_task_count'] != 6 or execution['workers'] != 1:
        raise RuntimeError('NATIVE_MISSING_ONLY_EXECUTION_METADATA')
    rows = prepared['task_rows']
    for name in ['PUBLISHED', 'FINISH', 'REUSE']:
        phase = phases[name]
        if phase['task_rows'] != rows or len(phase['attempts']) != 2:
            raise RuntimeError('PUBLICATION_RETRY_DISPATCHED_WORKLOAD:' + name)
        if next(iter(phase['prepared'].values())) != done:
            raise RuntimeError('BOUND_COMPLETION_REWRITTEN:' + name)
    if not phases['FINISH']['result']['reused'] or not phases['REUSE']['result']['reused']:
        raise RuntimeError('REUSE_NOT_NATIVE')
    if phases['FINISH']['completions'] != phases['REUSE']['completions']:
        raise RuntimeError('REUSE_COMPLETION_BYTES_CHANGED')
    result = {'status': 'BOUNDED_PAUSED_NATIVE_RECOVERY_GATE_PASS',
        'candidate_version': '0.8.0.dev155+lib',
        'capture_id': read('CAPTURE_PENDING.json')['capture']['capture_id'],
        'first_committed_tasks': 2, 'missing_tasks_executed_after_cold_restore': 6,
        'final_task_count': 8, 'execution_attempts': 2,
        'publication_retry_execution_attempts': 0,
        'prior_task_rows_unchanged': True, 'completion_bytes_unchanged': True,
        'completion_sha256': done['completion_sha256'],
        'paused_cold_restore': read('PAUSE_COLD.json'),
        'resource_scope': {'workers': 1, 'memory_budget_bytes': 4294967296,
            'start_method': 'spawn', 'overall_execution_deadline': None},
        'scientific_acceptance': 'NONE_ENGINEERING_FIXTURE',
        'full_p0_accepted': False, 'independent_host_rc': False,
        'remaining': ['Global identity V2-to-V3 impact and migration mapping',
            'Full registered RC including native context tests and independent-host runtime/source restore',
            'Provider outage and unknown-write outcome qualification',
            'Accepted promotion and controller transition']}
    (HERE / 'results/NATIVE_MATRIX.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
