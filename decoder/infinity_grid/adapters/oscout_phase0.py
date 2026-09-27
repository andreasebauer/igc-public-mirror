from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..canon import write_json_atomic
from ..controller import register_runner

EXPECTED_SCIENCE_SHA256 = '82bf4927956bf92696d9bc80b018bf56b9dba940becabab05ce94fe43fa07c51'
EXPECTED_TESTS = 151


def execute(fixture: Path, work: Path):
    """Replay the frozen OScout v0.1 Phase-0 specification verifier.

    This is a compatibility replay only.  It must not generate O7 or change any
    OScout scientific authority.  The exact historical Phase-0 verifier and
    specification packet are executed in a fresh staging copy.
    """
    fixture = Path(fixture)
    work = Path(work)
    target = work / 'oscout_phase0'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(fixture, target, copy_function=shutil.copyfile)
    # Dataset snapshots are intentionally read-only.  This replay executes in a
    # private worker-owned copy, so restore write permission only inside staging.
    for q in target.rglob('*'):
        try:
            os.chmod(q, 0o700 if q.is_dir() else 0o600)
        except OSError:
            pass
    os.chmod(target, 0o700)
    (target / '06_RESULTS').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = '0'
    p = subprocess.run(
        [sys.executable, str(target / '05_CODE/verify_oscout_phase0.py'), str(target)],
        cwd=str(target), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    (target / '06_RESULTS/UNIFIED_RUNTIME_STDOUT.txt').write_text(p.stdout, encoding='utf-8')
    (target / '06_RESULTS/UNIFIED_RUNTIME_STDERR.txt').write_text(p.stderr, encoding='utf-8')
    if p.returncode:
        raise RuntimeError(f'OScout Phase0 verifier failed rc={p.returncode}: {p.stderr[-2000:]}')

    result = json.loads((target / '06_RESULTS/OSCOUT_PHASE0_VERIFICATION_RESULT.json').read_text(encoding='utf-8'))
    science = (target / '06_RESULTS/SCIENCE_SHA256.txt').read_text(encoding='utf-8').strip()
    tests = result.get('tests', [])
    failures = [t for t in tests if t.get('status') != 'PASS']
    principal = {
        'status': 'PASS' if (
            result.get('status') == 'PASS'
            and len(tests) == EXPECTED_TESTS
            and not failures
            and science == EXPECTED_SCIENCE_SHA256
            and result.get('formal_status') == 'OSCOUT_V0_1_PHASE0_SPECIFICATION_FORMALIZATION_COMPLETE'
            and result.get('O7_status') == 'NOT_RUN_NOT_EARNED'
        ) else 'FAIL',
        'target': 'OSCOUT_V0_1_PHASE0_SELFTEST',
        'tests_total': len(tests),
        'tests_pass': len(tests) - len(failures),
        'tests_fail': len(failures),
        'science_sha256': science,
        'expected_science_sha256': EXPECTED_SCIENCE_SHA256,
        'protocol_label': 'OSCOUT_V0_1_PHASE0_SPECIFICATION_FORMALIZATION_COMPLETE',
        'authority': 'SCOPED_AUDIT_AUTHORITY',
        'o7_status': 'NOT_RUN_NOT_EARNED',
        'live_generation': False,
    }
    return (
        principal,
        target / '06_RESULTS/OSCOUT_PHASE0_VERIFICATION_RESULT.json',
        target / '06_RESULTS/SCIENCE_SHA256.txt',
    )


def _journal_payload(principal: dict) -> dict:
    return {'principal': principal}


@register_runner('adapter.oscout_phase0', v05_operation='OSCOUT_PHASE0_REFERENCE_REPLAY', call_style='context')
def stage_runner(*, context, plan, stage):
    ds = stage['params']['fixture_dataset_sha256']
    fixture = context.materialize_dataset(ds, 'fixture')
    journal = context.task_journal()
    task_id = 'oscout-phase0-reference-replay'
    rec = journal.load(task_id)
    reused = rec is not None
    if rec is None:
        principal, _, _ = execute(fixture, context.staging_path('execution'))
        if principal['status'] != 'PASS':
            raise RuntimeError(f'OScout Phase0 self-test mismatch: {principal}')
        journal.commit(task_id, _journal_payload(principal))
        rec = journal.load(task_id)
    principal = rec['payload']['principal']
    out = context.staging_path('OSCOUT_PHASE0_PRINCIPAL_RESULT.json')
    write_json_atomic(out, principal)
    js = journal.summary()
    return {
        'outputs': {'OSCOUT_PHASE0_PRINCIPAL_RESULT.json': out},
        'stage_result': {
            'status': 'PASS',
            'target': 'VS_OSCOUT_PHASE0',
            'protocol_label': 'OSCOUT_V0_1_PHASE0_SPECIFICATION_FORMALIZATION_COMPLETE',
            'authority': 'SCOPED_AUDIT_AUTHORITY',
            'o7_status': 'NOT_RUN_NOT_EARNED',
            'live_generation': False,
            'science_sha256': principal['science_sha256'],
            'logical_tasks': {
                'tasks_total': 1,
                'tasks_reused': 1 if reused else 0,
                'tasks_committed_this_attempt': 0 if reused else 1,
                'journal': js,
            },
        },
    }
