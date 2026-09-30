from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_SCIENCE_SHA256 = '82bf4927956bf92696d9bc80b018bf56b9dba940becabab05ce94fe43fa07c51'
EXPECTED_TESTS = 151

OSCOUT_PHASE0_COMPARISON_FIELDS = [
    'status', 'target', 'tests_total', 'tests_pass', 'tests_fail',
    'science_sha256', 'expected_science_sha256', 'protocol_label',
    'authority', 'o7_status', 'live_generation',
]


def replay_oscout_phase0(fixture: Path, root: Path) -> dict:
    """Cold replay of the exact frozen OScout Phase-0 verifier packet.

    This is deliberately a compatibility verifier, not an independent second-code
    theorem proof.  It rematerializes the immutable packet into a fresh directory
    and executes its historical verifier under deterministic hash seeding.
    """
    fixture = Path(fixture)
    root = Path(root)
    target = root / 'oscout_phase0'
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
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
        [sys.executable, '-B', str(target / '05_CODE/verify_oscout_phase0.py'), str(target)],
        cwd=str(target), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if p.returncode:
        raise RuntimeError(f'OScout cold replay failed rc={p.returncode}: {p.stderr[-2000:]}')
    result = json.loads((target / '06_RESULTS/OSCOUT_PHASE0_VERIFICATION_RESULT.json').read_text(encoding='utf-8'))
    science = (target / '06_RESULTS/SCIENCE_SHA256.txt').read_text(encoding='utf-8').strip()
    tests = result.get('tests', [])
    failures = [t for t in tests if t.get('status') != 'PASS']
    return {
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


def oscout_phase0_science_projection(principal: dict) -> dict:
    return {k: principal.get(k) for k in OSCOUT_PHASE0_COMPARISON_FIELDS}
