"""Preregister and capture; this entry never dispatches workload."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'decoder165'
sys.path.insert(0, str(SOURCE))
from infinity_grid import submission as sub, portable_registry as reg
from infinity_grid import sqlite_attestation as sqlite, v05_controller_event_loop as loop
from infinity_grid.canon import write_json_atomic

ROOT = Path('/tmp/ig_p0_capture166_r1')


def main():
    if ROOT.exists():
        raise RuntimeError('DISPOSABLE_ROOT_EXISTS')
    if shutil.disk_usage('/tmp').free < 10 * 1024**3:
        raise RuntimeError('MINIMUM_FREE_DISK')
    ROOT.mkdir()
    runtime = {'schema_id': 'IG_CAPTURE_SQLITE_BINDING_V1',
        'policy': 'REQUIRE_REVIEWED_WAL_BUILD', 'observation': sqlite.require_wal_fix()}
    binding = ROOT / 'sqlite-binding.json'
    write_json_atomic(binding, runtime)
    source_sha, package_sha = loop._source_ids(SOURCE)
    nodes = ['tests/test_decoder06_capture.py', 'tests/test_decoder06_invocation.py', 'tests/test_change_preservation_rebind.py']
    prereg = {'schema': 'IG_P0_NATIVE_CAPTURE_VALIDATION_V1', 'nodes': nodes,
        'source_sha256': source_sha, 'package_sha256': package_sha,
        'stop': 'First nonpass; retain evidence; no promotion',
        'scope': 'Same-host captured validation only; full RC and independent host open',
        'resources': {'workers': 1, 'memory_budget_bytes': 4*1024**3,
          'workspace_budget_bytes': 512*1024**2, 'start_method': 'fork',
          'execution_policy': 'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},
        'scripts': {p.name: sub._sha(p.read_bytes()) for p in HERE.glob('*.py')}}
    write_json_atomic(HERE / 'PREREGISTRATION.json', prereg)
    spec = {'schema_id': sub.SPEC_SCHEMA, 'job_id': 'P0.DEV165.CAPTURE.VALIDATION',
      'engine_source': str(SOURCE), 'project_source': None,
      'question': {'stage_id': 'ENG:P0-CAPTURE', 'description': 'Saved native capture and authority validation',
        'outcomes': ['PASS', 'FAIL'], 'stopping_rule': 'Three frozen test files; stop on nonpass'},
      'execution': {'kind': 'VALIDATION', 'nodes': nodes},
      'resources': prereg['resources'], 'inputs': [],
      'environment': {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
        'requirements': [], 'artifacts': [{'logical_name': 'sqlite_runtime_binding',
        'path': str(binding), 'sha256': sub._sha(binding.read_bytes())}]},
      'output_contract': {'scientific_acceptance': 'NONE_ENGINEERING_VALIDATION'}}
    write_json_atomic(HERE / 'results/SPECIFICATION.json', spec)
    store = ROOT / 'store'
    reg.initialize(store, 'P0 paused native engineering gate', SOURCE)
    capture = sub.capture(store, spec)
    work = Path(capture['workspace'])
    rows = []
    for obj in sub.required_objects(work):
        path = store / 'objects' / obj['object_name']
        if not path.is_file():
            raise RuntimeError('CAPTURE_OBJECT_PATH:' + obj['role'])
        rows.append(dict(obj, path=str(path)))
    write_json_atomic(HERE / 'results/CAPTURE_PENDING.json',
        {'capture': capture, 'workspace': str(work), 'root': str(ROOT), 'obligations': rows})
    print(json.dumps({'status': capture['status'], 'capture_id': capture['capture_id'],
        'obligations': len(rows), 'physical_objects': len({r['sha256'] for r in rows}),
        'source_sha256': source_sha}, sort_keys=True))


if __name__ == '__main__':
    main()
