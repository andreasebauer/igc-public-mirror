"""Preregister and capture; this entry never dispatches workload."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'candidate/source'
sys.path.insert(0, str(SOURCE))
from infinity_grid import submission as sub, portable_registry as reg
from infinity_grid import sqlite_attestation as sqlite, v05_controller_event_loop as loop
from infinity_grid.canon import write_json_atomic

ROOT = Path('/tmp/ig_p0_paused155_r1')


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
    prereg = {'schema': 'IG_P0_PAUSED_NATIVE_ENGINEERING_GATE_V1',
        'candidate_version': '0.8.0.dev155+lib', 'source_sha256': source_sha,
        'package_sha256': package_sha,
        'scripts': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([HERE / 'prepare.py', HERE / 'run_probe.py', HERE / 'project/fixture.py',
                HERE / 'checkpoint_tools.py', HERE / 'provider_readback.py'])},
        'runtime_binding_sha256': sub._sha(binding.read_bytes()),
        'registration_revision': 3, 'prior_refusal': 'dev154 automatic paused checkpoint missed final PAUSED claim; prior bytes retained', 'start_method': 'spawn', 'tasks': 8, 'pause_after_commits': 2, 'expected_missing_on_restart': 6,
        'faults': ['NATIVE_ABORT_AFTER_TWO_DURABLE_TASK_COMMITS',
            'PREPARED_COMPLETION_BEFORE_TERMINAL_CHECKPOINT',
            'COMPLETED_FILE_AFTER_TERMINAL_CHECKPOINT_BEFORE_CLAIM'],
        'acceptance': ['Actual provider raw readbacks before native admission',
            'PAUSED attempt with no completion after first abort',
            'Exact cold restore of committed rows and runtime bytes/modes',
            'Restart executes six missing tasks; two prior rows unchanged',
            'Publication retries never dispatch workload',
            'Completion and capsule hash unchanged on final reuse'],
        'stop': 'First unexplained mismatch; retain evidence and do not promote',
        'scope': 'Engineering fixture only; no master admission or scientific claim',
        'resources': {'workers': 1, 'memory_budget_bytes': 4 * 1024**3,
            'workspace_budget_bytes': 256 * 1024**2,
            'execution_policy': 'NO_AUTOMATIC_RUNTIME_DEADLINE_V1',
            'minimum_free_disk_bytes': 10 * 1024**3},
        'live_controller_changed': False, 'independent_host_rc': False}
    write_json_atomic(HERE / 'PREREGISTRATION.json', prereg)
    spec = {'schema_id': sub.SPEC_SCHEMA, 'job_id': 'P0.DEV155.PAUSED.NATIVE.ENGINEERING',
        'engine_source': str(SOURCE), 'project_source': str(HERE / 'project'),
        'question': {'stage_id': 'ENG:P0-PAUSED', 'description': 'Native missing-only recovery fixture',
            'outcomes': ['PASS'], 'stopping_rule': 'Eight finite engineering values; no scientific claim'},
        'execution': {'kind': 'STAGE', 'handler_ref': 'project.fixture:handler',
            'evaluator_refs': ['project.fixture:evaluate'], 'parameters': {}},
        'resources': {k: v for k, v in prereg['resources'].items() if k != 'minimum_free_disk_bytes'} | {'start_method': 'spawn'},
        'inputs': [], 'environment': {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
            'requirements': [], 'artifacts': [{'logical_name': 'sqlite_runtime_binding',
                'path': str(binding), 'sha256': sub._sha(binding.read_bytes())}]},
        'output_contract': {'schema_id': 'IG_DECODER_RESULT_CONTRACT_V1', 'claim': 'EXECUTION_ONLY',
            'required_artifacts': [], 'result_checks': [{'pointer': '/partition/task_count', 'equals': 8},
                {'pointer': '/partition/class_count', 'equals': 2}], 'prerequisites': [],
            'preservation': {'project_evidence_externalization': 1}}}
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
