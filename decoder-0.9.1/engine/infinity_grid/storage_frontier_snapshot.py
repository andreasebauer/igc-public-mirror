"""Read-only binding of a supported frontier to original runner snapshot bytes.

This is not dependency closure or authorization to resume science.
"""
import hashlib
from .storage_schema import strict_loads
from .replay_reference_data import verify_reference_record
from .storage_runner_snapshot import verify_runner_snapshot, RunnerSnapshotError


def verify_frontier_snapshot(frontier_raw, manifest_raw, state_raw, *,
                             checkpoints=(), capsules=(), decisions=(),
                             max_total_bytes=67108864, max_records=4096):
    if type(frontier_raw) is not bytes:
        raise RunnerSnapshotError('FRONTIER_BYTES_REQUIRED')
    if type(max_total_bytes) is not int or not 1 <= max_total_bytes <= 67108864 or type(max_records) is not int or not 3 <= max_records <= 4096:
        raise RunnerSnapshotError('INVALID_FRONTIER_BUDGET')
    if len(frontier_raw) >= max_total_bytes:
        raise RunnerSnapshotError('FRONTIER_BYTE_BUDGET')
    record = strict_loads(frontier_raw, max_bytes=4194304)
    if not isinstance(record, dict):
        raise RunnerSnapshotError('FRONTIER_OBJECT_REQUIRED')
    record = verify_reference_record(record)
    if record['record_type'] != 'RESUME_FRONTIER':
        raise RunnerSnapshotError('FRONTIER_TYPE_REQUIRED')
    snapshot = verify_runner_snapshot(manifest_raw, state_raw,
        checkpoints=checkpoints, capsules=capsules, decisions=decisions,
        max_total_bytes=max_total_bytes-len(frontier_raw), max_records=max_records-1)
    manifest = strict_loads(manifest_raw, max_bytes=4194304)
    state = strict_loads(state_raw, max_bytes=4194304)
    payload = record['payload']
    expected = {'root_run_id': state['root_run_id'],
        'manifest_dag_sha256': manifest['dag_sha256'],
        'runner_state_sha256': state['state_sha256'],
        'completed_node_ids': state['completed_node_ids'],
        'checkpoint_sha256_by_node': state['accepted_checkpoint_sha256_by_node']}
    for key, value in expected.items():
        if payload[key] != value:
            raise RunnerSnapshotError('FRONTIER_SNAPSHOT_MISMATCH:' + key)
    # This producer labels a semantic DAG seal as provenance, not a raw digest.
    if record['provenance']['source_hashes'] != [{'ref': 'compiled_manifest.dag_sha256', 'sha256': manifest['dag_sha256']}]:
        raise RunnerSnapshotError('UNSUPPORTED_FRONTIER_PROVENANCE')
    completed = set(state['completed_node_ids'])
    remaining = [n for n in manifest['topological_order'] if n not in completed]
    if not remaining:
        raise RunnerSnapshotError('UNSUPPORTED_TERMINAL_FRONTIER')
    next_id = remaining[0]
    node = next(n for n in manifest['nodes'] if n['canonical_id'] == next_id)
    if any(dep not in completed for dep in node['dependencies']):
        raise RunnerSnapshotError('FRONTIER_DEPENDENCY_ORDER')
    active = state['active_audit_capsule_sha256']
    if active is not None:
        capsule = next(strict_loads(raw, max_bytes=4194304) for raw in capsules
            if strict_loads(raw, max_bytes=4194304)['audit_capsule_sha256'] == active)
        if state['runner_status'] != 'WAITING_FOR_EXTERNAL_AUDIT' or payload['frontier_status'] != 'WAITING_FOR_EXTERNAL_AUDIT' or capsule['stopped_node_id'] != next_id:
            raise RunnerSnapshotError('FRONTIER_AUDIT_STATUS_MISMATCH')
    else:
        if state['runner_status'] != 'READY' or payload['frontier_status'] not in {'READY', 'WAITING_FOR_NODE_EXECUTOR_BINDING'}:
            raise RunnerSnapshotError('FRONTIER_READY_STATUS_MISMATCH')
        if node['node_kind'] == 'WORKFLOW_GATE':
            raise RunnerSnapshotError('FRONTIER_PENDING_GATE_TRANSITION')
    if payload['next_node_id'] != next_id:
        raise RunnerSnapshotError('FRONTIER_NEXT_NODE_MISMATCH')
    return {'status': 'FRONTIER_SNAPSHOT_BOUND', 'scope': 'FRONTIER_AND_RUNNER_INDEX_ONLY',
        'execution_authorized': False, 'scientific_acceptance': 'NOT_GRANTED',
        'dependency_closure_verified': False, 'record_id': record['record_id'],
        'record_sha256': record['record_sha256'], 'next_node_id': next_id,
        'frontier_status': payload['frontier_status'], 'snapshot': snapshot,
        'frontier_content_ref': {'sha256': hashlib.sha256(frontier_raw).hexdigest(), 'size_bytes': str(len(frontier_raw))},
        'records_checked': snapshot['records_checked'] + 1,
        'bytes_checked': snapshot['bytes_checked'] + len(frontier_raw)}
