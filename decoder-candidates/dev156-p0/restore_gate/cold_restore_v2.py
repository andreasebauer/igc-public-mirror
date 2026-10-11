"""Read-only dev156 restoration of the unchanged saved dev155 checkpoint."""
from pathlib import Path
import hashlib
import json
import stat
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'restore_candidate/source'))
from infinity_grid import preservation as pr, portable_registry as reg, submission as sub
from infinity_grid import v05_controller_event_loop as loop


def main():
    prereg = json.loads((HERE / 'RESTORE_PREREGISTRATION.json').read_text())
    source = HERE / 'restore_candidate/source'
    if loop._source_ids(source)[0] != prereg['restorer_source_sha256']:
        raise RuntimeError('RESTORER_SOURCE_PIN')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != prereg['probe_sha256']:
        raise RuntimeError('RESTORE_PROBE_PIN')
    pending = json.loads((HERE / 'results/CAPTURE_PENDING.json').read_text())
    work = Path(pending['resume_workspace'])
    archive = Path(pending['root']) / 'FINAL-full.zip'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != prereg['archive_sha256']:
        raise RuntimeError('SAVED_FINAL_EXPORT_PIN')
    destination = Path(pending['root']) / 'cold-final-v2'
    result = pr.restore_checkpoint(archive, destination, digest)
    packet = sub._read(pr._root(work) / 'commits' / (result['checkpoint_sha256'] + '.json'))
    state = pr._zip_files(pr._bytes(work, packet['state']))
    for shard in packet.get('state_shards', []):
        state[shard['path']] = pr._bytes(work, shard)
    for obj in sub.capture_record(work)['objects']:
        if obj['role'].startswith('input:'):
            name = 'runtime/intake/artifacts/' + obj['sha256'] + '.bin'
            state[name] = (work / name).read_bytes()
    for name, raw in state.items():
        if (destination / name).read_bytes() != raw:
            raise RuntimeError('COLD_STATE_BYTES:' + name)
        if stat.S_IMODE((destination / name).stat().st_mode) != packet['state_file_modes'][name]:
            raise RuntimeError('COLD_STATE_MODE:' + name)
    original_project, _ = reg.locate(work)
    restored_project, binding = reg.locate(destination)
    if reg.snapshot(original_project) != reg.snapshot(restored_project):
        raise RuntimeError('COLD_FULL_PROJECT_BYTES')
    _, claim = reg.current(restored_project, 'work:' + binding['work_id'])
    if claim['state'] != 'COMPLETED':
        raise RuntimeError('COLD_COMPLETED_CLAIM')
    capsule = reg.verify_capsule(restored_project, claim['capsule'], binding['identity'])
    # Reuse the existing completion administratively. This must succeed even
    # though the loaded dev156 source differs from captured dev155 source;
    # an attempted fresh dispatch would fail loaded-source admission.
    reused = loop.run_workspace_job(destination, pending['capture']['job_id'])
    if not reused['reused'] or reused['completion_sha256'] != claim['completion_sha256']:
        raise RuntimeError('COLD_EXACT_COMPLETION_REUSE')
    latest = sub._read(pr._root(destination) / 'RESTORE_PROVENANCE.json')
    if latest['checkpoint_sha256'] != result['checkpoint_sha256']:
        raise RuntimeError('LATEST_RESTORE_ADMINISTRATIVE_RECEIPT')
    row = {'status': 'PASS', 'scope': 'Independent directory on same host; not independent-host RC',
        'restorer_version': '0.8.0.dev156+lib',
        'captured_source_sha256': sub.capture_record(destination)['workspace']['source_sha256'],
        'captured_decoder_version': sub.capture_record(destination)['decoder_version'],
        'checkpoint_sha256': result['checkpoint_sha256'],
        'full_export_sha256': digest, 'full_export_bytes': archive.stat().st_size,
        'state_files_exact': len(state), 'state_modes_exact': True,
        'prior_provenance_bytes_preserved': True,
        'latest_restore_receipt_separate': True,
        'completion_sha256': capsule['completion_sha256'],
        'capsule_sha256': claim['capsule'], 'cold_administrative_reuse': True,
        'new_workload_dispatches': 0, 'restored_active_outbox': pr.status(destination)['status']}
    (HERE / 'results/FINAL_COLD_V2.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    main()
