"""Save planning, native ACK and exact cold-directory verification; no dispatch."""
from pathlib import Path
import json
import sqlite3
import stat
import sys

HERE = Path(__file__).resolve().parent
pending = json.loads((HERE / 'results/CAPTURE_PENDING.json').read_text())
WORK = Path(pending.get('resume_workspace', pending['workspace']))
ROOT = Path(pending['root'])
sys.path.insert(0, str(WORK / 'source'))
from infinity_grid import preservation as pr, portable_registry as reg, submission as sub
from infinity_grid import preservation_batch as batch
from infinity_grid.canon import write_json_atomic


def snapshot(stage):
    if stage in {'FINAL', 'REFUSAL'}:
        pr.make_checkpoint(WORK, 'P0_PAUSED_GATE_' + stage)
    plan = pr.drain_plan(WORK)
    write_json_atomic(HERE / 'results' / (stage + '_PLAN.json'), plan)
    print(json.dumps(plan))


def acknowledge(stage):
    plan = json.loads((HERE / 'results' / (stage + '_PLAN.json')).read_text())
    proofs = json.loads((HERE / 'results' / (stage + '_PROVIDER_READBACKS.json')).read_text())
    request = {'schema_id': batch.SCHEMA, 'capture_id': plan['capture_id'],
        'obligations': plan['obligations'], 'readbacks': [
            {'sha256': r['sha256'], 'kind': 'RAW', 'path': r['path'], 'drive_file_id': r['drive_file_id']}
            for r in proofs]}
    path = HERE / 'results' / (stage + '_ACK.json')
    write_json_atomic(path, request)
    if request['obligations']:
        pr.confirm_batch(WORK, path)
    status = pr.status(WORK)
    if status['status'] != 'PRESERVED' or status['pending_objects']:
        raise RuntimeError('NATIVE_DRAIN_NOT_PRESERVED')
    write_json_atomic(HERE / 'results' / (stage + '_PRESERVED.json'), status)
    print(json.dumps({'stage': stage, 'status': status['status'], 'pending_bytes': status['pending_bytes']}))


def cold(stage):
    full = ROOT / (stage + '-full.zip')
    repeat = ROOT / (stage + '-repeat.zip')
    slim = HERE / 'results' / (stage + '_CHECKPOINT_SLIM.zip')
    pr.export_checkpoint(WORK, full)
    pr.export_checkpoint(WORK, repeat)
    if full.read_bytes() != repeat.read_bytes():
        raise RuntimeError('EXPORT_RETRY_BYTES_CHANGED')
    pr.export_checkpoint(WORK, slim, slim=True)
    dest = ROOT / ('cold-' + stage.lower())
    result = pr.restore_checkpoint(full, dest, sub._sha(full.read_bytes()))
    manifest = sub._read(pr._root(WORK) / 'commits' / (result['checkpoint_sha256'] + '.json'))
    state_files = pr._zip_files(pr._bytes(WORK, manifest['state']))
    for shard in manifest.get('state_shards', []):
        state_files[shard['path']] = pr._bytes(WORK, shard)
    for obj in sub.capture_record(WORK)['objects']:
        if obj['role'].startswith('input:'):
            name = 'runtime/intake/artifacts/' + obj['sha256'] + '.bin'
            state_files[name] = (WORK / name).read_bytes()
    for name, raw in state_files.items():
        actual = dest / name
        if actual.read_bytes() != raw:
            raise RuntimeError('COLD_STATE_BYTES:' + name)
        if stat.S_IMODE(actual.stat().st_mode) != manifest['state_file_modes'][name]:
            raise RuntimeError('COLD_STATE_MODE:' + name)
    original_project, _ = reg.locate(WORK)
    restored_project, _ = reg.locate(dest)
    if reg.snapshot(original_project) != reg.snapshot(restored_project):
        raise RuntimeError('COLD_PROJECT_BYTES')
    write_json_atomic(HERE / 'results' / (stage + '_COLD.json'),
        {'status': 'PASS', 'scope': 'Independent directory on same host; not independent-host RC',
            'checkpoint_sha256': result['checkpoint_sha256'], 'validation_nodes': len(json.loads((HERE/'results/NATIVE_RESULT.json').read_text())['result']['result']['nodes']),
            'state_files_exact': len(state_files), 'state_modes_exact': True,
            'full_export_sha256': sub._sha(full.read_bytes()), 'full_export_bytes': full.stat().st_size,
            'slim_export_sha256': sub._sha(slim.read_bytes()), 'slim_export_bytes': slim.stat().st_size,
            'destination': str(dest), 'restored_active_outbox': pr.status(dest)['status']})
    if stage == 'PAUSE':
        if sub.save_status(dest)['status'] != 'SAVED':
            raise RuntimeError('COLD_CAPTURE_RECEIPTS_NOT_SAVED')
        pending['resume_workspace'] = str(dest)
        write_json_atomic(HERE / 'results/CAPTURE_PENDING.json', pending)
    repeat.unlink()
    print(json.dumps({'stage': stage, 'status': 'COLD_EXACT_PASS', 'validation_nodes': len(json.loads((HERE/'results/NATIVE_RESULT.json').read_text())['result']['result']['nodes'])}))


if __name__ == '__main__':
    {'snapshot': snapshot, 'ack': acknowledge, 'cold': cold}[sys.argv[1]](sys.argv[2])
