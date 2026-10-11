"""Verify actual raw Drive responses; private transfer references never persist."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request

HERE = Path(__file__).resolve().parent


def main(stage):
    private = HERE / 'results' / (stage + '_RAW_PRIVATE.json')
    rows = json.loads(private.read_text())
    pending = json.loads((HERE / 'results/CAPTURE_PENDING.json').read_text())
    folder = Path(pending['root']) / 'provider-readbacks'
    folder.mkdir(exist_ok=True)
    proofs = []
    for r in rows:
        if r.get('reuse'):
            raw = Path(r['reuse']['path']).read_bytes()
            provider_id = r['reuse']['drive_file_id']
        else:
            raw = urllib.request.urlopen(r['raw']['download_url']).read()
            provider_id = r['upload']['id']
        if (hashlib.sha256(raw).hexdigest() != r['sha256']
                or len(raw) != r['size_bytes'] or raw != Path(r['path']).read_bytes()):
            raise RuntimeError('PROVIDER_RAW_READBACK_MISMATCH')
        path = folder / (r['sha256'] + '.bin')
        path.write_bytes(raw)
        proofs.append({'sha256': r['sha256'], 'size_bytes': len(raw), 'path': str(path),
            'drive_file_id': provider_id, 'raw_exact_readback': True,
            'reused': bool(r.get('reuse'))})
    (HERE / 'results' / (stage + '_PROVIDER_READBACKS.json')).write_text(json.dumps(proofs, indent=2) + '\n')
    private.unlink()
    print(json.dumps({'stage': stage, 'physical_objects': len(proofs),
        'actual_uploads': sum(not r['reused'] for r in proofs), 'all_raw_exact': True}))


if __name__ == '__main__':
    main(sys.argv[1])
