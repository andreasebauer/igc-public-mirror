"""Restore this source overlay over the exact dev151 baseline; stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def inventory(root):
    rows = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.name.endswith('.pyc'):
            continue
        if path.is_symlink():
            raise RuntimeError('SOURCE_SYMLINK_REFUSED')
        if path.is_file():
            data = path.read_bytes()
            rows[path.relative_to(root).as_posix()] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-source', type=Path, required=True)
    parser.add_argument('--overlay-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.overlay_root / 'SOURCE_MANIFEST.json').read_text())
    if inventory(args.base_source) != manifest['baseline_files']:
        raise RuntimeError('BASE_SOURCE_BYTES_OR_FILE_SET_MISMATCH')
    if args.destination.exists():
        raise RuntimeError('DESTINATION_MUST_BE_ABSENT')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='p0-restore-', dir=args.destination.parent))
    try:
        shutil.copytree(args.base_source, stage / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in manifest['changed_source_files']:
            source = args.overlay_root / 'source' / name
            destination = stage / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if inventory(stage / 'source') != manifest['candidate_files']:
            raise RuntimeError('CANDIDATE_SOURCE_BYTES_OR_FILE_SET_MISMATCH')
        (stage / 'source').rename(args.destination)
    finally:
        shutil.rmtree(stage)
    print(json.dumps({'status': 'PASS_EXACT_CANDIDATE_SOURCE_RESTORE', 'files': len(manifest['candidate_files']), 'scientific_credit': 0}))


if __name__ == '__main__':
    main()
