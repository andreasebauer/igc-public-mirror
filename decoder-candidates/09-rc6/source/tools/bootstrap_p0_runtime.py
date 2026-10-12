#!/usr/bin/env python3
"""Restore the exact V5 bundled runtime; no science or ambient package install."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile

RUNTIME_ARCHIVE_SHA256 = '25f76fa883c03b2ccc8a902d0aca799abd167279dbc6108ee2dad8095ac90222'
LAUNCHER = '''#!/bin/sh
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
unset LD_PRELOAD VIRTUAL_ENV PYTHONSTARTUP PYTHONPATH PYTHONOPTIMIZE
export PYTHONHOME="$HERE/base" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$HERE/host:$HERE/native"
exec "$HERE/host/ld-linux-x86-64.so.2" --library-path "$HERE/host:$HERE/native" "$HERE/base/bin/python3.13" "$@"
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    bundle, runtime = args.bundle_root.resolve(strict=True), args.runtime_root.resolve()
    if not runtime.is_relative_to('/tmp') or runtime == Path('/tmp'):
        raise ValueError('RUNTIME_MUST_BE_DEDICATED_TMP_DIRECTORY')
    source = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(source))
    from infinity_grid import isolated_runtime as bound
    raw = bound.BUNDLED_MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != bound.BUNDLED_MANIFEST_SHA256:
        raise ValueError('BUNDLED_MANIFEST_HASH')
    manifest = json.loads(raw)
    archive = bundle / 'runtime/IG_RUNTIME_CPY3135_SQLITE3513.tar.gz'
    if sha(archive) != RUNTIME_ARCHIVE_SHA256:
        raise ValueError('RUNTIME_ARCHIVE_HASH')
    if runtime.exists():
        if not args.verify_existing:
            raise ValueError('RUNTIME_DESTINATION_EXISTS')
        bound._verify_tree(runtime, manifest)
    else:
        temporary = runtime.with_name(runtime.name + '.extracting')
        if temporary.exists():
            raise ValueError('RUNTIME_EXTRACTION_INCOMPLETE_RECONCILE_FIRST')
        temporary.mkdir(parents=True)
        try:
            top = 'ig_runtime_v55_fresh_20260930'
            with tarfile.open(archive, 'r:gz') as tar:
                for item in tar.getmembers():
                    path = Path(item.name)
                    if (path.is_absolute() or '..' in path.parts or
                            not (item.isdir() or item.isfile()) or
                            not (item.name == top or item.name.startswith(top + '/'))):
                        raise ValueError('UNSAFE_RUNTIME_MEMBER')
                tar.extractall(temporary, filter='data')
            root = temporary / top
            (root / 'host').mkdir()
            for name in manifest['bundled_host_files']:
                src = bundle / 'runtime/host_bindings' / Path(name).name
                shutil.copyfile(src, root / name)
            for name in ('python-fixed-host', 'python-bundled-host'):
                (root / name).write_text(LAUNCHER)
            for name in manifest['runtime_files']:
                os.chmod(root / name, 0o755 if name in manifest['executables'] else 0o644)
            bound._verify_tree(root, manifest)
            root.replace(runtime)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    probe = subprocess.run([str(runtime / 'python-bundled-host'), '-B', '-c',
                            'import sys,json,sqlite3,platform;from pathlib import Path;'
                            'sys.path.insert(0,sys.argv[1]);'
                            'from infinity_grid.isolated_runtime import verified_runtime;'
                            'print(json.dumps({"python":platform.python_version(),"sqlite":sqlite3.sqlite_version,"runtime":str(verified_runtime())}))',
                            str(source)], capture_output=True, text=True, check=True)
    result = json.loads(probe.stdout)
    if result['python'] != '3.13.5' or result['sqlite'] != '3.51.3':
        raise ValueError('RUNTIME_VERSION_MISMATCH')
    print(json.dumps({'status': 'PASS_EXACT_V5_BUNDLED_RUNTIME_RESTORE',
                      'runtime_files': len(manifest['runtime_files']),
                      'manifest_sha256': bound.BUNDLED_MANIFEST_SHA256,
                      'loader_mode': oct(stat.S_IMODE((runtime / 'host/ld-linux-x86-64.so.2').stat().st_mode)),
                      'scientific_execution': False,
                      'independent_host_qualification': False, **result}, indent=2))


if __name__ == '__main__':
    main()
