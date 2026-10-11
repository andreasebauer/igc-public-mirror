"""Bounded runtime-kit checks; no native science or independent-host claim."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from infinity_grid import isolated_runtime as bound


def test_v5_current_runtime_is_exact_and_includes_loader_executable_mode():
    root = bound.verified_runtime()
    if not (root / 'python-bundled-host').exists():
        pytest.skip('This check requires the V5 bundled runtime')
    manifest = json.loads(bound.BUNDLED_MANIFEST.read_text())
    assert len(manifest['runtime_files']) == 4729
    assert (root / 'host/ld-linux-x86-64.so.2').stat().st_mode & 0o777 == 0o755
    assert manifest['host'] == {}
    assert len(manifest['bundled_host_files']) == 7


def test_pinned_loader_launches_isolated_child_with_packages_and_sqlite():
    source = Path(bound.__file__).resolve().parents[1]
    code = '''import sys,json,sqlite3
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from infinity_grid.isolated_runtime import configure_isolated_paths,verified_runtime
configure_isolated_paths(Path(sys.argv[1]))
from cryptography.fernet import Fernet
c=Fernet(Fernet.generate_key())
if c.decrypt(c.encrypt(b'bound'))!=b'bound':raise RuntimeError('CRYPTO')
print(json.dumps({'sqlite':sqlite3.sqlite_version,'runtime':str(verified_runtime()),'flags':[sys.flags.isolated,sys.flags.no_site,sys.flags.dont_write_bytecode], 'maps':Path('/proc/self/maps').read_text()}))
'''
    result = subprocess.run(bound.subprocess_argv(['-B', '-I', '-S', '-c', code, str(source)]),
                            env=bound.subprocess_environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['sqlite'] == '3.51.3' and data['flags'] == [1, 1, 1]
    assert str(bound.verified_runtime() / 'host/ld-linux-x86-64.so.2') in data['maps']


def test_optimized_native_runtime_is_refused():
    source = Path(bound.__file__).resolve().parents[1]
    code = 'import sys;sys.path.insert(0,sys.argv[1]);from infinity_grid.isolated_runtime import verified_runtime;verified_runtime()'
    result = subprocess.run(bound.subprocess_argv(['-B', '-O', '-c', code, str(source)]),
                            env=bound.subprocess_environment(), capture_output=True, text=True)
    assert result.returncode != 0 and 'OPTIMIZED_RUNTIME' in result.stderr


def test_bundled_manifest_pin_cannot_be_changed(monkeypatch, tmp_path):
    manifest = tmp_path / 'altered.json'; manifest.write_text('{}')
    monkeypatch.setattr(bound, 'BUNDLED_MANIFEST', manifest)
    with pytest.raises(bound.IsolatedRuntimeError, match='BUNDLED_MANIFEST_HASH'):
        bound.verified_runtime()
