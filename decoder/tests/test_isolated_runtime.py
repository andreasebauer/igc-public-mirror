"""Engineering regressions; execution must be inside a saved native capture."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pytest

from infinity_grid import isolated_runtime as bound


def _small_tree(tmp_path):
    # Synthetic byte-check unit fixture, never an admitted executable runtime.
    root = tmp_path / "synthetic-runtime"
    root.mkdir()
    p = root / "package.py"
    p.write_bytes(b"value = 1\n")
    p.chmod(0o644)
    manifest = {"runtime_files": {"package.py": {
        "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}},
        "host": {}}
    return root, p, manifest


@pytest.mark.parametrize("corruption,reason", [
    ("changed", "BYTES"), ("extra", "FILE_SET"), ("missing", "FILE_SET"),
    ("mode", "MODE"), ("symlink", "SPECIAL_FILE"), ("bytecode", "FILE_SET"),
])
def test_runtime_tree_refuses_mutation(tmp_path, corruption, reason):
    root, p, manifest = _small_tree(tmp_path)
    bound._verify_tree(root, manifest)
    if corruption == "changed":
        p.write_bytes(b"value = 2\n")
    elif corruption == "extra":
        (root / "injected.py").write_bytes(b"pass\n")
    elif corruption == "missing":
        p.unlink()
    elif corruption == "mode":
        p.chmod(0o755)
    elif corruption == "symlink":
        (root / "link.py").symlink_to(p)
    else:
        cache = root / "__pycache__"
        cache.mkdir()
        (cache / "package.cpython-313.pyc").write_bytes(b"cache")
    with pytest.raises(bound.IsolatedRuntimeError, match=reason):
        bound._verify_tree(root, manifest)


def test_bootstrap_refuses_unpinned_manifest(tmp_path, monkeypatch):
    p = tmp_path / "manifest.json"
    p.write_bytes(b"{}")
    monkeypatch.setattr(bound, "MANIFEST", p)
    with pytest.raises(bound.IsolatedRuntimeError, match="MANIFEST_HASH"):
        bound.verified_runtime()


def test_bootstrap_refuses_unbound_interpreter_path(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.write_bytes(b"not a pinned interpreter")
    monkeypatch.setattr(sys, "executable", str(executable))
    with pytest.raises(bound.IsolatedRuntimeError, match="EXECUTABLE_LAYOUT"):
        bound.verified_runtime()


def test_isolated_process_uses_pinned_packages_and_sqlite():
    source = Path(bound.__file__).resolve().parents[1]
    env = bound.subprocess_environment()
    assert "LD_PRELOAD" not in env and "PYTHONPATH" not in env
    code = '''import sys,json
from pathlib import Path
source=Path(sys.argv[1]);sys.path.insert(0,str(source))
from infinity_grid.isolated_runtime import configure_isolated_paths
configure_isolated_paths(source)
from cryptography.fernet import Fernet
cipher=Fernet(Fernet.generate_key());assert cipher.decrypt(cipher.encrypt(b'bound'))==b'bound'
from infinity_grid.submission import capture_record
from infinity_grid.sqlite_attestation import admit_capture_runtime
observation=admit_capture_runtime(source.parent,capture_record(source.parent))
print(json.dumps({'sqlite':observation['sqlite_version'],'paths':sys.path,
'flags':[sys.flags.isolated,sys.flags.no_site,sys.flags.dont_write_bytecode]}))
'''
    done = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", code, str(source)],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["sqlite"] == "3.51.3"
    assert result["flags"] == [1, 1, 1]
    root = bound.verified_runtime()  # Exact post-exec bytes/modes/file set.
    assert result["paths"] == [str(source), str(root / "base/lib/python3.13"),
                               str(root / "base/lib/python3.13/lib-dynload"),
                               str(root / "base/lib/python3.13/dist-packages")]


@pytest.mark.parametrize("case,error", [
    ("flags", "STARTUP_FLAGS"), ("loader", "LOADER_ENVIRONMENT"),
])
def test_isolated_bootstrap_refuses_unsafe_launch(case, error):
    source = Path(bound.__file__).resolve().parents[1]
    env = bound.subprocess_environment()
    flags = ["-B", "-I", "-S"]
    if case == "flags":
        flags.remove("-I")
    else:
        env.pop("LD_LIBRARY_PATH")
    code = ("import sys;sys.dont_write_bytecode=True;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
            "from infinity_grid.isolated_runtime import configure_isolated_paths;"
            "configure_isolated_paths(Path(sys.argv[1]))")
    done = subprocess.run([sys.executable, *flags, "-c", code, str(source)],
                          env=env, capture_output=True, text=True)
    assert done.returncode != 0 and error in done.stderr
