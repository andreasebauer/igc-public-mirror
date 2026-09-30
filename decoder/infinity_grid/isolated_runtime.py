"""Source-pinned V4 bootstrap paths; no ambient package or loader discovery.

This admits one retained runtime layout, not arbitrary Python installations or
an independently qualified host. C6 origin/lease and SQLite checks still apply.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

MANIFEST = Path(__file__).parent / "resources/runtime/V4_BOOTSTRAP_MANIFEST.json"
MANIFEST_SHA256 = "58481bb12202404e0d88b65940ac965359c80492f75c7a6c8182dc19b040af58"
EXECUTABLES = {"python-fixed-host", "base/bin/python3.13", "base/bin/python-engineering3.13"}
INTERPRETERS = {"base/bin/python3.13", "base/bin/python-engineering3.13"}


class IsolatedRuntimeError(RuntimeError):
    pass


def _fail(reason: str) -> None:
    raise IsolatedRuntimeError("ISOLATED_RUNTIME_" + reason)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_tree(root: Path, manifest: dict) -> None:
    """Byte/mode/file-set check; never imports code from the checked tree."""
    if root.is_symlink() or not root.is_dir():
        _fail("ROOT")
    paths = list(root.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
        _fail("SPECIAL_FILE")
    expected = manifest["runtime_files"]
    if {p.relative_to(root).as_posix() for p in paths if p.is_file()} != set(expected):
        _fail("FILE_SET")
    for name, row in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            _fail("MANIFEST_PATH")
        p = root / name
        if p.stat().st_size != row["bytes"] or _sha(p) != row["sha256"]:
            _fail("BYTES:" + name)
        mode = 0o755 if name in EXECUTABLES else 0o644
        if stat.S_IMODE(p.stat().st_mode) != mode:
            _fail("MODE:" + name)
    for name, row in manifest["host"].items():
        p = Path(name)
        if not p.is_file() or p.stat().st_size != row["bytes"] or _sha(p) != row["sha256"]:
            _fail("HOST_BYTES:" + name)


def verified_runtime() -> Path:
    raw = MANIFEST.read_bytes()
    if MANIFEST.is_symlink() or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        _fail("MANIFEST_HASH")
    manifest = json.loads(raw)
    executable = Path(sys.executable).absolute()
    # V4 has a real interpreter at <runtime>/base/bin/<python>, never a link.
    root = executable.parent.parent.parent
    if executable != executable.resolve(strict=True) or root != root.resolve(strict=True):
        _fail("EXECUTABLE_PATH")
    if executable.relative_to(root).as_posix() not in INTERPRETERS:
        _fail("EXECUTABLE_LAYOUT")
    _verify_tree(root, manifest)
    return root


def subprocess_environment() -> dict[str, str]:
    root = verified_runtime()
    # LD_LIBRARY_PATH must be in the environment at exec, before native loading.
    # Never carry caller-provided loader paths or LD_PRELOAD into C6 children.
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1", "LD_LIBRARY_PATH": str(root / "native")}


def configure_isolated_paths(source: Path) -> None:
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        _fail("STARTUP_FLAGS")
    root = verified_runtime()
    if os.environ.get("LD_LIBRARY_PATH") != str(root / "native") or os.environ.get("LD_PRELOAD"):
        _fail("LOADER_ENVIRONMENT")
    stdlib = root / "base/lib/python3.13"
    packages = stdlib / "dist-packages"
    sys.path[:] = [str(source.resolve(strict=True)), str(stdlib),
                   str(stdlib / "lib-dynload"), str(packages)]
