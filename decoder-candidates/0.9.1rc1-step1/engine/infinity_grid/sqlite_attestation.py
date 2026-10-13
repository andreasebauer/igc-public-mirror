"""Read-only runtime observation and explicit WAL-build admission.

No version-range guess is an upstream-fix attestation. The allow-list below pins
one reviewed upstream source ID. Other patched builds need a reviewed policy
revision, not an environment variable bypass.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import sys
from pathlib import Path

POLICY_ID = 'IG_SQLITE_WAL_SOURCE_POLICY_V1'
UPSTREAM_FIXED_SOURCE = '2026-03-13 10:38:09 737ae4a34738ffa0c3ff7f9bb18df914dd1cad163f28fd6b6e114a344fe6d618'

class SQLiteAdmissionError(RuntimeError):
    pass

def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def assess_wal_build(version: str, source_id: str) -> dict:
    approved = version == '3.51.3' and source_id == UPSTREAM_FIXED_SOURCE
    return {'policy_id': POLICY_ID,
            'status': 'UPSTREAM_WAL_FIX_SOURCE_VERIFIED' if approved else 'UNAPPROVED_WAL_BUILD',
            'approved_for_wal_fix_gate': approved,
            'scope': 'WAL-reset fix source identity only; not complete runtime qualification',
            'basis': 'Exact upstream version/source ID allow-list',
            'reference': 'https://sqlite.org/releaselog/3_51_3.html'}

def observe_runtime() -> dict:
    import _sqlite3
    con = sqlite3.connect(':memory:')
    try:
        version, source_id = con.execute('SELECT sqlite_version(), sqlite_source_id()').fetchone()
        options = sorted(x[0] for x in con.execute('PRAGMA compile_options'))
        pragmas = {k: con.execute('PRAGMA ' + k).fetchone()[0] for k in
                   ('journal_mode', 'synchronous', 'foreign_keys', 'page_size', 'busy_timeout', 'wal_autocheckpoint')}
    finally: con.close()
    # Linux mapping observations bind the real loaded native library, not merely
    # the Python package name. Paths are operational observations, never IDs.
    libraries = []
    maps = Path('/proc/self/maps')
    if maps.is_file():
        names = set()
        for line in maps.read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith('/') and 'libsqlite3' in parts[5]:
                names.add(parts[5])
        for name in sorted(names):
            p = Path(name)
            libraries.append({'path': name, 'sha256': _digest(p), 'size_bytes': str(p.stat().st_size)})
    packages = {}
    for name in ('pytest', 'cryptography', 'igraph', 'sympy', 'networkx', 'jsonschema', 'referencing'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = 'NOT_INSTALLED'
    executable = Path(sys.executable).resolve(); extension = Path(_sqlite3.__file__).resolve()
    return {'schema_id': 'IG_SQLITE_RUNTIME_OBSERVATION_V1',
            'python_version': platform.python_version(), 'python_implementation': platform.python_implementation(),
            'executable_sha256': _digest(executable), 'sqlite_extension_sha256': _digest(extension),
            'platform': platform.platform(), 'machine': platform.machine(),
            'sqlite_version': version, 'sqlite_source_id': source_id,
            'sqlite_compile_options': options, 'sqlite_thread_safety': sqlite3.threadsafety,
            'loaded_sqlite_libraries': libraries, 'package_versions': packages,
            'observed_in_memory_connection_pragmas': pragmas,
            'pragma_scope': 'A new :memory: connection, not settings of a production WAL database',
            'wal_fix_gate': assess_wal_build(version, source_id),
            'scientific_execution': 'NONE', 'runtime_qualification': 'NOT_GRANTED'}

def require_wal_fix() -> dict:
    observation = observe_runtime()
    if not observation['wal_fix_gate']['approved_for_wal_fix_gate']:
        raise SQLiteAdmissionError('UNAPPROVED_WAL_BUILD:' + observation['sqlite_version'])
    return observation

def verify_runtime_binding(expected: dict) -> dict:
    actual = observe_runtime()
    keys = ('python_version', 'executable_sha256', 'sqlite_extension_sha256', 'sqlite_version',
            'sqlite_source_id', 'sqlite_compile_options', 'package_versions')
    if any(actual.get(k) != expected.get(k) for k in keys):
        raise SQLiteAdmissionError('PINNED_RUNTIME_MISMATCH')
    # Paths can move; library identities cannot.
    def libs(o): return sorted((x['sha256'], x['size_bytes']) for x in o['loaded_sqlite_libraries'])
    if libs(actual) != libs(expected): raise SQLiteAdmissionError('PINNED_SQLITE_LIBRARY_MISMATCH')
    return actual

def admit_capture_runtime(workspace: Path, capture: dict) -> dict:
    """Executed before native authority is granted in this candidate.

    An explicitly captured bounded VALIDATION can check refusal paths on an
    unapproved build. SCRIPT/STAGE cannot use that engineering-only policy.
    Existing completion reuse remains non-executing and does not call this gate.
    """
    entries = [x for x in capture['environment']['artifacts'] if x['logical_name']=='sqlite_runtime_binding']
    if len(entries) != 1: raise SQLiteAdmissionError('SQLITE_RUNTIME_BINDING_REQUIRED')
    digest = entries[0]['sha256']
    p = Path(workspace)/'runtime/intake/artifacts'/(digest+'.bin')
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 65536:
        raise SQLiteAdmissionError('SQLITE_BINDING_MISSING_OR_INVALID')
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest: raise SQLiteAdmissionError('SQLITE_BINDING_HASH_MISMATCH')
    from .storage_schema import strict_loads
    spec = strict_loads(raw, max_bytes=65536)
    if set(spec) != {'schema_id','policy','observation'} or spec['schema_id'] != 'IG_CAPTURE_SQLITE_BINDING_V1':
        raise SQLiteAdmissionError('SQLITE_BINDING_SCHEMA')
    actual = verify_runtime_binding(spec['observation'])
    if spec['policy'] == 'REQUIRE_REVIEWED_WAL_BUILD':
        require_wal_fix()
    elif spec['policy'] == 'ENGINEERING_VALIDATION_ONLY':
        if capture['job']['execution']['kind'] != 'VALIDATION':
            raise SQLiteAdmissionError('ENGINEERING_RUNTIME_POLICY_NOT_FOR_SCIENCE')
    else: raise SQLiteAdmissionError('UNKNOWN_SQLITE_RUNTIME_POLICY')
    return actual
