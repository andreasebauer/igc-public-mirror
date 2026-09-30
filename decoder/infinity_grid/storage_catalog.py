"""Closed, immutable, bounded read indexes over exact supplied record bytes.

The index is a REBUILDABLE_PROJECTION. A successful open/build grants no
scientific authority. Canonical collection/closure and seal admission remain
separate. No legacy live workspace is queried, migrated, or auto-rebuilt here.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from typing import Iterable
from .storage_schema import canonical_bytes, strict_loads, validate_record_bytes

INDEX_VERSION = 'IG_STORAGE_READ_INDEX_V1'
MAX_RECORD_BYTES = 64 * 1024
MAX_KEY_BYTES = 4096
DDL = '''
PRAGMA journal_mode=DELETE;
PRAGMA synchronous=FULL;
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE records(record_key TEXT PRIMARY KEY COLLATE BINARY,
 schema_id TEXT NOT NULL, raw BLOB NOT NULL, raw_sha256 TEXT NOT NULL,
 raw_size TEXT NOT NULL) WITHOUT ROWID;
CREATE INDEX records_type_key ON records(schema_id,record_key);
CREATE TABLE native_bindings(kind TEXT NOT NULL, native_ref TEXT NOT NULL,
 record_key TEXT NOT NULL REFERENCES records(record_key),
 PRIMARY KEY(kind,native_ref)) WITHOUT ROWID;
CREATE TABLE incidences(record_key TEXT PRIMARY KEY, endpoint_kind TEXT NOT NULL,
 endpoint_ref TEXT NOT NULL, relation_ref TEXT NOT NULL, direction TEXT NOT NULL);
CREATE INDEX incidence_endpoint_key ON incidences(endpoint_kind,endpoint_ref,record_key);
CREATE INDEX incidence_relation_key ON incidences(relation_ref,record_key);
CREATE TABLE graph_edges(record_key TEXT PRIMARY KEY, edge_id TEXT UNIQUE NOT NULL,
 edge_type TEXT NOT NULL, source_node_id TEXT NOT NULL,target_node_id TEXT NOT NULL);
CREATE INDEX edge_out_key ON graph_edges(source_node_id,record_key);
CREATE INDEX edge_in_key ON graph_edges(target_node_id,record_key);
CREATE INDEX edge_out_type_key ON graph_edges(source_node_id,edge_type,record_key);
CREATE INDEX edge_in_type_key ON graph_edges(target_node_id,edge_type,record_key);
'''

class ReadIndexError(RuntimeError):
    pass

@dataclass(frozen=True)
class IndexRecord:
    key: str
    raw: bytes
    profile: str = 'STORAGE_V1'

def file_ref(path: Path) -> dict:
    h = hashlib.sha256(); size = 0
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk); size += len(chunk)
    return {'sha256': h.hexdigest(), 'size_bytes': str(size)}

def _ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'sha256', 'size_bytes'}:
        raise ReadIndexError('CONTENT_REF_INVALID')
    import re
    if not isinstance(ref['sha256'], str) or re.fullmatch('[0-9a-f]{64}', ref['sha256']) is None:
        raise ReadIndexError('CONTENT_REF_INVALID')
    if not isinstance(ref['size_bytes'], str) or re.fullmatch('0|[1-9][0-9]*', ref['size_bytes']) is None:
        raise ReadIndexError('CONTENT_REF_INVALID')
    return ref

def _key(key):
    if not isinstance(key, str) or not key or len(key.encode('utf-8')) > MAX_KEY_BYTES:
        raise ReadIndexError('RECORD_KEY_BUDGET_OR_TYPE')
    p = PurePosixPath(key)
    if p.is_absolute() or '..' in p.parts or '\\' in key or '\x00' in key or str(p) != key:
        raise ReadIndexError('UNSAFE_LOGICAL_KEY')
    return key

def _feed(h, record):
    rawkey = record.key.encode('utf-8')
    h.update(len(rawkey).to_bytes(8, 'big')); h.update(rawkey)
    h.update(len(record.raw).to_bytes(8, 'big')); h.update(hashlib.sha256(record.raw).digest())
    # Profiles are part of interpretation and therefore part of the inventory.
    profile = record.profile.encode('ascii'); h.update(len(profile).to_bytes(8, 'big')); h.update(profile)

def inventory(records: Iterable[IndexRecord]) -> dict:
    """Streaming input commitment. The caller must freeze the result before build."""
    h = hashlib.sha256(); count = 0; prior = None
    for record in records:
        _key(record.key)
        if type(record.raw) is not bytes or len(record.raw) > MAX_RECORD_BYTES:
            raise ReadIndexError('RECORD_BUDGET_EXCEEDED')
        if prior is not None and record.key <= prior: raise ReadIndexError('KEY_ORDER_OR_DUPLICATE')
        _feed(h, record); prior = record.key; count += 1
    return {'format': 'KEY_RAW_PROFILE_SEQUENCE_V1', 'records_sha256': h.hexdigest(), 'record_count': str(count)}

def _closed_db(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file(): raise ReadIndexError('MISSING_OR_UNSAFE_INDEX')
    for suffix in ('-wal', '-shm', '-journal'):
        if os.path.lexists(str(p) + suffix): raise ReadIndexError('INDEX_NOT_CLOSED')
    with p.open('rb') as f: header = f.read(100)
    if len(header) != 100 or header[:16] != b'SQLite format 3\x00' or header[18:20] != b'\x01\x01':
        raise ReadIndexError('INDEX_NOT_CLOSED_DELETE_DATABASE')

def _native_text(ref):
    if not isinstance(ref, dict) or set(ref) != {'native_id', 'scope_ref', 'profile_ref'}:
        raise ReadIndexError('NATIVE_REF_INVALID')
    if not isinstance(ref['native_id'], str) or not ref['native_id']:
        raise ReadIndexError('NATIVE_REF_INVALID')
    _ref(ref['scope_ref']); _ref(ref['profile_ref'])
    text = canonical_bytes(ref).decode('utf-8')
    if len(text.encode('utf-8')) > MAX_KEY_BYTES: raise ReadIndexError('NATIVE_REF_BUDGET')
    return text

def _project(con, record):
    if record.profile == 'STORAGE_V1':
        obj = validate_record_bytes(record.raw, max_bytes=MAX_RECORD_BYTES)
        sid = obj['schema_id']
        kinds = {'IG_STORAGE_OBJECT_V1': ('OBJECT', 'object_ref'),
                 'IG_STORAGE_OCCURRENCE_V1': ('OCCURRENCE', 'occurrence_ref'),
                 'IG_STORAGE_RELATION_V1': ('RELATION', 'relation_ref'),
                 'IG_STORAGE_INCIDENCE_V1': ('INCIDENCE', 'incidence_ref')}
        if sid in kinds:
            kind, field = kinds[sid]
            con.execute('INSERT INTO native_bindings VALUES(?,?,?)',
                        (kind, _native_text(obj[field]), record.key))
        if sid == 'IG_STORAGE_INCIDENCE_V1':
            con.execute('INSERT INTO incidences VALUES(?,?,?,?,?)',
                        (record.key, obj['endpoint']['kind'], _native_text(obj['endpoint']['ref']),
                         _native_text(obj['relation_ref']), obj['direction']))
    elif record.profile == 'LEGACY_REFERENCE_RECORD_V1':
        from .storage_native_records import native_record
        obj,key=native_record(record.raw,max_bytes=MAX_RECORD_BYTES)
        if record.key!=key:raise ReadIndexError('NATIVE_INDEX_KEY_MISMATCH')
        sid=obj['schema_id']
    elif record.profile == 'LEGACY_GRAPH_EDGE_V1':
        from .graph_store import validate_edge
        obj = strict_loads(record.raw, max_bytes=MAX_RECORD_BYTES); validate_edge(obj)
        sid = obj['schema_id']
        con.execute('INSERT INTO graph_edges VALUES(?,?,?,?,?)',
                    (record.key, obj['edge_id'], obj['edge_type'], obj['source_node_id'], obj['target_node_id']))
    else:
        raise ReadIndexError('UNSUPPORTED_RECORD_PROFILE')
    con.execute('INSERT INTO records VALUES(?,?,?,?,?)',
                (record.key, sid, record.raw, hashlib.sha256(record.raw).hexdigest(), str(len(record.raw))))

@contextmanager
def _owned_writer(parent):
    """Writers coordinate only with other explicit writers; readers create no lock."""
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / '.index-writer.lock'
    if path.is_symlink(): raise ReadIndexError('UNSAFE_WRITER_LOCK')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)

def build_index(records: Iterable[IndexRecord], destination: Path, *,
                release_root: dict, source_collections: list[dict],
                expected_inventory: dict) -> dict:
    """Build/validate a NEW snapshot. Existing destinations are never destroyed.

    This bounded engineering binding covers supplied records only, not the full
    ReleaseRoot closure. The eventual publisher must establish that separately.
    """
    _ref(release_root)
    if not source_collections: raise ReadIndexError('SOURCE_COLLECTIONS_REQUIRED')
    for ref in source_collections: _ref(ref)
    dest = Path(destination)
    if dest.is_symlink(): raise ReadIndexError('UNSAFE_DESTINATION')
    binding = {'release_root': release_root, 'source_collections': source_collections,
               'input_inventory': expected_inventory}
    with _owned_writer(dest.parent):
        if dest.exists(): raise ReadIndexError('INDEX_DESTINATION_EXISTS')
        tmp = Path(tempfile.mkdtemp(prefix='.index-build-', dir=dest.parent))
        db = tmp / 'index.sqlite'; con = None
        try:
            con = sqlite3.connect(db)
            con.execute('PRAGMA trusted_schema=OFF')
            con.executescript(DDL)
            con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4 * MAX_RECORD_BYTES)
            h = hashlib.sha256(); count = 0; prior = None
            for record in records:
                _key(record.key)
                if type(record.raw) is not bytes or len(record.raw) > MAX_RECORD_BYTES:
                    raise ReadIndexError('RECORD_BUDGET_EXCEEDED')
                if prior is not None and record.key <= prior: raise ReadIndexError('KEY_ORDER_OR_DUPLICATE')
                _project(con, record); _feed(h, record); prior = record.key; count += 1
            actual = {'format': 'KEY_RAW_PROFILE_SEQUENCE_V1', 'records_sha256': h.hexdigest(), 'record_count': str(count)}
            if actual != expected_inventory: raise ReadIndexError('INPUT_INVENTORY_MISMATCH')
            con.execute('INSERT INTO meta VALUES(?,?)', ('index_version', INDEX_VERSION))
            con.execute('INSERT INTO meta VALUES(?,?)', ('binding', canonical_bytes(binding).decode()))
            con.commit()
            if con.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                raise ReadIndexError('INDEX_INTEGRITY_FAILED')
            con.close(); con = None
            _closed_db(db)
            manifest = {'format': INDEX_VERSION, 'binding': binding, 'index_ref': file_ref(db),
                        'authority': 'REBUILDABLE_PROJECTION',
                        'verification_scope': 'SUPPLIED_RECORD_BYTES_AND_INDEX_INTEGRITY',
                        'scientific_acceptance': 'NOT_GRANTED', 'record_bytes_max': str(MAX_RECORD_BYTES)}
            (tmp / 'INDEX.json').write_bytes(canonical_bytes(manifest))
            for p in (db, tmp / 'INDEX.json'):
                with p.open('rb') as f: os.fsync(f.fileno())
                p.chmod(0o444)
            fd = os.open(tmp, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
            # Explicit writer lock guards existence and rename. A failed build
            # leaves any prior snapshot/alias untouched.
            os.rename(tmp, dest)
            fd = os.open(dest.parent, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
            return {'status': 'PASS', 'directory': str(dest), 'manifest_ref': file_ref(dest/'INDEX.json'), **manifest}
        except Exception:
            if con is not None: con.close()
            # Preserve the failed staging directory for inspection/recovery.
            raise

def switch_alias(alias: Path, *, expected: dict | None, target: dict) -> None:
    """Owned compare-and-swap of a convenience alias, never scientific promotion."""
    _ref(target); p = Path(alias)
    with _owned_writer(p.parent):
        if p.is_symlink(): raise ReadIndexError('UNSAFE_ALIAS')
        current = strict_loads(p.read_bytes(), max_bytes=8192) if p.exists() else None
        if current != expected: raise ReadIndexError('ALIAS_CONFLICT')
        fd, name = tempfile.mkstemp(prefix='.alias-', dir=p.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(canonical_bytes(target)); f.flush(); os.fsync(f.fileno())
            os.replace(name, p)
            fd = os.open(p.parent, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        finally: Path(name).unlink(missing_ok=True)

class ReadIndex:
    def __init__(self, directory: Path, *, expected_manifest_ref: dict, release_root: dict):
        self.directory = Path(directory); self.db = self.directory / 'index.sqlite'
        if self.directory.is_symlink() or (self.directory/'INDEX.json').is_symlink():
            raise ReadIndexError('UNSAFE_INDEX_LOCATION')
        if file_ref(self.directory/'INDEX.json') != _ref(expected_manifest_ref):
            raise ReadIndexError('INDEX_MANIFEST_MISMATCH')
        m = strict_loads((self.directory/'INDEX.json').read_bytes(), max_bytes=65536)
        if m.get('format') != INDEX_VERSION: raise ReadIndexError('UNSUPPORTED_INDEX_SCHEMA')
        if m['binding']['release_root'] != _ref(release_root): raise ReadIndexError('STALE_INDEX')
        _closed_db(self.db)
        before = self.db.stat()
        if file_ref(self.db) != m['index_ref']: raise ReadIndexError('INDEX_BYTES_MISMATCH')
        self._con = sqlite3.connect(self.db.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
        self._con.execute('PRAGMA query_only=ON'); self._con.execute('PRAGMA trusted_schema=OFF')
        self._con.enable_load_extension(False)
        self._con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4 * MAX_RECORD_BYTES)
        self._con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 8192)
        try:
            if self._con.execute("SELECT value FROM meta WHERE key='binding'").fetchone() != (canonical_bytes(m['binding']).decode(),):
                raise ReadIndexError('INDEX_BINDING_MISMATCH')
            if self._con.execute("SELECT value FROM meta WHERE key='index_version'").fetchone() != (INDEX_VERSION,):
                raise ReadIndexError('UNSUPPORTED_INDEX_SCHEMA')
            after = self.db.stat()
            self._stat = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if self._stat != (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns):
                raise ReadIndexError('INDEX_CHANGED_DURING_OPEN')
        except Exception:
            self._con.close(); raise
        self.manifest = m; self._manifest_ref = expected_manifest_ref
        self._con.set_authorizer(lambda action, a, b, db, source:
            sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ) else sqlite3.SQLITE_DENY)

    def close(self): self._con.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _unchanged(self):
        st = self.db.stat()
        if self._stat != (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns):
            raise ReadIndexError('INDEX_CHANGED')

    def _page(self, query: dict, sql: str, params: tuple, *, after: str | None,
              limit: int, max_bytes: int, max_vm_steps: int) -> dict:
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ReadIndexError('PAGE_ROW_BUDGET')
        if type(max_bytes) is not int or not 1 <= max_bytes <= 16*1024*1024:
            raise ReadIndexError('PAGE_BYTE_BUDGET')
        if type(max_vm_steps) is not int or not 1000 <= max_vm_steps <= 10000000:
            raise ReadIndexError('QUERY_WORK_BUDGET')
        self._unchanged(); last = ''
        if after is not None:
            try:
                if not isinstance(after, str) or len(after) > 16384: raise ValueError()
                cur = strict_loads(base64.b64decode(after, altchars=b'-_', validate=True), max_bytes=12000)
                if set(cur) != {'index_ref','query','last_key'} or cur['index_ref'] != self._manifest_ref or cur['query'] != query:
                    raise ValueError()
                last = _key(cur['last_key'])
            except Exception as exc: raise ReadIndexError('CURSOR_BINDING_MISMATCH') from exc
        ticks = 0
        def progress():
            nonlocal ticks
            ticks += 1000
            return int(ticks >= max_vm_steps)
        self._con.set_progress_handler(progress, 1000)
        rows = []; total = 0; truncated = False
        cursor = None
        try:
            cursor = self._con.execute(sql, (*params, last, limit + 1))
            for key, raw, digest in cursor:
                if len(rows) == limit or total + len(raw) > max_bytes:
                    if not rows: raise ReadIndexError('ONE_RECORD_EXCEEDS_PAGE_BUDGET')
                    truncated = True; break
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ReadIndexError('INDEX_RECORD_HASH_MISMATCH')
                rows.append({'key': key, 'record': strict_loads(raw, max_bytes=MAX_RECORD_BYTES),
                             'content_ref': {'sha256': digest, 'size_bytes': str(len(raw))}})
                total += len(raw)
        except sqlite3.DatabaseError as exc:
            raise ReadIndexError('QUERY_REFUSED_OR_WORK_BUDGET:' + str(exc)) from exc
        finally:
            if cursor is not None: cursor.close()
            self._con.set_progress_handler(None, 0)
        self._unchanged()
        token = None
        if truncated:
            token = base64.urlsafe_b64encode(canonical_bytes({'index_ref': self._manifest_ref,
                'query': query, 'last_key': rows[-1]['key']})).decode('ascii')
        return {'rows': rows, 'next_cursor': token, 'truncated': truncated,
                'raw_record_bytes': total, 'verification_scope': 'INDEX_RAW_BYTES_AND_RETURNED_RECORDS',
                'scientific_acceptance': 'NOT_GRANTED', 'release_root': self.manifest['binding']['release_root']}

    def records(self, *, schema_id: str, after=None, limit=100, max_bytes=1024*1024, max_vm_steps=500000):
        if not isinstance(schema_id, str) or len(schema_id) > 256: raise ReadIndexError('SCHEMA_FILTER_INVALID')
        return self._page({'op':'records','schema_id':schema_id},
            'SELECT record_key,raw,raw_sha256 FROM records WHERE schema_id=? AND record_key>? ORDER BY record_key LIMIT ?',
            (schema_id,), after=after, limit=limit, max_bytes=max_bytes, max_vm_steps=max_vm_steps)

    def graph_neighbours(self, node_id: str, *, direction: str, edge_type: str | None = None,
                         after=None, limit=100, max_bytes=1024*1024, max_vm_steps=500000):
        if direction not in ('OUT','IN') or not isinstance(node_id, str) or not 1 <= len(node_id) <= MAX_KEY_BYTES:
            raise ReadIndexError('GRAPH_QUERY_INVALID')
        if edge_type is not None and (not isinstance(edge_type,str) or not 1 <= len(edge_type) <= 256):
            raise ReadIndexError('GRAPH_QUERY_INVALID')
        col = 'source_node_id' if direction == 'OUT' else 'target_node_id'
        params = (node_id,); condition = ''
        if edge_type is not None: condition = ' AND e.edge_type=?'; params += (edge_type,)
        return self._page({'op':'graph_neighbours','node_id':node_id,'direction':direction,'edge_type':edge_type},
            f'SELECT e.record_key,r.raw,r.raw_sha256 FROM graph_edges e JOIN records r USING(record_key) WHERE e.{col}=?'
            +condition+' AND e.record_key>? ORDER BY e.record_key LIMIT ?', params,
            after=after, limit=limit, max_bytes=max_bytes, max_vm_steps=max_vm_steps)

    def incidences(self, *, endpoint: dict | None = None, relation: dict | None = None,
                   after=None, limit=100, max_bytes=1024*1024, max_vm_steps=500000):
        if (endpoint is None) == (relation is None): raise ReadIndexError('ONE_INCIDENCE_FILTER_REQUIRED')
        if endpoint is not None:
            if set(endpoint) != {'kind','ref'} or endpoint['kind'] not in ('OBJECT','OCCURRENCE'):
                raise ReadIndexError('ENDPOINT_INVALID')
            condition = 'i.endpoint_kind=? AND i.endpoint_ref=?'; params=(endpoint['kind'],_native_text(endpoint['ref']))
        else: condition='i.relation_ref=?'; params=(_native_text(relation),)
        return self._page({'op':'incidences','endpoint':endpoint,'relation':relation},
            'SELECT i.record_key,r.raw,r.raw_sha256 FROM incidences i JOIN records r USING(record_key) WHERE '
            +condition+' AND i.record_key>? ORDER BY i.record_key LIMIT ?',params,
            after=after, limit=limit, max_bytes=max_bytes, max_vm_steps=max_vm_steps)
