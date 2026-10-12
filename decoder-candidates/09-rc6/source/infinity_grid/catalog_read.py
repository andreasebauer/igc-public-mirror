"""Closed snapshots for the existing Catalogue API (not scientific releases).

Publication is an explicit writer operation. Queries resolve a browsing alias
once, or bind an exact manifest supplied by the caller. Neither path opens the
legacy live database or creates a workspace. Original source bytes are retained
inside the snapshot; table paths are logical paths relative to IGPaths.root.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from .storage_catalog import ReadIndexError, _closed_db, _owned_writer, _ref, file_ref
from .storage_schema import canonical_bytes, strict_loads

FORMAT = 'IG_LEGACY_CATALOGUE_SNAPSHOT_V1'
MAX_RECORD_BYTES = 64 * 1024
MAX_INPUT_FILES = 100000
ALIAS = 'CURRENT_CATALOGUE.json'
EXTRA_DDL = '''
CREATE TABLE source_records(logical_path TEXT PRIMARY KEY COLLATE BINARY,
 raw BLOB NOT NULL, raw_sha256 TEXT NOT NULL) WITHOUT ROWID;
CREATE INDEX legacy_edge_out_path ON graph_edges(source_node_id,record_path);
CREATE INDEX legacy_edge_in_path ON graph_edges(target_node_id,record_path);
CREATE INDEX legacy_edge_out_type_path ON graph_edges(source_node_id,edge_type,record_path);
CREATE INDEX legacy_edge_in_type_path ON graph_edges(target_node_id,edge_type,record_path);
'''


def _stamp(path: Path) -> tuple:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _bounded_read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReadIndexError('MISSING_OR_UNSAFE_RECORD:' + str(path))
    if path.stat().st_size > limit:
        raise ReadIndexError('RECORD_BUDGET_EXCEEDED:' + str(path))
    with path.open('rb') as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ReadIndexError('RECORD_BUDGET_EXCEEDED:' + str(path))
    return raw


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _membership(paths) -> list[str]:
    patterns = [(paths.store/'artifacts','*.json'), (paths.store/'datasets','*.json'),
        (paths.store/'protocols','*.json'), (paths.store/'claims','**/*.json'),
        (paths.runs,'*/run.json'), (paths.runs,'*/stages/*/current.json'),
        (paths.releases,'*/release_manifest.json')]
    patterns += [(paths.store/'graph'/name,'*.json') for name in
        ('nodes','edges','recipes','aliases','reconstruction_proofs')]
    names = []
    for root, pattern in patterns:
        for p in root.glob(pattern):
            names.append(p.relative_to(paths.root).as_posix())
            if len(names) > MAX_INPUT_FILES:
                raise ReadIndexError('CATALOGUE_FILE_BUDGET_EXCEEDED')
    return sorted(names)


def rebuild_catalogue(catalogue) -> dict:
    """Build from bounded, stable legacy metadata; retain failed staging trees.

    Source writers must be cooperative (ordinary file timestamps, no mutation
    spoofing). This is a stable captured legacy view, not a transaction across
    arbitrary external writers or full canonical-release closure verification.
    """
    from .catalog import DDL, SCHEMA_VERSION
    paths = catalogue.paths
    with _owned_writer(paths.catalog):
        alias = paths.catalog/ALIAS
        if alias.is_symlink():
            raise ReadIndexError('UNSAFE_ALIAS')
        old_alias = _bounded_read(alias,8192) if alias.exists() else None
        before_members = _membership(paths)
        tmp = Path(tempfile.mkdtemp(prefix='.catalogue-build-',dir=paths.catalog))
        db = tmp/'index.sqlite'
        con = sqlite3.connect(db)
        stamps = {}
        def logical(p):
            p = Path(p)
            rel = p.relative_to(paths.root)
            if '..' in rel.parts:
                raise ReadIndexError('UNSAFE_SOURCE_PATH')
            current = paths.root
            for part in rel.parts:
                current = current/part
                if current.is_symlink():
                    raise ReadIndexError('UNSAFE_SOURCE_SYMLINK')
            return rel.as_posix()
        def read_record(p):
            key = logical(p)
            st = _stamp(p)
            raw = _bounded_read(p,MAX_RECORD_BYTES)
            if _stamp(p) != st:
                raise ReadIndexError('SOURCE_CHANGED_DURING_READ')
            if key in stamps:
                raise ReadIndexError('DUPLICATE_SOURCE_RECORD')
            if len(stamps) >= MAX_INPUT_FILES:
                raise ReadIndexError('CATALOGUE_FILE_BUDGET_EXCEEDED')
            obj = strict_loads(raw,max_bytes=MAX_RECORD_BYTES)
            # Native graph identity rules remain in their existing validators.
            if key.startswith('store/graph/nodes/'):
                from .graph_store import validate_node
                validate_node(obj)
            if key.startswith('store/graph/edges/'):
                from .graph_store import validate_edge
                validate_edge(obj)
            if key.startswith('store/graph/recipes/'):
                from .graph_store import validate_recipe
                validate_recipe(obj)
            con.execute('INSERT INTO source_records VALUES(?,?,?)',
                        (key,raw,hashlib.sha256(raw).hexdigest()))
            stamps[key] = st
            return obj
        try:
            con.executescript(DDL.replace('journal_mode=WAL','journal_mode=DELETE')+EXTRA_DDL)
            con.execute('PRAGMA synchronous=FULL')
            con.execute('PRAGMA trusted_schema=OFF')
            con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,4*MAX_RECORD_BYTES)
            counts = catalogue._populate_snapshot(con,read_record,logical)
            # Original code used INSERT OR REPLACE; the candidate's projection
            # uses INSERT so duplicate native keys fail instead of disappearing.
            for table,count in counts.items():
                if con.execute('SELECT count(*) FROM '+table).fetchone()[0] != count:
                    raise ReadIndexError('CATALOGUE_CARDINALITY_MISMATCH:'+table)
            h = hashlib.sha256()
            for key,raw,digest in con.execute('SELECT * FROM source_records ORDER BY logical_path'):
                encoded = key.encode('utf-8')
                h.update(len(encoded).to_bytes(8,'big')); h.update(encoded)
                h.update(len(raw).to_bytes(8,'big')); h.update(bytes.fromhex(digest))
            binding = {'format':FORMAT,'schema_version':SCHEMA_VERSION,
                'input_inventory':{'format':'LOGICAL_PATH_RAW_SEQUENCE_V1',
                    'sha256':h.hexdigest(),'file_count':str(len(stamps))},
                'counts':{k:str(v) for k,v in counts.items()},
                'scope':'CAPTURED_LEGACY_METADATA_ONLY',
                'scientific_acceptance':'NOT_GRANTED',
                'record_path_base':'IGPaths.root',
                'resource_profile':{'max_record_bytes':MAX_RECORD_BYTES,'max_input_files':MAX_INPUT_FILES}}
            con.execute("INSERT OR REPLACE INTO meta VALUES('schema_version',?)",(str(SCHEMA_VERSION),))
            con.execute("INSERT INTO meta VALUES('snapshot_binding',?)",(canonical_bytes(binding).decode(),))
            con.commit()
            if con.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                raise ReadIndexError('CATALOGUE_INTEGRITY_FAILED')
            con.close(); con = None
            _closed_db(db)
            if _membership(paths) != before_members:
                raise ReadIndexError('SOURCE_MEMBERSHIP_CHANGED')
            for key,st in stamps.items():
                if _stamp(paths.root/key) != st:
                    raise ReadIndexError('SOURCE_CHANGED_BEFORE_SEAL')
            manifest = dict(binding,index_ref=file_ref(db))
            (tmp/'INDEX.json').write_bytes(canonical_bytes(manifest))
            for p in tmp.iterdir():
                with p.open('rb') as f: os.fsync(f.fileno())
                p.chmod(0o444)
            _sync_dir(tmp)
            manifest_ref = file_ref(tmp/'INDEX.json')
            snapshots = paths.catalog/'snapshots'; snapshots.mkdir(exist_ok=True)
            if snapshots.is_symlink(): raise ReadIndexError('UNSAFE_SNAPSHOT_DIRECTORY')
            dest = snapshots/manifest_ref['sha256']
            if dest.exists():
                # Verify byte-identical reuse; retain redundant staging for later
                # authorized retention cleanup, rather than deleting evidence.
                if dest.is_symlink() or file_ref(dest/'INDEX.json') != manifest_ref or file_ref(dest/'index.sqlite') != manifest['index_ref']:
                    raise ReadIndexError('SNAPSHOT_IDENTITY_CONFLICT')
            else:
                os.rename(tmp,dest); _sync_dir(snapshots)
            if (_bounded_read(alias,8192) if alias.exists() else None) != old_alias:
                raise ReadIndexError('ALIAS_CONFLICT')
            fd,name = tempfile.mkstemp(prefix='.catalogue-alias-',dir=paths.catalog)
            with os.fdopen(fd,'wb') as f:
                f.write(canonical_bytes(manifest_ref)); f.flush(); os.fsync(f.fileno())
            os.replace(name,alias); _sync_dir(paths.catalog)
            return {'status':'PASS','counts':counts,'db':str(dest/'index.sqlite'),
                'schema_version':SCHEMA_VERSION,'snapshot_ref':manifest_ref,
                'input_inventory':binding['input_inventory'],'scientific_acceptance':'NOT_GRANTED'}
        except Exception:
            if con is not None: con.close()
            raise


class CatalogueReader:
    """A pinned, closed legacy view. No writer constructors are called."""
    def __init__(self,paths,*,snapshot_ref=None):
        self.paths = paths
        if snapshot_ref is None:
            alias = paths.catalog/ALIAS
            if not alias.exists(): raise ReadIndexError('CATALOGUE_SNAPSHOT_REQUIRED')
            snapshot_ref = strict_loads(_bounded_read(alias,8192),max_bytes=8192)
        self.snapshot_ref = dict(_ref(snapshot_ref))
        parent = paths.catalog/'snapshots'
        self.directory = parent/snapshot_ref['sha256']
        if paths.catalog.is_symlink() or parent.is_symlink() or self.directory.is_symlink():
            raise ReadIndexError('UNSAFE_SNAPSHOT_DIRECTORY')
        raw = _bounded_read(self.directory/'INDEX.json',65536)
        if {'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))} != snapshot_ref:
            raise ReadIndexError('CATALOGUE_MANIFEST_MISMATCH')
        self.manifest = strict_loads(raw,max_bytes=65536)
        from .catalog import SCHEMA_VERSION
        if self.manifest.get('format') != FORMAT or self.manifest.get('schema_version') != SCHEMA_VERSION:
            raise ReadIndexError('UNSUPPORTED_CATALOGUE_SCHEMA')
        self.db = self.directory/'index.sqlite'
        _closed_db(self.db); before = _stamp(self.db)
        if file_ref(self.db) != self.manifest['index_ref']:
            raise ReadIndexError('CATALOGUE_BYTES_MISMATCH')
        self._con = sqlite3.connect(self.db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            self._con.execute('PRAGMA query_only=ON'); self._con.execute('PRAGMA trusted_schema=OFF')
            self._con.enable_load_extension(False)
            self._con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,4*MAX_RECORD_BYTES)
            self._con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH,8192)
            binding = {k:v for k,v in self.manifest.items() if k!='index_ref'}
            if self._con.execute("SELECT value FROM meta WHERE key='snapshot_binding'").fetchone() != (canonical_bytes(binding).decode(),):
                raise ReadIndexError('CATALOGUE_BINDING_MISMATCH')
            self._stat = _stamp(self.db)
            if before != self._stat: raise ReadIndexError('CATALOGUE_CHANGED_DURING_OPEN')
            self._con.row_factory = sqlite3.Row
            self._con.set_authorizer(self._authorize)
        except Exception:
            self._con.close(); raise

    @staticmethod
    def _authorize(action,a,b,db,trigger):
        if action in (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ): return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (b or '').lower() in {
                'count','min','max','sum','avg','total','length','typeof','hex','coalesce','ifnull'}:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def close(self): self._con.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def _unchanged(self):
        if self.db.is_symlink() or _stamp(self.db) != self._stat:
            raise ReadIndexError('CATALOGUE_CHANGED')
        _closed_db(self.db)

    def query(self,sql,args=(),*,max_rows=1024,max_bytes=1048576,max_vm_steps=500000):
        if type(max_rows) is not int or not 1 <= max_rows <= 4096: raise ReadIndexError('QUERY_ROW_BUDGET')
        if type(max_bytes) is not int or not 1 <= max_bytes <= 16*1024*1024: raise ReadIndexError('QUERY_BYTE_BUDGET')
        if type(max_vm_steps) is not int or not 1000 <= max_vm_steps <= 10000000: raise ReadIndexError('QUERY_WORK_BUDGET')
        if not isinstance(sql,str) or len(sql.encode('utf-8'))>8192: raise ReadIndexError('QUERY_SQL_BUDGET')
        self._unchanged(); ticks = 0; cursor = None
        def progress():
            nonlocal ticks
            ticks += 1000
            return int(ticks >= max_vm_steps)
        self._con.set_progress_handler(progress,1000)
        rows=[]; size=0
        try:
            cursor=self._con.execute(sql,args)
            if cursor.description is None: raise ReadIndexError('READ_QUERY_REQUIRED')
            columns=[c[0] for c in cursor.description]
            if len(columns)!=len(set(columns)): raise ReadIndexError('DUPLICATE_QUERY_COLUMNS')
            for row in cursor:
                size += sum(len(x) if isinstance(x,bytes) else len(str(x).encode('utf-8')) for x in row)
                if len(rows)>=max_rows: raise ReadIndexError('QUERY_ROW_BUDGET_EXCEEDED_NO_PARTIAL_RESULT')
                if size>max_bytes: raise ReadIndexError('QUERY_BYTE_BUDGET_EXCEEDED_NO_PARTIAL_RESULT')
                rows.append(dict(row))
        except sqlite3.DatabaseError as exc:
            raise ReadIndexError('QUERY_REFUSED_OR_WORK_BUDGET:'+str(exc)) from exc
        finally:
            if cursor is not None: cursor.close()
            self._con.set_progress_handler(None,0)
        self._unchanged()
        return rows

    def graph_page(self,node_id,*,direction,edge_types=None,after=None,limit=100,max_bytes=1048576,max_vm_steps=500000):
        if direction not in ('OUT','IN'): raise ReadIndexError('GRAPH_DIRECTION')
        if not isinstance(node_id,str) or len(node_id.encode('utf-8'))>4096: raise ReadIndexError('GRAPH_NODE_FILTER')
        if type(limit) is not int or not 1<=limit<=1024: raise ReadIndexError('PAGE_ROW_BUDGET')
        from .graph_store import EDGE_TYPES,validate_edge
        types=sorted(set(edge_types)) if edge_types is not None else None
        if types is not None and any(t not in EDGE_TYPES for t in types): raise ReadIndexError('GRAPH_TYPE_FILTER')
        query={'node_id':node_id,'direction':direction,'edge_types':types}
        last=''
        if after is not None:
            try:
                if not isinstance(after,str) or len(after)>16384: raise ValueError()
                cursor=strict_loads(base64.b64decode(after,altchars=b'-_',validate=True),max_bytes=12000)
                if set(cursor)!={'snapshot_ref','query','last_path'} or cursor['snapshot_ref']!=self.snapshot_ref or cursor['query']!=query:
                    raise ValueError()
                last=cursor['last_path']
                if not isinstance(last,str) or len(last)>8192: raise ValueError()
            except Exception as exc: raise ReadIndexError('CURSOR_BINDING_MISMATCH') from exc
        column='source_node_id' if direction=='OUT' else 'target_node_id'
        sql='SELECT g.record_path,s.raw,s.raw_sha256 FROM graph_edges AS g JOIN source_records AS s ON s.logical_path=g.record_path WHERE g.'+column+'=? AND g.record_path>?'
        params=[node_id,last]
        if types is not None:
            sql+=' AND g.edge_type IN ('+','.join('?' for _ in types)+')'; params+=types
        sql+=' ORDER BY g.record_path LIMIT ?';params.append(limit+1)
        found=self.query(sql,params,max_rows=limit+1,max_bytes=max_bytes,max_vm_steps=max_vm_steps)
        more=len(found)>limit; rows=[]
        for row in found[:limit]:
            raw=row['raw']
            if hashlib.sha256(raw).hexdigest()!=row['raw_sha256']: raise ReadIndexError('SOURCE_RECORD_HASH_MISMATCH')
            edge=strict_loads(raw,max_bytes=MAX_RECORD_BYTES); validate_edge(edge)
            rows.append(edge)
        token=base64.urlsafe_b64encode(canonical_bytes({'snapshot_ref':self.snapshot_ref,'query':query,'last_path':found[limit-1]['record_path']})).decode() if more else None
        return {'rows':rows,'truncated':more,'next_cursor':token,'snapshot_ref':self.snapshot_ref,
                'scope':'CAPTURED_LEGACY_METADATA_ONLY','scientific_acceptance':'NOT_GRANTED'}

    def graph_list(self,node_id,*,direction,edge_types=None,max_rows=1024,max_bytes=1048576,max_vm_steps=500000):
        page=self.graph_page(node_id,direction=direction,edge_types=edge_types,limit=max_rows,
            max_bytes=max_bytes,max_vm_steps=max_vm_steps)
        if page['truncated']: raise ReadIndexError('GRAPH_LIST_BUDGET_EXCEEDED_USE_GRAPH_PAGE')
        return page['rows']
