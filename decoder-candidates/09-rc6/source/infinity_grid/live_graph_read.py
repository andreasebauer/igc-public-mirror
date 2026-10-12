"""Bounded, operation-scoped adjacency over current legacy edge records.

Fresh for every operation; no persistent cache or catalogue alias. Successful
context exit requires the captured directory/file stamps to remain unchanged.
This is an engineering projection, not a canonical scientific release.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import tempfile

from .graph_store import GraphValidationError, validate_edge, _record_filename
from .storage_schema import strict_loads

MAX_EDGES = 100000
MAX_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024


class LiveGraphChanged(GraphValidationError):
    pass


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _inventory(directory, max_edges):
    if directory.is_symlink() or not directory.is_dir():
        raise GraphValidationError('UNSAFE_LIVE_EDGE_DIRECTORY')
    before = _stamp(directory.stat())
    rows = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith('.json'): continue
            st = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode):
                raise GraphValidationError('UNSAFE_LIVE_EDGE_RECORD')
            if st.st_size > MAX_RECORD_BYTES:
                raise GraphValidationError('LIVE_EDGE_RECORD_BUDGET')
            rows.append((entry.name, _stamp(st)))
            if len(rows) > max_edges:
                raise GraphValidationError('LIVE_EDGE_COUNT_BUDGET')
    if _stamp(directory.stat()) != before:
        raise LiveGraphChanged('LIVE_GRAPH_CHANGED_DURING_INVENTORY')
    return before, sorted(rows)


class LiveGraphRead:
    def __init__(self, graph, *, max_edges=MAX_EDGES, max_bytes=MAX_BYTES):
        if type(max_edges) is not int or not 1 <= max_edges <= MAX_EDGES:
            raise GraphValidationError('LIVE_EDGE_COUNT_BUDGET')
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
            raise GraphValidationError('LIVE_EDGE_BYTE_BUDGET')
        self._graph=graph; self._max_edges=max_edges; self._max_bytes=max_bytes
        self._temp=None; self._con=None; self._inventory=None

    def __enter__(self):
        if self._con is not None: raise GraphValidationError('LIVE_READ_ALREADY_OPEN')
        self._inventory=_inventory(self._graph.edges_dir,self._max_edges)
        self._temp=tempfile.TemporaryDirectory(prefix='ig-live-adjacency-')
        try:
            self._con=sqlite3.connect(str(Path(self._temp.name)/'edges.sqlite'))
            self._con.executescript('''
                PRAGMA journal_mode=DELETE;
                PRAGMA trusted_schema=OFF;
                CREATE TABLE edges(k TEXT PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                  source TEXT NOT NULL, target TEXT NOT NULL, kind TEXT NOT NULL, raw BLOB NOT NULL);
                CREATE INDEX live_out ON edges(source,k);
                CREATE INDEX live_in ON edges(target,k);
            ''')
            total=0
            for name,stamp in self._inventory[1]:
                path=self._graph.edges_dir/name
                with path.open('rb') as handle:
                    if _stamp(os.fstat(handle.fileno()))!=stamp:
                        raise LiveGraphChanged('LIVE_GRAPH_CHANGED_DURING_READ')
                    raw=handle.read(MAX_RECORD_BYTES+1)
                    if _stamp(os.fstat(handle.fileno()))!=stamp:
                        raise LiveGraphChanged('LIVE_GRAPH_CHANGED_DURING_READ')
                if len(raw)>MAX_RECORD_BYTES: raise GraphValidationError('LIVE_EDGE_RECORD_BUDGET')
                total+=len(raw)
                if total>self._max_bytes: raise GraphValidationError('LIVE_EDGE_BYTE_BUDGET')
                edge=strict_loads(raw,max_bytes=MAX_RECORD_BYTES); validate_edge(edge)
                if _record_filename(edge['edge_id'])!=name:
                    raise GraphValidationError('LIVE_EDGE_PATH_IDENTITY_MISMATCH')
                self._con.execute('INSERT INTO edges VALUES(?,?,?,?,?,?)',
                    (name,edge['edge_id'],edge['source_node_id'],edge['target_node_id'],edge['edge_type'],raw))
            self._con.commit(); self._con.execute('PRAGMA query_only=ON')
            self._unchanged()
            return self
        except BaseException:
            self.close(); raise

    def _unchanged(self):
        if _inventory(self._graph.edges_dir,self._max_edges)!=self._inventory:
            raise LiveGraphChanged('LIVE_GRAPH_CHANGED_DURING_OPERATION')

    def __exit__(self,typ,value,tb):
        try:
            if typ is None:self._unchanged()
        finally:self.close()

    def close(self):
        if self._con is not None:self._con.close();self._con=None
        if self._temp is not None:self._temp.cleanup();self._temp=None

    @contextmanager
    def live_read(self):
        self._require_open()
        yield self

    def _require_open(self):
        if self._con is None:raise GraphValidationError('LIVE_READ_CLOSED')

    def _edges(self,column=None,node=None,edge_types=None):
        self._require_open()
        sql='SELECT kind,raw FROM edges';args=()
        if column is not None:
            sql+=' WHERE '+column+'=?';args=(node,)
        sql+=' ORDER BY k'
        kinds=None if edge_types is None else set(edge_types)
        out=[]
        for kind,raw in self._con.execute(sql,args):
            if kinds is None or kind in kinds:out.append(strict_loads(raw,max_bytes=MAX_RECORD_BYTES))
        return out

    def outgoing(self,node_id,edge_types=None):return self._edges('source',node_id,edge_types)
    def incoming(self,node_id,edge_types=None):return self._edges('target',node_id,edge_types)
    def list_edges(self):return self._edges()

    def __getattr__(self,name):
        if name not in {'get_node','get_recipe','node_available','resolve','paths'}:
            raise AttributeError(name)
        self._require_open()
        return getattr(self._graph,name)
