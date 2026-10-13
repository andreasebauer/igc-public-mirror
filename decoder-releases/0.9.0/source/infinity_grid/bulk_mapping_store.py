"""Decoder-owned atomic exact-row store with compressed full-witness deduplication."""
from pathlib import Path
import hashlib
import json
import sqlite3
import zlib
from .structural_encoding import structural_canonical_bytes as encode
from .canon import canonical_sha256
from .records import utc_now

SCHEMA = 'IG_EXACT_BULK_MAPPING_SQLITE_V1'

def _parsed_json_bytes(value):
    # Internal values originate in json.loads: the parser already restricts
    # container/key/value types. allow_nan=False still rejects nonfinite numbers.
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


class BulkMappingError(RuntimeError):
    pass


def witness_bytes(compressed, size, digest):
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(bytes(compressed), int(size) + 1)
    except zlib.error as exc:
        raise BulkMappingError('WITNESS_BYTES_INVALID') from exc
    if len(raw) != size or not decoder.eof or decoder.unused_data or hashlib.sha256(raw).hexdigest() != digest:
        raise BulkMappingError('WITNESS_BYTES_INVALID')
    value = json.loads(raw)
    if _parsed_json_bytes(value) != raw:
        raise BulkMappingError('WITNESS_ENCODING_INVALID')
    return raw


class MappingStore:
    def __init__(self, path, scope, tasks):
        self.path = Path(path)
        self.conn = sqlite3.connect(path, timeout=5)
        self._witness_cache = {}
        try:
            from .v05_stage_runtime import _configure_task_database
            _configure_task_database(self.conn)
            self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS witnesses (digest TEXT PRIMARY KEY,raw_size INTEGER NOT NULL,compressed BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS rows (ordinal INTEGER PRIMARY KEY,task_id TEXT NOT NULL,core BLOB NOT NULL,witness_digest TEXT NOT NULL REFERENCES witnesses(digest));
            CREATE INDEX IF NOT EXISTS rows_task_ordinal ON rows(task_id,ordinal);
            CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY,payload_sha256 TEXT NOT NULL,result_sha256 TEXT NOT NULL,row_count INTEGER NOT NULL,metrics TEXT NOT NULL,committed_utc TEXT NOT NULL);
            ''')
            self.tasks = {t.task_id:t for t in tasks}
            if len(self.tasks) != len(tasks):
                raise BulkMappingError('DUPLICATE_TASK')
            ordinals=[]
            for task in tasks:
                ordinals.extend(x['ordinal'] for x in task.payload['descriptors'])
            if any(type(x) is not int or x < 0 for x in ordinals) or len(set(ordinals)) != len(ordinals):
                raise BulkMappingError('DUPLICATE_OR_INVALID_ORDINAL')
            self.ordinals=sorted(ordinals)
            raw=encode(dict(scope,schema_id=SCHEMA,task_manifest=[{'task_id':t.task_id,'binding_sha256':t.binding_sha256,'payload_sha256':canonical_sha256(t.payload)} for t in sorted(tasks,key=lambda t:t.task_id)] )).decode()
            old=self.conn.execute("SELECT value FROM meta WHERE key='scope'").fetchone()
            if old is not None and old[0]!=raw:
                raise BulkMappingError('DURABLE_MAPPING_SCOPE_MISMATCH')
            if old is None:self.conn.execute("INSERT INTO meta VALUES ('scope',?)",(raw,))
            self.conn.commit()
            orphan = self.conn.execute('SELECT COUNT(*) FROM rows r LEFT JOIN tasks t ON r.task_id=t.task_id LEFT JOIN witnesses w ON r.witness_digest=w.digest WHERE t.task_id IS NULL OR w.digest IS NULL').fetchone()[0]
            if orphan:
                raise BulkMappingError('DURABLE_MAPPING_ORPHAN_ROW')
            for tid,sha in self.conn.execute('SELECT task_id,payload_sha256 FROM tasks'):
                if tid not in self.tasks or canonical_sha256(self.tasks[tid].payload)!=sha:
                    raise BulkMappingError('DURABLE_MAPPING_TASK_MISMATCH')
                self.verify_task(tid)
        except BaseException:
            self.conn.close()
            raise

    def close(self):
        self.conn.close()

    def pending(self):
        done={x[0] for x in self.conn.execute('SELECT task_id FROM tasks')}
        return [t for tid,t in sorted(self.tasks.items()) if tid not in done]

    def iter_rows(self, task_id=None):
        yield from iter_connection_rows(self.conn, task_id, witness_cache=self._witness_cache)

    def verify_task(self, task_id):
        saved=self.conn.execute('SELECT result_sha256,row_count FROM tasks WHERE task_id=?',(task_id,)).fetchone()
        rows=list(iter_encoded_rows(self.conn, task_id, witness_cache=self._witness_cache))
        digest = batch_digest([raw for _, raw in rows])
        if saved is None or len(rows)!=saved[1] or digest!=saved[0]:
            raise BulkMappingError('DURABLE_MAPPING_RESULT_MISMATCH')
        if [ordinal for ordinal, _ in rows]!=[x['ordinal'] for x in self.tasks[task_id].payload['descriptors']]:
            raise BulkMappingError('DURABLE_MAPPING_COVERAGE_MISMATCH')

    def commit(self, task_id, result):
        raw=bytes(result['signature_canonical_bytes'])
        signature=json.loads(raw)
        if _parsed_json_bytes(signature)!=raw or hashlib.sha256(raw).hexdigest()!=result['signature_sha256'] or result.get('engineering_digest_override'):
            raise BulkMappingError('MAPPING_RESULT_BYTES_INVALID')
        if signature.get('schema')!='IG_HET4_FULL_WITNESS_BATCH_V1':
            raise BulkMappingError('MAPPING_RESULT_SCHEMA')
        rows=signature['rows'];expected=self.tasks[task_id].payload['descriptors']
        if [r['ordinal'] for r in rows]!=[r['ordinal'] for r in expected] or len(rows)!=result['outcome_count']:
            raise BulkMappingError('MAPPING_TASK_COVERAGE_MISMATCH')
        for row,descriptor in zip(rows,expected):
            if encode(row['raw_edge'])!=encode(descriptor['edge']) or row.get('literal_descriptor_sha256')!=descriptor.get('literal_descriptor_sha256'):
                raise BulkMappingError('MAPPING_DESCRIPTOR_MISMATCH')
        if self.conn.execute('SELECT 1 FROM tasks WHERE task_id=?',(task_id,)).fetchone():
            self.verify_task(task_id)
            if batch_bytes([value for _, value in iter_encoded_rows(self.conn, task_id, witness_cache=self._witness_cache)])!=raw:
                raise BulkMappingError('MAPPING_TASK_RETRY_CHANGED')
            return
        with self.conn:
            for row in rows:
                core=dict(row);full=core.pop('relabeling_witness');wraw=_parsed_json_bytes(full);digest=hashlib.sha256(wraw).hexdigest()
                if core['relabeling_witness_sha256']!=digest:raise BulkMappingError('WITNESS_IDENTIFIER_MISMATCH')
                old=self.conn.execute('SELECT raw_size,compressed FROM witnesses WHERE digest=?',(digest,)).fetchone()
                if old is not None:
                    if checked_witness(old[1],old[0],digest,self._witness_cache)!=wraw:raise BulkMappingError('WITNESS_HASH_COLLISION')
                else:self.conn.execute('INSERT INTO witnesses VALUES (?,?,?)',(digest,len(wraw),sqlite3.Binary(zlib.compress(wraw))))
                self.conn.execute('INSERT INTO rows VALUES (?,?,?,?)',(row['ordinal'],task_id,sqlite3.Binary(encode(core)),digest))
            self.conn.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?)',(task_id,canonical_sha256(self.tasks[task_id].payload),hashlib.sha256(raw).hexdigest(),len(rows),json.dumps(result.get('metrics',{}),sort_keys=True),utc_now()))

    def summary(self):
        observed=[r[0] for r in self.conn.execute('SELECT ordinal FROM rows ORDER BY ordinal')]
        if observed!=self.ordinals:raise BulkMappingError('MAPPING_WINDOW_COVERAGE_MISMATCH')
        for tid in self.tasks:self.verify_task(tid)
        h=hashlib.sha256()
        for _, raw in iter_encoded_rows(self.conn, witness_cache=self._witness_cache):h.update(raw);h.update(b'\n')
        return {'schema_id':SCHEMA,'rows':len(observed),'tasks':len(self.tasks),'full_rows_stream_sha256':h.hexdigest(),'witness_objects':self.conn.execute('SELECT COUNT(*) FROM witnesses').fetchone()[0],'witness_raw_bytes':self.conn.execute('SELECT COALESCE(SUM(raw_size),0) FROM witnesses').fetchone()[0],'witness_compressed_bytes':self.conn.execute('SELECT COALESCE(SUM(LENGTH(compressed)),0) FROM witnesses').fetchone()[0]}


def checked_witness(compressed, size, digest, cache):
    compressed = bytes(compressed)
    previous = cache.get(digest)
    if previous is not None and previous[:2] == (size, compressed):
        return previous[2]
    raw = witness_bytes(compressed, size, digest)
    # Bounded to one witness by default: eviction changes only speed.
    cache.clear()
    cache[digest] = (size, compressed, raw)
    return raw


def batch_bytes(rows):
    return b'{"rows":[' + b','.join(rows) + b'],"schema":"IG_HET4_FULL_WITNESS_BATCH_V1"}'


def batch_digest(rows):
    return hashlib.sha256(batch_bytes(rows)).hexdigest()


def _joined_rows(conn, task_id):
    where='' if task_id is None else ' WHERE r.task_id=?'
    params=() if task_id is None else (task_id,)
    sql='SELECT r.ordinal,r.core,r.witness_digest,w.raw_size,w.compressed FROM rows r JOIN witnesses w ON r.witness_digest=w.digest'+where+' ORDER BY r.ordinal'
    yield from conn.execute(sql,params)


def iter_encoded_rows(conn, task_id=None, *, witness_cache=None):
    cache={} if witness_cache is None else witness_cache
    for ordinal,core,digest,size,compressed in _joined_rows(conn, task_id):
        raw=checked_witness(compressed,size,digest,cache)
        row=json.loads(core)
        if encode(row)!=bytes(core) or row.get('ordinal')!=ordinal or row.get('relabeling_witness_sha256')!=digest or 'relabeling_witness' in row:
            raise BulkMappingError('MAPPING_ROW_BYTES_INVALID')
        # Compose canonical JSON from independently validated components.
        # Key ordering/escaping are the same encoder used for the original row.
        keys=sorted([*row,'relabeling_witness'])
        fields=[encode(k)+b':' +(raw if k=='relabeling_witness' else encode(row[k])) for k in keys]
        yield ordinal,b'{' + b','.join(fields) + b'}'


def iter_connection_rows(conn, task_id=None, *, witness_cache=None):
    for _, raw in iter_encoded_rows(conn, task_id, witness_cache=witness_cache):
        yield json.loads(raw)



def read_mapping_rows(path):
    conn=sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)
    try:
        yield from iter_connection_rows(conn)
    finally:
        conn.close()

