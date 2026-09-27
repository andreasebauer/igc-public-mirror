"""Separate POSIX worker; request states are transport facts, never science results."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, closing
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import uuid

from .models import AcceptedRequest, Catalog, Settings
from .native import AdapterError, Command, NativeAdapter, canonical_hash, read_json

ACTIVE = ('queued', 'dispatching', 'running', 'needs_reconciliation')


def command_digest(command):
    return canonical_hash(asdict(command))


class Queue:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_absolute() or '..' in self.root.parts:
            raise ValueError('Absolute private queue directory required')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError('Queue symlinks prohibited')
        if self.root.stat().st_mode & 0o077:
            raise ValueError('Queue directory must have mode 0700')
        with closing(self.connect()) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, key TEXT UNIQUE NOT NULL, digest TEXT NOT NULL,
                    operation TEXT NOT NULL, target TEXT NOT NULL, lane TEXT NOT NULL,
                    command TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                    updated REAL NOT NULL, pid INTEGER, process_start TEXT,
                    exit_code INTEGER, classification TEXT, native TEXT, error TEXT);
                CREATE UNIQUE INDEX IF NOT EXISTS active_run ON requests(target)
                WHERE operation='run' AND status IN
                ('queued','dispatching','running','needs_reconciliation');
            ''')
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,capture_id TEXT UNIQUE NOT NULL,
                    registration TEXT NOT NULL,job TEXT NOT NULL,created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events(cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,kind TEXT NOT NULL,detail TEXT NOT NULL,created REAL NOT NULL);
            """)
            columns={r[1] for r in db.execute('PRAGMA table_info(requests)')}
            if 'run_key' not in columns:
                db.execute('ALTER TABLE requests ADD COLUMN run_key TEXT')
                for row in db.execute("SELECT id,command FROM requests WHERE operation='run'").fetchall():
                    db.execute('UPDATE requests SET run_key=? WHERE id=?',
                               (self.run_key(json.loads(row['command'])),row['id']))
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS active_native_run ON requests(run_key) "
                       "WHERE operation='run' AND status IN ('queued','dispatching','running','needs_reconciliation')")
            db.commit()

    @staticmethod
    def run_key(command):
        argv=command['argv']
        return canonical_hash(argv[-2:] if len(argv)>=2 else [command['target']])

    @staticmethod
    def event(db,rid,kind,detail):
        db.execute('INSERT INTO events(request_id,kind,detail,created) VALUES (?,?,?,?)',
                   (rid,kind,json.dumps(detail),time.time()))

    def events(self,rid,cursor=0,limit=100):
        self.get(rid)
        with closing(self.connect()) as db:
            rows=db.execute('SELECT * FROM events WHERE request_id=? AND cursor>? ORDER BY cursor LIMIT ?',
                            (rid,cursor,limit)).fetchall()
        return {'items':[dict(r,detail=json.loads(r['detail'])) for r in rows],
                'next_cursor':rows[-1]['cursor'] if rows else cursor}

    def logs(self,rid,stream,offset=0,limit=65536):
        self.get(rid)
        if stream not in ('stdout','stderr'): raise AdapterError('LOG_NOT_FOUND',404)
        from .native import contained
        path=self.root/rid/stream
        if not path.exists() and not path.is_symlink():
            return {'text':'','next_offset':offset,'available':False}
        with contained(self.root,str(path),directory=False).open('rb') as f:
            f.seek(offset);raw=f.read(limit)
        return {'text':raw.decode('utf-8','replace'),'next_offset':offset+len(raw),'available':True}

    def connect(self):
        db = sqlite3.connect(self.root / 'requests.sqlite3', timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def submit(self, command: Command, key: str, digest: str):
        if command.operation not in {'capture', 'run', 'pause', 'snapshot', 'export-full', 'export-slim'} or command_digest(command) != digest:
            raise AdapterError('INVALID_COMMAND', 400)
        with self.transaction() as db:
            row = db.execute('SELECT * FROM requests WHERE key=?', (key,)).fetchone()
            if row:
                if row['digest'] != digest:
                    raise AdapterError('IDEMPOTENCY_CONFLICT', 409)
                return AcceptedRequest(request_id=row['id'], status=row['status'])
            rid = 'req-' + uuid.uuid4().hex
            now = time.time()
            try:
                db.execute('''INSERT INTO requests
                    (id,key,digest,operation,target,lane,command,status,created,updated,run_key)
                    VALUES (?,?,?,?,?,?,?,'queued',?,?,?)''',
                    (rid,key,digest,command.operation,command.target,
                     'control' if command.operation == 'pause' else 'execution',
                     json.dumps(asdict(command)),now,now,self.run_key(asdict(command)) if command.operation=='run' else None))
                self.event(db,rid,'QUEUED',{'operation':command.operation})
            except sqlite3.IntegrityError as exc:
                raise AdapterError('JOB_ALREADY_ACTIVE', 409) from exc
        return AcceptedRequest(request_id=rid, status='queued')

    def get(self, request_id):
        with closing(self.connect()) as db:
            row = db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
        if row is None:
            raise AdapterError('REQUEST_NOT_FOUND',404)
        return {k: (json.loads(row[k]) if k == 'native' and row[k] else row[k]) for k in
                ('id','operation','target','status','created','updated','pid','process_start',
                 'exit_code','classification','native','error')} | {
                     'scientific_outcome': 'NOT_INFERRED',
                     'logs': {name: (self.root / request_id / name).stat().st_size
                              if (self.root / request_id / name).exists() else 0
                              for name in ('stdout','stderr')}}

    def update(self, rid, **values):
        allowed = {'status','pid','process_start','exit_code','classification','native','error'}
        if not set(values) <= allowed:
            raise ValueError('Invalid update')
        values['updated'] = time.time()
        with self.transaction() as db:
            db.execute('UPDATE requests SET '+','.join(k+'=?' for k in values)+' WHERE id=?',
                       (*values.values(),rid))
            self.event(db,rid,'STATE_UPDATED',{'status':values.get('status'),'error':values.get('error')})

    def claim(self, lane):
        blocked = False
        row = None
        with self.transaction() as db:
            # Persist ambiguity even when this call refuses further dispatch.
            db.execute("UPDATE requests SET status='needs_reconciliation',updated=? "
                       "WHERE lane=? AND status IN ('dispatching','running')",(time.time(),lane))
            blocked = bool(db.execute("SELECT 1 FROM requests WHERE lane=? AND status='needs_reconciliation'",
                                      (lane,)).fetchone())
            if not blocked:
                row = db.execute("SELECT * FROM requests WHERE lane=? AND status='queued' ORDER BY created,id LIMIT 1",
                                 (lane,)).fetchone()
                if row:
                    db.execute("UPDATE requests SET status='dispatching',updated=? WHERE id=?",(time.time(),row['id']))
                    self.event(db,row['id'],'DISPATCH_INTENT',{})
        if blocked:
            raise AdapterError('LANE_NEEDS_RECONCILIATION')
        return dict(row) if row else None


def prepare(settings, row, directory):
    """Resolve current server catalogue again; a stored argv is never executed."""
    adapter = NativeAdapter(settings)
    from .tracking import catalogue
    catalog = catalogue(settings,Queue(settings.worker_state) if settings.worker_state else None)
    for group in (catalog.jobs, catalog.tasks):
        if len({x.id for x in group}) != len(group):
            raise AdapterError('CATALOG_DUPLICATE_ID')
    stored = json.loads(row['command'])
    operation = row['operation']
    if operation == 'capture':
        task = next((x for x in catalog.tasks if x.id == row['target']), None)
        if task is None:
            raise AdapterError('TASK_NOT_FOUND',404)
        command = adapter.command(operation, task=task)
    else:
        job = next((x for x in catalog.jobs if x.id == row['target']), None)
        if job is None:
            raise AdapterError('JOB_NOT_FOUND',404)
        options = {'reason':stored['argv'][-1]} if operation == 'pause' else {}
        if operation in ('export-full','export-slim'): options['export_id'] = canonical_hash(row['key'])
        command = adapter.command(operation,job,**options)
    if command_digest(command) != row['digest']:
        raise AdapterError('COMMAND_CHANGED',409)
    adapter.verify_runtime()
    argv = list(command.argv)
    cwd = command.cwd
    if operation == 'capture':
        raw = Path(argv[-1]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != command.specification_sha256:
            raise AdapterError('SPECIFICATION_CHANGED',409)
        spec = json.loads(raw)
        if Path(spec['engine_source']).resolve() != adapter.source.resolve() or spec['project_source']:
            raise AdapterError('UNSUPPORTED_CAPTURE_SOURCE')
        frozen = directory / 'specification.json'
        with frozen.open('xb') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
        argv[-1] = str(frozen)
    else:
        from .native import contained, PIN_SOURCE
        source = contained(Path(settings.workspace_root), str(Path(job.workspace) / 'source'), directory=True)
        adapter.verify_source(source)
        workspace = read_json(source.parent / 'WORKSPACE.json')
        if workspace.get('source_sha256') != PIN_SOURCE:
            raise AdapterError('CAPTURE_SOURCE_NOT_SUPPORTED')
        cwd = str(source)
    if operation in ('export-full','export-slim'):
        from .exports import export_path
        output = export_path(settings.worker_state, canonical_hash(row['key']))
        output.parent.mkdir(mode=0o700, exist_ok=True)
        export_path(settings.worker_state, canonical_hash(row['key']))
        if output.exists(): raise AdapterError('EXPORT_ALREADY_EXISTS',409)
    return argv, cwd, adapter.environment(source=Path(cwd))


def process_identity(pid):
    try:
        # Field 22 follows the parenthesized comm, which may contain spaces.
        start = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return boot + ':' + start
    except (OSError, IndexError):
        return None


def execute(settings, queue, row, lock_fd):
    rid = row['id']
    directory = queue.root / rid
    spawned = False
    try:
        directory.mkdir(mode=0o700)
        argv, cwd, env = prepare(settings,row,directory)
        # Logs exist before spawn; direct file descriptors avoid memory growth,
        # pipe deadlocks, and coupling child lifetime to the HTTP or worker process.
        with (directory/'stdout').open('xb',buffering=0) as stdout, (directory/'stderr').open('xb',buffering=0) as stderr:
            process = subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                                       stdout=stdout,stderr=stderr,shell=False,
                                       start_new_session=True,pass_fds=(lock_fd,))
            spawned = True
            queue.update(rid,status='running',pid=process.pid,process_start=process_identity(process.pid))
            rc = process.wait()  # Intentionally no science deadline.
            os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
        from .tracking import atomic_record, file_hash, finalize
        atomic_record(directory/'exit.json',{
            'schema':'IG_WEB_EXIT_V1','request_id':rid,'digest':row['digest'],'exit_code':rc,
            'stdout_sha256':file_hash(directory/'stdout'),
            'stderr_sha256':file_hash(directory/'stderr')})
        finalize(settings,queue,row)
    except Exception as exc:
        queue.update(rid,status='needs_reconciliation' if spawned else 'refused',
                     error=exc.code if isinstance(exc,AdapterError) else type(exc).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--lane',choices=['execution','control'])
    parser.add_argument('--reconcile',metavar='REQUEST_ID')
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    settings = Settings.model_validate(read_json(Path(args.config)))
    if not settings.worker_state:
        parser.error('worker_state must name a private directory')
    if process_identity(os.getpid()) is None:
        parser.error('Linux process identity unavailable; worker startup refused')
    os.umask(0o077)
    queue = Queue(settings.worker_state)
    if args.reconcile:
        from .tracking import reconcile
        print(json.dumps(reconcile(settings,queue,args.reconcile)))
        return
    if not args.lane: parser.error('--lane or --reconcile is required')
    with (queue.root / (args.lane+'.lock')).open('a+b') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1,'Worker lane is already held\n')
        while True:
            row = queue.claim(args.lane)
            if row:
                execute(settings,queue,row,lock.fileno())
            if args.once:
                return
            if row is None:
                time.sleep(0.5)


if __name__ == '__main__':
    main()
