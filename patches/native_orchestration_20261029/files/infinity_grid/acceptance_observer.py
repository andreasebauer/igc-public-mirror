"""Operational-only acceptance progress. Never contributes scientific bytes."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import time, os, uuid
from .canon import write_json_atomic

_ACTIVE = ContextVar('acceptance_observer', default=None)

def workspace_bytes(root):
    """Stream exact accounting for recognized, individually bounded status files."""
    import re,stat
    from .complete_result_storage import require
    root=Path(root)
    require(root.is_dir() and not root.is_symlink(),'ACCEPTANCE_TELEMETRY_LAYOUT')
    def status_bytes(p):
        try:info=p.lstat()
        except FileNotFoundError:return 0 # Atomic writer already renamed it.
        require(stat.S_ISREG(info.st_mode) and info.st_size<=65536 and
            (p.name=='ACCEPTANCE_STATUS.json' or re.fullmatch(r'\.ACCEPTANCE_STATUS\.json\.tmp-[0-9]+-[0-9a-f]{32}',p.name)),
            'ACCEPTANCE_TELEMETRY_FILE')
        return info.st_size
    total=0
    for p in root.iterdir():
        if p.name=='workers':
            require(p.is_dir() and not p.is_symlink(),'ACCEPTANCE_TELEMETRY_WORKERS')
            # Completed native pools leave diagnostic records under prior PIDs.
            # Stream their exact byte accounting; history is not a worker limit.
            for w in p.iterdir():
                require(re.fullmatch(r'[0-9]{1,10}',w.name) and w.is_dir() and not w.is_symlink(),'ACCEPTANCE_TELEMETRY_WORKER')
                total+=sum(status_bytes(item) for item in w.iterdir())
        else:
            # Interrupted atomic writes remain diagnostic evidence. Their
            # count is not a science limit; every original byte stays accounted.
            total+=status_bytes(p)
    return total

class Observer:
    def __init__(self, root):
        self.path = Path(root) / 'ACCEPTANCE_STATUS.json'
        self.identity = {"observer_id":uuid.uuid4().hex,"pid":os.getpid()}
        self.invocation_serial = 0
        self.invocations = []
        self.operations = {}
        self.started = time.monotonic()
        self.last_write = 0.0
        self.last_progress = None
        self.counts = {}
        self.stack = []
        self.details = {}
        self.durations = {}
        self.work_plan = None
        self.progress = {}
        self.started_phases = []
        self.status = 'RUNNING'
        self.termination = None
        self.write(True)

    def write(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 5:
            return
        try:
            write_json_atomic(self.path, {'schema_id':'IG_ACCEPTANCE_OPERATIONAL_STATUS_V1',
            'termination':self.termination, 'identity':dict(self.identity), 'invocations':list(self.invocations),
            'operations':dict(self.operations), 'heartbeat_unix':time.time(),
            'status':self.status, 'elapsed_seconds':now-self.started,
            'active_phases':list(self.stack), 'counters':dict(self.counts),
            'context':dict(self.details), 'timing_seconds':dict(self.durations),
            'registered_work_plan':self.work_plan, 'registered_progress':dict(self.progress),
            'active_phase_elapsed_seconds': None if not self.started_phases else now-self.started_phases[-1],
            'last_progress_unix':self.last_progress,
            'last_progress_semantics':'LATEST_OBSERVED_COUNTER_ADVANCE; NOT_HEARTBEAT_OR_COMPLETION_AUTHORITY',
            'science_hash_excluded':True})
        except OSError:
            pass # Diagnostics must not stop or alter scientific work.
        self.last_write = now

def tick(name, count=1):
    observer = _ACTIVE.get()
    if observer is not None:
        observer.counts[name] = observer.counts.get(name, 0) + count
        if count>0:observer.last_progress=time.time()
        observer.write()
        return observer.counts[name]

@contextmanager
def phase(name, *, force=True):
    observer = _ACTIVE.get()
    if observer is None:
        yield
        return
    observer.invocation_serial += 1
    invocation={'phase':name,'invocation_id':observer.invocation_serial,
        'parent_invocation_id':observer.invocations[-1]['invocation_id'] if observer.invocations else None}
    observer.invocations.append(invocation)
    observer.stack.append(name)
    observer.started_phases.append(time.monotonic())
    observer.write(force)
    try:
        yield
    finally:
        started=observer.started_phases.pop()
        observer.durations[name]=observer.durations.get(name,0.0)+time.monotonic()-started
        observer.stack.pop()
        observer.invocations.pop()
        observer.write(force)

@contextmanager
def session(root):
    if _ACTIVE.get() is not None:
        yield
        return
    observer = Observer(root)
    token = _ACTIVE.set(observer)
    try:
        yield
    except BaseException as exc:
        from .validation_frontier import ValidationYield
        observer.status = 'CHECKPOINT_PAUSE' if isinstance(exc,ValidationYield) else 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        observer.termination={'exception_type':type(exc).__name__,'reason':str(exc)[:1000],
            'exit_code':130 if isinstance(exc,KeyboardInterrupt) else exc.code if isinstance(exc,SystemExit) and type(exc.code)is int else 1,
            'sqlite_errorcode':getattr(exc,'sqlite_errorcode',None),'sqlite_errorname':getattr(exc,'sqlite_errorname',None)}
        raise
    else:
        observer.status = 'COMPLETE'
        observer.termination={'exit_code':0}
    finally:
        observer.write(True)
        _ACTIVE.reset(token)

def observed(name):
    from functools import wraps
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _ACTIVE.get() is None and args:
                owner = args[0]
                store = getattr(owner, 'store', owner)
                directory = getattr(store, 'directory', None)
                runtime = getattr(owner, 'runtime_root', None)
                root = Path(directory).parent/'telemetry' if directory is not None else (Path(runtime)/'telemetry' if runtime is not None else None)
                if root is not None:
                    with session(root), phase(name):
                        return function(*args, **kwargs)
            with phase(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def detail(**fields):
    observer=_ACTIVE.get()
    if observer is not None:
        observer.details.update(fields)
        observer.write()

@contextmanager
def context(**fields):
    observer=_ACTIVE.get()
    if observer is None:
        yield
        return
    previous=dict(observer.details)
    observer.details.update(fields)
    observer.write(True)
    try:yield
    finally:
        observer.details=previous
        observer.write(True)

def contract_observed(function):
    from functools import wraps
    @wraps(function)
    def wrapped(store,task,ref):
        observer=_ACTIVE.get()
        caller=observer.stack[-1] if observer is not None and observer.stack else 'UNINSTRUMENTED_CALLER'
        with context(task_id=task['task_id'],contract_caller=caller),phase('contract_validation'):
            return function(store,task,ref)
    return wrapped


def register_plan(plan):
    observer=_ACTIVE.get()
    if observer is not None:
        # Keep the status bounded: task detail remains in the preserved plan input.
        observer.work_plan={k:plan[k] for k in ('plan_sha256','manifest_sha256','stages','whole_run_eta_seconds','eta_reason')}
        observer.write(True)


def progress(stage_id,done,total,elapsed,*,homogeneous=False):
    observer=_ACTIVE.get()
    if observer is None or observer.work_plan is None:return
    if stage_id not in {s['stage_id'] for s in observer.work_plan['stages']}:return
    from .accepted_dispatch_plan import stage_eta
    if done>observer.progress.get(stage_id,{}).get('done',0):observer.last_progress=time.time()
    observer.progress[stage_id]={'done':done,'total':total,'elapsed_seconds':elapsed,
        'eta':stage_eta(done,total,elapsed,homogeneous=homogeneous),
        'completion_authority':'NONE; OBSERVATION_ONLY'}
    observer.write()


def bind_identity(**fields):
    observer=_ACTIVE.get()
    if observer is not None:
        observer.identity.update(fields)
        observer.write(True)


def operation(name, done, total=None, **fields):
    """Bounded latest-operation snapshot, scoped to the active invocation."""
    observer=_ACTIVE.get()
    if observer is None:return
    row={'done':done,'total':total,**fields,
        'task_id':observer.details.get('task_id'),
        'invocation_id':observer.invocations[-1]['invocation_id'] if observer.invocations else None,
        'updated_unix':time.time(),'completion_authority':'NONE; OBSERVATION_ONLY'}
    previous=observer.operations.get(name,{})
    if any(row.get(k)!=previous.get(k) for k in ('task_id','invocation_id')):previous={}
    # Refresh work freshness only for measured completed work or consumed bytes.
    # Context, elapsed time, an unchanged count, and a heartbeat are not progress.
    for key in ('done','bytes_done','formations_done'):
        value=row.get(key);old=previous.get(key,0)
        if type(value) in (int,float) and type(old) in (int,float) and value>old:
            observer.last_progress=row['updated_unix'];break
    observer.operations[name]=row
    observer.write()


def measured_digest(value, name='input_hash'):
    from .complete_result_storage import exact_digest
    if _ACTIVE.get() is None:return exact_digest(value)
    def update(size, complete):
        operation(name,int(complete),1,bytes_done=size,
            bytes_total=size if complete else None,bytes_total_reason='Known on serialization completion')
    return exact_digest(value,on_progress=update)
