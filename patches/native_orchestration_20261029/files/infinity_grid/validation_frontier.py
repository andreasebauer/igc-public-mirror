"""Verifier-owned durable units. Never put this authority in producer CAS.

The authority directory is trusted native runtime state and must be restored only
through the authenticated native checkpoint closure, never imported from a
producer result. Source/runtime/scope changes create a new trust epoch.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib,hmac,json,os,sqlite3,sys,time
from .library_service_protocol import digest

_ACTIVE=ContextVar('validation_frontier',default=None)
class ValidationYield(RuntimeError):pass

def implementation_binding():
    from .acceptance_observer import phase,operation
    def hash_file(hasher,path,operation_name,files_done,bytes_done):
        # Preserve the exact previous byte concatenation while bounding each
        # read and exposing progress before a complete runtime scan finishes.
        with path.open('rb') as stream:
            while chunk:=stream.read(1048576):
                hasher.update(chunk);bytes_done+=len(chunk)
                operation(operation_name,files_done,bytes_done=bytes_done)
        files_done+=1
        operation(operation_name,files_done,bytes_done=bytes_done)
        return files_done,bytes_done
    root=Path(__file__).parent
    h=hashlib.sha256()
    with phase('validation_source_binding'):
        files_done=bytes_done=0
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
            h.update(p.relative_to(root).as_posix().encode());h.update(b'\0')
            files_done,bytes_done=hash_file(h,p,'implementation_source_hash',files_done,bytes_done)
        operation('implementation_source_hash',files_done,files_done,bytes_done=bytes_done,bytes_total=bytes_done)
    # Native session admission already verifies the pinned runtime. Retain the
    # actual runtime file binding as well, so a changed extension cannot reuse credit.
    rh=hashlib.sha256();prefix=Path(sys.prefix)
    with phase('validation_runtime_binding'):
        files_done=bytes_done=0
        for directory in (prefix,prefix.parent/'native',prefix.parent/'host'):
            if not directory.exists():continue
            for p in sorted(directory.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:
                    rh.update(str(p.relative_to(prefix.parent)).encode());rh.update(b'\0')
                    files_done,bytes_done=hash_file(rh,p,'implementation_runtime_hash',files_done,bytes_done)
        operation('implementation_runtime_hash',files_done,files_done,bytes_done=bytes_done,bytes_total=bytes_done)
    return dict(source_sha256=h.hexdigest(),runtime_sha256=rh.hexdigest(),python=sys.version,sqlite=sqlite3.sqlite_version,protocol='IG_VALIDATION_FRONTIER_V1')

class Frontier:
    def __init__(self,directory,binding,*,max_new_units=32):
        if max_new_units is not None and (type(max_new_units)is not int or max_new_units<1):raise ValueError('VALIDATION_UNIT_BUDGET')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.binding=binding;self.epoch=digest(binding);self.limit=max_new_units;self.new=self.reused=0
        key=self.directory/'authority.key';db=self.directory/'units.sqlite'
        if not key.exists():
            if db.exists():raise ValueError('VALIDATION_AUTHORITY_KEY_MISSING')
            fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f:f.write(os.urandom(32));f.flush();os.fsync(f.fileno())
        self.key=key.read_bytes()
        if len(self.key)!=32 or key.stat().st_mode&0o077:raise ValueError('VALIDATION_AUTHORITY_KEY_INVALID')
        self.db=sqlite3.connect(db);self.db.executescript('PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS units(epoch TEXT,role TEXT,input_sha TEXT,value TEXT,signature TEXT,PRIMARY KEY(epoch,role,input_sha)) WITHOUT ROWID;')
    def close(self):self.db.close()
    def _signature(self,role,pin,value):return hmac.new(self.key,json.dumps([self.epoch,role,pin,value],separators=(',',':')).encode(),hashlib.sha256).hexdigest()
    def status(self,role):
        data=dict(schema_id='IG_VALIDATION_FRONTIER_STATUS_V1',epoch=self.epoch,role=role,new_units=self.new,reused_units=self.reused,unit_budget=self.limit,unix=time.time())
        p=self.directory/'STATUS.json';tmp=p.with_suffix('.tmp')
        try:tmp.write_text(json.dumps(data));tmp.replace(p)
        except OSError as exc:
            # Diagnostic failure cannot revoke or mask a durable verifier commit.
            if not getattr(self,'status_failed',False):
                print('VALIDATION_TELEMETRY_DEGRADED:'+type(exc).__name__,file=sys.stderr,flush=True)
            self.status_failed=True
    def unit(self,role,inputs,verify):
        if not isinstance(role,str) or not role:raise ValueError('VALIDATION_ROLE_REQUIRED')
        from .acceptance_observer import phase,tick,operation,measured_digest
        with phase('validation_input_hash',force=False):
            pin=measured_digest(inputs,"unit_input_hash") # Always derives from actual current complete unit bytes.
        tick('validation_input_hashes')
        row=self.db.execute('SELECT value,signature FROM units WHERE epoch=? AND role=? AND input_sha=?',(self.epoch,role,pin)).fetchone()
        if row:
            value,signature=row
            with phase('validation_receipt_authentication',force=False):
                if not hmac.compare_digest(signature,self._signature(role,pin,value)):raise ValueError('VALIDATION_RECEIPT_AUTHENTICATION')
            tick('validation_units_reused:'+role)
            tick('validation_units_reused')
            self.reused+=1;self.status(role);return json.loads(value)
        if self.limit is not None and self.new>=self.limit:
            self.status(role);raise ValidationYield('VALIDATION_UNIT_BUDGET_REACHED')
        with phase('validation_new_verification:'+role,force=False):
            result=verify() # interrupted units earn no credit
        value=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
        with self.db:self.db.execute('INSERT INTO units VALUES(?,?,?,?,?)',(self.epoch,role,pin,value,self._signature(role,pin,value)))
        self.new+=1
        tick("validation_units_committed:"+role)
        tick("validation_units_committed")
        # Save an expensive structural verification at the next native boundary.
        if role=='ROOTED_EXACT_BYTES' and self.limit is not None:self.new=self.limit
        self.status(role);return result

@contextmanager
def session(directory,binding,*,max_new_units=32):
    f=Frontier(directory,binding,max_new_units=max_new_units);token=_ACTIVE.set(f)
    try:yield f
    finally:_ACTIVE.reset(token);f.close()

def unit(role,inputs,verify):
    f=_ACTIVE.get();return verify() if f is None else f.unit(role,inputs,verify)

def verified_result(store,ref,on_ref):
    """Persist full-verifier authority plus exact consumed bytes, never just PASS.

    Every reuse authenticates the verifier receipt and every saved byte copy,
    then rereads the current owner bytes and compares them in full. Callbacks
    run before that final byte comparison, preserving mutation detection.
    """
    f=_ACTIVE.get()
    if f is None:return None
    from .complete_result_storage import (_rooted_proof_policy,_RootedVerifiedSnapshot,_RootedSnapshotStore,require)
    policy=_rooted_proof_policy(store)
    # Storage location is not verifier identity. Cold restore may relocate the
    # same authenticated byte closure; purpose/limits/budgets stay bound.
    durable_policy=json.loads(policy)
    durable_policy.pop('directory',None)
    identity={'root':ref,'policy':durable_policy}
    destination=f.directory/'snapshots'/digest((f.epoch,identity))
    def tag(sha,raw):return hmac.new(f.key,b'ROOTED_BYTES_V1\0'+sha.encode()+raw,hashlib.sha256).hexdigest()
    def verify():
        proof=_RootedVerifiedSnapshot(ref,policy)
        try:
            tracker=_RootedSnapshotStore(store,proof)
            require(tracker._verify_result_exact(ref,on_ref=proof.observer(None)) is True,'ROOTED_FULL_VERIFICATION_REQUIRED')
            proof.recheck(store)
            destination.mkdir(parents=True,exist_ok=True,mode=0o700);blobs={}
            for sha in proof.refs:
                raw=(Path(proof.temporary.name)/sha).read_bytes()
                path=destination/sha
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
                temporary.chmod(0o400);temporary.replace(path);blobs[sha]=tag(sha,raw)
            return {'refs':list(proof.refs.values()),'events':proof.events,'blobs':blobs}
        finally:proof.close()
    from .acceptance_observer import phase,detail
    with phase('rooted_exact_bytes_verification'):
        metadata=f.unit('ROOTED_EXACT_BYTES',identity,verify)
    if on_ref is not None:
        for event in metadata['events']:on_ref(dict(event))
    from .runtime_capacity import host_memory_ceiling_bytes
    total_bytes=sum(int(item['size_bytes']) for item in metadata['refs']);checked_bytes=0
    detail(rooted_objects_total=len(metadata['refs']),rooted_bytes_total=total_bytes)
    for object_index,item in enumerate(metadata['refs'],1):
        sha=item['sha256'];path=destination/sha
        require(path.is_file() and not path.is_symlink(),'ROOTED_FRONTIER_BYTES_MISSING')
        raw=path.read_bytes()
        require(hmac.compare_digest(tag(sha,raw),metadata['blobs'][sha]),'ROOTED_FRONTIER_BYTES_AUTHENTICATION')
        require(raw==store.read(item,host_memory_ceiling_bytes()),'ROOTED_VERIFICATION_BYTES_CHANGED')
        checked_bytes+=len(raw)
        detail(rooted_objects_checked=object_index,rooted_bytes_checked=checked_bytes)
    return True

def workspace_bytes(directory):
    """Account native authority state without admitting arbitrary workspace files."""
    import stat,re
    from .complete_result_storage import require
    root=Path(directory);require(root.is_dir() and not root.is_symlink(),'VALIDATION_AUTHORITY_LAYOUT')
    total=0;owner=root.stat().st_uid
    def regular(p):
        info=p.lstat();require(stat.S_ISREG(info.st_mode) and info.st_uid==owner,'VALIDATION_AUTHORITY_FILE_TYPE');return info.st_size
    for p in root.iterdir():
        require(not p.is_symlink(),'VALIDATION_AUTHORITY_SYMLINK')
        if p.name=='snapshots':
            require(p.is_dir(),'VALIDATION_SNAPSHOT_LAYOUT')
            for folder in p.iterdir():
                require(not folder.is_symlink() and folder.is_dir() and re.fullmatch('[0-9a-f]{64}',folder.name),'VALIDATION_SNAPSHOT_LAYOUT')
                for blob in folder.iterdir():
                    require(re.fullmatch('[0-9a-f]{64}(?:\\.tmp)?',blob.name),'VALIDATION_SNAPSHOT_MEMBER');total+=regular(blob)
        else:
            require(p.name in ('authority.key','units.sqlite','units.sqlite-journal','STATUS.json','STATUS.tmp'),'VALIDATION_AUTHORITY_MEMBER')
            size=regular(p)
            if p.name=='authority.key':require(size==32 and not p.stat().st_mode&0o077,'VALIDATION_AUTHORITY_KEY_INVALID')
            total+=size
    return total
