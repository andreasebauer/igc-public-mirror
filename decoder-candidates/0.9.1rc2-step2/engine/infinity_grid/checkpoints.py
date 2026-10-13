from __future__ import annotations
import hashlib, json, os, socket, time, uuid
from pathlib import Path
from .canon import canonical_sha256, canonical_text, write_json_atomic
from .store import ArtifactStore
from .records import utc_now
from .safety import contained_path, validate_identifier

class DuplicateController(RuntimeError): pass
class CorruptCheckpoint(RuntimeError): pass


def _boot_id() -> str | None:
    p = Path('/proc/sys/kernel/random/boot_id')
    try:
        v = p.read_text(encoding='utf-8').strip()
        return v or None
    except OSError:
        return None


def _process_start_token(pid: int) -> str | None:
    # Linux /proc starttime (field 22) is stable for the life of one process and changes on PID reuse.
    p = Path(f'/proc/{int(pid)}/stat')
    try:
        raw = p.read_text(encoding='utf-8')
        # comm may contain spaces inside parentheses; fields after the final ')' are unambiguous.
        tail = raw.rsplit(')', 1)[1].strip().split()
        return tail[19] if len(tail) > 19 else None
    except (OSError, ValueError, IndexError):
        return None


def current_process_identity() -> dict:
    pid = os.getpid()
    return {
        'pid': pid,
        'host': socket.gethostname(),
        'boot_id': _boot_id(),
        'process_start_token': _process_start_token(pid),
    }


class RunLock:
    def __init__(self,run_dir:Path):
        self.run_dir=Path(run_dir); self.path=self.run_dir/'controller.lock'; self.held=False; self.payload=None
    def acquire(self):
        self.run_dir.mkdir(parents=True,exist_ok=True)
        ident=current_process_identity()
        payload={**ident,'created_utc':utc_now(),'lock_token':uuid.uuid4().hex}
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
        try: fd=os.open(str(self.path),flags,0o644)
        except FileExistsError: raise DuplicateController(f'controller lock already exists: {self.path}')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as h:
                json.dump(payload,h,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
            try:
                dfd=os.open(str(self.run_dir),os.O_RDONLY)
                try: os.fsync(dfd)
                finally: os.close(dfd)
            except OSError: pass
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self.held=True; self.payload=payload; return payload
    def release(self):
        if not self.held: return
        # Never unlink a lock that has been replaced by another owner.
        try:
            observed=json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            self.held=False
            return
        except Exception as exc:
            self.held=False
            raise DuplicateController('controller lock unreadable before release; refusing unlink') from exc
        if not isinstance(observed,dict) or not self.payload or observed.get('lock_token')!=self.payload.get('lock_token'):
            self.held=False
            raise DuplicateController('controller lock ownership changed before release; refusing unlink')
        self.path.unlink(); self.held=False
    def __enter__(self): self.acquire(); return self
    def __exit__(self,*exc): self.release()


def recover_stale_lock(run_dir:Path, *, external_liveness_decision:dict|None=None, operator_override:dict|None=None):
    p=Path(run_dir)/'controller.lock'
    if not p.exists(): return {'status':'NO_LOCK'}
    try:
        obj=json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        raise DuplicateController('controller lock is unreadable; recovery requires explicit forensic/operator action') from exc
    if not isinstance(obj, dict) or not isinstance(obj.get('lock_token'), str) or not obj.get('lock_token'):
        raise DuplicateController('controller lock lacks a valid ownership token; refusing automatic recovery')
    host=socket.gethostname()
    try:
        pid=int(obj.get('pid'))
    except Exception as exc:
        raise DuplicateController('controller lock PID is unresolved; refusing automatic recovery') from exc
    foreign=obj.get('host')!=host
    if foreign:
        decision=external_liveness_decision or operator_override
        if not isinstance(decision,dict) or decision.get('allow_recovery') is not True or not decision.get('reason'):
            raise DuplicateController('foreign-host lock recovery requires explicit auditable liveness decision or operator override')
    else:
        if pid <= 0:
            raise DuplicateController('controller lock PID is nonpositive; refusing automatic recovery')
        alive=False
        try:
            os.kill(pid,0); alive=True
        except ProcessLookupError:
            alive=False
        except PermissionError:
            raise DuplicateController(f'cannot establish process absence for {pid}')
        if alive:
            locked_start=obj.get('process_start_token'); current_start=_process_start_token(pid)
            locked_boot=obj.get('boot_id'); current_boot=_boot_id()
            if locked_start is None or current_start is None or locked_boot is None or current_boot is None:
                raise DuplicateController('live PID identity cannot be resolved completely; refusing recovery')
            if locked_start == current_start and locked_boot == current_boot:
                raise DuplicateController(f'process {pid} is still alive; refusing recovery')
    hist=Path(run_dir)/'lock_recovery'; hist.mkdir(parents=True,exist_ok=True)
    target=hist/f"recovered-{int(time.time())}-{uuid.uuid4().hex}.json"
    audit={'previous':obj,'recovered_utc':utc_now(),'external_liveness_decision':external_liveness_decision,'operator_override':operator_override}
    write_json_atomic(target,audit)
    # Compare-and-delete: re-read immediately before unlinking and require the
    # same ownership token.  Another controller may have acquired a new lock
    # while the recovery audit record was being committed.
    try:
        current=json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        raise DuplicateController('controller lock changed/unreadable during recovery; refusing unlink') from exc
    if current.get('lock_token') != obj.get('lock_token'):
        return {'status':'OWNERSHIP_CHANGED_REFUSED','audit_copy':str(target),'previous':obj,'current':current}
    p.unlink()
    return {'status':'RECOVERED','audit_copy':str(target),'previous':obj}

class CheckpointManager:
    def __init__(self,run_dir:Path,store:ArtifactStore): self.run_dir=Path(run_dir); self.store=store
    def _stage_dir(self,stage_id): validate_identifier(stage_id,field='stage_id'); return contained_path(self.run_dir/'stages',stage_id,field='stage_id')
    def current_pointer(self,stage_id):
        p=self._stage_dir(stage_id)/'current.json'
        if not p.exists(): return None
        return json.loads(p.read_text())
    def current(self,stage_id):
        ptr=self.current_pointer(stage_id)
        if ptr is None: return None
        ap=self._stage_dir(stage_id)/'attempts'/f"{int(ptr['attempt']):06d}.json"
        if not ap.is_file(): raise CorruptCheckpoint(f'current pointer missing attempt {ap}')
        raw=ap.read_bytes(); observed=hashlib.sha256(raw).hexdigest()
        if observed!=ptr['checkpoint_sha256']: raise CorruptCheckpoint('checkpoint record hash mismatch')
        cp=json.loads(raw)
        if canonical_sha256({k:v for k,v in cp.items() if k!='checkpoint_content_sha256'})!=cp.get('checkpoint_content_sha256'): raise CorruptCheckpoint('checkpoint content identity mismatch')
        if cp.get('stage_id') != stage_id:
            raise CorruptCheckpoint(f'checkpoint stage binding mismatch: requested {stage_id}, body {cp.get("stage_id")}')
        if cp.get('run_id') != self.run_dir.name:
            raise CorruptCheckpoint('checkpoint run binding mismatch')
        if int(cp.get('attempt', -1)) != int(ptr.get('attempt', -2)):
            raise CorruptCheckpoint('checkpoint attempt binding mismatch')
        return cp
    def dependency_bindings(self,dependency_stage_ids:list[str]):
        out=[]
        for sid in dependency_stage_ids:
            ptr=self.current_pointer(sid); cp=self.current(sid)
            if not ptr or not cp or cp.get('status')!='COMPLETE_VALID': raise CorruptCheckpoint(f'dependency {sid} is not complete')
            out.append({'stage_id':sid,'attempt':cp['attempt'],'checkpoint_sha256':ptr['checkpoint_sha256'],'checkpoint_content_sha256':cp['checkpoint_content_sha256'],'output_artifacts':[{'logical_name':a.get('logical_name'),'sha256':a['sha256'],'size_bytes':a.get('size_bytes')} for a in cp.get('output_artifacts',[])]})
        return out
    def reuse_status(self,stage_id,identity:dict):
        cp=self.current(stage_id)
        if cp is None: return {'status':'MISSING'}
        for key in ['plan_sha256','code_sha256','environment_sha256','run_core_sha256','stage_spec_sha256']:
            if cp.get(key)!=identity.get(key): return {'status':'INVALID_IDENTITY','field':key,'expected':identity.get(key),'observed':cp.get(key)}
        if cp.get('input_artifacts')!=identity.get('input_artifacts'): return {'status':'INVALID_IDENTITY','field':'input_artifacts'}
        if cp.get('dependency_bindings')!=identity.get('dependency_bindings'): return {'status':'INVALID_IDENTITY','field':'dependency_bindings'}
        if cp.get('status')!='COMPLETE_VALID': return {'status':'INVALID_STATUS','checkpoint_status':cp.get('status')}
        failures=[]
        for a in cp.get('output_artifacts',[]):
            v=self.store.verify(a['sha256'],a.get('size_bytes'))
            if v['status']!='PASS': failures.append(v)
        if failures: raise CorruptCheckpoint(f'checkpoint output corruption: {failures}')
        return {'status':'REUSABLE','checkpoint':cp}
    def publish(self,stage_id,*,identity,input_artifacts,output_artifacts,dependencies,resource_usage,stage_result,status='COMPLETE_VALID'):
        sd=self._stage_dir(stage_id); ad=sd/'attempts'; ad.mkdir(parents=True,exist_ok=True); existing=sorted(ad.glob('*.json')); attempt=(int(existing[-1].stem)+1 if existing else 1); ru=dict(resource_usage); ru.setdefault('checkpoint_bytes',0)
        cp={'schema_id':'IG_STAGE_CHECKPOINT_V0_17','run_id':self.run_dir.name,'stage_id':stage_id,'attempt':attempt,'status':status,'plan_sha256':identity['plan_sha256'],'code_sha256':identity['code_sha256'],'environment_sha256':identity['environment_sha256'],'run_core_sha256':identity['run_core_sha256'],'stage_spec_sha256':identity['stage_spec_sha256'],'input_artifacts':input_artifacts,'output_artifacts':output_artifacts,'dependencies':dependencies,'dependency_bindings':identity['dependency_bindings'],'resource_usage':ru,'stage_result':stage_result,'created_utc':utc_now()}
        cp['checkpoint_content_sha256']=canonical_sha256(cp)
        for _ in range(8):
            size=len(canonical_text(cp,pretty=True).encode('utf-8'))
            if ru.get('checkpoint_bytes')==size: break
            ru['checkpoint_bytes']=size; cp['checkpoint_content_sha256']=canonical_sha256({k:v for k,v in cp.items() if k!='checkpoint_content_sha256'})
        ap=ad/f'{attempt:06d}.json'
        if ap.exists(): raise RuntimeError(f'checkpoint attempt path already exists: {ap}')
        write_json_atomic(ap,cp); sha=hashlib.sha256(ap.read_bytes()).hexdigest(); write_json_atomic(sd/'current.json',{'attempt':attempt,'checkpoint_sha256':sha})
        return cp
