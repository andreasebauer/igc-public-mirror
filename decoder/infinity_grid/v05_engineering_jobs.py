from __future__ import annotations

"""Decoder-owned registered engineering job implementation.

Every mutable engineering operation targets a controller-created child source
workspace. The running parent source is never edited. The exact operation,
source identities, optional overlay, worker identity and validation groups are
bound in the installed chain registration and execution receipt.
"""

import hashlib, json, os, shutil, stat, subprocess, sys, tempfile, time, traceback, zipfile
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_execution_authority import source_tree_digest
from .v05_engineering_worker import engineering_source_tree_digest
from .v05_validation_runtime import run_registered_validation_groups

JOB_SPEC_SCHEMA="IG_DECODER_REGISTERED_ENGINEERING_JOB_V2"
ACCEPT_SPEC_SCHEMA="IG_DECODER_REGISTERED_ENGINEERING_ACCEPTANCE_V1"
JOB_RESULT_SCHEMA="IG_DECODER_REGISTERED_ENGINEERING_JOB_RESULT_V1"
ACCEPT_RESULT_SCHEMA="IG_DECODER_REGISTERED_ENGINEERING_ACCEPTANCE_RESULT_V1"
SAME_IDENTITY_AUTHORIZATION_SCHEMA="IG_DECODER_ENGINEERING_SAME_IDENTITY_AUTHORIZATION_V1"
SAME_IDENTITY_AUTHORIZATION_SCOPE="ONE_REGISTERED_ENGINEERING_JOB_SAME_IDENTITY"
SAME_IDENTITY_AUTHORIZATION_TEXT="I authorize this one registered engineering job to use the controller UID/GID."

class EngineeringJobError(RuntimeError): pass

def _sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def _safe_id(value: Any) -> str:
    s=str(value)
    if not s or len(s)>128 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in s):
        raise EngineeringJobError("ENGINEERING_JOB_ID")
    return s

def _chown_tree(root: Path, uid: int, gid: int) -> None:
    for p in [root,*sorted(root.rglob("*"))]:
        if p.is_symlink(): raise EngineeringJobError("ENGINEERING_SOURCE_SYMLINK")
        os.chown(p,uid,gid)
        os.chmod(p,0o700 if p.is_dir() else 0o600)

def _readonly_tree(root: Path) -> None:
    for p in sorted(root.rglob("*"),reverse=True):
        if p.is_symlink(): raise EngineeringJobError("ENGINEERING_SOURCE_SYMLINK")
        os.chmod(p,0o500 if p.is_dir() else 0o400)
    os.chmod(root,0o500)

def _copy_source(src: Path, dst: Path) -> None:
    def ignore(_dir,names): return {n for n in names if n in {"__pycache__",".pytest_cache","build","dist",".git",".engineering_tmp"}}
    shutil.copytree(src,dst,ignore=ignore)

def _owned_overlay(path_value: Any, expected_sha: str | None) -> Path | None:
    if path_value is None:
        if expected_sha is not None: raise EngineeringJobError("ENGINEERING_OVERLAY_BINDING")
        return None
    if type(path_value) is not str or not Path(path_value).is_absolute() or ".." in Path(path_value).parts:
        raise EngineeringJobError("ENGINEERING_OVERLAY_ABSOLUTE_PATH")
    p=Path(path_value).resolve(strict=True); st=p.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.geteuid() or st.st_mode & 0o077:
        raise EngineeringJobError("ENGINEERING_OVERLAY_PERMISSIONS")
    if _sha_file(p)!=expected_sha: raise EngineeringJobError("ENGINEERING_OVERLAY_SHA_MISMATCH")
    return p

def _validate_same_identity_authorization(value: Any, p: Mapping[str,Any]) -> dict[str,Any]:
    if type(value) is not dict:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_REQUIRED")
    fields={"schema_id","authorization_id","authorized_by","scope","authorization_text",
            "job_id","parent_source_sha256","worker_uid","worker_gid","reason","authorization_sha256"}
    if set(value)!=fields or value.get("schema_id")!=SAME_IDENTITY_AUTHORIZATION_SCHEMA:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID")
    auth=dict(value)
    try: auth["authorization_id"]=_safe_id(auth["authorization_id"])
    except EngineeringJobError as exc: raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID") from exc
    if type(auth["authorized_by"]) is not str or not auth["authorized_by"].strip() or len(auth["authorized_by"])>200:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID")
    if auth["scope"]!=SAME_IDENTITY_AUTHORIZATION_SCOPE or auth["authorization_text"]!=SAME_IDENTITY_AUTHORIZATION_TEXT:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID")
    if auth["job_id"]!=p["job_id"] or auth["parent_source_sha256"]!=p["parent_source_sha256"]:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_BINDING")
    if auth["worker_uid"]!=p["worker_uid"] or auth["worker_gid"]!=p["worker_gid"]:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_BINDING")
    if type(auth["reason"]) is not str or not auth["reason"].strip() or len(auth["reason"])>1000:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID")
    if type(auth["authorization_sha256"]) is not str or len(auth["authorization_sha256"])!=64:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID")
    sealed={k:v for k,v in auth.items() if k!="authorization_sha256"}
    if canonical_sha256(sealed)!=auth["authorization_sha256"]:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_HASH_MISMATCH")
    return auth

def _worker_identity_mode(p: Mapping[str,Any], controller_uid: int, controller_gid: int) -> tuple[bool,dict[str,Any]|None]:
    same_uid=p["worker_uid"]==controller_uid; same_gid=p["worker_gid"]==controller_gid
    if same_uid!=same_gid:
        raise EngineeringJobError("ENGINEERING_WORKER_IDENTITY_PARTIALLY_SHARED")
    same_identity=same_uid and same_gid
    if same_identity:
        return True,_validate_same_identity_authorization(p["same_identity_authorization"],p)
    if p["same_identity_authorization"] is not None:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_NOT_APPLICABLE")
    return False,None

def validate_job_parameters(parameters: Mapping[str,Any]) -> dict[str,Any]:
    p=dict(parameters); fields={"schema_id","job_id","operation","parent_source_sha256","parent_package_sha256",
        "expected_candidate_source_sha256","expected_candidate_package_sha256","overlay_path","overlay_sha256",
        "validation_groups","worker_uid","worker_gid","wall_seconds_max","same_identity_authorization"}
    if set(p)!=fields or p.get("schema_id")!=JOB_SPEC_SCHEMA: raise EngineeringJobError("ENGINEERING_JOB_PARAMETERS")
    p["job_id"]=_safe_id(p["job_id"])
    if p["operation"] not in {"VALIDATE_SOURCE","APPLY_OVERLAY_AND_VALIDATE"}: raise EngineeringJobError("ENGINEERING_OPERATION_NOT_REGISTERED")
    for k in ("parent_source_sha256","parent_package_sha256","expected_candidate_source_sha256","expected_candidate_package_sha256"):
        if type(p[k]) is not str or len(p[k])!=64: raise EngineeringJobError("ENGINEERING_SOURCE_SHA")
    if p["operation"]=="VALIDATE_SOURCE" and (p["overlay_path"] is not None or p["overlay_sha256"] is not None): raise EngineeringJobError("ENGINEERING_OVERLAY_NOT_ALLOWED")
    if p["operation"]=="APPLY_OVERLAY_AND_VALIDATE" and (type(p["overlay_path"]) is not str or type(p["overlay_sha256"]) is not str or len(p["overlay_sha256"])!=64): raise EngineeringJobError("ENGINEERING_OVERLAY_REQUIRED")
    groups=p["validation_groups"]
    if type(groups) is not list or not groups or len(groups)!=len(set(groups)) or any(type(x) is not str for x in groups): raise EngineeringJobError("ENGINEERING_VALIDATION_GROUPS")
    if type(p["worker_uid"]) is not int or type(p["worker_gid"]) is not int or p["worker_uid"]<0 or p["worker_gid"]<0: raise EngineeringJobError("ENGINEERING_WORKER_IDENTITY")
    if p["same_identity_authorization"] is not None and type(p["same_identity_authorization"]) is not dict:
        raise EngineeringJobError("ENGINEERING_SAME_IDENTITY_AUTHORIZATION_INVALID")
    no_deadline=os.environ.get('IG_DECODER_EXECUTION_POLICY')=='NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
    if not no_deadline and (type(p["wall_seconds_max"]) not in {int,float} or p["wall_seconds_max"]<=0): raise EngineeringJobError("ENGINEERING_WALL_BUDGET")
    return p

def _wait_child(pid: int, timeout: float|None) -> int:
    deadline=None if timeout is None else time.monotonic()+float(timeout)
    while True:
        got, st=os.waitpid(pid, os.WNOHANG)
        if got:
            return os.waitstatus_to_exitcode(st)
        if deadline is not None and time.monotonic()>=deadline:
            try: os.kill(pid, 9)
            except OSError: pass
            try: os.waitpid(pid,0)
            except OSError: pass
            return 124
        time.sleep(0.02)


def _forked_write_probe(target: Path, uid: int, gid: int) -> int:
    from .v05_origin_guard import require_controller_execution_origin, _forked_worker_scope
    require_controller_execution_origin("engineering-write-probe")
    pid=os.fork()
    if pid==0:
        rc=9
        try:
            with _forked_worker_scope("engineering-write-probe"):
                os.setgroups([]); os.setgid(int(gid)); os.setuid(int(uid)); os.umask(0o077)
                try: target.write_text("forbidden",encoding="utf-8")
                except (PermissionError,OSError): rc=0
                else: target.unlink(missing_ok=True); rc=9
        except BaseException: rc=8
        os._exit(rc)
    return _wait_child(pid,30)


def _forked_engineering_worker(job_path: Path, candidate: Path, uid: int, gid: int, timeout: float, error_path: Path, *, drop_identity: bool=True) -> int:
    from .v05_origin_guard import require_controller_execution_origin, _forked_worker_scope
    require_controller_execution_origin("engineering-worker-fork")
    pid=os.fork()
    if pid==0:
        rc=2
        try:
            os.chdir(candidate)
            with error_path.open("w",encoding="utf-8") as err:
                os.dup2(err.fileno(),2)
                with _forked_worker_scope("engineering-worker"):
                    if drop_identity:
                        os.setgroups([]); os.setgid(int(gid)); os.setuid(int(uid))
                    os.umask(0o077)
                    from .v05_engineering_worker import worker_main
                    worker_main(job_path); rc=0
        except BaseException:
            traceback.print_exc(); rc=2
        os._exit(rc)
    no_deadline=os.environ.get('IG_DECODER_EXECUTION_POLICY')=='NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
    return _wait_child(pid,None if no_deadline else float(timeout)+30.0)


def execute_registered_engineering_job(*, runtime, parameters: Mapping[str,Any]) -> dict[str,Any]:
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin("execute_registered_engineering_job")
    runtime._require_execution(); p=validate_job_parameters(parameters)
    if os.name!="posix" or not hasattr(os,"geteuid") or os.geteuid()!=0: raise EngineeringJobError("ENGINEERING_CONTROLLER_PERMISSION_SEPARATION_REQUIRED")
    same_identity,authorization=_worker_identity_mode(p,os.geteuid(),os.getegid())
    source_root=Path(__file__).resolve().parent.parent
    package_root=source_root/"infinity_grid"
    if engineering_source_tree_digest(source_root)!=p["parent_source_sha256"]: raise EngineeringJobError("ENGINEERING_PARENT_SOURCE_MISMATCH")
    if source_tree_digest(package_root)!=p["parent_package_sha256"]: raise EngineeringJobError("ENGINEERING_PARENT_PACKAGE_MISMATCH")
    overlay=_owned_overlay(p["overlay_path"],p["overlay_sha256"])
    stage_root=runtime._root/"engineering_job"; stage_root.mkdir(parents=True,exist_ok=True)
    durable=stage_root/"candidate_source"; record_path=stage_root/"ENGINEERING_JOB_RESULT.json"
    if record_path.is_file() and durable.is_dir():
        rec=json.loads(record_path.read_text(encoding="utf-8"))
        if rec.get("job_id")!=p["job_id"] or rec.get("candidate_source_sha256")!=p["expected_candidate_source_sha256"]: raise EngineeringJobError("ENGINEERING_JOB_REENTRY_MISMATCH")
        if engineering_source_tree_digest(durable)!=p["expected_candidate_source_sha256"]: raise EngineeringJobError("ENGINEERING_DURABLE_CANDIDATE_MISMATCH")
        return rec
    exchange=Path(tempfile.mkdtemp(prefix="ig-decoder-engineering-",dir="/tmp")); os.chmod(exchange,0o711)
    try:
        candidate=exchange/"candidate"; _copy_source(source_root,candidate); _chown_tree(candidate,p["worker_uid"],p["worker_gid"])
        input_overlay=None
        if overlay is not None:
            input_overlay=exchange/"overlay.zip"; shutil.copyfile(overlay,input_overlay); os.chmod(input_overlay,0o444)
        result_path=exchange/"worker_result.json"; result_path.touch(mode=0o600); os.chown(result_path,p["worker_uid"],p["worker_gid"])
        job={"schema_id":"IG_DECODER_REGISTERED_ENGINEERING_WORKER_JOB_V1","job_id":p["job_id"],"operation":p["operation"],
             "candidate_root":str(candidate),"overlay_path":str(input_overlay) if input_overlay else None,"overlay_sha256":p["overlay_sha256"],
             "expected_parent_source_sha256":p["parent_source_sha256"],"expected_candidate_source_sha256":p["expected_candidate_source_sha256"],
             "validation_groups":p["validation_groups"],"result_path":str(result_path)}
        job["job_sha256"]=canonical_sha256(job); job_path=exchange/"job.json"; job_path.write_text(json.dumps(job,sort_keys=True)+"\n",encoding="utf-8"); os.chmod(job_path,0o444)
        if not same_identity:
            probe_target=source_root/"infinity_grid"/"__init__.py"
            if _forked_write_probe(probe_target,p["worker_uid"],p["worker_gid"])!=0:
                raise EngineeringJobError("ENGINEERING_PARENT_WRITE_SEPARATION_FAILED")
        worker_error=exchange/"engineering_worker.stderr.log"
        worker_rc=_forked_engineering_worker(job_path,candidate,p["worker_uid"],p["worker_gid"],float(p["wall_seconds_max"]),worker_error,drop_identity=not same_identity)
        if worker_rc!=0:
            tail=worker_error.read_text(encoding="utf-8",errors="replace")[-3000:] if worker_error.exists() else ""
            raise EngineeringJobError("ENGINEERING_WORKER_FAILED:"+tail)
        raw=json.loads(result_path.read_text(encoding="utf-8"))
        if raw.get("status")!="PASS" or raw.get("candidate_source_sha256")!=p["expected_candidate_source_sha256"]: raise EngineeringJobError("ENGINEERING_WORKER_RESULT_MISMATCH")
        if raw.get("worker_identity")!={"uid":p["worker_uid"],"gid":p["worker_gid"]}: raise EngineeringJobError("ENGINEERING_WORKER_IDENTITY_MISMATCH")
        if engineering_source_tree_digest(source_root)!=p["parent_source_sha256"] or source_tree_digest(package_root)!=p["parent_package_sha256"]:
            raise EngineeringJobError("ENGINEERING_PARENT_SOURCE_CHANGED")
        # Controller owns validation topology and reduction. The worker has already
        # stopped mutating the child source. Freeze it before validation.
        _readonly_tree(candidate)
        no_deadline=os.environ.get('IG_DECODER_EXECUTION_POLICY')=='NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
        validation=run_registered_validation_groups(candidate,p["validation_groups"],workers=runtime.default_workers,worker_uid=None if same_identity else p["worker_uid"],worker_gid=None if same_identity else p["worker_gid"],wall_seconds_max=None if no_deadline else float(p["wall_seconds_max"]))
        if validation["status"]!="PASS": raise EngineeringJobError("ENGINEERING_VALIDATION_FAILED")
        if engineering_source_tree_digest(source_root)!=p["parent_source_sha256"] or source_tree_digest(package_root)!=p["parent_package_sha256"]:
            raise EngineeringJobError("ENGINEERING_PARENT_SOURCE_CHANGED")
        from .v05_optimization_acceptance import run_s8_optimization_acceptance_gate
        optimization_acceptance=run_s8_optimization_acceptance_gate(source_root,candidate)
        if durable.exists(): shutil.rmtree(durable)
        shutil.copytree(candidate,durable); _readonly_tree(durable)
        package_sha=source_tree_digest(durable/"infinity_grid")
        if package_sha!=p["expected_candidate_package_sha256"]: raise EngineeringJobError("ENGINEERING_CANDIDATE_PACKAGE_MISMATCH")
        rec={"schema_id":JOB_RESULT_SCHEMA,"status":"PASS","job_id":p["job_id"],"operation":p["operation"],
             "parent_source_sha256":p["parent_source_sha256"],"parent_package_sha256":p["parent_package_sha256"],
             "candidate_source_sha256":p["expected_candidate_source_sha256"],"candidate_package_sha256":package_sha,
             "validation_results":validation["result"]["groups"],"validation_result_sha256":validation["result_sha256"],
             "validation_execution_metadata":validation["execution_metadata"],"optimization_acceptance":optimization_acceptance,
             "worker_separation":{"separate_identity":not same_identity,"parent_write_denied":not same_identity,
                 "same_identity_authorized":same_identity,
                 "authorization_id":authorization["authorization_id"] if authorization else None,
                 "authorization_sha256":authorization["authorization_sha256"] if authorization else None,
                 "compensating_control":"PARENT_SOURCE_DIGEST_BEFORE_AND_AFTER" if same_identity else None}}
        write_json_atomic(record_path,rec); return rec
    finally:
        try:
            for q in exchange.rglob("*"):
                try: os.chown(q,os.geteuid(),os.getegid()); os.chmod(q,0o700 if q.is_dir() else 0o600)
                except OSError: pass
            shutil.rmtree(exchange,ignore_errors=True)
        except Exception: pass

def _deterministic_source_zip(source: Path, destination: Path) -> str:
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in sorted(x for x in source.rglob("*") if x.is_file()):
            rel=f.relative_to(source).as_posix(); zi=zipfile.ZipInfo("source/"+rel,date_time=(1980,1,1,0,0,0)); zi.compress_type=zipfile.ZIP_DEFLATED; zi.external_attr=(0o444&0xFFFF)<<16; z.writestr(zi,f.read_bytes())
    return _sha_file(destination)

def validate_accept_parameters(parameters: Mapping[str,Any]) -> dict[str,Any]:
    p=dict(parameters); fields={"schema_id","job_id","source_stage_id","expected_candidate_source_sha256","expected_candidate_package_sha256"}
    if set(p)!=fields or p.get("schema_id")!=ACCEPT_SPEC_SCHEMA: raise EngineeringJobError("ENGINEERING_ACCEPT_PARAMETERS")
    p["job_id"]=_safe_id(p["job_id"])
    for k in ("expected_candidate_source_sha256","expected_candidate_package_sha256"):
        if type(p[k]) is not str or len(p[k])!=64: raise EngineeringJobError("ENGINEERING_SOURCE_SHA")
    return p

def accept_registered_engineering_candidate(*, runtime, parameters: Mapping[str,Any]) -> dict[str,Any]:
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin("accept_registered_engineering_candidate")
    runtime._require_execution(); p=validate_accept_parameters(parameters)
    dep=runtime.dependency_commit(p["source_stage_id"]); job=(dep.get("result") or {}).get("engineering_job")
    if type(job) is not dict or job.get("status")!="PASS" or job.get("job_id")!=p["job_id"]: raise EngineeringJobError("ENGINEERING_ACCEPT_DEPENDENCY")
    if job.get("candidate_source_sha256")!=p["expected_candidate_source_sha256"] or job.get("candidate_package_sha256")!=p["expected_candidate_package_sha256"]: raise EngineeringJobError("ENGINEERING_ACCEPT_SOURCE_BINDING")
    source_runtime=runtime._chain_dir/"decoder_stage_runtime"/p["source_stage_id"].replace(":","__")/"engineering_job"
    candidate=source_runtime/"candidate_source"
    if engineering_source_tree_digest(candidate)!=p["expected_candidate_source_sha256"] or source_tree_digest(candidate/"infinity_grid")!=p["expected_candidate_package_sha256"]: raise EngineeringJobError("ENGINEERING_ACCEPT_CANDIDATE_CHANGED")
    release_root=runtime._chain_dir/"engineering_releases"/p["expected_candidate_source_sha256"]
    if release_root.exists():
        rec=json.loads((release_root/"ACCEPTANCE.json").read_text(encoding="utf-8")); return rec
    release_root.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for f in sorted(x for x in candidate.rglob("*") if x.is_file()): manifest.append({"path":f.relative_to(candidate).as_posix(),"sha256":_sha_file(f),"size_bytes":f.stat().st_size})
    man={"schema_id":"IG_DECODER_REGISTERED_ENGINEERING_SOURCE_MANIFEST_V1","source_sha256":p["expected_candidate_source_sha256"],"files":manifest}
    write_json_atomic(release_root/"SOURCE_MANIFEST.json",man)
    zip_sha=_deterministic_source_zip(candidate,release_root/"source.zip")
    rec={"schema_id":ACCEPT_RESULT_SCHEMA,"status":"ACCEPTED_ENGINEERING_CANDIDATE","authoritative_science_effect":"NONE",
         "job_id":p["job_id"],"source_stage_id":p["source_stage_id"],"candidate_source_sha256":p["expected_candidate_source_sha256"],
         "candidate_package_sha256":p["expected_candidate_package_sha256"],"source_manifest_sha256":canonical_sha256(man),"release_zip_sha256":zip_sha,
         "source_stage_commit_sha256":dep["commit_sha256"],"validation_results":job["validation_results"]}
    write_json_atomic(release_root/"ACCEPTANCE.json",rec); _readonly_tree(release_root); return rec

def copy_release_for_publication(controller, commits: list[dict[str,Any]], publication: dict[str,Any], publication_root: Path) -> None:
    if not commits: return
    acc=(commits[-1].get("result") or {}).get("engineering_acceptance")
    if type(acc) is not dict or acc.get("schema_id")!=ACCEPT_RESULT_SCHEMA: return
    chain=controller.chain_dir(publication["record"]["chain_id"]); src=chain/"engineering_releases"/acc["candidate_source_sha256"]
    if _sha_file(src/"source.zip")!=acc["release_zip_sha256"]: raise EngineeringJobError("ENGINEERING_PUBLICATION_RELEASE_MISMATCH")
    root=publication_root/"engineering_releases"/publication["record_sha256"]; root.mkdir(parents=True,exist_ok=True)
    for name in ("source.zip","SOURCE_MANIFEST.json","ACCEPTANCE.json"):
        dst=root/name
        if not dst.exists(): shutil.copyfile(src/name,dst)
    bind={"schema_id":"IG_DECODER_REGISTERED_ENGINEERING_PUBLICATION_BINDING_V1","record_sha256":publication["record_sha256"],
          "release_zip_sha256":acc["release_zip_sha256"],"source_manifest_sha256":acc["source_manifest_sha256"],"candidate_source_sha256":acc["candidate_source_sha256"]}
    write_json_atomic(root/"PUBLICATION_BINDING.json",bind); _readonly_tree(root)


CONTROLLER_SOURCE_TRANSITION_SCHEMA="IG_DECODER_CONTROLLER_SOURCE_TRANSITION_REGISTRATION_V1"

def _public_readonly_tree(root: Path) -> None:
    for q in sorted(root.rglob('*'),reverse=True):
        if q.is_symlink(): raise EngineeringJobError('ENGINEERING_SOURCE_SYMLINK')
        os.chmod(q,0o555 if q.is_dir() else 0o444)
    os.chmod(root,0o555)

def run_controller_registered_source_transition(*,runtime_root: str|Path,registration: Mapping[str,Any],overlay_path: str|Path,worker_uid: int|None,worker_gid: int|None,workers: int=4) -> dict[str,Any]:
    """Controller-root registered source transition used by passive C5+ engineering."""
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin('run_controller_registered_source_transition')
    reg=dict(registration)
    fields={'schema_id','job_id','operation','parent_source_sha256','parent_package_sha256','expected_candidate_source_sha256','expected_candidate_package_sha256','overlay_sha256','validation_groups'}
    if set(reg)!=fields or reg.get('schema_id')!=CONTROLLER_SOURCE_TRANSITION_SCHEMA or reg.get('operation')!='APPLY_OVERLAY_AND_VALIDATE': raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_REGISTRATION')
    registration_sha=canonical_sha256(reg)
    source_root=Path(__file__).resolve().parent.parent; package_root=source_root/'infinity_grid'
    if engineering_source_tree_digest(source_root)!=reg['parent_source_sha256'] or source_tree_digest(package_root)!=reg['parent_package_sha256']: raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_PARENT')
    overlay=Path(overlay_path).resolve(strict=True)
    if _sha_file(overlay)!=reg['overlay_sha256']: raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_OVERLAY')
    root=Path(runtime_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    final_record=root/'SOURCE_TRANSITION_RESULT.json'
    if final_record.is_file():
        obj=json.loads(final_record.read_text(encoding='utf-8'))
        if obj.get('registration_sha256')!=registration_sha: raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_REENTRY')
        return obj
    exchange=Path(tempfile.mkdtemp(prefix='ig-decoder-controller-transition-',dir='/tmp')); os.chmod(exchange,0o711)
    try:
        candidate=exchange/'candidate'; _copy_source(source_root,candidate)
        uid=os.geteuid() if worker_uid is None else int(worker_uid); gid=os.getegid() if worker_gid is None else int(worker_gid)
        _chown_tree(candidate,uid,gid)
        input_overlay=exchange/'overlay.zip'; shutil.copyfile(overlay,input_overlay); os.chmod(input_overlay,0o444)
        result_path=exchange/'worker_result.json'; result_path.touch(mode=0o600); os.chown(result_path,uid,gid)
        job={'schema_id':'IG_DECODER_REGISTERED_ENGINEERING_WORKER_JOB_V1','job_id':str(reg['job_id']),'operation':'APPLY_OVERLAY_AND_VALIDATE','candidate_root':str(candidate),'overlay_path':str(input_overlay),'overlay_sha256':reg['overlay_sha256'],'expected_parent_source_sha256':reg['parent_source_sha256'],'expected_candidate_source_sha256':reg['expected_candidate_source_sha256'],'validation_groups':list(reg['validation_groups']),'result_path':str(result_path)}
        job['job_sha256']=canonical_sha256(job); job_path=exchange/'job.json'; job_path.write_text(json.dumps(job,sort_keys=True)+'\n',encoding='utf-8'); os.chmod(job_path,0o444)
        error=exchange/'engineering_worker.stderr.log'
        if worker_uid is None:
            # Same-identity controller-descended worker is allowed only when OS separation is unavailable.
            from .v05_origin_guard import _forked_worker_scope
            pid=os.fork()
            if pid==0:
                rc=2
                try:
                    os.chdir(candidate)
                    with error.open('w',encoding='utf-8') as err:
                        os.dup2(err.fileno(),2)
                        with _forked_worker_scope('controller-source-transition-worker'):
                            from .v05_engineering_worker import worker_main
                            worker_main(job_path); rc=0
                except BaseException: traceback.print_exc(); rc=2
                os._exit(rc)
            worker_rc=_wait_child(pid,900)
        else:
            worker_rc=_forked_engineering_worker(job_path,candidate,uid,gid,900,error)
        if worker_rc!=0: raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_WORKER_FAILED')
        raw=json.loads(result_path.read_text(encoding='utf-8'))
        if raw.get('status')!='PASS' or raw.get('candidate_source_sha256')!=reg['expected_candidate_source_sha256']: raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_WORKER_RESULT')
        _readonly_tree(candidate)
        validation=run_registered_validation_groups(candidate,reg['validation_groups'],workers=int(workers),worker_uid=worker_uid,worker_gid=worker_gid,wall_seconds_max=900)
        observed_source=engineering_source_tree_digest(candidate); observed_pkg=source_tree_digest(candidate/'infinity_grid')
        if observed_source!=reg['expected_candidate_source_sha256'] or observed_pkg!=reg['expected_candidate_package_sha256']: raise EngineeringJobError('CONTROLLER_SOURCE_TRANSITION_CANDIDATE_MISMATCH')
        durable=root/'candidate_source';
        if durable.exists(): shutil.rmtree(durable)
        shutil.copytree(candidate,durable); _public_readonly_tree(durable)
        manifest=[]
        for f in sorted(x for x in durable.rglob('*') if x.is_file()): manifest.append({'path':f.relative_to(durable).as_posix(),'sha256':_sha_file(f),'size_bytes':f.stat().st_size})
        man={'schema_id':'IG_DECODER_CONTROLLER_SOURCE_TRANSITION_MANIFEST_V1','source_sha256':observed_source,'files':manifest}; write_json_atomic(root/'SOURCE_MANIFEST.json',man)
        zip_sha=_deterministic_source_zip(durable,root/'source.zip')
        out={'schema_id':'IG_DECODER_CONTROLLER_SOURCE_TRANSITION_RESULT_V1','status':'PASS','registration_sha256':registration_sha,'job_id':reg['job_id'],'parent_source_sha256':reg['parent_source_sha256'],'candidate_source_sha256':observed_source,'candidate_package_sha256':observed_pkg,'candidate_source_path':str(durable),'source_manifest_sha256':canonical_sha256(man),'source_zip_sha256':zip_sha,'validation_results':validation['result']['groups'],'validation_result_sha256':validation['result_sha256'],'validation_execution_metadata':validation['execution_metadata'],'g6_science_effect':'NONE'}
        out['result_sha256']=canonical_sha256(out); write_json_atomic(final_record,out); os.chmod(final_record,0o444); return out
    finally:
        try:
            for q in exchange.rglob('*'):
                try: os.chown(q,os.geteuid(),os.getegid()); os.chmod(q,0o700 if q.is_dir() else 0o600)
                except OSError: pass
            shutil.rmtree(exchange,ignore_errors=True)
        except Exception: pass
