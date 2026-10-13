from __future__ import annotations

"""Unprivileged worker for registered Decoder engineering jobs.

The worker can modify only its controller-prepared child source tree. It accepts
no command line from a service request; validation groups map to frozen commands.
"""

import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256

JOB_SCHEMA = "IG_DECODER_REGISTERED_ENGINEERING_WORKER_JOB_V1"
RESULT_SCHEMA = "IG_DECODER_REGISTERED_ENGINEERING_WORKER_RESULT_V1"
OVERLAY_SCHEMA = "IG_DECODER_REGISTERED_ENGINEERING_OVERLAY_V1"
SOURCE_SCHEMA = "IG_DECODER_ENGINEERING_SOURCE_TREE_V1"

from .v05_validation_runtime import REGISTERED_VALIDATION_GROUPS as VALIDATION_GROUPS


class EngineeringWorkerError(RuntimeError): pass

def _sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def _safe_rel(value: str) -> str:
    p=Path(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value:
        raise EngineeringWorkerError("ENGINEERING_RELATIVE_PATH_REQUIRED")
    return p.as_posix()

def engineering_source_tree_digest(root: str | Path) -> str:
    root=Path(root).resolve(strict=True); rows=[]
    skip={".git","__pycache__",".pytest_cache","build","dist",".engineering_tmp"}
    for p in sorted(root.rglob("*")):
        rel=p.relative_to(root)
        if any(x in skip for x in rel.parts) or p.suffix in {".pyc",".pyo"}: continue
        if p.is_symlink(): raise EngineeringWorkerError("ENGINEERING_SOURCE_SYMLINK")
        if p.is_file(): rows.append([rel.as_posix(),_sha_file(p)])
    return canonical_sha256({"schema_id":SOURCE_SCHEMA,"files":rows})

def validate_overlay_archive(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    p=Path(path).resolve(strict=True)
    if _sha_file(p)!=expected_sha256: raise EngineeringWorkerError("ENGINEERING_OVERLAY_SHA_MISMATCH")
    with zipfile.ZipFile(p) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or "OVERLAY_MANIFEST.json" not in names:
            raise EngineeringWorkerError("ENGINEERING_OVERLAY_LAYOUT")
        manifest=json.loads(z.read("OVERLAY_MANIFEST.json"))
        if type(manifest) is not dict or set(manifest)!={"schema_id","files","delete_paths","overlay_sha256"} or manifest["schema_id"]!=OVERLAY_SCHEMA:
            raise EngineeringWorkerError("ENGINEERING_OVERLAY_MANIFEST")
        if type(manifest["overlay_sha256"]) is not str or len(manifest["overlay_sha256"])!=64: raise EngineeringWorkerError("ENGINEERING_OVERLAY_BINDING")
        files=manifest["files"]
        if type(files) is not list or any(type(x) is not dict or set(x)!={"path","sha256"} for x in files):
            raise EngineeringWorkerError("ENGINEERING_OVERLAY_FILES")
        declared=[]
        for row in files:
            rel=_safe_rel(row["path"]); declared.append("files/"+rel)
            if len(row["sha256"])!=64: raise EngineeringWorkerError("ENGINEERING_OVERLAY_FILE_SHA")
        deletes=manifest["delete_paths"]
        if type(deletes) is not list or any(type(x) is not str for x in deletes): raise EngineeringWorkerError("ENGINEERING_OVERLAY_DELETES")
        deletes=[_safe_rel(x) for x in deletes]
        allowed={"OVERLAY_MANIFEST.json",*declared}
        if set(names)!=allowed: raise EngineeringWorkerError("ENGINEERING_OVERLAY_UNLISTED_MEMBER")
        for row in files:
            data=z.read("files/"+_safe_rel(row["path"]))
            if hashlib.sha256(data).hexdigest()!=row["sha256"]: raise EngineeringWorkerError("ENGINEERING_OVERLAY_FILE_MISMATCH")
        return {"files":files,"delete_paths":deletes}

def _apply_overlay(candidate: Path, overlay: Path, expected_sha: str) -> None:
    manifest=validate_overlay_archive(overlay,expected_sha)
    with zipfile.ZipFile(overlay) as z:
        for rel in manifest["delete_paths"]:
            p=(candidate/rel).resolve()
            if not p.is_relative_to(candidate.resolve()): raise EngineeringWorkerError("ENGINEERING_OVERLAY_DESTINATION")
            if p.exists():
                if p.is_dir(): shutil.rmtree(p)
                else: p.unlink()
        for row in manifest["files"]:
            rel=_safe_rel(row["path"]); dst=(candidate/rel).resolve()
            if not dst.is_relative_to(candidate.resolve()): raise EngineeringWorkerError("ENGINEERING_OVERLAY_DESTINATION")
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(z.read("files/"+rel))

def compact_validation_tail(output: str, limit: int = 16000) -> str:
    lines=output.splitlines()
    failed=[line for line in lines if line.startswith("FAILED ")]
    summaries=[line for line in lines if (" failed" in line or " passed" in line or " error" in line or " skipped" in line) and (" in " in line or line.startswith("="))]
    selected=failed + summaries[-3:]
    if not selected:
        selected=lines[-40:]
    return "\n".join(selected)[-limit:]

def _run_group(*_args, **_kwargs):
    raise EngineeringWorkerError("ENGINEERING_VALIDATION_OWNED_BY_CONTROLLER")

def worker_main(job_path: str | Path) -> dict[str, Any]:
    from .v05_origin_guard import require_worker_execution_origin
    require_worker_execution_origin("v05_engineering_worker.worker_main")
    job=json.loads(Path(job_path).read_text(encoding="utf-8"))
    required={"schema_id","job_id","operation","candidate_root","overlay_path","overlay_sha256",
              "expected_parent_source_sha256","expected_candidate_source_sha256","validation_groups","result_path","job_sha256"}
    if type(job) is not dict or set(job)!=required or job.get("schema_id")!=JOB_SCHEMA:
        raise EngineeringWorkerError("ENGINEERING_WORKER_JOB_SCHEMA")
    base={k:v for k,v in job.items() if k!="job_sha256"}
    if canonical_sha256(base)!=job["job_sha256"]: raise EngineeringWorkerError("ENGINEERING_WORKER_JOB_HASH")
    candidate=Path(job["candidate_root"]).resolve(strict=True)
    if engineering_source_tree_digest(candidate)!=job["expected_parent_source_sha256"]:
        raise EngineeringWorkerError("ENGINEERING_PARENT_COPY_MISMATCH")
    op=job["operation"]
    if op=="APPLY_OVERLAY_AND_VALIDATE":
        if not isinstance(job["overlay_path"],str) or not isinstance(job["overlay_sha256"],str): raise EngineeringWorkerError("ENGINEERING_OVERLAY_REQUIRED")
        _apply_overlay(candidate,Path(job["overlay_path"]),job["overlay_sha256"])
    elif op=="VALIDATE_SOURCE":
        if job["overlay_path"] is not None or job["overlay_sha256"] is not None: raise EngineeringWorkerError("ENGINEERING_OVERLAY_NOT_ALLOWED")
    else: raise EngineeringWorkerError("ENGINEERING_OPERATION_NOT_REGISTERED")
    observed=engineering_source_tree_digest(candidate)
    if observed!=job["expected_candidate_source_sha256"]: raise EngineeringWorkerError("ENGINEERING_CANDIDATE_SOURCE_MISMATCH")
    groups=job["validation_groups"]
    if type(groups) is not list or not groups or len(groups)!=len(set(groups)) or any(type(x) is not str for x in groups):
        raise EngineeringWorkerError("ENGINEERING_VALIDATION_GROUPS")
    # C5: the engineering worker does not execute validation. It only applies
    # the registered source mutation and reports the observed candidate digest.
    shutil.rmtree(candidate/".engineering_tmp",ignore_errors=True)
    out={"schema_id":RESULT_SCHEMA,"status":"PASS","job_id":job["job_id"],"operation":op,
         "candidate_source_sha256":observed,
         "worker_identity":{"uid":os.getuid() if hasattr(os,"getuid") else None,"gid":os.getgid() if hasattr(os,"getgid") else None}}
    rp=Path(job["result_path"]); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(out,sort_keys=True)+"\n",encoding="utf-8")
    return out
