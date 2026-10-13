from __future__ import annotations
import hashlib, json
from importlib.resources import files
from pathlib import Path
from typing import Any
from .canon import canonical_sha256
from .errors import ArtifactCorruptError, ContractError
from .schema import validate

MANIFEST_SCHEMA_ID="IG_V026_SCIENTIFIC_ORACLE_MANIFEST_V1"

def _resource_json(name:str)->dict:
    return json.loads(files("infinity_grid").joinpath("resources/oracle").joinpath(name).read_text(encoding="utf-8"))

def gate2_contract_registry()->dict:
    reg=_resource_json("GATE2_NATIVE_CONTRACT_REGISTRY_V1.json")
    if reg.get("schema_id")!="IG_GATE2_NATIVE_CONTRACT_REGISTRY_V1": raise ContractError("unexpected Gate2 contract registry schema")
    body={k:v for k,v in reg.items() if k!="registry_sha256"}
    if canonical_sha256(body)!=reg.get("registry_sha256"): raise ArtifactCorruptError("Gate2 contract registry hash mismatch")
    return reg

def gate2_payload_schema(schema_id:str)->dict:
    reg=gate2_contract_registry(); ent=reg.get("payload_schemas",{}).get(schema_id)
    if not ent: raise KeyError(schema_id)
    sch=_resource_json(ent["schema_path"])
    if canonical_sha256(sch)!=ent["schema_sha256"]: raise ArtifactCorruptError(f"Gate2 schema hash mismatch: {schema_id}")
    return sch

def validate_gate2_payload(payload:dict)->str:
    sid=payload.get("schema_id") if isinstance(payload,dict) else None
    if not isinstance(sid,str): raise ContractError("Gate2 payload lacks schema_id")
    try: sch=gate2_payload_schema(sid)
    except KeyError as exc: raise ContractError(f"unregistered Gate2 payload schema: {sid}") from exc
    errs=validate(sch,payload,raise_on_error=False)
    if errs: raise ContractError(f"Gate2 payload {sid} failed strict contract: {'; '.join(errs[:20])}")
    return sid

def oracle_manifest()->dict:
    m=_resource_json("V026_SCIENTIFIC_ORACLE_MANIFEST.json")
    validate_gate2_payload(m)
    if m.get("schema_id")!=MANIFEST_SCHEMA_ID: raise ContractError("unexpected oracle manifest schema")
    body={k:v for k,v in m.items() if k!="oracle_sha256"}
    if canonical_sha256(body)!=m.get("oracle_sha256"): raise ArtifactCorruptError("oracle manifest self hash mismatch")
    return m

def _file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def _pointer(obj:Any,pointer:str)->Any:
    if pointer in ("", "/"): return obj
    cur=obj
    for raw in pointer.lstrip("/").split("/"):
        key=raw.replace("~1","/").replace("~0","~")
        if isinstance(cur,list): cur=cur[int(key)]
        elif isinstance(cur,dict): cur=cur[key]
        else: raise KeyError(pointer)
    return cur

def verify_embedded_oracle()->dict:
    m=oracle_manifest()
    checked=[]
    for name in ["SCIENTIFIC_TEST_ORACLE_RESULT.json","O1_O7_PRIMARY_ORACLE.json","DETERMINISM_MATRIX_RESULT.json","REPLAY_MANIFEST_ORACLE.json","L2_ORACLE_INDEX.json","CALIBRATION_RECORD_INDEX.json"]:
        obj=_resource_json(name); validate_gate2_payload(obj); checked.append({"resource":name,"schema_id":obj["schema_id"],"canonical_sha256":canonical_sha256(obj)})
    return {"schema_id":"IG_GATE2_EMBEDDED_ORACLE_VERIFICATION_V1","status":"PASS","oracle_sha256":m["oracle_sha256"],"resources":checked}

def verify_corpus(corpus_root:str|Path)->dict:
    root=Path(corpus_root).resolve(); m=oracle_manifest(); failures=[]; checked=0; assertions=0
    external_manifest=root/"V026_SCIENTIFIC_ORACLE_MANIFEST.json"
    if not external_manifest.is_file(): failures.append({"reason":"missing_external_manifest","path":str(external_manifest)})
    else:
        try:
            ext=json.loads(external_manifest.read_text(encoding="utf-8"))
            if ext!=m: failures.append({"reason":"external_manifest_not_identical_to_embedded"})
        except Exception as exc: failures.append({"reason":"external_manifest_parse","error":f"{type(exc).__name__}: {exc}"})
    base_for_parent=root.parent
    for rec in m["corpus_files"]:
        rel=rec["path"]
        p=(root/rel).resolve() if not rel.startswith("../") else (base_for_parent/rel[3:]).resolve()
        if not p.is_file(): failures.append({"reason":"missing_file","path":rel}); continue
        checked+=1
        if p.stat().st_size!=rec["size_bytes"]: failures.append({"reason":"size_mismatch","path":rel})
        got=_file_sha(p)
        if got!=rec["sha256"]: failures.append({"reason":"sha256_mismatch","path":rel,"expected":rec["sha256"],"observed":got})
        if rec.get("canonical_sha256"):
            try: obj=json.loads(p.read_text(encoding="utf-8")); c=canonical_sha256(obj)
            except Exception as exc: failures.append({"reason":"canonical_json_error","path":rel,"error":f"{type(exc).__name__}: {exc}"}); continue
            if c!=rec["canonical_sha256"]: failures.append({"reason":"canonical_sha256_mismatch","path":rel})
    for a in m["assertions"]:
        p=root/a["file"]
        try:
            obj=json.loads(p.read_text(encoding="utf-8")); got=_pointer(obj,a["pointer"]); assertions+=1
            if got!=a["equals"]: failures.append({"reason":"assertion_mismatch","file":a["file"],"pointer":a["pointer"],"expected":a["equals"],"observed":got})
        except Exception as exc: failures.append({"reason":"assertion_error","file":a["file"],"pointer":a["pointer"],"error":f"{type(exc).__name__}: {exc}"})
    return {"schema_id":"IG_GATE2_CORPUS_VERIFICATION_V1","status":"PASS" if not failures else "FAIL","oracle_sha256":m["oracle_sha256"],"files_checked":checked,"assertions_checked":assertions,"failure_count":len(failures),"failures":failures}
