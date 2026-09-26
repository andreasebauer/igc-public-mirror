from __future__ import annotations
import ast, importlib, inspect, json, textwrap
from importlib.resources import files
from typing import Any
from .canon import canonical_sha256

ACTION_REGISTRY_RESOURCE = "decoder/O_NATIVE_ACTION_REGISTRY_V1.json"
BASELINE_RESOURCE = "decoder/O_NATIVE_SEMANTICS_BASELINE_V1.json"

class NativeSemanticsError(RuntimeError): pass
class NativeSemanticsReopenRequired(NativeSemanticsError): pass

def _resource_json(rel: str) -> dict:
    return json.loads(files("infinity_grid").joinpath("resources").joinpath(rel).read_text(encoding="utf-8"))

def _resolve_callable(ref: str):
    if ":" not in ref: raise NativeSemanticsError(f"invalid callable ref: {ref}")
    module_name, path = ref.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in path.split("."): obj = getattr(obj, part)
    if not callable(obj): raise NativeSemanticsError(f"not callable: {ref}")
    return obj

CALLABLE_HASH_SCHEME = "EXACT_NORMALIZED_SOURCE_V1"

def callable_semantic_sha256(ref: str) -> str:
    """Hash exact callable source in a Python-minor-version-independent form.

    AST dumps are not a stable scientific identity across supported CPython
    releases.  The repair scheme binds the exact dedented source text with
    normalized newlines/trailing whitespace.  Any source edit therefore
    reopens fail-closed, while identical bytes hash identically on 3.11-3.13.
    """
    try:
        source = textwrap.dedent(inspect.getsource(_resolve_callable(ref)))
    except (OSError, TypeError) as exc:
        raise NativeSemanticsError(f"cannot inspect {ref}") from exc
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    source = "\n".join(line.rstrip() for line in source.split("\n")).strip("\n") + "\n"
    return canonical_sha256({"scheme": CALLABLE_HASH_SCHEME, "callable": ref, "source": source})

def action_registry() -> dict:
    reg = _resource_json(ACTION_REGISTRY_RESOURCE)
    if reg.get("schema_id") != "IG_O_NATIVE_ACTION_REGISTRY_V1": raise NativeSemanticsError("unexpected action registry")
    if canonical_sha256({k:v for k,v in reg.items() if k != "registry_sha256"}) != reg.get("registry_sha256"):
        raise NativeSemanticsError("action registry hash mismatch")
    return reg

def native_semantics_snapshot() -> dict:
    reg = action_registry(); actions=[]
    for row in reg["actions"]:
        actions.append({
            "action_id": row["action_id"], "callable": row["callable"],
            "callable_semantic_sha256": callable_semantic_sha256(row["callable"]),
            "relation_arity": row["relation_arity"], "successor_semantics": row["successor_semantics"],
            "read_set": sorted(row["read_set"]), "writes": sorted(row["writes"]),
            "hidden_reads": sorted(row.get("hidden_reads", [])),
            "features": dict(sorted(row.get("features", {}).items())), "scientific_role": row["scientific_role"],
        })
    actions.sort(key=lambda x:x["action_id"])
    out={"schema_id":"IG_O_NATIVE_SEMANTICS_SNAPSHOT_V1","hash_scheme":CALLABLE_HASH_SCHEME,"action_registry_sha256":reg["registry_sha256"],"actions":actions}
    out["science_sha256"]=canonical_sha256(out); return out

def native_semantics_baseline() -> dict:
    b=_resource_json(BASELINE_RESOURCE)
    if b.get("schema_id") != "IG_O_NATIVE_SEMANTICS_BASELINE_V1": raise NativeSemanticsError("unexpected native baseline")
    if canonical_sha256({k:v for k,v in b.items() if k != "baseline_sha256"}) != b.get("baseline_sha256"):
        raise NativeSemanticsError("native baseline hash mismatch")
    return b

def compare_native_semantics(previous: dict, current: dict) -> dict:
    prev={x["action_id"]:x for x in previous.get("actions",[])}; cur={x["action_id"]:x for x in current.get("actions",[])}; changes=[]
    for aid in sorted(set(prev)|set(cur)):
        if aid not in prev: changes.append({"action_id":aid,"change":"NEW_ACTION"})
        elif aid not in cur: changes.append({"action_id":aid,"change":"REMOVED_ACTION"})
        elif prev[aid] != cur[aid]: changes.append({"action_id":aid,"change":"ACTION_CONTRACT_OR_IMPLEMENTATION_CHANGED","fields":[k for k in sorted(set(prev[aid])|set(cur[aid])) if prev[aid].get(k)!=cur[aid].get(k)]})
    return {"schema_id":"IG_O_NATIVE_SEMANTICS_COMPARISON_V1","classification":"REOPEN_REQUIRED" if changes else "NATIVE_SEMANTICS_UNCHANGED","changes":changes,"previous_science_sha256":previous.get("science_sha256"),"current_science_sha256":current.get("science_sha256")}

def verify_native_semantics(*, raise_on_change: bool=True) -> dict:
    b=native_semantics_baseline(); cur=native_semantics_snapshot(); cmp=compare_native_semantics(b["snapshot"],cur)
    out={"schema_id":"IG_O_NATIVE_SEMANTICS_SENTINEL_RESULT_V1","status":"PASS" if cmp["classification"]=="NATIVE_SEMANTICS_UNCHANGED" else "REOPEN_REQUIRED","classification":cmp["classification"],"baseline_sha256":b["baseline_sha256"],"comparison":cmp,"current_snapshot":cur,"reopen_action":"STOP_FAIL_CLOSED_AND_REQUIRE_THEOREM_IMPACT_ANALYSIS_THEN_TARGETED_EXACT_AUDIT_IF_NEEDED"}
    out["science_sha256"]=canonical_sha256(out)
    if raise_on_change and out["status"]!="PASS": raise NativeSemanticsReopenRequired(f"native semantics changed: {cmp['changes']}")
    return out

def current_action_contract() -> dict:
    v=verify_native_semantics(raise_on_change=True); rel=[a for a in v["current_snapshot"]["actions"] if a["scientific_role"]=="RELATION_ADD_SEMANTICS"]
    if not rel: raise NativeSemanticsError("no relation action")
    ar={a["relation_arity"] for a in rel}; ss={a["successor_semantics"] for a in rel}
    if len(ar)!=1 or len(ss)!=1: raise NativeSemanticsReopenRequired("relation actions disagree")
    fkeys=sorted(set().union(*(set(a["features"]) for a in rel)))
    return {"relation_arity":next(iter(ar)),"successor_semantics":next(iter(ss)),"read_set":sorted(set().union(*(set(a["read_set"]) for a in rel))),"hidden_reads":sorted(set().union(*(set(a["hidden_reads"]) for a in rel))),"writes":sorted(set().union(*(set(a["writes"]) for a in rel))),"forbidden_features_present":{k:any(bool(a["features"].get(k)) for a in rel) for k in fkeys},"native_semantics_science_sha256":v["current_snapshot"]["science_sha256"],"native_semantics_baseline_sha256":v["baseline_sha256"]}
