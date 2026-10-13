from __future__ import annotations
import json
from importlib.resources import files
from .canon import canonical_sha256
from .semantic_sentinel import verify_native_semantics
REGISTRY_RESOURCE="decoder/THEOREM_PREMISE_REGISTRY_V1.json"
class TheoremPremiseError(RuntimeError): pass
class StaleTheoremError(TheoremPremiseError): pass

def _r(rel): return json.loads(files("infinity_grid").joinpath("resources").joinpath(rel).read_text(encoding="utf-8"))
def theorem_premise_registry():
    r=_r(REGISTRY_RESOURCE)
    if r.get("schema_id")!="IG_THEOREM_PREMISE_REGISTRY_V1": raise TheoremPremiseError("unexpected registry")
    if canonical_sha256({k:v for k,v in r.items() if k!="registry_sha256"})!=r.get("registry_sha256"): raise TheoremPremiseError("registry hash mismatch")
    return r
def current_premise_snapshot():
    sent=verify_native_semantics(raise_on_change=False)
    out={"schema_id":"IG_THEOREM_PREMISE_SNAPSHOT_V1","resources":{n:canonical_sha256(_r("decoder/"+n+".json")) for n in ["O_REGIME_EARNED_LAW_REGISTRY_v1","O_REGIME_ADAPTIVE_SCANNER_SPEC_v1","GRRL_APPLICATION_GATE_RECONCILED_v0.1"]},"native_semantics":{"status":sent["status"],"snapshot_science_sha256":sent["current_snapshot"]["science_sha256"],"baseline_sha256":sent["baseline_sha256"]}}
    out["science_sha256"]=canonical_sha256(out); return out
def premise_impact(previous,current):
    changes=[]
    for k in sorted(set(previous.get("resources",{}))|set(current.get("resources",{}))):
        if previous.get("resources",{}).get(k)!=current.get("resources",{}).get(k): changes.append({"premise":"resource:"+k,"previous":previous.get("resources",{}).get(k),"current":current.get("resources",{}).get(k)})
    if previous.get("native_semantics")!=current.get("native_semantics"): changes.append({"premise":"native_semantics","previous":previous.get("native_semantics"),"current":current.get("native_semantics")})
    return {"schema_id":"IG_THEOREM_PREMISE_IMPACT_RESULT_V1","classification":"STALE_REOPEN_REQUIRED" if changes else "PREMISES_UNCHANGED","changes":changes}
def verify_theorem(theorem_id,*,raise_on_stale=True):
    reg=theorem_premise_registry(); rows={x["theorem_id"]:x for x in reg["theorems"]}
    if theorem_id not in rows: raise TheoremPremiseError("unknown theorem: "+theorem_id)
    e=rows[theorem_id]; cur=current_premise_snapshot(); imp=premise_impact(e["premise_snapshot"],cur); stale=imp["classification"]!="PREMISES_UNCHANGED"
    out={"schema_id":"IG_THEOREM_PREMISE_VERIFICATION_RESULT_V1","theorem_id":theorem_id,"status":"STALE_REOPEN_REQUIRED" if stale else "PASS","registered_scope":e["scope"],"impact":imp,"reopen_triggers":e["reopen_triggers"],"registry_sha256":reg["registry_sha256"],"current_premise_science_sha256":cur["science_sha256"]}; out["science_sha256"]=canonical_sha256(out)
    if stale and raise_on_stale: raise StaleTheoremError(f"{theorem_id} stale: {imp['changes']}")
    return out
def verify_all_theorems(*,raise_on_stale=True):
    reg=theorem_premise_registry(); rr=[verify_theorem(x["theorem_id"],raise_on_stale=False) for x in reg["theorems"]]; stale=[x["theorem_id"] for x in rr if x["status"]!="PASS"]
    out={"schema_id":"IG_THEOREM_PREMISE_REGISTRY_VERIFICATION_V1","status":"PASS" if not stale else "STALE_REOPEN_REQUIRED","stale_theorems":stale,"results":rr,"registry_sha256":reg["registry_sha256"]}; out["science_sha256"]=canonical_sha256(out)
    if stale and raise_on_stale: raise StaleTheoremError(f"stale theorem certificates: {stale}")
    return out
