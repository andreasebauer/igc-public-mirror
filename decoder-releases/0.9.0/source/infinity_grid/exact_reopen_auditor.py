from __future__ import annotations
from pathlib import Path
from .canon import canonical_sha256
from .exact_carrier_unblinding import run_exact_carrier_targeted_audit
from .semantic_sentinel import verify_native_semantics
from .theorem_registry import verify_all_theorems
class ExactReopenEvidenceMissing(RuntimeError): pass
def reopen_status():
    s=verify_native_semantics(raise_on_change=False); t=verify_all_theorems(raise_on_stale=False); req=s["status"]!="PASS" or t["status"]!="PASS"
    out={"schema_id":"IG_EXACT_REOPEN_STATUS_V1","reopen_required":req,"native_semantics_status":s["status"],"theorem_status":t["status"],"scientific_rule":"EXACT_AUDIT_ONLY_AFTER_EXPLICIT_REOPEN; NEVER_INVENT_MISSING_PHASE8_OR_PRE_O7_EXACT_HISTORY"}; out["science_sha256"]=canonical_sha256(out); return out
def run_exact_reopen_audit(*,phase8_seed:Path|None,output:Path,reopen_reason:str,start_level:int=7,through:int=9,allow_full_panel:bool=False):
    if not reopen_reason: raise ValueError("explicit reopen_reason is mandatory")
    if phase8_seed is None or not Path(phase8_seed).is_file(): raise ExactReopenEvidenceMissing("targeted exact audit needs full Phase8 seed bytes; expected digest is pinned but archive bytes are absent. Fail closed.")
    return run_exact_carrier_targeted_audit(Path(phase8_seed),Path(output),start_level=start_level,through=through,reopen_reason=reopen_reason,allow_full_panel=allow_full_panel)
