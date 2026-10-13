from __future__ import annotations

import argparse
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .checkpoints import CheckpointManager
from .datasets import DatasetStore
from .records import live_source_sha256, runtime_sha256
from .store import ArtifactStore
from .v05 import V05RegistrationStore
from .v05_evidence import ENVELOPE_SCHEMA
from .v05_verifier import VERIFICATION_SCHEMA
from .v05_workflow import ControlledWorkflowEngine, validate_workflow_registration

VERIFIER_ID = "v05.g5-s1.independent"
ORACLE_RESOURCE = "resources/v05/G5_S1_PREREGISTERED_ORACLE_V1.json"
SPEC_RESOURCE = "resources/uplift/G5_S1_PAIR_CONNECTION_CENSUS_SPEC_V1.json"
ORACLE_RESOURCE_V2 = "resources/v05/G5_S1_PREREGISTERED_ORACLE_V2.json"
SPEC_RESOURCE_V2 = "resources/uplift/G5_S1_PAIR_CONNECTION_CENSUS_SPEC_V2.json"
PASS_ALT = "G5_S0_G4_AUTHORITY_FROZEN_CAPS7_PLUS_H_CLASS_BAG_PUBLIC_ACTUAL_G4_HIDDEN_DECORATED_TREE_CHALLENGE_CORPUS_EARNED_S1_DESIGN_REVIEW"
FAIL_ALT = "G5_S0_AUTHORITY_OR_INTERFACE_FREEZE_FAILURE"
LOCK_ALT = "G5_S1_PREREGISTRATION_REQUIRED"


def _check(checks: list[dict], check_id: str, ok: bool, detail: Any = None) -> None:
    rec = {"check_id": check_id, "status": "PASS" if ok else "FAIL"}
    if detail is not None:
        rec["detail"] = detail
    checks.append(rec)


def _load_hashed_resource(name: str, field: str = "science_sha256") -> dict:
    obj = json.loads(files("infinity_grid").joinpath(name).read_text(encoding="utf-8"))
    declared = obj.get(field)
    observed = canonical_sha256({k: v for k, v in obj.items() if k != field})
    if declared != observed:
        raise RuntimeError(f"resource hash mismatch: {name}")
    return obj


def _load_single_registration(root: Path) -> dict:
    found = sorted(p for p in Path(root).rglob("*.json") if p.is_file())
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one workflow registration JSON, found {len(found)}")
    return validate_workflow_registration(json.loads(found[0].read_text(encoding="utf-8")))


def _science_predicate(result: dict, wf_reg: dict, oracle: dict, spec: dict) -> tuple[bool, dict]:
    details: dict[str, Any] = {}
    stages=result.get("stage_results") or []
    details["stage_count"]=len(stages)
    if len(stages)<2 or stages[0].get("stage")!="S0" or stages[0].get("outcome")!="PASS" or stages[1].get("stage")!="S1":
        return False, dict(details,failure="S0_S1_RESULT_MISSING")
    s1=stages[1]; details.update(s1_outcome=s1.get("outcome"),s1_alternative=s1.get("observed_alternative"))
    if s1.get("outcome") not in set(oracle["allowed_s1_outcomes"]):
        return False,dict(details,failure="S1_OUTCOME_OUTSIDE_ORACLE")
    if s1.get("observed_alternative") not in set(oracle["allowed_s1_alternatives"]):
        return False,dict(details,failure="S1_ALTERNATIVE_OUTSIDE_ORACLE")
    if result.get("authority_effect")!=oracle["required_authority_effect"] or bool(result.get("graduated")):
        return False,dict(details,failure="AUTHORITY_OR_GRADUATION_MISMATCH")
    q=wf_reg.get("question",{}).get("stage_rows",[])[1]
    if q.get("parameters",{}).get("spec_science_sha256")!=spec["science_sha256"] or oracle.get("spec_science_sha256")!=spec["science_sha256"]:
        return False,dict(details,failure="SPEC_BINDING_MISMATCH")
    params=q.get("parameters",{})
    if "required_s0_stage_science_sha256" in params:
        if params.get("required_s0_stage_science_sha256")!=oracle.get("required_s0_stage_science_sha256") or stages[0].get("science_sha256")!=oracle.get("required_s0_stage_science_sha256"):
            return False,dict(details,failure="S0_STAGE_AUTHORITY_BINDING_MISMATCH")
    elif params.get("required_s0_science_sha256")!=oracle.get("required_s0_science_sha256"):
        return False,dict(details,failure="S0_AUTHORITY_BINDING_MISMATCH")
    sr=s1.get("result") or {}
    hist=oracle.get("historical_compatibility")
    if hist is not None and s1.get("outcome")=="PASS":
        proj=[{"left":r["left"],"right":r["right"],"operator":r["operator"],"input_public_key":r["input_public_key"],"public_signature":r["public_signature"]} for r in (sr.get("rows") or [])]
        details["historical_public_projection_sha256"]=canonical_sha256(proj)
        if details["historical_public_projection_sha256"] != hist.get("historical_public_projection_sha256"):
            return False,dict(details,failure="HISTORICAL_PUBLIC_PROJECTION_MISMATCH")
        if s1.get("observed_alternative") != hist.get("required_historical_s1_classification"):
            return False,dict(details,failure="HISTORICAL_S1_CLASSIFICATION_MISMATCH")
    if s1.get("outcome")=="PASS":
        if sr.get("census_row_count")!=496 or sr.get("ordered_pair_count")!=16 or sr.get("operator_count")!=31 or sr.get("all_latent_owner_pairs_quantified") is not True:
            return False,dict(details,failure="CENSUS_COMPLETENESS_MISMATCH")
        if any(bool(sr.get(k)) for k in ("hidden_owner_identity_promoted","hidden_topology_promoted","g4_changed","g5_composition_law_earned","g5_graduated")):
            return False,dict(details,failure="FORBIDDEN_PROMOTION_FLAG")
        if len(stages)!=3 or stages[2].get("stage")!="S2" or stages[2].get("outcome")!="REVIEW_REQUIRED" or stages[2].get("observed_alternative")!=oracle["required_post_s1_stop"]:
            return False,dict(details,failure="S2_LOCK_MISMATCH")
        stop=result.get("stop_record") or {}
        if stop.get("stage")!="S2" or stop.get("reason")!="REVIEW_REQUIRED":
            return False,dict(details,failure="POST_S1_STOP_MISMATCH")
        details["scientific_disposition"]="S1_PASS_S2_REVIEW_REQUIRED"
    else:
        stop=result.get("stop_record") or {}
        if stop.get("stage")!="S1" or stop.get("reason") not in {"REVIEW_REQUIRED","BLOCKED"}:
            return False,dict(details,failure="S1_NONPASS_STOP_MISMATCH")
        details["scientific_disposition"]="S1_REVIEW_OR_BLOCKED"
    return True,details


def verify_g5_s1_run(paths, run_id: str) -> dict:
    checks: list[dict] = []
    run_dir = Path(paths.runs) / run_id
    required = {name: run_dir / name for name in ("run.json", "plan.json", "run_core.json", "v05_envelope.json")}
    _check(checks, "records_present", all(p.is_file() for p in required.values()))
    if not all(p.is_file() for p in required.values()):
        return _finish(run_id, None, None, checks, None, None)
    try:
        run = json.loads(required["run.json"].read_text())
        plan = json.loads(required["plan.json"].read_text())
        core = json.loads(required["run_core.json"].read_text())
        env = json.loads(required["v05_envelope.json"].read_text())
        _check(checks, "records_json", True)
    except Exception as exc:
        _check(checks, "records_json", False, str(exc))
        return _finish(run_id, None, None, checks, None, None)

    env_sha = env.get("envelope_sha256")
    _check(checks, "envelope_hash", env.get("schema_id") == ENVELOPE_SCHEMA and env_sha == canonical_sha256({k:v for k,v in env.items() if k != "envelope_sha256"}))
    _check(checks, "execution_terminal", run.get("lifecycle") == "COMPLETE_VALID" and env.get("execution_lifecycle") == "COMPLETE_VALID")

    reg = None
    try:
        reg_sha = plan.get("v05", {}).get("registration_sha256")
        reg = V05RegistrationStore(paths.store).get(reg_sha)
        _check(checks, "registration_binding", env.get("registration_sha256") == reg_sha and reg.get("runner") == "adapter.v05_workflow" and reg.get("verification_policy", {}).get("verifier_id") == VERIFIER_ID and reg.get("protocol",{}).get("protocol_id") == "G5_WORKFLOW")
    except Exception as exc:
        _check(checks, "registration_binding", False, str(exc))

    if reg is not None:
        _check(checks, "source_identity", reg["source_sha256"] == run.get("code_identity",{}).get("source_sha256") == env.get("source_sha256") == live_source_sha256())
        _check(checks, "environment_identity", reg["environment_sha256"] == run.get("environment_identity",{}).get("runtime_sha256") == env.get("environment_sha256") == runtime_sha256())
        _check(checks, "input_binding", reg["input_datasets"] == plan.get("input_datasets") == run.get("input_artifacts") == env.get("input_artifacts"))
        _check(checks, "subject_binding", reg.get("subject") == plan.get("subject") == run.get("subject") and reg.get("subject",{}).get("phase") == "G5")
    else:
        for cid in ("source_identity","environment_identity","input_binding","subject_binding"):
            _check(checks, cid, False, "registration unavailable")

    _check(checks, "plan_hash", plan.get("plan_sha256") == canonical_sha256({k:v for k,v in plan.items() if k != "plan_sha256"}) == env.get("plan_sha256"))
    _check(checks, "run_core_hash", core.get("run_core_sha256") == canonical_sha256({k:v for k,v in core.items() if k != "run_core_sha256"}) == run.get("run_core_sha256") == env.get("run_core_sha256"))

    store = ArtifactStore(paths.store); ds_store = DatasetStore(store)
    _check(checks, "input_dataset_bytes", all(ds_store.verify(d["dataset_sha256"]).get("status") == "PASS" for d in run.get("input_artifacts", [])))

    result_obj = None
    refs = run.get("result_artifacts", [])
    logical = reg.get("output_contract",{}).get("logical_outputs",[]) if reg else []
    _check(checks, "logical_output_set", len(logical) == 1 and {x.get("logical_name") for x in refs} == set(logical))
    result_ref = next((x for x in refs if logical and x.get("logical_name") == logical[0]), None)
    if result_ref:
        try:
            _check(checks, "result_artifact_bytes", store.verify(result_ref["sha256"], result_ref.get("size_bytes")).get("status") == "PASS")
            result_obj = json.loads(store.blob_path(result_ref["sha256"]).read_text())
            _check(checks, "result_json", isinstance(result_obj, dict) and result_obj.get("science_sha256") == canonical_sha256({k:v for k,v in result_obj.items() if k != "science_sha256"}))
        except Exception as exc:
            _check(checks, "result_artifact_bytes", False, str(exc)); _check(checks, "result_json", False, str(exc))
    else:
        _check(checks, "result_artifact_bytes", False, "principal result missing"); _check(checks, "result_json", False, "principal result missing")

    try:
        cm = CheckpointManager(run_dir, store)
        cp = cm.current(reg["stage_id"]) if reg else None
        ptr = cm.current_pointer(reg["stage_id"]) if reg else None
        hits = [x for x in env.get("checkpoint_bindings",[]) if reg and x.get("stage_id") == reg["stage_id"]]
        ok = bool(cp and ptr and cp.get("status") == "COMPLETE_VALID" and len(hits) == 1 and hits[0]["checkpoint_sha256"] == ptr["checkpoint_sha256"] and hits[0]["checkpoint_content_sha256"] == cp["checkpoint_content_sha256"])
        _check(checks, "checkpoint_binding", ok)
        wb = (cp or {}).get("stage_result",{}).get("worker_boundary",{})
        _check(checks, "worker_boundary", bool(reg and wb.get("kind") == "POSIX_DROP_PRIVILEGE" and wb.get("uid") == reg["worker_policy"]["uid"] and wb.get("gid") == reg["worker_policy"]["gid"] and wb.get("store_write_access") == "DENIED_BY_POSIX" and wb.get("publication_write_access") == "DENIED_BY_POSIX"))
    except Exception as exc:
        _check(checks, "checkpoint_binding", False, str(exc)); _check(checks, "worker_boundary", False, str(exc))

    recomputed = None; predicate_detail = None
    if reg is not None and len(reg.get("input_datasets",[])) == 1:
        try:
            with tempfile.TemporaryDirectory(prefix="ig-g5-s1-verify-") as td:
                fixture = ds_store.materialize(reg["input_datasets"][0]["dataset_sha256"], Path(td)/"workflow")
                wf_reg = _load_single_registration(fixture)
                recomputed = ControlledWorkflowEngine().run(wf_reg)
            _check(checks, "cold_recompute_exact", result_obj is not None and canonical_sha256(result_obj) == canonical_sha256(recomputed))
            repair_v2 = wf_reg.get("registration_id") == "g5-s1-pair-census-repair-v2"
            oracle = _load_hashed_resource(ORACLE_RESOURCE_V2 if repair_v2 else ORACLE_RESOURCE)
            spec = _load_hashed_resource(SPEC_RESOURCE_V2 if repair_v2 else SPEC_RESOURCE)
            _check(checks, "preregistered_oracle_binding", reg.get("output_contract",{}).get("science_oracle") == oracle)
            ok, predicate_detail = _science_predicate(recomputed, wf_reg, oracle, spec)
            _check(checks, "science_predicate", ok, predicate_detail)
            _check(checks, "stored_science_predicate", bool(result_obj is not None and _science_predicate(result_obj, wf_reg, oracle, spec)[0]))
            _check(checks, "g4_authority_frozen", spec["authority"]["g4_status"] == "GRADUATED_AND_FROZEN" and spec["authority"]["g4_public_descriptor"] == "CAPS7_PLUS_H_CLASS_BAG" and spec["authority"]["g4_completion_science_sha256"] == "44db26880449be8b98e6ac332849db8bfb6ff7f731ced6b58c0a6394dcf7ac59")
        except Exception as exc:
            for cid in ("cold_recompute_exact","preregistered_oracle_binding","science_predicate","stored_science_predicate","g4_authority_frozen"):
                _check(checks, cid, False, str(exc))
    else:
        for cid in ("cold_recompute_exact","preregistered_oracle_binding","science_predicate","stored_science_predicate","g4_authority_frozen"):
            _check(checks, cid, False, "input registration unavailable")

    req = reg.get("verification_policy",{}).get("required_checks",[]) if reg else []
    seen = {c["check_id"] for c in checks}
    _check(checks, "required_check_coverage", bool(req) and all(x in seen for x in req), {"required":req,"seen":sorted(seen)})
    return _finish(run_id, env_sha, reg, checks, recomputed, predicate_detail)


def _finish(run_id: str, env_sha: str | None, reg: dict | None, checks: list[dict], recomputed: dict | None, predicate_detail: dict | None) -> dict:
    required = reg.get("verification_policy",{}).get("required_checks",[]) if reg else []
    cmap = {c["check_id"]:c["status"] for c in checks}
    status = "PASS" if required and all(c["status"] == "PASS" for c in checks) and all(cmap.get(x) == "PASS" for x in required) and cmap.get("required_check_coverage") == "PASS" else "FAIL"
    base = {
        "schema_id": VERIFICATION_SCHEMA,"contract_version": reg.get("contract_version") if reg else None,
        "verifier_id": VERIFIER_ID,"run_id": run_id,"registration_sha256": reg.get("registration_sha256") if reg else None,
        "envelope_sha256": env_sha,"status": status,"comparison_mode": "COLD_RECOMPUTE_EXACT_PLUS_PREREGISTERED_G5_S1_PREDICATE",
        "checks": checks,"required_checks": required,
        "verifier_implementation":{"kind":"G5_S1_COLD_RECOMPUTE_AND_PREREGISTERED_PREDICATE","independent_process_required":True},
        "invocation":{"pid":os.getpid(),"uid":os.getuid() if hasattr(os,"getuid") else None,"gid":os.getgid() if hasattr(os,"getgid") else None,"live_source_sha256":live_source_sha256(),"runtime_sha256":runtime_sha256()},
        "cold_recompute_science_sha256": recomputed.get("science_sha256") if recomputed else None,
        "scientific_disposition": (predicate_detail or {}).get("scientific_disposition"),
        "authority_effect":"NONE",
        "limitations":["G5_S1_ONLY","NO_G5_COMPOSITION_LAW","NO_G5_GRADUATION","NO_G4_DESCRIPTOR_CHANGE","NO_HIDDEN_G4_TOPOLOGY_PROMOTION","NO_S2_EXECUTION"],
    }
    return dict(base, verification_sha256=canonical_sha256(base))


def _cli() -> int:
    from .paths import resolve_root
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--run-id",required=True); ap.add_argument("--report",required=True); ns=ap.parse_args()
    report=verify_g5_s1_run(resolve_root(ns.root),ns.run_id); write_json_atomic(Path(ns.report),report)
    print(json.dumps({"status":report["status"],"verification_sha256":report["verification_sha256"],"scientific_disposition":report.get("scientific_disposition")},sort_keys=True))
    return 0 if report["status"] == "PASS" else 3

if __name__ == "__main__":
    raise SystemExit(_cli())
