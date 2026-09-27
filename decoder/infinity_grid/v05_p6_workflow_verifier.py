from __future__ import annotations

import argparse
import json
import os
import tempfile
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

VERIFIER_ID = "v05.p6-workflow.independent"


def _check(checks: list[dict], check_id: str, ok: bool, detail: Any = None) -> None:
    rec = {"check_id": check_id, "status": "PASS" if ok else "FAIL"}
    if detail is not None:
        rec["detail"] = detail
    checks.append(rec)


def _projection(result: dict) -> dict:
    return {
        "workflow_registration_sha256": result.get("registration_sha256"),
        "workflow_science_sha256": result.get("science_sha256"),
        "workflow_status": result.get("status"),
        "stop_reason": (result.get("stop_record") or {}).get("reason"),
        "authority_effect": result.get("authority_effect"),
        "graduated": bool(result.get("graduated", False)),
    }


def _load_single_registration(root: Path) -> dict:
    files = sorted(p for p in Path(root).rglob("*.json") if p.is_file())
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one workflow registration JSON, found {len(files)}")
    return validate_workflow_registration(json.loads(files[0].read_text(encoding="utf-8")))


def verify_p6_workflow_run(paths, run_id: str) -> dict:
    checks: list[dict] = []
    run_dir = Path(paths.runs) / run_id
    required = {name: run_dir / name for name in ("run.json", "plan.json", "run_core.json", "v05_envelope.json")}
    _check(checks, "records_present", all(p.is_file() for p in required.values()))
    if not all(p.is_file() for p in required.values()):
        return _finish(run_id, None, None, checks, None)

    try:
        run = json.loads(required["run.json"].read_text())
        plan = json.loads(required["plan.json"].read_text())
        core = json.loads(required["run_core.json"].read_text())
        env = json.loads(required["v05_envelope.json"].read_text())
        _check(checks, "records_json", True)
    except Exception as exc:
        _check(checks, "records_json", False, str(exc))
        return _finish(run_id, None, None, checks, None)

    env_sha = env.get("envelope_sha256")
    _check(checks, "envelope_hash", env.get("schema_id") == ENVELOPE_SCHEMA and env_sha == canonical_sha256({k:v for k,v in env.items() if k != "envelope_sha256"}))
    _check(checks, "execution_terminal", run.get("lifecycle") == "COMPLETE_VALID" and env.get("execution_lifecycle") == "COMPLETE_VALID")

    reg = None
    try:
        reg_sha = plan.get("v05", {}).get("registration_sha256")
        reg = V05RegistrationStore(paths.store).get(reg_sha)
        _check(checks, "registration_binding", env.get("registration_sha256") == reg_sha and reg.get("runner") == "adapter.v05_workflow" and reg.get("verification_policy", {}).get("verifier_id") == VERIFIER_ID)
    except Exception as exc:
        _check(checks, "registration_binding", False, str(exc))

    if reg is not None:
        _check(checks, "source_identity", reg["source_sha256"] == run.get("code_identity",{}).get("source_sha256") == env.get("source_sha256") == live_source_sha256())
        _check(checks, "environment_identity", reg["environment_sha256"] == run.get("environment_identity",{}).get("runtime_sha256") == env.get("environment_sha256") == runtime_sha256())
        _check(checks, "input_binding", reg["input_datasets"] == plan.get("input_datasets") == run.get("input_artifacts") == env.get("input_artifacts"))
    else:
        for cid in ("source_identity","environment_identity","input_binding"):
            _check(checks, cid, False, "registration unavailable")

    _check(checks, "plan_hash", plan.get("plan_sha256") == canonical_sha256({k:v for k,v in plan.items() if k != "plan_sha256"}) == env.get("plan_sha256"))
    _check(checks, "run_core_hash", core.get("run_core_sha256") == canonical_sha256({k:v for k,v in core.items() if k != "run_core_sha256"}) == run.get("run_core_sha256") == env.get("run_core_sha256"))

    store = ArtifactStore(paths.store); ds_store = DatasetStore(store)
    input_ok = all(ds_store.verify(d["dataset_sha256"]).get("status") == "PASS" for d in run.get("input_artifacts", []))
    _check(checks, "input_dataset_bytes", input_ok)

    result_obj = None
    refs = run.get("result_artifacts", [])
    logical = reg.get("output_contract",{}).get("logical_outputs",[]) if reg else []
    _check(checks, "logical_output_set", len(logical) == 1 and {x.get("logical_name") for x in refs} == set(logical))
    result_ref = next((x for x in refs if logical and x.get("logical_name") == logical[0]), None)
    if result_ref:
        try:
            _check(checks, "result_artifact_bytes", store.verify(result_ref["sha256"], result_ref.get("size_bytes")).get("status") == "PASS")
            result_obj = json.loads(store.blob_path(result_ref["sha256"]).read_text())
            _check(checks, "result_json", isinstance(result_obj, dict))
        except Exception as exc:
            _check(checks, "result_artifact_bytes", False, str(exc)); _check(checks, "result_json", False, str(exc))
    else:
        _check(checks, "result_artifact_bytes", False, "principal result missing"); _check(checks, "result_json", False, "principal result missing")

    checkpoint_ok = False
    try:
        cm = CheckpointManager(run_dir, store)
        cp = cm.current(reg["stage_id"]) if reg else None
        ptr = cm.current_pointer(reg["stage_id"]) if reg else None
        hits = [x for x in env.get("checkpoint_bindings",[]) if reg and x.get("stage_id") == reg["stage_id"]]
        checkpoint_ok = bool(cp and ptr and cp.get("status") == "COMPLETE_VALID" and len(hits) == 1 and hits[0]["checkpoint_sha256"] == ptr["checkpoint_sha256"] and hits[0]["checkpoint_content_sha256"] == cp["checkpoint_content_sha256"])
        _check(checks, "checkpoint_binding", checkpoint_ok)
        wb = (cp or {}).get("stage_result",{}).get("worker_boundary",{})
        _check(checks, "worker_boundary", bool(reg and wb.get("kind") == "POSIX_DROP_PRIVILEGE" and wb.get("uid") == reg["worker_policy"]["uid"] and wb.get("gid") == reg["worker_policy"]["gid"] and wb.get("store_write_access") == "DENIED_BY_POSIX" and wb.get("publication_write_access") == "DENIED_BY_POSIX"))
    except Exception as exc:
        _check(checks, "checkpoint_binding", False, str(exc)); _check(checks, "worker_boundary", False, str(exc))

    recomputed = None
    if reg is not None and len(reg.get("input_datasets",[])) == 1:
        try:
            with tempfile.TemporaryDirectory(prefix="ig-p6-workflow-verify-") as td:
                fixture = ds_store.materialize(reg["input_datasets"][0]["dataset_sha256"], Path(td)/"workflow")
                wf_reg = _load_single_registration(fixture)
                recomputed = ControlledWorkflowEngine().run(wf_reg)
            _check(checks, "cold_recompute_exact", result_obj is not None and canonical_sha256(result_obj) == canonical_sha256(recomputed))
            proj = _projection(recomputed)
            _check(checks, "science_projection_exact", proj == reg["output_contract"]["science_oracle"] and result_obj is not None and _projection(result_obj) == proj)
        except Exception as exc:
            _check(checks, "cold_recompute_exact", False, str(exc)); _check(checks, "science_projection_exact", False, str(exc))
    else:
        _check(checks, "cold_recompute_exact", False, "input registration unavailable"); _check(checks, "science_projection_exact", False, "input registration unavailable")

    req = reg.get("verification_policy",{}).get("required_checks",[]) if reg else []
    seen = {c["check_id"] for c in checks}
    _check(checks, "required_check_coverage", bool(req) and all(x in seen for x in req), {"required":req,"seen":sorted(seen)})
    return _finish(run_id, env_sha, reg, checks, recomputed)


def _finish(run_id: str, env_sha: str | None, reg: dict | None, checks: list[dict], recomputed: dict | None) -> dict:
    required = reg.get("verification_policy",{}).get("required_checks",[]) if reg else []
    cmap = {c["check_id"]:c["status"] for c in checks}
    status = "PASS" if required and all(c["status"] == "PASS" for c in checks) and all(cmap.get(x) == "PASS" for x in required) and cmap.get("required_check_coverage") == "PASS" else "FAIL"
    base = {
        "schema_id": VERIFICATION_SCHEMA,
        "contract_version": reg.get("contract_version") if reg else None,
        "verifier_id": VERIFIER_ID,
        "run_id": run_id,
        "registration_sha256": reg.get("registration_sha256") if reg else None,
        "envelope_sha256": env_sha,
        "status": status,
        "comparison_mode": "COLD_RECOMPUTE_EXACT",
        "checks": checks,
        "required_checks": required,
        "verifier_implementation": {"kind":"P6_GENERIC_WORKFLOW_COLD_RECOMPUTE","independent_process_required":True},
        "invocation": {"pid":os.getpid(),"uid":os.getuid() if hasattr(os,"getuid") else None,"gid":os.getgid() if hasattr(os,"getgid") else None,"live_source_sha256":live_source_sha256(),"runtime_sha256":runtime_sha256()},
        "cold_recompute_projection": _projection(recomputed) if recomputed else None,
        "authority_effect": "NONE",
        "limitations": ["P6_ENGINEERING_GATE_ONLY","WORKFLOW_CAPABILITIES_REUSE_P5_ACCEPTED_SEMANTICS","NO_NEW_G3_OR_G4_SCIENCE","NO_AUTO_GRADUATION","NO_G5"],
    }
    return dict(base, verification_sha256=canonical_sha256(base))


def _cli() -> int:
    from .paths import resolve_root
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--run-id",required=True); ap.add_argument("--report",required=True); ns=ap.parse_args()
    report=verify_p6_workflow_run(resolve_root(ns.root),ns.run_id); write_json_atomic(Path(ns.report),report)
    print(json.dumps({"status":report["status"],"verification_sha256":report["verification_sha256"]},sort_keys=True))
    return 0 if report["status"] == "PASS" else 3

if __name__ == "__main__":
    raise SystemExit(_cli())
