from __future__ import annotations

import json
from pathlib import Path

from .canon import canonical_sha256, write_json_atomic
from .checkpoints import CheckpointManager
from .store import ArtifactStore
from .v05 import V05RegistrationStore
from .v05_packaging import initialize_packaging_state


ENVELOPE_SCHEMA_V12 = "IG_DECODER_V05_EXECUTION_ENVELOPE_V1_2"
ENVELOPE_SCHEMA_V13 = "IG_DECODER_V05_EXECUTION_ENVELOPE_V1_3"
ENVELOPE_SCHEMA_V14 = "IG_DECODER_V05_EXECUTION_ENVELOPE_V1_4"
ENVELOPE_SCHEMA = "IG_DECODER_V05_EXECUTION_ENVELOPE_V1_5"


def seal_execution_envelope(paths, run_id: str, registration_sha256: str) -> dict:
    run_dir = Path(paths.runs) / run_id
    run_path = run_dir / "run.json"; plan_path = run_dir / "plan.json"; core_path = run_dir / "run_core.json"; events_path = run_dir / "events.jsonl"
    if not all(p.is_file() for p in (run_path, plan_path, core_path)):
        raise RuntimeError("cannot seal v0.5 envelope before run/plan/core exist")
    run = json.loads(run_path.read_text(encoding="utf-8")); plan = json.loads(plan_path.read_text(encoding="utf-8")); core = json.loads(core_path.read_text(encoding="utf-8"))
    if run.get("lifecycle") != "COMPLETE_VALID": raise RuntimeError("cannot seal non-complete v0.5 execution")
    reg = V05RegistrationStore(paths.store).get(registration_sha256)
    if plan.get("v05", {}).get("registration_sha256") != registration_sha256: raise RuntimeError("envelope registration/plan mismatch")
    store = ArtifactStore(paths.store); cm = CheckpointManager(run_dir, store); checkpoints=[]
    for st in plan.get("stages", []):
        sid=st["stage_id"]; ptr=cm.current_pointer(sid); cp=cm.current(sid)
        if not ptr or not cp or cp.get("status") != "COMPLETE_VALID": raise RuntimeError(f"cannot seal incomplete checkpoint {sid}")
        checkpoints.append({
            "stage_id":sid,"attempt":cp["attempt"],"checkpoint_sha256":ptr["checkpoint_sha256"],"checkpoint_content_sha256":cp["checkpoint_content_sha256"],
            "stage_spec_sha256":cp["stage_spec_sha256"],
            "output_artifacts":[{"logical_name":a.get("logical_name"),"sha256":a["sha256"],"size_bytes":a.get("size_bytes")} for a in cp.get("output_artifacts",[])],
            "stage_result":cp.get("stage_result",{}),
        })
    result_artifacts=[{"logical_name":a.get("logical_name"),"sha256":a["sha256"],"size_bytes":a.get("size_bytes"),"media_type":a.get("media_type")} for a in run.get("result_artifacts",[])]
    base={
        "schema_id": ENVELOPE_SCHEMA if reg.get("contract_version")=="1.5.0" else ENVELOPE_SCHEMA_V14 if reg.get("contract_version")=="1.4.0" else ENVELOPE_SCHEMA_V13 if reg.get("contract_version")=="1.3.0" else ENVELOPE_SCHEMA_V12,
        "contract_version":reg["contract_version"],"run_id":run_id,"registration_sha256":registration_sha256,
        "plan_sha256":plan["plan_sha256"],"run_core_sha256":core["run_core_sha256"],
        "source_sha256":run["code_identity"]["source_sha256"],"environment_sha256":run["environment_identity"]["runtime_sha256"],
        "execution_lifecycle":run["lifecycle"],"input_artifacts":run.get("input_artifacts",[]),"result_artifacts":result_artifacts,"checkpoint_bindings":checkpoints,
        "telemetry_reference":{"path_role":"RUN_EVENTS_JSONL","identity_scope":"OUTSIDE_ENVELOPE_SCIENCE_IDENTITY","present_at_seal":events_path.is_file()},
        "publication_state":"UNPUBLISHED","authority_effect":"NONE",
    }
    if reg.get("contract_version") in {"1.3.0", "1.4.0", "1.5.0"}:
        packaging=initialize_packaging_state(run_dir)
        telemetry_files=sorted((run_dir/"v05_telemetry").glob("*.json")) if (run_dir/"v05_telemetry").is_dir() else []
        terminal=[]
        for p in telemetry_files:
            obj=json.loads(p.read_text(encoding="utf-8")); terminal.append({"file":p.name,"telemetry_sha256":obj.get("telemetry_sha256"),"terminal_status":obj.get("terminal_status"),"heartbeat_count":obj.get("heartbeat_count")})
        base.update({
            "terminal_telemetry_bindings":terminal,
            "packaging_state_reference":{"path_role":"V05_PACKAGING_STATE","identity_scope":"SEPARATE_MUTABLE_OPERATIONAL_STATE","schema_id":packaging.get("schema_id")},
        })
        if reg.get("contract_version") in {"1.4.0", "1.5.0"}:
            base.update(resource_budget=reg.get("resource_budget"), stop_rules=reg.get("stop_rules"), comparison_contract=reg.get("comparison_contract"))
    env=dict(base,envelope_sha256=canonical_sha256(base)); out=run_dir/"v05_envelope.json"
    if out.exists():
        old=json.loads(out.read_text(encoding="utf-8"))
        if old!=env: raise RuntimeError("existing v0.5 envelope differs; allocate successor execution envelope")
    else: write_json_atomic(out,env)
    return env
