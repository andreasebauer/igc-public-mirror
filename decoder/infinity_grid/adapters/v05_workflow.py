from __future__ import annotations

import json
from pathlib import Path

from ..canon import write_json_atomic
from ..controller import register_runner

RUNNER_ID = "adapter.v05_workflow"
OPERATION = "V05_GENERIC_SCIENCE_WORKFLOW"
LOGICAL_OUTPUT = "P6_WORKFLOW_RESULT.json"


def _load_single_registration(root: Path) -> dict:
    files = sorted(p for p in Path(root).rglob("*.json") if p.is_file())
    if len(files) != 1:
        raise RuntimeError(f"P6 workflow dataset must contain exactly one JSON registration, found {len(files)}")
    from ..v05_workflow import validate_workflow_registration
    reg = json.loads(files[0].read_text(encoding="utf-8"))
    return validate_workflow_registration(reg)


@register_runner(RUNNER_ID, v05_operation=OPERATION, call_style="context", dataset_param="workflow_dataset_sha256")
def run_v05_workflow(*, context, plan: dict, stage: dict) -> dict:
    """Run one registered P6 workflow under the existing v0.5 worker boundary.

    This runner owns no scheduling, checkpoint, publication or authority logic.
    Those remain in the existing Controller/runtime.  A new bounded workflow is
    supplied as data; no dispatcher branch is required for each registration.
    """
    ds_sha = stage.get("params", {}).get("workflow_dataset_sha256")
    if not isinstance(ds_sha, str) or len(ds_sha) != 64:
        raise RuntimeError("P6 workflow_dataset_sha256 missing")
    with context.phase("MATERIALIZATION", component="p6.workflow.registration"):
        fixture = context.materialize_dataset(ds_sha, "workflow")
        reg = _load_single_registration(fixture)
    from ..v05_workflow import ControlledWorkflowEngine, get_workflow_semantic_contract
    journal = context.task_journal()
    with context.phase("CENSUS", component="p6.workflow.generic-engine"):
        result = ControlledWorkflowEngine().run(reg, task_journal=journal)
    semantic_contracts = []
    if reg["workflow_kind"] == "STRUCTURAL_S0_S6":
        ids = [row["capability_id"] for row in reg["question"]["stage_rows"]]
    else:
        ids = [reg["question"]["capability_id"]]
    for cid in ids:
        c = get_workflow_semantic_contract(cid)
        if c is not None:
            semantic_contracts.append({"capability_id": cid, "semantic_contract_sha256": c["semantic_contract_sha256"]})
    with context.phase("SERIALIZATION", component="p6.workflow.result"):
        out = context.staging_path(LOGICAL_OUTPUT)
        write_json_atomic(out, result)
    return {
        "outputs": {LOGICAL_OUTPUT: str(out)},
        "stage_result": {
            "status": result.get("status"),
            "workflow_registration_sha256": reg["registration_sha256"],
            "workflow_science_sha256": result.get("science_sha256"),
            "workflow_stop_reason": (result.get("stop_record") or {}).get("reason"),
            "authority_effect": result.get("authority_effect"),
            "graduated": bool(result.get("graduated", False)),
            "controller_owned": True,
            "durable_task_summary": journal.summary(),
            "resolved_semantic_contracts": semantic_contracts,
        },
    }
