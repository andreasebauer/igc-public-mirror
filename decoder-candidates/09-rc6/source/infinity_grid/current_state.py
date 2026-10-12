from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .errors import ArtifactCorruptError, ContractError, StateConflictError
from .schema import validate
from .contracts import _resource_json

CURRENT_STATE_SCHEMA_ID = "IG_DECODER_CURRENT_STATE_V2"
CURRENT_STATE_VERSION = "2.0.0"


def current_state_schema(schema_id: str = CURRENT_STATE_SCHEMA_ID) -> dict:
    if schema_id == "IG_DECODER_CURRENT_STATE_V1":
        return _resource_json("contracts/DECODER_CURRENT_STATE_V1.schema.json")
    if schema_id == "IG_DECODER_CURRENT_STATE_V2":
        return _resource_json("contracts/DECODER_CURRENT_STATE_V2.schema.json")
    if schema_id == "IG_DECODER_CURRENT_STATE_V3":
        return _resource_json("contracts/DECODER_CURRENT_STATE_V3.schema.json")
    if schema_id == "IG_DECODER_CURRENT_STATE_V4":
        return _resource_json("contracts/DECODER_CURRENT_STATE_V4.schema.json")
    if schema_id == "IG_DECODER_CURRENT_STATE_V5":
        return _resource_json("contracts/DECODER_CURRENT_STATE_V5.schema.json")
    raise ContractError(f"unsupported current-state schema: {schema_id}")


def attach_state_hash(state: dict) -> dict:
    body = dict(state)
    body.pop("state_sha256", None)
    out = dict(body)
    out["state_sha256"] = canonical_sha256(body)
    return out


def verify_current_state(state: dict) -> dict:
    errors = validate(current_state_schema(state.get("schema_id") if isinstance(state, dict) else ""), state, raise_on_error=False)
    if errors:
        raise ContractError("current-state schema failure: " + "; ".join(errors[:20]))
    body = dict(state)
    got = body.pop("state_sha256")
    expected = canonical_sha256(body)
    if got != expected:
        raise ArtifactCorruptError(f"current-state hash mismatch: {got} != {expected}")
    return {"status": "PASS", "schema_id": state["schema_id"], "state_sha256": got}


def load_current_state(path: str | Path) -> dict:
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactCorruptError(f"cannot parse current state {p}: {type(exc).__name__}: {exc}") from exc
    verify_current_state(obj)
    return obj


def write_current_state(path: str | Path, state: dict) -> dict:
    state = attach_state_hash(state)
    verify_current_state(state)
    write_json_atomic(Path(path), state)
    return state


def _load_optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile_legacy_context(context_dir: str | Path, *, gate1_release: dict, as_of_utc: str) -> tuple[dict, dict]:
    """Collapse legacy split context into one authority file without inventing live status.

    HANDOFF_CURRENT is preferred over WORK_STATUS only where the handoff explicitly
    carries a later snapshot of the same run. If a live runtime locator is unavailable,
    active-run information is labelled stale and is not promoted into current science.
    """
    cdir = Path(context_dir)
    handoff = _load_optional(cdir / "HANDOFF_CURRENT.json")
    work = _load_optional(cdir / "WORK_STATUS.json")
    science = _load_optional(cdir / "SCIENCE_STATUS.json")
    code = _load_optional(cdir / "CODE_STATUS.json")
    authority = _load_optional(cdir / "AUTHORITY_POLICY.json")
    if handoff is None and work is None:
        raise StateConflictError("legacy context contains neither HANDOFF_CURRENT nor WORK_STATUS")

    conflicts = []
    active_snapshot: dict[str, Any] = {}
    if handoff and isinstance(handoff.get("active_runs"), dict):
        active_snapshot = handoff["active_runs"]
    elif work:
        active_snapshot = {r.get("run_id", f"run-{i}"): r for i, r in enumerate(work.get("active_runs", []))}

    if handoff and work:
        hrun = handoff.get("active_runs", {}).get("blind_control_O14_to_O59", {})
        wrun = next((r for r in work.get("active_runs", []) if r.get("run_id") == "blind-o-regime-o14-to-o59-control"), {})
        if hrun and wrun and hrun.get("last_completed_level") != wrun.get("last_completed_level"):
            conflicts.append({
                "field": "blind_control.last_completed_level",
                "older_work_status": wrun.get("last_completed_level"),
                "newer_handoff": hrun.get("last_completed_level"),
                "resolution": "HANDOFF_CURRENT_PREFERRED_AS_LATER_EXPLICIT_HANDOFF_SNAPSHOT",
            })

    hard_rules = []
    if handoff:
        hard_rules.extend(handoff.get("hard_rules", []))
    if authority:
        for rule in authority.get("hard_rules", []):
            if rule not in hard_rules:
                hard_rules.append(rule)

    state = {
        "schema_id": CURRENT_STATE_SCHEMA_ID,
        "state_version": CURRENT_STATE_VERSION,
        "authority_class": "CURRENT_CONTEXT",
        "as_of_utc": as_of_utc,
        "purpose": "Single authoritative machine-readable Decoder state. Legacy split context files are inputs/provenance only after this file exists.",
        "canonical_code": {
            "baseline_decoder_version": "0.26.0",
            "gate1_contract_release": gate1_release,
            "gate2_oracle_release": {"status": "NOT_BOUND_IN_LEGACY_RECONCILIATION"},
            "semantic_delta_from_v0_26": "NONE",
            "next_semantic_target": "0.27.0",
        },
        "gate_status": {
            "gate_0_baseline_freeze": "COMPLETE",
            "gate_1_machine_contracts": "COMPLETE",
            "gate_2_v0_26_oracle": "COMPLETE",
            "mig_001_compatibility_spec": "NEXT",
            "gate_3_refactor": "BLOCKED_UNTIL_MIG_001",
            "v0_27_semantic_integration": "BLOCKED_UNTIL_LATER_GATES",
        },
        "science_status": (handoff or {}).get("science_status") or (science or {}).get("o_frontier", {}),
        "operational_snapshot": {
            "freshness": "HISTORICAL_SNAPSHOT_REQUIRES_LIVE_REFRESH",
            "source": "HANDOFF_CURRENT.json" if handoff else "WORK_STATUS.json",
            "active_runs": active_snapshot,
            "rule": "Do not infer present process liveness or scientific conclusions from this historical snapshot. Refresh from declared runtime source before operational action.",
        },
        "hard_rules": hard_rules,
        "legacy_state_reconciliation": {
            "conflicts": conflicts,
            "superseded_authority_files": [x for x in ["INDEX.json", "BOOTSTRAP_CURRENT.json", "WORK_STATUS.json", "HANDOFF_CURRENT.json"] if (cdir / x).exists()],
            "rule": "CURRENT_STATE.json is authoritative after Gate 1; legacy files remain immutable provenance snapshots.",
        },
    }
    state = attach_state_hash(state)
    verify_current_state(state)
    report = {
        "schema_id": "IG_DECODER_STATE_RECONCILIATION_RESULT_V1",
        "status": "PASS",
        "context_dir": str(cdir),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "current_state_sha256": state["state_sha256"],
    }
    return state, report
