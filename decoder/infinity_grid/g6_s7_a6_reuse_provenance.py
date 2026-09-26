from __future__ import annotations

"""Decoder-owned provenance verification for G6:S7 A6 certified reuse.

This validation is intentionally separate from S7D2 science.  It verifies every
available structural/binding claim without reconstructing the A6 census or
observer.  Missing original partition-store bytes remain an explicit gap; a
self-consistent reuse artifact is never relabeled as an independent membership
comparison.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from .canon import canonical_sha256, write_json_atomic
from .structural_encoding import structural_canonical_sha256
from .v05_origin_guard import require_controller_execution_origin

PLAN_SCHEMA = "IG_G6_S7_A6_REUSE_PROVENANCE_VERIFY_PLAN_V1"
REUSE_SCHEMA = "IG_G6_S7_A6_DEPTH1_CERTIFIED_REUSE_INPUT_V1"
RECEIPT_SCHEMA = "IG_G6_S7_A6_REUSE_PROVENANCE_VERIFICATION_RECEIPT_V1"
OBSERVER_ID = "ORDINARY_BRANCH_MULTISET_FUTURE_V1"
BRANCH_SEMANTICS = "MULTISET_OF_SUCCESSOR_BLOCKS"
_CLASS_ID = re.compile(r"^(S1|HIGHER):D1:[0-9]{4}:[0-9a-f]{64}$")


class G6S7A6ReuseProvenanceError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_id", "date", "status", "role", "scientific_effect",
        "a6_completion_file_sha256", "reuse_input_file_sha256",
        "expected_a6_source_sha256", "expected_internal_execution_id",
        "expected_result_sha256", "expected_request_id", "observer",
        "expected_panels", "phase_evidence_policy", "question_sha256",
    }
    if type(plan) is not dict or set(plan) != required or plan.get("schema_id") != PLAN_SCHEMA:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_PLAN_SCHEMA")
    if plan.get("scientific_effect") != "NONE" or plan.get("role") != "VALIDATION_ONLY_NO_RECONSTRUCTION":
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_ROLE")
    if canonical_sha256({k: v for k, v in plan.items() if k != "question_sha256"}) != plan.get("question_sha256"):
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_PLAN_HASH")
    for key in ("a6_completion_file_sha256", "reuse_input_file_sha256", "expected_a6_source_sha256", "expected_result_sha256"):
        value = plan.get(key)
        if type(value) is not str or len(value) != 64:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_PLAN_SHA:" + key)
    obs = plan.get("observer") or {}
    if obs != {"id": OBSERVER_ID, "branch_semantics": BRANCH_SEMANTICS, "ordinary_context_count": 248, "ordinary_operator_count": 31}:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_PLAN_OBSERVER")
    expected = {
        "s1": {"exact_state_count": 4520, "depth1_class_count": 363, "depth1_multi_class_count": 307, "collision_member_count": 4464, "max_class_size": 54},
        "higher": {"selected_exact_state_count": 256, "depth1_class_count": 30, "depth1_multi_class_count": 21, "collision_member_count": 247, "max_class_size": 46},
    }
    if plan.get("expected_panels") != expected:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_PLAN_PANELS")
    if plan.get("phase_evidence_policy") != {"inherited_partition_required": True, "component_batch_count": 31, "partition_store_bytes_required_for_independent_membership": True}:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PROVENANCE_PHASE_POLICY")
    return dict(plan)


def _tree_from_state(state: Mapping[str, Any]) -> DecoratedG4Tree:
    if type(state) is not dict or set(state) != {"n", "edges", "edge_operators", "H_classes"}:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_MEMBER_STATE_FIELDS")
    return DecoratedG4Tree(
        n=int(state["n"]),
        edges=tuple(tuple(int(y) for y in x) for x in state["edges"]),
        H_classes=tuple(str(x) for x in state["H_classes"]),
        edge_operators=tuple(tuple(int(y) for y in x) for x in state["edge_operators"]),
    )


def recompute_member_identity(row: Mapping[str, Any], adapter: G4AcceptedAdapter | None = None) -> str:
    """Recompute the exact accepted structural identity for one stored A6 member."""
    if type(row) is not dict:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_MEMBER_ROW")
    adapter = adapter or G4AcceptedAdapter()
    tree = _tree_from_state(row.get("state"))
    canon = adapter.unrooted_canon(tree)
    digest = structural_canonical_sha256(canon)
    if row.get("identity_sha256") != digest or row.get("identity_canonical_bytes_sha256") != digest:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_MEMBER_IDENTITY_MISMATCH")
    if row.get("state_token") != digest + ":0":
        raise G6S7A6ReuseProvenanceError("A6_REUSE_MEMBER_TOKEN_MISMATCH")
    return digest


def _expected_phase_ids(panel: str) -> list[str]:
    if panel == "s1":
        inherited = "S7_S1_INHERITED_PUBLIC_PARTITION"
        prefix = "S7_S1_BRANCH_MULTISET_D1_COMPONENT_BATCH_"
    elif panel == "higher":
        inherited = "S7_HIGHER_HOLDOUT_INHERITED_PUBLIC_PARTITION"
        prefix = "S7_HIGHER_HOLDOUT_BRANCH_MULTISET_D1_COMPONENT_BATCH_"
    else:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PANEL")
    return [inherited] + [prefix + f"{i:02d}" for i in range(31)]


def _completion_science(a6_completion: Mapping[str, Any]) -> dict[str, Any]:
    if type(a6_completion) is not dict or a6_completion.get("schema_id") != "IG_DECODER_CONTROLLER_REQUEST_COMPLETION_V1" or a6_completion.get("status") != "PASS":
        raise G6S7A6ReuseProvenanceError("A6_REUSE_COMPLETION_SCHEMA")
    result = a6_completion.get("result") or {}
    science = result.get("science_result") or {}
    if type(science) is not dict:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_COMPLETION_SCIENCE")
    return science


def verify_reuse_provenance(*, reuse: Mapping[str, Any], a6_completion: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    plan = _validate_plan(plan)
    if type(reuse) is not dict or reuse.get("schema_id") != REUSE_SCHEMA:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_SCHEMA")
    expected_payload = canonical_sha256({k: v for k, v in reuse.items() if k != "reuse_payload_sha256"})
    if reuse.get("reuse_payload_sha256") != expected_payload:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_PAYLOAD_HASH")

    science = _completion_science(a6_completion)
    src = reuse.get("source_execution") or {}
    bindings = {
        "accepted_source": plan["expected_a6_source_sha256"],
        "internal_execution_id": plan["expected_internal_execution_id"],
        "result_sha256": plan["expected_result_sha256"],
        "request_id": plan["expected_request_id"],
    }
    if a6_completion.get("accepted_source_sha256") != bindings["accepted_source"] or science.get("accepted_decoder_source_sha256") != bindings["accepted_source"] or src.get("accepted_decoder_source_sha256") != bindings["accepted_source"]:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_SOURCE_BINDING")
    if science.get("internal_execution_id") != bindings["internal_execution_id"] or src.get("internal_execution_id") != bindings["internal_execution_id"]:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_EXECUTION_BINDING")
    if science.get("result_sha256") != bindings["result_sha256"] or src.get("result_sha256") != bindings["result_sha256"]:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_RESULT_BINDING")
    if a6_completion.get("request_id") != bindings["request_id"] or src.get("request_id") != bindings["request_id"]:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_REQUEST_BINDING")
    if science.get("classification") != "BUDGET_INSUFFICIENT_FOR_REGISTERED_DEPTH2_REFINEMENT" or src.get("classification") != science.get("classification"):
        raise G6S7A6ReuseProvenanceError("A6_REUSE_CLASSIFICATION_BINDING")
    if science.get("g6_graduated") is not False or science.get("r_series_started") is not False:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_STAGE_SEPARATION")

    expected_obs = plan["observer"]
    actual_obs = reuse.get("observer") or {}
    if actual_obs != expected_obs:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_OBSERVER_BINDING")
    if science.get("observer_id") != OBSERVER_ID or science.get("branch_semantics") != BRANCH_SEMANTICS or int(science.get("ordinary_context_count", -1)) != 248 or int(science.get("ordinary_operator_count", -1)) != 31:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_COMPLETION_OBSERVER_BINDING")

    adapter = G4AcceptedAdapter()
    panel_receipts: dict[str, Any] = {}
    global_seen: dict[str, str] = {}
    for panel in ("s1", "higher"):
        obj = reuse.get(panel) or {}
        exp = plan["expected_panels"][panel]
        classes = list(obj.get("collision_classes") or ())
        if len(classes) != exp["depth1_multi_class_count"]:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_CLASS_COUNT:" + panel)
        seen: set[str] = set()
        computed_classes = []
        sizes = []
        for c in classes:
            cid = str(c.get("depth1_class_id", ""))
            wanted_prefix = "S1:" if panel == "s1" else "HIGHER:"
            if not _CLASS_ID.fullmatch(cid) or not cid.startswith(wanted_prefix):
                raise G6S7A6ReuseProvenanceError("A6_REUSE_CLASS_ID:" + panel)
            members = list(c.get("members") or ())
            if int(c.get("size", -1)) != len(members) or len(members) <= 1:
                raise G6S7A6ReuseProvenanceError("A6_REUSE_CLASS_SHAPE:" + panel)
            tokens = []
            for row in members:
                digest = recompute_member_identity(row, adapter)
                token = digest + ":0"
                if token in seen or token in global_seen:
                    raise G6S7A6ReuseProvenanceError("A6_REUSE_DUPLICATE_MEMBER:" + panel)
                seen.add(token); global_seen[token] = panel; tokens.append(token)
            tokens.sort(); sizes.append(len(tokens))
            computed_classes.append({
                "declared_depth1_class_id": cid,
                "computed_lazy_class_token": "LAZY:" + canonical_sha256(["S7_LAZY_CLASS", tokens]),
                "member_count": len(tokens),
                "representative_state_token": tokens[0],
                "member_tokens_sha256": canonical_sha256(tokens),
            })
        if sum(sizes) != exp["collision_member_count"] or max(sizes, default=0) != exp["max_class_size"]:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_MEMBER_COVERAGE:" + panel)
        if panel == "s1":
            if int(obj.get("original_exact_state_count", -1)) != exp["exact_state_count"]:
                raise G6S7A6ReuseProvenanceError("A6_REUSE_PANEL_COUNT:s1")
            science_part = science.get("s1_depth1_partition") or {}; collision = science.get("s1_depth1_collision_info") or {}
            rep_prefix = "S1D1-"
        else:
            if int(obj.get("selected_exact_state_count", -1)) != exp["selected_exact_state_count"]:
                raise G6S7A6ReuseProvenanceError("A6_REUSE_PANEL_COUNT:higher")
            science_part = science.get("higher_depth1_partition") or {}; collision = science.get("higher_depth1_collision_info") or {}
            rep_prefix = "HOLD1-"
        ps = science_part.get("science") or {}
        if int(obj.get("depth1_class_count", -1)) != exp["depth1_class_count"] or int(obj.get("depth1_multi_class_count", -1)) != exp["depth1_multi_class_count"] or int(obj.get("depth1_collision_member_count", -1)) != exp["collision_member_count"] or int(obj.get("depth1_max_class_size", -1)) != exp["max_class_size"]:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_DECLARED_STATS:" + panel)
        if int(ps.get("class_count", -1)) != exp["depth1_class_count"] or int(ps.get("multi_class_count", -1)) != exp["depth1_multi_class_count"] or int(ps.get("max_class_size", -1)) != exp["max_class_size"] or int(collision.get("collision_member_count", -1)) != exp["collision_member_count"]:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_COMPLETION_STATS:" + panel)

        phase_rows = list(obj.get("depth1_phase_evidence") or ())
        expected_phase_ids = _expected_phase_ids(panel)
        if [str(x.get("phase_id", "")) for x in phase_rows] != expected_phase_ids:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_PHASE_SEQUENCE:" + panel)
        if any(type(x.get("partition_db_sha256")) is not str or not re.fullmatch(r"[0-9a-f]{64}", x["partition_db_sha256"]) for x in phase_rows):
            raise G6S7A6ReuseProvenanceError("A6_REUSE_PHASE_HASH:" + panel)

        computed_classes.sort(key=lambda x: x["representative_state_token"])
        first = computed_classes[0] if computed_classes else None
        anchor = collision.get("first_multi_class")
        if first is None or type(anchor) is not dict:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_FIRST_CLASS_MISSING:" + panel)
        if anchor.get("class_token") != first["computed_lazy_class_token"] or int(anchor.get("size", -1)) != first["member_count"] or anchor.get("representative_task_id") != rep_prefix + first["representative_state_token"]:
            raise G6S7A6ReuseProvenanceError("A6_REUSE_FIRST_CLASS_ANCHOR:" + panel)

        panel_receipts[panel] = {
            "collision_class_count": len(computed_classes),
            "collision_member_count": len(seen),
            "max_class_size": max(sizes, default=0),
            "computed_membership_binding_sha256": canonical_sha256(computed_classes),
            "phase_reference_binding_sha256": canonical_sha256(phase_rows),
            "generation_store_db_sha256": str(obj.get("generation_store_db_sha256", "")),
            "first_multi_class_anchor_verified": True,
        }

    receipt = {
        "schema_id": RECEIPT_SCHEMA,
        "status": "PASS_AVAILABLE_EVIDENCE_WITH_EXPLICIT_GAP",
        "scientific_effect": "NONE",
        "accepted_decoder_source_sha256": None,
        "validation_internal_execution_id": None,
        "a6_source_sha256": bindings["accepted_source"],
        "a6_internal_execution_id": bindings["internal_execution_id"],
        "a6_result_sha256": bindings["result_sha256"],
        "a6_request_id": bindings["request_id"],
        "reuse_payload_sha256": expected_payload,
        "structural_member_identity_verified": True,
        "state_token_binding_verified": True,
        "reuse_payload_integrity_verified": True,
        "a6_source_result_execution_binding_verified": True,
        "observer_binding_verified": True,
        "panel_statistics_verified": True,
        "phase_reference_shape_verified": True,
        "first_multi_class_anchor_verified": True,
        "mapping_self_consistency_verified": True,
        "original_partition_store_bytes_verified": False,
        "independent_member_to_class_verification": False,
        "science_admission": "PAUSE_PENDING_VERIFICATION",
        "gap_reason": "ORIGINAL_A6_PARTITION_STORE_BYTES_OR_INDEPENDENT_MAPPING_EXPORT_NOT_RECOVERED",
        "panels": panel_receipts,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def run_g6_s7_a6_reuse_provenance_verify(*, plan_path: str | Path, artifacts: Mapping[str, str | Path], output_dir: str | Path, accepted_source_sha256: str, internal_execution_id: str) -> dict[str, Any]:
    require_controller_execution_origin("g6-s7-a6-reuse-provenance-verify")
    plan_path = Path(plan_path).resolve(strict=True)
    plan = _validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    paths = {k: Path(v).resolve(strict=True) for k, v in artifacts.items()}
    if _sha_file(paths["a6_completion"]) != plan["a6_completion_file_sha256"]:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_COMPLETION_FILE_BINDING")
    if _sha_file(paths["reuse_input"]) != plan["reuse_input_file_sha256"]:
        raise G6S7A6ReuseProvenanceError("A6_REUSE_INPUT_FILE_BINDING")
    a6_completion = json.loads(paths["a6_completion"].read_text(encoding="utf-8"))
    reuse = json.loads(paths["reuse_input"].read_text(encoding="utf-8"))
    receipt = verify_reuse_provenance(reuse=reuse, a6_completion=a6_completion, plan=plan)
    receipt["accepted_decoder_source_sha256"] = str(accepted_source_sha256)
    receipt["validation_internal_execution_id"] = str(internal_execution_id)
    receipt["receipt_sha256"] = canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "G6_S7_A6_REUSE_PROVENANCE_VERIFICATION_RECEIPT.json", receipt)
    return receipt
