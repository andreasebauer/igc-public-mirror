from __future__ import annotations

"""P4 executor for the four mapped L0 source-integrity obligations.

The compiled catalogue classifies every current L0 obligation as
SOURCE_INTEGRITY_ONLY and excludes it from fresh empty-root science counts.
This executor preserves that boundary: it verifies pinned bytes and semantic
anchors, regenerates the declared observation, compares it with the historical
record, and stores both sides under the P3 contract.  It does not claim a fresh
J3 census or grant audit authority.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256
from .replay_dag_runner import RESULT_SCHEMA
from .replay_reference_data import RECORD_SCHEMA, seal_reference_record


HANDLER_REF = "infinity_grid.replay_l0_executor:execute_l0_source_integrity_node"
L0_NODE_IDS = (
    "IG/L0/S/ASSERTION_MAPPED_STRUCTURE",
    "IG/L0/R/ASSERTION_MAPPED_RELATIONS",
    "IG/L0/C/ASSERTION_MAPPED_INTERFACES",
    "IG/L0/F/ASSERTION_MAPPED_FALSIFICATIONS",
)
NODE_EXECUTOR_BINDINGS = [
    {"node_id": node_id, "handler_ref": HANDLER_REF} for node_id in L0_NODE_IDS
]
L0_ANCHORS = {
    L0_NODE_IDS[0]: ("### Primitive exactness", "T_P = { [B(x)] : x in Mic(P) }"),
    L0_NODE_IDS[1]: ("## 9. The bridge law", "### Bridge Sufficiency"),
    L0_NODE_IDS[2]: ("## 8. Boundary interfaces and anonymous ports", "### No provenance variable"),
    L0_NODE_IDS[3]: ("### L0 - the local compositor", "no authority, geometry, or physical interpretation"),
}


class L0ReplayExecutorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise L0ReplayExecutorError(message)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    record_id: str,
    record_type: str,
    payload: Mapping[str, Any],
    *,
    provenance_class: str,
    source_hashes: list[dict[str, str]],
    dependencies: tuple[str, ...] = (),
    epistemic_status: str = "REPLAYED",
    authority_effect: str = "NONE",
    nonclaims: tuple[str, ...] = (),
) -> dict[str, Any]:
    return seal_reference_record({
        "schema_id": RECORD_SCHEMA,
        "record_id": record_id,
        "record_type": record_type,
        "layer": "L0",
        "payload": deepcopy(dict(payload)),
        "provenance": {
            "classification": provenance_class,
            "status": "PINNED",
            "source_hashes": deepcopy(source_hashes),
            "explanation": "P4 exact replay of the mapped L0 source-integrity obligation",
        },
        "scope": {"layer": "L0", "execution_class": "SOURCE_INTEGRITY_ONLY"},
        "nonclaims": list(nonclaims),
        "dependencies": list(dependencies),
        "epistemic_status": epistemic_status,
        "science_execution": "NONE",
        "authority_effect": authority_effect,
    })


def _verify_hash_rows(rows: list[Mapping[str, Any]], repository_root: Path) -> None:
    for row in rows:
        path = repository_root / str(row["ref"])
        _require(path.is_file(), f"L0 pinned source unavailable: {row['ref']}")
        _require(_sha_file(path) == row["sha256"], f"L0 pinned source hash mismatch: {row['ref']}")


def _shared_records(
    authorization: Mapping[str, Any],
    source_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _record(
        "IGRD/L0/SOURCE/FORMAL_MONOGRAPH",
        "CANONICAL_OBJECT",
        {
            "object_identity": "FORMAL_MONOGRAPH_FINAL_2026-08-27",
            "carrier_schema": "UTF8_MARKDOWN_PINNED_BYTES",
            "canonical_bytes_sha256": source_rows[0]["sha256"],
            "formation_provenance": "PINNED_HISTORICAL_SOURCE",
        },
        provenance_class="DERIVED_THEOREM",
        source_hashes=source_rows,
        epistemic_status="HISTORICAL",
        nonclaims=("SOURCE_BYTES_ARE_NOT_AN_INDEPENDENT_PROOF",),
    )
    audit_sources = list(authorization["audit_provenance"])
    audit = _record(
        "IGRD/L0/AUDIT/P1C2_V2",
        "AUDIT_AUTHORIZATION",
        {
            "authorization_id": authorization["authorization_id"],
            "obligation_ids": list(authorization["canonical_obligation_ids"]),
            "decision": authorization["decision"],
            "authorized_scope": {
                "scope_and_bounds": authorization["scope_and_bounds"],
                "strength": authorization["strength"],
                "comparison_contract": authorization["expected_result_identity_or_comparison_contract"],
            },
            "limitations": list(authorization["limitations_and_nonclaims"]),
            "evidence_hashes": list(authorization["audit_evidence"]),
        },
        provenance_class="EXTERNAL_AUDIT_DECISION",
        source_hashes=audit_sources,
        epistemic_status="EXTERNAL_DECISION",
        authority_effect="EXTERNAL_DECISION_RECORDED",
        nonclaims=("DECODER_DID_NOT_GRANT_THIS_AUTHORITY",),
    )
    return source, audit


def execute_l0_source_integrity_node(
    *,
    node: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    repository_root: Path,
    attempt: int,
    inject_stop: bool = False,
) -> dict[str, Any]:
    """Execute one exact mapped L0 obligation and return a runner result."""
    node_id = node.get("canonical_id")
    _require(node_id in L0_NODE_IDS, "P4 executor only accepts the four L0 pilot nodes")
    _require(node.get("effective_execution_class") == "SOURCE_INTEGRITY_ONLY",
             "L0 execution class changed")
    _require(node.get("counts_toward_empty_root_science_replay") is False,
             "L0 source-integrity node cannot be promoted to fresh science")
    mapping_ids = node.get("assertion_mapping_ids")
    _require(isinstance(mapping_ids, list) and len(mapping_ids) == 1, "L0 assertion mapping cardinality")
    mappings = {row["assertion_id"]: row for row in catalogue["historical_assertion_mappings"]}
    mapping = mappings.get(mapping_ids[0])
    _require(mapping is not None and mapping["canonical_obligation_ids"] == [node_id],
             "L0 assertion mapping mismatch")
    auth_ids = node.get("audit_authorization_ids")
    _require(isinstance(auth_ids, list) and len(auth_ids) == 1, "L0 audit authorization cardinality")
    authorizations = {row["authorization_id"]: row for row in catalogue["historical_audit_authorizations"]}
    authorization = authorizations.get(auth_ids[0])
    _require(authorization is not None and node_id in authorization["canonical_obligation_ids"],
             "L0 audit authorization mismatch")
    _require(authorization["decision"] == "CONTINUE", "L0 historical decision is not CONTINUE")

    repository_root = Path(repository_root)
    source_rows = list(mapping["source_hashes"])
    _verify_hash_rows(source_rows, repository_root)
    _verify_hash_rows(list(authorization["historical_source_hashes"]), repository_root)
    _verify_hash_rows(list(authorization["audit_evidence"]), repository_root)
    _verify_hash_rows(list(authorization["audit_provenance"]), repository_root)
    source_text = (repository_root / source_rows[0]["ref"]).read_text(encoding="utf-8")
    for anchor in L0_ANCHORS[node_id]:
        _require(anchor in source_text, f"L0 semantic anchor missing: {anchor}")

    source, audit = _shared_records(authorization, source_rows)
    records = [source, audit]
    lane = node["series"]
    observation = {
        "node_id": node_id,
        "assertion_id": mapping["assertion_id"],
        "claim_summary": mapping["claim_summary"],
        "expected_outcome": mapping["expected_outcome"],
        "evidence_mode": mapping["evidence_mode"],
        "scope_and_bounds": mapping["scope_and_bounds"],
        "locator": mapping["locator"],
        "source_hashes": source_rows,
        "semantic_anchors": list(L0_ANCHORS[node_id]),
    }
    historical_identity = canonical_sha256(observation)
    replay_observation = deepcopy(observation)
    if inject_stop:
        replay_observation["injected_fault"] = "P4_INTENTIONAL_RESULT_MISMATCH"
    replay_identity = canonical_sha256(replay_observation)

    recipe = _record(
        f"IGRD/L0/RECIPE/{lane}",
        "GENERATION_RECIPE",
        {
            "recipe_id": f"P4-L0-{lane}-SOURCE-INTEGRITY",
            "implementation_ref": HANDLER_REF,
            "implementation_sha256": _sha_file(Path(__file__)),
            "input_record_ids": [source["record_id"]],
            "parameters": {"assertion_id": mapping["assertion_id"], "anchors": list(L0_ANCHORS[node_id])},
            "expected_record_types": ["SRCF_EVIDENCE", "COMPARISON"],
            "determinism_contract": "BYTE_IDENTICAL",
        },
        provenance_class="CERTIFIED_CONSTRUCTION",
        source_hashes=[{"ref": "infinity_grid/replay_l0_executor.py", "sha256": _sha_file(Path(__file__))}],
        dependencies=(source["record_id"],),
        nonclaims=("RECIPE_EXECUTES_SOURCE_INTEGRITY_ONLY",),
    )
    records.append(recipe)
    historical = _record(
        f"IGRD/L0/EVIDENCE/{lane}/HISTORICAL",
        "SRCF_EVIDENCE",
        {
            "series": lane,
            "obligation_id": node_id,
            "result_identity": historical_identity,
            "equality_contract": {
                "mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                "compression": "EXACT", "observer": node["observer"],
                "certificate_sha256": None,
            },
            "evidence_mode": "HISTORICAL_RESULT_ONLY",
            "outcome": "REPRODUCED",
        },
        provenance_class="DERIVED_THEOREM",
        source_hashes=source_rows,
        epistemic_status="HISTORICAL",
        authority_effect="HISTORICAL_AUTHORITY_RECORDED",
        nonclaims=("HISTORICAL_RESULT_IS_NOT_FRESH_RECOMPUTATION",),
    )
    records.append(historical)
    suffix = "/INJECTED" if inject_stop else ""
    replay = _record(
        f"IGRD/L0/EVIDENCE/{lane}/REPLAY{suffix}",
        "SRCF_EVIDENCE",
        {
            "series": lane,
            "obligation_id": node_id,
            "result_identity": replay_identity,
            "equality_contract": {
                "mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                "compression": "EXACT", "observer": node["observer"],
                "certificate_sha256": None,
            },
            "evidence_mode": "SOURCE_INTEGRITY_ONLY",
            "outcome": "DISCREPANCY" if inject_stop else "REPRODUCED",
        },
        provenance_class="FINITE_COMPUTATIONAL_OBSERVATION",
        source_hashes=source_rows,
        nonclaims=("NO_FRESH_J3_CENSUS", "REPRODUCED_DOES_NOT_MEAN_CONFIRMED_CORRECT"),
    )
    records.append(replay)
    comparison = _record(
        f"IGRD/L0/COMPARISON/{lane}{suffix}",
        "COMPARISON",
        {
            "obligation_id": node_id,
            "historical_record_ids": [historical["record_id"]],
            "replay_record_ids": [replay["record_id"]],
            "equality_contract": {
                "mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                "compression": "EXACT", "observer": node["observer"],
                "certificate_sha256": None,
            },
            "outcome": "MISMATCH" if inject_stop else "REPRODUCED",
            "qualification_ids": list(node.get("known_qualification_ids", [])),
        },
        provenance_class="FINITE_COMPUTATIONAL_OBSERVATION",
        source_hashes=source_rows,
        dependencies=(historical["record_id"], replay["record_id"]),
        nonclaims=("COMPARISON_DOES_NOT_GRANT_AUDIT_AUTHORITY",),
    )
    records.append(comparison)
    if lane == "F":
        negative = _record(
            "IGRD/L0/NEGATIVE/F",
            "NEGATIVE_RESULT",
            {
                "obligation_id": node_id,
                "tested_scope": {"scope_and_bounds": mapping["scope_and_bounds"]},
                "negative_statement": mapping["claim_summary"],
                "witnesses": list(L0_ANCHORS[node_id]),
                "does_not_establish": [
                    "ACTUALIZATION_AUTHORITY", "GEOMETRY", "PHYSICAL_INTERPRETATION",
                    "GLOBAL_OR_UNBOUNDED_NO_GO",
                ],
            },
            provenance_class="DERIVED_THEOREM",
            source_hashes=source_rows,
            nonclaims=("BOUNDED_DECLARED_NONCLAIMS_ONLY",),
        )
        records.append(negative)

    return {
        "schema_id": RESULT_SCHEMA,
        "node_id": node_id,
        "attempt": attempt,
        "comparison_outcome": "RESULT_MISMATCH" if inject_stop else "EXACT_HISTORICAL_REPLAY_AUTHORIZED",
        "result_sha256": comparison["record_sha256"],
        "evidence_sha256": replay["record_sha256"],
        "audit_authorization_record_sha256": audit["record_sha256"],
        "execution_class": "SOURCE_INTEGRITY_ONLY",
        "counts_toward_empty_root_science_replay": False,
        "reference_records": records,
    }


def seal_l0_frontier(
    *,
    runner_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    next_node_id: str,
    waiting_for_audit: bool,
) -> dict[str, Any]:
    """Seal the P4 resume frontier without changing scientific authority."""
    name = "AUDIT_STOP" if waiting_for_audit else "AUTOMATIC_PASS"
    frontier = _record(
        f"IGRD/L0/FRONTIER/{name}",
        "RESUME_FRONTIER",
        {
            "root_run_id": runner_state["root_run_id"],
            "manifest_dag_sha256": manifest["dag_sha256"],
            "runner_state_sha256": runner_state["state_sha256"],
            "completed_node_ids": list(runner_state["completed_node_ids"]),
            "checkpoint_sha256_by_node": dict(runner_state["accepted_checkpoint_sha256_by_node"]),
            "next_node_id": next_node_id,
            "frontier_status": "WAITING_FOR_EXTERNAL_AUDIT" if waiting_for_audit else "READY",
        },
        provenance_class="FINITE_COMPUTATIONAL_OBSERVATION",
        source_hashes=[{
            "ref": "compiled_manifest.dag_sha256",
            "sha256": manifest["dag_sha256"],
        }],
        nonclaims=("FRONTIER_DOES_NOT_AUTHORIZE_THE_NEXT_LAYER",),
    )
    return frontier
