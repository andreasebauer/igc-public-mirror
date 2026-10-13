from __future__ import annotations

"""P5 executors for the four mapped C0_HISTORICAL obligations."""

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256
from .replay_dag_runner import RESULT_SCHEMA
from .replay_reference_data import RECORD_SCHEMA, seal_reference_record


HANDLER_REF = "infinity_grid.replay_c0_historical_executor:execute_c0_historical_node"
C0_NODE_IDS = (
    "IG/C0_HISTORICAL/S/ASSERTION_MAPPED_STRUCTURE",
    "IG/C0_HISTORICAL/R/ASSERTION_MAPPED_RELATIONS",
    "IG/C0_HISTORICAL/C/ASSERTION_MAPPED_INTERFACES",
    "IG/C0_HISTORICAL/F/ASSERTION_MAPPED_FALSIFICATIONS",
)
NODE_EXECUTOR_BINDINGS = [{"node_id": node_id, "handler_ref": HANDLER_REF} for node_id in C0_NODE_IDS]
C0_ANCHORS = {
    C0_NODE_IDS[0]: ("### 27.3 Local Sufficiency Gate", "`C0`: very cheap or unlabelled local structure"),
    C0_NODE_IDS[1]: ("`C1`: one-carrier or one-support local profile", "`C2`: pairwise overlap or shared-support profile"),
    C0_NODE_IDS[2]: ("If `C1` closes the distinction", "If `C2` is necessary, `C1` must be falsified"),
    C0_NODE_IDS[3]: ("L12 again had an exact first-occurrence skeleton", "`C0` failed", "C2 required"),
}


class C0HistoricalExecutorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C0HistoricalExecutorError(message)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(record_id: str, record_type: str, payload: Mapping[str, Any], *,
            provenance_class: str, source_hashes: list[dict[str, str]],
            dependencies: tuple[str, ...] = (), epistemic_status: str = "REPLAYED",
            authority_effect: str = "NONE", nonclaims: tuple[str, ...] = ()) -> dict[str, Any]:
    return seal_reference_record({
        "schema_id": RECORD_SCHEMA, "record_id": record_id, "record_type": record_type,
        "layer": "C0_HISTORICAL", "payload": deepcopy(dict(payload)),
        "provenance": {"classification": provenance_class, "status": "PINNED",
                       "source_hashes": deepcopy(source_hashes),
                       "explanation": "P5 exact replay of a mapped C0_HISTORICAL obligation"},
        "scope": {"layer": "C0_HISTORICAL", "execution_class": "HISTORICAL_EVIDENCE_REPLAY"},
        "nonclaims": list(nonclaims), "dependencies": list(dependencies),
        "epistemic_status": epistemic_status, "science_execution": "NONE",
        "authority_effect": authority_effect,
    })


def _verify(rows: list[Mapping[str, Any]], root: Path) -> None:
    for row in rows:
        path = root / str(row["ref"])
        _require(path.is_file(), f"C0 pinned source unavailable: {row['ref']}")
        _require(_sha_file(path) == row["sha256"], f"C0 pinned source hash mismatch: {row['ref']}")


def execute_c0_historical_node(*, node: Mapping[str, Any], catalogue: Mapping[str, Any],
                               repository_root: Path, attempt: int,
                               inject_stop: bool = False) -> dict[str, Any]:
    node_id = node.get("canonical_id")
    _require(node_id in C0_NODE_IDS, "P5 executor only accepts the four C0_HISTORICAL nodes")
    execution_class = node.get("effective_execution_class")
    _require(execution_class in {"SOURCE_INTEGRITY_ONLY", "HISTORICAL_RESULT_ONLY"},
             "C0 execution class changed")
    _require(node.get("counts_toward_empty_root_science_replay") is False,
             "C0 historical replay cannot be promoted to fresh science")
    mapping_ids = node.get("assertion_mapping_ids")
    _require(isinstance(mapping_ids, list) and len(mapping_ids) == 1, "C0 assertion mapping cardinality")
    mappings = {row["assertion_id"]: row for row in catalogue["historical_assertion_mappings"]}
    mapping = mappings.get(mapping_ids[0])
    _require(mapping is not None and mapping["canonical_obligation_ids"] == [node_id],
             "C0 assertion mapping mismatch")
    _require(mapping["execution_class"] == execution_class, "C0 mapping execution-class mismatch")
    auth_ids = node.get("audit_authorization_ids")
    _require(isinstance(auth_ids, list) and len(auth_ids) == 1, "C0 audit authorization cardinality")
    authorizations = {row["authorization_id"]: row for row in catalogue["historical_audit_authorizations"]}
    authorization = authorizations.get(auth_ids[0])
    _require(authorization is not None and node_id in authorization["canonical_obligation_ids"],
             "C0 audit authorization mismatch")
    _require(authorization["decision"] == "CONTINUE", "C0 historical decision is not CONTINUE")

    root = Path(repository_root)
    source_rows = list(mapping["source_hashes"])
    _verify(source_rows, root)
    _verify(list(authorization["historical_source_hashes"]), root)
    _verify(list(authorization["audit_evidence"]), root)
    _verify(list(authorization["audit_provenance"]), root)
    text = (root / source_rows[0]["ref"]).read_text(encoding="utf-8")
    for anchor in C0_ANCHORS[node_id]:
        _require(anchor in text, f"C0 semantic anchor missing: {anchor}")

    lane = node["series"]
    source = _record("IGRD/C0_HISTORICAL/SOURCE/FORMAL_MONOGRAPH", "CANONICAL_OBJECT", {
        "object_identity": "FORMAL_MONOGRAPH_FINAL_2026-08-27", "carrier_schema": "UTF8_MARKDOWN_PINNED_BYTES",
        "canonical_bytes_sha256": source_rows[0]["sha256"], "formation_provenance": "PINNED_HISTORICAL_SOURCE",
    }, provenance_class="DERIVED_THEOREM", source_hashes=source_rows, epistemic_status="HISTORICAL",
       nonclaims=("SOURCE_BYTES_ARE_NOT_AN_INDEPENDENT_PROOF",))
    audit = _record("IGRD/C0_HISTORICAL/AUDIT/P1C2_V2", "AUDIT_AUTHORIZATION", {
        "authorization_id": authorization["authorization_id"],
        "obligation_ids": list(authorization["canonical_obligation_ids"]), "decision": authorization["decision"],
        "authorized_scope": {"scope_and_bounds": authorization["scope_and_bounds"],
                             "strength": authorization["strength"],
                             "comparison_contract": authorization["expected_result_identity_or_comparison_contract"]},
        "limitations": list(authorization["limitations_and_nonclaims"]),
        # C0 is explicitly classified USER_BLANKET_REPLAY_AUTHORIZATION_ONLY;
        # its decision/provenance record is the evidence, not a nonexistent
        # per-layer audit report.
        "evidence_hashes": (list(authorization["audit_evidence"])
                            or list(authorization["audit_provenance"])),
    }, provenance_class="EXTERNAL_AUDIT_DECISION", source_hashes=list(authorization["audit_provenance"]),
       epistemic_status="EXTERNAL_DECISION", authority_effect="EXTERNAL_DECISION_RECORDED",
       nonclaims=("DECODER_DID_NOT_GRANT_THIS_AUTHORITY",))
    records = [source, audit]
    observation = {"node_id": node_id, "assertion_id": mapping["assertion_id"],
                   "claim_summary": mapping["claim_summary"], "expected_outcome": mapping["expected_outcome"],
                   "evidence_mode": mapping["evidence_mode"], "scope_and_bounds": mapping["scope_and_bounds"],
                   "locator": mapping["locator"], "source_hashes": source_rows,
                   "semantic_anchors": list(C0_ANCHORS[node_id])}
    historical_identity = canonical_sha256(observation)
    replay_observation = deepcopy(observation)
    if inject_stop:
        replay_observation["injected_fault"] = "P5_INTENTIONAL_RESULT_MISMATCH"
    replay_identity = canonical_sha256(replay_observation)
    implementation_sha = _sha_file(Path(__file__))
    recipe = _record(f"IGRD/C0_HISTORICAL/RECIPE/{lane}", "GENERATION_RECIPE", {
        "recipe_id": f"P5-C0-{lane}-HISTORICAL-EVIDENCE", "implementation_ref": HANDLER_REF,
        "implementation_sha256": implementation_sha, "input_record_ids": [source["record_id"]],
        "parameters": {"assertion_id": mapping["assertion_id"], "anchors": list(C0_ANCHORS[node_id]),
                       "execution_class": execution_class},
        "expected_record_types": ["SRCF_EVIDENCE", "COMPARISON"], "determinism_contract": "BYTE_IDENTICAL",
    }, provenance_class="CERTIFIED_CONSTRUCTION",
       source_hashes=[{"ref": "infinity_grid/replay_c0_historical_executor.py", "sha256": implementation_sha}],
       dependencies=(source["record_id"],), nonclaims=("RECIPE_DOES_NOT_REGENERATE_HISTORICAL_SCIENCE",))
    records.append(recipe)
    historical = _record(f"IGRD/C0_HISTORICAL/EVIDENCE/{lane}/HISTORICAL", "SRCF_EVIDENCE", {
        "series": lane, "obligation_id": node_id, "result_identity": historical_identity,
        "equality_contract": {"mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                              "compression": "EXACT", "observer": node["observer"], "certificate_sha256": None},
        "evidence_mode": "HISTORICAL_RESULT_ONLY", "outcome": "REPRODUCED",
    }, provenance_class="DERIVED_THEOREM", source_hashes=source_rows, epistemic_status="HISTORICAL",
       authority_effect="HISTORICAL_AUTHORITY_RECORDED",
       nonclaims=("HISTORICAL_RESULT_IS_NOT_FRESH_RECOMPUTATION",))
    records.append(historical)
    suffix = "/INJECTED" if inject_stop else ""
    replay = _record(f"IGRD/C0_HISTORICAL/EVIDENCE/{lane}/REPLAY{suffix}", "SRCF_EVIDENCE", {
        "series": lane, "obligation_id": node_id, "result_identity": replay_identity,
        "equality_contract": {"mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                              "compression": "EXACT", "observer": node["observer"], "certificate_sha256": None},
        "evidence_mode": execution_class, "outcome": "DISCREPANCY" if inject_stop else "REPRODUCED",
    }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
       nonclaims=("NO_FRESH_C0_SCIENCE", "REPRODUCED_DOES_NOT_MEAN_CONFIRMED_CORRECT"))
    records.append(replay)
    comparison = _record(f"IGRD/C0_HISTORICAL/COMPARISON/{lane}{suffix}", "COMPARISON", {
        "obligation_id": node_id, "historical_record_ids": [historical["record_id"]],
        "replay_record_ids": [replay["record_id"]],
        "equality_contract": {"mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                              "compression": "EXACT", "observer": node["observer"], "certificate_sha256": None},
        "outcome": "MISMATCH" if inject_stop else "REPRODUCED",
        "qualification_ids": list(node.get("known_qualification_ids", [])),
    }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
       dependencies=(historical["record_id"], replay["record_id"]),
       nonclaims=("COMPARISON_DOES_NOT_GRANT_AUDIT_AUTHORITY",))
    records.append(comparison)
    if lane == "F":
        negative = _record("IGRD/C0_HISTORICAL/NEGATIVE/F", "NEGATIVE_RESULT", {
            "obligation_id": node_id, "tested_scope": {"scope_and_bounds": mapping["scope_and_bounds"]},
            "negative_statement": mapping["claim_summary"], "witnesses": list(C0_ANCHORS[node_id]),
            "does_not_establish": ["UNIVERSAL_C0_SUFFICIENCY", "FRESH_RECOMPUTATION", "UNBOUNDED_NO_GO"],
        }, provenance_class="DERIVED_THEOREM", source_hashes=source_rows,
           nonclaims=("SPECIFIC_AUDITED_L12_L13_SCOPE_ONLY",))
        records.append(negative)
    return {"schema_id": RESULT_SCHEMA, "node_id": node_id, "attempt": attempt,
            "comparison_outcome": "RESULT_MISMATCH" if inject_stop else "EXACT_HISTORICAL_REPLAY_AUTHORIZED",
            "result_sha256": comparison["record_sha256"], "evidence_sha256": replay["record_sha256"],
            "audit_authorization_record_sha256": audit["record_sha256"], "execution_class": execution_class,
            "counts_toward_empty_root_science_replay": False, "reference_records": records}


def seal_c0_frontier(*, runner_state: Mapping[str, Any], manifest: Mapping[str, Any],
                     next_node_id: str, waiting_for_audit: bool) -> dict[str, Any]:
    name = "AUDIT_STOP" if waiting_for_audit else "AUTOMATIC_PASS"
    return _record(f"IGRD/C0_HISTORICAL/FRONTIER/{name}", "RESUME_FRONTIER", {
        "root_run_id": runner_state["root_run_id"], "manifest_dag_sha256": manifest["dag_sha256"],
        "runner_state_sha256": runner_state["state_sha256"],
        "completed_node_ids": list(runner_state["completed_node_ids"]),
        "checkpoint_sha256_by_node": dict(runner_state["accepted_checkpoint_sha256_by_node"]),
        "next_node_id": next_node_id,
        "frontier_status": "WAITING_FOR_EXTERNAL_AUDIT" if waiting_for_audit else "READY",
    }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION",
       source_hashes=[{"ref": "compiled_manifest.dag_sha256", "sha256": manifest["dag_sha256"]}],
       nonclaims=("FRONTIER_DOES_NOT_AUTHORIZE_THE_NEXT_LAYER",))
