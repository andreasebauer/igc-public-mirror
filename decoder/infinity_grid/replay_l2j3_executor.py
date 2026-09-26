from __future__ import annotations

"""P6 historical-evidence executors for the four mapped L2J3 obligations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256
from .replay_dag_runner import RESULT_SCHEMA
from .replay_reference_data import RECORD_SCHEMA, seal_reference_record


HANDLER_REF = "infinity_grid.replay_l2j3_executor:execute_l2j3_node"
L2J3_NODE_IDS = (
    "IG/L2J3/S/ASSERTION_MAPPED_STRUCTURE",
    "IG/L2J3/R/ASSERTION_MAPPED_RELATIONS",
    "IG/L2J3/C/ASSERTION_MAPPED_INTERFACES",
    "IG/L2J3/F/ASSERTION_MAPPED_FALSIFICATIONS",
)
NODE_EXECUTOR_BINDINGS = [{"node_id": node_id, "handler_ref": HANDLER_REF} for node_id in L2J3_NODE_IDS]
ORACLE_SCHEMA = "IG_V026_SCIENTIFIC_ORACLE_MANIFEST_V1"


class L2J3ExecutorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise L2J3ExecutorError(message)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(record_id: str, record_type: str, payload: Mapping[str, Any], *,
            provenance_class: str, source_hashes: list[dict[str, str]],
            dependencies: tuple[str, ...] = (), epistemic_status: str = "REPLAYED",
            authority_effect: str = "NONE", nonclaims: tuple[str, ...] = ()) -> dict[str, Any]:
    return seal_reference_record({
        "schema_id": RECORD_SCHEMA, "record_id": record_id, "record_type": record_type,
        "layer": "L2J3", "payload": deepcopy(dict(payload)),
        "provenance": {"classification": provenance_class, "status": "PINNED",
                       "source_hashes": deepcopy(source_hashes),
                       "explanation": "P6 exact replay of mapped L2J3 historical evidence"},
        "scope": {"layer": "L2J3", "execution_class": "HISTORICAL_EVIDENCE_REPLAY"},
        "nonclaims": list(nonclaims), "dependencies": list(dependencies),
        "epistemic_status": epistemic_status, "science_execution": "NONE",
        "authority_effect": authority_effect,
    })


def _verify(rows: list[Mapping[str, Any]], root: Path) -> None:
    for row in rows:
        path = root / str(row["ref"])
        _require(path.is_file(), f"L2J3 pinned source unavailable: {row['ref']}")
        _require(_sha_file(path) == row["sha256"], f"L2J3 pinned source hash mismatch: {row['ref']}")


def _oracle_assertion(mapping: Mapping[str, Any], root: Path) -> dict[str, Any]:
    locator = mapping.get("locator")
    _require(isinstance(locator, Mapping), "L2J3 locator missing")
    selector = locator.get("selector")
    _require(isinstance(selector, str) and selector.startswith("/assertions/"), "L2J3 selector form")
    try:
        index = int(selector.rsplit("/", 1)[1])
    except ValueError as exc:
        raise L2J3ExecutorError("L2J3 selector index") from exc
    source_ref = locator.get("source_ref")
    _require(isinstance(source_ref, str), "L2J3 source reference missing")
    oracle = json.loads((root / source_ref).read_text(encoding="utf-8"))
    _require(oracle.get("schema_id") == ORACLE_SCHEMA, "L2J3 oracle schema mismatch")
    assertions = oracle.get("assertions")
    _require(isinstance(assertions, list) and 0 <= index < len(assertions), "L2J3 oracle selector range")
    assertion = assertions[index]
    _require(isinstance(assertion, dict), "L2J3 oracle assertion form")
    expected = json.dumps(assertion.get("equals"), ensure_ascii=False, separators=(",", ":"))
    _require(expected == mapping.get("expected_outcome"), "L2J3 oracle expected outcome mismatch")
    summary = f"{assertion.get('file')} {assertion.get('pointer')} equals {expected}"
    _require(summary == mapping.get("claim_summary"), "L2J3 oracle claim summary mismatch")
    return deepcopy(assertion)


def execute_l2j3_node(*, node: Mapping[str, Any], catalogue: Mapping[str, Any],
                      repository_root: Path, attempt: int,
                      inject_stop: bool = False) -> dict[str, Any]:
    node_id = node.get("canonical_id")
    _require(node_id in L2J3_NODE_IDS, "P6 executor only accepts the four L2J3 nodes")
    execution_class = node.get("effective_execution_class")
    _require(execution_class == "HISTORICAL_RESULT_ONLY", "L2J3 execution class changed")
    _require(node.get("counts_toward_empty_root_science_replay") is False,
             "L2J3 historical replay cannot be promoted to fresh science")
    mapping_ids = node.get("assertion_mapping_ids")
    _require(isinstance(mapping_ids, list) and mapping_ids, "L2J3 assertion mappings missing")
    all_mappings = {row["assertion_id"]: row for row in catalogue["historical_assertion_mappings"]}
    mappings = [all_mappings.get(mapping_id) for mapping_id in mapping_ids]
    _require(all(row is not None for row in mappings), "L2J3 assertion mapping unavailable")
    for mapping in mappings:
        _require(mapping["canonical_obligation_ids"] == [node_id], "L2J3 assertion mapping mismatch")
        _require(mapping["execution_class"] == execution_class, "L2J3 mapping execution-class mismatch")
    auth_ids = node.get("audit_authorization_ids")
    _require(isinstance(auth_ids, list) and len(auth_ids) == 1, "L2J3 authorization cardinality")
    authorizations = {row["authorization_id"]: row for row in catalogue["historical_audit_authorizations"]}
    authorization = authorizations.get(auth_ids[0])
    _require(authorization is not None and node_id in authorization["canonical_obligation_ids"],
             "L2J3 audit authorization mismatch")
    _require(authorization["decision"] == "CONTINUE", "L2J3 historical decision is not CONTINUE")

    root = Path(repository_root)
    source_rows: list[dict[str, str]] = []
    for mapping in mappings:
        _verify(list(mapping["source_hashes"]), root)
        for row in mapping["source_hashes"]:
            if row not in source_rows:
                source_rows.append(deepcopy(row))
    _verify(list(authorization["historical_source_hashes"]), root)
    _verify(list(authorization["audit_evidence"]), root)
    _verify(list(authorization["audit_provenance"]), root)
    observations = [{"assertion_id": mapping["assertion_id"],
                     "oracle_assertion": _oracle_assertion(mapping, root),
                     "locator": mapping["locator"], "scope_and_bounds": mapping["scope_and_bounds"],
                     "expected_outcome": mapping["expected_outcome"]} for mapping in mappings]

    source = _record("IGRD/L2J3/SOURCE/V026_SCIENTIFIC_ORACLE", "CANONICAL_OBJECT", {
        "object_identity": "V026_SCIENTIFIC_ORACLE_MANIFEST", "carrier_schema": ORACLE_SCHEMA,
        "canonical_bytes_sha256": source_rows[0]["sha256"],
        "formation_provenance": "PINNED_HISTORICAL_SOURCE",
    }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
       epistemic_status="HISTORICAL", nonclaims=("ORACLE_BYTES_ARE_NOT_FRESH_RECOMPUTATION",))
    audit = _record("IGRD/L2J3/AUDIT/P1C2_V2", "AUDIT_AUTHORIZATION", {
        "authorization_id": authorization["authorization_id"], "decision": authorization["decision"],
        "obligation_ids": list(authorization["canonical_obligation_ids"]),
        "authorized_scope": {"scope_and_bounds": authorization["scope_and_bounds"],
                             "strength": authorization["strength"],
                             "comparison_contract": authorization["expected_result_identity_or_comparison_contract"]},
        "limitations": list(authorization["limitations_and_nonclaims"]),
        "evidence_hashes": (list(authorization["audit_evidence"])
                            or list(authorization["audit_provenance"])),
    }, provenance_class="EXTERNAL_AUDIT_DECISION",
       source_hashes=list(authorization["audit_provenance"]), epistemic_status="EXTERNAL_DECISION",
       authority_effect="EXTERNAL_DECISION_RECORDED",
       nonclaims=("DECODER_DID_NOT_GRANT_THIS_AUTHORITY",))
    lane = node["series"]
    implementation_sha = _sha_file(Path(__file__))
    recipe = _record(f"IGRD/L2J3/RECIPE/{lane}", "GENERATION_RECIPE", {
        "recipe_id": f"P6-L2J3-{lane}-HISTORICAL-EVIDENCE", "implementation_ref": HANDLER_REF,
        "implementation_sha256": implementation_sha, "input_record_ids": [source["record_id"]],
        "parameters": {"assertion_ids": list(mapping_ids), "execution_class": execution_class},
        "expected_record_types": ["SRCF_EVIDENCE", "COMPARISON"],
        "determinism_contract": "BYTE_IDENTICAL",
    }, provenance_class="CERTIFIED_CONSTRUCTION",
       source_hashes=[{"ref": "infinity_grid/replay_l2j3_executor.py", "sha256": implementation_sha}],
       dependencies=(source["record_id"],), nonclaims=("RECIPE_DOES_NOT_REGENERATE_HISTORICAL_SCIENCE",))
    records = [source, audit, recipe]
    historical_ids: list[str] = []
    replay_ids: list[str] = []
    replay_hashes: list[str] = []
    for position, observation in enumerate(observations):
        assertion_id = observation["assertion_id"].rsplit("/", 1)[-1]
        historical_identity = canonical_sha256(observation)
        replay_observation = deepcopy(observation)
        if inject_stop and position == 0:
            replay_observation["injected_fault"] = "P6_INTENTIONAL_RESULT_MISMATCH"
        replay_identity = canonical_sha256(replay_observation)
        historical = _record(f"IGRD/L2J3/EVIDENCE/{lane}/{assertion_id}/HISTORICAL", "SRCF_EVIDENCE", {
            "series": lane, "obligation_id": node_id,
            "result_identity": historical_identity, "outcome": "REPRODUCED",
            "equality_contract": {"mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                                  "compression": "EXACT", "observer": node["observer"],
                                  "certificate_sha256": None},
            "evidence_mode": "HISTORICAL_RESULT_ONLY",
        }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
           epistemic_status="HISTORICAL", authority_effect="HISTORICAL_AUTHORITY_RECORDED",
           nonclaims=("HISTORICAL_RESULT_IS_NOT_FRESH_RECOMPUTATION",))
        replay = _record(f"IGRD/L2J3/EVIDENCE/{lane}/{assertion_id}/REPLAY"
                         + ("/INJECTED" if inject_stop and position == 0 else ""), "SRCF_EVIDENCE", {
            "series": lane, "obligation_id": node_id,
            "result_identity": replay_identity,
            "outcome": "DISCREPANCY" if inject_stop and position == 0 else "REPRODUCED",
            "equality_contract": {"mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                                  "compression": "EXACT", "observer": node["observer"],
                                  "certificate_sha256": None},
            "evidence_mode": "HISTORICAL_RESULT_ONLY",
        }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
           nonclaims=("NO_FRESH_L2J3_SCIENCE", "REPRODUCED_DOES_NOT_MEAN_CONFIRMED_CORRECT"))
        records.extend([historical, replay])
        historical_ids.append(historical["record_id"])
        replay_ids.append(replay["record_id"])
        replay_hashes.append(replay["record_sha256"])
    comparison = _record(f"IGRD/L2J3/COMPARISON/{lane}" + ("/INJECTED" if inject_stop else ""),
                         "COMPARISON", {
        "obligation_id": node_id,
        "historical_record_ids": historical_ids, "replay_record_ids": replay_ids,
        "equality_contract": {"mode": "CANONICAL_JSON", "cardinality_semantics": "ORDERED",
                              "compression": "EXACT", "observer": node["observer"],
                              "certificate_sha256": None},
        "outcome": "MISMATCH" if inject_stop else "REPRODUCED",
        "qualification_ids": list(node.get("known_qualification_ids", [])),
    }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
       dependencies=tuple(historical_ids + replay_ids),
       nonclaims=("COMPARISON_DOES_NOT_GRANT_AUDIT_AUTHORITY",))
    records.append(comparison)
    if lane == "F":
        records.append(_record("IGRD/L2J3/NEGATIVE/F", "NEGATIVE_RESULT", {
            "obligation_id": node_id,
            "tested_scope": {"assertion_ids": list(mapping_ids), "bounded": True},
            "negative_statement": "Pinned bounded challenge outcomes were reproduced",
            "witnesses": [observation["oracle_assertion"] for observation in observations],
            "does_not_establish": ["UNBOUNDED_NO_GO", "FRESH_RECOMPUTATION", "GLOBAL_COMPLETENESS"],
        }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION", source_hashes=source_rows,
           nonclaims=("PINNED_HISTORICAL_SCOPE_ONLY",)))
    return {"schema_id": RESULT_SCHEMA, "node_id": node_id, "attempt": attempt,
            "comparison_outcome": "RESULT_MISMATCH" if inject_stop else "EXACT_HISTORICAL_REPLAY_AUTHORIZED",
            "result_sha256": comparison["record_sha256"],
            "evidence_sha256": canonical_sha256(replay_hashes),
            "audit_authorization_record_sha256": audit["record_sha256"],
            "execution_class": execution_class, "assertion_count": len(mapping_ids),
            "counts_toward_empty_root_science_replay": False, "reference_records": records}


def seal_l2j3_frontier(*, runner_state: Mapping[str, Any], manifest: Mapping[str, Any],
                       next_node_id: str, waiting_for_audit: bool) -> dict[str, Any]:
    name = "AUDIT_STOP" if waiting_for_audit else "AUTOMATIC_PASS"
    return _record(f"IGRD/L2J3/FRONTIER/{name}", "RESUME_FRONTIER", {
        "root_run_id": runner_state["root_run_id"], "manifest_dag_sha256": manifest["dag_sha256"],
        "runner_state_sha256": runner_state["state_sha256"],
        "completed_node_ids": list(runner_state["completed_node_ids"]),
        "checkpoint_sha256_by_node": dict(runner_state["accepted_checkpoint_sha256_by_node"]),
        "next_node_id": next_node_id,
        "frontier_status": "WAITING_FOR_EXTERNAL_AUDIT" if waiting_for_audit else "READY",
    }, provenance_class="FINITE_COMPUTATIONAL_OBSERVATION",
       source_hashes=[{"ref": "compiled_manifest.dag_sha256", "sha256": manifest["dag_sha256"]}],
       nonclaims=("FRONTIER_DOES_NOT_AUTHORIZE_THE_NEXT_LAYER",))
