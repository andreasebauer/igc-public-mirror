from __future__ import annotations

"""Compile the L0-upward S/R/C/F science catalogue and replay authorities.

This module is engineering infrastructure only. It does not execute or
interpret science. Historical audits are external decisions: the compiler only
validates their exact bindings and exposes automatic-replay versus external-
audit boundaries to the future runner.
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import argparse
import hashlib
import json
import re

from .canon import canonical_sha256, write_json_atomic


SOURCE_SCHEMA = "IG_REPLAY_OBLIGATION_SOURCE_V3"
LEGACY_SOURCE_SCHEMA = "IG_REPLAY_OBLIGATION_SOURCE_V2"
COMPILED_SCHEMA = "IG_REPLAY_OBLIGATION_DAG_V3"
REQUIRED_SERIES = ("S", "R", "C", "F")
MAPPING_STATES = {"MAPPED", "UNMAPPED"}
ASSERTION_MAPPING_STATES = {"MAPPED", "AMBIGUOUS", "EXCLUDED"}
EVIDENCE_MODES = {
    "INDEPENDENT_RECOMPUTATION", "DIRECT_TARGET_REPLAY", "EMBEDDED_EVIDENCE_CHECK",
    "EMBEDDED_GATE", "HISTORICAL_RESULT_ONLY", "THEOREM", "DESIGN_SPECIFICATION",
    "DOCUMENTARY_CLOSEOUT", "EXCLUDED_NONSCIENTIFIC",
}
EXECUTION_CLASSES = (
    "FRESH_RECOMPUTE", "VERIFIED_RESTORED_BLOCK",
    "SOURCE_INTEGRITY_ONLY", "HISTORICAL_RESULT_ONLY",
)
EXECUTION_CLASS_RANK = {name: index for index, name in enumerate(EXECUTION_CLASSES)}
COST_BUDGET_OUTCOMES = {
    "LIKELY_WITHIN_BUDGET", "BUDGET_REQUIRED_BEFORE_RUN",
    "PERFORMANCE_BUDGET_EXCEEDED_EXPECTED", "NOT_AUTHORIZED",
}
EVIDENCE_MODE_EXECUTION_CLASS = {
    "INDEPENDENT_RECOMPUTATION": "FRESH_RECOMPUTE",
    "DIRECT_TARGET_REPLAY": "FRESH_RECOMPUTE",
    "EMBEDDED_EVIDENCE_CHECK": "VERIFIED_RESTORED_BLOCK",
    "EMBEDDED_GATE": "VERIFIED_RESTORED_BLOCK",
    "THEOREM": "SOURCE_INTEGRITY_ONLY",
    "DESIGN_SPECIFICATION": "SOURCE_INTEGRITY_ONLY",
    "DOCUMENTARY_CLOSEOUT": "SOURCE_INTEGRITY_ONLY",
    "HISTORICAL_RESULT_ONLY": "HISTORICAL_RESULT_ONLY",
}
STRENGTHS = {"EXACT", "EXHAUSTIVE", "THEOREM_BACKED", "BOUNDED_EMPIRICAL", "CONDITIONAL"}
AUDIT_DECISIONS = {"CONTINUE", "CERTIFY_AND_ADVANCE"}
EXTERNAL_DECISIONS = {
    "CONTINUE", "REPEAT", "PATCH_REQUIRED", "NEW_SCIENCE_REQUIRED",
    "CERTIFY_AND_ADVANCE",
}
AUTOMATIC_CONTINUE_OUTCOMES = {
    "EXACT_HISTORICAL_REPLAY_AUTHORIZED",
    "CERTIFIED_SEMANTIC_EQUIVALENCE_AUTHORIZED",
    "REPRODUCED_WITH_DECLARED_EVIDENCE_MODE",
}
STOP_OUTCOMES = {
    "RESULT_MISMATCH",
    "PROVENANCE_MISMATCH",
    "SCOPE_OR_NONCLAIM_MISMATCH",
    "MISSING_HISTORICAL_AUDIT_AUTHORIZATION",
    "AMBIGUOUS_HISTORICAL_MAPPING",
    "NEW_OR_EXPANDED_SCIENTIFIC_CLAIM",
    "EQUIVALENCE_NOT_CERTIFIED",
    "NONDETERMINISM",
    "CODE_OR_DATA_FAILURE",
    "RESOURCE_OR_PERFORMANCE_EXHAUSTION",
    "UNAUDITED_G8_FRONTIER",
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9_.-]*$")
AUTH_ID_RE = re.compile(r"^IGA(?:/[A-Z0-9][A-Z0-9_.-]*)+$")
QUALIFICATION_ID_RE = re.compile(r"^IGKQ(?:/[A-Z0-9][A-Z0-9_.-]*)+$")
ASSERTION_ID_RE = re.compile(r"^IGAM(?:/[A-Z0-9][A-Z0-9_.-]*)+$")
CANONICAL_ID_RE = re.compile(
    r"^IG/(?P<layer>[A-Z0-9][A-Z0-9_]*)/(?P<series>S|R|C|F|GATE)/"
    r"(?P<node>[A-Z0-9][A-Z0-9_.-]*)$"
)

SCIENTIFIC_FIELDS = (
    "canonical_id", "layer", "series", "historical_aliases", "mapping_reason",
    "source_hashes", "evidence_hashes", "dependencies", "carrier_type",
    "producer", "consumer", "observer", "equality_mode",
    "set_or_multiset_semantics", "exact_or_compressed", "formation_provenance",
    "permitted_outcomes", "mapping_status", "node_kind", "f_targets",
    "audit_authorization_ids",
)

ASSERTION_MAPPING_FIELDS = (
    "assertion_id", "canonical_obligation_ids", "historical_aliases",
    "source_hashes", "locator", "claim_summary", "classification_reason",
    "scope_and_bounds", "expected_outcome", "evidence_mode", "mapping_status",
    "ambiguity_reason", "exclusion_category",
)

KNOWN_QUALIFICATION_FIELDS = (
    "qualification_id", "category", "status", "affected_assertion_ids",
    "affected_obligation_ids", "evidence_pins", "evidence_status",
    "required_result_language", "replay_effect", "boundary", "nonclaims",
    "record_hash",
)

AUDIT_AUTHORIZATION_FIELDS = (
    "authorization_id", "canonical_obligation_ids", "historical_aliases",
    "historical_source_hashes", "expected_result_identity_or_comparison_contract",
    "carrier", "equality_or_observer", "scope_and_bounds",
    "earned_algebra_statement", "strength", "preserved_falsifications",
    "limitations_and_nonclaims", "accepted_equivalence_certificates",
    "authorized_downstream_nodes", "audit_provenance", "decision",
    "decision_hash",
)


class ReplayObligationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayObligationError(message)


def _validate_hashes(value: Any, field: str, owner_id: str) -> list[dict[str, str]]:
    _require(isinstance(value, list), f"{owner_id}: {field} must be a list")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        _require(isinstance(raw, Mapping), f"{owner_id}: {field}[{index}] must be an object")
        ref = raw.get("ref")
        digest = raw.get("sha256")
        _require(isinstance(ref, str) and bool(ref.strip()), f"{owner_id}: {field}[{index}] missing ref")
        _require(isinstance(digest, str) and HASH_RE.fullmatch(digest) is not None,
                 f"{owner_id}: {field}[{index}] has invalid sha256")
        rows.append({"ref": ref, "sha256": digest})
    return rows


def _validate_string_list(value: Any, field: str, owner_id: str) -> list[str]:
    _require(isinstance(value, list), f"{owner_id}: {field} must be a list")
    _require(all(isinstance(item, str) and item for item in value),
             f"{owner_id}: {field} must contain nonempty strings")
    _require(len(value) == len(set(value)), f"{owner_id}: {field} contains duplicates")
    return list(value)


def _validate_scientific_node(raw: Any, layers: set[str]) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "obligation must be an object")
    missing = [field for field in SCIENTIFIC_FIELDS if field not in raw]
    node_id = str(raw.get("canonical_id", "<missing-id>"))
    _require(not missing, f"{node_id}: missing required fields {missing}")

    match = CANONICAL_ID_RE.fullmatch(node_id)
    _require(match is not None, f"{node_id}: invalid canonical ID")
    layer = raw["layer"]
    series = raw["series"]
    _require(layer in layers, f"{node_id}: unknown layer {layer!r}")
    _require(match.group("layer") == layer, f"{node_id}: layer field disagrees with ID")
    _require(series in REQUIRED_SERIES, f"{node_id}: scientific series must be S, R, C or F")
    _require(match.group("series") == series, f"{node_id}: series field disagrees with ID")
    _require(raw["node_kind"] in {"SCIENTIFIC", "CATALOGUE_GAP"},
             f"{node_id}: invalid scientific node_kind")
    _require(raw["mapping_status"] in MAPPING_STATES, f"{node_id}: invalid mapping_status")
    if raw["node_kind"] == "CATALOGUE_GAP":
        _require(raw["mapping_status"] == "UNMAPPED", f"{node_id}: catalogue gap cannot be mapped")
    if raw["mapping_status"] == "MAPPED":
        _require(raw["node_kind"] == "SCIENTIFIC", f"{node_id}: mapped node must be scientific")
        _require(bool(raw["mapping_reason"].strip()), f"{node_id}: mapped node needs mapping_reason")
        _require(bool(raw["source_hashes"]), f"{node_id}: mapped node needs a pinned source")

    node = deepcopy(dict(raw))
    for field in ("historical_aliases", "dependencies", "permitted_outcomes",
                  "f_targets", "audit_authorization_ids"):
        node[field] = _validate_string_list(raw[field], field, node_id)
    if "known_qualification_ids" in raw:
        node["known_qualification_ids"] = _validate_string_list(
            raw["known_qualification_ids"], "known_qualification_ids", node_id)
    execution_fields = {
        "assertion_execution_class_counts", "execution_classes_present",
        "effective_execution_class", "execution_class_is_mixed",
        "counts_toward_empty_root_science_replay",
    }
    if execution_fields.intersection(raw):
        missing_execution = sorted(execution_fields.difference(raw))
        _require(not missing_execution,
                 f"{node_id}: incomplete node execution classification {missing_execution}")
        node["execution_classes_present"] = _validate_string_list(
            raw["execution_classes_present"], "execution_classes_present", node_id)
        _require(all(value in EXECUTION_CLASSES for value in node["execution_classes_present"]),
                 f"{node_id}: invalid execution class in execution_classes_present")
        _require(raw["effective_execution_class"] in EXECUTION_CLASSES,
                 f"{node_id}: invalid effective_execution_class")
        _require(isinstance(raw["execution_class_is_mixed"], bool),
                 f"{node_id}: execution_class_is_mixed must be boolean")
        _require(isinstance(raw["counts_toward_empty_root_science_replay"], bool),
                 f"{node_id}: empty-root counting flag must be boolean")
        counts = raw["assertion_execution_class_counts"]
        _require(isinstance(counts, Mapping) and bool(counts),
                 f"{node_id}: assertion_execution_class_counts must be nonempty")
        _require(all(key in EXECUTION_CLASSES and isinstance(value, int) and value > 0
                     for key, value in counts.items()),
                 f"{node_id}: invalid assertion execution class counts")
    if "cost_budget" in raw:
        budget = raw["cost_budget"]
        _require(isinstance(budget, Mapping), f"{node_id}: cost_budget must be an object")
        expected_scenarios = {
            "current_evidence_replay", "prospective_full_empty_root_recompute",
        }
        _require(expected_scenarios.issubset(budget),
                 f"{node_id}: incomplete cost_budget scenarios")
        for scenario in sorted(expected_scenarios):
            row = budget[scenario]
            _require(isinstance(row, Mapping),
                     f"{node_id}: cost_budget.{scenario} must be an object")
            for field in ("cost_band", "expected_outcome", "basis"):
                _require(isinstance(row.get(field), str) and bool(row[field].strip()),
                         f"{node_id}: cost_budget.{scenario}.{field} must be nonempty")
            _require(row["expected_outcome"] in COST_BUDGET_OUTCOMES,
                     f"{node_id}: invalid cost outcome in {scenario}")
            for field in ("wall_seconds", "memory_gib"):
                if field in row:
                    _require(isinstance(row[field], (int, float)) and row[field] > 0,
                             f"{node_id}: cost_budget.{scenario}.{field} must be positive")
            if "measurement_evidence" in row:
                node["cost_budget"][scenario]["measurement_evidence"] = _validate_hashes(
                    row["measurement_evidence"], "measurement_evidence", node_id)
    node["source_hashes"] = _validate_hashes(raw["source_hashes"], "source_hashes", node_id)
    node["evidence_hashes"] = _validate_hashes(raw["evidence_hashes"], "evidence_hashes", node_id)
    for field in (
        "mapping_reason", "carrier_type", "producer", "consumer", "observer",
        "equality_mode", "set_or_multiset_semantics", "exact_or_compressed",
        "formation_provenance",
    ):
        _require(isinstance(raw[field], str), f"{node_id}: {field} must be a string")
    if series == "F" and raw["mapping_status"] == "MAPPED":
        _require(bool(node["f_targets"]), f"{node_id}: mapped F node requires f_targets")
    if series != "F":
        _require(not node["f_targets"], f"{node_id}: only F nodes may declare f_targets")
    return node


def _validate_authorization(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "historical audit authorization must be an object")
    missing = [field for field in AUDIT_AUTHORIZATION_FIELDS if field not in raw]
    auth_id = str(raw.get("authorization_id", "<missing-authorization-id>"))
    _require(not missing, f"{auth_id}: missing required fields {missing}")
    _require(AUTH_ID_RE.fullmatch(auth_id) is not None, f"{auth_id}: invalid authorization_id")
    auth = deepcopy(dict(raw))
    for field in (
        "canonical_obligation_ids", "historical_aliases", "preserved_falsifications",
        "limitations_and_nonclaims", "authorized_downstream_nodes",
    ):
        auth[field] = _validate_string_list(raw[field], field, auth_id)
    _require(bool(auth["canonical_obligation_ids"]), f"{auth_id}: no bound obligations")
    auth["historical_source_hashes"] = _validate_hashes(
        raw["historical_source_hashes"], "historical_source_hashes", auth_id)
    auth["accepted_equivalence_certificates"] = _validate_hashes(
        raw["accepted_equivalence_certificates"], "accepted_equivalence_certificates", auth_id)
    auth["audit_provenance"] = _validate_hashes(raw["audit_provenance"], "audit_provenance", auth_id)
    p1c2_fields = {
        "authority_basis", "audit_record_status", "audit_evidence",
        "complete_layer_audit_recorded", "independent_replay_recorded",
    }
    if p1c2_fields.intersection(raw):
        missing_p1c2 = sorted(p1c2_fields.difference(raw))
        _require(not missing_p1c2, f"{auth_id}: incomplete P1C2 audit classification {missing_p1c2}")
        _require(raw["authority_basis"] == "USER_BLANKET_REPLAY_AUTHORIZATION",
                 f"{auth_id}: unsupported P1C2 authority basis")
        _require(isinstance(raw["audit_record_status"], str) and bool(raw["audit_record_status"]),
                 f"{auth_id}: invalid audit_record_status")
        auth["audit_evidence"] = _validate_hashes(raw["audit_evidence"], "audit_evidence", auth_id)
        _require(isinstance(raw["complete_layer_audit_recorded"], bool),
                 f"{auth_id}: complete_layer_audit_recorded must be boolean")
        _require(isinstance(raw["independent_replay_recorded"], bool),
                 f"{auth_id}: independent_replay_recorded must be boolean")
        if raw["audit_record_status"] == "USER_BLANKET_REPLAY_AUTHORIZATION_ONLY":
            _require(not auth["audit_evidence"],
                     f"{auth_id}: blanket-only status cannot carry audit evidence")
        else:
            _require(bool(auth["audit_evidence"]),
                     f"{auth_id}: recorded-audit status requires pinned audit evidence")
    _require(bool(auth["historical_source_hashes"]), f"{auth_id}: historical sources are not pinned")
    _require(bool(auth["audit_provenance"]), f"{auth_id}: audit provenance is not pinned")
    contract = auth["expected_result_identity_or_comparison_contract"]
    _require(isinstance(contract, Mapping) and bool(contract), f"{auth_id}: empty comparison contract")
    for field in ("carrier", "equality_or_observer", "scope_and_bounds", "earned_algebra_statement"):
        _require(isinstance(auth[field], str) and bool(auth[field].strip()),
                 f"{auth_id}: {field} must be a nonempty string")
    _require(auth["strength"] in STRENGTHS, f"{auth_id}: invalid strength")
    _require(auth["decision"] in AUDIT_DECISIONS, f"{auth_id}: invalid replay decision")
    decision_hash = auth["decision_hash"]
    _require(isinstance(decision_hash, str) and HASH_RE.fullmatch(decision_hash) is not None,
             f"{auth_id}: invalid decision_hash")
    payload = {key: value for key, value in auth.items() if key != "decision_hash"}
    _require(canonical_sha256(payload) == decision_hash, f"{auth_id}: decision hash mismatch")
    return auth


def _validate_assertion_mapping(raw: Any, exclusions: set[str]) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "historical assertion mapping must be an object")
    missing = [field for field in ASSERTION_MAPPING_FIELDS if field not in raw]
    assertion_id = str(raw.get("assertion_id", "<missing-assertion-id>"))
    _require(not missing, f"{assertion_id}: missing required fields {missing}")
    _require(ASSERTION_ID_RE.fullmatch(assertion_id) is not None,
             f"{assertion_id}: invalid assertion_id")
    mapping = deepcopy(dict(raw))
    if "known_qualification_ids" in raw:
        mapping["known_qualification_ids"] = _validate_string_list(
            raw["known_qualification_ids"], "known_qualification_ids", assertion_id)
    execution_fields = {
        "execution_class", "counts_toward_empty_root_science_replay",
        "execution_class_reason",
    }
    if execution_fields.intersection(raw):
        missing_execution = sorted(execution_fields.difference(raw))
        _require(not missing_execution,
                 f"{assertion_id}: incomplete assertion execution classification {missing_execution}")
        _require(raw["execution_class"] in EXECUTION_CLASSES,
                 f"{assertion_id}: invalid execution_class")
        _require(raw["execution_class"] == EVIDENCE_MODE_EXECUTION_CLASS[raw["evidence_mode"]],
                 f"{assertion_id}: execution_class disagrees with evidence_mode")
        _require(isinstance(raw["counts_toward_empty_root_science_replay"], bool),
                 f"{assertion_id}: empty-root counting flag must be boolean")
        _require(raw["counts_toward_empty_root_science_replay"]
                 is (raw["execution_class"] == "FRESH_RECOMPUTE"),
                 f"{assertion_id}: empty-root counting flag disagrees with execution_class")
        _require(isinstance(raw["execution_class_reason"], str)
                 and bool(raw["execution_class_reason"].strip()),
                 f"{assertion_id}: execution_class_reason must be nonempty")
    for field in ("canonical_obligation_ids", "historical_aliases"):
        mapping[field] = _validate_string_list(raw[field], field, assertion_id)
    mapping["source_hashes"] = _validate_hashes(raw["source_hashes"], "source_hashes", assertion_id)
    _require(bool(mapping["source_hashes"]), f"{assertion_id}: source hashes are not pinned")
    locator = mapping["locator"]
    _require(isinstance(locator, Mapping) and bool(locator), f"{assertion_id}: locator must be an object")
    _require(isinstance(locator.get("source_ref"), str) and bool(locator["source_ref"].strip()),
             f"{assertion_id}: locator.source_ref must be a nonempty string")
    _require(isinstance(locator.get("selector"), str) and bool(locator["selector"].strip()),
             f"{assertion_id}: locator.selector must be a nonempty string")
    for field in ("claim_summary", "classification_reason", "scope_and_bounds", "expected_outcome",
                  "ambiguity_reason", "exclusion_category"):
        _require(isinstance(mapping[field], str), f"{assertion_id}: {field} must be a string")
    _require(bool(mapping["claim_summary"].strip()), f"{assertion_id}: claim_summary is empty")
    _require(bool(mapping["classification_reason"].strip()),
             f"{assertion_id}: classification_reason is empty")
    _require(bool(mapping["scope_and_bounds"].strip()), f"{assertion_id}: scope_and_bounds is empty")
    _require(bool(mapping["expected_outcome"].strip()), f"{assertion_id}: expected_outcome is empty")
    _require(mapping["evidence_mode"] in EVIDENCE_MODES, f"{assertion_id}: invalid evidence_mode")
    state = mapping["mapping_status"]
    _require(state in ASSERTION_MAPPING_STATES, f"{assertion_id}: invalid mapping_status")
    if state == "MAPPED":
        _require(bool(mapping["canonical_obligation_ids"]), f"{assertion_id}: mapped assertion has no obligation")
        _require(not mapping["ambiguity_reason"], f"{assertion_id}: mapped assertion cannot be ambiguous")
        _require(not mapping["exclusion_category"], f"{assertion_id}: mapped assertion cannot be excluded")
        _require(mapping["evidence_mode"] != "EXCLUDED_NONSCIENTIFIC",
                 f"{assertion_id}: mapped assertion cannot use excluded evidence mode")
    elif state == "AMBIGUOUS":
        _require(bool(mapping["canonical_obligation_ids"]), f"{assertion_id}: ambiguous assertion needs candidates")
        _require(bool(mapping["ambiguity_reason"].strip()), f"{assertion_id}: ambiguity reason is empty")
        _require(not mapping["exclusion_category"], f"{assertion_id}: ambiguous assertion cannot be excluded")
    else:
        _require(not mapping["canonical_obligation_ids"], f"{assertion_id}: excluded assertion cannot bind science")
        _require(not mapping["ambiguity_reason"], f"{assertion_id}: excluded assertion cannot be ambiguous")
        _require(mapping["exclusion_category"] in exclusions,
                 f"{assertion_id}: exclusion category is not declared")
        _require(mapping["evidence_mode"] == "EXCLUDED_NONSCIENTIFIC",
                 f"{assertion_id}: excluded assertion needs EXCLUDED_NONSCIENTIFIC mode")
    return mapping


def _validate_known_qualification(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "known replay qualification must be an object")
    qualification_id = str(raw.get("qualification_id", "<missing-qualification-id>"))
    missing = [field for field in KNOWN_QUALIFICATION_FIELDS if field not in raw]
    _require(not missing, f"{qualification_id}: missing required fields {missing}")
    _require(QUALIFICATION_ID_RE.fullmatch(qualification_id) is not None,
             f"{qualification_id}: invalid qualification_id")
    row = deepcopy(dict(raw))
    for field in ("affected_assertion_ids", "affected_obligation_ids", "nonclaims"):
        row[field] = _validate_string_list(raw[field], field, qualification_id)
    row["evidence_pins"] = _validate_hashes(raw["evidence_pins"], "evidence_pins", qualification_id)
    _require(bool(row["affected_assertion_ids"]), f"{qualification_id}: no affected assertions")
    _require(bool(row["affected_obligation_ids"]), f"{qualification_id}: no affected obligations")
    _require(bool(row["evidence_pins"]), f"{qualification_id}: no pinned evidence")
    _require(raw["status"] == "ACTIVE", f"{qualification_id}: qualification must be ACTIVE")
    for field in ("category", "evidence_status", "required_result_language", "replay_effect", "boundary"):
        _require(isinstance(raw[field], str) and bool(raw[field].strip()),
                 f"{qualification_id}: {field} must be nonempty")
    expected = canonical_sha256({key: value for key, value in row.items() if key != "record_hash"})
    _require(raw["record_hash"] == expected, f"{qualification_id}: record hash mismatch")
    return row


def _gap_node(layer: str, series: str, previous_gate: str | None) -> dict[str, Any]:
    return {
        "canonical_id": f"IG/{layer}/{series}/UNMAPPED_CATALOGUE_GAP",
        "layer": layer,
        "series": series,
        "historical_aliases": [],
        "mapping_reason": "No source-grounded obligation has yet been assigned to this required layer/series lane.",
        "source_hashes": [],
        "evidence_hashes": [],
        "dependencies": [previous_gate] if previous_gate else [],
        "carrier_type": "UNRESOLVED",
        "producer": "UNRESOLVED",
        "consumer": "UNRESOLVED",
        "observer": "UNRESOLVED",
        "equality_mode": "UNRESOLVED",
        "set_or_multiset_semantics": "UNRESOLVED",
        "exact_or_compressed": "UNRESOLVED",
        "formation_provenance": "UNRESOLVED",
        "permitted_outcomes": ["REPLAY_GAP"],
        "mapping_status": "UNMAPPED",
        "node_kind": "CATALOGUE_GAP",
        "f_targets": [],
        "audit_authorization_ids": [],
        "audit_authorization_state": "NOT_APPLICABLE_UNMAPPED",
    }


def _gate_node(layer: str, dependencies: list[str]) -> dict[str, Any]:
    return {
        "canonical_id": f"IG/{layer}/GATE/CERTIFY",
        "layer": layer,
        "series": "GATE",
        "dependencies": sorted(dependencies),
        "node_kind": "WORKFLOW_GATE",
        "mapping_status": "MAPPED",
        "permitted_outcomes": ["CERTIFIED_REPLAY_PASS", "WAITING_FOR_EXTERNAL_AUDIT"],
        "audit_owner": "EXTERNAL_CHAT_WITH_USER",
    }


def _topological_order(nodes: Mapping[str, Mapping[str, Any]], layer_rank: Mapping[str, int]) -> list[str]:
    indegree = {node_id: 0 for node_id in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        for dependency in node["dependencies"]:
            _require(dependency in nodes, f"{node_id}: unknown dependency {dependency}")
            _require(dependency != node_id, f"{node_id}: self dependency")
            indegree[node_id] += 1
            children[dependency].append(node_id)

    def key(node_id: str) -> tuple[int, int, str]:
        node = nodes[node_id]
        series_order = {"S": 0, "R": 1, "C": 2, "F": 3, "GATE": 4}[node["series"]]
        return (layer_rank[node["layer"]], series_order, node_id)

    ready = sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=key)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for child in sorted(children[node_id], key=key):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=key)
    if len(order) != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise ReplayObligationError(f"dependency cycle involving {cyclic}")
    return order


def _compiled_status(unmapped: list[str], unauthorized: list[str], unresolved_assertions: list[str] | None = None) -> str:
    unresolved_assertions = unresolved_assertions or []
    if unresolved_assertions and (unmapped or unauthorized):
        return "BLOCKED_MAPPING_CATALOGUE_OR_AUTHORIZATION"
    if unresolved_assertions:
        return "BLOCKED_ASSERTION_MAPPING"
    if unmapped and unauthorized:
        return "BLOCKED_UNMAPPED_AND_UNAUTHORIZED"
    if unmapped:
        return "BLOCKED_UNMAPPED"
    if unauthorized:
        return "BLOCKED_UNAUTHORIZED"
    return "READY"


def compile_obligations(source: Mapping[str, Any]) -> dict[str, Any]:
    source_schema = source.get("schema_id")
    _require(source_schema in {SOURCE_SCHEMA, LEGACY_SOURCE_SCHEMA}, "bad obligation source schema")
    _require(source.get("canonical_id_syntax") == "IG/<layer>/<S|R|C|F>/<node>",
             "canonical ID syntax is not the frozen v1.3 syntax")
    exclusions = source.get("excluded_from_scientific_catalogue")
    _require(isinstance(exclusions, list) and bool(exclusions),
             "excluded_from_scientific_catalogue must be explicit")
    _validate_string_list(exclusions, "excluded_from_scientific_catalogue", SOURCE_SCHEMA)

    layers_raw = source.get("layers")
    _require(isinstance(layers_raw, list) and layers_raw, "layers must be a nonempty list")
    layers: list[dict[str, str]] = []
    for index, raw in enumerate(layers_raw):
        _require(isinstance(raw, Mapping), f"layer {index} must be an object")
        order = raw.get("order")
        layer = raw.get("layer")
        _require(isinstance(order, str) and TOKEN_RE.fullmatch(order) is not None, f"layer {index}: invalid order")
        _require(isinstance(layer, str) and TOKEN_RE.fullmatch(layer) is not None, f"layer {index}: invalid layer")
        layers.append({"order": order, "layer": layer})
    _require(len({row["order"] for row in layers}) == len(layers), "duplicate layer order")
    _require(len({row["layer"] for row in layers}) == len(layers), "duplicate layer ID")
    _require([row["order"] for row in layers] == sorted(row["order"] for row in layers),
             "layers must be listed in ascending workflow order")

    layer_names = {row["layer"] for row in layers}
    layer_rank = {row["layer"]: index for index, row in enumerate(layers)}
    explicit = source.get("obligations")
    _require(isinstance(explicit, list), "obligations must be a list")
    nodes: dict[str, dict[str, Any]] = {}
    for raw in explicit:
        node = _validate_scientific_node(raw, layer_names)
        assertion_mapping_ids = raw.get("assertion_mapping_ids", [])
        node["assertion_mapping_ids"] = _validate_string_list(
            assertion_mapping_ids, "assertion_mapping_ids", node["canonical_id"])
        if source_schema == SOURCE_SCHEMA and node["mapping_status"] == "MAPPED":
            _require(bool(node["assertion_mapping_ids"]),
                     f"{node['canonical_id']}: mapped V3 node needs assertion mappings")
        node_id = node["canonical_id"]
        _require(node_id not in nodes, f"duplicate canonical ID {node_id}")
        nodes[node_id] = node

    previous_gate: str | None = None
    for layer_row in layers:
        layer = layer_row["layer"]
        if previous_gate:
            for node in nodes.values():
                if node["layer"] == layer and node["series"] in REQUIRED_SERIES:
                    if previous_gate not in node["dependencies"]:
                        node["dependencies"].append(previous_gate)
                        node["dependencies"].sort()
        for series in REQUIRED_SERIES:
            lane = [node for node in nodes.values() if node["layer"] == layer and node["series"] == series]
            if not lane:
                gap = _gap_node(layer, series, previous_gate)
                nodes[gap["canonical_id"]] = gap
        layer_science = [node_id for node_id, node in nodes.items() if node["layer"] == layer]
        gate = _gate_node(layer, layer_science)
        nodes[gate["canonical_id"]] = gate
        previous_gate = gate["canonical_id"]

    assertion_mappings_raw = source.get("historical_assertion_mappings", [])
    _require(isinstance(assertion_mappings_raw, list), "historical_assertion_mappings must be a list")
    if source_schema == SOURCE_SCHEMA:
        _require(bool(assertion_mappings_raw), "V3 source requires historical_assertion_mappings")
    assertion_mappings: dict[str, dict[str, Any]] = {}
    for raw in assertion_mappings_raw:
        mapping = _validate_assertion_mapping(raw, set(exclusions))
        assertion_id = mapping["assertion_id"]
        _require(assertion_id not in assertion_mappings, f"duplicate assertion ID {assertion_id}")
        assertion_mappings[assertion_id] = mapping

    qualifications_raw = source.get("known_replay_qualifications", [])
    _require(isinstance(qualifications_raw, list), "known_replay_qualifications must be a list")
    qualifications: dict[str, dict[str, Any]] = {}
    for raw in qualifications_raw:
        row = _validate_known_qualification(raw)
        qualification_id = row["qualification_id"]
        _require(qualification_id not in qualifications,
                 f"duplicate known qualification ID {qualification_id}")
        qualifications[qualification_id] = row

    for assertion_id, mapping in assertion_mappings.items():
        for node_id in mapping["canonical_obligation_ids"]:
            _require(node_id in nodes, f"{assertion_id}: unknown bound obligation {node_id}")
            _require(nodes[node_id]["node_kind"] == "SCIENTIFIC",
                     f"{assertion_id}: may bind only mapped scientific obligations")
            _require(assertion_id in nodes[node_id]["assertion_mapping_ids"],
                     f"{assertion_id}: obligation binding is not reciprocal for {node_id}")
    for node_id, node in nodes.items():
        for assertion_id in node.get("assertion_mapping_ids", []):
            _require(assertion_id in assertion_mappings,
                     f"{node_id}: unknown assertion mapping {assertion_id}")
            _require(node_id in assertion_mappings[assertion_id]["canonical_obligation_ids"],
                     f"{node_id}: assertion mapping {assertion_id} is not reciprocal")

    if "execution_classification_policy" in source:
        _require(isinstance(source["execution_classification_policy"], Mapping),
                 "execution_classification_policy must be an object")
        for assertion_id, mapping in assertion_mappings.items():
            _require("execution_class" in mapping,
                     f"{assertion_id}: execution classification missing under P1C4 policy")
        for node_id, node in nodes.items():
            if node["node_kind"] != "SCIENTIFIC":
                continue
            classes = [assertion_mappings[assertion_id]["execution_class"]
                       for assertion_id in node.get("assertion_mapping_ids", [])]
            expected_counts = {key: classes.count(key) for key in EXECUTION_CLASSES if key in classes}
            expected_present = [key for key in EXECUTION_CLASSES if key in classes]
            expected_effective = max(expected_present, key=EXECUTION_CLASS_RANK.get)
            _require(node.get("assertion_execution_class_counts") == expected_counts,
                     f"{node_id}: assertion execution class counts disagree with mappings")
            _require(node.get("execution_classes_present") == expected_present,
                     f"{node_id}: execution_classes_present disagree with mappings")
            _require(node.get("effective_execution_class") == expected_effective,
                     f"{node_id}: effective execution class is not conservative")
            _require(node.get("execution_class_is_mixed") is (len(expected_present) > 1),
                     f"{node_id}: mixed execution-class flag disagrees")
            _require(node.get("counts_toward_empty_root_science_replay")
                     is (expected_effective == "FRESH_RECOMPUTE"),
                     f"{node_id}: node empty-root counting flag disagrees")

    if "cost_budget_policy" in source:
        _require(isinstance(source["cost_budget_policy"], Mapping)
                 and bool(source["cost_budget_policy"]),
                 "cost_budget_policy must be a nonempty object")
        _require(isinstance(source.get("cost_budget_programme_summary"), Mapping),
                 "cost_budget_programme_summary must be an object")
        for node_id, node in nodes.items():
            if node["node_kind"] == "SCIENTIFIC":
                _require("cost_budget" in node,
                         f"{node_id}: cost budget missing under P1C5 policy")

    for qualification_id, row in qualifications.items():
        for assertion_id in row["affected_assertion_ids"]:
            _require(assertion_id in assertion_mappings,
                     f"{qualification_id}: unknown affected assertion {assertion_id}")
            _require(qualification_id in assertion_mappings[assertion_id].get("known_qualification_ids", []),
                     f"{qualification_id}: assertion qualification binding is not reciprocal")
        for node_id in row["affected_obligation_ids"]:
            _require(node_id in nodes, f"{qualification_id}: unknown affected obligation {node_id}")
            _require(qualification_id in nodes[node_id].get("known_qualification_ids", []),
                     f"{qualification_id}: obligation qualification binding is not reciprocal")
    for assertion_id, mapping in assertion_mappings.items():
        for qualification_id in mapping.get("known_qualification_ids", []):
            _require(qualification_id in qualifications,
                     f"{assertion_id}: unknown known qualification {qualification_id}")
            _require(assertion_id in qualifications[qualification_id]["affected_assertion_ids"],
                     f"{assertion_id}: known qualification binding is not reciprocal")
    for node_id, node in nodes.items():
        for qualification_id in node.get("known_qualification_ids", []):
            _require(qualification_id in qualifications,
                     f"{node_id}: unknown known qualification {qualification_id}")
            _require(node_id in qualifications[qualification_id]["affected_obligation_ids"],
                     f"{node_id}: known qualification binding is not reciprocal")

    authorizations_raw = source.get("historical_audit_authorizations")
    _require(isinstance(authorizations_raw, list), "historical_audit_authorizations must be a list")
    authorizations: dict[str, dict[str, Any]] = {}
    for raw in authorizations_raw:
        auth = _validate_authorization(raw)
        auth_id = auth["authorization_id"]
        _require(auth_id not in authorizations, f"duplicate authorization ID {auth_id}")
        authorizations[auth_id] = auth

    for node_id, node in nodes.items():
        for dependency in node["dependencies"]:
            _require(dependency in nodes, f"{node_id}: unknown dependency {dependency}")
        if node["series"] == "F":
            for target in node["f_targets"]:
                _require(target in nodes, f"{node_id}: unknown F target {target}")
                _require(nodes[target]["series"] in {"S", "R", "C"},
                         f"{node_id}: F target must be S, R or C")
        for auth_id in node.get("audit_authorization_ids", []):
            _require(auth_id in authorizations, f"{node_id}: unknown audit authorization {auth_id}")
            _require(node_id in authorizations[auth_id]["canonical_obligation_ids"],
                     f"{node_id}: audit authorization {auth_id} is not reciprocal")

    for auth_id, auth in authorizations.items():
        for node_id in auth["canonical_obligation_ids"]:
            _require(node_id in nodes, f"{auth_id}: unknown bound obligation {node_id}")
            _require(nodes[node_id]["node_kind"] == "SCIENTIFIC",
                     f"{auth_id}: may bind only mapped scientific obligations")
            _require(auth_id in nodes[node_id]["audit_authorization_ids"],
                     f"{auth_id}: obligation binding is not reciprocal for {node_id}")
        for downstream in auth["authorized_downstream_nodes"]:
            _require(downstream in nodes, f"{auth_id}: unknown authorized downstream node {downstream}")

    for node_id, node in nodes.items():
        for dependency in node["dependencies"]:
            dep_match = CANONICAL_ID_RE.fullmatch(dependency)
            if dep_match and dep_match.group("layer") in layer_rank:
                _require(layer_rank[dep_match.group("layer")] <= layer_rank[node["layer"]],
                         f"{node_id}: dependency points forward to {dependency}")

    for node in nodes.values():
        if node["node_kind"] == "SCIENTIFIC":
            node["audit_authorization_state"] = (
                "HISTORICALLY_AUTHORIZED" if node["audit_authorization_ids"] else
                "MISSING_HISTORICAL_AUDIT_AUTHORIZATION"
            )

    order = _topological_order(nodes, layer_rank)
    aliases: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        for alias in node.get("historical_aliases", []):
            aliases[alias].append(node["canonical_id"])
    alias_index = {alias: sorted(node_ids) for alias, node_ids in sorted(aliases.items())}
    unmapped = sorted(node_id for node_id, node in nodes.items() if node["mapping_status"] == "UNMAPPED")
    unauthorized = sorted(
        node_id for node_id, node in nodes.items()
        if node["node_kind"] == "SCIENTIFIC" and not node["audit_authorization_ids"]
    )
    unresolved_assertions = sorted(
        assertion_id for assertion_id, mapping in assertion_mappings.items()
        if mapping["mapping_status"] == "AMBIGUOUS"
    )
    unresolved_candidate_nodes = {
        node_id
        for assertion_id in unresolved_assertions
        for node_id in assertion_mappings[assertion_id]["canonical_obligation_ids"]
    }
    authorized_replay_prefix: list[str] = []
    for layer_row in layers:
        layer = layer_row["layer"]
        layer_nodes = [
            node for node in nodes.values()
            if node["layer"] == layer and node["series"] in REQUIRED_SERIES
        ]
        layer_ready = (
            len(layer_nodes) == len(REQUIRED_SERIES)
            and all(node["node_kind"] == "SCIENTIFIC" for node in layer_nodes)
            and all(node["audit_authorization_ids"] for node in layer_nodes)
            and not any(node["canonical_id"] in unresolved_candidate_nodes for node in layer_nodes)
        )
        if not layer_ready:
            break
        authorized_replay_prefix.append(layer)
    authorized_through = authorized_replay_prefix[-1] if authorized_replay_prefix else None
    next_wait_layer = (
        layers[len(authorized_replay_prefix)]["layer"]
        if len(authorized_replay_prefix) < len(layers) else None
    )
    status = _compiled_status(unmapped, unauthorized, unresolved_assertions)
    compiled = {
        "schema_id": COMPILED_SCHEMA,
        "source_schema_id": source_schema,
        "source_catalogue_sha256": canonical_sha256(source),
        "status": status,
        "science_executed": False,
        "canonical_id_syntax": source["canonical_id_syntax"],
        "series_meanings": deepcopy(source.get("series_meanings", {})),
        "historical_axis_warning": deepcopy(source.get("historical_axis_warning", {})),
        "excluded_from_scientific_catalogue": list(exclusions),
        "audit_policy": {
            "owner": "EXTERNAL_CHAT_WITH_USER",
            "waiting_state": "WAITING_FOR_EXTERNAL_AUDIT",
            "automatic_continue_outcomes": sorted(AUTOMATIC_CONTINUE_OUTCOMES),
            "stop_outcomes": sorted(STOP_OUTCOMES),
            "external_decisions": sorted(EXTERNAL_DECISIONS),
            "decoder_may_interpret_earned_algebra": False,
        },
        "layers": layers,
        "nodes": [nodes[node_id] for node_id in order],
        "topological_order": order,
        "historical_alias_index": alias_index,
        "historical_assertion_mappings": [assertion_mappings[key] for key in sorted(assertion_mappings)],
        "result_language_policy": deepcopy(source.get("result_language_policy", {})),
        "known_replay_qualifications": [qualifications[key] for key in sorted(qualifications)],
        "execution_classification_policy": deepcopy(source.get("execution_classification_policy", {})),
        "empty_root_science_coverage": deepcopy(source.get("empty_root_science_coverage", {})),
        "cost_budget_policy": deepcopy(source.get("cost_budget_policy", {})),
        "cost_budget_programme_summary": deepcopy(source.get("cost_budget_programme_summary", {})),
        "assertion_mapping_index": {
            node_id: list(nodes[node_id].get("assertion_mapping_ids", []))
            for node_id in sorted(nodes) if nodes[node_id].get("assertion_mapping_ids")
        },
        "historical_audit_authorizations": [authorizations[key] for key in sorted(authorizations)],
        "counts": {
            "layers": len(layers),
            "nodes": len(nodes),
            "scientific": sum(node["node_kind"] == "SCIENTIFIC" for node in nodes.values()),
            "catalogue_gaps": sum(node["node_kind"] == "CATALOGUE_GAP" for node in nodes.values()),
            "workflow_gates": sum(node["node_kind"] == "WORKFLOW_GATE" for node in nodes.values()),
            "historical_audit_authorizations": len(authorizations),
            "historical_assertion_mappings": len(assertion_mappings),
            "mapped_assertions": sum(row["mapping_status"] == "MAPPED" for row in assertion_mappings.values()),
            "ambiguous_assertions": len(unresolved_assertions),
            "excluded_assertions": sum(row["mapping_status"] == "EXCLUDED" for row in assertion_mappings.values()),
            "unmapped": len(unmapped),
            "unauthorized_scientific": len(unauthorized),
        },
        "unmapped_nodes": unmapped,
        "unauthorized_scientific_nodes": unauthorized,
        "unresolved_assertion_mappings": unresolved_assertions,
        "authorized_replay_prefix": authorized_replay_prefix,
        "authorized_replay_through_layer": authorized_through,
        "next_wait_layer": next_wait_layer,
        "prefix_replay_ready": bool(authorized_replay_prefix),
        "execution_authorized": not unmapped and not unauthorized and not unresolved_assertions,
        "catalogue_complete": not unmapped,
        "replay_authorizations_complete": not unauthorized,
    }
    if source_schema == LEGACY_SOURCE_SCHEMA:
        for key in ("historical_assertion_mappings", "mapped_assertions", "ambiguous_assertions", "excluded_assertions"):
            compiled["counts"].pop(key)
    if "known_replay_qualifications" in source:
        compiled["counts"]["known_replay_qualifications"] = len(qualifications)
    if "execution_classification_policy" in source:
        assertion_class_counts = {
            key: sum(mapping.get("execution_class") == key for mapping in assertion_mappings.values())
            for key in EXECUTION_CLASSES
        }
        node_class_counts = {
            key: sum(node.get("effective_execution_class") == key
                     for node in nodes.values() if node["node_kind"] == "SCIENTIFIC")
            for key in EXECUTION_CLASSES
        }
        compiled["counts"]["assertion_execution_classes"] = assertion_class_counts
        compiled["counts"]["effective_node_execution_classes"] = node_class_counts
    if "cost_budget_policy" in source:
        scientific = [node for node in nodes.values() if node["node_kind"] == "SCIENTIFIC"]
        for scenario in ("current_evidence_replay", "prospective_full_empty_root_recompute"):
            compiled["counts"][f"{scenario}_cost_outcomes"] = {
                outcome: sum(
                    node["cost_budget"][scenario]["expected_outcome"] == outcome
                    for node in scientific
                )
                for outcome in sorted(COST_BUDGET_OUTCOMES)
            }
    compiled["dag_sha256"] = canonical_sha256(compiled)
    return compiled


def verify_compiled_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_id") == COMPILED_SCHEMA, "bad compiled manifest schema")
    expected = manifest.get("dag_sha256")
    payload = {key: value for key, value in manifest.items() if key != "dag_sha256"}
    _require(isinstance(expected, str) and HASH_RE.fullmatch(expected) is not None,
             "compiled manifest has invalid dag_sha256")
    _require(canonical_sha256(payload) == expected, "compiled manifest hash mismatch")
    unmapped = manifest.get("unmapped_nodes")
    unauthorized = manifest.get("unauthorized_scientific_nodes")
    unresolved_assertions = manifest.get("unresolved_assertion_mappings", [])
    _require(isinstance(unmapped, list), "compiled manifest missing unmapped_nodes")
    _require(isinstance(unauthorized, list), "compiled manifest missing unauthorized_scientific_nodes")
    _require(isinstance(unresolved_assertions, list),
             "compiled manifest has invalid unresolved_assertion_mappings")
    ready = not unmapped and not unauthorized and not unresolved_assertions
    _require(manifest.get("execution_authorized") is ready,
             "compiled manifest authorization disagrees with blockers")
    _require(manifest.get("catalogue_complete") is (not bool(unmapped)),
             "compiled manifest completeness disagrees with unmapped nodes")
    _require(manifest.get("replay_authorizations_complete") is (not bool(unauthorized)),
             "compiled manifest audit completeness disagrees with unauthorized nodes")
    _require(manifest.get("status") == _compiled_status(unmapped, unauthorized, unresolved_assertions),
             "compiled manifest status disagrees with blockers")
    nodes = manifest.get("nodes")
    layers = manifest.get("layers")
    _require(isinstance(nodes, list), "compiled manifest missing nodes")
    _require(isinstance(layers, list), "compiled manifest missing layers")
    unresolved_rows = {
        row["assertion_id"]: row
        for row in manifest.get("historical_assertion_mappings", [])
        if row.get("mapping_status") == "AMBIGUOUS"
    }
    unresolved_candidates = {
        node_id for row in unresolved_rows.values() for node_id in row["canonical_obligation_ids"]
    }
    expected_prefix = []
    for layer_row in layers:
        layer = layer_row["layer"]
        layer_nodes = [node for node in nodes if node["layer"] == layer and node["series"] in REQUIRED_SERIES]
        if not (
            len(layer_nodes) == len(REQUIRED_SERIES)
            and all(node["node_kind"] == "SCIENTIFIC" for node in layer_nodes)
            and all(node["audit_authorization_ids"] for node in layer_nodes)
            and not any(node["canonical_id"] in unresolved_candidates for node in layer_nodes)
        ):
            break
        expected_prefix.append(layer)
    _require(manifest.get("authorized_replay_prefix") == expected_prefix,
             "compiled manifest authorized prefix disagrees with node state")
    expected_through = expected_prefix[-1] if expected_prefix else None
    expected_wait = layers[len(expected_prefix)]["layer"] if len(expected_prefix) < len(layers) else None
    _require(manifest.get("authorized_replay_through_layer") == expected_through,
             "compiled manifest authorized-through layer disagrees with prefix")
    _require(manifest.get("next_wait_layer") == expected_wait,
             "compiled manifest next-wait layer disagrees with prefix")
    _require(manifest.get("prefix_replay_ready") is bool(expected_prefix),
             "compiled manifest prefix readiness disagrees with prefix")


def verify_local_source_bindings(source: Mapping[str, Any], project_root: Path) -> None:
    """Verify every local mapping and audit pin against candidate bytes.

    This is an integrity check only. It does not interpret a historical result or
    turn a mapped assertion into replay authorization.
    """
    root = Path(project_root).resolve()
    owners = list(source.get("obligations", [])) + list(source.get("historical_assertion_mappings", []))
    owners += list(source.get("historical_audit_authorizations", []))
    owners += list(source.get("known_replay_qualifications", []))
    if source.get("audit_provenance_ledger"):
        owners.append({"authorization_id": "SOURCE_AUDIT_PROVENANCE_LEDGER",
                       "audit_provenance": [source["audit_provenance_ledger"]]})
    if source.get("known_replay_qualification_ledger"):
        owners.append({"authorization_id": "SOURCE_KNOWN_REPLAY_QUALIFICATION_LEDGER",
                       "audit_provenance": [source["known_replay_qualification_ledger"]]})
    if source.get("execution_class_ledger"):
        owners.append({"authorization_id": "SOURCE_EXECUTION_CLASS_LEDGER",
                       "audit_provenance": [source["execution_class_ledger"]]})
    if source.get("cost_budget_ledger"):
        owners.append({"authorization_id": "SOURCE_COST_BUDGET_LEDGER",
                       "audit_provenance": [source["cost_budget_ledger"]]})
    checked: dict[str, str] = {}
    for owner in owners:
        owner_id = (
            owner.get("canonical_id") or owner.get("assertion_id")
            or owner.get("authorization_id") or "<unknown>"
        )
        pin_fields = ["source_hashes"]
        if owner.get("authorization_id"):
            pin_fields = [
                "historical_source_hashes", "accepted_equivalence_certificates",
                "audit_provenance", "audit_evidence",
            ]
        if owner.get("qualification_id"):
            pin_fields = ["evidence_pins"]
        for field in pin_fields:
            for row in owner.get(field, []):
                ref = row["ref"]
                expected = row["sha256"]
                prior = checked.get(ref)
                if prior is not None:
                    _require(prior == expected, f"{owner_id}: conflicting source hash for {ref}")
                    continue
                path = (root / ref).resolve()
                _require(path.is_relative_to(root), f"{owner_id}: source ref escapes project root: {ref}")
                _require(path.is_file(), f"{owner_id}: pinned source is missing: {ref}")
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                _require(actual == expected, f"{owner_id}: source hash mismatch for {ref}")
                checked[ref] = expected
    _require(bool(checked), "no local source bindings were checked")


def load_and_compile(path: Path) -> dict[str, Any]:
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    return compile_obligations(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile an Infinity Grid L0-upward S/R/C/F replay DAG")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    compiled = load_and_compile(args.source)
    write_json_atomic(args.output, compiled)
    print(json.dumps({
        "status": compiled["status"],
        "dag_sha256": compiled["dag_sha256"],
        "counts": compiled["counts"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if compiled["execution_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
