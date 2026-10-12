from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .hashing import sha256_file
from .protocols import ProtocolRegistry
from .records import utc_now


CHAIN_SCHEMA = "IG_SCOUT_CONVEYOR_CHAIN_PLAN_V0_1"
PACKET_SCHEMA = "IG_SCOUT_CONVEYOR_LEVEL_PACKET_V0_1"
CHAIN_USAGE_SCHEMA = "IG_SCOUT_CONVEYOR_CHAIN_USAGE_V0_19_5"
PARENT_CERT_SCHEMA = "IG_SCOUT_CONVEYOR_PARENT_CERTIFICATE_V0_1"
REGISTERED_SUMMARIES = {
    "PERSISTS (bounded panel only)",
    "PERSISTS + EXPANDS (bounded panel only)",
    "PERSISTS + REORGANIZES (bounded panel only)",
    "PERSISTS + EXPANDS + REORGANIZES (bounded panel only)",
}


def _canon_hash_excluding(obj: dict[str, Any], field: str) -> str:
    return canonical_sha256({k: v for k, v in obj.items() if k != field})


def spec_hash(resource_dir: Path, name: str) -> str:
    obj = json.loads((Path(resource_dir) / name).read_text(encoding="utf-8"))
    observed = _canon_hash_excluding(obj, "spec_sha256")
    declared = obj.get("spec_sha256")
    if declared is not None and declared != observed:
        raise RuntimeError(f"frozen conveyor spec hash mismatch: {name}")
    return observed


def verify_bootstrap_parent(*, parent_pin_path: Path, expected_pin_file_sha256: str, closeout_bundle_path: Path | None = None, expected_closeout_bundle_sha256: str | None = None) -> dict[str, Any]:
    """Verify the historical L16 bootstrap handoff.

    This remains for exact backwards compatibility.  Later chain starts use an
    ELIGIBLE_AUTO_PARENT certificate and a verified minimal parent capsule.
    """
    parent_pin_path = Path(parent_pin_path)
    if sha256_file(parent_pin_path) != expected_pin_file_sha256:
        raise RuntimeError("bootstrap parent pin file SHA mismatch")
    pin = json.loads(parent_pin_path.read_text(encoding="utf-8"))
    if pin.get("status") != "ELIGIBLE_FOR_SEPARATELY_FROZEN_L17_PARENT":
        raise RuntimeError("bootstrap parent does not have the Phase-0 accepted L16 status")
    if pin.get("authority") != "RECONNAISSANCE_ONLY" or pin.get("evidence_label") != "SCOUT_OBSERVED" or pin.get("population_complete") is not False:
        raise RuntimeError("bootstrap parent evidence boundary mismatch")
    if closeout_bundle_path is not None:
        observed = sha256_file(closeout_bundle_path)
        if expected_closeout_bundle_sha256 and observed != expected_closeout_bundle_sha256:
            raise RuntimeError("bootstrap closeout bundle SHA mismatch")
    return pin


def validate_eligible_parent_certificate(parent: dict[str, Any]) -> bool:
    if parent.get("schema_id") != PARENT_CERT_SCHEMA:
        raise RuntimeError("eligible parent certificate schema mismatch")
    if parent.get("status") != "ELIGIBLE_AUTO_PARENT":
        raise RuntimeError("chain start requires ELIGIBLE_AUTO_PARENT")
    if _canon_hash_excluding(parent, "certificate_sha256") != parent.get("certificate_sha256"):
        raise RuntimeError("eligible parent certificate SHA mismatch")
    if parent.get("authority") != "RECONNAISSANCE_ONLY" or parent.get("evidence_label") != "SCOUT_OBSERVED":
        raise RuntimeError("eligible parent authority/evidence mismatch")
    if parent.get("population_complete") is not False or int(parent.get("claim_record_count", -1)) != 0:
        raise RuntimeError("eligible parent evidence firewall mismatch")
    root = parent.get("root_material", {})
    for key in ("file_sha256", "canonical_science_sha256", "selected_object_count"):
        if key not in root:
            raise RuntimeError(f"eligible parent missing root material: {key}")
    if parent.get("trigger_evaluation", {}).get("active"):
        raise RuntimeError("eligible parent has an active trigger")
    reviewed = parent.get("reviewed_markers_cumulative", [])
    if not isinstance(reviewed, list) or any(not isinstance(x, str) or not x for x in reviewed):
        raise RuntimeError("eligible parent reviewed-marker registry malformed")
    if reviewed != sorted(set(reviewed)):
        raise RuntimeError("eligible parent reviewed-marker registry is not canonical")
    return True


def load_verified_parent_capsule(path: Path) -> dict[str, Any]:
    """Load an exact later-level chain-start capsule and verify all bindings."""
    import zipfile
    path = Path(path)
    check = verify_minimal_parent_capsule(path)
    if check.get("status") != "PASS":
        raise RuntimeError(f"parent capsule verification failed: {check}")
    with zipfile.ZipFile(path, "r") as z:
        parent_raw = z.read("parent_certificate.json")
        selected_raw = z.read("SELECTED_RELATION.json")
        baseline_raw = z.read("PARENT_NORMALIZED_BASELINE.json")
    parent = json.loads(parent_raw)
    validate_eligible_parent_certificate(parent)
    root = parent["root_material"]
    if hashlib.sha256(selected_raw).hexdigest() != root["file_sha256"]:
        raise RuntimeError("parent capsule selected relation disagrees with certificate")
    baseline_sha = hashlib.sha256(baseline_raw).hexdigest()
    expected_baseline = parent.get("normalized_baseline_artifact_sha256")
    if expected_baseline and baseline_sha != expected_baseline:
        raise RuntimeError("parent capsule normalized baseline disagrees with certificate")
    return {
        "status": "PASS",
        "capsule_sha256": sha256_file(path),
        "parent": parent,
        "parent_file_sha256": hashlib.sha256(parent_raw).hexdigest(),
        "selected_relation_bytes": selected_raw,
        "normalized_baseline_bytes": baseline_raw,
    }


def freeze_chain_plan(*, chain_id: str, bootstrap_parent_pin: dict[str, Any], bootstrap_parent_pin_sha256: str, runtime_identity: dict[str, Any], primitive_identity: dict[str, Any], resource_dir: Path, max_target_level: int, finite_limits: dict[str, int], bootstrap_material: dict[str, Any] | None = None) -> dict[str, Any]:
    level_value = bootstrap_parent_pin.get("level", bootstrap_parent_pin.get("subject", {}).get("level"))
    if level_value is None:
        raise ValueError("chain-start parent level unavailable")
    start_level = int(level_value)
    if max_target_level <= start_level:
        raise ValueError("max_target_level must be greater than start level")
    required_limits = [
        "max_level_attempts", "max_level_wall_seconds", "max_total_wall_seconds",
        "max_total_storage_bytes", "max_unique_candidates", "minimum_free_bytes",
        "generation_chunk_parent_count",
    ]
    for key in required_limits:
        if not isinstance(finite_limits.get(key), int) or finite_limits[key] <= 0:
            raise ValueError(f"finite positive limit required: {key}")
    # Explicit max is copied into the limits as an integrity mirror.
    limits = dict(finite_limits, max_target_level=int(max_target_level))
    resources = {
        "protocol_template_sha256": spec_hash(resource_dir, "SCOUT_GENERIC_LEVEL_PROTOCOL_TEMPLATE_v0.1.json"),
        "selection_rules_sha256": spec_hash(resource_dir, "SCOUT_CONVEYOR_SELECTION_RULES_v0.1.json"),
        "observer_sha256": spec_hash(resource_dir, "SCOUT_CONVEYOR_GENERIC_OBSERVER_v0.1.json"),
        "evidence_boundary_sha256": spec_hash(resource_dir, "SCOUT_CONVEYOR_EVIDENCE_CLAIM_FIREWALL_v0.1.json"),
        "trigger_registry_sha256": spec_hash(resource_dir, "SCOUT_CONVEYOR_ESCALATION_TRIGGER_REGISTRY_v0.1.json"),
        "state_machine_sha256": spec_hash(resource_dir, "SCOUT_CONVEYOR_STATE_MACHINE_v0.1.json"),
    }
    plan = {
        "schema_id": CHAIN_SCHEMA,
        "chain_id": chain_id,
        "start_level": start_level,
        "max_target_level": int(max_target_level),
        "bootstrap_parent_pin": {
            "sha256": bootstrap_parent_pin_sha256,
            "certificate_sha256": bootstrap_parent_pin.get("certificate_sha256"),
            "level": start_level,
            "selected_relation_file_sha256": bootstrap_parent_pin["root_material"]["file_sha256"],
            "selected_relation_canonical_science_sha256": bootstrap_parent_pin["root_material"]["canonical_science_sha256"],
            "selected_object_count": bootstrap_parent_pin["root_material"]["selected_object_count"],
            "status": bootstrap_parent_pin["status"],
        },
        "runtime_identity": runtime_identity,
        "primitive_identity": primitive_identity,
        "bootstrap_material": bootstrap_material or {},
        "reviewed_markers_cumulative": sorted(set(
            bootstrap_parent_pin.get(
                "reviewed_markers_cumulative",
                bootstrap_parent_pin.get("classification", {}).get("current_level_escalated_for_review", []),
            )
        )),
        **resources,
        "finite_limits": limits,
        "execution_policy": {
            "auto_advance": True,
            "stop_on_any_integrity_failure": True,
            "stop_on_any_scientific_trigger": True,
            "resume_verified_work_only": True,
            "spectroscope_non_feedback": True,
            "claim_generation": "FORBIDDEN",
            "fail_closed_on_mismatch": True,
        },
        "created_utc": utc_now(),
    }
    plan["chain_plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_chain_plan(plan: dict[str, Any]) -> bool:
    if plan.get("schema_id") != CHAIN_SCHEMA:
        raise ValueError("bad conveyor chain schema")
    if _canon_hash_excluding(plan, "chain_plan_sha256") != plan.get("chain_plan_sha256"):
        raise ValueError("chain plan hash mismatch")
    if int(plan.get("max_target_level", -1)) <= int(plan.get("start_level", -1)):
        raise ValueError("finite target cap missing or invalid")
    if int(plan.get("finite_limits", {}).get("max_target_level", -1)) != int(plan["max_target_level"]):
        raise ValueError("max target integrity mirror mismatch")
    if plan.get("execution_policy", {}).get("claim_generation") != "FORBIDDEN":
        raise ValueError("claim generation must be forbidden")
    return True


def parent_certificate_sha256(parent: dict[str, Any]) -> str:
    if parent.get("schema_id") == PARENT_CERT_SCHEMA:
        declared = parent.get("certificate_sha256")
        observed = _canon_hash_excluding(parent, "certificate_sha256")
        if declared != observed:
            raise RuntimeError("parent certificate SHA mismatch")
        return declared
    # Bootstrap L16 handoff is file-pinned by the chain plan; use canonical content
    # only for deterministic packet derivation after its file identity is verified.
    return canonical_sha256(parent)


def _packet_seed(chain_plan_sha256: str, parent_cert_sha256: str, target_level: int) -> str:
    return canonical_sha256({"chain_plan_sha256": chain_plan_sha256, "parent_certificate_sha256": parent_cert_sha256, "target_level": int(target_level)})


def _parent_root(parent: dict[str, Any]) -> tuple[int, str]:
    if parent.get("schema_id") == PARENT_CERT_SCHEMA:
        rm = parent["root_material"]
    else:
        rm = parent["root_material"]
    return int(rm["selected_object_count"]), rm["file_sha256"]


def derive_child_packet(*, chain_plan: dict[str, Any], parent_certificate: dict[str, Any], fixture_dataset_sha256: str, protocol_descriptor_sha256: str, protocol_version: str = "0.19.0") -> dict[str, Any]:
    validate_chain_plan(chain_plan)
    source_level = int(parent_certificate.get("level", parent_certificate.get("subject", {}).get("level", -1)))
    if source_level < 0:
        raise ValueError("parent level unavailable")
    target_level = source_level + 1
    if target_level > int(chain_plan["max_target_level"]):
        raise ValueError("target exceeds frozen chain cap")
    if parent_certificate.get("schema_id") == PARENT_CERT_SCHEMA:
        if parent_certificate.get("status") != "ELIGIBLE_AUTO_PARENT":
            raise RuntimeError("child derivation requires ELIGIBLE_AUTO_PARENT, not a candidate or paused parent")
        psha = parent_certificate_sha256(parent_certificate)
    else:
        if parent_certificate.get("status") != "ELIGIBLE_FOR_SEPARATELY_FROZEN_L17_PARENT":
            raise RuntimeError("bootstrap child derivation requires the frozen accepted L16 parent status")
        # The frozen L16 bootstrap parent is identified by the exact handoff-pin
        # file SHA-256 recorded in the chain plan.  Content canonicalization is
        # useful for validation but must not silently replace that frozen file
        # identity in child-packet derivation.
        psha = chain_plan["bootstrap_parent_pin"]["sha256"]
        if source_level != int(chain_plan["bootstrap_parent_pin"]["level"]):
            raise RuntimeError("bootstrap parent level disagrees with chain plan")
        rm = parent_certificate.get("root_material", {})
        if rm.get("file_sha256") != chain_plan["bootstrap_parent_pin"]["selected_relation_file_sha256"]:
            raise RuntimeError("bootstrap selected-relation file identity disagreement")
        if rm.get("canonical_science_sha256") != chain_plan["bootstrap_parent_pin"]["selected_relation_canonical_science_sha256"]:
            raise RuntimeError("bootstrap selected-relation canonical identity disagreement")
        if int(rm.get("selected_object_count", -1)) != int(chain_plan["bootstrap_parent_pin"]["selected_object_count"]):
            raise RuntimeError("bootstrap selected-object count disagreement")
    seed = _packet_seed(chain_plan["chain_plan_sha256"], psha, target_level)
    packet_id = f"scout-l{target_level}-{seed[:16]}"
    run_id = f"run-scout-l{target_level}-chain-{chain_plan['chain_plan_sha256'][:12]}-{seed[:16]}"
    root_count, root_sha = _parent_root(parent_certificate)
    question_text = f"How do the five frozen bounded Scout relation families change from L{source_level} to L{target_level} under one complete lawful primitive-attachment step, exact D/B/S/O/L selection, sealed shallow relational analysis, and recognition-only Spectroscope?"
    stopping_rules = {
        "complete_one_step_generation": True,
        "no_adaptive_panel_extension": True,
        "no_deep_ancestry_search": True,
        "no_population_census": True,
        "selection_sealed_before_relation_analysis": True,
        "spectroscope_after_seal_only": True,
    }
    subject = {"hierarchy": "L", "level": target_level, "source_level": source_level, "live_generation": True, "conveyor_managed": True}
    question = {"protocol_id": "SCOUT", "subject": subject, "protocol_label": "SCOUT_OBSERVED", "input_dataset_sha256": fixture_dataset_sha256, "stopping_rules": stopping_rules, "question_text": question_text}
    qsha = canonical_sha256(question)
    common = {
        "fixture_dataset_sha256": fixture_dataset_sha256,
        "source_level": source_level,
        "target_level": target_level,
        "root_sha256": root_sha,
        "primitive_sha256": chain_plan["primitive_identity"]["sha256"],
        "selection_rules_sha256": chain_plan["selection_rules_sha256"],
        "observer_sha256": chain_plan["observer_sha256"],
        "claim_boundary_sha256": chain_plan["evidence_boundary_sha256"],
    }
    stages = [
        {"stage_id": "preflight", "depends_on": [], "runner": "adapter.scout_level.preflight", "params": dict(common, root_count=root_count)},
        {"stage_id": "wide", "depends_on": ["preflight"], "runner": "adapter.scout_level.wide_chunked", "params": dict(common, per_lane=256, lane_ids=["D", "B", "S", "O", "L"], generation_chunk_parent_count=chain_plan["finite_limits"]["generation_chunk_parent_count"])},
        {"stage_id": "spectroscope", "depends_on": ["wide"], "runner": "adapter.scout_level.spectroscope", "params": {"fixture_dataset_sha256": fixture_dataset_sha256, "target_level": target_level, "recognition_only": True, "cannot_feedback": True}},
        {"stage_id": "classify", "depends_on": ["wide", "spectroscope"], "runner": "adapter.scout_level.classify", "params": {"fixture_dataset_sha256": fixture_dataset_sha256, "source_level": source_level, "target_level": target_level, "observer_sha256": chain_plan["observer_sha256"], "claim_boundary_sha256": chain_plan["evidence_boundary_sha256"]}},
        {"stage_id": "verify", "depends_on": ["classify"], "runner": "adapter.scout_level.verify_independent", "params": {"fixture_dataset_sha256": fixture_dataset_sha256, "source_level": source_level, "target_level": target_level, "per_lane": 256, "selection_rules_sha256": chain_plan["selection_rules_sha256"], "observer_sha256": chain_plan["observer_sha256"], "claim_boundary_sha256": chain_plan["evidence_boundary_sha256"]}},
    ]
    execution_plan = {
        "schema_id": "IG_EXECUTION_PLAN_V0_17",
        "protocol_id": "SCOUT",
        "protocol_version": protocol_version,
        "descriptor_sha256": protocol_descriptor_sha256,
        "protocol_label": "SCOUT_OBSERVED",
        "subject": subject,
        "question": question,
        "question_sha256": qsha,
        "input_datasets": [{"dataset_sha256": fixture_dataset_sha256, "logical_role": "SCOUT_CONVEYOR_LEVEL_FIXTURE"}],
        "input_flags": {"bounded_preregistered": True, "parent_certified": True, "separate_lineage": True, "observer_frozen": True, "selection_sealed_before_relation_analysis": True, "spectroscope_non_feedback": True, "population_complete": False, "conveyor_managed": True},
        "requested_claims": [],
        "observer": {"observer_sha256": chain_plan["observer_sha256"], "role": "BOUNDED_SHALLOW_SCOUT_RELATIONAL_OBSERVER"},
        "evidence": {"origin": "DIRECT_EXECUTION", "scope": "BOUNDED_PREREGISTERED", "disposition": "OBSERVED", "authority": "RECONNAISSANCE_ONLY", "protocol_label": "SCOUT_OBSERVED"},
        "stages": stages,
        "run_id": run_id,
    }
    execution_plan["plan_sha256"] = canonical_sha256(execution_plan)
    packet = {
        "schema_id": PACKET_SCHEMA,
        "packet_id": packet_id,
        "packet_seed_sha256": seed,
        "chain_id": chain_plan["chain_id"],
        "chain_plan_sha256": chain_plan["chain_plan_sha256"],
        "parent_certificate_sha256": psha,
        "source_level": source_level,
        "target_level": target_level,
        "root_count": root_count,
        "root_sha256": root_sha,
        "fixture_dataset_sha256": fixture_dataset_sha256,
        "evidence": ["DIRECT_EXECUTION", "BOUNDED_PREREGISTERED", "OBSERVED", "RECONNAISSANCE_ONLY", "SCOUT_OBSERVED"],
        "question": question,
        "question_sha256": qsha,
        "execution_plan": execution_plan,
        "execution_plan_sha256": execution_plan["plan_sha256"],
        "run_id": run_id,
        "status": "FROZEN_NOT_EXECUTED",
        "L17_or_target_executed_by_derivation": False,
    }
    packet["packet_sha256"] = canonical_sha256(packet)
    return packet


def derive_child_packet_independent(**kwargs) -> dict[str, Any]:
    # Independent construction path intentionally reserializes and re-derives
    # the output rather than aliasing the returned object.
    first = derive_child_packet(**kwargs)
    raw = json.loads(canonical_text(first))
    # Validate every deterministic identity from its ingredients.
    if raw["packet_seed_sha256"] != _packet_seed(raw["chain_plan_sha256"], raw["parent_certificate_sha256"], raw["target_level"]):
        raise RuntimeError("independent packet seed disagreement")
    if raw["question_sha256"] != canonical_sha256(raw["question"]):
        raise RuntimeError("independent question disagreement")
    ep = raw["execution_plan"]
    if ep["plan_sha256"] != _canon_hash_excluding(ep, "plan_sha256"):
        raise RuntimeError("independent execution plan disagreement")
    if raw["packet_sha256"] != _canon_hash_excluding(raw, "packet_sha256"):
        raise RuntimeError("independent packet identity disagreement")
    return raw


def freeze_child_packet(directory: Path, packet: dict[str, Any]) -> dict[str, Any]:
    """Freeze a child packet, or verify/reuse an identical dry packet.

    This makes a frozen execution handoff directly startable.  The dry packet
    stays in place and is never manually removed merely to begin execution.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    verify = {
        "status": "PASS",
        "packet_sha256": packet["packet_sha256"],
        "question_sha256": packet["question_sha256"],
        "execution_plan_sha256": packet["execution_plan_sha256"],
        "run_id": packet["run_id"],
        "target_level": packet["target_level"],
        "science_executed": False,
    }
    expected = {
        "packet.json": packet,
        "question.json": packet["question"],
        "plan.json": packet["execution_plan"],
        "packet_verification.json": verify,
    }
    present = {q.name for q in directory.iterdir() if q.is_file()}
    if present:
        if present != set(expected):
            raise RuntimeError("pre-existing child packet inventory mismatch")
        for name, obj in expected.items():
            observed = json.loads((directory / name).read_text(encoding="utf-8"))
            if canonical_text(observed) != canonical_text(obj):
                raise RuntimeError(f"pre-existing child packet mismatch: {name}")
        return dict(verify, reused=True)
    for name, obj in expected.items():
        write_json_atomic(directory / name, obj)
    return dict(verify, reused=False)

def registered_recognition_tokens(resource_dir: Path) -> set[str]:
    vec = json.loads((Path(resource_dir) / "SCOUT_CONVEYOR_REFERENCE_TEST_VECTORS_v0.1.json").read_text(encoding="utf-8"))
    return set(vec["registered_recognition_tokens"])


def evaluate_scientific_triggers(*, metrics: dict[str, Any], classification: dict[str, Any], parent: dict[str, Any], spectroscope: dict[str, Any], resource_dir: Path, representation_collision: bool = False) -> list[str]:
    """Evaluate frozen scientific stop predicates.

    Ordinary classification, including ``R_PARENT_FIBER.EXPANDS``, is never a
    review escalation.  ``S_NEW_REVIEW_ESCALATION`` requires a distinct
    explicit review predicate absent from the cumulative reviewed registry.
    """
    out: list[str] = []
    lawful = int(metrics.get("lawful", 0))
    unique = int(metrics.get("unique_candidates", 0))
    selected = int(metrics.get("selected_union", 0))
    if lawful == 0: out.append("S_ENDPOINT_NO_LAWFUL_CHILDREN")
    if unique == 0: out.append("S_ENDPOINT_NO_UNIQUE_CANDIDATES")
    if unique < 256: out.append("S_LOW_CANDIDATE_SUPPORT")
    if selected == 0: out.append("S_EMPTY_SELECTED_RELATION")
    parent_persistent = set(parent.get("persistent", parent.get("classification", {}).get("persistent", [])))
    current_persistent = set(classification.get("persistent", []))
    if parent_persistent - current_persistent: out.append("S_TRACKED_RELATION_DISAPPEARS")
    known = registered_recognition_tokens(resource_dir)
    unknown_tokens = set(spectroscope.get("recognition_summary", [])) - known
    if classification.get("observed_or_new") or unknown_tokens:
        out.append("S_NEW_RELATION_OR_UNREGISTERED_TOKEN")
    reviewed = set(parent.get("reviewed_markers_cumulative", parent.get("reviewed_markers", [])))
    explicit = set(classification.get("explicit_review_escalations", classification.get("explicit_review_markers", [])))
    if explicit - reviewed: out.append("S_NEW_REVIEW_ESCALATION")
    parent_max = int(parent.get("max_parent_count", parent.get("relations", {}).get("R_PARENT_FIBER", {}).get("selected_max_parent_count", 0)))
    current_max = int(metrics.get("selected_max_parent_count", 0))
    if parent_max > 1 and current_max <= 1: out.append("S_PARENT_FIBER_COLLAPSE")
    pair_count = int(metrics.get("distinct_fiber_pairs", 0)); both = int(metrics.get("distinct_fiber_pairs_both_realized", 0))
    if pair_count > 0 and both == pair_count: out.append("S_EXACT_PAIRWISE_CLOSURE_CANDIDATE")
    depth_status = classification.get("depth_status")
    if depth_status is None and "R_LOCAL_DEPTH_REORGANIZATION" not in classification.get("unresolved", []): depth_status = "UNKNOWN"
    if depth_status not in {None, "UNRESOLVED_DEPTH_SCOPE"}: out.append("S_DEPTH_MARKER_RESOLVED_OR_CHANGED")
    if classification.get("classification_defined") is False or classification.get("summary") == "UNDEFINED": out.append("S_CLASSIFICATION_UNDEFINED")
    if representation_collision: out.append("S_REPRESENTATION_OR_FUTURE_COLLISION")
    return sorted(set(out))

def decide_chain_state(*, integrity: list[str] | None = None, operational: list[str] | None = None, scientific: list[str] | None = None, operator_stop: bool = False, parent_eligible: bool = True, target_level: int, max_target_level: int) -> str:
    integrity = integrity or []
    operational = operational or []
    scientific = scientific or []
    if integrity:
        return "FAIL_CLOSED"
    if operator_stop:
        return "PAUSED_OPERATOR"
    if operational:
        # Chain-wide resource exhaustion has a dedicated terminal state.
        if any(x in {"O_CHAIN_TIME_BUDGET"} for x in operational):
            return "COMPLETE_RESOURCE_LIMIT"
        return "PAUSED_OPERATIONAL"
    if scientific:
        return "PAUSED_SCIENTIFIC"
    if not parent_eligible:
        return "FAIL_CLOSED"
    if int(target_level) >= int(max_target_level):
        return "COMPLETE_LIMIT_REACHED"
    return "AUTO_ADVANCE_AUTHORIZED"


def write_chain_event(path: Path, *, event: str, chain_plan_sha256: str, current_parent_certificate_sha256: str, **fields) -> dict[str, Any]:
    rec = {"event": event, "utc": utc_now(), "chain_plan_sha256": chain_plan_sha256, "current_parent_certificate_sha256": current_parent_certificate_sha256, **fields}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as h:
        h.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        h.flush()
        try:
            os.fsync(h.fileno())
        except OSError:
            pass
    return rec


def build_parent_candidate(*, level: int, source_level: int, selected_relation_file_sha256: str, selected_relation_canonical_sha256: str, selected_object_count: int, baseline_sha256: str, classification_sha256: str, runtime_identity: dict[str, Any], protocol_hashes: dict[str, Any], bounded_relation_summary: dict[str, Any], spectroscope_summary: dict[str, Any], trigger_evaluation: dict[str, Any], release_verification: dict[str, Any], catalog_verification: dict[str, Any], classification: dict[str, Any], reviewed_markers_cumulative: list[str] | None = None) -> dict[str, Any]:
    cert = {
        "schema_id": PARENT_CERT_SCHEMA,
        "status": "PARENT_CANDIDATE",
        "level": int(level),
        "source_level": int(source_level),
        "evidence_label": "SCOUT_OBSERVED",
        "authority": "RECONNAISSANCE_ONLY",
        "population_complete": False,
        "claim_record_count": 0,
        "root_material": {"file_sha256": selected_relation_file_sha256, "canonical_science_sha256": selected_relation_canonical_sha256, "selected_object_count": int(selected_object_count)},
        "baseline_sha256": baseline_sha256,
        "classification_sha256": classification_sha256,
        "runtime_identity": runtime_identity,
        "protocol_hashes": protocol_hashes,
        "bounded_relation_summary": bounded_relation_summary,
        "spectroscope_summary": spectroscope_summary,
        "trigger_evaluation": trigger_evaluation,
        "release_verification": release_verification,
        "catalog_verification": catalog_verification,
        "classification": {
            "persistent": classification.get("persistent", []),
            "current_level_escalated_for_review": classification.get("current_level_escalated_for_review", []),
            "explicit_review_escalations": classification.get("explicit_review_escalations", classification.get("explicit_review_markers", [])),
            "summary": classification.get("summary"),
        },
        "reviewed_markers_cumulative": sorted(set(reviewed_markers_cumulative or [])),
        "forbidden_parent_inference": ["first occurrence", "population-wide absence", "mechanism", "geometry", "physical topology", "transition level", "graduation", "lattice or closed algebra"],
    }
    cert["certificate_sha256"] = canonical_sha256(cert)
    return cert

def prepare_eligible_parent_certificate(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("status") != "PARENT_CANDIDATE":
        raise RuntimeError("phase-1 parent candidate required")
    if candidate.get("trigger_evaluation", {}).get("active"):
        raise RuntimeError("cannot commit parent with active trigger")
    if candidate.get("claim_record_count") != 0 or candidate.get("authority") != "RECONNAISSANCE_ONLY" or candidate.get("population_complete") is not False:
        raise RuntimeError("parent candidate evidence boundary violation")
    eligible = dict(candidate)
    eligible.pop("certificate_sha256", None)
    eligible["status"] = "ELIGIBLE_AUTO_PARENT"
    eligible["committed_utc"] = utc_now()
    eligible["certificate_sha256"] = canonical_sha256(eligible)
    validate_eligible_parent_certificate(eligible)
    return eligible


def commit_prepared_parent_certificate(*, eligible: dict[str, Any], eligible_path: Path, current_pointer_path: Path) -> dict[str, Any]:
    validate_eligible_parent_certificate(eligible)
    write_json_atomic(eligible_path, eligible)
    pointer = {"status": "COMMITTED", "level": eligible["level"], "certificate_sha256": eligible["certificate_sha256"], "certificate_path": str(Path(eligible_path).name)}
    tmp = Path(current_pointer_path).with_suffix(".tmp")
    write_json_atomic(tmp, pointer)
    os.replace(tmp, current_pointer_path)
    return eligible


def commit_parent_certificate(*, candidate_path: Path, eligible_path: Path, current_pointer_path: Path) -> dict[str, Any]:
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    eligible = prepare_eligible_parent_certificate(candidate)
    return commit_prepared_parent_certificate(eligible=eligible, eligible_path=eligible_path, current_pointer_path=current_pointer_path)


def rebuild_chain_state(*, chain_plan: dict[str, Any], events_path: Path, parent_certificates_dir: Path) -> dict[str, Any]:
    validate_chain_plan(chain_plan)
    events = []
    if Path(events_path).is_file():
        for line in Path(events_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("chain_plan_sha256") != chain_plan["chain_plan_sha256"]:
                    raise RuntimeError("event chain binding mismatch")
                events.append(rec)
    certs = []
    for p in sorted(Path(parent_certificates_dir).glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("schema_id") != PARENT_CERT_SCHEMA:
            continue
        if _canon_hash_excluding(obj, "certificate_sha256") != obj.get("certificate_sha256"):
            raise RuntimeError("parent certificate corruption")
        certs.append(obj)
    eligible = sorted((c for c in certs if c.get("status") == "ELIGIBLE_AUTO_PARENT"), key=lambda c: (c["level"], c["certificate_sha256"]))
    current = eligible[-1] if eligible else None
    state = events[-1].get("state") if events else "FROZEN"
    return {"status": "PASS", "chain_id": chain_plan["chain_id"], "state": state, "event_count": len(events), "eligible_parent_count": len(eligible), "current_parent_level": current.get("level") if current else chain_plan["start_level"], "current_parent_certificate_sha256": current.get("certificate_sha256") if current else chain_plan["bootstrap_parent_pin"]["sha256"]}


def create_minimal_parent_capsule(*, destination_zip: Path, certificate_path: Path, selected_relation_path: Path, normalized_baseline_path: Path) -> dict[str, Any]:
    import zipfile
    destination_zip = Path(destination_zip)
    files = [("parent_certificate.json", Path(certificate_path)), ("SELECTED_RELATION.json", Path(selected_relation_path)), ("PARENT_NORMALIZED_BASELINE.json", Path(normalized_baseline_path))]
    manifest = []
    for name, p in files:
        manifest.append({"path": name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    man = {"schema_id": "IG_SCOUT_FORWARD_PARENT_CAPSULE_V0_1", "files": manifest, "authority": "RECONNAISSANCE_ONLY", "population_complete": False, "claims": []}
    man_bytes = (json.dumps(man, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(destination_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, p in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo("capsule_manifest.json", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
        z.writestr(info, man_bytes)
    return {"status": "PASS", "archive": str(destination_zip), "sha256": sha256_file(destination_zip), "files": len(files) + 1}


def verify_minimal_parent_capsule(path: Path) -> dict[str, Any]:
    import zipfile
    path = Path(path)
    failures: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            bad = z.testzip()
            if bad:
                return {"status": "FAIL", "bad_member": bad, "sha256": sha256_file(path)}
            required = {"parent_certificate.json", "SELECTED_RELATION.json", "PARENT_NORMALIZED_BASELINE.json", "capsule_manifest.json"}
            if not required.issubset(set(z.namelist())):
                failures.append("missing_required_member")
            man = json.loads(z.read("capsule_manifest.json"))
            for f in man.get("files", []):
                raw = z.read(f["path"])
                if hashlib.sha256(raw).hexdigest() != f["sha256"] or len(raw) != f["size_bytes"]:
                    failures.append(f["path"])
            if man.get("claims") != [] or man.get("authority") != "RECONNAISSANCE_ONLY" or man.get("population_complete") is not False:
                failures.append("evidence_boundary")
            parent = json.loads(z.read("parent_certificate.json"))
            try:
                validate_eligible_parent_certificate(parent)
            except Exception:
                failures.append("eligible_parent_certificate")
            selected_raw = z.read("SELECTED_RELATION.json")
            baseline_raw = z.read("PARENT_NORMALIZED_BASELINE.json")
            if parent.get("root_material", {}).get("file_sha256") != hashlib.sha256(selected_raw).hexdigest():
                failures.append("selected_relation_binding")
            expected_baseline = parent.get("normalized_baseline_artifact_sha256")
            if expected_baseline and expected_baseline != hashlib.sha256(baseline_raw).hexdigest():
                failures.append("normalized_baseline_binding")
    except Exception as exc:
        failures.append(f"exception:{type(exc).__name__}")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "sha256": sha256_file(path)}


def verify_chain_runtime_identity(chain_plan: dict[str, Any], *, source_sha256: str, wheel_sha256: str, runtime_sha256: str) -> bool:
    rid = chain_plan["runtime_identity"]
    if rid.get("source_sha256") != source_sha256 or rid.get("wheel_sha256") != wheel_sha256 or rid.get("runtime_sha256") != runtime_sha256:
        raise RuntimeError("I_RUNTIME_IDENTITY_DRIFT")
    return True


def _chain_usage_path(chain_dir: Path) -> Path:
    return Path(chain_dir) / "chain_usage.json"


def record_verified_level_usage(*, chain_dir: Path, chain_plan_sha256: str, level: int, run_id: str, wall_seconds: float) -> dict[str, Any]:
    """Idempotently record one verified level and return cumulative wall time."""
    path = _chain_usage_path(chain_dir)
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("schema_id") != CHAIN_USAGE_SCHEMA or obj.get("chain_plan_sha256") != chain_plan_sha256:
            raise RuntimeError("chain usage binding mismatch")
    else:
        obj = {"schema_id": CHAIN_USAGE_SCHEMA, "chain_plan_sha256": chain_plan_sha256, "levels": {}}
    key = str(int(level)); entry = {"level": int(level), "run_id": str(run_id), "wall_seconds": float(wall_seconds)}
    prior = obj["levels"].get(key)
    if prior is not None and canonical_text(prior) != canonical_text(entry):
        raise RuntimeError("verified level usage changed across resume")
    obj["levels"][key] = entry
    obj["total_verified_level_wall_seconds"] = float(sum(float(v["wall_seconds"]) for v in obj["levels"].values()))
    write_json_atomic(path, obj)
    return obj


def cumulative_reviewed_markers(parent: dict[str, Any], classification: dict[str, Any] | None = None) -> list[str]:
    reviewed = set(parent.get("reviewed_markers_cumulative", parent.get("reviewed_markers", [])))
    if classification is not None:
        reviewed.update(classification.get("explicit_review_escalations", classification.get("explicit_review_markers", [])))
    return sorted(reviewed)


def operational_triggers(*, chain_plan: dict[str, Any], attempts: int, unique_candidates: int, level_wall_seconds: float, chain_wall_seconds: float, total_storage_bytes: int, free_bytes: int) -> list[str]:
    lim = chain_plan["finite_limits"]
    out = []
    if attempts > lim["max_level_attempts"]:
        out.append("O_ATTEMPT_BUDGET")
    if unique_candidates > lim["max_unique_candidates"]:
        out.append("O_CANDIDATE_BUDGET")
    no_deadline=os.environ.get('IG_DECODER_EXECUTION_POLICY')=='NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
    if not no_deadline and level_wall_seconds > lim["max_level_wall_seconds"]:
        out.append("O_LEVEL_TIME_BUDGET")
    if not no_deadline and chain_wall_seconds > lim["max_total_wall_seconds"]:
        out.append("O_CHAIN_TIME_BUDGET")
    if total_storage_bytes > lim["max_total_storage_bytes"]:
        out.append("O_STORAGE_BUDGET")
    if free_bytes < lim["minimum_free_bytes"]:
        out.append("O_FREE_SPACE_RESERVE")
    return out


def _artifact_by_logical(run_record: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [a for a in run_record.get("result_artifacts", []) if a.get("logical_name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one run artifact {name}, found {len(matches)}")
    return matches[0]


class ScoutChainController:
    """Finite outer controller for the frozen Scout Level Conveyor.

    New-level execution is disabled until an explicit conveyor graduation record is
    installed.  This class is therefore implementation-complete but safe by default.
    """

    def __init__(self, paths):
        from .store import ArtifactStore
        from .datasets import DatasetStore
        from .controller import Controller
        from .catalog import Catalogue
        from .releases import ReleaseGenerator, verify_release_archive
        self.paths = paths.ensure()
        self.store = ArtifactStore(paths.store)
        self.datasets = DatasetStore(self.store)
        self.runtime = Controller(paths)
        self.catalogue = Catalogue(paths)
        self.release_gen = ReleaseGenerator(paths)
        self.verify_release_archive = verify_release_archive

    def chain_dir(self, chain_id: str) -> Path:
        from .safety import contained_path, validate_identifier
        root = self.paths.runs / "chains"
        root.mkdir(parents=True, exist_ok=True)
        validate_identifier(chain_id, field="chain_id")
        p = contained_path(root, chain_id, field="chain_id")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def graduation_record(self) -> dict[str, Any] | None:
        p = self.paths.store / "conveyor" / "graduation.json"
        if not p.is_file():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("status") != "GRADUATED" or obj.get("new_level_execution_enabled") is not True:
            return None
        return obj

    def assert_execution_enabled(self) -> None:
        if self.graduation_record() is None:
            raise RuntimeError("SCOUT_CONVEYOR_IMPLEMENTATION_ONLY_NOT_GRADUATED_FOR_NEW_LEVEL_EXECUTION")

    def _build_fixture(self, chain_plan: dict[str, Any], parent: dict[str, Any], chain_dir: Path, target_level: int) -> str:
        tmp = self.paths.workspace / f"chain-{chain_plan['chain_id']}" / f"fixture-l{target_level}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        if parent.get("schema_id") == PARENT_CERT_SCHEMA:
            root_sha = parent["root_material"]["artifact_sha256"]
            baseline_sha = parent["normalized_baseline_artifact_sha256"]
        else:
            mat = chain_plan["bootstrap_material"]
            root_sha = mat["selected_relation_artifact_sha256"]
            baseline_sha = mat["normalized_baseline_artifact_sha256"]
        self.store.materialize(root_sha, tmp / "PARENT_SELECTED_RELATION.json")
        self.store.materialize(baseline_sha, tmp / "PARENT_NORMALIZED_BASELINE.json")
        self.store.materialize(chain_plan["primitive_identity"]["artifact_sha256"], tmp / "primitive.pkl")
        ds = self.datasets.import_directory(tmp, logical_role=f"SCOUT_CONVEYOR_L{target_level}_FIXTURE")
        return ds["dataset_sha256"]

    def _current_parent(self, chain_plan: dict[str, Any], chain_dir: Path) -> dict[str, Any]:
        ptr = chain_dir / "current_parent.json"
        if ptr.is_file():
            obj = json.loads(ptr.read_text(encoding="utf-8"))
            cp = chain_dir / "parents" / obj["certificate_path"]
            return json.loads(cp.read_text(encoding="utf-8"))
        return json.loads((chain_dir / "bootstrap_parent.json").read_text(encoding="utf-8"))

    def install_execution_handoff(self, chain_plan: dict[str, Any], chain_dir: Path) -> dict[str, Any]:
        """Verify and install the immutable wheel/graduation handoff."""
        from .lifecycle import ExecutionModeManager
        hdir = Path(chain_dir) / "handoff"; manifest_path = hdir / "HANDOFF_MANIFEST.json"
        if not manifest_path.is_file():
            self.assert_execution_enabled()
            return {"status": "PASS", "mode": "PREINSTALLED_ROOT"}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("chain_plan_sha256") != chain_plan["chain_plan_sha256"]:
            raise RuntimeError("execution handoff chain-plan mismatch")
        for rec in manifest.get("files", []):
            q = hdir / rec["path"]
            if not q.is_file() or sha256_file(q) != rec["sha256"] or q.stat().st_size != rec["size_bytes"]:
                raise RuntimeError(f"execution handoff file mismatch: {rec['path']}")
        runtime_grad = json.loads((hdir / "UNIFIED_RUNTIME_GRADUATION_RECORD.json").read_text(encoding="utf-8"))
        ExecutionModeManager(self.paths).install_graduation(runtime_grad)
        conveyor_grad = json.loads((hdir / "CONVEYOR_GRADUATION_RECORD.json").read_text(encoding="utf-8"))
        if conveyor_grad.get("status") != "GRADUATED" or conveyor_grad.get("new_level_execution_enabled") is not True:
            raise RuntimeError("execution handoff conveyor graduation invalid")
        observed_identity = conveyor_grad.get("runtime_identity", {})
        for key in ("source_sha256", "wheel_sha256", "runtime_sha256"):
            expected = chain_plan["runtime_identity"].get(key); observed = observed_identity.get(key, conveyor_grad.get(key))
            if observed is not None and observed != expected:
                raise RuntimeError(f"execution handoff conveyor runtime mismatch: {key}")
        cgpath = self.paths.store / "conveyor" / "graduation.json"; cgpath.parent.mkdir(parents=True, exist_ok=True)
        if cgpath.is_file() and canonical_text(json.loads(cgpath.read_text(encoding="utf-8"))) != canonical_text(conveyor_grad):
            raise RuntimeError("conflicting installed conveyor graduation")
        write_json_atomic(cgpath, conveyor_grad)
        wheels = [rec for rec in manifest.get("files", []) if rec["path"].startswith("wheelhouse/") and rec["path"].endswith(".whl")]
        if len(wheels) != 1: raise RuntimeError("execution handoff must contain exactly one wheel")
        wrec = wheels[0]; src = hdir / wrec["path"]
        if wrec["sha256"] != chain_plan["runtime_identity"]["wheel_sha256"]: raise RuntimeError("execution handoff wheel identity mismatch")
        wh = self.paths.app / "wheelhouse"; wh.mkdir(parents=True, exist_ok=True)
        for q in wh.glob("*.whl"):
            if sha256_file(q) != wrec["sha256"]: raise RuntimeError("conflicting runtime wheel already installed")
        dst = wh / src.name
        if not dst.is_file(): shutil.copy2(src, dst)
        return {"status": "PASS", "mode": "HANDOFF_INSTALLED", "wheel_sha256": wrec["sha256"]}

    def derive_and_freeze_next_packet(self, chain_plan: dict[str, Any], chain_dir: Path) -> tuple[dict[str, Any], Path]:
        parent = self._current_parent(chain_plan, chain_dir)
        source_level = int(parent.get("level", parent.get("subject", {}).get("level")))
        target = source_level + 1
        fixture = self._build_fixture(chain_plan, parent, chain_dir, target)
        desc = ProtocolRegistry().get("SCOUT", version="0.19.0")
        kwargs = dict(chain_plan=chain_plan, parent_certificate=parent, fixture_dataset_sha256=fixture, protocol_descriptor_sha256=desc["descriptor_sha256"], protocol_version="0.19.0")
        a = derive_child_packet(**kwargs)
        b = derive_child_packet_independent(**kwargs)
        if canonical_text(a) != canonical_text(b):
            raise RuntimeError("I_PACKET_DERIVATION_MISMATCH")
        pdir = chain_dir / "packets" / a["packet_id"]
        freeze_child_packet(pdir, a)
        return a, pdir

    def execute_one_level(self, chain_plan: dict[str, Any], chain_dir: Path) -> dict[str, Any]:
        self.assert_execution_enabled()
        packet, _ = self.derive_and_freeze_next_packet(chain_plan, chain_dir)
        parent = self._current_parent(chain_plan, chain_dir)
        run = self.runtime.run(packet["execution_plan"])
        if run.get("lifecycle") != "COMPLETE_VALID": raise RuntimeError("level run did not complete valid")
        level = packet["target_level"]; ldir = chain_dir / "levels" / f"L{level}"; ldir.mkdir(parents=True, exist_ok=True)
        objs = {}
        names = ["SCOUT_LEVEL_WIDE_RESULT.json", "SELECTED_RELATION.json", "SCOUT_SPECTROSCOPE_RESULT.json", "SCOUT_LEVEL_BASELINE.json", "SCOUT_LEVEL_CLASSIFICATION.json", "SCOUT_LEVEL_TERMINAL_VERIFICATION.json"]
        for name in names:
            art = _artifact_by_logical(run, name); dst = ldir / name
            self.store.materialize(art["sha256"], dst, expected_size=art.get("size_bytes")); objs[name] = json.loads(dst.read_text(encoding="utf-8"))
        wide=objs["SCOUT_LEVEL_WIDE_RESULT.json"]; relation=objs["SELECTED_RELATION.json"]; spec=objs["SCOUT_SPECTROSCOPE_RESULT.json"]; baseline=objs["SCOUT_LEVEL_BASELINE.json"]; classification=objs["SCOUT_LEVEL_CLASSIFICATION.json"]
        compact = self.release_gen.create(run["run_id"], "compact"); compact_v = self.verify_release_archive(Path(compact["archive"])); cat = self.catalogue.rebuild()
        psummary = {
            "max_parent_count": parent.get("bounded_relation_summary", {}).get("selected_max_parent_count", 0),
            "persistent": parent.get("classification", {}).get("persistent", []),
            "reviewed_markers_cumulative": cumulative_reviewed_markers(parent),
        }
        inc=spec["closure_breakdown"]["incomparable"]
        metrics={"lawful":wide["generation"]["lawful"],"unique_candidates":wide["generation"]["unique_candidates"],"selected_union":wide["bounded_relation"]["selected_union"],"selected_max_parent_count":wide["bounded_relation"]["selected_max_parent_count"],"distinct_fiber_pairs":inc["pairs"],"distinct_fiber_pairs_both_realized":inc["both_realized"]}
        sci=evaluate_scientific_triggers(metrics=metrics,classification=classification,parent=psummary,spectroscope=spec,resource_dir=Path(__file__).resolve().parent/"resources"/"conveyor")
        usage=run.get("resource_summary",{}); level_wall=float(usage.get("wall_seconds",0))
        chain_usage=record_verified_level_usage(chain_dir=chain_dir,chain_plan_sha256=chain_plan["chain_plan_sha256"],level=level,run_id=run["run_id"],wall_seconds=level_wall)
        chain_wall=float(chain_usage["total_verified_level_wall_seconds"])
        storage=sum(q.stat().st_size for q in chain_dir.rglob("*") if q.is_file()); free=shutil.disk_usage(self.paths.root).free
        op=operational_triggers(chain_plan=chain_plan,attempts=wide["generation"]["attempts"],unique_candidates=metrics["unique_candidates"],level_wall_seconds=level_wall,chain_wall_seconds=chain_wall,total_storage_bytes=storage,free_bytes=free)
        state=decide_chain_state(operational=op,scientific=sci,parent_eligible=(compact_v.get("status")=="PASS" and cat.get("status","PASS")=="PASS"),target_level=level,max_target_level=chain_plan["max_target_level"])
        trig={"active":bool(op or sci),"operational":op,"scientific":sci,"decision":state,"level_wall_seconds":level_wall,"chain_wall_seconds":chain_wall,"reviewed_markers_cumulative":cumulative_reviewed_markers(parent),"explicit_review_escalations":classification.get("explicit_review_escalations",[])}
        write_json_atomic(ldir/"TRIGGER_EVALUATION.json",trig)
        if op or sci:
            write_chain_event(chain_dir/"events.jsonl",event="CHAIN_PAUSED",chain_plan_sha256=chain_plan["chain_plan_sha256"],current_parent_certificate_sha256=parent_certificate_sha256(parent),state=state,level=level,triggers=trig,level_wall_seconds=level_wall,chain_wall_seconds=chain_wall)
            return {"status":state,"level":level,"triggers":trig,"run_id":run["run_id"]}
        from .adapters.scout_level import normalize_parent_baseline
        normalized=normalize_parent_baseline(level=level,baseline=baseline,classification=classification); nb=ldir/"PARENT_NORMALIZED_BASELINE.json"; write_json_atomic(nb,normalized)
        nb_art=self.store.put_file(nb,logical_role="SCOUT_FORWARD_PARENT_BASELINE",source_name=nb.name,created_by_run_id=run["run_id"])
        rel_art=_artifact_by_logical(run,"SELECTED_RELATION.json"); base_art=_artifact_by_logical(run,"SCOUT_LEVEL_BASELINE.json"); cls_art=_artifact_by_logical(run,"SCOUT_LEVEL_CLASSIFICATION.json")
        cert=build_parent_candidate(level=level,source_level=packet["source_level"],selected_relation_file_sha256=rel_art["sha256"],selected_relation_canonical_sha256=canonical_sha256(relation),selected_object_count=len(relation["selected"]),baseline_sha256=base_art["sha256"],classification_sha256=cls_art["sha256"],runtime_identity=chain_plan["runtime_identity"],protocol_hashes={"question_sha256":packet["question_sha256"],"plan_sha256":packet["execution_plan_sha256"],"protocol_descriptor_sha256":packet["execution_plan"]["descriptor_sha256"],"selection_rules_sha256":chain_plan["selection_rules_sha256"],"observer_sha256":chain_plan["observer_sha256"],"evidence_boundary_sha256":chain_plan["evidence_boundary_sha256"]},bounded_relation_summary=wide["bounded_relation"],spectroscope_summary={"recognition_summary":spec["recognition_summary"],"components":spec["incidence"]["components"],"abstract_cycle_rank_recognition_only":spec["incidence"]["abstract_cycle_rank"]},trigger_evaluation=trig,release_verification=compact_v,catalog_verification=cat,classification=classification,reviewed_markers_cumulative=cumulative_reviewed_markers(parent,classification))
        cert["root_material"]["artifact_sha256"]=rel_art["sha256"]; cert["normalized_baseline_artifact_sha256"]=nb_art["sha256"]; cert.pop("certificate_sha256",None); cert["certificate_sha256"]=canonical_sha256(cert)
        parents=chain_dir/"parents"; parents.mkdir(parents=True,exist_ok=True); cand=parents/f"L{level}-PARENT_CANDIDATE.json"; write_json_atomic(cand,cert)
        eligible_obj=prepare_eligible_parent_certificate(cert); prepared=chain_dir/"workspace"/f"L{level}-ELIGIBLE_AUTO_PARENT.prepared"; prepared.parent.mkdir(parents=True,exist_ok=True); write_json_atomic(prepared,eligible_obj)
        capsule=chain_dir/"parent_capsules"/f"L{level}-forward-parent.zip"; capsule.parent.mkdir(parents=True,exist_ok=True); create_minimal_parent_capsule(destination_zip=capsule,certificate_path=prepared,selected_relation_path=ldir/"SELECTED_RELATION.json",normalized_baseline_path=nb)
        cv=verify_minimal_parent_capsule(capsule)
        if cv["status"]!="PASS": raise RuntimeError("I_RELEASE_OR_CATALOG_MISMATCH")
        eligible=parents/f"L{level}-ELIGIBLE_AUTO_PARENT.json"; committed=commit_prepared_parent_certificate(eligible=eligible_obj,eligible_path=eligible,current_pointer_path=chain_dir/"current_parent.json"); prepared.unlink(missing_ok=True)
        write_chain_event(chain_dir/"events.jsonl",event="PARENT_COMMITTED",chain_plan_sha256=chain_plan["chain_plan_sha256"],current_parent_certificate_sha256=committed["certificate_sha256"],state=state,level=level,level_wall_seconds=level_wall,chain_wall_seconds=chain_wall)
        return {"status":state,"level":level,"run_id":run["run_id"],"parent_certificate_sha256":committed["certificate_sha256"],"parent_capsule_sha256":cv["sha256"]}

    def start(self, chain_plan: dict[str, Any]) -> dict[str, Any]:
        validate_chain_plan(chain_plan)
        cdir=self.chain_dir(chain_plan["chain_id"])
        self.install_execution_handoff(chain_plan,cdir)
        self.assert_execution_enabled()
        results=[]
        while True:
            result=self.execute_one_level(chain_plan,cdir); results.append(result)
            if result["status"]!="AUTO_ADVANCE_AUTHORIZED":
                return {"status":result["status"],"chain_id":chain_plan["chain_id"],"levels":results}
