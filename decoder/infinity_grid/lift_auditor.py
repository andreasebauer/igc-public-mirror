from __future__ import annotations

"""Independent Lift auditor.

A certified Plateau is necessary but never sufficient for a Lift.  LiftAuditor requires an
explicit LiftCandidate, evidence for every declared required audit, and a fully frozen
certification payload describing the next EntityClass boundary/action semantics.  It never
mutates ResearchFrontier automatically.
"""

from typing import Any, Iterable, Mapping

from .canon import canonical_sha256
from .scientific_architecture import (
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    validate_scientific_artifact,
)


class LiftAuditError(ScientificArchitectureError):
    pass


_INDEPENDENT_CLASSES = {
    "INDEPENDENT_EXACT_ORACLE",
    "INDEPENDENT_IMPLEMENTATION_REPLAY",
    "FORMAL_ARGUMENT",
}


def _is_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        u = value.upper()
        return any(tok in u for tok in ("TO_BE_", "TEMPLATE", "PLACEHOLDER", "DO_NOT_TREAT_AS_EARNED"))
    if isinstance(value, Mapping):
        return any(_is_placeholder(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_is_placeholder(v) for v in value)
    return False


def _validate_lift_evidence(
    ev: Mapping[str, Any],
    *,
    requirement: str,
    allow_synthetic: bool,
) -> tuple[bool, str]:
    try:
        validate_scientific_artifact(ev)
    except Exception as exc:
        return False, f"invalid evidence artifact: {type(exc).__name__}: {exc}"
    if ev.get("audit_kind") != "LIFT_REQUIREMENT":
        return False, "audit_kind must be LIFT_REQUIREMENT"
    if ev.get("requirement_ref") != requirement:
        return False, "requirement_ref mismatch"
    if ev.get("status") != "PASS":
        return False, "evidence status is not PASS"
    observed_sha = canonical_sha256({k: v for k, v in ev.items() if k != "evidence_sha256"})
    if ev.get("evidence_sha256") != observed_sha:
        return False, "evidence_sha256 does not bind the evidence payload"
    cls = ev.get("independence_class")
    if cls == "TEST_ONLY_SYNTHETIC" and allow_synthetic:
        return True, "synthetic evidence admitted only by explicit software-test override"
    if cls not in _INDEPENDENT_CLASSES:
        return False, f"independence_class {cls!r} is insufficient for Lift certification"
    return True, f"accepted {cls} evidence"


def audit_lift(
    candidate: Mapping[str, Any] | None,
    *,
    plateau_certificates: Iterable[Mapping[str, Any]] = (),
    audit_evidence: Iterable[Mapping[str, Any]] = (),
    certification_input: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    registry = registry or ScientificProtocolRegistry()
    if candidate is None:
        result = {
            "schema_id": "IG_LIFT_AUDIT_RESULT_V1",
            "schema_version": "1.0.0",
            "audit_id": canonical_sha256({"candidate": None}),
            "status": "PASS",
            "decision": "NOT_READY",
            "lift_candidate_id": None,
            "checks": [{"check_id": "LIFT_CANDIDATE_PRESENT", "status": "FAIL", "detail": "No LiftCandidate supplied."}],
            "evidence_refs": [],
            "lift_certificate": None,
            "creates_entity_class": False,
            "research_frontier_mutated": False,
            "new_scientific_claim": False,
            "nonclaim": "NO_LIFT_CANDIDATE_NO_LIFT",
        }
        result["science_sha256"] = canonical_sha256(result)
        validate_scientific_artifact(result)
        return result

    candidate = dict(candidate)
    checks: list[dict[str, Any]] = []
    try:
        validate_scientific_artifact(candidate)
        candidate_schema_ok = candidate.get("status") in {"PROPOSED", "UNDER_AUDIT", "CERTIFIED"}
    except Exception as exc:
        candidate_schema_ok = False
        checks.append({"check_id": "CANDIDATE_SCHEMA", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})
    else:
        checks.append({"check_id": "CANDIDATE_SCHEMA", "status": "PASS" if candidate_schema_ok else "FAIL", "detail": "LiftCandidate schema/status accepted." if candidate_schema_ok else "LiftCandidate status not auditable."})

    placeholder_free = not _is_placeholder({
        "proposed_entity_class": candidate.get("proposed_entity_class"),
        "proposed_action_language_ref": candidate.get("proposed_action_language_ref"),
    })
    checks.append({"check_id": "NO_PLACEHOLDERS", "status": "PASS" if placeholder_free else "FAIL", "detail": "Proposed EntityClass/action language are concrete." if placeholder_free else "LiftCandidate still contains template/placeholders."})

    proposed_caps = set(candidate.get("proposed_capabilities", []))
    caps_known = proposed_caps.issubset(registry.capability_ids)
    checks.append({"check_id": "CAPABILITIES_KNOWN", "status": "PASS" if caps_known else "FAIL", "detail": "All proposed capabilities are in the frozen vocabulary." if caps_known else f"Unknown capabilities: {sorted(proposed_caps - registry.capability_ids)}"})

    certs = []
    for cert in plateau_certificates:
        try:
            validate_scientific_artifact(cert)
        except Exception:
            continue
        if cert.get("status") == "CERTIFIED_IN_DECLARED_SCOPE":
            certs.append(dict(cert))
    needed_refs = set(candidate.get("plateau_certificate_refs", []))
    present_refs = {str(c.get("plateau_id")) for c in certs}
    plateau_ok = bool(needed_refs) and needed_refs.issubset(present_refs)
    checks.append({"check_id": "CERTIFIED_PLATEAU_PRESENT", "status": "PASS" if plateau_ok else "FAIL", "detail": "All referenced Plateau certificates are certified and supplied." if plateau_ok else f"Missing certified Plateau refs: {sorted(needed_refs - present_refs)}"})

    source_match = all(
        c.get("entity_class_ref") == candidate.get("source_entity_class_ref")
        and c.get("regime_ref") == candidate.get("source_regime_ref")
        for c in certs if c.get("plateau_id") in needed_refs
    ) if plateau_ok else False
    checks.append({"check_id": "PLATEAU_SOURCE_MATCH", "status": "PASS" if source_match else "FAIL", "detail": "Plateau source EntityClass/Regime match LiftCandidate." if source_match else "Plateau source does not match LiftCandidate source."})

    evidence = list(audit_evidence)
    accepted_refs: list[str] = []
    all_requirements = True
    for requirement in candidate.get("required_audits", []):
        matches = []
        details = []
        for ev in evidence:
            ok, detail = _validate_lift_evidence(ev, requirement=requirement, allow_synthetic=allow_synthetic)
            if ev.get("requirement_ref") == requirement:
                details.append(detail)
            if ok:
                matches.append(ev)
        passed = bool(matches)
        all_requirements = all_requirements and passed
        accepted_refs.extend(str(x["evidence_id"]) for x in matches)
        checks.append({
            "check_id": "AUDIT_REQUIREMENT:" + requirement,
            "status": "PASS" if passed else "FAIL",
            "detail": "; ".join(details) if details else "No qualifying evidence supplied.",
        })

    input_ok = False
    if certification_input is not None:
        try:
            validate_scientific_artifact(certification_input)
            input_ok = not _is_placeholder(certification_input)
        except Exception as exc:
            checks.append({"check_id": "CERTIFICATION_INPUT", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})
        else:
            checks.append({"check_id": "CERTIFICATION_INPUT", "status": "PASS" if input_ok else "FAIL", "detail": "Frozen Lift certification payload is complete." if input_ok else "Certification payload contains placeholders."})
    else:
        checks.append({"check_id": "CERTIFICATION_INPUT", "status": "FAIL", "detail": "No frozen Lift certification payload supplied."})

    all_pass = candidate_schema_ok and placeholder_free and caps_known and plateau_ok and source_match and all_requirements and input_ok and all(x["status"] == "PASS" for x in checks)
    certificate = None
    if all_pass:
        ci = dict(certification_input)
        cert = {
            "schema_id": "IG_LIFT_CERTIFICATE_V1",
            "schema_version": "1.0.0",
            "lift_certificate_id": canonical_sha256({
                "candidate": candidate,
                "plateaus": sorted(needed_refs),
                "evidence_refs": sorted(set(accepted_refs)),
                "certification_input": ci,
            }),
            "status": "CERTIFIED_IN_DECLARED_SCOPE",
            "lift_candidate_ref": candidate["lift_candidate_id"],
            "certificate_scope": ci["certificate_scope"],
            "creates_entity_class": True,
            "sufficiency_and_closure": list(ci["sufficiency_and_closure"]),
            "representative_independence": ci["representative_independence"],
            "observer_action_multiplicity_scope": ci["observer_action_multiplicity_scope"],
            "canonicalization_rule": ci["canonicalization_rule"],
            "retained_relations_and_fields": list(ci["retained_relations_and_fields"]),
            "authority_refs": list(ci["authority_refs"]),
            "premise_set_ref": ci["premise_set_ref"],
            "reopen_conditions": list(ci["reopen_conditions"]),
            "limitations_and_nonclaims": list(ci["limitations_and_nonclaims"]),
        }
        validate_scientific_artifact(cert)
        certificate = cert

    result = {
        "schema_id": "IG_LIFT_AUDIT_RESULT_V1",
        "schema_version": "1.0.0",
        "audit_id": canonical_sha256({"candidate": candidate, "checks": checks, "evidence_refs": sorted(set(accepted_refs))}),
        "status": "PASS",
        "decision": "CERTIFIED_IN_DECLARED_SCOPE" if certificate else "NOT_READY",
        "lift_candidate_id": candidate.get("lift_candidate_id"),
        "checks": checks,
        "evidence_refs": sorted(set(accepted_refs)),
        "lift_certificate": certificate,
        "creates_entity_class": bool(certificate),
        "research_frontier_mutated": False,
        "new_scientific_claim": bool(certificate) and not allow_synthetic,
        "nonclaim": "PLATEAU_IS_NECESSARY_NOT_SUFFICIENT_AND_LIFT_NEVER_AUTO_PROMOTES_FRONTIER",
    }
    result["science_sha256"] = canonical_sha256(result)
    validate_scientific_artifact(result)
    return result
