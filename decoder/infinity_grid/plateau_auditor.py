from __future__ import annotations

"""Independent Plateau certification auditor.

MaturationAuditor may recognize a bounded Plateau *candidate*.  PlateauAuditor is a
separate gate: it replays the candidate against the completed level records, requires an
independent confirmation authority, and only then may emit a bounded observer-relative
PlateauCertificate.  It never performs a Lift and never mutates ResearchFrontier.
"""

from typing import Any, Iterable, Mapping

from .canon import canonical_sha256
from .scientific_architecture import (
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    validate_scientific_artifact,
)
from .maturation_auditor import LevelTestRecord, audit_maturation


class PlateauAuditError(ScientificArchitectureError):
    pass


_INDEPENDENT_CLASSES = {
    "INDEPENDENT_EXACT_ORACLE",
    "INDEPENDENT_IMPLEMENTATION_REPLAY",
    "FORMAL_ARGUMENT",
}


def _check_record(checks: list[dict[str, Any]], check_id: str, passed: bool, detail: str) -> None:
    checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})


def _validate_audit_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_requirement: str,
    allow_synthetic: bool,
) -> tuple[bool, str]:
    try:
        validate_scientific_artifact(evidence)
    except Exception as exc:
        return False, f"invalid evidence artifact: {type(exc).__name__}: {exc}"
    if evidence.get("audit_kind") != expected_kind:
        return False, f"audit_kind must be {expected_kind}"
    if evidence.get("requirement_ref") != expected_requirement:
        return False, f"requirement_ref must be {expected_requirement}"
    if evidence.get("status") != "PASS":
        return False, "evidence status is not PASS"
    observed_sha = canonical_sha256({k: v for k, v in evidence.items() if k != "evidence_sha256"})
    if evidence.get("evidence_sha256") != observed_sha:
        return False, "evidence_sha256 does not bind the evidence payload"
    cls = evidence.get("independence_class")
    if cls == "TEST_ONLY_SYNTHETIC" and allow_synthetic:
        return True, "synthetic evidence admitted only by explicit software-test override"
    if cls not in _INDEPENDENT_CLASSES:
        return False, f"independence_class {cls!r} is not independent enough for Plateau certification"
    return True, f"independent confirmation accepted: {cls}"


def audit_plateau(
    levels: Iterable[LevelTestRecord],
    *,
    maturation_audit: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
    independent_evidence: Iterable[Mapping[str, Any]] = (),
    test_pack_ref: str | None = None,
    plateau_window: int | None = None,
    registry: ScientificProtocolRegistry | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Audit a MaturationAuditor Plateau candidate without changing scientific authority.

    Certification is bounded to the frozen EntityClass/Regime/TestPack/depth window.  An
    independently supplied PASS authority is mandatory.  The same-provider recognition run
    can never certify itself.
    """
    registry = registry or ScientificProtocolRegistry()
    rows = sorted(list(levels), key=lambda x: x.depth)
    if not rows:
        raise PlateauAuditError("Plateau audit requires completed level records")
    if maturation_audit.get("schema_id") != "IG_MATURATION_AUDIT_RESULT_V1":
        raise PlateauAuditError("Plateau audit requires IG_MATURATION_AUDIT_RESULT_V1")
    if maturation_audit.get("status") != "PASS":
        raise PlateauAuditError("Maturation audit is not PASS")

    candidate = dict(candidate or maturation_audit.get("plateau_candidate") or {}) or None
    checks: list[dict[str, Any]] = []
    if candidate is None:
        result = {
            "schema_id": "IG_PLATEAU_AUDIT_RESULT_V1",
            "schema_version": "1.0.0",
            "audit_id": canonical_sha256({"depths": [x.depth for x in rows], "candidate": None}),
            "status": "PASS",
            "decision": "NOT_CERTIFIED",
            "candidate_id": None,
            "checks": [{"check_id": "CANDIDATE_PRESENT", "status": "FAIL", "detail": "MaturationAuditor emitted no Plateau candidate."}],
            "independent_evidence_refs": [],
            "plateau_certificate": None,
            "research_frontier_mutated": False,
            "lift_authorized": False,
            "new_scientific_claim": False,
            "nonclaim": "NO_PLATEAU_CANDIDATE_NO_CERTIFICATION",
        }
        result["science_sha256"] = canonical_sha256(result)
        validate_scientific_artifact(result)
        return result

    try:
        validate_scientific_artifact(candidate)
        valid_candidate = candidate.get("status") == "CANDIDATE"
    except Exception as exc:
        valid_candidate = False
        _check_record(checks, "CANDIDATE_SCHEMA", False, f"{type(exc).__name__}: {exc}")
    else:
        _check_record(checks, "CANDIDATE_SCHEMA", valid_candidate, "Candidate schema valid and status=CANDIDATE." if valid_candidate else "Candidate status is not CANDIDATE.")

    pack_ref = test_pack_ref or (candidate.get("observer_or_test_pack_ref") if candidate else None)
    if not isinstance(pack_ref, str) or not pack_ref:
        raise PlateauAuditError("Plateau candidate/test_pack_ref is missing")
    window = plateau_window or int(candidate["depth_window"]["width"])

    replay = audit_maturation(rows, test_pack_ref=pack_ref, plateau_window=window, registry=registry)
    replay_candidate = replay.get("plateau_candidate")
    same_candidate = replay_candidate is not None and canonical_sha256(replay_candidate) == canonical_sha256(candidate)
    _check_record(checks, "CANDIDATE_REPLAY", same_candidate, "Independent audit replay reproduces the candidate byte-semantically." if same_candidate else "Candidate does not match a fresh maturation replay.")

    supplied_maturation = maturation_audit.get("maturation_record")
    replay_maturation = replay.get("maturation_record")
    maturation_match = canonical_sha256(supplied_maturation) == canonical_sha256(replay_maturation)
    _check_record(checks, "MATURATION_RECORD_REPLAY", maturation_match, "MaturationRecord reproduced exactly." if maturation_match else "MaturationRecord replay mismatch.")

    unresolved = list((supplied_maturation or {}).get("unresolved_findings", []))
    _check_record(checks, "NO_UNRESOLVED_FINDINGS", not unresolved, "No unresolved findings in declared maturation scope." if not unresolved else f"Unresolved findings remain: {unresolved}")

    grammar_ok = (supplied_maturation or {}).get("grammar_status") == "STABLE_IN_DECLARED_TEST_SCOPE"
    _check_record(checks, "GRAMMAR_STABLE", grammar_ok, "Grammar stable in declared TestPack scope." if grammar_ok else "Grammar status is not stable in declared TestPack scope.")

    start = int(candidate["depth_window"]["start"])
    end = int(candidate["depth_window"]["end"])
    win = [x for x in rows if start <= x.depth <= end]
    window_complete = len(win) == window and [x.depth for x in win] == list(range(start, end + 1))
    _check_record(checks, "WINDOW_COMPLETE", window_complete, f"Complete consecutive window O{start}..O{end}." if window_complete else "Plateau window is incomplete or non-consecutive.")
    seam_free = bool(win) and not any(x.provider_seam for x in win[1:]) and len({x.provider_ref for x in win}) == 1
    _check_record(checks, "NO_PROVIDER_SEAM_IN_WINDOW", seam_free, "Single provider across certification window." if seam_free else "Provider seam or mixed providers inside certification window.")
    signature_stable = bool(win) and len({x.stabilization_signature_sha256 for x in win}) == 1
    _check_record(checks, "STABILIZATION_PERSISTS", signature_stable, "Normalized TestPack signature persists across the full window." if signature_stable else "Stabilization signature changes inside window.")
    complexity_ok = bool(win) and all(x.complexity_growth_from_previous is True for x in win[1:])
    _check_record(checks, "COMPLEXITY_GROWTH_GUARD", complexity_ok, "Complexity guard passes at every step in the window." if complexity_ok else "Complexity growth guard does not pass at every step.")

    evidence_refs: list[str] = []
    independent_ok = False
    independent_details: list[str] = []
    accepted_evidence: list[Mapping[str, Any]] = []
    for ev in independent_evidence:
        ok, detail = _validate_audit_evidence(
            ev,
            expected_kind="PLATEAU_CONFIRMATION",
            expected_requirement="INDEPENDENT_PLATEAU_CONFIRMATION",
            allow_synthetic=allow_synthetic,
        )
        independent_details.append(detail)
        if ok:
            independent_ok = True
            accepted_evidence.append(ev)
            evidence_refs.append(str(ev["evidence_id"]))
    _check_record(checks, "INDEPENDENT_CONFIRMATION", independent_ok, "; ".join(independent_details) if independent_details else "No independent Plateau confirmation supplied.")

    all_pass = all(x["status"] == "PASS" for x in checks)
    certificate = None
    if all_pass:
        cert = dict(candidate)
        cert["status"] = "CERTIFIED_IN_DECLARED_SCOPE"
        cert["plateau_id"] = canonical_sha256({
            "candidate_id": candidate["plateau_id"],
            "candidate_sha256": canonical_sha256(candidate),
            "independent_evidence": [canonical_sha256(dict(x)) for x in accepted_evidence],
        })
        cert["evidence_refs"] = list(candidate.get("evidence_refs", [])) + [
            {
                "id": str(ev["evidence_id"]),
                "version": str(ev["schema_version"]),
                "role": "independent plateau confirmation",
                "scope": str(ev.get("scope", "declared plateau window")),
                "sha256": str(ev["evidence_sha256"]),
            }
            for ev in accepted_evidence
        ]
        cert["reopen_conditions"] = list(dict.fromkeys(list(cert.get("reopen_conditions", [])) + [
            "independent confirmation authority is withdrawn or invalidated",
        ]))
        validate_scientific_artifact(cert)
        certificate = cert

    result = {
        "schema_id": "IG_PLATEAU_AUDIT_RESULT_V1",
        "schema_version": "1.0.0",
        "audit_id": canonical_sha256({"candidate": candidate, "checks": checks, "evidence_refs": evidence_refs}),
        "status": "PASS",
        "decision": "CERTIFIED_IN_DECLARED_SCOPE" if certificate else "NOT_CERTIFIED",
        "candidate_id": candidate.get("plateau_id"),
        "checks": checks,
        "independent_evidence_refs": evidence_refs,
        "plateau_certificate": certificate,
        "research_frontier_mutated": False,
        "lift_authorized": False,
        "new_scientific_claim": bool(certificate) and not allow_synthetic,
        "nonclaim": "PLATEAU_CERTIFICATION_IS_BOUNDED_OBSERVER_RELATIVE_AND_NEVER_AUTOMATIC_LIFT",
    }
    result["science_sha256"] = canonical_sha256(result)
    validate_scientific_artifact(result)
    return result
