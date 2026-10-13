from __future__ import annotations

from typing import Any

from .canon import canonical_sha256


COMPARISON_RESULT_SCHEMA_V14 = "IG_DECODER_V05_OLD_NEW_COMPARISON_RESULT_V1_4"
COMPARISON_RESULT_SCHEMA = "IG_DECODER_V05_OLD_NEW_COMPARISON_RESULT_V1_5"


def science_projection(result: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        if field == "compatible_paths":
            out[field] = result.get("diagnostics", {}).get("internally_compatible_three_event_paths")
        else:
            out[field] = result.get(field)
    return out


def evaluate_comparison_contract(*, contract: dict[str, Any], baseline_result: dict[str, Any], candidate_result: dict[str, Any], contract_version: str = "1.4.0") -> dict[str, Any]:
    fields = list(contract["projection_fields"])
    baseline = science_projection(baseline_result, fields)
    candidate = science_projection(candidate_result, fields)
    exact_match = baseline == candidate
    role = contract["role"]
    witness = None
    if role in {"POSITIVE", "RECOVERY"}:
        disposition = "MATCH" if exact_match else "MISMATCH"
        status = "PASS" if disposition == contract["expected_disposition"] else "FAIL"
    elif role == "EXPECTED_FALSIFICATION":
        pred = contract["candidate_predicate"]
        observed = candidate.get(pred["field"])
        survived = observed == pred["equals"]
        disposition = "SURVIVED_UNEXPECTEDLY" if survived else "FALSIFIED_AS_EXPECTED"
        witness = {"field": pred["field"], "candidate_equals": pred["equals"], "observed": observed}
        status = "PASS" if disposition == contract["expected_disposition"] and exact_match else "FAIL"
    elif role == "REVIEW_REQUIRED" and contract_version == "1.5.0":
        disposition = "REVIEW_REQUIRED" if exact_match else "MISMATCH"
        witness = {"review_reason": contract.get("review_reason")}
        status = "PASS" if exact_match and disposition == contract["expected_disposition"] else "FAIL"
    else:
        disposition = "UNSUPPORTED_ROLE"
        status = "FAIL"
    base = {
        "schema_id": COMPARISON_RESULT_SCHEMA if contract_version == "1.5.0" else COMPARISON_RESULT_SCHEMA_V14,
        "comparison_id": contract["comparison_id"],
        "role": role,
        "status": status,
        "disposition": disposition,
        "exact_old_new_projection_match": exact_match,
        "baseline_projection": baseline,
        "candidate_projection": candidate,
        "witness": witness,
        "authority_effect": "NONE_P2_6" if contract_version == "1.5.0" else "NONE_P2_5",
    }
    return dict(base, comparison_sha256=canonical_sha256(base))
