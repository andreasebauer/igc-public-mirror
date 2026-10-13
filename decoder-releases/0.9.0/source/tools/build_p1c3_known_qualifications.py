from __future__ import annotations

import hashlib
import json
from pathlib import Path

from infinity_grid.canon import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V6 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V6.json"
SOURCE_V7 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V7.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C3_KNOWN_REPLAY_QUALIFICATIONS_V1.json"
DECISION = "p1c3_qualification_records/KNOWN_REPLAY_QUALIFICATIONS_DECISION_2026-09-22.txt"


def pin(ref: str) -> dict[str, str]:
    return {"ref": ref, "sha256": hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()}


def make_qualification(
    qualification_id: str,
    category: str,
    affected_assertion_ids: list[str],
    affected_obligation_ids: list[str],
    evidence_refs: list[str],
    evidence_status: str,
    required_result_language: str,
    replay_effect: str,
    boundary: str,
) -> dict:
    row = {
        "qualification_id": qualification_id,
        "category": category,
        "status": "ACTIVE",
        "affected_assertion_ids": affected_assertion_ids,
        "affected_obligation_ids": affected_obligation_ids,
        "evidence_pins": [pin(ref) for ref in evidence_refs],
        "evidence_status": evidence_status,
        "required_result_language": required_result_language,
        "replay_effect": replay_effect,
        "boundary": boundary,
        "nonclaims": [
            "A reproduced result is not independently confirmed.",
            "This qualification does not alter frozen historical result bytes.",
        ],
    }
    row["record_hash"] = canonical_sha256(row)
    return row


def main() -> None:
    source = json.loads(SOURCE_V6.read_text(encoding="utf-8"))
    qualifications = [
        make_qualification(
            "IGKQ/G6/FEX1_S7_A30_IN_SAMPLE/V1",
            "IN_SAMPLE_SHARED_DEFECT_HISTORY",
            ["IGAM/G6/F/TEST_G6_S8_INTRINSIC_DESCRIPTOR_PY"],
            ["IG/G6/F/ASSERTION_MAPPED_FALSIFICATIONS"],
            [
                DECISION,
                "RELEASE_0.6_READ_FIRST.txt",
                "tests/test_g6_s7d2_compact_family_a30.py",
                "tests/test_g6_r0_post_graduation_fiber.py",
            ],
            "REPORTED_LINK_NOT_INDEPENDENTLY_REVERIFIED_IN_THIS_STEP",
            "QUALIFIED_REPRODUCTION_IN_SAMPLE",
            "CONTINUE_WITH_QUALIFICATION",
            "G6 S7 A30/FEX1 only; the unrelated L2/J3 oracle assertion named A30 is explicitly excluded.",
        ),
        make_qualification(
            "IGKQ/G7/RECEIPT_HASH_DISCREPANCY/V1",
            "UNRESOLVED_PROVENANCE_DISCREPANCY",
            [
                "IGAM/G7/C/HANDOFF_CONTRACT",
                "IGAM/G7/F/OPEN_NONCLAIMS",
                "IGAM/G7/R/C9C_C9F_NECESSITY",
                "IGAM/G7/S/C9A_C9B_STATE",
            ],
            [
                "IG/G7/C/ASSERTION_MAPPED_INTERFACES",
                "IG/G7/F/ASSERTION_MAPPED_FALSIFICATIONS",
                "IG/G7/R/ASSERTION_MAPPED_RELATIONS",
                "IG/G7/S/ASSERTION_MAPPED_STRUCTURE",
            ],
            [DECISION, "p1b_external_sources/G7_FINAL_CLOSEOUT_REPORT_V1_2026-09-16.txt"],
            "REPORTED_MISMATCH_EXACT_RECEIPT_FILE_PAIR_NOT_LOCATED",
            "REPRODUCED_DOCUMENTARY_CLOSEOUT_WITH_UNRESOLVED_PROVENANCE_QUALIFICATION",
            "SOURCE_INTEGRITY_MAY_CONTINUE_RECEIPT_BASED_REUSE_MUST_STOP",
            "No historical receipt may certify current bytes until its exact hashes are reconciled.",
        ),
        make_qualification(
            "IGKQ/G8/CAP37_ZERO_DATA_FRONTIER/V1",
            "NEGATIVE_INCOMPLETE_FRONTIER",
            ["IGAM/G8/F/CAP37_ZERO_DATA"],
            ["IG/G8/F/ASSERTION_MAPPED_FALSIFICATIONS"],
            [DECISION, "p1b_external_sources/G8_CAP37_32768_ZERO_DATA_HANDOFF_2026-09-20.txt"],
            "PINNED_HISTORICAL_NEGATIVE_HANDOFF",
            "REPRODUCED_NEGATIVE_INCOMPLETE_OUTCOME",
            "CONTINUE_TO_RECORDED_FRONTIER_THEN_STOP_UNAUDITED",
            "Zero-data is not CAP37 completion; crossing the recorded frontier is UNAUDITED_G8_FRONTIER.",
        ),
    ]

    ledger = {
        "schema_id": "IG/P1C3/KNOWN_REPLAY_QUALIFICATIONS/V1",
        "version": "1.0",
        "status": "ACTIVE_QUALIFICATIONS_BOUND",
        "result_language_policy": {
            "success_term": "REPRODUCED",
            "forbidden_unqualified_terms": ["CONFIRMED", "INDEPENDENTLY_CONFIRMED", "MATCH"],
            "meaning": "Agreement with pinned expected evidence under the declared evidence mode and scope.",
        },
        "qualifications": qualifications,
        "counts": {"active": 3, "blocking_all_replay": 0, "conditional_continue": 3},
    }
    ledger["ledger_hash"] = canonical_sha256(ledger)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    q_by_assertion: dict[str, list[str]] = {}
    q_by_obligation: dict[str, list[str]] = {}
    for row in qualifications:
        for assertion_id in row["affected_assertion_ids"]:
            q_by_assertion.setdefault(assertion_id, []).append(row["qualification_id"])
        for obligation_id in row["affected_obligation_ids"]:
            q_by_obligation.setdefault(obligation_id, []).append(row["qualification_id"])

    for mapping in source["historical_assertion_mappings"]:
        mapping["known_qualification_ids"] = sorted(q_by_assertion.get(mapping["assertion_id"], []))
    for node in source["obligations"]:
        node["known_qualification_ids"] = sorted(q_by_obligation.get(node["canonical_id"], []))
    for auth in source["historical_audit_authorizations"]:
        auth["expected_result_identity_or_comparison_contract"]["on_match"] = (
            "REPRODUCED_WITH_DECLARED_EVIDENCE_MODE"
        )
        auth["limitations_and_nonclaims"] = sorted(set(auth["limitations_and_nonclaims"] + [
            "REPRODUCED does not mean independently confirmed or proved correct.",
        ]))
        auth["decision_hash"] = canonical_sha256({k: v for k, v in auth.items() if k != "decision_hash"})

    source["version"] = "7.0"
    source["status"] = "P1C3_KNOWN_QUALIFICATIONS_BOUND_GLOBAL_PENDING"
    source["result_language_policy"] = ledger["result_language_policy"]
    source["known_replay_qualifications"] = qualifications
    source["known_replay_qualification_ledger"] = pin(str(LEDGER.relative_to(ROOT)))
    SOURCE_V7.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
