from __future__ import annotations

import hashlib
import json
from pathlib import Path

from infinity_grid.canon import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V5 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V5.json"
SOURCE_V6 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V6.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C2_LAYER_AUDIT_PROVENANCE_LEDGER_V1.json"
BLANKET_DECISION = "p1c_historical_sources/EXTERNAL_AUDIT_DECISION_THROUGH_G8_2026-09-22.txt"


def pin(ref: str) -> dict[str, str]:
    return {"ref": ref, "sha256": hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()}


FORMAL = [
    "p1c2_audit_records/formal/00_READ_FIRST/AUDIT_AND_CLOSEOUT_v2.2.md",
    "p1c2_audit_records/formal/00_READ_FIRST/CLOSEOUT_CERTIFICATION_v2.2.txt",
]
ALGEBRA = [
    "p1c2_audit_records/algebra/Infinity_Grid_Algebra_Final_2026-08-15/02_CORE_AUDITS/ig_rm1_L2_boundary_closure_code_audit_results.json",
    "p1c2_audit_records/algebra/Infinity_Grid_Algebra_Final_2026-08-15/02_CORE_AUDITS/ig_rm1_L2_bridge_sufficiency_code_audit_results.json",
    "p1c2_audit_records/algebra/Infinity_Grid_Algebra_Final_2026-08-15/02_CORE_AUDITS/ig_rm1_L2_compositor_congruence_code_audit_results.json",
    "p1c2_audit_records/algebra/Infinity_Grid_Algebra_Final_2026-08-15/02_CORE_AUDITS/ig_rm1_L2_output_sufficiency_code_audit_results.json",
]
NODE = [
    "p1c2_audit_records/node/NODE_GRADUATION_AUDIT_2026-08-26/results/HUMAN_SUMMARY.txt",
    "p1c2_audit_records/node/NODE_GRADUATION_AUDIT_2026-08-26/results/NODE_GRADUATION_RESULT.json",
]
SCOUT = [
    "p1c2_audit_records/scout_o3/AUDIT_RESULT.json",
    "p1c2_audit_records/scout_o3/HUMAN_REPORT.txt",
]
O3 = [
    "p1c2_audit_records/scout_o3/SCOUT3_V1_3_O3_GRADUATION_CLASSIFICATION_AUDIT_2026-08-27/results/O3_GRADUATION_AUDIT_RESULT.json",
    "p1c2_audit_records/scout_o3/SCOUT3_V1_3_O3_GRADUATION_CLASSIFICATION_AUDIT_2026-08-27/report/HUMAN_REPORT.txt",
]
O7 = [
    "p1c2_audit_records/o7/Infinity_Grid_Post_O7_Structural_Audit_v0.2_RECONCILED_2026-08-29/00_READ_FIRST/UPDATED_AUDIT_REPORT.txt",
    "p1c2_audit_records/o7/Infinity_Grid_Post_O7_Structural_Audit_v0.2_RECONCILED_2026-08-29/01_RESULTS/POST_O7_STRUCTURAL_AUDIT_RECONCILED_RESULT.json",
]
G7 = ["p1c2_audit_records/g7_g8/G7_FINAL_CLOSEOUT_REPORT_V1_2026-09-16.txt"]
G8 = ["p1c2_audit_records/g7_g8/IG_G8_INTEGRATION_CLOSEOUT_2026-09-17.txt"]


# Status is deliberately about records, not about whether the user authorized replay.
PROFILES = {
    "L0": ("RECORDED_AUDIT_AND_CLOSEOUT", FORMAL,
           "The formal-foundation closeout records a certified replay and scoped scientific verdict.", True),
    "C0_HISTORICAL": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
                      "No separate C0 layer audit decision was located in the available evidence.", False),
    "L2J3": ("RECORDED_TECHNICAL_AUDIT_NO_LAYER_AUTHORIZATION", ALGEBRA,
             "Core code-audit results are recorded; they are evidence, not a separate replay-authority decision.", False),
    "NODE_IN": ("RECORDED_LAYER_GRADUATION_AUDIT", NODE,
                "The mature-node graduation audit result and human summary are pinned.", True),
    "SCOUT_HISTORICAL": ("RECORDED_ADVERSARIAL_AUDIT", SCOUT,
                         "The bounded Scout adversarial audit result and report are pinned.", True),
    "O1_O3": ("RECORDED_O3_GRADUATION_AUDIT_PARTIAL_COMBINED_LAYER", O3,
              "O3 has a recorded graduation audit; this does not by itself document every O1/O2 decision.", False),
    "O4_O7": ("RECORDED_POST_O7_AUDIT_PARTIAL_COMBINED_LAYER", O7,
              "A reconciled post-O7 structural audit is pinned; separate O4-O6 audit decisions are not all pinned here.", False),
    "G1": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
           "No separate G1 audit decision record was located in the available evidence.", False),
    "G2": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
           "No separate G2 audit decision record was located in the available evidence.", False),
    "G3": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
           "No separate G3 audit decision record was located in the available evidence.", False),
    "G4": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
           "No separate G4 audit decision record was located in the available evidence.", False),
    "G5": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
           "No separate G5 audit decision record was located in the available evidence.", False),
    "G6": ("USER_BLANKET_REPLAY_AUTHORIZATION_ONLY", [],
           "A partial review was reported in conversation, but no exact G6 layer audit decision file is pinned.", False),
    "G7": ("RECORDED_SCOPED_CLOSEOUT_NOT_INDEPENDENT_REPLAY", G7,
           "The scoped documentary closeout is pinned and explicitly says it did not rerun the science.", False),
    "G8": ("RECORDED_CONDITIONAL_CLOSEOUT_NOT_INDEPENDENT_REPLAY", G8,
           "The conditional bounded integration closeout is pinned; intrinsic graduation remains unearned.", False),
}


def main() -> None:
    source = json.loads(SOURCE_V5.read_text(encoding="utf-8"))
    blanket_pin = pin(BLANKET_DECISION)
    entries = []
    auths = {row["authorization_id"].split("/")[1]: row
             for row in source["historical_audit_authorizations"]}

    for layer in [row["layer"] for row in source["layers"] if row["layer"] != "GLOBAL"]:
        status, refs, statement, complete = PROFILES[layer]
        entry = {
            "layer": layer,
            "replay_authority_basis": "USER_BLANKET_REPLAY_AUTHORIZATION",
            "replay_authority": [blanket_pin],
            "audit_record_status": status,
            "complete_layer_audit_recorded": complete,
            "independent_replay_recorded": False,
            "audit_evidence": [pin(ref) for ref in refs],
            "coverage_statement": statement,
            "nonclaims": [
                "Replay authorization is not proof that an independent layer audit exists.",
                "Pinned scientific sources and closeouts are not silently promoted to independent confirmation.",
            ],
        }
        entry["entry_hash"] = canonical_sha256(entry)
        entries.append(entry)

        old = auths[layer]
        new_id = f"IGA/{layer}/P1C2/REPLAY_AUTHORIZATION/V2"
        old["authorization_id"] = new_id
        old["historical_aliases"] = [f"P1C2 {layer} explicit authority and audit-record classification"]
        old["authority_basis"] = entry["replay_authority_basis"]
        old["audit_record_status"] = status
        old["audit_evidence"] = entry["audit_evidence"]
        old["complete_layer_audit_recorded"] = complete
        old["independent_replay_recorded"] = False
        old["audit_provenance"] = [blanket_pin]
        old["earned_algebra_statement"] = (
            f"The user decision authorizes replay of mapped {layer} obligations within the pinned scope; "
            f"audit-record status is {status}."
        )
        if not complete:
            note = "A complete layer-specific historical audit decision is not claimed by this authorization."
            if note not in old["limitations_and_nonclaims"]:
                old["limitations_and_nonclaims"].append(note)
        old["decision_hash"] = canonical_sha256({k: v for k, v in old.items() if k != "decision_hash"})
        for node in source["obligations"]:
            if node["layer"] == layer and node["node_kind"] == "SCIENTIFIC":
                node["audit_authorization_ids"] = [new_id]

    ledger = {
        "schema_id": "IG/P1C2/LAYER_AUDIT_PROVENANCE_LEDGER/V1",
        "version": "1.0",
        "status": "AUDIT_RECORDS_CLASSIFIED_REPLAY_AUTHORITY_SEPARATED",
        "rule": (
            "Replay authority, recorded audit evidence, scoped closeout evidence and independent replay "
            "are separate properties and must not be inferred from one another."
        ),
        "entries": entries,
        "counts": {
            "layers": len(entries),
            "complete_layer_audit_recorded": sum(row["complete_layer_audit_recorded"] for row in entries),
            "blanket_authorization_only": sum(
                row["audit_record_status"] == "USER_BLANKET_REPLAY_AUTHORIZATION_ONLY"
                for row in entries
            ),
            "independent_replay_recorded": 0,
        },
    }
    ledger["ledger_hash"] = canonical_sha256(ledger)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source["version"] = "6.0"
    source["status"] = "P1C2_AUTHORITY_AND_AUDIT_RECORDS_SEPARATED_GLOBAL_PENDING"
    source["audit_provenance_ledger"] = pin(str(LEDGER.relative_to(ROOT)))
    source["historical_audit_authorizations"] = [auths[layer] for layer in PROFILES]
    SOURCE_V6.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
