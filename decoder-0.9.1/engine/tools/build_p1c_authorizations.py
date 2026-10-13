from __future__ import annotations

import hashlib
import json
from pathlib import Path

from infinity_grid.canon import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V4 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V4.json"
SOURCE_V5 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V5.json"
DECISION_REF = "p1c_historical_sources/EXTERNAL_AUDIT_DECISION_THROUGH_G8_2026-09-22.txt"

LAYERS = [
    "L0", "C0_HISTORICAL", "L2J3", "NODE_IN", "SCOUT_HISTORICAL",
    "O1_O3", "O4_O7", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
]

PROFILES = {
    "L0": ("THEOREM_BACKED", "Exact primitive boundary, bridge legality and closure claims only.",
           "No claim above the formal L0 observer boundary."),
    "C0_HISTORICAL": ("CONDITIONAL", "Historical local-certificate ladder and its recorded failures.",
                      "Historical C0 is retained as an alias and is not equated with later axes."),
    "L2J3": ("EXHAUSTIVE", "Frozen L2/J3 canonical and finite-context result identities.",
             "No extrapolation beyond the pinned finite and theorem-backed scopes."),
    "NODE_IN": ("THEOREM_BACKED", "Mature-node and finite inter-node grammar obligations.",
                "No geometry or unrestricted infinite grammar is claimed."),
    "SCOUT_HISTORICAL": ("BOUNDED_EMPIRICAL", "Pinned Scout2B/Scout3 audit populations and theorem statements.",
                         "Non-associativity and recorded counterexamples remain active."),
    "O1_O3": ("THEOREM_BACKED", "Reconciled O1-O3 carrier, quotient, parity, Euler and compatibility scope.",
              "No O4+ claim is inherited from this authorization."),
    "O4_O7": ("THEOREM_BACKED", "Frozen O4-O7 registry/oracle and GRRL H1-H13 instantiation scope.",
              "Historical O7.Gn labels remain O7 gates, not later G-axis layers."),
    "G1": ("CONDITIONAL", "Explicit O7-to-G1 crosswalk and frozen public-interface contract.",
           "G1 remains pre-geometric and does not authorize G2."),
    "G2": ("CONDITIONAL", "Mapped G2 structural, relational, interface and falsification contracts.",
           "Replay tests the declared contracts; design text is not promoted to new empirical evidence."),
    "G3": ("CONDITIONAL", "Mapped G3 recursive-closure and structural-algebra contracts.",
           "Only the registered historical scope is authorized."),
    "G4": ("CONDITIONAL", "Mapped G4 rebase, observer quotient and challenge scope.",
           "Blocked or negative historical outcomes remain preserved outcomes."),
    "G5": ("CONDITIONAL", "Mapped G5 closure and marker-free separation scope.",
           "No stronger predictive or geometric claim is authorized."),
    "G6": ("BOUNDED_EMPIRICAL", "Mapped G6 chaining, observer congruence and intrinsic-descriptor scope.",
           "Optimization and plateau results remain bounded by their registered fixtures."),
    "G7": ("CONDITIONAL", "Final scoped G7 closeout, typed handoff and preserved open boundaries.",
           "Conditional and nonclaim language in the closeout remains binding."),
    "G8": ("CONDITIONAL", "Only the catalogue's bounded/conditional G8 S/R/C/F claims.",
           "CAP37 32768 zero-data remains negative; no unbounded or open-frontier completion is claimed."),
}


def file_hash(ref: str) -> str:
    return hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(SOURCE_V4.read_text(encoding="utf-8"))
    mapping_by_id = {row["assertion_id"]: row for row in source["historical_assertion_mappings"]}
    obligations = {row["canonical_id"]: row for row in source["obligations"]}
    decision_pin = {"ref": DECISION_REF, "sha256": file_hash(DECISION_REF)}
    authorizations = []

    for layer in LAYERS:
        node_ids = sorted(
            node_id for node_id, node in obligations.items()
            if node["layer"] == layer and node["node_kind"] == "SCIENTIFIC"
        )
        if len(node_ids) != 4:
            raise RuntimeError(f"{layer}: expected four scientific obligations, got {node_ids}")

        source_pins = {}
        f_assertions = []
        for node_id in node_ids:
            node = obligations[node_id]
            for assertion_id in node["assertion_mapping_ids"]:
                mapping = mapping_by_id[assertion_id]
                for row in mapping["source_hashes"]:
                    prior = source_pins.setdefault(row["ref"], row["sha256"])
                    if prior != row["sha256"]:
                        raise RuntimeError(f"conflicting source hash for {row['ref']}")
                if node["series"] == "F":
                    f_assertions.append(assertion_id)

        strength, scope, limitation = PROFILES[layer]
        auth_id = f"IGA/{layer}/P1C/EXTERNAL_AUDIT_REPLAY/V1"
        auth = {
            "authorization_id": auth_id,
            "canonical_obligation_ids": node_ids,
            "historical_aliases": [f"P1C {layer} external audit decision"],
            "historical_source_hashes": [
                {"ref": ref, "sha256": source_pins[ref]} for ref in sorted(source_pins)
            ],
            "expected_result_identity_or_comparison_contract": {
                "mode": "ASSERTION_SPECIFIC_EXACT_OR_DECLARED_COMPARISON",
                "required_match": "Every bound assertion reproduces its pinned expected_outcome under its recorded evidence_mode and scope_and_bounds.",
                "on_match": "EXACT_HISTORICAL_REPLAY_AUTHORIZED",
                "on_mismatch": "RESULT_MISMATCH",
                "g8_frontier_rule": "UNAUDITED_G8_FRONTIER" if layer == "G8" else "NOT_APPLICABLE",
            },
            "carrier": f"ASSERTION_MAPPED_{layer}_SRCF_CATALOGUE",
            "equality_or_observer": "Per-obligation pinned observer and exact-or-declared comparison contract.",
            "scope_and_bounds": scope,
            "earned_algebra_statement": (
                f"External audit authorizes automatic historical replay of mapped {layer} S/R/C/F obligations within the pinned scope."
            ),
            "strength": strength,
            "preserved_falsifications": sorted(f_assertions),
            "limitations_and_nonclaims": [
                limitation,
                "No mismatch may be waived automatically and no new or expanded scientific claim is authorized.",
            ],
            "accepted_equivalence_certificates": [],
            "authorized_downstream_nodes": [],
            "audit_provenance": [decision_pin],
            "decision": "CONTINUE",
        }
        auth["decision_hash"] = canonical_sha256(auth)
        authorizations.append(auth)
        for node_id in node_ids:
            obligations[node_id]["audit_authorization_ids"] = [auth_id]

    source["version"] = "5.0"
    source["status"] = "P1C_HISTORICAL_AUDIT_AUTHORIZED_THROUGH_QUALIFIED_G8_GLOBAL_PENDING"
    source["historical_audit_authorizations"] = authorizations
    SOURCE_V5.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
