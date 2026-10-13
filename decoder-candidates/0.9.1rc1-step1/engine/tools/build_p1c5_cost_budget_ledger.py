from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from infinity_grid.canon import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V8 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V8.json"
SOURCE_V9 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C5_COST_AND_BUDGET_LEDGER_V1.json"
G6_MEASUREMENT = "qualification_evidence/P1C4_RUNNER_PROBE_RESULT_V3.json"

CURRENT_BUDGETS = {
    "FRESH_RECOMPUTE": {"wall_seconds": 1800, "memory_gib": 8, "cost_band": "UNKNOWN"},
    "VERIFIED_RESTORED_BLOCK": {"wall_seconds": 900, "memory_gib": 8, "cost_band": "LOW"},
    "SOURCE_INTEGRITY_ONLY": {"wall_seconds": 120, "memory_gib": 2, "cost_band": "TINY"},
    "HISTORICAL_RESULT_ONLY": {"wall_seconds": 120, "memory_gib": 2, "cost_band": "TINY"},
}
G6_MEASURED_NODES = {
    "IG/G6/F/ASSERTION_MAPPED_FALSIFICATIONS",
    "IG/G6/R/ASSERTION_MAPPED_RELATIONS",
}
CAP37_NODE = "IG/G8/F/ASSERTION_MAPPED_FALSIFICATIONS"


def pin(ref: str) -> dict[str, str]:
    return {"ref": ref, "sha256": hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()}


def main() -> None:
    source = json.loads(SOURCE_V8.read_text(encoding="utf-8"))
    g6_probe = json.loads((ROOT / G6_MEASUREMENT).read_text(encoding="utf-8"))
    g6_record = g6_probe["independent_test_records"][0]
    g6_elapsed = g6_record["elapsed_seconds"]
    node_rows = []

    for node in source["obligations"]:
        if node["node_kind"] != "SCIENTIFIC":
            continue
        node_id = node["canonical_id"]
        execution_class = node["effective_execution_class"]
        current = dict(CURRENT_BUDGETS[execution_class])
        current.update({
            "basis": "POLICY_CEILING_NOT_MEASURED_RUNTIME",
            "expected_outcome": "LIKELY_WITHIN_BUDGET",
        })
        if execution_class == "FRESH_RECOMPUTE":
            current["expected_outcome"] = "BUDGET_REQUIRED_BEFORE_RUN"
            current["basis"] = "NO_NODE_SPECIFIC_RUNTIME_MEASUREMENT"
        if node_id in G6_MEASURED_NODES:
            current.update({
                "cost_band": "TINY",
                "expected_outcome": "LIKELY_WITHIN_BUDGET",
                "basis": "MEASURED_SHARED_TWO_MODULE_PROBE_NOT_NODE_SEPARABLE",
                "measurement_group_id": "G6_DEV45_TWO_MODULE_PROBE",
                "measurement_group_elapsed_seconds": g6_elapsed,
                "measurement_group_tests_passed": 22,
                "measurement_evidence": [pin(G6_MEASUREMENT)],
            })

        if node["layer"] == "GLOBAL":
            full = {
                "cost_band": "NOT_APPLICABLE",
                "expected_outcome": "NOT_AUTHORIZED",
                "basis": "GLOBAL_COMPONENTS_REPLAY_NOT_AUTHORIZED",
            }
        elif node_id in G6_MEASURED_NODES:
            full = {
                "cost_band": "TINY_FOR_CURRENT_MAPPED_SUBSET",
                "expected_outcome": "LIKELY_WITHIN_BUDGET",
                "basis": "MEASURED_MAPPED_SUBSET_ONLY_NOT_FULL_HISTORICAL_G6_SCIENCE",
            }
        elif node_id == CAP37_NODE:
            full = {
                "cost_band": "EXTREME_OR_UNBOUNDED_AT_CURRENT_FRONTIER",
                "expected_outcome": "PERFORMANCE_BUDGET_EXCEEDED_EXPECTED",
                "basis": "CAP37_ZERO_DATA_INCOMPLETE_FRONTIER_AND_NO_COMPLETION_COST_MODEL",
            }
        else:
            full = {
                "cost_band": "UNKNOWN",
                "expected_outcome": "BUDGET_REQUIRED_BEFORE_RUN",
                "basis": "NO_COMPLETE_FRESH_HANDLER_AND_EMPIRICAL_COST_MODEL_BOUND_TO_THIS_NODE",
            }

        node["cost_budget"] = {
            "current_evidence_replay": current,
            "prospective_full_empty_root_recompute": full,
        }
        node_rows.append({
            "canonical_id": node_id,
            "layer": node["layer"],
            "series": node["series"],
            "effective_execution_class": execution_class,
            "current_evidence_replay": current,
            "prospective_full_empty_root_recompute": full,
        })

    priority = {
        "LIKELY_WITHIN_BUDGET": 0,
        "BUDGET_REQUIRED_BEFORE_RUN": 1,
        "PERFORMANCE_BUDGET_EXCEEDED_EXPECTED": 2,
        "NOT_AUTHORIZED": 3,
    }
    layer_rows = []
    for layer in [row["layer"] for row in source["layers"]]:
        rows = [row for row in node_rows if row["layer"] == layer]
        current_counts = Counter(row["current_evidence_replay"]["expected_outcome"] for row in rows)
        full_counts = Counter(row["prospective_full_empty_root_recompute"]["expected_outcome"] for row in rows)
        current_status = max(current_counts, key=priority.get)
        full_status = max(full_counts, key=priority.get)
        layer_rows.append({
            "layer": layer,
            "scientific_nodes": len(rows),
            "current_evidence_replay_outcomes": dict(sorted(current_counts.items())),
            "current_evidence_replay_layer_status": current_status,
            "prospective_full_empty_root_outcomes": dict(sorted(full_counts.items())),
            "prospective_full_empty_root_layer_status": full_status,
        })

    pre_global = [row for row in node_rows if row["layer"] != "GLOBAL"]
    current_counts = Counter(row["current_evidence_replay"]["expected_outcome"] for row in pre_global)
    full_counts = Counter(row["prospective_full_empty_root_recompute"]["expected_outcome"] for row in pre_global)
    summary = {
        "current_evidence_replay": {
            "status": "BUDGET_INCOMPLETE_ONE_FRESH_NODE_UNMEASURED",
            "pre_global_node_outcomes": dict(sorted(current_counts.items())),
            "total_runtime_estimate": "NOT_SUMMABLE_FROM_AVAILABLE_MEASUREMENTS",
        },
        "prospective_full_empty_root_recompute": {
            "status": "DO_NOT_LAUNCH_COST_MODEL_INCOMPLETE",
            "pre_global_node_outcomes": dict(sorted(full_counts.items())),
            "expected_performance_budget_exceeded_nodes": [CAP37_NODE],
            "total_runtime_estimate": "UNAVAILABLE",
        },
    }
    policy = {
        "purpose": "Operational ceilings and launch-readiness classifications, not fabricated runtime predictions.",
        "default_current_node_budgets": CURRENT_BUDGETS,
        "budget_exceeded_outcome": "PERFORMANCE_BUDGET_EXCEEDED",
        "unknown_cost_outcome": "BUDGET_REQUIRED_BEFORE_RUN",
        "measurement_rule": "Do not sum shared measurements or infer full-science cost from focused tests.",
        "launch_rule": "A prospective full empty-root run may not launch while any authorized pre-GLOBAL node is BUDGET_REQUIRED_BEFORE_RUN or PERFORMANCE_BUDGET_EXCEEDED_EXPECTED without an explicit external budget decision.",
    }
    ledger = {
        "schema_id": "IG/P1C5/COST_AND_BUDGET_LEDGER/V1",
        "version": "1.0",
        "status": "COST_MODEL_INCOMPLETE_FULL_EMPTY_ROOT_RUN_BLOCKED",
        "cost_budget_policy": policy,
        "measurement_evidence": [pin(G6_MEASUREMENT)],
        "node_records": node_rows,
        "layer_summary": layer_rows,
        "programme_summary": summary,
        "counts": {
            "scientific_nodes": len(node_rows),
            "pre_global_scientific_nodes": len(pre_global),
            "current_pre_global_outcomes": dict(sorted(current_counts.items())),
            "prospective_full_empty_root_pre_global_outcomes": dict(sorted(full_counts.items())),
        },
    }
    ledger["ledger_hash"] = canonical_sha256(ledger)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source["version"] = "9.0"
    source["status"] = "P1C5_COST_BUDGET_LEDGER_FULL_EMPTY_ROOT_RUN_BLOCKED_GLOBAL_PENDING"
    source["cost_budget_policy"] = policy
    source["cost_budget_programme_summary"] = summary
    source["cost_budget_ledger"] = pin(str(LEDGER.relative_to(ROOT)))
    SOURCE_V9.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
