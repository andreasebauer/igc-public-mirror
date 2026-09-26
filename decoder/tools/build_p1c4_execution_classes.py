from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from infinity_grid.canon import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V7 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V7.json"
SOURCE_V8 = ROOT / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V8.json"
LEDGER = ROOT / "infinity_grid/resources/replay/P1C4_EXECUTION_CLASS_LEDGER_V1.json"

MODE_TO_CLASS = {
    "INDEPENDENT_RECOMPUTATION": "FRESH_RECOMPUTE",
    "DIRECT_TARGET_REPLAY": "FRESH_RECOMPUTE",
    "EMBEDDED_EVIDENCE_CHECK": "VERIFIED_RESTORED_BLOCK",
    "EMBEDDED_GATE": "VERIFIED_RESTORED_BLOCK",
    "THEOREM": "SOURCE_INTEGRITY_ONLY",
    "DESIGN_SPECIFICATION": "SOURCE_INTEGRITY_ONLY",
    "DOCUMENTARY_CLOSEOUT": "SOURCE_INTEGRITY_ONLY",
    "HISTORICAL_RESULT_ONLY": "HISTORICAL_RESULT_ONLY",
}
CLASS_RANK = {
    "FRESH_RECOMPUTE": 0,
    "VERIFIED_RESTORED_BLOCK": 1,
    "SOURCE_INTEGRITY_ONLY": 2,
    "HISTORICAL_RESULT_ONLY": 3,
}
CLASS_REASON = {
    "FRESH_RECOMPUTE": "The declared target is executed again from registered code and permitted inputs.",
    "VERIFIED_RESTORED_BLOCK": "A previously produced block is restored or embedded and verified; its science is not regenerated from an empty generated-data root.",
    "SOURCE_INTEGRITY_ONLY": "Only pinned theorem, design or documentary source bytes and declarations are checked.",
    "HISTORICAL_RESULT_ONLY": "A frozen historical result/disposition is compared or preserved without regenerating its science.",
}


def pin(ref: str) -> dict[str, str]:
    return {"ref": ref, "sha256": hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()}


def main() -> None:
    source = json.loads(SOURCE_V7.read_text(encoding="utf-8"))
    mapping_by_id = {row["assertion_id"]: row for row in source["historical_assertion_mappings"]}
    assertion_rows = []
    for mapping in source["historical_assertion_mappings"]:
        execution_class = MODE_TO_CLASS[mapping["evidence_mode"]]
        mapping["execution_class"] = execution_class
        mapping["counts_toward_empty_root_science_replay"] = execution_class == "FRESH_RECOMPUTE"
        mapping["execution_class_reason"] = CLASS_REASON[execution_class]
        assertion_rows.append({
            "assertion_id": mapping["assertion_id"],
            "evidence_mode": mapping["evidence_mode"],
            "execution_class": execution_class,
            "counts_toward_empty_root_science_replay": mapping["counts_toward_empty_root_science_replay"],
        })

    node_rows = []
    for node in source["obligations"]:
        if node["node_kind"] != "SCIENTIFIC":
            continue
        classes = [mapping_by_id[assertion_id]["execution_class"]
                   for assertion_id in node.get("assertion_mapping_ids", [])]
        counts = Counter(classes)
        present = sorted(counts, key=CLASS_RANK.get)
        effective = max(present, key=CLASS_RANK.get)
        node["assertion_execution_class_counts"] = {key: counts[key] for key in present}
        node["execution_classes_present"] = present
        node["effective_execution_class"] = effective
        node["execution_class_is_mixed"] = len(present) > 1
        node["counts_toward_empty_root_science_replay"] = effective == "FRESH_RECOMPUTE"
        node_rows.append({
            "canonical_id": node["canonical_id"],
            "layer": node["layer"],
            "series": node["series"],
            "execution_classes_present": present,
            "assertion_execution_class_counts": node["assertion_execution_class_counts"],
            "effective_execution_class": effective,
            "execution_class_is_mixed": node["execution_class_is_mixed"],
            "counts_toward_empty_root_science_replay": node["counts_toward_empty_root_science_replay"],
        })

    layer_rows = []
    for layer in [row["layer"] for row in source["layers"]]:
        rows = [row for row in node_rows if row["layer"] == layer]
        counts = Counter(row["effective_execution_class"] for row in rows)
        fresh = sum(row["counts_toward_empty_root_science_replay"] for row in rows)
        layer_rows.append({
            "layer": layer,
            "scientific_nodes": len(rows),
            "effective_node_class_counts": {key: counts[key] for key in CLASS_RANK if counts[key]},
            "empty_root_counting_nodes": fresh,
            "complete_empty_root_science_layer": fresh == len(rows),
        })

    assertion_counts = Counter(row["execution_class"] for row in assertion_rows)
    node_counts = Counter(row["effective_execution_class"] for row in node_rows)
    pre_global_nodes = [row for row in node_rows if row["layer"] != "GLOBAL"]
    pre_global_fresh = sum(row["counts_toward_empty_root_science_replay"] for row in pre_global_nodes)
    policy = {
        "classes": list(CLASS_RANK),
        "evidence_mode_mapping": MODE_TO_CLASS,
        "node_effective_class_rule": "Use the least fresh/most historically dependent assertion class present in the node.",
        "empty_root_counting_rule": "A scientific node counts only when every mapped assertion is FRESH_RECOMPUTE.",
        "empty_root_definition": "The generated-data root starts empty. Source code, specifications and declared immutable input fixtures may be present; restored or historical outputs do not count as freshly generated science.",
    }
    coverage = {
        "status": "FULL_EMPTY_ROOT_REPLAY_NOT_ACHIEVED",
        "pre_global_scientific_nodes": len(pre_global_nodes),
        "pre_global_empty_root_counting_nodes": pre_global_fresh,
        "pre_global_empty_root_fraction": f"{pre_global_fresh}/{len(pre_global_nodes)}",
        "complete_pre_global_empty_root_layers": [
            row["layer"] for row in layer_rows
            if row["layer"] != "GLOBAL" and row["complete_empty_root_science_layer"]
        ],
        "interpretation": "The authorized evidence replay reaches G8, but the current graph does not represent a complete empty-root regeneration of all mapped science.",
    }
    ledger = {
        "schema_id": "IG/P1C4/EXECUTION_CLASS_LEDGER/V1",
        "version": "1.0",
        "status": "ALL_ASSERTIONS_AND_SCIENTIFIC_NODES_CLASSIFIED",
        "execution_classification_policy": policy,
        "assertion_records": assertion_rows,
        "node_records": node_rows,
        "layer_summary": layer_rows,
        "counts": {
            "assertions": len(assertion_rows),
            "assertion_execution_classes": {key: assertion_counts[key] for key in CLASS_RANK},
            "scientific_nodes": len(node_rows),
            "effective_node_execution_classes": {key: node_counts[key] for key in CLASS_RANK},
            "mixed_class_nodes": sum(row["execution_class_is_mixed"] for row in node_rows),
        },
        "empty_root_science_coverage": coverage,
    }
    ledger["ledger_hash"] = canonical_sha256(ledger)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source["version"] = "8.0"
    source["status"] = "P1C4_EXECUTION_CLASSES_BOUND_GLOBAL_PENDING"
    source["execution_classification_policy"] = policy
    source["empty_root_science_coverage"] = coverage
    source["execution_class_ledger"] = pin(str(LEDGER.relative_to(ROOT)))
    SOURCE_V8.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
