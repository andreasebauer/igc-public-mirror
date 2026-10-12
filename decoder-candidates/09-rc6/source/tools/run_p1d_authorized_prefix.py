#!/usr/bin/env python3
"""Qualify the authorized replay prefix through G8 without overstating evidence.

Empirical executors are rerun where the assertion map names one. Design,
theorem, documentary, and historical-only mappings receive source-integrity
qualification only; they are never relabelled as fresh empirical results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


G_LAYERS = {f"G{i}" for i in range(1, 9)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def layer_of(assertion_id: str) -> str:
    return assertion_id.split("/", 3)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--formal-checkpoint", type=Path, required=True)
    ap.add_argument("--l2-checkpoint", type=Path, required=True)
    ap.add_argument("--o-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dag_path = source_root / "infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json"
    dag = load(dag_path)
    qualifications = {
        row["qualification_id"]: row for row in dag.get("known_replay_qualifications", [])
    }

    upstream = {
        "formal_foundation": {"path": str(args.formal_checkpoint.resolve()), "payload": load(args.formal_checkpoint.resolve())},
        "finite_l2": {"path": str(args.l2_checkpoint.resolve()), "payload": load(args.l2_checkpoint.resolve())},
        "o1_o7": {"path": str(args.o_checkpoint.resolve()), "payload": load(args.o_checkpoint.resolve())},
    }
    upstream_records = []
    for name, item in upstream.items():
        p = Path(item["path"])
        payload = item.pop("payload")
        status = payload.get("status")
        complete = status == "PASS"
        upstream_records.append({
            "checkpoint_id": name,
            "path": str(p),
            "sha256": sha256(p),
            "status": "PASS" if complete else "FAIL",
            "reported_status": status,
        })
        if not complete:
            result = {
                "schema_id": "IG_P1D_AUTHORIZED_PREFIX_REPLAY_RESULT_V4",
                "status": "STOPPED_UPSTREAM_CHECKPOINT",
                "failed_checkpoint": name,
                "upstream_checkpoints": upstream_records,
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return 2

    mappings = [m for m in dag["historical_assertion_mappings"] if layer_of(m["assertion_id"]) in G_LAYERS]
    source_records: list[dict[str, Any]] = []
    source_failed = False
    for mapping in sorted(mappings, key=lambda x: x["assertion_id"]):
        pins = []
        for pin in mapping.get("source_hashes", []):
            path = source_root / pin["ref"]
            observed = sha256(path) if path.is_file() else None
            pins.append({
                "ref": pin["ref"],
                "expected_sha256": pin["sha256"],
                "observed_sha256": observed,
                "status": "PASS" if observed == pin["sha256"] else "FAIL",
            })
        locator = mapping["locator"]
        locator_path = source_root / locator["source_ref"]
        record = {
            "assertion_id": mapping["assertion_id"],
            "layer": layer_of(mapping["assertion_id"]),
            "evidence_mode": mapping["evidence_mode"],
            "qualification": (
                "FRESH_RECOMPUTATION_QUALIFIED" if mapping["evidence_mode"] == "INDEPENDENT_RECOMPUTATION" and mapping.get("known_qualification_ids")
                else "FRESH_RECOMPUTATION" if mapping["evidence_mode"] == "INDEPENDENT_RECOMPUTATION"
                else "SOURCE_INTEGRITY_ONLY"
            ),
            "known_qualification_ids": mapping.get("known_qualification_ids", []),
            "known_qualifications": [
                qualifications[qid]["required_result_language"]
                for qid in mapping.get("known_qualification_ids", [])
            ],
            "execution_class": mapping["execution_class"],
            "counts_toward_empty_root_science_replay": mapping[
                "counts_toward_empty_root_science_replay"
            ],
            "locator": locator,
            "locator_present": locator_path.is_file(),
            "source_pins": pins,
        }
        record["status"] = "PASS" if record["locator_present"] and pins and all(p["status"] == "PASS" for p in pins) else "FAIL"
        if record["status"] != "PASS":
            record["comparison_result"] = "NOT_REPRODUCED"
        elif record["known_qualifications"]:
            record["comparison_result"] = record["known_qualifications"][0]
        else:
            record["comparison_result"] = "REPRODUCED"
        source_records.append(record)
        if record["status"] != "PASS":
            source_failed = True
            break

    test_records: list[dict[str, Any]] = []
    if not source_failed:
        independent = [m for m in mappings if m["evidence_mode"] == "INDEPENDENT_RECOMPUTATION"]
        test_paths = sorted({m["locator"]["source_ref"] for m in independent})
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *test_paths],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        test_records.append({
            "layer": "G6",
            "mode": "FRESH_RECOMPUTE",
            "independent_confirmation": False,
            "test_paths": test_paths,
            "test_file_sha256": {p: sha256(source_root / p) for p in test_paths},
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - t0, 3),
            "stdout": proc.stdout,
            "status": "PASS" if proc.returncode == 0 else "FAIL",
        })

    tests_pass = bool(test_records) and all(r["status"] == "PASS" for r in test_records)
    status = (
        "QUALIFIED_EVIDENCE_REPRODUCTION_BUDGET_INCOMPLETE"
        if not source_failed and tests_pass else "STOPPED"
    )
    layer_qualifications = {}
    for layer in [f"G{i}" for i in range(1, 9)]:
        rows = [r for r in source_records if r["layer"] == layer]
        has_fresh = any(r["evidence_mode"] == "INDEPENDENT_RECOMPUTATION" for r in rows)
        layer_qualifications[layer] = {
            "status": "PASS" if rows and all(r["status"] == "PASS" for r in rows) and (not has_fresh or tests_pass) else "FAIL",
            "assertion_mappings": len(rows),
            "fresh_recomputation": has_fresh,
            "independent_confirmation": False,
            "source_integrity_only_assertions": sum(r["qualification"] == "SOURCE_INTEGRITY_ONLY" for r in rows),
        }

    result = {
        "schema_id": "IG_P1D_AUTHORIZED_PREFIX_REPLAY_RESULT_V4",
        "status": status,
        "fail_closed": True,
        "dag_path": str(dag_path),
        "dag_sha256": sha256(dag_path),
        "dag_declared_sha256": dag.get("dag_sha256"),
        "authorized_replay_through_layer": dag.get("authorized_replay_through_layer"),
        "next_wait_layer": dag.get("next_wait_layer"),
        "science_executed": True,
        "result_language_policy": dag.get("result_language_policy", {}),
        "known_replay_qualifications": list(qualifications.values()),
        "execution_classification_policy": dag.get("execution_classification_policy", {}),
        "empty_root_science_coverage": dag.get("empty_root_science_coverage", {}),
        "cost_budget_policy": dag.get("cost_budget_policy", {}),
        "cost_budget_programme_summary": dag.get("cost_budget_programme_summary", {}),
        "qualification_note": "REPRODUCED means agreement under the declared evidence mode and scope, not independent confirmation. Fresh computation is claimed only for the formal, finite-L2, O-target, and named G6 executable checks. Other G1-G8 assertions are source-integrity qualifications.",
        "upstream_checkpoints": upstream_records,
        "g_layer_qualifications": layer_qualifications,
        "source_records": source_records,
        "independent_test_records": test_records,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_id": result["schema_id"],
        "status": status,
        "authorized_replay_through_layer": result["authorized_replay_through_layer"],
        "next_wait_layer": result["next_wait_layer"],
        "source_assertions": len(source_records),
        "g6_test_status": test_records[0]["status"] if test_records else "NOT_RUN",
    }, indent=2))
    return 0 if status == "QUALIFIED_EVIDENCE_REPRODUCTION_BUDGET_INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
