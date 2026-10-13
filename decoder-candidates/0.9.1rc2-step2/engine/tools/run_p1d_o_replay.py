#!/usr/bin/env python3
"""Run every registered O1--O7 target once and evaluate all 36 obligations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from infinity_grid.frontier import run_o7_topology_read_probe
from infinity_grid.jumpstart import JumpstartRuntime, _safe_extract_zip
from infinity_grid.paths import resolve_root
from infinity_grid.scientific_tests import ScientificTestStore


def pointer(obj: Any, value: str) -> Any:
    if value in ("", "/"):
        return obj
    cur = obj
    for raw in value.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        cur = cur[int(token)] if isinstance(cur, list) else cur[token]
    return cur


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_topology_read(runtime: JumpstartRuntime, target_alias: str, workspace: Path) -> dict[str, Any]:
    """Execute the registered science through the current controller seam.

    The retained launcher still calls the removed historical CLI.  The science
    implementation itself is current and registered, so prepare the immutable
    payload and call that implementation directly.
    """
    prep = runtime.prepare(target_alias, workspace=workspace, clean=True)
    work_root = Path(prep["work_root"])
    replay_zip = work_root / "replay/Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29.zip"
    extracted = work_root / "_frontier_runtime"
    shutil.rmtree(extracted, ignore_errors=True)
    extracted.mkdir()
    _safe_extract_zip(replay_zip, extracted)
    tops = [p for p in extracted.iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise RuntimeError(f"bad nested O7 replay root: {tops}")
    output = work_root / "_frontier_replay_results"
    result = run_o7_topology_read_probe(work_root / "graduation_compact", work_root / "authority", tops[0], output)
    authority = json.loads((work_root / "authority/O7_TOPOLOGY_AWARE_READ_PROBE_RESULT.json").read_text())
    matched = result.get("status") == "PASS" and result.get("science_sha256") == authority.get("science_sha256")
    return {
        "status": "COMPLETE" if matched else "FAIL_CLOSED",
        "returncode": 0 if matched else 2,
        "workspace": prep["workspace"],
        "work_root": prep["work_root"],
        "plan_sha256": prep["plan"]["plan_sha256"],
        "fresh_evidence": str(output / "O7_TOPOLOGY_AWARE_READ_PROBE_RESULT.json"),
        "fresh_science_sha256": result.get("science_sha256"),
        "authority_science_sha256": authority.get("science_sha256"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-root", type=Path, required=True)
    ap.add_argument("--workspace-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    runtime_root = args.runtime_root.resolve()
    workspace_root = args.workspace_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    workspace_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    paths = resolve_root(runtime_root)
    store = ScientificTestStore(paths)
    runtime = JumpstartRuntime(paths)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in store.list():
        groups[spec["parent_target_alias"]].append(spec)

    targets: list[dict[str, Any]] = []
    obligations: list[dict[str, Any]] = []
    started = time.time()
    stopped = False
    for target_alias in sorted(groups, key=lambda x: (groups[x][0]["level"], x)):
        specs = groups[target_alias]
        target_workspace = workspace_root / target_alias.rsplit("/", 1)[-1]
        t0 = time.time()
        current_controller_binding = target_alias == "ig:test/O7_TOPOLOGY_READ"
        try:
            run = (
                run_topology_read(runtime, target_alias, target_workspace)
                if current_controller_binding
                else runtime.launch(target_alias, workspace=target_workspace, clean=True, detach=False)
            )
        except Exception as exc:
            targets.append({
                "target_alias": target_alias,
                "status": "FAIL_EXECUTION",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - t0, 3),
            })
            stopped = True
            break
        target_record = {
            "target_alias": target_alias,
            "status": "PASS" if run.get("status") == "COMPLETE" and int(run.get("returncode", 1)) == 0 else "FAIL_EXECUTION",
            "returncode": run.get("returncode"),
            "work_root": run.get("work_root"),
            "plan_sha256": run.get("plan_sha256"),
            "elapsed_seconds": round(time.time() - t0, 3),
            "current_controller_binding": current_controller_binding,
        }
        if current_controller_binding:
            target_record.update({k: run[k] for k in ("fresh_evidence", "fresh_science_sha256", "authority_science_sha256")})
        targets.append(target_record)
        if target_record["status"] != "PASS":
            stopped = True
            break
        work_root = Path(run["work_root"])
        for spec in sorted(specs, key=lambda x: x["test_id"]):
            record: dict[str, Any] = {
                "test_id": spec["test_id"],
                "level": spec["level"],
                "parent_target_alias": target_alias,
                "replay_mode": spec["replay_mode"],
                "authority_note": spec.get("authority_note", ""),
                "replay_semantics": spec.get("replay_semantics", ""),
            }
            try:
                observed = None
                evidence_path = None
                if spec.get("evidence_relative_path"):
                    evidence_path = work_root / spec["evidence_relative_path"]
                    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    observed = pointer(evidence, spec.get("json_pointer", "/"))
                    record["evidence_relative_path"] = spec["evidence_relative_path"]
                    record["evidence_sha256"] = file_sha(evidence_path)
                    record["json_pointer"] = spec.get("json_pointer", "/")
                record["observed"] = observed
                record["expected"] = spec.get("expected_value")
                record["status"] = "PASS" if "expected_value" not in spec or observed == spec["expected_value"] else "FAIL_DISCREPANCY"
            except Exception as exc:
                record["status"] = "FAIL_EVIDENCE"
                record["error"] = f"{type(exc).__name__}: {exc}"
            obligations.append(record)
            if record["status"] != "PASS":
                stopped = True
                break
        if stopped:
            break

    complete = len(obligations) == len(store.list())
    status = "PASS" if complete and not stopped and all(x["status"] == "PASS" for x in targets + obligations) else "STOPPED"
    checkpoint = {
        "schema_id": "IG_P1D_O1_O7_FRESH_REPLAY_CHECKPOINT_V1",
        "status": status,
        "fail_closed": True,
        "registry_sha256": store.registry["registry_sha256"],
        "registered_targets": len(groups),
        "completed_targets": sum(x["status"] == "PASS" for x in targets),
        "registered_obligations": len(store.list()),
        "completed_obligations": sum(x["status"] == "PASS" for x in obligations),
        "elapsed_seconds": round(time.time() - started, 3),
        "targets": targets,
        "obligations": obligations,
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: checkpoint[k] for k in ("schema_id", "status", "registered_targets", "completed_targets", "registered_obligations", "completed_obligations", "elapsed_seconds")}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
