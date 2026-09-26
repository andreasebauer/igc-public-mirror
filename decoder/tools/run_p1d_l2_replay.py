#!/usr/bin/env python3
"""Run the canonical finite-L2 replay in an isolated flat workspace.

The historical scripts use /mnt/data as their declared execution root.  This
driver refuses to use an existing /mnt/data, creates a temporary compatibility
symlink, runs only explicitly registered checks, and removes the symlink on
exit.  Original source archives are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


REGISTERED = (
    ("bridge_sufficiency", "ig_rm1_L2_bridge_sufficiency_code_audit.py", "ig_rm1_L2_bridge_sufficiency_code_audit_results.json"),
    ("output_sufficiency", "ig_rm1_L2_output_sufficiency_code_audit.py", "ig_rm1_L2_output_sufficiency_code_audit_results.json"),
    ("compositor_congruence", "ig_rm1_L2_compositor_congruence_code_audit.py", "ig_rm1_L2_compositor_congruence_code_audit_results.json"),
    ("disjoint_attachment_interchange", "ig_rm1_L2_disjoint_attachment_interchange_audit.py", "ig_rm1_L2_disjoint_attachment_interchange_audit_results.json"),
    ("canonical_alignment", "ig_rm1_L2_canonical_alignment_sufficiency_test.py", "ig_rm1_L2_canonical_alignment_sufficiency_results.json"),
    ("iso_bridge_sufficiency", "ig_rm1_L2_iso_bridge_sufficiency_test.py", "ig_rm1_L2_iso_bridge_sufficiency_results.json"),
    ("iso_output_collision", "ig_rm1_L2_iso_output_collision_sweep.py", "ig_rm1_L2_iso_output_collision_sweep_results.json"),
    ("bijection_independence", "ig_rm1_L2_bijection_independence_test.py", "ig_rm1_L2_bijection_independence_results.json"),
    ("recursive_completeness", "ig_rm1_L2_recursive_completeness_constructor_audit.py", "ig_rm1_L2_recursive_completeness_constructor_audit_results.json"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_sha(path: Path) -> str:
    value = json.loads(path.read_text())
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def extract_flat(source_zip: Path, workspace: Path) -> None:
    with zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or "/" in info.filename:
                continue
            target = workspace / info.filename
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def overlay_canonical(canonical_root: Path, workspace: Path) -> None:
    for src in sorted(canonical_root.rglob("*")):
        if src.is_file():
            shutil.copy2(src, workspace / src.name)


def extract_handoff_core(handoff_zip: Path, workspace: Path) -> None:
    marker = "/core/"
    with zipfile.ZipFile(handoff_zip) as zf:
        for info in zf.infolist():
            if info.is_dir() or marker not in info.filename or info.filename.startswith("__MACOSX/"):
                continue
            rel = info.filename.split(marker, 1)[1]
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--handoff-zip", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    args.source_zip = args.source_zip.resolve()
    args.handoff_zip = args.handoff_zip.resolve()
    args.canonical_root = args.canonical_root.resolve()
    args.workspace = args.workspace.resolve()
    args.checkpoint = args.checkpoint.resolve()

    if Path("/mnt/data").exists() or Path("/mnt/data").is_symlink():
        raise SystemExit("fail closed: /mnt/data already exists")
    args.workspace.mkdir(parents=True, exist_ok=False)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    extract_flat(args.source_zip, args.workspace)
    extract_handoff_core(args.handoff_zip, args.workspace)
    overlay_canonical(args.canonical_root, args.workspace)
    # One audit retained the historical subdirectory spelling although the
    # canonical archive stores the dependency at its flat root.
    # The source upload is flat, while a few scripts retain their original
    # workspace directory prefixes. Relative self-links reproduce that layout
    # without copying or mutating evidence files.
    os.symlink(".", args.workspace / "algebra_scripts")

    frozen = args.workspace / "_frozen"
    frozen.mkdir()
    for _, _, result_name in REGISTERED:
        src = args.workspace / result_name
        if not src.is_file():
            raise SystemExit(f"fail closed: missing frozen result {result_name}")
        shutil.copy2(src, frozen / result_name)

    records: list[dict[str, object]] = []
    started = time.time()
    os.symlink(args.workspace.resolve(), "/mnt/data")
    try:
        for check_id, script_name, result_name in REGISTERED:
            script = args.workspace / script_name
            if not script.is_file():
                records.append({"check_id": check_id, "status": "BLOCKED_MISSING_EXECUTOR", "script": script_name})
                break
            before = sha256(args.workspace / result_name)
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=args.workspace,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            generated = args.workspace / result_name
            record: dict[str, object] = {
                "check_id": check_id,
                "script": script_name,
                "result": result_name,
                "returncode": proc.returncode,
                "elapsed_seconds": round(time.time() - t0, 3),
                "stdout_tail": proc.stdout[-4000:],
                "frozen_file_sha256": before,
            }
            if proc.returncode != 0 or not generated.is_file():
                record["status"] = "FAIL_EXECUTION"
                records.append(record)
                break
            record["generated_file_sha256"] = sha256(generated)
            record["frozen_canonical_json_sha256"] = canonical_json_sha(frozen / result_name)
            record["generated_canonical_json_sha256"] = canonical_json_sha(generated)
            record["status"] = (
                "PASS_EXACT_SEMANTIC_MATCH"
                if record["frozen_canonical_json_sha256"] == record["generated_canonical_json_sha256"]
                else "FAIL_DISCREPANCY"
            )
            records.append(record)
            if record["status"] != "PASS_EXACT_SEMANTIC_MATCH":
                break
    finally:
        Path("/mnt/data").unlink(missing_ok=True)

    overall = "PASS" if len(records) == len(REGISTERED) and all(r["status"] == "PASS_EXACT_SEMANTIC_MATCH" for r in records) else "STOPPED"
    checkpoint = {
        "schema_id": "IG_P1D_L2_FRESH_REPLAY_CHECKPOINT_V1",
        "status": overall,
        "fail_closed": True,
        "registered_checks": len(REGISTERED),
        "completed_checks": sum(r["status"] == "PASS_EXACT_SEMANTIC_MATCH" for r in records),
        "elapsed_seconds": round(time.time() - started, 3),
        "records": records,
    }
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: checkpoint[k] for k in ("schema_id", "status", "registered_checks", "completed_checks", "elapsed_seconds")}, indent=2))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
