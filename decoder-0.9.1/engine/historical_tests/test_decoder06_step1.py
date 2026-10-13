"""Four bounded Step 1 checks, executed only by the saved Decoder VALIDATION job.

No scientific kernel run, replacement controller, private pool, or execution
context is opened here. Observations are captured in Decoder-owned worker logs.
"""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import sys
import tomllib
import zipfile

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / "source"
JOB_ID = "DECODER.06.STEP1.VALIDATION.R2"


def _inputs():
    job = json.loads((WORKSPACE / "registry" / (JOB_ID + ".json")).read_text())
    row = next(a for a in job["input_artifacts"] if a["logical_name"] == "step1_evidence")
    p = WORKSPACE / "runtime/intake/artifacts" / (row["sha256"] + ".zip")
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    with zipfile.ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist()}


def _emit(kind, value):
    print("IG_STEP1_EVIDENCE_JSON:" + json.dumps({"kind": kind, "value": value}, sort_keys=True))


def test_parent_source_unchanged_and_input_closure():
    data = _inputs()
    man = json.loads(data["PARENT_SOURCE_MANIFEST.json"])
    for row in man["files"]:
        p = SOURCE / row["path"]
        assert p.is_file(), row["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"], row["path"]
    skip = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".engineering_tmp"}
    actual = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*")
              if p.is_file() and not any(x in skip for x in p.relative_to(SOURCE).parts)
              and p.suffix not in {".pyc", ".pyo"}}
    expected = {r["path"] for r in man["files"]}
    assert actual - expected == {"tests/test_decoder06_step1.py"}
    _emit("parent_source_integrity", {"baseline_files_verified": len(expected),
        "changed_baseline_files": [], "added_files": sorted(actual - expected),
        "parent_source_sha256": man["source_sha256"]})


def test_version_inventory_and_unified_target():
    data = _inputs()
    project = tomllib.loads((SOURCE / "pyproject.toml").read_text())["project"]
    init = ast.parse((SOURCE / "infinity_grid/__init__.py").read_text())
    values = [ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    assert values == ["0.30.88.dev6"]
    assert project["version"] == "0.30.88.dev5"
    change = json.loads(data["REPAIR_CHANGE_RECORD.json"])
    assert change["target_release"] == "0.6.0"
    assert "one canonical release-version value" in data["VERSIONING_06_CONTRACT.txt"].decode().lower()
    _emit("version_inventory", {"release_label": "DECODER_V05_CUTDOWN_1",
        "package_metadata_version": project["version"], "runtime_version": values[0],
        "mismatch_confirmed": True, "target_release": "0.6.0",
        "version_alignment_implemented": False,
        "interpretation": "Known baseline mismatch inventoried; this PASS is not a version-alignment acceptance."})


def test_parent_receipts_and_branch_repair_disposition():
    data = _inputs()
    selected = json.loads(data["PARENT_SELECTION.json"])
    assert selected["status"] == "SELECTED_FOR_ENGINEERING_ONLY"
    assert selected["science_run_authorized_this_step"] is False
    expected = {
        "parent_acceptance_reuse.json": "7315f8d2eb634a31ab985041072dbe710c1c6e67ea2a467163c2684c3763a98e",
        "parent_regression_reuse.json": "fa43ff43a2852b72404ef92ddf1ce78a31e8af9467e35b7b888dbf9954f073f8",
    }
    for name, digest in expected.items():
        rec = json.loads(data[name])
        assert rec["reused"] is True and rec["status"] == "COMPLETED"
        assert rec["source_sha256"] == selected["source_sha256"]
        assert rec["completion_sha256"] == digest
    branch = selected["carry_forward"][0]
    assert branch["global_activation_claimed"] is False
    delta = json.loads(data["A30_CARRY_FORWARD_MANIFEST.json"])
    assert len(delta["files"]) == 2
    for row in delta["files"]:
        assert hashlib.sha256(data["carry_forward/" + row["path"]]).hexdigest() == row["sha256"]
    _emit("parent_and_branch_disposition", {"saved_acceptance_records_reused": 2,
        "historical_test_cases_represented": 107, "historical_tests_rerun": 0,
        "a30_delta_files_preserved": 2, "a30_globally_activated": False,
        "a30_correctness_not_rerun": True})


def test_actual_environment_capabilities():
    def text_if_present(path):
        p = Path(path)
        return p.read_text().strip() if p.is_file() else None
    disk = shutil.disk_usage(WORKSPACE)
    affin = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    assert sys.version_info >= (3, 11)
    assert disk.free > 16 * 1024 * 1024
    deps = {n: importlib.metadata.version(n) for n in ("pytest", "cryptography", "setuptools")}
    assert deps == {"pytest": "9.0.2", "cryptography": "46.0.4", "setuptools": "82.0.1"}
    _emit("runtime_capabilities", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "dependencies": deps,
        "worker_pid": os.getpid(), "parent_pid": os.getppid(),
        "cpu_affinity_count": len(affin) if affin else None,
        "cpu_quota": text_if_present("/sys/fs/cgroup/cpu.max"),
        "memory_limit": text_if_present("/sys/fs/cgroup/memory.max"),
        "multiprocessing_methods_reported": multiprocessing.get_all_start_methods(),
        "free_workspace_bytes": disk.free,
        "detached_execution": "Established by launch record and controller completion, not this test alone",
        "drive_connector": "Parent and preregistered source roundtrips recorded externally",
        "unattended_drive_mirroring": "NOT_IMPLEMENTED_IN_SELECTED_PARENT",
        "persistent_shared_authority": "NOT_CONFIGURED_OR_VERIFIED_THIS_STEP",
        "remote_compute_host": "NOT_INSPECTED_OR_DEPLOYED_THIS_STEP"})
