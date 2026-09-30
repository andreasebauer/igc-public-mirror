from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .canon import canonical_sha256, write_json_atomic
from .spectroscope import run_scout_spectroscope, scientific_projection

FROZEN_CALIBRATION_SOURCE_SHA256 = "547498ded6c19b914447f2f005c3e0719e7eea2004cc4b3a0c997772eb62fc3e"
CALIBRATION_ACCELERATOR_ID = "SCOUT_L15_FARTHEST_MATH_DIST_EXACT_SCIENCE_EQUIVALENCE_V1"


def _read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _science(o: dict) -> dict:
    return {k: v for k, v in o.items() if k != "cost"}


def _run(script: Path, cwd: Path, args=None):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    return subprocess.run(
        [sys.executable, "-B", str(script)] + list(args or []),
        cwd=str(cwd), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _classification(b: dict) -> dict:
    out = {
        "status": "PASS", "level": 15, "evidence_label": "SCOUT_OBSERVED", "authoritative": False,
        "classification_semantics": "Release-level grouping of already frozen Scout maturation/promotion labels; not a new scientific test.",
        "observed": [], "persistent": [], "reorganized_or_refined": [], "destroyed_or_relieved": [],
        "escalated_for_review": [], "unresolved": [], "promotion_summary": b.get("promotion_summary", {}),
        "prohibitions": b.get("prohibitions", []),
    }
    for name, item in b["relations"].items():
        states = set(item.get("states", [])); promo = item.get("promotion")
        if "SCOUT_NEW" in states: out["observed"].append(name)
        if states & {"PERSISTS", "EXPANDS"}: out["persistent"].append(name)
        if states & {"REFINES", "REORGANIZES", "COMBINES", "CLOSES"}: out["reorganized_or_refined"].append(name)
        if "DESTROYS_OR_RELIEVES" in states: out["destroyed_or_relieved"].append(name)
        if promo in {"HIGH_PRIORITY_MATURATION_REVIEW", "HIGH_PRIORITY_OFFICIAL_TEST"}: out["escalated_for_review"].append(name)
        if "UNRESOLVED_DEPTH_SCOPE" in states or promo == "UNRESOLVED_DEPTH_SCOPE": out["unresolved"].append(name)
    for k in ("observed", "persistent", "reorganized_or_refined", "destroyed_or_relieved", "escalated_for_review", "unresolved"):
        out[k] = sorted(set(out[k]))
    return out


def _accelerated_calibration_source(source_path: Path) -> tuple[str, dict]:
    raw = Path(source_path).read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != FROZEN_CALIBRATION_SOURCE_SHA256:
        raise RuntimeError(f"frozen Scout calibration source identity mismatch: {observed}")
    src = raw.decode("utf-8")
    old_import = "import ast,hashlib,json,pickle,time,resource"
    if src.count(old_import) != 1:
        raise RuntimeError("unexpected Scout calibration import surface")
    src = src.replace(old_import, old_import + ",math")
    old = """    ch=[s]; used={s}; md=[sum((a-b)**2 for a,b in zip(nv[i],nv[s])) for i in range(len(rows))];md[s]=-1\n    while len(ch)<n:\n        k=max((i for i in range(len(rows)) if i not in used),key=lambda i:(md[i],rows[i]['sha256']))\n        ch.append(k);used.add(k);md[k]=-1\n        for i in range(len(rows)):\n            if md[i]>=0:md[i]=min(md[i],sum((a-b)**2 for a,b in zip(nv[i],nv[k])))\n"""
    new = """    ch=[s]; used={s}; md=[math.dist(nv[i],nv[s]) for i in range(len(rows))];md[s]=-1\n    while len(ch)<n:\n        k=max((i for i in range(len(rows)) if i not in used),key=lambda i:(md[i],rows[i]['sha256']))\n        ch.append(k);used.add(k);md[k]=-1\n        nk=nv[k]\n        for i in range(len(rows)):\n            if md[i]>=0:\n                d=math.dist(nv[i],nk)\n                if d<md[i]:md[i]=d\n"""
    if src.count(old) != 1:
        raise RuntimeError("unexpected Scout calibration farthest kernel")
    src = src.replace(old, new)
    meta = {
        "accelerator_id": CALIBRATION_ACCELERATOR_ID,
        "frozen_source_sha256": observed,
        "accelerated_source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "science_acceptance_rule": "FULL_NON_COST_CALIBRATION_ARTIFACT_MUST_EQUAL_FROZEN_EXPECTED",
        "optimization_scope": "DISTANCE_KERNEL_ONLY",
        "new_third_party_dependency": False,
    }
    return src, meta


def replay_scout_l15(fixture: Path, root: Path) -> dict:
    fixture = Path(fixture); root = Path(root)
    # WIDE
    wide = root / "wide"
    for q in ("code", "inputs", "results", "evidence", "checkpoints", "metrics", "logs"):
        (wide / q).mkdir(parents=True, exist_ok=True)
    _copy(fixture / "code/l15_scout_wide_reference.py", wide / "code/l15_scout_wide.py")
    _copy(fixture / "inputs/GLOBAL_L14_PANEL_CHILDREN.json", wide / "inputs/GLOBAL_L14_PANEL_CHILDREN.json")
    _copy(fixture / "inputs/primitive.pkl", wide / "inputs/primitive.pkl")
    p = _run(wide / "code/l15_scout_wide.py", wide)
    if p.returncode: raise RuntimeError(f"wide failed rc={p.returncode}: {p.stderr[-2000:]}")
    wide_obs = _read(wide / "results/L15_SCOUT_WIDE_RESULT.json")
    wide_exp = _read(fixture / "expected/L15_SCOUT_WIDE_RESULT.json")

    # CALIBRATION with exact-science-checked distance-kernel acceleration.
    cal = root / "calibration"
    for q in ("code", "inputs", "results", "evidence", "metrics", "logs"):
        (cal / q).mkdir(parents=True, exist_ok=True)
    patched, accel_meta = _accelerated_calibration_source(fixture / "code/l15_scout_calibration_port.py")
    (cal / "code/l15_scout_calibration.py").write_text(patched, encoding="utf-8")
    write_json_atomic(cal / "evidence/V05_CALIBRATION_ACCELERATOR.json", accel_meta)
    for src, dst in [
        (fixture / "inputs/GLOBAL_L14_PANEL_CHILDREN.json", cal / "inputs/GLOBAL_L14_PANEL_CHILDREN.json"),
        (fixture / "inputs/primitive.pkl", cal / "inputs/primitive.pkl"),
        (wide / "results/L15_SCOUT_WIDE_RESULT.json", cal / "inputs/L15_SCOUT_WIDE_RESULT.json"),
        (wide / "evidence/SEALED_LANE_SELECTION.json", cal / "inputs/SEALED_LANE_SELECTION.json"),
        (wide / "evidence/SELECTED_RELATION.json", cal / "inputs/SELECTED_RELATION.json"),
    ]: _copy(src, dst)
    p = _run(cal / "code/l15_scout_calibration.py", cal)
    if p.returncode: raise RuntimeError(f"calibration failed rc={p.returncode}: {p.stderr[-2000:]}")
    cal_obs = _read(cal / "results/L15_SCOUT_CALIBRATION_RESULT.json")
    cal_exp = _read(fixture / "expected/L15_SCOUT_CALIBRATION_RESULT.json")

    # SPECTROSCOPE
    spec = root / "spectroscope"
    spec_obs = run_scout_spectroscope(wide / "evidence/SELECTED_RELATION.json", spec, 15)
    spec_exp = _read(fixture / "expected/SCOUT_SPECTROSCOPE_RESULT.json")
    spec_obs_normalized = json.loads(json.dumps(spec_obs, sort_keys=True))

    # BASELINE
    base = root / "baseline"; (base / "code").mkdir(parents=True, exist_ok=True); (base / "results").mkdir(parents=True, exist_ok=True)
    _copy(fixture / "code/scout_baseline.py", base / "code/scout_baseline.py")
    args = ["--wide", str(wide / "results/L15_SCOUT_WIDE_RESULT.json"), "--calibration", str(cal / "results/L15_SCOUT_CALIBRATION_RESULT.json"), "--spectroscope", str(spec / "SCOUT_SPECTROSCOPE_RESULT.json"), "--out", str(base / "results"), "--level", "15"]
    p = _run(base / "code/scout_baseline.py", base, args)
    if p.returncode: raise RuntimeError(f"baseline failed rc={p.returncode}: {p.stderr[-2000:]}")
    base_obs = _read(base / "results/SCOUT_BASELINE.json"); base_exp = _read(fixture / "expected/SCOUT_BASELINE.json")

    # DEPTH COMPACT
    depth = root / "depth_compact"
    for q in ("code", "inputs", "evidence", "logs"): (depth / q).mkdir(parents=True, exist_ok=True)
    _copy(fixture / "code/replay_depth_compact.py", depth / "code/replay_depth_compact.py")
    _copy(wide / "evidence/SELECTED_RELATION.json", depth / "inputs/L15_SELECTED_RELATION.json")
    _copy(fixture / "evidence/BOUNDED_L14_L13_EDGES.json", depth / "evidence/BOUNDED_L14_L13_EDGES.json")
    p = _run(depth / "code/replay_depth_compact.py", depth)
    if p.returncode: raise RuntimeError(f"depth failed rc={p.returncode}: {p.stderr[-2000:]}")
    depth_obs = json.loads(p.stdout.strip().splitlines()[-1])
    depth_exp = _read(fixture / "expected/L15_SCOUT_DEPTH_RESULT.json")
    classification = _classification(base_obs); class_exp = _read(fixture / "expected/L15_SCOUT_CLASSIFICATION.json")
    depth_expected_projection = {
        "C0_classes": depth_exp.get("profile_refinement", {}).get("C0_classes"),
        "C1_classes": depth_exp.get("profile_refinement", {}).get("C1_classes"),
        "C2_classes": depth_exp.get("profile_refinement", {}).get("C2_classes"),
        "L14_used": depth_exp.get("bounded_depth_relation", {}).get("L14_objects"),
        "L15": depth_exp.get("bounded_depth_relation", {}).get("L15_objects"),
        "all_used_L14_single_source": "BOUNDED_DEPTH_ROOT_COLLAPSE_ONE_L13_SOURCE_PER_L14_ROOT" in depth_exp.get("roadmap", {}).get("maturation_labels", []),
    }
    comparisons = {
        "wide_science_fields_equal": _science(wide_obs) == _science(wide_exp),
        "calibration_science_fields_equal": _science(cal_obs) == _science(cal_exp),
        "spectroscope_science_fields_equal": scientific_projection(spec_obs_normalized) == scientific_projection(spec_exp),
        "baseline_science_fields_equal": _science(base_obs) == _science(base_exp),
        "depth_compact_replay_pass": depth_obs.get("status") == "PASS",
        "depth_compact_science_fields_equal": depth_obs.get("replay") == depth_expected_projection,
        "classification_equal": classification == class_exp,
    }
    return {
        "status": "PASS" if all(comparisons.values()) else "FAIL",
        "target": "L15_ROADMAP_SCOUT_VERTICAL_SLICE", "evidence_label": "SCOUT_OBSERVED", "authoritative": False,
        "authority": "RECONNAISSANCE_ONLY", "level": 15, "comparisons": comparisons,
        "science_hashes": {
            "wide": canonical_sha256(_science(wide_obs)), "calibration": canonical_sha256(_science(cal_obs)),
            "spectroscope": canonical_sha256(scientific_projection(spec_obs_normalized)), "baseline": canonical_sha256(_science(base_obs)),
            "depth": canonical_sha256(_science(depth_obs)), "classification": canonical_sha256(classification),
        },
        "generation": wide_obs.get("generation"), "selection": wide_obs.get("selection"), "bounded_relation": wide_obs.get("bounded_relation"),
        "promotion_summary": base_obs.get("promotion_summary"), "classification": classification, "depth_scope": cal_obs.get("depth_ancestry"),
        "spectroscope_recognition": spec_obs.get("recognition_summary"),
        "prohibitions": sorted(set(wide_obs.get("prohibitions", []) + spec_obs.get("prohibitions", []))),
        "handoff": "L16_DEFERRED_UNTIL_PHASE1_GRADUATION",
        "v05_migration": {"calibration_accelerator": accel_meta},
    }


SCOUT_COMPARISON_FIELDS = [
    "status", "science_hashes", "generation", "selection", "bounded_relation", "promotion_summary",
    "classification", "depth_scope", "spectroscope_recognition", "prohibitions", "handoff",
]


def scout_science_projection(principal: dict) -> dict:
    return {k: principal.get(k) for k in SCOUT_COMPARISON_FIELDS}
