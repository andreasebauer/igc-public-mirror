from __future__ import annotations

import ast
import hashlib
import json
import pickle
import resource
import time
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..canon import canonical_sha256, write_json_atomic
from ..checkpoints import CheckpointManager
from ..controller import register_runner
from ..hashing import sha256_file
from ..spectroscope import run_scout_spectroscope, scientific_projection

ROOT_COUNT = 1089
ROOT_SHA256 = "a1b54f2ee2841f029e9bb5b222f8c2b559bd0d8377c6522f59968488bea3a1f3"
PRIMITIVE_SHA256 = "eff6d9ad96fa2a69bce437fcea03249c0b25f19b56d1c99e0902e298ddda6fec"
INPUT_DATASET_SHA256 = "914236c3c31305e2bd72cf4ce0a0f4e464613743530e581fc9511d6c5f2eecda"
SELECTION_RULES_SHA256 = "6a69ffc31ddf396a54c95142169a7e2a01b0153ba9c2d3d45e397177bbaa8499"
OBSERVER_SHA256 = "3ce59d392231d927939120498807bda857abc91bb4f707f86fdd7730ca805820"
CLAIM_BOUNDARY_SHA256 = "917f3bbe3cbd3dd2d4c5a3fb4d5047502b1693d92c5b7cb95d15808453d05c75"
GOLDEN_SHA256 = "7ef85b13a3b2f12e6bf2bba148076d00282a8d3855e790a58ad1cd588d9b993d"
PROHIBITIONS = [
    "NO_FIRST_OCCURRENCE_CLAIM",
    "NO_C_PROVED",
    "NO_POPULATION_WIDE_NEGATIVE",
    "NO_MECHANISM_PROMOTION",
    "NO_GEOMETRY_PROMOTION",
]


def _resource_json(name: str) -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources/scout_l16").joinpath(name)
    return json.loads(p.read_text(encoding="utf-8"))


def _resource_sha(name: str) -> str:
    obj = _resource_json(name)
    # Frozen protocol resources carry their own identity field.  The pinned
    # identity is the canonical object with that identity field removed.
    identity_keys = {
        "L16_SCOUT_SELECTION_AND_STOPPING_RULES_v1.0.json": "rules_sha256",
        "L16_SCOUT_OBSERVER_AND_METRIC_SPEC_v1.0.json": "observer_sha256",
        "L16_SCOUT_EVIDENCE_AND_CLAIM_BOUNDARY_v1.0.json": "boundary_sha256",
        "L16_GOLDEN_COMMITMENTS_v1.0.json": "golden_commitment_sha256",
    }
    k = identity_keys.get(name)
    return canonical_sha256({x: y for x, y in obj.items() if x != k}) if k else canonical_sha256(obj)


from ..core.boundary import destination_tuple as _dtup, canonical_boundary_record as _canon, record_sha256 as _sha_record, bridge_ports as _bridge, compose_boundary_records as _compose

def _vector(r):
    pc = Counter(r[1])
    tc = Counter(_dtup(r[3]))
    return (
        len(r[1]),
        len(pc),
        max(pc.values()),
        sum(v * v for v in pc.values()),
        len(_dtup(r[3])),
        len(tc),
        max(tc.values()),
        sum(v * v for v in tc.values()),
        int(r[0]),
        int(r[2]),
        int(r[4]),
    )


def _d2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _farthest(rows, n):
    if len(rows) <= n:
        return list(rows)
    vs = [_vector(x["record"]) for x in rows]
    mins = [min(v[j] for v in vs) for j in range(len(vs[0]))]
    maxs = [max(v[j] for v in vs) for j in range(len(vs[0]))]
    nv = [
        tuple(0 if maxs[j] == mins[j] else (v[j] - mins[j]) / (maxs[j] - mins[j]) for j in range(len(v)))
        for v in vs
    ]
    st = min(range(len(rows)), key=lambda i: rows[i]["sha256"])
    ch = [st]
    S = {st}
    mind = [_d2(nv[i], nv[st]) for i in range(len(rows))]
    mind[st] = -1
    while len(ch) < n:
        k = max((i for i in range(len(rows)) if i not in S), key=lambda i: (mind[i], rows[i]["sha256"]))
        ch.append(k)
        S.add(k)
        mind[k] = -1
        for i in range(len(rows)):
            if mind[i] >= 0:
                mind[i] = min(mind[i], _d2(nv[i], nv[k]))
    return [rows[i] for i in ch]


def _dependency_artifact(controller, run_dir: Path, stage_id: str, logical_name: str, destination: Path) -> Path:
    cp = CheckpointManager(run_dir, controller.store).current(stage_id)
    if not cp or cp.get("status") != "COMPLETE_VALID":
        raise RuntimeError(f"dependency {stage_id} is not COMPLETE_VALID")
    hits = [a for a in cp.get("output_artifacts", []) if a.get("logical_name") == logical_name]
    if len(hits) != 1:
        raise RuntimeError(f"dependency artifact {stage_id}/{logical_name} not unique: {len(hits)}")
    a = hits[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    controller.store.materialize(a["sha256"], destination, expected_size=a.get("size_bytes"))
    return destination


def _science(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k != "cost"}


def _preflight(fixture: Path, params: dict[str, Any]) -> dict[str, Any]:
    root_path = fixture / "inputs/L15_SELECTED_RELATION.json"
    prim_path = fixture / "inputs/primitive.pkl"
    if sha256_file(root_path) != params["root_sha256"] or params["root_sha256"] != ROOT_SHA256:
        raise RuntimeError("L15 selected-relation identity mismatch")
    if sha256_file(prim_path) != params["primitive_sha256"] or params["primitive_sha256"] != PRIMITIVE_SHA256:
        raise RuntimeError("primitive identity mismatch")
    raw = json.loads(root_path.read_text(encoding="utf-8"))
    if len(raw.get("selected", [])) != int(params["root_count"]) or int(params["root_count"]) != ROOT_COUNT:
        raise RuntimeError("root count mismatch")
    bad = 0
    prev = None
    for x in sorted(raw["selected"], key=lambda z: z["sha256"]):
        r = _canon(ast.literal_eval(x["record"]))
        if _sha_record(r) != x["sha256"]:
            bad += 1
        if prev is not None and x["sha256"] < prev:
            raise RuntimeError("source ordering failure")
        prev = x["sha256"]
    if bad:
        raise RuntimeError(f"{bad} source-record SHA mismatches")
    # Resource identities are frozen before live generation.
    if _resource_sha("L16_SCOUT_SELECTION_AND_STOPPING_RULES_v1.0.json") != SELECTION_RULES_SHA256:
        raise RuntimeError("selection/stopping resource hash mismatch")
    if _resource_sha("L16_SCOUT_OBSERVER_AND_METRIC_SPEC_v1.0.json") != OBSERVER_SHA256:
        raise RuntimeError("observer resource hash mismatch")
    if _resource_sha("L16_SCOUT_EVIDENCE_AND_CLAIM_BOUNDARY_v1.0.json") != CLAIM_BOUNDARY_SHA256:
        raise RuntimeError("claim boundary resource hash mismatch")
    return {
        "status": "PASS",
        "target": "L16_SCOUT_PREFLIGHT",
        "root_count": ROOT_COUNT,
        "root_sha256": ROOT_SHA256,
        "primitive_sha256": PRIMITIVE_SHA256,
        "input_dataset_sha256": INPUT_DATASET_SHA256,
        "selection_rules_sha256": SELECTION_RULES_SHA256,
        "observer_sha256": OBSERVER_SHA256,
        "claim_boundary_sha256": CLAIM_BOUNDARY_SHA256,
        "population_complete": False,
        "evidence_label": "SCOUT_OBSERVED",
    }


def _wide(fixture: Path, per_lane: int, lane_ids: list[str]):
    t0 = time.perf_counter()
    c0 = time.process_time()
    raw = json.loads((fixture / "inputs/L15_SELECTED_RELATION.json").read_text(encoding="utf-8"))
    with (fixture / "inputs/primitive.pkl").open("rb") as h:
        prim = pickle.load(h)
    sources = [{"sha256": x["sha256"], "record": _canon(ast.literal_eval(x["record"]))} for x in raw["selected"]]
    sources.sort(key=lambda x: x["sha256"])
    if len(sources) != ROOT_COUNT:
        raise RuntimeError("L16 root count changed")
    children = {}
    pars = defaultdict(set)
    wit = defaultdict(int)
    src_child = defaultdict(set)
    attempts = 0
    lawful = 0
    prim_records = [_canon(p["record"]) for p in prim]
    for s in sources:
        a = s["record"]
        for b in prim_records:  # preserve primitive record order exactly
            for sa in range(len(a[1])):
                for sp in range(len(b[1])):
                    attempts += 1
                    ch = _compose(a, sa, b, sp)
                    if ch is None:
                        continue
                    lawful += 1
                    h = _sha_record(ch)
                    children[h] = ch
                    pars[h].add(s["sha256"])
                    wit[h] += 1
                    src_child[s["sha256"]].add(h)
    rows = [
        {"sha256": h, "record": r, "parent_count_bounded": len(pars[h]), "witness_count": wit[h]}
        for h, r in children.items()
    ]
    rows.sort(key=lambda x: x["sha256"])
    N = min(int(per_lane), len(rows))
    if N != 256:
        raise RuntimeError(f"frozen per-lane size not reached: {N}")
    D = _farthest(rows, N)
    B = sorted(rows, key=lambda x: (-x["parent_count_bounded"], -x["witness_count"], x["sha256"]))[:N]
    S = sorted(rows, key=lambda x: (len(set(x["record"][1])), max(Counter(x["record"][1]).values()), x["sha256"]))[:N]
    O = sorted(
        rows,
        key=lambda x: (
            -sum(v * (v - 1) // 2 for v in Counter(x["record"][1]).values()),
            -x["parent_count_bounded"],
            -x["witness_count"],
            x["sha256"],
        ),
    )[:N]
    line = []
    seen = set()
    row_by_sha = {x["sha256"]: x for x in rows}
    for s in sources:
        hs = sorted(src_child[s["sha256"]])
        if hs and len(line) < N:
            h = hs[0]
            if h not in seen:
                line.append(row_by_sha[h])
                seen.add(h)
    for x in rows:
        if len(line) >= N:
            break
        if x["sha256"] not in seen:
            line.append(x)
            seen.add(x["sha256"])
    lanes = {"D": D, "B": B, "S": S, "O": O, "L": line}
    if lane_ids != ["D", "B", "S", "O", "L"]:
        raise RuntimeError("lane order differs from frozen plan")
    union = {x["sha256"]: x for lane in lane_ids for x in lanes[lane]}
    selected = sorted(union.values(), key=lambda x: x["sha256"])
    selset = set(union)
    edges = sorted((s["sha256"], h) for s in sources for h in src_child[s["sha256"]] if h in selset)
    pair = Counter()
    for s in sources:
        hs = sorted(h for h in src_child[s["sha256"]] if h in selset)
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                pair[(hs[i], hs[j])] += 1
    fib = Counter(tuple(sorted(pars[x["sha256"]])) for x in selected)
    br = {
        "selected_union": len(selected),
        "relation_edges": len(edges),
        "selected_multi_parent": sum(x["parent_count_bounded"] >= 2 for x in selected),
        "selected_max_parent_count": max(x["parent_count_bounded"] for x in selected),
        "shared_parent_pair_count": len(pair),
        "shared_parent_multiplicity_histogram": dict(sorted(Counter(pair.values()).items())),
        "distinct_exact_parent_fibers": len(fib),
        "selected_parent_histogram": dict(sorted(Counter(x["parent_count_bounded"] for x in selected).items())),
    }
    gen = {
        "L15_sources": len(sources),
        "attempts": attempts,
        "lawful": lawful,
        "global_unique_L16_candidates": len(rows),
        "multi_parent_candidates": sum(len(v) >= 2 for v in pars.values()),
        "max_bounded_parent_count": max(map(len, pars.values())),
    }
    relation = {
        "selected": [
            {
                "sha256": x["sha256"],
                "record": repr(x["record"]),
                "bounded_parent_count": x["parent_count_bounded"],
                "witness_count": x["witness_count"],
            }
            for x in selected
        ],
        "edges": edges,
    }
    lane_seal = {
        "schema_id": "IG_L16_SCOUT_SEALED_LANES_V1",
        "selection_rules_sha256": SELECTION_RULES_SHA256,
        "per_lane": N,
        "lane_order": lane_ids,
        "lanes": {k: [x["sha256"] for x in lanes[k]] for k in lane_ids},
        "selected_union": [x["sha256"] for x in selected],
        "selected_union_count": len(selected),
        "sealed_before_relation_analysis": True,
        "no_outcome_feedback": True,
    }
    res = {
        "stage": "L16_ROADMAP_SCOUT_WIDE",
        "status": "SCOUT_COMPLETE",
        "evidence_label": "SCOUT_OBSERVED",
        "authoritative": False,
        "source_level": 15,
        "target_level": 16,
        "root_scope": "1089 sealed L15 scout objects; NOT complete L15 population",
        "generation": gen,
        "selection": {"per_lane": N, "union_selected": len(selected)},
        "bounded_relation": br,
        # Runtime cost/resource measurements are deliberately excluded from
        # the scientific result artifact.  They are recorded by the unified
        # controller/checkpoint envelope, which keeps release bytes replayable
        # across machines while preserving resource provenance separately.
        "prohibitions": PROHIBITIONS,
    }
    return res, relation, lane_seal


def _build_l16_baseline(wide: dict[str, Any], spec: dict[str, Any], l15: dict[str, Any]) -> dict[str, Any]:
    br = wide["bounded_relation"]
    clo = spec["closure"]
    order = spec["order"]
    ref = spec["refinement"]
    scope = spec["scope"]
    l15_rel = l15["relations"]
    l15_parent_max = l15_rel["R_PARENT_FIBER"]["metrics"]["selected_max_parent_count"]
    l15_inc_h = l15_rel["R_FIBER_INCLUSION"]["metrics"]["height"]
    l15_inter = l15_rel["R_PARTIAL_CLOSURE"]["metrics"]["intersection_realized_fraction"]
    l15_union = l15_rel["R_PARTIAL_CLOSURE"]["metrics"]["union_realized_fraction"]
    l15_right_frac = l15_rel["R_REFINEMENT_INDIVIDUATION"]["metrics"]["right_singletons"] / l15_rel["R_REFINEMENT_INDIVIDUATION"]["metrics"]["right_nodes"]
    right_frac = ref["right_singletons"] / scope["right_nodes"]
    relations = {
        "R_FIBER_INCLUSION": {
            "comparison_to_L15": {
                "height_L15": l15_inc_h,
                "height_L16": order["height"],
                "note": "Counts/rates not population-comparable because bounded root scopes differ.",
            },
            "metrics": {
                "cover_relations": order["cover_relations"],
                "height": order["height"],
                "strict_inclusion_pairs": order["strict_inclusion_pairs"],
                "width": order["width"],
            },
            "promotion": "TRACK_UPWARD",
            "states": ["PERSISTS", "REORGANIZES"],
        },
        "R_LOCAL_DEPTH_REORGANIZATION": {
            "metrics": {},
            "promotion": "UNRESOLVED_DEPTH_SCOPE",
            "states": ["UNRESOLVED_DEPTH_SCOPE"],
        },
        "R_PARENT_FIBER": {
            "comparison_to_L15": {
                "fraction_comparison": "NOT_PROMOTED_DIFFERENT_ROOT_SCOPE",
                "max_parent_count": {"L15": l15_parent_max, "L16": br["selected_max_parent_count"], "direction": "UP" if br["selected_max_parent_count"] > l15_parent_max else "NOT_UP"},
            },
            "metrics": {
                "distinct_parent_fibers": br["distinct_exact_parent_fibers"],
                "multi_parent_fraction": br["selected_multi_parent"] / br["selected_union"],
                "selected_max_parent_count": br["selected_max_parent_count"],
                "selected_multi_parent": br["selected_multi_parent"],
                "selected_union": br["selected_union"],
            },
            "promotion": "TRACK_UPWARD",
            "states": ["PERSISTS", "EXPANDS"],
        },
        "R_PARTIAL_CLOSURE": {
            "comparison_to_L15": {
                "intersection_fraction_L15": l15_inter,
                "intersection_fraction_L16": clo["intersection_realized_fraction"],
                "note": "Directional comparison diagnostic only; bounded root scopes differ.",
                "union_fraction_L15": l15_union,
                "union_fraction_L16": clo["union_realized_fraction"],
            },
            "metrics": {
                "both_realized": clo["both_realized"],
                "intersection_realized_fraction": clo["intersection_realized_fraction"],
                "union_realized_fraction": clo["union_realized_fraction"],
            },
            "promotion": "TRACK_UPWARD",
            "states": ["PERSISTS", "REORGANIZES"],
        },
        "R_REFINEMENT_INDIVIDUATION": {
            "comparison_to_L15": {
                "note": "Diagnostic only; bounded root scopes differ.",
                "right_singleton_fraction_L15": l15_right_frac,
                "right_singleton_fraction_L16": right_frac,
            },
            "metrics": {
                "left_nodes": scope["left_nodes"],
                "left_singletons": ref["left_singletons"],
                "right_nodes": scope["right_nodes"],
                "right_singleton_fraction": right_frac,
                "right_singletons": ref["right_singletons"],
                "stable_left_classes": ref["stable_left_classes"],
                "stable_right_classes": ref["stable_right_classes"],
            },
            "promotion": "TRACK_UPWARD",
            "states": ["PERSISTS", "REORGANIZES"],
        },
        "R_SHARED_SUPPORT": {
            "metrics": {
                "max_shared_parent_multiplicity": max(map(int, br["shared_parent_multiplicity_histogram"].keys())),
                "shared_parent_pair_count": br["shared_parent_pair_count"],
            },
            "promotion": "TRACK_UPWARD",
            "states": ["PERSISTS"],
        },
    }
    baseline = {
        "authoritative": False,
        "evidence_label": "SCOUT_OBSERVED",
        "level": 16,
        "prohibitions": PROHIBITIONS,
        "promotion_summary": {
            "HIGH_PRIORITY_MATURATION_REVIEW": ["R_PARENT_FIBER"],
            "HIGH_PRIORITY_OFFICIAL_TEST": [],
            "SCOUT_STRUCTURAL_SHOCK": False,
            "TRACK_UPWARD": [
                "R_PARENT_FIBER",
                "R_SHARED_SUPPORT",
                "R_FIBER_INCLUSION",
                "R_PARTIAL_CLOSURE",
                "R_REFINEMENT_INDIVIDUATION",
            ],
            "UNRESOLVED_DEPTH_SCOPE": ["R_LOCAL_DEPTH_REORGANIZATION"],
        },
        "relations": relations,
        "roadmap_summary": [
            "ALTERNATIVE_PARENT_ORGANIZATION_PERSISTS_BOUNDED",
            "PARENT_MULTIPLICITY_EXPANDS_TO_16_BOUNDED",
            "SHARED_SUPPORT_FACTORIZATION_PERSISTS_BOUNDED",
            "FIBER_INCLUSION_ORDER_PERSISTS_BUT_REORGANIZES_BOUNDED",
            "PARTIAL_CLOSURE_PERSISTS_BUT_NOT_MONOTONICALLY_STRENGTHENED_IN_THIS_PANEL",
            "RELATIONAL_INDIVIDUATION_PERSISTS_BUT_IS_LOOSER_IN_THIS_PANEL",
            "DEEP_MATURATION_UNRESOLVED_DEPTH_SCOPE",
        ],
        "scope": {
            "candidate_count": wide["generation"]["global_unique_L16_candidates"],
            "relation_edges": br["relation_edges"],
            "root_scope": wide["root_scope"],
            "selected_union": br["selected_union"],
        },
        "spectroscope": {
            "abstract_cycle_rank_recognition_only": spec["incidence"]["abstract_cycle_rank"],
            "components": spec["incidence"]["components"],
            "recognition_summary": spec["recognition_summary"],
        },
        "stage": "L16_SCOUT_LEVEL_BASELINE",
    }
    # Legacy L16 baseline serialization computed its self-hash *after* adding
    # cost, while the science projection later removed cost but retained that
    # self-hash.  That accidentally made a nominal science hash depend on
    # historical timing.  Preserve the historical non-scientific cost bytes
    # solely for replay compatibility; current execution cost lives in the
    # universal stage/run resource records.
    compat = _resource_json("L16_LEGACY_BASELINE_COST_COMPAT_v1.0.json")
    baseline["cost"] = compat["cost"]
    baseline["baseline_sha256"] = canonical_sha256(baseline)
    return baseline


def _classification(baseline: dict[str, Any]) -> dict[str, Any]:
    persistent = []
    expanded = []
    reorganized = []
    destroyed = []
    observed = []
    unresolved = []
    current_escalated = []
    for name, item in baseline["relations"].items():
        states = set(item.get("states", []))
        if "SCOUT_NEW" in states:
            observed.append(name)
        if "PERSISTS" in states:
            persistent.append(name)
        if "EXPANDS" in states:
            expanded.append(name)
        if states & {"REFINES", "REORGANIZES", "COMBINES", "CLOSES"}:
            reorganized.append(name)
        if "DESTROYS_OR_RELIEVES" in states:
            destroyed.append(name)
        if "UNRESOLVED_DEPTH_SCOPE" in states:
            unresolved.append(name)
        if item.get("promotion") in {"HIGH_PRIORITY_MATURATION_REVIEW", "HIGH_PRIORITY_OFFICIAL_TEST"}:
            current_escalated.append(name)
    return {
        "authoritative": False,
        "carried_program_priority": baseline["promotion_summary"],
        "current_level_escalated_for_review": sorted(current_escalated),
        "destroyed_or_relieved": sorted(destroyed),
        "evidence_label": "SCOUT_OBSERVED",
        "expanded": sorted(expanded),
        "level": 16,
        "observed_or_new": sorted(observed),
        "persistent": sorted(persistent),
        "prohibitions": PROHIBITIONS,
        "reorganized_or_refined": sorted(reorganized),
        "schema_id": "IG_L16_SCOUT_CLASSIFICATION_PROJECTION_V1",
        "summary": "PERSISTS + EXPANDS + REORGANIZES (bounded panel only)",
        "unresolved": sorted(unresolved),
    }


@register_runner("adapter.scout_l16.preflight")
def preflight_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["fixture_dataset_sha256"] != INPUT_DATASET_SHA256:
        raise RuntimeError("frozen L16 dataset identity changed")
    fixture = controller.datasets.materialize(INPUT_DATASET_SHA256, work_dir / "fixture")
    result = _preflight(fixture, stage["params"])
    out = work_dir / "L16_PREFLIGHT_RESULT.json"
    write_json_atomic(out, result)
    return {"outputs": {"L16_PREFLIGHT_RESULT.json": out}, "stage_result": result}


@register_runner("adapter.scout_l16.wide")
def wide_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["selection_rules_sha256"] != SELECTION_RULES_SHA256:
        raise RuntimeError("selection-rules parameter mismatch")
    fixture = controller.datasets.materialize(stage["params"]["fixture_dataset_sha256"], work_dir / "fixture")
    wide, relation, lane_seal = _wide(fixture, int(stage["params"]["per_lane"]), list(stage["params"]["lane_ids"]))
    p_w = work_dir / "L16_SCOUT_WIDE_RESULT.json"
    p_r = work_dir / "SELECTED_RELATION.json"
    p_l = work_dir / "SEALED_LANE_SELECTION.json"
    write_json_atomic(p_w, wide)
    write_json_atomic(p_r, relation)
    write_json_atomic(p_l, lane_seal)
    return {
        "outputs": {
            "L16_SCOUT_WIDE_RESULT.json": p_w,
            "SELECTED_RELATION.json": p_r,
            "SEALED_LANE_SELECTION.json": p_l,
        },
        "stage_result": {
            "status": "PASS",
            "target": "L16_SCOUT_WIDE",
            "candidate_count": wide["generation"]["global_unique_L16_candidates"],
            "selected_union": wide["bounded_relation"]["selected_union"],
            "relation_edges": wide["bounded_relation"]["relation_edges"],
            "selection_sealed_before_relation_analysis": True,
        },
    }


@register_runner("adapter.scout_l16.spectroscope")
def spectroscope_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"].get("recognition_only") is not True or stage["params"].get("cannot_feedback") is not True:
        raise RuntimeError("Spectroscope non-feedback contract violated")
    rel = _dependency_artifact(controller, run_dir, "wide", "SELECTED_RELATION.json", work_dir / "inputs/SELECTED_RELATION.json")
    result = run_scout_spectroscope(rel, work_dir / "spectroscope", 16)
    out = work_dir / "spectroscope/SCOUT_SPECTROSCOPE_RESULT.json"
    # The shared Spectroscope records local runtime cost for interactive use.
    # Strip that non-scientific field at the L16 publication boundary; the
    # common runtime already records resource usage in run/checkpoint records.
    result = {k: v for k, v in result.items() if k != "cost"}
    write_json_atomic(out, result)
    return {
        "outputs": {"SCOUT_SPECTROSCOPE_RESULT.json": out},
        "stage_result": {
            "status": "PASS",
            "target": "L16_SCOUT_SPECTROSCOPE",
            "recognition_only": True,
            "authority": "RECONNAISSANCE_ONLY",
            "recognition_summary": result["recognition_summary"],
        },
    }


@register_runner("adapter.scout_l16.classify")
def classify_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["observer_sha256"] != OBSERVER_SHA256 or stage["params"]["claim_boundary_sha256"] != CLAIM_BOUNDARY_SHA256:
        raise RuntimeError("frozen observer/claim boundary changed")
    fixture = controller.datasets.materialize(stage["params"]["fixture_dataset_sha256"], work_dir / "fixture")
    pwide = _dependency_artifact(controller, run_dir, "wide", "L16_SCOUT_WIDE_RESULT.json", work_dir / "inputs/L16_SCOUT_WIDE_RESULT.json")
    prel = _dependency_artifact(controller, run_dir, "wide", "SELECTED_RELATION.json", work_dir / "inputs/SELECTED_RELATION.json")
    pspec = _dependency_artifact(controller, run_dir, "spectroscope", "SCOUT_SPECTROSCOPE_RESULT.json", work_dir / "inputs/SCOUT_SPECTROSCOPE_RESULT.json")
    wide = json.loads(pwide.read_text(encoding="utf-8"))
    relation = json.loads(prel.read_text(encoding="utf-8"))
    spec = json.loads(pspec.read_text(encoding="utf-8"))
    l15 = json.loads((fixture / "baseline/L15_SCOUT_BASELINE.json").read_text(encoding="utf-8"))
    baseline = _build_l16_baseline(wide, spec, l15)
    classification = _classification(baseline)
    # Normalize through JSON exactly as the historical file boundary did;
    # histogram keys become strings on disk.
    norm = lambda obj: json.loads(json.dumps(obj, sort_keys=True))
    projections = {
        "schema_id": "IG_L16_SCOUT_SCIENCE_PROJECTIONS_V1",
        "projection_version": "L16_SCOUT_SCIENCE_PROJECTION_V1",
        "wide_science_no_cost_sha256": canonical_sha256(norm(_science(wide))),
        "selected_relation_canonical_sha256": canonical_sha256(norm(relation)),
        "spectroscope_science_no_cost_sha256": canonical_sha256(norm(scientific_projection(spec))),
        "baseline_science_no_cost_sha256": canonical_sha256(norm(_science(baseline))),
        "classification_canonical_sha256": canonical_sha256(classification),
        "evidence_label": "SCOUT_OBSERVED",
        "authority": "RECONNAISSANCE_ONLY",
        "population_complete": False,
    }
    pbase = work_dir / "L16_SCOUT_BASELINE.json"
    pcl = work_dir / "L16_SCOUT_CLASSIFICATION.json"
    pproj = work_dir / "L16_SCIENCE_PROJECTIONS.json"
    write_json_atomic(pbase, baseline)
    write_json_atomic(pcl, classification)
    write_json_atomic(pproj, projections)
    return {
        "outputs": {
            "L16_SCOUT_BASELINE.json": pbase,
            "L16_SCOUT_CLASSIFICATION.json": pcl,
            "L16_SCIENCE_PROJECTIONS.json": pproj,
        },
        "stage_result": {
            "status": "PASS",
            "target": "L16_SCOUT_CLASSIFY",
            "summary": classification["summary"],
            "authority": "RECONNAISSANCE_ONLY",
            "evidence_label": "SCOUT_OBSERVED",
        },
    }


@register_runner("adapter.scout_l16.verify")
def verify_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["golden_commitment_sha256"] != GOLDEN_SHA256:
        raise RuntimeError("golden commitment parameter mismatch")
    if stage["params"].get("projection_version") != "L16_SCOUT_SCIENCE_PROJECTION_V1":
        raise RuntimeError("projection version mismatch")
    # Golden commitments become accessible only in this terminal runner.
    golden = _resource_json("L16_GOLDEN_COMMITMENTS_v1.0.json")
    if _resource_sha("L16_GOLDEN_COMMITMENTS_v1.0.json") != GOLDEN_SHA256:
        raise RuntimeError("packaged golden commitment identity mismatch")
    pproj = _dependency_artifact(controller, run_dir, "classify", "L16_SCIENCE_PROJECTIONS.json", work_dir / "inputs/L16_SCIENCE_PROJECTIONS.json")
    pcl = _dependency_artifact(controller, run_dir, "classify", "L16_SCOUT_CLASSIFICATION.json", work_dir / "inputs/L16_SCOUT_CLASSIFICATION.json")
    projections = json.loads(pproj.read_text(encoding="utf-8"))
    classification = json.loads(pcl.read_text(encoding="utf-8"))
    commitments = golden["commitments"]
    comparisons = {
        "wide_science_no_cost": projections["wide_science_no_cost_sha256"] == commitments["wide"]["science_no_cost_sha256"],
        "selected_relation": projections["selected_relation_canonical_sha256"] == commitments["selected_relation"]["canonical_sha256"],
        "spectroscope_science_no_cost": projections["spectroscope_science_no_cost_sha256"] == commitments["spectroscope"]["science_no_cost_sha256"],
        "baseline_science_no_cost": projections["baseline_science_no_cost_sha256"] == commitments["baseline"]["science_no_cost_sha256"],
        "classification": projections["classification_canonical_sha256"] == commitments["classification_projection"]["canonical_sha256"],
    }
    status = "PASS" if all(comparisons.values()) else "FAIL"
    verification = {
        "schema_id": "IG_L16_SCOUT_TERMINAL_VERIFICATION_V1",
        "status": status,
        "golden_commitment_sha256": GOLDEN_SHA256,
        "projection_version": projections["projection_version"],
        "comparisons": comparisons,
        "observed_projection_hashes": projections,
        "mismatch_action": golden["mismatch_action"],
        "golden_access_stage": "verify_only",
    }
    principal = {
        "schema_id": "IG_L16_SCOUT_PRINCIPAL_RESULT_V1",
        "status": status,
        "target": "L16_ROADMAP_SCOUT_HISTORICAL_SCOPE_DIRECT_REPLAY",
        "level": 16,
        "source_level": 15,
        "execution_class": "HISTORICAL_SCOPE_DIRECT_REPLAY_AND_CURRENT_UNIFIED_RUNTIME_INTEGRATION",
        "evidence_label": "SCOUT_OBSERVED",
        "authority": "RECONNAISSANCE_ONLY",
        "authoritative": False,
        "scope": "BOUNDED_PREREGISTERED",
        "population_complete": False,
        "root_count": ROOT_COUNT,
        "root_sha256": ROOT_SHA256,
        "primitive_sha256": PRIMITIVE_SHA256,
        "verification": comparisons,
        "science_projection_hashes": projections,
        "classification": classification,
        "summary": classification["summary"],
        "prohibitions": PROHIBITIONS + ["NO_TOPOLOGY_PROMOTION_FROM_CYCLE_RANK"],
        "claims_licensed": [
            "bounded persistence of tracked relation families",
            "bounded parent-fiber expansion to observed maximum 16",
            "bounded reorganization of fiber inclusion, partial closure, and refinement individuation",
        ],
        "claims_forbidden": [
            "first occurrence",
            "population-wide absence",
            "mechanism",
            "geometry",
            "physical topology",
            "transition level",
            "graduation",
        ],
        "deep_maturation": "UNRESOLVED_DEPTH_SCOPE",
    }
    pv = work_dir / "L16_SCOUT_TERMINAL_VERIFICATION.json"
    pp = work_dir / "L16_SCOUT_PRINCIPAL_RESULT.json"
    write_json_atomic(pv, verification)
    write_json_atomic(pp, principal)
    if status != "PASS":
        raise RuntimeError(f"L16 golden science-projection mismatch: {comparisons}")
    return {
        "outputs": {
            "L16_SCOUT_TERMINAL_VERIFICATION.json": pv,
            "L16_SCOUT_PRINCIPAL_RESULT.json": pp,
        },
        "stage_result": {
            "status": "PASS",
            "target": "L16_SCOUT_TERMINAL_VERIFY",
            "comparisons": comparisons,
            "authority": "RECONNAISSANCE_ONLY",
            "evidence_label": "SCOUT_OBSERVED",
        },
    }
