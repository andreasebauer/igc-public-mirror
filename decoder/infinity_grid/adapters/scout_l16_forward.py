from __future__ import annotations

import ast
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..canon import canonical_sha256, write_json_atomic
from ..controller import register_runner
from ..datasets import DatasetStore
from ..hashing import sha256_file
from ..spectroscope import run_scout_spectroscope, scientific_projection
from .scout_l16 import _canon, _sha_record, _compose, _vector, _dependency_artifact, _dtup

ROOT_COUNT = 1089
ROOT_SHA256 = "a1b54f2ee2841f029e9bb5b222f8c2b559bd0d8377c6522f59968488bea3a1f3"
PRIMITIVE_SHA256 = "eff6d9ad96fa2a69bce437fcea03249c0b25f19b56d1c99e0902e298ddda6fec"
INPUT_DATASET_SHA256 = "914236c3c31305e2bd72cf4ce0a0f4e464613743530e581fc9511d6c5f2eecda"
SELECTION_RULES_SHA256 = "8fd2b9d647e074e68f048c79c1320a15019dd484f35ce8dc3fdf1b9646b85bd2"
OBSERVER_SHA256 = "c13e4a9f7e815cfe3692ac9628175dd939215efe155d59f5c449c0eb673eda7a"
CLAIM_BOUNDARY_SHA256 = "5991a811beac9b81835bb620127bf8d445b09cdf95bcce13f47d3f66ae9a452a"
AUDIT_REFERENCE_SHA256 = "7ede6ed7b4d11110187a0a8cec2492ceed717df9e6fd9da5427de6ebbe85d86b"
PRO_AUDIT_SCIENCE_SHA256 = "1d8982aff4be4678b9aadef73f6810687c77b5bdc65daba9b591369ff1df2163"
PROHIBITIONS = [
    "NO_FIRST_OCCURRENCE_CLAIM",
    "NO_C_PROVED",
    "NO_POPULATION_WIDE_NEGATIVE",
    "NO_MECHANISM_PROMOTION",
    "NO_GEOMETRY_PROMOTION",
    "NO_TOPOLOGY_PROMOTION_FROM_CYCLE_RANK",
    "NO_LATTICE_OR_CLOSED_ALGEBRA_PROMOTION_FROM_AGGREGATE_CLOSURE_COUNTS",
]


def _resource_json(name: str) -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources/scout_l16_forward").joinpath(name)
    return json.loads(p.read_text(encoding="utf-8"))


def _resource_sha(name: str, identity_key: str) -> str:
    obj = _resource_json(name)
    return canonical_sha256({k: v for k, v in obj.items() if k != identity_key})


def _dist_exact(a: tuple[int, ...], b: tuple[int, ...], weights: list[int]) -> int:
    return sum((a[i] - b[i]) ** 2 * weights[i] for i in range(len(a)))


def _farthest_exact(rows: list[dict[str, Any]], n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic farthest-point selection using exact integer arithmetic.

    Each feature dimension is min/max normalized rationally. A common integer
    denominator converts squared Euclidean distances into exact integers.
    First seed: minimum SHA. Exact distance ties thereafter: maximum SHA.
    No float and no tolerance enters the ordering key.
    """
    if len(rows) <= n:
        return list(rows), {
            "arithmetic": "EXACT_INTEGER_RATIONAL_EQUIVALENT",
            "initial_seed": "MINIMUM_SHA256",
            "tie_direction": "MAXIMUM_SHA256",
            "exact_tie_steps": 0,
            "common_denominator": 1,
        }
    vec = [_vector(x["record"]) for x in rows]
    dims = len(vec[0])
    lo = [min(v[i] for v in vec) for i in range(dims)]
    hi = [max(v[i] for v in vec) for i in range(dims)]
    ranges = [hi[i] - lo[i] for i in range(dims)]
    common = 1
    for r in ranges:
        if r:
            common = math.lcm(common, r * r)
    weights = [0 if r == 0 else common // (r * r) for r in ranges]
    start = min(range(len(rows)), key=lambda i: rows[i]["sha256"])
    chosen = [start]
    nearest = [_dist_exact(vec[i], vec[start], weights) for i in range(len(rows))]
    nearest[start] = -1
    exact_ties = 0
    while len(chosen) < n:
        best = max(nearest)
        inds = [i for i, d in enumerate(nearest) if d == best]
        if len(inds) > 1:
            exact_ties += 1
        k = max(inds, key=lambda i: rows[i]["sha256"])
        chosen.append(k)
        nearest[k] = -1
        for i in range(len(rows)):
            if nearest[i] >= 0:
                d = _dist_exact(vec[i], vec[k], weights)
                if d < nearest[i]:
                    nearest[i] = d
    return [rows[i] for i in chosen], {
        "arithmetic": "EXACT_INTEGER_RATIONAL_EQUIVALENT",
        "initial_seed": "MINIMUM_SHA256",
        "tie_direction": "MAXIMUM_SHA256",
        "exact_tie_steps": exact_ties,
        "common_denominator": common,
        "common_denominator_digits": len(str(common)),
        "binary_float_in_ordering_key": False,
        "tolerance_pseudo_ties": False,
    }


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
    prev = None
    bad = 0
    # Check actual on-disk order before any sorting.
    for x in raw["selected"]:
        if prev is not None and x["sha256"] < prev:
            raise RuntimeError("source on-disk canonical ordering failure")
        prev = x["sha256"]
        r = _canon(ast.literal_eval(x["record"]))
        if _sha_record(r) != x["sha256"]:
            bad += 1
    if bad:
        raise RuntimeError(f"{bad} source-record SHA mismatches")
    checks = [
        (_resource_sha("L16_FORWARD_SELECTION_RULES_v1.1.json", "rules_sha256"), SELECTION_RULES_SHA256, "selection rules"),
        (_resource_sha("L16_FORWARD_OBSERVER_METRIC_SPEC_v1.1.json", "observer_sha256"), OBSERVER_SHA256, "observer"),
        (_resource_sha("L16_FORWARD_EVIDENCE_AND_CLAIM_BOUNDARY_v1.1.json", "boundary_sha256"), CLAIM_BOUNDARY_SHA256, "claim boundary"),
        (_resource_sha("L16_FORWARD_AUDIT_REFERENCE_v1.1.json", "reference_sha256"), AUDIT_REFERENCE_SHA256, "audit reference"),
    ]
    for observed, expected, label in checks:
        if observed != expected:
            raise RuntimeError(f"{label} resource hash mismatch")
    return {
        "status": "PASS",
        "target": "L16_FORWARD_PARENT_PREFLIGHT",
        "root_count": ROOT_COUNT,
        "root_sha256": ROOT_SHA256,
        "primitive_sha256": PRIMITIVE_SHA256,
        "input_dataset_sha256": INPUT_DATASET_SHA256,
        "selection_rules_sha256": SELECTION_RULES_SHA256,
        "observer_sha256": OBSERVER_SHA256,
        "claim_boundary_sha256": CLAIM_BOUNDARY_SHA256,
        "audit_reference_sha256": AUDIT_REFERENCE_SHA256,
        "source_order_check": "ORIGINAL_ON_DISK_ORDER_BEFORE_SORT",
        "population_complete": False,
        "evidence_label": "SCOUT_OBSERVED",
    }


def _wide_exact(fixture: Path, per_lane: int, lane_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = json.loads((fixture / "inputs/L15_SELECTED_RELATION.json").read_text(encoding="utf-8"))
    with (fixture / "inputs/primitive.pkl").open("rb") as h:
        prim = pickle.load(h)
    sources = [{"sha256": x["sha256"], "record": _canon(ast.literal_eval(x["record"]))} for x in raw["selected"]]
    sources.sort(key=lambda x: x["sha256"])
    children: dict[str, Any] = {}
    pars: dict[str, set[str]] = defaultdict(set)
    wit: Counter[str] = Counter()
    src_child: dict[str, set[str]] = defaultdict(set)
    attempts = lawful = 0
    prim_records = [_canon(p["record"]) for p in prim]
    for s in sources:
        a = s["record"]
        for b in prim_records:
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
    rows = [{"sha256": h, "record": r, "parent_count_bounded": len(pars[h]), "witness_count": wit[h]} for h, r in children.items()]
    rows.sort(key=lambda x: x["sha256"])
    N = min(int(per_lane), len(rows))
    if N != 256 or lane_ids != ["D", "B", "S", "O", "L"]:
        raise RuntimeError("forward lane contract mismatch")
    D, exact_meta = _farthest_exact(rows, N)
    B = sorted(rows, key=lambda x: (-x["parent_count_bounded"], -x["witness_count"], x["sha256"]))[:N]
    S = sorted(rows, key=lambda x: (len(set(x["record"][1])), max(Counter(x["record"][1]).values()), x["sha256"]))[:N]
    O = sorted(rows, key=lambda x: (-sum(v * (v - 1) // 2 for v in Counter(x["record"][1]).values()), -x["parent_count_bounded"], -x["witness_count"], x["sha256"]))[:N]
    line, seen = [], set()
    row_by_sha = {x["sha256"]: x for x in rows}
    for s in sources:
        hs = sorted(src_child[s["sha256"]])
        if hs and len(line) < N and hs[0] not in seen:
            line.append(row_by_sha[hs[0]]); seen.add(hs[0])
    for x in rows:
        if len(line) >= N: break
        if x["sha256"] not in seen:
            line.append(x); seen.add(x["sha256"])
    lanes = {"D": D, "B": B, "S": S, "O": O, "L": line}
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
        "L15_sources": len(sources), "attempts": attempts, "lawful": lawful,
        "global_unique_L16_candidates": len(rows),
        "multi_parent_candidates": sum(len(v) >= 2 for v in pars.values()),
        "max_bounded_parent_count": max(map(len, pars.values())),
    }
    relation = {
        "selected": [{"sha256": x["sha256"], "record": repr(x["record"]), "bounded_parent_count": x["parent_count_bounded"], "witness_count": x["witness_count"]} for x in selected],
        "edges": edges,
    }
    lane_seal = {
        "schema_id": "IG_L16_SCOUT_FORWARD_SEALED_LANES_V1_1",
        "selection_rules_sha256": SELECTION_RULES_SHA256,
        "per_lane": N, "lane_order": lane_ids,
        "lanes": {k: [x["sha256"] for x in lanes[k]] for k in lane_ids},
        "selected_union": [x["sha256"] for x in selected], "selected_union_count": len(selected),
        "D_lane_exactness": exact_meta,
        "sealed_before_relation_analysis": True, "no_outcome_feedback": True,
    }
    result = {
        "stage": "L16_FORWARD_PARENT_RESEAL_WIDE",
        "status": "SCOUT_COMPLETE", "evidence_label": "SCOUT_OBSERVED", "authoritative": False,
        "source_level": 15, "target_level": 16,
        "root_scope": "1089 sealed L15 scout objects; NOT complete L15 population",
        "generation": gen, "selection": {"per_lane": N, "union_selected": len(selected), "D_lane_exactness": exact_meta},
        "bounded_relation": br, "prohibitions": PROHIBITIONS,
    }
    return result, relation, lane_seal


def _closure_breakdown(relation: dict[str, Any]) -> dict[str, Any]:
    l2r, r2l = defaultdict(set), defaultdict(set)
    for a, b in relation["edges"]:
        l2r[a].add(b); r2l[b].add(a)
    right_pairs = set()
    for rs in l2r.values():
        s = sorted(rs)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                right_pairs.add((s[i], s[j]))
    fibers = {r: frozenset(r2l[r]) for r in r2l}
    unique = set(fibers.values())
    fiber_pairs = set()
    for a, b in right_pairs:
        A, B = fibers[a], fibers[b]
        if A != B:
            key = (A, B) if tuple(sorted(A)) < tuple(sorted(B)) else (B, A)
            fiber_pairs.add(key)
    classes = {"comparable": [], "incomparable": []}
    for A, B in fiber_pairs:
        key = "comparable" if A < B or B < A else "incomparable"
        classes[key].append(((A & B) in unique, (A | B) in unique))
    out = {}
    for key, vals in classes.items():
        n = len(vals); inter = sum(i for i, _ in vals); union = sum(u for _, u in vals); both = sum(i and u for i, u in vals)
        out[key] = {"pairs": n, "intersection_realized": inter, "union_realized": union, "both_realized": both,
                    "intersection_fraction": inter / n if n else 0.0, "union_fraction": union / n if n else 0.0, "both_fraction": both / n if n else 0.0}
    return out


def _forward_recognition(spec: dict[str, Any], breakdown: dict[str, Any]) -> list[str]:
    out = []
    if spec["order"]["strict_inclusion_pairs"]:
        out.append("NONTRIVIAL_PARENT_FIBER_INCLUSION_ORDER")
    if spec["closure"]["formal_pair_closure_exact_pair"] or spec["closure"]["formal_pair_closure_enlarged"]:
        out.append("NONTRIVIAL_FORMAL_CONCEPT_STYLE_PAIR_CLOSURE")
    inc = breakdown["incomparable"]
    if inc["intersection_realized"]:
        out.append("MEET_SKEWED_PARTIAL_CLOSURE_NONTRIVIAL_INTERSECTIONS_PRESENT")
    if inc["union_realized"]:
        out.append("NONTRIVIAL_UNIONS_EXTREMELY_SPARSE")
    if inc["both_realized"] == 0:
        out.append("NO_INCOMPARABLE_FIBER_PAIR_WITH_BOTH_MEET_AND_JOIN_REALIZED")
    if spec["incidence"]["abstract_cycle_rank"] > 0:
        out.append("CYCLE_RICH_ABSTRACT_INCIDENCE_GRAPH_RECOGNITION_ONLY")
    out.append("INCLUSION_POSET_WIDTH_COMPUTED")
    return out


def _build_baseline(wide: dict[str, Any], spec: dict[str, Any], l15: dict[str, Any]) -> dict[str, Any]:
    br, order, ref, scope = wide["bounded_relation"], spec["order"], spec["refinement"], spec["scope"]
    inc = spec["closure_breakdown"]["incomparable"]
    l15_rel = l15["relations"]
    l15_parent_max = l15_rel["R_PARENT_FIBER"]["metrics"]["selected_max_parent_count"]
    l15_inc_h = l15_rel["R_FIBER_INCLUSION"]["metrics"]["height"]
    l15_inter = l15_rel["R_PARTIAL_CLOSURE"]["metrics"]["intersection_realized_fraction"]
    l15_union = l15_rel["R_PARTIAL_CLOSURE"]["metrics"]["union_realized_fraction"]
    l15_right_frac = l15_rel["R_REFINEMENT_INDIVIDUATION"]["metrics"]["right_singletons"] / l15_rel["R_REFINEMENT_INDIVIDUATION"]["metrics"]["right_nodes"]
    right_frac = ref["right_singletons"] / scope["right_nodes"]
    relations = {
        "R_PARENT_FIBER": {"metrics": {"distinct_parent_fibers": br["distinct_exact_parent_fibers"], "selected_max_parent_count": br["selected_max_parent_count"], "selected_multi_parent": br["selected_multi_parent"], "selected_union": br["selected_union"], "multi_parent_fraction": br["selected_multi_parent"] / br["selected_union"]}, "comparison_to_L15": {"max_parent_count": {"L15": l15_parent_max, "L16": br["selected_max_parent_count"]}, "fraction_comparison": "NOT_PROMOTED_DIFFERENT_ROOT_SCOPE"}},
        "R_SHARED_SUPPORT": {"metrics": {"shared_parent_pair_count": br["shared_parent_pair_count"], "max_shared_parent_multiplicity": max(map(int, br["shared_parent_multiplicity_histogram"].keys()))}},
        "R_FIBER_INCLUSION": {"metrics": {"strict_inclusion_pairs": order["strict_inclusion_pairs"], "cover_relations": order["cover_relations"], "height": order["height"], "width": order["width"]}, "comparison_to_L15": {"height_L15": l15_inc_h, "height_L16": order["height"], "note": "Counts/rates not population-comparable because bounded root scopes differ."}},
        "R_PARTIAL_CLOSURE": {"metrics": {"incomparable_pairs": inc["pairs"], "incomparable_intersection_realized": inc["intersection_realized"], "incomparable_union_realized": inc["union_realized"], "incomparable_both_realized": inc["both_realized"], "incomparable_intersection_fraction": inc["intersection_fraction"], "incomparable_union_fraction": inc["union_fraction"], "aggregate_intersection_realized_fraction": spec["closure"]["intersection_realized_fraction"], "aggregate_union_realized_fraction": spec["closure"]["union_realized_fraction"]}, "comparison_to_L15": {"aggregate_intersection_fraction_L15": l15_inter, "aggregate_intersection_fraction_L16": spec["closure"]["intersection_realized_fraction"], "aggregate_union_fraction_L15": l15_union, "aggregate_union_fraction_L16": spec["closure"]["union_realized_fraction"], "note": "Directional comparison diagnostic only; bounded root scopes differ. Nontrivial closure claims use incomparable pairs."}},
        "R_REFINEMENT_INDIVIDUATION": {"metrics": {"left_nodes": scope["left_nodes"], "right_nodes": scope["right_nodes"], "left_singletons": ref["left_singletons"], "right_singletons": ref["right_singletons"], "right_singleton_fraction": right_frac, "stable_left_classes": ref["stable_left_classes"], "stable_right_classes": ref["stable_right_classes"]}, "comparison_to_L15": {"right_singleton_fraction_L15": l15_right_frac, "right_singleton_fraction_L16": right_frac, "note": "Diagnostic only; bounded root scopes differ."}},
        "R_LOCAL_DEPTH_REORGANIZATION": {"metrics": {}, "status": "UNRESOLVED_DEPTH_SCOPE"},
    }
    return {
        "schema_id": "IG_L16_SCOUT_FORWARD_BASELINE_V1_1",
        "stage": "L16_FORWARD_PARENT_BASELINE", "level": 16, "authoritative": False, "evidence_label": "SCOUT_OBSERVED",
        "relations": relations,
        "scope": {"candidate_count": wide["generation"]["global_unique_L16_candidates"], "relation_edges": br["relation_edges"], "selected_union": br["selected_union"], "root_scope": wide["root_scope"]},
        "spectroscope": {"components": spec["incidence"]["components"], "abstract_cycle_rank_recognition_only": spec["incidence"]["abstract_cycle_rank"], "recognition_summary": spec["recognition_summary"], "closure_breakdown": spec["closure_breakdown"]},
        "prohibitions": PROHIBITIONS,
    }


def _classification_from_predicates(baseline: dict[str, Any]) -> dict[str, Any]:
    r = baseline["relations"]
    predicates = {
        "R_PARENT_FIBER.PERSISTS": r["R_PARENT_FIBER"]["metrics"]["distinct_parent_fibers"] > 0,
        "R_PARENT_FIBER.EXPANDS": r["R_PARENT_FIBER"]["metrics"]["selected_max_parent_count"] > r["R_PARENT_FIBER"]["comparison_to_L15"]["max_parent_count"]["L15"],
        "R_SHARED_SUPPORT.PERSISTS": r["R_SHARED_SUPPORT"]["metrics"]["shared_parent_pair_count"] > 0,
        "R_FIBER_INCLUSION.PERSISTS": r["R_FIBER_INCLUSION"]["metrics"]["strict_inclusion_pairs"] > 0,
        "R_FIBER_INCLUSION.REORGANIZES": r["R_FIBER_INCLUSION"]["comparison_to_L15"]["height_L16"] != r["R_FIBER_INCLUSION"]["comparison_to_L15"]["height_L15"],
        "R_PARTIAL_CLOSURE.PERSISTS": (r["R_PARTIAL_CLOSURE"]["metrics"]["incomparable_intersection_realized"] + r["R_PARTIAL_CLOSURE"]["metrics"]["incomparable_union_realized"]) > 0,
        "R_PARTIAL_CLOSURE.REORGANIZES": (r["R_PARTIAL_CLOSURE"]["comparison_to_L15"]["aggregate_intersection_fraction_L16"] != r["R_PARTIAL_CLOSURE"]["comparison_to_L15"]["aggregate_intersection_fraction_L15"] or r["R_PARTIAL_CLOSURE"]["comparison_to_L15"]["aggregate_union_fraction_L16"] != r["R_PARTIAL_CLOSURE"]["comparison_to_L15"]["aggregate_union_fraction_L15"]),
        "R_REFINEMENT_INDIVIDUATION.PERSISTS": r["R_REFINEMENT_INDIVIDUATION"]["metrics"]["stable_right_classes"] > 0,
        "R_REFINEMENT_INDIVIDUATION.REORGANIZES": r["R_REFINEMENT_INDIVIDUATION"]["comparison_to_L15"]["right_singleton_fraction_L16"] != r["R_REFINEMENT_INDIVIDUATION"]["comparison_to_L15"]["right_singleton_fraction_L15"],
        "R_LOCAL_DEPTH_REORGANIZATION.UNRESOLVED_DEPTH_SCOPE": r["R_LOCAL_DEPTH_REORGANIZATION"]["status"] == "UNRESOLVED_DEPTH_SCOPE",
    }
    persistent = sorted({k.split('.')[0] for k, v in predicates.items() if v and k.endswith('.PERSISTS')})
    expanded = sorted({k.split('.')[0] for k, v in predicates.items() if v and k.endswith('.EXPANDS')})
    reorganized = sorted({k.split('.')[0] for k, v in predicates.items() if v and k.endswith('.REORGANIZES')})
    unresolved = sorted({k.split('.')[0] for k, v in predicates.items() if v and k.endswith('.UNRESOLVED_DEPTH_SCOPE')})
    parts = []
    if persistent: parts.append("PERSISTS")
    if expanded: parts.append("EXPANDS")
    if reorganized: parts.append("REORGANIZES")
    summary = " + ".join(parts) + " (bounded panel only)"
    return {
        "schema_id": "IG_L16_SCOUT_FORWARD_CLASSIFICATION_V1_1", "level": 16,
        "evidence_label": "SCOUT_OBSERVED", "authoritative": False,
        "classification_method": "EXPLICIT_METRIC_PREDICATES", "predicates": predicates,
        "persistent": persistent, "expanded": expanded, "reorganized_or_refined": reorganized, "unresolved": unresolved,
        "observed_or_new": [], "destroyed_or_relieved": [],
        "current_level_escalated_for_review": ["R_PARENT_FIBER"] if predicates["R_PARENT_FIBER.EXPANDS"] else [],
        "summary": summary, "prohibitions": PROHIBITIONS,
    }


def _science_projection_v2(obj: dict[str, Any]) -> dict[str, Any]:
    # Explicitly remove non-scientific cost and any cost-derived/self identity fields.
    return {k: v for k, v in obj.items() if k not in {"cost", "baseline_sha256", "runtime_seconds", "wall_seconds", "cpu_seconds"}}


@register_runner("adapter.scout_l16_forward.preflight")
def preflight_stage(*, controller, plan, stage, work_dir, run_dir):
    fixture = controller.datasets.materialize(stage["params"]["fixture_dataset_sha256"], work_dir / "fixture")
    result = _preflight(fixture, stage["params"])
    out = work_dir / "L16_FORWARD_PREFLIGHT_RESULT.json"; write_json_atomic(out, result)
    return {"outputs": {"L16_FORWARD_PREFLIGHT_RESULT.json": out}, "stage_result": result}


@register_runner("adapter.scout_l16_forward.wide")
def wide_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["selection_rules_sha256"] != SELECTION_RULES_SHA256:
        raise RuntimeError("forward selection-rules parameter mismatch")
    fixture = controller.datasets.materialize(stage["params"]["fixture_dataset_sha256"], work_dir / "fixture")
    wide, relation, lane_seal = _wide_exact(fixture, int(stage["params"]["per_lane"]), list(stage["params"]["lane_ids"]))
    outs = {}
    for name, obj in [("L16_FORWARD_WIDE_RESULT.json", wide), ("SELECTED_RELATION.json", relation), ("SEALED_LANE_SELECTION.json", lane_seal)]:
        p = work_dir / name; write_json_atomic(p, obj); outs[name] = p
    return {"outputs": outs, "stage_result": {"status": "PASS", "target": "L16_FORWARD_WIDE", "candidate_count": wide["generation"]["global_unique_L16_candidates"], "selected_union": wide["bounded_relation"]["selected_union"], "relation_edges": wide["bounded_relation"]["relation_edges"], "D_lane_exactness": wide["selection"]["D_lane_exactness"]}}


@register_runner("adapter.scout_l16_forward.spectroscope")
def spectroscope_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"].get("recognition_only") is not True or stage["params"].get("cannot_feedback") is not True:
        raise RuntimeError("Spectroscope non-feedback contract violated")
    relp = _dependency_artifact(controller, run_dir, "wide", "SELECTED_RELATION.json", work_dir / "inputs/SELECTED_RELATION.json")
    result = run_scout_spectroscope(relp, work_dir / "spectroscope", 16)
    result = {k: v for k, v in result.items() if k != "cost"}
    relation = json.loads(relp.read_text(encoding="utf-8"))
    result["closure_breakdown"] = _closure_breakdown(relation)
    result["recognition_summary"] = _forward_recognition(result, result["closure_breakdown"])
    result["test_id"] = "SCOUT_SPECTROSCOPE_L16_FORWARD_V1_1"
    out = work_dir / "SCOUT_SPECTROSCOPE_RESULT.json"; write_json_atomic(out, result)
    return {"outputs": {"SCOUT_SPECTROSCOPE_RESULT.json": out}, "stage_result": {"status": "PASS", "target": "L16_FORWARD_SPECTROSCOPE", "recognition_only": True, "authority": "RECONNAISSANCE_ONLY", "recognition_summary": result["recognition_summary"]}}


@register_runner("adapter.scout_l16_forward.classify")
def classify_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["observer_sha256"] != OBSERVER_SHA256 or stage["params"]["claim_boundary_sha256"] != CLAIM_BOUNDARY_SHA256:
        raise RuntimeError("forward observer/claim boundary mismatch")
    fixture = controller.datasets.materialize(stage["params"]["fixture_dataset_sha256"], work_dir / "fixture")
    pwide = _dependency_artifact(controller, run_dir, "wide", "L16_FORWARD_WIDE_RESULT.json", work_dir / "inputs/L16_FORWARD_WIDE_RESULT.json")
    prel = _dependency_artifact(controller, run_dir, "wide", "SELECTED_RELATION.json", work_dir / "inputs/SELECTED_RELATION.json")
    pspec = _dependency_artifact(controller, run_dir, "spectroscope", "SCOUT_SPECTROSCOPE_RESULT.json", work_dir / "inputs/SCOUT_SPECTROSCOPE_RESULT.json")
    wide, relation, spec = [json.loads(p.read_text(encoding="utf-8")) for p in (pwide, prel, pspec)]
    l15 = json.loads((fixture / "baseline/L15_SCOUT_BASELINE.json").read_text(encoding="utf-8"))
    baseline = _build_baseline(wide, spec, l15)
    classification = _classification_from_predicates(baseline)
    projections = {
        "schema_id": "IG_L16_SCOUT_SCIENCE_PROJECTIONS_V2", "projection_version": "L16_SCOUT_SCIENCE_PROJECTION_V2",
        "wide_science_sha256": canonical_sha256(_science_projection_v2(wide)),
        "selected_relation_canonical_sha256": canonical_sha256(relation),
        "spectroscope_science_sha256": canonical_sha256(_science_projection_v2(spec)),
        "baseline_science_sha256": canonical_sha256(_science_projection_v2(baseline)),
        "classification_canonical_sha256": canonical_sha256(classification),
        "timing_independent": True, "evidence_label": "SCOUT_OBSERVED", "authority": "RECONNAISSANCE_ONLY", "population_complete": False,
    }
    outs = {}
    for name, obj in [("L16_FORWARD_BASELINE.json", baseline), ("L16_FORWARD_CLASSIFICATION.json", classification), ("L16_SCIENCE_PROJECTIONS_V2.json", projections)]:
        p = work_dir / name; write_json_atomic(p, obj); outs[name] = p
    return {"outputs": outs, "stage_result": {"status": "PASS", "target": "L16_FORWARD_CLASSIFY", "summary": classification["summary"], "classification_method": classification["classification_method"], "authority": "RECONNAISSANCE_ONLY"}}


@register_runner("adapter.scout_l16_forward.verify")
def verify_stage(*, controller, plan, stage, work_dir, run_dir):
    if stage["params"]["audit_reference_sha256"] != AUDIT_REFERENCE_SHA256:
        raise RuntimeError("audit reference parameter mismatch")
    ref = _resource_json("L16_FORWARD_AUDIT_REFERENCE_v1.1.json")
    if _resource_sha("L16_FORWARD_AUDIT_REFERENCE_v1.1.json", "reference_sha256") != AUDIT_REFERENCE_SHA256:
        raise RuntimeError("audit reference packaged identity mismatch")
    prel = _dependency_artifact(controller, run_dir, "wide", "SELECTED_RELATION.json", work_dir / "inputs/SELECTED_RELATION.json")
    pwide = _dependency_artifact(controller, run_dir, "wide", "L16_FORWARD_WIDE_RESULT.json", work_dir / "inputs/L16_FORWARD_WIDE_RESULT.json")
    pspec = _dependency_artifact(controller, run_dir, "spectroscope", "SCOUT_SPECTROSCOPE_RESULT.json", work_dir / "inputs/SCOUT_SPECTROSCOPE_RESULT.json")
    pcl = _dependency_artifact(controller, run_dir, "classify", "L16_FORWARD_CLASSIFICATION.json", work_dir / "inputs/L16_FORWARD_CLASSIFICATION.json")
    pproj = _dependency_artifact(controller, run_dir, "classify", "L16_SCIENCE_PROJECTIONS_V2.json", work_dir / "inputs/L16_SCIENCE_PROJECTIONS_V2.json")
    relation, wide, spec, classification, projections = [json.loads(p.read_text(encoding="utf-8")) for p in (prel, pwide, pspec, pcl, pproj)]
    em = ref["expected_exact_union_metrics"]
    metric_cmp = {k: wide["bounded_relation"].get(k) == v for k, v in em.items()}
    sref = ref["expected_spectroscope_core"]
    spec_cmp = {
        "scope": spec["scope"] == sref["scope"],
        "incidence": all(spec["incidence"].get(k) == v for k, v in sref["incidence_summary"].items()),
        "order": all(spec["order"].get(k) == v for k, v in sref["order_summary"].items()),
        "refinement": all(spec["refinement"].get(k) == v for k, v in sref["refinement_summary"].items()),
        "closure": all(spec["closure"].get(k) == v for k, v in sref["closure_summary"].items()),
        "incomparable_closure": all(spec["closure_breakdown"]["incomparable"].get(k) == v for k, v in sref["incomparable_closure"].items()),
    }
    comparisons = {
        "selected_relation_exact_audit_reference": canonical_sha256(relation) == ref["exact_counterfactual_selected_relation_canonical_sha256"],
        "exact_union_metrics": all(metric_cmp.values()),
        "spectroscope_core": all(spec_cmp.values()),
        "predicate_summary": classification["summary"] == ref["expected_summary"],
        "projection_v2": projections["projection_version"] == "L16_SCOUT_SCIENCE_PROJECTION_V2" and projections["timing_independent"] is True,
        "no_historical_golden_equality_claim": True,
    }
    status = "PASS" if all(comparisons.values()) else "FAIL"
    verification = {"schema_id": "IG_L16_FORWARD_PARENT_TERMINAL_VERIFICATION_V1_1", "status": status, "audit_reference_sha256": AUDIT_REFERENCE_SHA256, "source_pro_audit_science_sha256": PRO_AUDIT_SCIENCE_SHA256, "comparisons": comparisons, "metric_comparisons": metric_cmp, "spectroscope_comparisons": spec_cmp, "reference_role": ref["reference_use"]}
    principal = {
        "schema_id": "IG_L16_FORWARD_PARENT_PRINCIPAL_RESULT_V1_1", "status": status, "target": "L16_EXACT_FORWARD_PARENT_RESEAL", "level": 16, "source_level": 15,
        "execution_class": "VERSIONED_FORWARD_PARENT_RESEAL_FROM_FROZEN_BOUNDED_L15_INPUTS", "evidence_label": "SCOUT_OBSERVED", "authority": "RECONNAISSANCE_ONLY", "authoritative": False,
        "scope": "BOUNDED_PREREGISTERED", "population_complete": False, "root_count": ROOT_COUNT, "root_sha256": ROOT_SHA256, "primitive_sha256": PRIMITIVE_SHA256,
        "selection_contract": {"arithmetic": "EXACT_INTEGER_RATIONAL_EQUIVALENT", "initial_seed": "MINIMUM_SHA256", "tie_direction_after_seed": "MAXIMUM_SHA256", "binary_float_in_ordering_key": False},
        "verification": comparisons, "science_projection_hashes": projections, "classification": classification, "summary": classification["summary"],
        "closure_interpretation": spec["recognition_summary"], "prohibitions": PROHIBITIONS,
        "claims_licensed": ["bounded persistence of tracked relation families", "bounded parent-fiber expansion to observed maximum 16", "bounded reorganization under the exact-D forward panel", "eligibility as bounded L16 Scout parent input after closeout"],
        "claims_forbidden": ["first occurrence", "population-wide absence", "mechanism", "geometry", "physical topology", "transition level", "graduation", "lattice or closed-algebra promotion"],
        "deep_maturation": "UNRESOLVED_DEPTH_SCOPE",
    }
    pv = work_dir / "L16_FORWARD_TERMINAL_VERIFICATION.json"; pp = work_dir / "L16_FORWARD_PARENT_PRINCIPAL_RESULT.json"
    write_json_atomic(pv, verification); write_json_atomic(pp, principal)
    if status != "PASS":
        raise RuntimeError(f"L16 forward-parent audit-reference mismatch: {comparisons}")
    return {"outputs": {"L16_FORWARD_TERMINAL_VERIFICATION.json": pv, "L16_FORWARD_PARENT_PRINCIPAL_RESULT.json": pp}, "stage_result": {"status": "PASS", "target": "L16_FORWARD_PARENT_VERIFY", "comparisons": comparisons, "authority": "RECONNAISSANCE_ONLY"}}
