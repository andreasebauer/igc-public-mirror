from __future__ import annotations

"""Execution-only parallel helpers for O-regime maturation.

This module deliberately owns no scientific semantics.  It evaluates the exact frozen
regime-scanner functions on independent candidate/state tasks and returns canonical pure-data
results.  Completion order and worker count are erased before any scientific hash is computed.
"""

from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any, Mapping
import math
import statistics
import json
import tempfile
import multiprocessing as _mp
import threading
from pathlib import Path

from .canon import canonical_sha256
from .execution import ExecutionPolicy, TaskSpec, execute_tasks
from . import regime_scanner as rs

_CANDIDATE_CONTEXT: dict[str, Any] | None = None


def _jsonify_tuple_tree(x: Any) -> Any:
    if isinstance(x, tuple):
        return [_jsonify_tuple_tree(v) for v in x]
    if isinstance(x, list):
        return [_jsonify_tuple_tree(v) for v in x]
    return x


def _tupleify_tree(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_tupleify_tree(v) for v in x)
    return x


def state_dag_wire(roots: list[Any]) -> dict[str, Any]:
    """Serialize exact selected states as a content-addressed DAG for spawn/forkserver workers.

    O7 leaves are identified by immutable authority source_id plus reservation counters; higher
    lifts are represented by child digests and their exact top-edge witnesses.  Shared children
    are emitted once, so deep maturation does not expand repeated subtrees exponentially.
    """
    nodes: dict[str, Any] = {}

    def visit(s: Any) -> str:
        digest = str(s.construction_digest)
        if digest in nodes:
            return digest
        if isinstance(s, rs.O7State):
            row = {
                "kind": "O7",
                "construction_digest": digest,
                "source_id": str(s.source_id),
                "reserve_counts": list(map(int, s.reserve_counts)),
                "lane": str(s.lane),
            }
        elif isinstance(s, rs.LiftState):
            child_ids = [visit(c) for c in s.children]
            row = {
                "kind": "LIFT",
                "construction_digest": digest,
                "level": int(s.level),
                "children": child_ids,
                "top_edges_full": _jsonify_tuple_tree(s.top_edges_full),
                "lane": str(s.lane),
                "motif_id": str(s.motif_id),
            }
        else:
            raise TypeError(f"unsupported maturation state type for DAG serialization: {type(s).__name__}")
        nodes[digest] = row
        return digest

    root_ids = [visit(s) for s in roots]
    out = {"schema_id": "IG_MATURATION_STATE_DAG_V1", "roots": root_ids, "nodes": nodes}
    out["science_sha256"] = canonical_sha256(out)
    return out


def _states_from_dag(obj: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    raw = dict(obj)
    expected = raw.pop("science_sha256", None)
    if expected != canonical_sha256(raw):
        raise RuntimeError("maturation state DAG identity mismatch")
    from .materialized_discovery import _get_process_o7_runtime
    spec = rs.load_regime_scanner_spec()
    runtime = _get_process_o7_runtime(spec)
    engine = runtime["engine"]
    base_by_source = {str(s.source_id): s for s in runtime["base_states"]}
    nodes = dict(obj["nodes"])
    memo: dict[str, Any] = {}

    def build(digest: str) -> Any:
        if digest in memo:
            return memo[digest]
        row = nodes[digest]
        kind = row["kind"]
        if kind == "O7":
            source_id = str(row["source_id"])
            if source_id not in base_by_source:
                raise RuntimeError(f"O7 authority source_id not found during worker reconstruction: {source_id}")
            base = base_by_source[source_id]
            state = rs.O7State(
                engine, base.ctx, base.edges,
                tuple(map(int, row["reserve_counts"])), source_id, str(row["lane"]),
            )
        elif kind == "LIFT":
            children = tuple(build(str(x)) for x in row["children"])
            state = rs.LiftState(
                engine, int(row["level"]), children,
                _tupleify_tree(row["top_edges_full"]), str(row["lane"]), str(row["motif_id"]),
            )
        else:
            raise RuntimeError(f"unknown maturation DAG node kind: {kind}")
        if str(state.construction_digest) != digest:
            raise RuntimeError(f"maturation DAG reconstruction digest mismatch: {digest}")
        memo[digest] = state
        return state

    roots = [build(str(x)) for x in obj["roots"]]
    return engine, roots


def _candidate_initializer(payload: Mapping[str, Any]) -> None:
    path = Path(str(payload["context_path"]))
    data = path.read_bytes()
    if canonical_sha256(json.loads(data)) != str(payload["context_sha256"]):
        raise RuntimeError("candidate context file hash mismatch")
    context = json.loads(data)
    engine, prev = _states_from_dag(context["state_dag"])
    center_digest = str(context["center_digest"])
    center = next((s for s in prev if str(s.construction_digest) == center_digest), None)
    if center is None:
        raise RuntimeError("candidate context center missing from reconstructed roots")
    install_candidate_context(
        engine=engine,
        prev=prev,
        level=int(context["level"]),
        pairs=[tuple(map(int, x)) for x in context["pairs"]],
        motifs=[],
        center=center,
    )


def state_wire(state: Any) -> dict[str, Any]:
    return {
        "construction_digest": str(state.construction_digest),
        "skin": str(state.skin),
        "lane": str(getattr(state, "lane", "")),
        "motif_id": str(getattr(state, "motif_id", "")),
        "owner_caps": [list(map(int, x)) for x in state.owner_caps],
        "total_caps": list(map(int, state.total_caps)),
        "top_pairs": [list(map(int, x)) for x in state.top_pairs],
        "typed_edges": [list(map(int, x)) for x in state.typed_edges],
        "leaf_count": int(state.leaf_count),
        "relation_count_total": int(state.relation_count_total),
    }


def _wire_state(w: Mapping[str, Any]) -> Any:
    return SimpleNamespace(
        construction_digest=str(w["construction_digest"]),
        skin=str(w["skin"]),
        lane=str(w.get("lane", "")),
        motif_id=str(w.get("motif_id", "")),
        owner_caps=[tuple(map(int, x)) for x in w["owner_caps"]],
        total_caps=tuple(map(int, w["total_caps"]),),
        top_pairs=[tuple(map(int, x)) for x in w["top_pairs"]],
        typed_edges=[tuple(map(int, x)) for x in w["typed_edges"]],
        leaf_count=int(w["leaf_count"]),
        relation_count_total=int(w["relation_count_total"]),
    )


def cohort_structural_seed_from_wire(w: Mapping[str, Any]) -> dict[str, Any]:
    s = _wire_state(w)
    graph = rs._graph_basic(len(s.owner_caps), s.top_pairs)
    return {
        "motif_id": s.motif_id,
        "lane": s.lane,
        "owner_count": len(s.owner_caps),
        "top_relation_count": len(s.top_pairs),
        "top_pairs": [list(map(int, e)) for e in s.top_pairs],
        "typed_top_edges": [list(map(int, e)) for e in s.typed_edges],
        "uncolored_topology_sha256": rs._sha(rs._uncolored_graph_canon(len(s.owner_caps), s.top_pairs)),
        "colored_owner_resource_topology_sha256": canonical_sha256(
            rs._colored_typed_canon(
                len(s.owner_caps),
                [tuple(int(x > 0) for x in c) for c in s.owner_caps],
                list(s.typed_edges),
            )
        ),
        "graph": {
            "degree": list(map(int, graph["degree"])),
            "diameter": int(graph["diameter"]),
            "radius": int(graph["radius"]),
            "articulations": int(graph["articulations"]),
            "bridges": int(graph["bridges"]),
            "triangles": int(graph["triangles"]),
            "cycle_rank": int(graph["beta"]),
        },
    }


def _selected_state_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = _wire_state(payload["state"])
    bridge_pairs = [tuple(map(int, x)) for x in payload["bridge_pairs"]]
    action = rs._action_aggregate(s, bridge_pairs)
    graph = rs._graph_basic(len(s.owner_caps), s.top_pairs)
    fiber = rs._factor_fiber(s)
    org = rs._state_organizational_signature(s, action)
    return {
        "construction_digest": s.construction_digest,
        "action": action,
        "graph": graph,
        "fiber": sorted(fiber),
        "organizational_sha256": rs._sha(org),
        "topology_canon": rs._uncolored_graph_canon(len(s.owner_caps), s.top_pairs),
    }


def analyze_selected_states(
    states: list[Any],
    bridge_pairs: list[tuple[int, int]],
    *,
    policy: ExecutionPolicy,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    tasks: list[TaskSpec] = []
    for s in sorted(states, key=lambda x: x.construction_digest):
        wire = state_wire(s)
        binding = canonical_sha256({"op": "O_REGIME_SELECTED_STATE_ANALYSIS_V1", "state": wire, "bridge_pairs": bridge_pairs})
        tasks.append(TaskSpec(
            task_id=f"state-{s.construction_digest[:24]}",
            task_kind="O_REGIME_SELECTED_STATE_ANALYSIS_V1",
            binding_sha256=binding,
            payload={"state": wire, "bridge_pairs": [list(x) for x in bridge_pairs]},
            cost_weight=max(1.0, float(len(s.owner_caps) + len(s.top_pairs))),
        ))
    batch = execute_tasks(
        tasks,
        worker_ref="infinity_grid.maturation_parallel:_selected_state_worker",
        policy=policy,
    )
    by_digest = {str(v["construction_digest"]): v for v in batch.results.values()}
    return by_digest, dict(batch.metadata)


def scan_level_from_analysis(
    states: list[Any],
    level: int,
    bridge_pairs: list[tuple[int, int]],
    grammar_hash: str,
    analysis: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Reassemble the frozen ``regime_scanner._scan_level`` output exactly from independent state analyses."""
    states = sorted(states, key=lambda s: s.construction_digest)
    rows = [analysis[s.construction_digest] for s in states]
    actions = [dict(r["action"]) for r in rows]
    gms = [dict(r["graph"]) for r in rows]
    fibers = [set(r["fiber"]) for r in rows]
    overlap = rs._panel_overlap(fibers)
    byskin: dict[str, list[Any]] = defaultdict(list)
    for s in states:
        byskin[s.skin].append(s)
    same_skin = [v for v in byskin.values() if len(v) > 1]
    topo_collision = 0
    for vals in same_skin:
        if len({rs._uncolored_graph_canon(len(s.owner_caps), s.top_pairs) for s in vals}) > 1:
            topo_collision += 1
    endpoint_support = [sum(x > 0 for x in s.total_caps) for s in states]
    min_caps = [min(s.total_caps[t] for s in states) for t in range(7)]
    org_hashes = sorted(str(r["organizational_sha256"]) for r in rows)
    raw = {
        "level": level,
        "states": len(states),
        "grammar_sha256": grammar_hash,
        "diversity": {
            "construction_states": len({s.construction_digest for s in states}),
            "resource_skins": len(byskin),
            "same_skin_groups": len(same_skin),
            "same_skin_topology_collision_groups": topo_collision,
            "topology_classes": len({rs._uncolored_graph_canon(len(s.owner_caps), s.top_pairs) for s in states}),
            "organizational_classes": len(set(org_hashes)),
            "owner_count_hist": dict(sorted(Counter(len(s.owner_caps) for s in states).items())),
            "edge_count_hist": dict(sorted(Counter(len(s.top_pairs) for s in states).items())),
            "endpoint_support_min": min(endpoint_support),
            "endpoint_support_max": max(endpoint_support),
            "min_total_free_by_type": min_caps,
            "leaf_count_min": min(s.leaf_count for s in states),
            "leaf_count_median": rs._median([s.leaf_count for s in states]),
            "leaf_count_max": max(s.leaf_count for s in states),
            "relation_count_total_median": rs._median([s.relation_count_total for s in states]),
            "relation_count_total_max": max(s.relation_count_total for s in states),
        },
        "branching": {
            "legal_action_labels_median": rs._median([a["legal_action_labels"] for a in actions]),
            "legal_action_labels_max": max(a["legal_action_labels"] for a in actions),
            "action_orbits_median": rs._median([a["action_orbits"] for a in actions]),
            "action_orbits_max": max(a["action_orbits"] for a in actions),
            "type_pair_support_min": min(a["type_pair_support"] for a in actions),
            "type_pair_support_max": max(a["type_pair_support"] for a in actions),
            "total_action_copies_log10_median": rs._median([math.log10(max(1, a["total_action_copies"])) for a in actions]),
            "total_action_copies_log10_p90": rs._p([math.log10(max(1, a["total_action_copies"])) for a in actions], .9),
            "service_classes_median": rs._median([a["service_classes"] for a in actions]),
            "service_classes_max": max(a["service_classes"] for a in actions),
        },
        "symmetry": {
            "automorphism_size_median": rs._median([a["automorphism_size"] for a in actions]),
            "automorphism_size_max": max(a["automorphism_size"] for a in actions),
            "owner_orbits_median": rs._median([a["owner_orbits"] for a in actions]),
        },
        "overlap_gluing": overlap,
        "lineage": {
            "factor_fiber_median": rs._median([len(f) for f in fibers]),
            "factor_fiber_max": max(map(len, fibers)),
            "bridge_fraction_median": rs._median([g["bridges"] / max(1, len(s.top_pairs)) for g, s in zip(gms, states)]),
            "cycle_rank_median": rs._median([g["beta"] for g in gms]),
        },
        "quotient_observer": {
            "same_skin_topology_hidden_present": topo_collision > 0,
            "resource_future_equivalence_basis": "INHERITED_GRRL_THEOREM_NOT_RECOMPUTED_FULL_COUNTER_PROFILE",
            "organizational_to_skin_class_ratio": len(set(org_hashes)) / max(1, len(byskin)),
        },
        "topology_services": {
            "diameter_hist": dict(sorted(Counter(g["diameter"] for g in gms).items())),
            "articulation_hist": dict(sorted(Counter(g["articulations"] for g in gms).items())),
            "beta_hist": dict(sorted(Counter(g["beta"] for g in gms).items())),
            "service_signature_classes": len({a["service_signature_sha256"] for a in actions}),
            "access_bottleneck_service_present": any(a["service_classes"] > 1 for a in actions),
        },
        "obstruction_relief": {
            "all_bridge_types_supported_everywhere": all(a["type_pair_support"] == len(set(bridge_pairs)) for a in actions),
            "all_owner_pairs_have_some_action": all(a["owner_pair_support"] == len(s.owner_caps) * (len(s.owner_caps) - 1) // 2 for a, s in zip(actions, states)),
            "zero_action_states": sum(a["legal_action_labels"] == 0 for a in actions),
        },
        "raw_growth": {
            "median_total_free": rs._median([sum(s.total_caps) for s in states]),
            "median_leaf_count": rs._median([s.leaf_count for s in states]),
            "median_total_relations": rs._median([s.relation_count_total for s in states]),
        },
    }
    normalized = {
        "grammar": grammar_hash,
        "owner_count_hist": raw["diversity"]["owner_count_hist"],
        "edge_count_hist": raw["diversity"]["edge_count_hist"],
        "topology_classes": raw["diversity"]["topology_classes"],
        "organizational_classes": raw["diversity"]["organizational_classes"],
        "endpoint_support_min": raw["diversity"]["endpoint_support_min"],
        "endpoint_support_max": raw["diversity"]["endpoint_support_max"],
        "branching": {k: raw["branching"][k] for k in ["legal_action_labels_median", "legal_action_labels_max", "action_orbits_median", "action_orbits_max", "type_pair_support_min", "type_pair_support_max", "service_classes_median", "service_classes_max"]},
        "symmetry": raw["symmetry"],
        "overlap_gluing": {k: raw["overlap_gluing"][k] for k in ["shared_fiber_pairs", "strict_inclusions", "intersection_realized_fraction", "union_realized_fraction"]},
        "lineage": {k: raw["lineage"][k] for k in ["factor_fiber_median", "factor_fiber_max", "bridge_fraction_median", "cycle_rank_median"]},
        "quotient": {k: raw["quotient_observer"][k] for k in ["same_skin_topology_hidden_present", "organizational_to_skin_class_ratio"]},
        "topology_services": raw["topology_services"],
        "obstruction_relief": raw["obstruction_relief"],
    }
    nh = rs._sha(normalized)
    raw["normalized_signature"] = normalized
    raw["normalized_signature_sha256"] = nh
    return raw, nh


def install_candidate_context(*, engine: Any, prev: list[Any], level: int, pairs: list[tuple[int, int]], motifs: list[dict[str, Any]], center: Any) -> None:
    global _CANDIDATE_CONTEXT
    _CANDIDATE_CONTEXT = {
        "engine": engine,
        "prev": prev,
        "level": int(level),
        "pairs": pairs,
        "motifs": motifs,
        "center": center,
    }


def clear_candidate_context() -> None:
    global _CANDIDATE_CONTEXT
    _CANDIDATE_CONTEXT = None


def _build_recipe_state(recipe: Mapping[str, Any]) -> Any:
    if _CANDIDATE_CONTEXT is None:
        raise RuntimeError("parallel candidate context unavailable; fork-inherited context required")
    c = _CANDIDATE_CONTEXT
    engine = c["engine"]
    prev = c["prev"]
    center = c["center"]
    level = int(c["level"])
    pairs = c["pairs"]
    lane = str(recipe["lane"])
    motif_id = str(recipe["motif_id"])
    edges = [tuple(map(int, e)) for e in recipe["edges"]]
    n = int(recipe["n"])
    if lane in {"HOM", "TWIN"}:
        owners = [center] * n
    elif lane == "HET":
        owners = [prev[j % len(prev)] for j in range(n)]
    elif lane == "MIX":
        owners = [prev[0], prev[1], prev[0], prev[1]]
    else:
        raise RuntimeError(f"unknown candidate lane {lane}")
    force_pair = None if recipe.get("force_pair") is None else tuple(map(int, recipe["force_pair"]))
    return rs._build_lift(
        engine, level, owners, edges, pairs, lane, motif_id,
        schedule_seed=int(recipe.get("schedule_seed", 0)), force_pair=force_pair,
    )


def _candidate_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    recipe = dict(payload["recipe"])
    state = _build_recipe_state(recipe)
    if state is None:
        return {"success": False, "recipe": recipe}
    wire = state_wire(state)
    seed = cohort_structural_seed_from_wire(wire)
    return {
        "success": True,
        "recipe": recipe,
        "construction_digest": str(state.construction_digest),
        "motif_id": str(state.motif_id),
        "lane": str(state.lane),
        # Execution-only exact public capacity vector.  This lets higher-stage infrastructure
        # select the same frozen R100 growth seed from descriptors before rematerializing only
        # the winning exact state.  It does not alter candidate selection or science hashes.
        "total_caps": [int(x) for x in state.total_caps],
        "pre_features": [float(x) for x in rs._state_pre_features(state)],
        "structural_seed_sha256": canonical_sha256(seed),
    }


def enumerate_candidate_recipes(motifs: list[dict[str, Any]], *, center_supports_twin: bool, pairs_count: int) -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    for mi, m in enumerate(motifs):
        n = int(m["n"])
        edges = [list(map(int, e)) for e in m["edges"]]
        recipes.append({"lane": "HOM", "motif_id": f"HOM:{n}:{mi}", "n": n, "edges": edges, "schedule_seed": mi % pairs_count, "force_pair": None})
        if n in (4, 5) or (n == 6 and mi % 6 == 0):
            recipes.append({"lane": "HET", "motif_id": f"HET:{n}:{mi}", "n": n, "edges": edges, "schedule_seed": (mi * 3 + 1) % pairs_count, "force_pair": None})
        if n == 4:
            recipes.append({"lane": "MIX", "motif_id": f"MIX:4:{mi}", "n": 4, "edges": edges, "schedule_seed": (mi * 5 + 2) % pairs_count, "force_pair": None})
    if center_supports_twin:
        recipes.extend([
            {"lane": "TWIN", "motif_id": "TWIN:A", "n": 6, "edges": [list(map(int, e)) for e in rs.GA], "schedule_seed": 0, "force_pair": [0, 0]},
            {"lane": "TWIN", "motif_id": "TWIN:B", "n": 6, "edges": [list(map(int, e)) for e in rs.GB], "schedule_seed": 0, "force_pair": [0, 0]},
        ])
    return recipes


def _farthest_select_descriptors(rows: list[dict[str, Any]], k: int, *, must_include_motif_ids: set[str]) -> list[dict[str, Any]]:
    uniq = {str(r["construction_digest"]): r for r in rows}
    pool = [uniq[x] for x in sorted(uniq)]
    if len(pool) <= k:
        return pool
    vec = [list(map(float, r["pre_features"])) for r in pool]
    dims = len(vec[0])
    mins = [min(v[j] for v in vec) for j in range(dims)]
    maxs = [max(v[j] for v in vec) for j in range(dims)]
    norm = [[(v[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] > mins[j] else 0.0 for j in range(dims)] for v in vec]
    chosen: list[int] = []
    # Frozen scanner passes topology twins as [TWIN:A, TWIN:B]; preserve that exact
    # must-include order before the farthest-point recurrence.
    ordered_must = ["TWIN:A", "TWIN:B"] + sorted(x for x in must_include_motif_ids if x not in {"TWIN:A", "TWIN:B"})
    for motif_id in ordered_must:
        for i, r in enumerate(pool):
            if str(r["motif_id"]) == motif_id and i not in chosen:
                chosen.append(i)
                break
    if not chosen:
        chosen = [0]
    while len(chosen) < k:
        best = None
        for i in range(len(pool)):
            if i in chosen:
                continue
            dmin = min(sum((norm[i][j] - norm[c][j]) ** 2 for j in range(dims)) ** 0.5 for c in chosen)
            if best is None or dmin > best[0] + 1e-15 or (abs(dmin - best[0]) <= 1e-15 and pool[i]["construction_digest"] < pool[best[1]]["construction_digest"]):
                best = (dmin, i)
        chosen.append(best[1])
    return [pool[i] for i in chosen[:k]]


def parallel_candidate_descriptors(
    *,
    engine: Any,
    prev: list[Any],
    level: int,
    pairs: list[tuple[int, int]],
    motifs: list[dict[str, Any]],
    spec: Mapping[str, Any],
    policy: ExecutionPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any]:
    prev = sorted(prev, key=lambda s: s.construction_digest)
    totals = [sum(s.total_caps) for s in prev]
    med = statistics.median(totals)
    center = min(prev, key=lambda s: (abs(sum(s.total_caps) - med), s.construction_digest))
    recipes = enumerate_candidate_recipes(motifs, center_supports_twin=(center.total_caps[0] >= 6), pairs_count=len(pairs))
    # v0.28.6 performance repair: the AUTO policy intentionally selects SERIAL candidate
    # construction for the dynamic O7 lifecycle.  In that mode the parent process already owns
    # the exact engine and ``prev`` state objects, so serializing the complete recursive DAG to
    # JSON and immediately reconstructing it in the *same process* is pure transport overhead.
    # It grows with depth and dominated O7->O69 process-warm replay.  Use the in-process exact
    # objects directly for SERIAL/AUTO while retaining the content-addressed DAG transport for
    # explicit process-pool policies.  Candidate bindings still include a canonical context
    # identity derived from the exact root construction digests and frozen controls.
    start_method = str(policy.start_method).lower()
    fork_inherited = policy.backend.upper() != "SERIAL" and start_method == "fork"
    serial_auto = policy.backend.upper() == "SERIAL" or (
        policy.backend.upper() != "SERIAL" and str(policy.start_method).upper() == "AUTO"
    )
    use_policy = policy
    fallback_reason = None
    if fork_inherited:
        # v0.30.32: process-pool acceleration without recursive DAG transport.  On Linux
        # fork workers inherit the immutable exact engine/prev-state context copy-on-write.
        # This preserves the same pure recipe evaluator while avoiding serialization and
        # reconstruction of the recursive state DAG at every maturation depth.
        context_identity = {
            "schema_id": "IG_MATURATION_CANDIDATE_CONTEXT_IDENTITY_V2",
            "level": int(level),
            "pairs": [list(x) for x in pairs],
            "center_digest": str(center.construction_digest),
            "prev_root_digests": [str(s.construction_digest) for s in prev],
        }
        ctx_sha = canonical_sha256(context_identity)
        tasks = [TaskSpec(
            task_id=f"candidate-{i:04d}",
            task_kind="O_REGIME_CANDIDATE_BUILD_V1",
            binding_sha256=canonical_sha256({"context": ctx_sha, "recipe": recipe}),
            payload={"recipe": recipe},
            cost_weight=max(1.0, float(int(recipe["n"]) + len(recipe["edges"]))),
        ) for i, recipe in enumerate(recipes)]
        install_candidate_context(
            engine=engine, prev=prev, level=int(level), pairs=pairs, motifs=motifs, center=center
        )
        try:
            batch = execute_tasks(
                tasks,
                worker_ref="infinity_grid.maturation_parallel:_candidate_worker",
                policy=use_policy,
            )
        finally:
            clear_candidate_context()
        transport_mode = "FORK_INHERITED_EXACT_STATE_REFERENCE_V1"
    elif serial_auto:
        if policy.backend.upper() != "SERIAL":
            use_policy = ExecutionPolicy(**{**policy.__dict__, "backend": "SERIAL", "requested_workers": 1})
            fallback_reason = "FORK_REQUIRES_SINGLE_THREADED_PARENT"
        context_identity = {
            "schema_id": "IG_MATURATION_CANDIDATE_CONTEXT_IDENTITY_V2",
            "level": int(level),
            "pairs": [list(x) for x in pairs],
            "center_digest": str(center.construction_digest),
            "prev_root_digests": [str(s.construction_digest) for s in prev],
        }
        ctx_sha = canonical_sha256(context_identity)
        tasks = [TaskSpec(
            task_id=f"candidate-{i:04d}",
            task_kind="O_REGIME_CANDIDATE_BUILD_V1",
            binding_sha256=canonical_sha256({"context": ctx_sha, "recipe": recipe}),
            payload={"recipe": recipe},
            cost_weight=max(1.0, float(int(recipe["n"]) + len(recipe["edges"]))),
        ) for i, recipe in enumerate(recipes)]
        install_candidate_context(
            engine=engine, prev=prev, level=int(level), pairs=pairs, motifs=motifs, center=center
        )
        try:
            batch = execute_tasks(
                tasks,
                worker_ref="infinity_grid.maturation_parallel:_candidate_worker",
                policy=use_policy,
            )
        finally:
            clear_candidate_context()
        transport_mode = "IN_PROCESS_EXACT_STATE_REFERENCE_SERIAL_V2"
    else:
        context = {
            "schema_id": "IG_MATURATION_CANDIDATE_CONTEXT_V1",
            "level": int(level),
            "pairs": [list(x) for x in pairs],
            "center_digest": str(center.construction_digest),
            "state_dag": state_dag_wire(prev),
        }
        ctx_sha = canonical_sha256(context)
        with tempfile.TemporaryDirectory(prefix=f"ig-maturation-O{int(level)}-") as td:
            context_path = Path(td) / "candidate_context.json"
            context_path.write_text(json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            tasks = [TaskSpec(
                task_id=f"candidate-{i:04d}",
                task_kind="O_REGIME_CANDIDATE_BUILD_V1",
                binding_sha256=canonical_sha256({"context": ctx_sha, "recipe": recipe}),
                payload={"recipe": recipe},
                cost_weight=max(1.0, float(int(recipe["n"]) + len(recipe["edges"]))),
            ) for i, recipe in enumerate(recipes)]
            try:
                batch = execute_tasks(
                    tasks,
                    worker_ref="infinity_grid.maturation_parallel:_candidate_worker",
                    policy=use_policy,
                    initializer_ref="infinity_grid.maturation_parallel:_candidate_initializer",
                    initializer_payload={"context_path": str(context_path), "context_sha256": ctx_sha},
                )
            finally:
                clear_candidate_context()
        transport_mode = "CONTENT_ADDRESSED_STATE_DAG_RECONSTRUCTION"
    successful = [dict(x) for x in batch.results.values() if x.get("success")]
    failures = len(recipes) - len(successful)
    twins = [x for x in successful if x["motif_id"] in {"TWIN:A", "TWIN:B"}]
    cap = int(spec["panel"]["candidate_cap"])
    candidates = list(successful)
    if len(candidates) > cap:
        candidates = sorted(candidates, key=lambda r: r["construction_digest"])[:cap]
        for t in twins:
            if all(x["construction_digest"] != t["construction_digest"] for x in candidates):
                candidates[-1] = t
    selected = _farthest_select_descriptors(candidates, int(spec["panel"]["beam"]), must_include_motif_ids={"TWIN:A", "TWIN:B"})
    meta = dict(batch.metadata)
    meta.update({"recipe_count": len(recipes), "successful_candidates": len(successful), "candidate_count_after_cap": len(candidates), "build_failures": failures})
    meta["candidate_context_sha256"] = ctx_sha
    meta["candidate_context_transport"] = transport_mode
    meta["spawn_safe_reference_transport"] = "CONTENT_ADDRESSED_STATE_DAG_RECONSTRUCTION"
    if fallback_reason is not None:
        meta["parallel_fallback_reason"] = fallback_reason
        if transport_mode == "CONTENT_ADDRESSED_STATE_DAG_RECONSTRUCTION_SERIAL_AUTO":
            meta["parallel_fallback_detail"] = "AUTO_SERIAL_FOR_DYNAMIC_O7_LIFECYCLE_SAFETY"
    return candidates, selected, meta, center


def rebuild_selected_states(
    selected: list[Mapping[str, Any]],
    *, engine: Any, prev: list[Any], level: int, pairs: list[tuple[int, int]], motifs: list[dict[str, Any]], center: Any,
) -> list[Any]:
    install_candidate_context(engine=engine, prev=sorted(prev, key=lambda s: s.construction_digest), level=level, pairs=pairs, motifs=motifs, center=center)
    try:
        states = []
        for row in selected:
            state = _build_recipe_state(row["recipe"])
            if state is None:
                raise RuntimeError(f"selected candidate failed deterministic parent rebuild: {row['motif_id']}")
            if state.construction_digest != row["construction_digest"]:
                raise RuntimeError(f"selected candidate digest mismatch on parent rebuild: {row['motif_id']}")
            states.append(state)
        return states
    finally:
        clear_candidate_context()
