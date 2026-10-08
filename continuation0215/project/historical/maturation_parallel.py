from __future__ import annotations
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
from infinity_grid.canon import canonical_sha256
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
