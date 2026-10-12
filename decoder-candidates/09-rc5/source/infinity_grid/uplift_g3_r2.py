from __future__ import annotations

"""Registered non-promoting G3:R2 adaptive post-graduation maturation sweep.

R2 continues the certified hidden G3 tree-fiber grammar beyond the exhaustive R1 n<=12
census without rematerializing exact G3 relation frontiers.  It exhaustively advances
unlabelled tree ranks through an adaptive dense window and, only if no dense event fires,
runs deterministic sparse sentinel families at larger ranks.  The selected R1 read
SHELL_PROFILE_MULTISET remains challenge-only and is never promoted here.
"""

from collections import defaultdict, deque
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import heapq
import json

from .canon import canonical_sha256, write_json_atomic
from .uplift_g3_r0 import _tree_canon


class G3R2Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G3_R2_ADAPTIVE_MATURATION_SPEC_V1.json"


def r2_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G3_R2_ADAPTIVE_MATURATION_SPEC_V1":
        raise G3R2Error("bad G3:R2 spec schema")
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if observed != expected:
        raise G3R2Error(f"G3:R2 spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def verify_r2_authority(r1_result: Mapping[str, Any], r1_replay: Mapping[str, Any]) -> dict[str, Any]:
    spec = r2_spec()
    failures: list[str] = []
    auth = spec["authority"]
    if r1_result.get("schema_id") != "IG_G3_R1_FIBER_MODULI_AUDIT_RESULT_V1" or r1_result.get("status") != "PASS":
        failures.append("R1_SCHEMA_OR_STATUS")
    if str(r1_result.get("classification")) != str(auth["g3_r1_classification"]):
        failures.append("R1_CLASSIFICATION")
    if str(r1_result.get("science_sha256")) != str(auth["g3_r1_science_sha256"]):
        failures.append("R1_IDENTITY")
    if r1_result.get("g3_graduation_preserved") is not True or r1_result.get("g4_started") is not False:
        failures.append("R1_FIREWALL")
    selected = list(r1_result.get("candidate_tree_read_audit", {}).get("selected_candidates") or [])
    if selected != [str(auth["selected_read"])]:
        failures.append("R1_SELECTED_READ")
    census = r1_result.get("tree_fiber_grafting_census", {})
    if int(census.get("max_g2_units", -1)) != int(auth["r1_exhaustive_max_g2_units"]):
        failures.append("R1_MAX_RANK")
    counts = list(census.get("unlabelled_tree_counts") or [])
    if not counts or int(counts[-1]) != int(auth["r1_unlabelled_tree_count_at_12"]):
        failures.append("R1_N12_COUNT")

    if r1_replay.get("schema_id") != "IG_G3_R1_REPLAY_COMPARISON_V1" or r1_replay.get("status") != "PASS":
        failures.append("R1_REPLAY_SCHEMA_OR_STATUS")
    if str(r1_replay.get("science_sha256")) != str(auth["g3_r1_replay_science_sha256"]):
        failures.append("R1_REPLAY_IDENTITY")
    if r1_replay.get("stable_scientific_payload_exact_equal") is not True or r1_replay.get("science_sha256_equal") is not True:
        failures.append("R1_REPLAY_NOT_EXACT")
    if str(r1_replay.get("primary_science_sha256")) != str(r1_result.get("science_sha256")) or str(r1_replay.get("cold_science_sha256")) != str(r1_result.get("science_sha256")):
        failures.append("R1_REPLAY_RESULT_MISMATCH")

    out = {
        "schema_id": "IG_G3_R2_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "g3_r1_science_sha256": r1_result.get("science_sha256"),
        "g3_r1_replay_science_sha256": r1_replay.get("science_sha256"),
        "selected_read": selected,
        "g3_graduation_preserved": True,
        "promotion": False,
        "g4_started": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G3R2Error("G3:R2 authority failed: " + ",".join(failures))
    return out


def _adjacency(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for u0, v0 in edges:
        u, v = int(u0), int(v0)
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            raise G3R2Error("bad tree edge")
        adj[u].append(v); adj[v].append(u)
    return adj


def _shell_profile_multiset(n: int, edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    if n <= 0:
        return tuple()
    adj = _adjacency(n, edges)
    profiles: list[tuple[int, ...]] = []
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        maxd = 0
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = du + 1
                    if du + 1 > maxd:
                        maxd = du + 1
                    q.append(v)
        if any(d < 0 for d in dist):
            raise G3R2Error("shell profile received disconnected graph")
        counts = [0] * (maxd + 1)
        for d in dist:
            counts[d] += 1
        profiles.append(tuple(counts))
    return tuple(sorted(profiles))


def _shell_key(value: Sequence[Sequence[int]]) -> str:
    return json.dumps([[int(y) for y in row] for row in value], separators=(",", ":"), ensure_ascii=False)


def _child_topology_set(n: int, edges: Sequence[tuple[int, int]]) -> list[str]:
    return sorted({_tree_canon(n + 1, list(edges) + [(v, n)]) for v in range(n)})


def _collision_continuation_witness(n: int, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    lchildren = _child_topology_set(n, left["edges"])
    rchildren = _child_topology_set(n, right["edges"])
    return {
        "parent_g2_unit_count": int(n),
        "left_topology_canon": str(left["canon"]),
        "right_topology_canon": str(right["canon"]),
        "shell_profile_multiset": [list(x) for x in left["shell"]],
        "left_child_topology_set": lchildren,
        "right_child_topology_set": rchildren,
        "one_step_continuation_equal": lchildren == rchildren,
    }


def _seed_rank12(r1_result: Mapping[str, Any]) -> tuple[dict[str, list[tuple[int, int]]], dict[str, tuple[tuple[int, ...], ...]]]:
    rows = r1_result["tree_fiber_grafting_census"]["rows"]
    row = next((r for r in rows if int(r.get("g2_unit_count", -1)) == 12), None)
    if row is None:
        raise G3R2Error("certified R1 n=12 row absent")
    shapes: dict[str, list[tuple[int, int]]] = {}
    shells: dict[str, tuple[tuple[int, ...], ...]] = {}
    for topo in row["topologies"]:
        canon = str(topo["topology_canon"])
        edges = [tuple(map(int, e)) for e in topo["example_edges"]]
        shell = tuple(tuple(int(y) for y in x) for x in topo["candidate_reads"]["SHELL_PROFILE_MULTISET"])
        if _tree_canon(12, edges) != canon:
            raise G3R2Error("R1 n=12 topology canon reproduction failed")
        if _shell_profile_multiset(12, edges) != shell:
            raise G3R2Error("R1 n=12 shell-profile reproduction failed")
        shapes[canon] = edges
        shells[canon] = shell
    if len(shapes) != 551 or len({_shell_key(v) for v in shells.values()}) != 551:
        raise G3R2Error("R1 n=12 prefix gate failed")
    return shapes, shells


def dense_adaptive_maturation(r1_result: Mapping[str, Any], *, progress_path: str | Path | None = None) -> dict[str, Any]:
    spec = r2_spec()["dense_phase"]
    start = int(spec["start_g2_units"])
    min_plateau = int(spec["minimum_plateau_decision_rank"])
    hard_max = int(spec["hard_max_g2_units"])
    window = int(spec["stabilization_window_ranks"])
    if start != 13 or min_plateau < start or hard_max < min_plateau:
        raise G3R2Error("unexpected frozen dense range")

    current, current_shells = _seed_rank12(r1_result)
    stable_ranks: list[int] = []
    rows: list[dict[str, Any]] = []
    event: dict[str, Any] | None = None
    plateau_rank: int | None = None

    def progress(obj: Mapping[str, Any]) -> None:
        if progress_path is not None:
            write_json_atomic(Path(progress_path), {"schema_id": "IG_G3_R2_OPERATIONAL_PROGRESS_V1", **obj})

    for n in range(start, hard_max + 1):
        nxt: dict[str, list[tuple[int, int]]] = {}
        transition_edges = 0
        min_children: int | None = None
        max_children: int | None = None
        for pcanon in sorted(current):
            pedges = current[pcanon]
            child_set: set[str] = set()
            for v in range(n - 1):
                e2 = list(pedges) + [(int(v), int(n - 1))]
                c2 = _tree_canon(n, e2)
                child_set.add(c2)
                nxt.setdefault(c2, e2)
            k = len(child_set)
            transition_edges += k
            min_children = k if min_children is None else min(min_children, k)
            max_children = k if max_children is None else max(max_children, k)

        shell_seen: dict[str, dict[str, Any]] = {}
        next_shells: dict[str, tuple[tuple[int, ...], ...]] = {}
        collision: dict[str, Any] | None = None
        for canon in sorted(nxt):
            edges = nxt[canon]
            shell = _shell_profile_multiset(n, edges)
            next_shells[canon] = shell
            key = _shell_key(shell)
            if key in shell_seen and shell_seen[key]["canon"] != canon and collision is None:
                collision = {
                    "g2_unit_count": int(n),
                    "left": shell_seen[key],
                    "right": {"canon": canon, "edges": edges, "shell": shell},
                }
            else:
                shell_seen.setdefault(key, {"canon": canon, "edges": edges, "shell": shell})

        row = {
            "g2_unit_count": int(n),
            "unlabelled_tree_topology_count": len(nxt),
            "shell_profile_class_count": len(shell_seen),
            "shell_profile_topology_separating": collision is None,
            "distinct_leaf_attachment_transition_edges_from_previous_rank": int(transition_edges),
            "min_distinct_children_per_parent": min_children,
            "max_distinct_children_per_parent": max_children,
        }
        rows.append(row)
        progress({
            "status": "RUNNING",
            "phase": "DENSE",
            "completed_rank": int(n),
            "topology_count": len(nxt),
            "shell_profile_class_count": len(shell_seen),
            "event_detected": collision is not None,
        })

        if collision is not None:
            cont = _collision_continuation_witness(n, collision["left"], collision["right"])
            event = {
                "event_scope": "EXHAUSTIVE_DENSE",
                "g2_unit_count": int(n),
                "kind": "SHELL_PROFILE_CONTINUATION_BREAK" if not cont["one_step_continuation_equal"] else "SHELL_PROFILE_TOPOLOGY_COLLISION_CONTINUATION_SURVIVES",
                "witness": cont,
            }
            break

        stable_ranks.append(n)
        if n >= min_plateau and len(stable_ranks) >= window and stable_ranks[-window:] == list(range(n - window + 1, n + 1)):
            plateau_rank = n
            current, current_shells = nxt, next_shells
            break

        current, current_shells = nxt, next_shells

    if event is None and plateau_rank is None:
        raise G3R2Error("dense phase reached hard maximum without event or frozen plateau")

    out = {
        "schema_id": "IG_G3_R2_DENSE_ADAPTIVE_MATURATION_V1",
        "status": "EVENT" if event is not None else "LOCAL_PLATEAU",
        "seed_rank": 12,
        "seed_topology_count": 551,
        "rows": rows,
        "stable_no_break_ranks": stable_ranks,
        "plateau_rank": plateau_rank,
        "event": event,
        "exhaustive_scope_end_rank": int(rows[-1]["g2_unit_count"]),
        "selected_read": "SHELL_PROFILE_MULTISET",
        "selected_read_promoted": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    progress({"status": "DENSE_COMPLETE", "phase": "DENSE", "science_sha256": out["science_sha256"], "event": event, "plateau_rank": plateau_rank})
    return out


def _path_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _star_edges(n: int) -> list[tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def _double_star_edges(n: int, a: int) -> list[tuple[int, int]]:
    b = n - 2 - a
    if a < 1 or b < 1:
        raise G3R2Error("bad double-star split")
    edges = [(0, 1)]
    v = 2
    for _ in range(a):
        edges.append((0, v)); v += 1
    for _ in range(b):
        edges.append((1, v)); v += 1
    return edges


def _broom_edges(n: int, leaf_count: int) -> list[tuple[int, int]]:
    k = int(leaf_count)
    path_n = n - k
    if path_n < 2 or k < 1:
        raise G3R2Error("bad broom")
    edges = [(i, i + 1) for i in range(path_n - 1)]
    for v in range(path_n, n):
        edges.append((0, v))
    return edges


def _spider_edges(arms: Sequence[int]) -> list[tuple[int, int]]:
    if any(int(x) <= 0 for x in arms):
        raise G3R2Error("spider arms must be positive")
    edges: list[tuple[int, int]] = []
    nxt = 1
    for L0 in arms:
        L = int(L0)
        parent = 0
        for _ in range(L):
            edges.append((parent, nxt))
            parent = nxt
            nxt += 1
    return edges


def _partitions3(total: int):
    for a in range(1, total + 1):
        for b in range(a, total + 1):
            c = total - a - b
            if c < b:
                break
            if c >= b:
                yield (a, b, c)


def _partitions4(total: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for a in range(1, total + 1):
        for b in range(a, total + 1):
            for c in range(b, total + 1):
                d = total - a - b - c
                if d < c:
                    break
                if d >= c:
                    out.append((a, b, c, d))
    return out


def _even_subsample(rows: Sequence[Any], cap: int) -> list[Any]:
    if len(rows) <= cap:
        return list(rows)
    if cap <= 1:
        return [rows[0]]
    idx = sorted({round(i * (len(rows) - 1) / (cap - 1)) for i in range(cap)})
    return [rows[i] for i in idx]


def _det_prufer(n: int, sample_index: int) -> list[int]:
    out: list[int] = []
    for pos in range(max(0, n - 2)):
        raw = f"IG_G3_R2_SENTINEL_V1|{n}|{sample_index}|{pos}".encode("utf-8")
        out.append(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % n)
    return out


def _prufer_edges(seq: Sequence[int], n: int) -> list[tuple[int, int]]:
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    deg = [1] * n
    for x in seq:
        deg[int(x)] += 1
    heap = [i for i, d in enumerate(deg) if d == 1]
    heapq.heapify(heap)
    edges: list[tuple[int, int]] = []
    for x0 in seq:
        x = int(x0)
        leaf = heapq.heappop(heap)
        edges.append((leaf, x))
        deg[leaf] -= 1; deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(heap, x)
    a = heapq.heappop(heap); b = heapq.heappop(heap)
    edges.append((a, b))
    return edges


def _sentinel_family(n: int) -> tuple[dict[str, list[tuple[int, int]]], dict[str, int]]:
    cfg = r2_spec()["sparse_sentinels"]
    raw: list[tuple[str, list[tuple[int, int]]]] = [("PATH", _path_edges(n)), ("STAR", _star_edges(n))]
    for a in range(1, n - 2):
        raw.append(("DOUBLE_STAR", _double_star_edges(n, a)))
    for k in range(1, min(int(cfg["broom_cap_per_rank"]), n - 2) + 1):
        raw.append(("BROOM", _broom_edges(n, k)))
    for arms in _partitions3(n - 1):
        raw.append(("SPIDER3", _spider_edges(arms)))
    p4 = _even_subsample(_partitions4(n - 1), int(cfg["four_arm_partition_cap_per_rank"]))
    for arms in p4:
        raw.append(("SPIDER4", _spider_edges(arms)))
    for i in range(int(cfg["random_sample_count_per_rank"])):
        raw.append(("PRUFER_FIXED", _prufer_edges(_det_prufer(n, i), n)))

    dedup: dict[str, list[tuple[int, int]]] = {}
    kind_counts: defaultdict[str, int] = defaultdict(int)
    for kind, edges in raw:
        canon = _tree_canon(n, edges)
        if canon not in dedup:
            dedup[canon] = edges
            kind_counts[kind] += 1
    return dedup, dict(sorted(kind_counts.items()))


def sentinel_rank_audit_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    n = int(payload["g2_unit_count"])
    expected_spec = str(payload["r2_spec_science_sha256"])
    if r2_spec()["science_sha256"] != expected_spec:
        raise G3R2Error("sentinel worker spec identity mismatch")
    shapes, kind_counts = _sentinel_family(n)
    shell_seen: dict[str, dict[str, Any]] = {}
    collision: dict[str, Any] | None = None
    for canon in sorted(shapes):
        edges = shapes[canon]
        shell = _shell_profile_multiset(n, edges)
        key = _shell_key(shell)
        if key in shell_seen and shell_seen[key]["canon"] != canon and collision is None:
            collision = {
                "left": shell_seen[key],
                "right": {"canon": canon, "edges": edges, "shell": shell},
            }
            break
        shell_seen.setdefault(key, {"canon": canon, "edges": edges, "shell": shell})
    event = None
    if collision is not None:
        cont = _collision_continuation_witness(n, collision["left"], collision["right"])
        event = {
            "event_scope": "SPARSE_SENTINEL",
            "g2_unit_count": n,
            "kind": "SHELL_PROFILE_CONTINUATION_BREAK" if not cont["one_step_continuation_equal"] else "SHELL_PROFILE_TOPOLOGY_COLLISION_CONTINUATION_SURVIVES",
            "witness": cont,
        }
    out = {
        "schema_id": "IG_G3_R2_SPARSE_SENTINEL_RANK_AUDIT_V1",
        "status": "EVENT" if event is not None else "NO_EVENT_IN_FROZEN_SENTINEL_FAMILY",
        "g2_unit_count": n,
        "sentinel_family_unique_topology_count": len(shapes),
        "sentinel_family_unique_by_generator_kind": kind_counts,
        "shell_profile_class_count_before_first_event_or_full_family": len(shell_seen),
        "event": event,
        "scope_is_exhaustive": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def finalize_r2_result(*, authority: Mapping[str, Any], dense: Mapping[str, Any], sentinel_rows: Sequence[Mapping[str, Any]], sentinel_batch_science: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = r2_spec()
    dense_event = dense.get("event")
    first_sentinel_event = next((r.get("event") for r in sorted(sentinel_rows, key=lambda x: int(x["g2_unit_count"])) if r.get("event") is not None), None)
    if dense_event is not None:
        if dense_event["kind"] == "SHELL_PROFILE_CONTINUATION_BREAK":
            classification = spec["outcomes"]["dense_continuation_event"]
        else:
            classification = spec["outcomes"]["dense_topology_only_event"]
    elif first_sentinel_event is not None:
        if first_sentinel_event["kind"] == "SHELL_PROFILE_CONTINUATION_BREAK":
            classification = spec["outcomes"]["sentinel_continuation_event"]
        else:
            classification = spec["outcomes"]["sentinel_topology_only_event"]
    else:
        classification = spec["outcomes"]["plateau"]

    result = {
        "schema_id": "IG_G3_R2_ADAPTIVE_MATURATION_RESULT_V1",
        "status": "PASS",
        "classification": classification,
        "promotion": False,
        "g3_graduation_preserved": True,
        "g4_started": False,
        "authority": dict(authority),
        "frozen_question_sha256": spec["science_sha256"],
        "lanes": list(spec["lanes"]),
        "selected_read_under_stress": "SHELL_PROFILE_MULTISET",
        "dense_adaptive_maturation": dict(dense),
        "sparse_sentinel_audits": list(sorted(sentinel_rows, key=lambda x: int(x["g2_unit_count"]))),
        "sparse_sentinel_execution_science": None if sentinel_batch_science is None else dict(sentinel_batch_science),
        "plateau_status": {
            "dense_local_plateau": dense.get("status") == "LOCAL_PLATEAU",
            "sparse_sentinel_no_event": bool(sentinel_rows) and all(r.get("event") is None for r in sentinel_rows),
            "global_shell_profile_completeness_claimed": False,
        },
        "next_recommendation": (
            "DESIGN_G4_PHASE0_USING_SHELL_PROFILE_MULTISET_AS_PREREGISTERED_CHALLENGE_READ_NOT_ASSUMED_STATE"
            if classification == spec["outcomes"]["plateau"]
            else "REVIEW_FIRST_R2_SHELL_PROFILE_EVENT_BEFORE_G4_PHASE0_OR_ANY_READ_REFINEMENT"
        ),
        "topology_promoted": False,
        "geometry_status": "PHYSICAL_GEOMETRY_NOT_EARNED",
        "nonclaims": list(spec["nonclaims"]),
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
