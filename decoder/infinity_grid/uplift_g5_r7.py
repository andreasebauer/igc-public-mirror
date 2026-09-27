from __future__ import annotations

"""G5:R7 larger-domain adversarial escalation of single C/(0,0) hidden separation.

R6 established bounded marker-free separation by one ordinary payload on fresh n=9
and endpoint-typed n=5 panels. R7 does not promote that observation to a theorem.
It escalates the same exact observer on two larger fresh domains, using Decoder-owned
process-pool execution with partitions that provably do not split public-Q fibers.
"""
from collections import defaultdict
from importlib.resources import files
from itertools import combinations, product
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g5_r5 import _q_key, _relation_child_canons
from .execution import ExecutionPolicy, TaskSpec, execute_tasks


class G5R7Error(RuntimeError):
    pass


_SPEC = "G5_R7_SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION_SPEC_V1.json"
_PRIMARY_GROUPS: tuple[tuple[int, ...], ...] = ((5,), (4,), (6,), (3, 7), (1, 2, 8, 9))
_PARENT_OPS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (2, 4), (4, 2))
_NON00: tuple[tuple[int, int], ...] = tuple(x for x in _PARENT_OPS if x != (0, 0))


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def r7_spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_R7_SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION_SPEC_V1":
        raise G5R7Error("bad G5:R7 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G5R7Error("G5:R7 spec hash mismatch")
    return obj


def verify_authority(
    r6_primary: Mapping[str, Any],
    r6_verification: Mapping[str, Any],
    r6_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    s = r7_spec()
    a = s["authority"]
    failures: list[str] = []
    if r6_primary.get("schema_id") != "IG_G5_R6_SINGLE_PAYLOAD_MARKER_FREE_SEPARATION_RESULT_V1" or r6_primary.get("status") != "PASS":
        failures.append("R6_PRIMARY_SCHEMA_OR_STATUS")
    if str(r6_primary.get("science_sha256")) != a["g5_r6_primary_science_sha256"]:
        failures.append("R6_PRIMARY_IDENTITY")
    if r6_primary.get("single_payload_marker_free_separation_earned_on_frozen_scope") is not True:
        failures.append("R6_SINGLE_PAYLOAD_AUTHORITY")
    if r6_primary.get("promotion") is not False or r6_primary.get("g5_graduation_preserved") is not True:
        failures.append("R6_FIREWALL")
    if r6_verification.get("status") != "PASS" or str(r6_verification.get("verification_sha256")) != a["g5_r6_independent_verification_sha256"]:
        failures.append("R6_VERIFICATION")
    if r6_closeout.get("status") != "CERTIFIED_PASS" or str(r6_closeout.get("closeout_sha256")) != a["g5_r6_closeout_sha256"]:
        failures.append("R6_CLOSEOUT")
    out = {
        "schema_id": "IG_G5_R7_AUTHORITY_CHECK_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "r6_primary_science_sha256": r6_primary.get("science_sha256"),
        "r6_verification_sha256": r6_verification.get("verification_sha256"),
        "r6_closeout_sha256": r6_closeout.get("closeout_sha256"),
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R7Error("R6 authority mismatch: " + ",".join(failures))
    return out


def _tree_record(t: DecoratedG4Tree) -> dict[str, Any]:
    return {
        "n": int(t.n),
        "edges": [list(map(int, e)) for e in t.edges],
        "H_classes": list(t.H_classes),
        "edge_operators": [list(map(int, o)) for o in t.edge_operators],
    }


def _single_relation(ad: G4AcceptedAdapter, t: DecoratedG4Tree) -> tuple[tuple[Any, ...], ...]:
    return _relation_child_canons(ad, t, "C", (0, 0))


def _colors_exact_c(n: int, c_count: int):
    for idxs in combinations(range(n), c_count):
        ss = set(idxs)
        yield tuple("C" if i in ss else "D" for i in range(n))


def _dec_vector(edge_ops: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    v = [0] * 7
    for a, b in edge_ops:
        v[int(a)] += 1
        v[int(b)] += 1
    return tuple(v)


def _endpoint_assignments() -> list[tuple[tuple[int, int], ...]]:
    rows: list[tuple[tuple[int, int], ...]] = []
    for pos in combinations(range(5), 2):
        for chosen in product(_NON00, repeat=2):
            eo = [(0, 0)] * 5
            eo[pos[0]] = chosen[0]
            eo[pos[1]] = chosen[1]
            rows.append(tuple(eo))
    rows.sort()
    if len(rows) != 160 or len(set(rows)) != 160:
        raise G5R7Error("endpoint assignment construction must yield exactly 160 unique assignments")
    return rows


def _endpoint_vector_bins() -> tuple[tuple[tuple[int, ...], ...], ...]:
    byv: dict[tuple[int, ...], int] = defaultdict(int)
    for eo in _endpoint_assignments():
        byv[_dec_vector(eo)] += 1
    # Greedy deterministic load balance by number of assignments in each public-Q-defining vector.
    bins: list[list[tuple[int, ...]]] = [[] for _ in range(3)]
    loads = [0, 0, 0]
    for v, weight in sorted(byv.items(), key=lambda kv: (-kv[1], kv[0])):
        i = min(range(3), key=lambda j: (loads[j], j))
        bins[i].append(v)
        loads[i] += weight
    return tuple(tuple(sorted(x)) for x in bins)


def task_partition_certificate() -> dict[str, Any]:
    ad = G4AcceptedAdapter()
    failures: list[str] = []
    if tuple(map(int, ad._rows["C"]["caps7"])) != tuple(map(int, ad._rows["D"]["caps7"])):
        failures.append("C_D_CAPS7_NOT_EQUAL_PARTITION_SAFETY_LOST")
    flat_c = [x for g in _PRIMARY_GROUPS for x in g]
    if sorted(flat_c) != list(range(1, 10)) or len(flat_c) != len(set(flat_c)):
        failures.append("PRIMARY_C_COUNT_PARTITION_NOT_EXACT")
    assignments = _endpoint_assignments()
    all_vecs = sorted({_dec_vector(eo) for eo in assignments})
    bins = _endpoint_vector_bins()
    flat_v = [v for b in bins for v in b]
    if sorted(flat_v) != all_vecs or len(flat_v) != len(set(flat_v)):
        failures.append("ENDPOINT_VECTOR_PARTITION_NOT_EXACT")
    counts = defaultdict(int)
    for eo in assignments:
        counts[_dec_vector(eo)] += 1
    out = {
        "schema_id": "IG_G5_R7_TASK_PARTITION_CERTIFICATE_V1",
        "status": "PASS" if not failures else "FAIL",
        "primary_c_count_groups": [list(g) for g in _PRIMARY_GROUPS],
        "primary_c_count_coverage": sorted(flat_c),
        "endpoint_unique_reservation_vector_count": len(all_vecs),
        "endpoint_total_assignment_count": len(assignments),
        "endpoint_vector_groups": [[list(v) for v in b] for b in bins],
        "endpoint_vector_group_assignment_counts": [sum(counts[v] for v in b) for b in bins],
        "public_Q_fiber_split_across_tasks": False,
        "partition_safety_basis": "PRIMARY_FIXED_00_Q_IS_DETERMINED_BY_H_BAG_C_COUNT; ENDPOINT_TYPED_Q_IS_DETERMINED_BY_H_BAG_C_COUNT_AND_AGGREGATE_ENDPOINT_RESERVATION_VECTOR_BECAUSE_C_AND_D_CAPS7_ARE_EQUAL",
        "failures": failures,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R7Error("partition certificate failed: " + ",".join(failures))
    return out


def _audit_exact(
    exact: Mapping[tuple[Any, ...], DecoratedG4Tree],
    *,
    panel_id: str,
    raw: int,
    illegal: int,
    domain_meta: Mapping[str, Any],
) -> dict[str, Any]:
    ad = G4AcceptedAdapter()
    byq: dict[tuple[Any, ...], list[tuple[tuple[Any, ...], DecoratedG4Tree]]] = defaultdict(list)
    failures: list[str] = []
    for p, t in exact.items():
        pub = ad.public_read(t)
        if not pub.get("legal"):
            failures.append("ILLEGAL_PARENT_AFTER_FILTER")
            continue
        byq[_q_key(pub)].append((p, t))
    q_rows: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    total_collisions = 0
    empty = 0
    maxkids = 0
    for q in sorted(byq):
        owners: dict[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], DecoratedG4Tree]] = {}
        collisions = 0
        for p, t in sorted(byq[q], key=lambda x: x[0]):
            rel = _single_relation(ad, t)
            empty += int(not rel)
            maxkids = max(maxkids, len(rel))
            old = owners.get(rel)
            if old is None:
                owners[rel] = (p, t)
            elif old[0] != p:
                collisions += 1
                total_collisions += 1
                w = {
                    "Q": {"caps7": list(q[0]), "H_class_bag": [list(x) for x in q[1]]},
                    "parent_a": _tree_record(old[1]),
                    "parent_b": _tree_record(t),
                    "parent_a_exact_canon": old[0],
                    "parent_b_exact_canon": p,
                    "common_single_payload_child_relation": rel,
                }
                if first is None or json.dumps(w, sort_keys=True, separators=(",", ":")) < json.dumps(first, sort_keys=True, separators=(",", ":")):
                    first = w
        q_rows.append({
            "Q_caps7": list(q[0]),
            "Q_H_class_bag": [list(x) for x in q[1]],
            "exact_parent_count": len(byq[q]),
            "distinct_single_payload_relation_count": len(owners),
            "collision_count": collisions,
        })
    out = {
        "schema_id": "IG_G5_R7_SINGLE_PAYLOAD_PANEL_SHARD_RESULT_V1",
        "panel_id": panel_id,
        "status": "PASS" if not failures else "FAIL",
        "raw_parent_count": int(raw),
        "rejected_illegal_parent_count": int(illegal),
        "exact_parent_canon_count": len(exact),
        "public_Q_fiber_count": len(byq),
        "single_payload": {"new_H_class": "C", "operator": [0, 0]},
        "q_rows": q_rows,
        "evaluated_exact_parent_relations": len(exact),
        "empty_child_relation_count": empty,
        "max_exact_child_canons_in_relation": maxkids,
        "collision_count": total_collisions,
        "within_Q_single_payload_relation_injective": first is None,
        "collision_witness": first,
        "structural_equality_used": True,
        "digest_only_equality_used": False,
        "domain_meta": dict(domain_meta),
        "failures": failures,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R7Error(panel_id + " failed: " + ",".join(failures))
    return out


def _primary_shard(c_counts: Sequence[int], shard_id: str) -> dict[str, Any]:
    ad = G4AcceptedAdapter()
    shapes = _generate_tree_shapes(10)[10]
    exact: dict[tuple[Any, ...], DecoratedG4Tree] = {}
    raw = illegal = 0
    for edges in shapes.values():
        for c in c_counts:
            for colors in _colors_exact_c(10, int(c)):
                raw += 1
                t = DecoratedG4Tree(10, tuple(edges), tuple(colors), tuple((0, 0) for _ in range(9)))
                pub = ad.public_read(t)
                if not pub.get("legal"):
                    illegal += 1
                    continue
                exact.setdefault(ad.unrooted_canon(t), t)
    return _audit_exact(
        exact,
        panel_id="FRESH_N10_MIXED_CD_FIXED_00_EDGES_SINGLE_C00_ACTION:" + shard_id,
        raw=raw,
        illegal=illegal,
        domain_meta={
            "n": 10,
            "unlabelled_shape_count": len(shapes),
            "c_counts": list(map(int, c_counts)),
            "raw_colorings_per_shape": sum(comb(10, int(c)) for c in c_counts),
            "action_label_nonfresh_by_construction": True,
        },
    )


def _endpoint_shard(vectors: Sequence[Sequence[int]], shard_id: str) -> dict[str, Any]:
    ad = G4AcceptedAdapter()
    shapes = _generate_tree_shapes(6)[6]
    wanted = {tuple(map(int, v)) for v in vectors}
    assignments = [eo for eo in _endpoint_assignments() if _dec_vector(eo) in wanted]
    exact: dict[tuple[Any, ...], DecoratedG4Tree] = {}
    raw = illegal = 0
    for eo in assignments:
        for edges in shapes.values():
            for c in range(1, 6):
                for colors in _colors_exact_c(6, c):
                    raw += 1
                    t = DecoratedG4Tree(6, tuple(edges), tuple(colors), tuple(eo))
                    pub = ad.public_read(t)
                    if not pub.get("legal"):
                        illegal += 1
                        continue
                    exact.setdefault(ad.unrooted_canon(t), t)
    return _audit_exact(
        exact,
        panel_id="FRESH_N6_TWO_NON00_FIVE_OPERATOR_PARENT_PANEL_SINGLE_C00_ACTION:" + shard_id,
        raw=raw,
        illegal=illegal,
        domain_meta={
            "n": 6,
            "unlabelled_shape_count": len(shapes),
            "reservation_vectors": [list(v) for v in sorted(wanted)],
            "edge_assignment_count": len(assignments),
            "parent_operator_subset": [list(x) for x in _PARENT_OPS],
            "exactly_two_non00_edges": True,
            "action_label_nonfresh_by_construction": True,
        },
    )


def g5_r7_panel_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(payload["kind"])
    shard = str(payload["shard_id"])
    if kind == "PRIMARY_N10":
        return _primary_shard([int(x) for x in payload["c_counts"]], shard)
    if kind == "ENDPOINT_N6":
        return _endpoint_shard(payload["reservation_vectors"], shard)
    raise G5R7Error("unknown G5:R7 panel worker kind " + kind)


def _merge_shards(rows: Sequence[Mapping[str, Any]], *, panel_id: str, expected_q_disjoint: bool = True) -> dict[str, Any]:
    failures: list[str] = []
    q_seen: set[str] = set()
    q_rows: list[dict[str, Any]] = []
    witnesses: list[Mapping[str, Any]] = []
    raw = illegal = exact = fibers = evals = empty = collisions = 0
    maxkids = 0
    shard_hashes: list[str] = []
    for r in rows:
        if r.get("status") != "PASS":
            failures.append("SHARD_STATUS")
        shard_hashes.append(str(r.get("science_sha256")))
        raw += int(r.get("raw_parent_count", 0))
        illegal += int(r.get("rejected_illegal_parent_count", 0))
        exact += int(r.get("exact_parent_canon_count", 0))
        fibers += int(r.get("public_Q_fiber_count", 0))
        evals += int(r.get("evaluated_exact_parent_relations", 0))
        empty += int(r.get("empty_child_relation_count", 0))
        collisions += int(r.get("collision_count", 0))
        maxkids = max(maxkids, int(r.get("max_exact_child_canons_in_relation", 0)))
        if r.get("collision_witness") is not None:
            witnesses.append(r["collision_witness"])
        for q in r.get("q_rows", []):
            key = json.dumps([q["Q_caps7"], q["Q_H_class_bag"]], sort_keys=True, separators=(",", ":"))
            if expected_q_disjoint and key in q_seen:
                failures.append("PUBLIC_Q_FIBER_SPLIT_ACROSS_SHARDS")
            q_seen.add(key)
            q_rows.append(dict(q))
    if fibers != len(q_rows) or len(q_rows) != len(q_seen):
        failures.append("Q_FIBER_COUNT_MERGE_MISMATCH")
    first = None
    if witnesses:
        first = min(witnesses, key=lambda w: json.dumps(w, sort_keys=True, separators=(",", ":")))
    out = {
        "schema_id": "IG_G5_R7_SINGLE_PAYLOAD_PANEL_RESULT_V1",
        "panel_id": panel_id,
        "status": "PASS" if not failures else "FAIL",
        "shard_count": len(rows),
        "shard_science_sha256": sorted(shard_hashes),
        "raw_parent_count": raw,
        "rejected_illegal_parent_count": illegal,
        "exact_parent_canon_count": exact,
        "public_Q_fiber_count": fibers,
        "evaluated_exact_parent_relations": evals,
        "empty_child_relation_count": empty,
        "max_exact_child_canons_in_relation": maxkids,
        "collision_count": collisions,
        "within_Q_single_payload_relation_injective": first is None,
        "collision_witness": first,
        "q_rows": sorted(q_rows, key=lambda q: (q["Q_caps7"], q["Q_H_class_bag"])),
        "public_Q_fibers_partitioned_disjointly": expected_q_disjoint,
        "structural_equality_used": True,
        "digest_only_equality_used": False,
        "failures": failures,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R7Error(panel_id + " merge failed: " + ",".join(failures))
    return out


def _tasks(partition: Mapping[str, Any], binding: str) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for i, g in enumerate(partition["primary_c_count_groups"]):
        raw_per_shape = sum(comb(10, int(c)) for c in g)
        tasks.append(TaskSpec(
            task_id=f"r7-primary-{i:02d}",
            task_kind="G5_R7_PRIMARY_N10",
            binding_sha256=binding,
            payload={"kind": "PRIMARY_N10", "shard_id": f"C_COUNTS_{'_'.join(map(str, g))}", "c_counts": g},
            cost_weight=float(raw_per_shape * 106),
        ))
    # Reservation-vector groups are already deterministically load-balanced by assignment count.
    counts = defaultdict(int)
    for eo in _endpoint_assignments():
        counts[_dec_vector(eo)] += 1
    for i, g in enumerate(partition["endpoint_vector_groups"]):
        vecs = [tuple(map(int, v)) for v in g]
        acount = sum(counts[v] for v in vecs)
        tasks.append(TaskSpec(
            task_id=f"r7-endpoint-{i:02d}",
            task_kind="G5_R7_ENDPOINT_N6",
            binding_sha256=binding,
            payload={"kind": "ENDPOINT_N6", "shard_id": f"VECTOR_GROUP_{i}", "reservation_vectors": [list(v) for v in vecs]},
            cost_weight=float(acount * 6 * 62),
        ))
    return tasks


def run_g5_r7_adversarial_escalation(
    *,
    engine: Any,
    r6_primary: Mapping[str, Any],
    r6_verification: Mapping[str, Any],
    r6_closeout: Mapping[str, Any],
    telemetry: Any = None,
) -> dict[str, Any]:
    s = r7_spec()
    auth = verify_authority(r6_primary, r6_verification, r6_closeout)
    partition = task_partition_certificate()
    binding = canonical_sha256({
        "stage": "G5:R7",
        "spec_science_sha256": s["science_sha256"],
        "authority_science_sha256": auth["science_sha256"],
        "partition_science_sha256": partition["science_sha256"],
    })
    tasks = _tasks(partition, binding)
    if len(tasks) != 8:
        raise G5R7Error(f"expected 8 registered R7 tasks, got {len(tasks)}")
    if engine is not None:
        policy = engine.policy
    else:
        policy = ExecutionPolicy(
            backend="AUTO", requested_workers=4, scheduler="COST_WEIGHTED_SHARDS",
            start_method="fork", reserve_cores=0, owner="g5-r7-cold-v03080",
        )
    batch = execute_tasks(
        tasks,
        worker_ref="infinity_grid.uplift_g5_r7:g5_r7_panel_worker",
        policy=policy,
        telemetry=telemetry,
        scientific_task_units={t.task_id: 1 for t in tasks},
    )
    primary_rows = [batch.results[t.task_id] for t in tasks if t.task_kind == "G5_R7_PRIMARY_N10"]
    endpoint_rows = [batch.results[t.task_id] for t in tasks if t.task_kind == "G5_R7_ENDPOINT_N6"]
    primary = _merge_shards(primary_rows, panel_id="FRESH_N10_MIXED_CD_FIXED_00_EDGES_SINGLE_C00_ACTION")
    endpoint = _merge_shards(endpoint_rows, panel_id="FRESH_N6_TWO_NON00_FIVE_OPERATOR_PARENT_PANEL_SINGLE_C00_ACTION")
    injective = bool(primary["within_Q_single_payload_relation_injective"] and endpoint["within_Q_single_payload_relation_injective"])
    witness = primary.get("collision_witness") or endpoint.get("collision_witness")
    classification = s["pass_classifications"]["injective" if injective else "collision"]
    out = {
        "schema_id": "IG_G5_R7_SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION_RESULT_V1",
        "status": "PASS",
        "stage_ref": "G5:R7",
        "classification": classification,
        "promotion": False,
        "g5_graduation_preserved": True,
        "g6_started": False,
        "authority": auth,
        "task_partition_certificate": partition,
        "fresh_n10_panel": primary,
        "endpoint_typed_n6_panel": endpoint,
        "single_payload_separation_survives_larger_fresh_scope": injective,
        "single_payload_collision_earned_on_r7_scope": not injective,
        "first_collision_witness": witness,
        "global_single_payload_minimality_claim": False,
        "all_finite_carrier_single_payload_separation_claim": False,
        "global_hidden_state_minimality_claim": False,
        "exact_hidden_canon_promoted_to_public": False,
        "topology_promoted": False,
        "public_descriptor_changed": False,
        "next_authorized_stage": s["next_authorized_stage"],
        "cost_control": s["cost_control"],
        "nonclaims": s["nonclaims"],
    }
    out["science_sha256"] = canonical_sha256(out)
    out["parallel_batch_metadata"] = batch.metadata  # execution-only; stripped before cold comparison
    return out


def stable_payload(r: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in {"source_sha256", "source_version", "registry_sha256", "execution_metadata", "parallel_batch_metadata"}}


def compare_cold_replay(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    ps = canonical_sha256(stable_payload(primary))
    cs = canonical_sha256(stable_payload(cold))
    failures: list[str] = []
    if primary.get("science_sha256") != cold.get("science_sha256"):
        failures.append("SCIENCE_SHA")
    if primary.get("classification") != cold.get("classification"):
        failures.append("CLASSIFICATION")
    if ps != cs:
        failures.append("STABLE_PAYLOAD")
    out = {
        "schema_id": "IG_G5_R7_COLD_REPLAY_COMPARISON_V1",
        "status": "PASS" if not failures else "FAIL",
        "certification": "CERTIFIED_PASS" if not failures else "CERTIFICATION_FAILED",
        "failures": failures,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
        "stable_scientific_payload_exact_equal": ps == cs,
        "stable_scientific_payload_sha256": ps,
    }
    out["comparison_sha256"] = canonical_sha256(out)
    return out


def certified_closeout(
    primary: Mapping[str, Any],
    cold: Mapping[str, Any],
    comparison: Mapping[str, Any],
    independent: Mapping[str, Any],
) -> dict[str, Any]:
    ok = (
        primary.get("status") == "PASS"
        and cold.get("status") == "PASS"
        and comparison.get("certification") == "CERTIFIED_PASS"
        and independent.get("status") == "PASS"
    )
    out = {
        "schema_id": "IG_G5_R7_CERTIFIED_CLOSEOUT_V1",
        "status": "CERTIFIED_PASS" if ok else "CERTIFICATION_FAILED",
        "experiment_id": "G5:R7.SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION",
        "classification": primary.get("classification") if ok else "G5_R7_REVIEW_REQUIRED_NO_PROMOTION",
        "science_sha256": primary.get("science_sha256"),
        "comparison_sha256": comparison.get("comparison_sha256"),
        "independent_verification_sha256": independent.get("verification_sha256"),
        "source_sha256": primary.get("source_sha256"),
        "registry_sha256": primary.get("registry_sha256"),
        "promotion": False,
        "g5_graduation_preserved": True,
        "single_payload_separation_survives_larger_fresh_scope": bool(primary.get("single_payload_separation_survives_larger_fresh_scope")) if ok else False,
        "single_payload_collision_earned_on_r7_scope": bool(primary.get("single_payload_collision_earned_on_r7_scope")) if ok else False,
        "global_single_payload_minimality_claim": False,
        "global_hidden_state_minimality_claim": False,
        "topology_promoted": False,
        "g6_started": False,
        "next_authorized_stage": "G5_POST_R7_CLOSEOUT_REVIEW" if ok else None,
        "failures": comparison.get("failures", []),
    }
    out["closeout_sha256"] = canonical_sha256(out)
    return out
