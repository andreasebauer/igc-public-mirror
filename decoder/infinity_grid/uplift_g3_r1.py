from __future__ import annotations

"""Registered non-promoting G3:R1 finite fiber/moduli reconnaissance.

R1 treats the exact hierarchical G3 realizations hidden behind the graduated CAPS7
quotient as finite combinatorial fibers over a frozen public/resource base.  It studies
how one-cross leaf attachment acts on coarse tree-shape strata and audits a preregistered
library of intrinsic tree reads for topology separation and one-step continuation
sufficiency.  Nothing here promotes topology or changes G3 action semantics.
"""

from collections import Counter, defaultdict
from importlib.resources import files
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .uplift_g3_r0 import _tree_canon, _graph_metrics


class G3R1Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G3_R1_FIBER_MODULI_AUDIT_SPEC_V1.json"


def r1_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G3_R1_FIBER_MODULI_AUDIT_SPEC_V1":
        raise G3R1Error("bad G3:R1 spec schema")
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if observed != expected:
        raise G3R1Error(f"G3:R1 spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def verify_r1_authority(r0_result: Mapping[str, Any], r0_replay: Mapping[str, Any]) -> dict[str, Any]:
    spec = r1_spec()
    failures: list[str] = []
    if r0_result.get("schema_id") != "IG_G3_R0_STRUCTURAL_RECON_RESULT_V1" or r0_result.get("status") != "PASS":
        failures.append("R0_SCHEMA_OR_STATUS")
    if r0_result.get("classification") != "INTRINSIC_G3_TWO_SCALE_GRADED_TREE_PREGEOMETRY_PRESENT_CAPS7_TOPOLOGY_BLIND":
        failures.append("R0_CLASSIFICATION")
    if str(r0_result.get("science_sha256")) != str(spec["authority"]["g3_r0_science_sha256"]):
        failures.append("R0_IDENTITY")
    if r0_result.get("g3_graduation_preserved") is not True or r0_result.get("topology_promoted") is not False or r0_result.get("g4_started") is not False:
        failures.append("R0_FIREWALL")

    if r0_replay.get("schema_id") != "IG_G3_R0_REPLAY_COMPARISON_V1" or r0_replay.get("status") != "PASS":
        failures.append("R0_REPLAY_SCHEMA_OR_STATUS")
    if str(r0_replay.get("science_sha256")) != str(spec["authority"]["g3_r0_replay_science_sha256"]):
        failures.append("R0_REPLAY_IDENTITY")
    if r0_replay.get("stable_scientific_payload_exact_equal") is not True or r0_replay.get("science_sha256_equal") is not True:
        failures.append("R0_REPLAY_NOT_EXACT")
    if str(r0_replay.get("primary_science_sha256")) != str(r0_result.get("science_sha256")) or str(r0_replay.get("cold_science_sha256")) != str(r0_result.get("science_sha256")):
        failures.append("R0_REPLAY_RESULT_MISMATCH")

    rows = r0_result.get("exact_growth_probe", {}).get("rows", [])
    n4 = next((r for r in rows if int(r.get("g2_unit_count", -1)) == 4), None)
    if n4 is None:
        failures.append("R0_N4_ROW_MISSING")
    else:
        if int(n4.get("exact_branch_count", -1)) != int(spec["authority"]["r0_n4_exact_branch_count"]):
            failures.append("R0_N4_BRANCH_COUNT")
        if int(n4.get("unique_coarse_unlabelled_tree_topologies", -1)) != 2:
            failures.append("R0_N4_COARSE_TOPOLOGY_COUNT")
        if int(n4.get("unique_flattened_g1_tree_topologies", -1)) != int(spec["authority"]["r0_n4_fine_topology_count"]):
            failures.append("R0_N4_FINE_TOPOLOGY_COUNT")
        if int(n4.get("unique_caps7_states", -1)) != 1:
            failures.append("R0_N4_CAPS7_COUNT")

    out = {
        "schema_id": "IG_G3_R1_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "g3_r0_science_sha256": r0_result.get("science_sha256"),
        "g3_r0_replay_science_sha256": r0_replay.get("science_sha256"),
        "g3_graduation_preserved": True,
        "promotion": False,
        "g4_started": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G3R1Error("G3:R1 authority failed: " + ",".join(failures))
    return out


def _leaf_count(deg: Sequence[int]) -> int:
    if len(deg) <= 1:
        return 1
    return sum(1 for x in deg if int(x) == 1)


def _shell_multiset(metrics: Mapping[str, Any]) -> list[list[int]]:
    return [list(x) for x in sorted(tuple(int(y) for y in row) for row in metrics.get("shell_profiles", []))]


def _candidate_value(name: str, *, metrics: Mapping[str, Any], topology_canon: str) -> Any:
    deg = [int(x) for x in metrics.get("degree_sequence", [])]
    if name == "MAX_DEGREE":
        return max(deg) if deg else 0
    if name == "LEAF_COUNT":
        return _leaf_count(deg)
    if name == "DIAMETER":
        return int(metrics.get("diameter", 0))
    if name == "RADIUS":
        return int(metrics.get("radius", 0))
    if name == "ARTICULATION_COUNT":
        return int(metrics.get("articulations", 0))
    if name == "WIENER_INDEX":
        return int(metrics.get("distance_sum", 0))
    if name == "DEGREE_SEQUENCE":
        return deg
    if name == "DEGREE_SEQUENCE_PLUS_WIENER":
        return {"degree_sequence": deg, "wiener_index": int(metrics.get("distance_sum", 0))}
    if name == "SHELL_PROFILE_MULTISET":
        return _shell_multiset(metrics)
    if name == "FULL_TREE_CANON":
        return str(topology_canon)
    raise G3R1Error(f"unknown G3:R1 candidate read {name}")


def _key(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _generate_tree_shapes(max_n: int) -> dict[int, dict[str, list[tuple[int, int]]]]:
    if max_n < 1:
        raise G3R1Error("max_n must be positive")
    shapes: dict[int, dict[str, list[tuple[int, int]]]] = {1: {_tree_canon(1, []): []}}
    for n in range(1, max_n):
        nxt: dict[str, list[tuple[int, int]]] = {}
        for canon in sorted(shapes[n]):
            edges = shapes[n][canon]
            for v in range(n):
                e2 = list(edges) + [(int(v), int(n))]
                c2 = _tree_canon(n + 1, e2)
                nxt.setdefault(c2, e2)
        if not nxt:
            raise G3R1Error(f"tree-shape closure exhausted at n={n+1}")
        shapes[n + 1] = nxt
    return shapes


def _resource_consumption(op: Sequence[int], edge_count: int) -> list[int]:
    a, b = map(int, op)
    out = [0] * 7
    out[a] += int(edge_count)
    out[b] += int(edge_count)
    return out


def _rank_caps(seed_caps: Sequence[int], op: Sequence[int], n: int) -> list[int]:
    a, b = map(int, op)
    out = [int(n) * int(x) for x in seed_caps]
    out[a] -= int(n) - 1
    out[b] -= int(n) - 1
    return out


def _rank_base_key(seed_caps: Sequence[int], op: Sequence[int], n: int) -> dict[str, Any]:
    payload = {
        "g2_unit_count": int(n),
        "caps7": _rank_caps(seed_caps, op, n),
        "g3_bridge_consumption_vector": _resource_consumption(op, int(n) - 1),
        "g3_bridge_operator": [int(op[0]), int(op[1])],
        "g3_bridge_count": int(n) - 1,
    }
    payload["base_key_sha256"] = canonical_sha256(payload)
    return payload


def _shape_rows(shapes: Mapping[int, Mapping[str, list[tuple[int, int]]]], max_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in range(1, max_n + 1):
        topo_rows: list[dict[str, Any]] = []
        transition_edge_count = 0
        branch_counts: list[int] = []
        for canon in sorted(shapes[n]):
            edges = list(shapes[n][canon])
            metrics = _graph_metrics(n, edges)
            child_counts: Counter[str] = Counter()
            if n < max_n:
                for v in range(n):
                    child_counts[_tree_canon(n + 1, edges + [(v, n)])] += 1
                transition_edge_count += len(child_counts)
                branch_counts.append(len(child_counts))
            topo_rows.append({
                "topology_canon": canon,
                "example_edges": [list(e) for e in edges],
                "metrics": metrics,
                "candidate_reads": {
                    name: _candidate_value(name, metrics=metrics, topology_canon=canon)
                    for name in r1_spec()["candidate_read_audit"]["ordered_candidates"]
                },
                "leaf_attachment_children": [
                    {"topology_canon": c, "attachment_vertex_multiplicity": int(child_counts[c])}
                    for c in sorted(child_counts)
                ],
            })
        rows.append({
            "g2_unit_count": n,
            "unlabelled_tree_topology_count": len(shapes[n]),
            "distinct_leaf_attachment_transition_edges": transition_edge_count if n < max_n else None,
            "min_distinct_children_per_topology": min(branch_counts) if branch_counts else None,
            "max_distinct_children_per_topology": max(branch_counts) if branch_counts else None,
            "topologies": topo_rows,
        })
    return rows


def _first_collision(rows: Sequence[Mapping[str, Any]], candidate: str, *, continuation: bool) -> dict[str, Any] | None:
    max_n = max(int(r["g2_unit_count"]) for r in rows)
    for row in rows:
        n = int(row["g2_unit_count"])
        if continuation and n >= max_n:
            continue
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for topo in row["topologies"]:
            value = topo["candidate_reads"][candidate]
            groups[_key(value)].append(topo)
        for key in sorted(groups):
            group = groups[key]
            if len(group) <= 1:
                continue
            if not continuation:
                return {
                    "g2_unit_count": n,
                    "candidate_value": group[0]["candidate_reads"][candidate],
                    "topology_canons": [str(x["topology_canon"]) for x in group[:4]],
                }
            sigs: dict[str, list[str]] = {}
            for topo in group:
                sig = sorted(str(x["topology_canon"]) for x in topo["leaf_attachment_children"])
                sigs[str(topo["topology_canon"])] = sig
            unique = {_key(v) for v in sigs.values()}
            if len(unique) > 1:
                return {
                    "g2_unit_count": n,
                    "candidate_value": group[0]["candidate_reads"][candidate],
                    "topology_child_signatures": [
                        {"topology_canon": c, "child_topology_set": sigs[c]}
                        for c in sorted(sigs)[:4]
                    ],
                }
    return None


def _candidate_audit(rows: Sequence[Mapping[str, Any]], r0_result: Mapping[str, Any]) -> dict[str, Any]:
    spec = r1_spec()
    ordered = list(spec["candidate_read_audit"]["ordered_candidates"])
    witness = r0_result["caps7_topology_separation_witness"]
    A = witness["topology_A"]
    B = witness["topology_B"]
    results: list[dict[str, Any]] = []
    for name in ordered:
        a_val = _candidate_value(name, metrics=A["metrics"], topology_canon=str(A["topology_canon"]))
        b_val = _candidate_value(name, metrics=B["metrics"], topology_canon=str(B["topology_canon"]))
        sep_collision = _first_collision(rows, name, continuation=False)
        cont_collision = _first_collision(rows, name, continuation=True)
        results.append({
            "candidate": name,
            "complexity_tier": int(spec["candidate_read_audit"]["complexity_tiers"][name]),
            "separates_exact_r0_n4_path_star_witness": a_val != b_val,
            "witness_value_A": a_val,
            "witness_value_B": b_val,
            "topology_separating_through_max_rank": sep_collision is None,
            "one_step_continuation_sufficient_through_max_rank_minus_one": cont_collision is None,
            "first_topology_collision": sep_collision,
            "first_continuation_collision": cont_collision,
        })
    passing = [r for r in results if r["topology_separating_through_max_rank"] and r["one_step_continuation_sufficient_through_max_rank_minus_one"]]
    if not passing:
        selected = None
        selected_tier = None
    else:
        selected_tier = min(int(r["complexity_tier"]) for r in passing)
        tier = [r["candidate"] for r in passing if int(r["complexity_tier"]) == selected_tier]
        selected = sorted(tier)
    out = {
        "schema_id": "IG_G3_R1_CANDIDATE_TREE_READ_AUDIT_V1",
        "status": "PASS" if passing else "NO_COMPACT_CANDIDATE_FOUND",
        "scope": {
            "topology_separation_through_g2_unit_count": int(spec["shape_transition_census"]["max_g2_units"]),
            "one_step_continuation_through_parent_g2_unit_count": int(spec["shape_transition_census"]["max_g2_units"]) - 1,
        },
        "candidate_results": results,
        "selected_lowest_complexity_tier": selected_tier,
        "selected_candidates": selected,
        "selection_rule": spec["candidate_read_audit"]["selection_rule"],
        "minimality_scope": "MINIMAL_ONLY_WITHIN_FROZEN_CANDIDATE_LIBRARY_AND_COMPLEXITY_TIERS; NO_GLOBAL_MINIMALITY_CLAIM",
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _first_fiber_branching(rows: Sequence[Mapping[str, Any]], base_keys: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    for row in rows:
        n = int(row["g2_unit_count"])
        if n < 1 or n >= max(base_keys):
            continue
        for topo in row["topologies"]:
            children = topo["leaf_attachment_children"]
            if len(children) > 1:
                return {
                    "parent_g2_unit_count": n,
                    "child_g2_unit_count": n + 1,
                    "parent_topology_canon": topo["topology_canon"],
                    "parent_public_resource_base": base_keys[n],
                    "child_public_resource_base": base_keys[n + 1],
                    "distinct_child_topology_count": len(children),
                    "child_topologies": children,
                    "meaning": "One identical graduated CAPS7/resource transition has multiple intrinsic topology outcomes; composition on the hidden fiber is relation-valued even though the public G3 law is single-valued on CAPS7.",
                }
    raise G3R1Error("no relation-valued fiber branching found in frozen scope")


def run_g3_r1_fiber_moduli_audit(*, engine: Any, r0_result: Mapping[str, Any], r0_replay: Mapping[str, Any]) -> dict[str, Any]:
    spec = r1_spec()
    auth = verify_r1_authority(r0_result, r0_replay)
    max_n = int(spec["shape_transition_census"]["max_g2_units"])
    shapes = _generate_tree_shapes(max_n)
    rows = _shape_rows(shapes, max_n)

    # Pin the already-certified n<=7 R0 shape census exactly before extending it.
    certified = r0_result["bounded_tree_shape_realisability"]["rows"]
    observed_prefix = [int(r["unlabelled_tree_topology_count"]) for r in rows[: len(certified)]]
    certified_prefix = [int(r["unlabelled_tree_topology_count"]) for r in certified]
    if observed_prefix != certified_prefix:
        raise G3R1Error(f"R1 tree generator failed certified R0 prefix: {observed_prefix} != {certified_prefix}")

    seed_caps = [int(x) for x in r0_result["exact_growth_probe"]["seed_caps7"]]
    op = [int(x) for x in r0_result["exact_growth_probe"]["repeated_g3_bridge_operator"]]
    a, b = op
    required_degree = max_n - 1
    if seed_caps[a] < required_degree or seed_caps[b] < required_degree:
        raise G3R1Error("frozen R0 seed/operator lacks R1 leaf-attachment headroom")
    base_keys = {n: _rank_base_key(seed_caps, op, n) for n in range(1, max_n + 1)}

    n4 = next(r for r in r0_result["exact_growth_probe"]["rows"] if int(r["g2_unit_count"]) == 4)
    exact_fiber = {
        "schema_id": "IG_G3_R1_CERTIFIED_R0_N4_FIBER_STRATIFICATION_V1",
        "status": "PASS",
        "fiber_base": base_keys[4],
        "exact_construction_digest_branch_count": int(n4["exact_branch_count"]),
        "public_caps7_state_count": int(n4["unique_caps7_states"]),
        "coarse_g2_tree_topology_stratum_count": int(n4["unique_coarse_unlabelled_tree_topologies"]),
        "flattened_g1_tree_topology_substratum_count": int(n4["unique_flattened_g1_tree_topologies"]),
        "stratification_chain": "24576 exact construction-digest branches -> 196 flattened G1 topology classes -> 2 coarse G2 topology classes -> 1 CAPS7 plus repeated-bridge resource base",
        "coarse_topology_witness": r0_result["caps7_topology_separation_witness"],
        "stratum_branch_cardinalities": "NOT_RECONSTRUCTED_FROM_R0_CLOSEOUT; R1 DOES NOT REMATERIALIZE THE 24576 EXACT BRANCHES",
    }
    exact_fiber["science_sha256"] = canonical_sha256(exact_fiber)

    candidate = _candidate_audit(rows, r0_result)
    first_branch = _first_fiber_branching(rows, base_keys)

    shape_summary = {
        "schema_id": "IG_G3_R1_TREE_FIBER_GRAFTING_CENSUS_V1",
        "status": "PASS",
        "method": "RECURSIVE_UNLABELLED_TREE_GENERATION_BY_ALL_LEAF_ATTACHMENTS_WITH_CANONICAL_DEDUPLICATION",
        "max_g2_units": max_n,
        "r0_certified_prefix_counts": certified_prefix,
        "unlabelled_tree_counts": [int(r["unlabelled_tree_topology_count"]) for r in rows],
        "transition_edge_counts_n_to_nplus1": [int(r["distinct_leaf_attachment_transition_edges"]) for r in rows[:-1]],
        "all_shapes_generated_from_previous_rank_by_leaf_attachment": True,
        "all_shapes_structurally_realisable_with_frozen_r0_seed_operator": True,
        "seed_ref": r0_result["exact_growth_probe"]["seed_g2_carrier_ref"],
        "seed_caps7": seed_caps,
        "repeated_g3_bridge_operator": op,
        "headroom_required_max_degree": required_degree,
        "rank_public_resource_base_keys": [base_keys[n] for n in range(1, max_n + 1)],
        "rows": rows,
    }
    shape_summary["science_sha256"] = canonical_sha256(shape_summary)

    selected = candidate.get("selected_candidates") or []
    if "SHELL_PROFILE_MULTISET" in selected:
        classification = "G3_HIDDEN_TREE_MODULI_PRESENT_RELATION_VALUED_FIBER_GRAFTING_SHELL_PROFILE_READ_CANDIDATE_EARNED_G4_NOT_STARTED"
    elif selected:
        classification = "G3_HIDDEN_TREE_MODULI_PRESENT_RELATION_VALUED_FIBER_GRAFTING_COMPACT_TREE_READ_CANDIDATE_EARNED_G4_NOT_STARTED"
    else:
        classification = "G3_HIDDEN_TREE_MODULI_PRESENT_RELATION_VALUED_FIBER_GRAFTING_NO_COMPACT_READ_FOUND_ON_FROZEN_LIBRARY_G4_NOT_STARTED"

    result = {
        "schema_id": "IG_G3_R1_FIBER_MODULI_AUDIT_RESULT_V1",
        "status": "PASS",
        "classification": classification,
        "promotion": False,
        "g2_graduation_preserved": True,
        "g3_graduation_preserved": True,
        "r0_complete": True,
        "g4_started": False,
        "authority": auth,
        "frozen_question_sha256": spec["science_sha256"],
        "fiber_definition": {
            "kind": "FINITE_COMBINATORIAL_SET_FIBER_NOT_AN_ALGEBRAIC_VARIETY_SCHEME_OR_STACK",
            "base_coordinates": ["G2_UNIT_COUNT", "CAPS7", "G3_TYPED_BRIDGE_CONSUMPTION_VECTOR"],
            "coarse_strata": "UNLABELLED_INTRINSIC_G2_UNIT_TREE_TOPOLOGY",
            "fine_substrata": "UNLABELLED_FLATTENED_G1_TREE_TOPOLOGY_WITH_G2_INTERNAL_G3_CROSS_GRADING",
            "public_map": "exact hierarchical G3 realization -> CAPS7 plus typed bridge-resource degree",
        },
        "certified_exact_n4_fiber": exact_fiber,
        "fiber_composition_law": {
            "law": "TREE_GRAFT_RELATION: connect one vertex of coarse tree T to one vertex of coarse tree U by one frozen G3 cross edge; for one-unit growth this is all leaf attachments T -> {T+leaf@v}",
            "public_shadow": "C_ab(f,g)=f+g-e_a-e_b is independent of attachment vertex and therefore collapses the topology-grafting relation",
            "first_relation_valued_branching": first_branch,
            "transition_graph_science_sha256": shape_summary["science_sha256"],
        },
        "tree_fiber_grafting_census": shape_summary,
        "candidate_tree_read_audit": candidate,
        "operational_read_recommendation": {
            "status": "CANDIDATE_ONLY_NOT_PROMOTED",
            "selected_candidates": selected,
            "meaning": "If a later G4 observer is allowed to read hidden topology, test the selected R1 candidate(s) before promoting full topology. Current G3 does not read them.",
        },
        "topology_promoted": False,
        "geometry_status": "PHYSICAL_GEOMETRY_NOT_EARNED",
        "moduli_status": "FINITE_COMBINATORIAL_FIBER_STRATIFICATION_EARNED; ALGEBRAIC_MODULI_SPACE_NOT_CLAIMED",
        "next_recommendation": "REVIEW_G3_R1_FIBER_MODULI_AUDIT_BEFORE_G4_PHASE0; IF_G4_IS_DESIGNED_USE_R1_SELECTED_TREE_READ_AS_A_PREREGISTERED_CHALLENGE_READ_NOT_AS_AN_ASSUMED_PUBLIC_STATE",
        "nonclaims": spec["nonclaims"],
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
