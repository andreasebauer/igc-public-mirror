from __future__ import annotations

"""Registered non-promoting G3:R0 post-graduation structural reconnaissance.

G3:R0 asks what intrinsic incidence structure exists behind the graduated G3 CAPS7
quotient. Exact topology is read only by this frozen reconnaissance observer. Nothing in
this module promotes topology into the G3 public state or changes the graduated grammar.
"""

from collections import Counter, deque
from importlib.resources import files
from itertools import product
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g2_relation import compose_binary_relation, is_g2_composite
from .uplift_r0 import flattened_g1_incidence, incidence_tree_theorem
from .uplift_g3_s1 import reproduce_s0_challenge_states
from . import regime_scanner as rs


class G3R0Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G3_R0_STRUCTURAL_RECON_SPEC_V1.json"
_G3_R0_LANE = "G3_R0_TREE_GROWTH"


def r0_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if observed != expected:
        raise G3R0Error(f"G3:R0 spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def verify_r0_authority(
    graduation_certificate: Mapping[str, Any], s0_result: Mapping[str, Any]
) -> dict[str, Any]:
    spec = r0_spec()
    failures: list[str] = []
    if graduation_certificate.get("schema_id") != "IG_G3_GRADUATION_CERTIFICATE_V1":
        failures.append("G3_CERT_SCHEMA")
    if graduation_certificate.get("status") != "PASS" or graduation_certificate.get("g3_graduated") is not True:
        failures.append("G3_NOT_GRADUATED")
    if graduation_certificate.get("r0_unlocked") is not True:
        failures.append("R0_NOT_UNLOCKED")
    if graduation_certificate.get("authorizes") != "G3:R0_POST_GRADUATION_STRUCTURAL_RECONNAISSANCE":
        failures.append("R0_AUTHORIZATION")
    if graduation_certificate.get("graduated_descriptor") != "IG_G3_CAPS7_READ_WRITE_STATE_V1":
        failures.append("G3_DESCRIPTOR")
    if graduation_certificate.get("graduated_observer") != "G3_TRANSITION_GRAMMAR_CAPS7_OBSERVER_V1":
        failures.append("G3_OBSERVER")
    grammar = graduation_certificate.get("graduated_grammar", {})
    if int(grammar.get("binary_operator_count", -1)) != 31 or int(grammar.get("reservation_actions", -1)) != 7:
        failures.append("G3_GRAMMAR_BASIS")
    if grammar.get("routing_semantics") != "RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1":
        failures.append("G3_ROUTING")
    if str(graduation_certificate.get("science_sha256")) != str(spec["authority"]["g3_graduation_certificate_science_sha256"]):
        failures.append("G3_CERT_IDENTITY")

    if s0_result.get("schema_id") != "IG_G3_S0_INTERFACE_EXTRACTION_RESULT_V1" or s0_result.get("status") != "PASS":
        failures.append("S0_SCHEMA_OR_STATUS")
    if str(s0_result.get("science_sha256")) != str(spec["authority"]["g3_s0_science_sha256"]):
        failures.append("S0_IDENTITY")
    if str(s0_result.get("authority", {}).get("g2_r0_science_sha256")) != str(spec["authority"]["g2_r0_science_sha256"]):
        failures.append("G2_R0_IDENTITY")
    corpus = s0_result.get("challenge_corpus", {})
    if int(corpus.get("exact_carrier_count", -1)) != int(spec["authority"]["s0_exact_g2_unit_count"]):
        failures.append("S0_CARRIER_COUNT")
    if int(corpus.get("g1_unit_count", -1)) != int(spec["authority"]["g1_units_per_g2_seed"]):
        failures.append("S0_G1_UNIT_COUNT")

    g2_theorem = incidence_tree_theorem()
    if g2_theorem.get("status") != "PASS" or str(g2_theorem.get("science_sha256")) != str(spec["authority"]["g2_r0_incidence_tree_theorem_science_sha256"]):
        failures.append("G2_TREE_THEOREM_IDENTITY")

    out = {
        "schema_id": "IG_G3_R0_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "g3_graduation_certificate_science_sha256": graduation_certificate.get("science_sha256"),
        "g3_s0_science_sha256": s0_result.get("science_sha256"),
        "g2_r0_science_sha256": s0_result.get("authority", {}).get("g2_r0_science_sha256"),
        "g2_r0_incidence_tree_theorem_science_sha256": g2_theorem.get("science_sha256"),
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G3R0Error("G3:R0 authority failed: " + ",".join(failures))
    return out


def _is_g3_r0_node(state: Any) -> bool:
    return is_g2_composite(state) and str(getattr(state, "lane", "")) == _G3_R0_LANE


def _g2_unit_paths(state: Any, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """Treat every non-R0 recursive G2 carrier as one atomic graduated G2 unit."""
    if not _is_g3_r0_node(state):
        return [prefix]
    out: list[tuple[int, ...]] = []
    for i, child in enumerate(state.children):
        out.extend(_g2_unit_paths(child, prefix + (i,)))
    return out


def _g3_owner_unit_path(state: Any, witness: Any) -> tuple[int, ...]:
    """Resolve only the G3:R0 owner portion of a reservation witness to a G2-unit path."""
    if not _is_g3_r0_node(state):
        return tuple()
    if not isinstance(witness, tuple) or len(witness) != 3 or not isinstance(witness[0], int):
        raise G3R0Error("malformed G3 owner witness")
    i = int(witness[0])
    if i < 0 or i >= len(state.children):
        raise G3R0Error("G3 owner witness child outside range")
    return (i,) + _g3_owner_unit_path(state.children[i], witness[1])


def flattened_g2_unit_incidence(state: Any) -> dict[str, Any]:
    """Contract every graduated G2 base unit to one vertex and retain only G3 cross edges."""
    paths = _g2_unit_paths(state)
    path_to_id = {p: i for i, p in enumerate(paths)}
    edges: list[tuple[int, int, int, int]] = []

    def rec(st: Any, prefix: tuple[int, ...]) -> None:
        if not _is_g3_r0_node(st):
            return
        for i, child in enumerate(st.children):
            rec(child, prefix + (i,))
        for e in st.top_edges_full:
            if len(e) != 6:
                raise G3R0Error("unexpected G3 edge shape")
            u, v, a, b, wu, wv = e
            u = int(u); v = int(v); a = int(a); b = int(b)
            lp = prefix + (u,) + _g3_owner_unit_path(st.children[u], wu)
            rp = prefix + (v,) + _g3_owner_unit_path(st.children[v], wv)
            if lp not in path_to_id or rp not in path_to_id:
                raise G3R0Error("G3 owner witness did not resolve to a graduated G2-unit leaf")
            edges.append((path_to_id[lp], path_to_id[rp], a, b))

    rec(state, tuple())
    return {
        "g2_unit_paths": [list(p) for p in paths],
        "edges": [list(e) for e in edges],
        "node_count": len(paths),
        "edge_count": len(edges),
    }


def flattened_graded_g1_incidence(state: Any) -> dict[str, Any]:
    """Flatten all the way to G1 units and grade each bridge by G2-internal vs G3-cross origin."""
    def leaf_paths(st: Any, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
        if not is_g2_composite(st):
            return [prefix]
        out: list[tuple[int, ...]] = []
        for i, ch in enumerate(st.children):
            out.extend(leaf_paths(ch, prefix + (i,)))
        return out

    def witness_leaf_path(st: Any, witness: Any) -> tuple[int, ...]:
        if not is_g2_composite(st):
            return tuple()
        if not isinstance(witness, tuple) or len(witness) != 3 or not isinstance(witness[0], int):
            raise G3R0Error("malformed recursive owner witness")
        i = int(witness[0])
        if i < 0 or i >= len(st.children):
            raise G3R0Error("recursive owner witness child outside range")
        return (i,) + witness_leaf_path(st.children[i], witness[1])

    paths = leaf_paths(state)
    path_to_id = {p: i for i, p in enumerate(paths)}
    edges: list[dict[str, Any]] = []

    def rec(st: Any, prefix: tuple[int, ...]) -> None:
        if not is_g2_composite(st):
            return
        for i, ch in enumerate(st.children):
            rec(ch, prefix + (i,))
        grade = "G3_CROSS" if _is_g3_r0_node(st) else "G2_INTERNAL"
        for e in st.top_edges_full:
            if len(e) != 6:
                raise G3R0Error("unexpected recursive edge shape")
            u, v, a, b, wu, wv = e
            u = int(u); v = int(v); a = int(a); b = int(b)
            lp = prefix + (u,) + witness_leaf_path(st.children[u], wu)
            rp = prefix + (v,) + witness_leaf_path(st.children[v], wv)
            if lp not in path_to_id or rp not in path_to_id:
                raise G3R0Error("recursive edge witness did not resolve to a G1 leaf")
            edges.append({"u": path_to_id[lp], "v": path_to_id[rp], "a": a, "b": b, "grade": grade})

    rec(state, tuple())
    return {
        "g1_leaf_paths": [list(p) for p in paths],
        "edges": edges,
        "node_count": len(paths),
        "edge_count": len(edges),
        "grade_histogram": dict(sorted(Counter(e["grade"] for e in edges).items())),
    }


def _graph_metrics(n: int, pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    gm = rs._graph_basic(n, pairs)
    return {
        "connected": bool(gm.get("connected")),
        "beta": int(gm.get("beta", 0)) if gm.get("beta") is not None else None,
        "diameter": gm.get("diameter"),
        "radius": gm.get("radius"),
        "degree_sequence": list(gm.get("degree", [])),
        "articulations": int(gm.get("articulations", 0)),
        "bridges": int(gm.get("bridges", 0)),
        "triangles": int(gm.get("triangles", 0)),
        "distance_sum": gm.get("distance_sum"),
        "shell_profiles": [] if gm.get("shells") is None else [list(x) for x in gm["shells"]],
    }


def _tree_canon(n: int, pairs: Sequence[tuple[int, int]]) -> str:
    if n == 0:
        return "()"
    if n == 1:
        return "(())"
    adj = [set() for _ in range(n)]
    for u, v in pairs:
        u = int(u); v = int(v)
        if u == v:
            raise G3R0Error("self-loop in G3 coarse incidence graph")
        adj[u].add(v); adj[v].add(u)
    deg = [len(x) for x in adj]
    leaves = deque(i for i, d in enumerate(deg) if d <= 1)
    remaining = n
    while remaining > 2:
        k = len(leaves)
        if k == 0:
            raise G3R0Error("tree canonicalization received non-tree graph")
        remaining -= k
        for _ in range(k):
            u = leaves.popleft()
            for v in adj[u]:
                deg[v] -= 1
                if deg[v] == 1:
                    leaves.append(v)
    centers = list(leaves)

    def rooted(u: int, parent: int) -> str:
        return "(" + "".join(sorted(rooted(v, u) for v in adj[u] if v != parent)) + ")"

    return min(rooted(c, -1) for c in centers)


def _prufer_pairs(seq: Sequence[int], n: int) -> list[tuple[int, int]]:
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    deg = [1] * n
    for x in seq:
        deg[int(x)] += 1
    edges: list[tuple[int, int]] = []
    for x0 in seq:
        x = int(x0)
        leaf = min(i for i, d in enumerate(deg) if d == 1)
        edges.append((leaf, x))
        deg[leaf] -= 1; deg[x] -= 1
    leaves = [i for i, d in enumerate(deg) if d == 1]
    if len(leaves) != 2:
        raise G3R0Error("Pruefer decoder invariant failed")
    edges.append((leaves[0], leaves[1]))
    return edges


def unlabeled_tree_shape_census(max_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in range(1, int(max_n) + 1):
        shapes: dict[str, dict[str, Any]] = {}
        seqs = [()] if n <= 2 else product(range(n), repeat=n - 2)
        labelled_count = 0
        for seq in seqs:
            labelled_count += 1
            pairs = _prufer_pairs(seq, n)
            canon = _tree_canon(n, pairs)
            if canon not in shapes:
                shapes[canon] = {"topology_canon": canon, "metrics": _graph_metrics(n, pairs), "example_edges": [list(x) for x in pairs]}
        rows.append({
            "g2_unit_count": n,
            "labelled_pruefer_tree_count": labelled_count,
            "unlabelled_tree_topology_count": len(shapes),
            "topologies": [shapes[k] for k in sorted(shapes)],
        })
    return rows


def hierarchical_tree_theorem() -> dict[str, Any]:
    g2 = incidence_tree_theorem()
    out = {
        "schema_id": "IG_G3_R0_HIERARCHICAL_TREE_THEOREM_V1",
        "status": "PASS",
        "proof_method": "TWO_LEVEL_STRUCTURAL_INDUCTION_ON_GRADUATED_G3_BINARY_RELATION_GRAMMAR",
        "coarse_base": "One graduated G2 unit contracts to one vertex and has no G3 cross edge.",
        "coarse_step": "Every admitted G3 binary constructor joins two disjoint connected complete G3 terms with exactly one typed cross bridge between one G2 owner unit on the left and one on the right; no second cross edge, deletion, or rewiring is introduced.",
        "coarse_conclusion": "After contracting each graduated G2 base unit, every finite connected G3 construction has a finite tree incidence graph: m=n-1, beta=0, every G3 cross edge is a bridge.",
        "fine_base_authority": g2["science_sha256"],
        "fine_step": "Each coarse G2 vertex expands to its certified finite G1-unit tree. Replacing vertices by disjoint trees and joining those trees along the one G3 cross edge for every coarse tree edge preserves connectedness and acyclicity.",
        "fine_conclusion": "The full flattened G1-unit graph is also a finite tree, naturally edge-graded into G2-internal and G3-cross relations. Contracting G2-internal edges recovers the coarse G2-unit tree.",
        "scope": "all finite complete G3 relation terms in the certified frozen G3 grammar; topology remains a non-promoting reconnaissance read",
        "nonclaims": ["NO_PHYSICAL_SPACE", "NO_DIMENSION", "NO_METRIC_EMBEDDING", "NO_TIME", "NO_TOPOLOGY_PROMOTION"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _choose_seed_and_operator(by_ref: Mapping[str, Any], bridge_pairs: Sequence[tuple[int, int]]) -> tuple[str, Any, tuple[int, int]]:
    if not by_ref:
        raise G3R0Error("empty certified S0 G2-unit corpus")
    seed_ref = sorted(by_ref)[0]
    seed = by_ref[seed_ref]
    caps = tuple(int(x) for x in seed.total_caps)
    best = None
    for a0, b0 in bridge_pairs:
        a = int(a0); b = int(b0)
        score = (min(caps[a], caps[b]), caps[a] + caps[b], -a, -b)
        if best is None or score > best[0]:
            best = (score, (a, b))
    if best is None:
        raise G3R0Error("no G3:R0 bridge operator")
    return seed_ref, seed, best[1]


def _expected_caps(seed_caps: Sequence[int], n: int, a: int, b: int) -> list[int]:
    out = [int(n) * int(x) for x in seed_caps]
    out[int(a)] -= int(n) - 1
    out[int(b)] -= int(n) - 1
    return out


def run_g3_r0_recon(
    *, engine: Any, graduation_certificate: Mapping[str, Any], s0_result: Mapping[str, Any]
) -> dict[str, Any]:
    spec = r0_spec()
    auth = verify_r0_authority(graduation_certificate, s0_result)
    theorem = hierarchical_tree_theorem()
    by_ref, reproduction = reproduce_s0_challenge_states(engine=engine, s0_result=s0_result)
    bridge_pairs = sorted({tuple(map(int, x)) for x in engine.session.bridge_pairs})
    if len(bridge_pairs) != 31:
        raise G3R0Error("G3:R0 requires the complete frozen 31-operator bridge basis")
    seed_ref, seed, op = _choose_seed_and_operator(by_ref, bridge_pairs)
    a, b = op
    seed_caps = [int(x) for x in seed.total_caps]
    required_headroom = int(spec["shape_census"]["max_g2_units"]) - 1
    if min(seed_caps[a], seed_caps[b]) < required_headroom + 1:
        raise G3R0Error("chosen G3:R0 seed/operator lacks preregistered tree-shape headroom")

    states = [seed]
    rows: list[dict[str, Any]] = []
    topology_witness = None
    max_exact_n = int(spec["exact_growth_probe"]["max_g2_units"])
    for n in range(2, max_exact_n + 1):
        nxt: dict[str, Any] = {}
        for st in states:
            for out in compose_binary_relation(
                engine.session.engine,
                300 + n,
                st,
                seed,
                a,
                b,
                lane=_G3_R0_LANE,
                motif_id=f"G3:R0:TREE:{n}:{a}>{b}",
            ):
                nxt.setdefault(str(out.construction_digest), out)
                if len(nxt) > int(spec["exact_growth_probe"]["hard_exact_state_cap_per_rank"]):
                    raise G3R0Error(f"G3:R0 exact growth exceeded frozen cap at n={n}")
        if not nxt:
            raise G3R0Error(f"G3:R0 exact growth exhausted at n={n}")
        states = [nxt[k] for k in sorted(nxt)]
        coarse_topo: dict[str, dict[str, Any]] = {}
        coarse_caps = set()
        fine_grade_hist = Counter()
        fine_topo_count = Counter()
        for st in states:
            coarse = flattened_g2_unit_incidence(st)
            cpairs = [(int(e[0]), int(e[1])) for e in coarse["edges"]]
            cgm = _graph_metrics(coarse["node_count"], cpairs)
            if not (coarse["node_count"] == n and coarse["edge_count"] == n - 1 and cgm["connected"] and cgm["beta"] == 0 and cgm["bridges"] == coarse["edge_count"]):
                raise G3R0Error("G3:R0 coarse G2-unit tree invariant failed")
            ccanon = _tree_canon(n, cpairs)
            caps = tuple(int(x) for x in st.total_caps)
            coarse_caps.add(caps)

            fine = flattened_graded_g1_incidence(st)
            fpairs = [(int(e["u"]), int(e["v"])) for e in fine["edges"]]
            fgm = _graph_metrics(fine["node_count"], fpairs)
            expected_g1 = int(spec["authority"]["g1_units_per_g2_seed"]) * n
            expected_g2_edges = (int(spec["authority"]["g1_units_per_g2_seed"]) - 1) * n
            expected_g3_edges = n - 1
            gh = fine["grade_histogram"]
            if not (
                fine["node_count"] == expected_g1 and fine["edge_count"] == expected_g1 - 1
                and fgm["connected"] and fgm["beta"] == 0 and fgm["bridges"] == fine["edge_count"]
                and int(gh.get("G2_INTERNAL", 0)) == expected_g2_edges
                and int(gh.get("G3_CROSS", 0)) == expected_g3_edges
            ):
                raise G3R0Error("G3:R0 flattened graded G1 tree invariant failed")
            fine_grade_hist.update({f"G2_INTERNAL={expected_g2_edges};G3_CROSS={expected_g3_edges}": 1})
            fine_topo_count[_tree_canon(fine["node_count"], fpairs)] += 1

            coarse_topo.setdefault(ccanon, {
                "topology_canon": ccanon,
                "example_construction_digest": str(st.construction_digest),
                "metrics": cgm,
                "caps7": list(caps),
                "typed_g3_edges": coarse["edges"],
                "flattened_g1_metrics": fgm,
                "flattened_edge_grade_histogram": gh,
            })
        expected_caps = tuple(_expected_caps(seed_caps, n, a, b))
        if coarse_caps != {expected_caps}:
            raise G3R0Error(f"G3:R0 fixed-rank CAPS7 congruence failed at n={n}")
        row = {
            "g2_unit_count": n,
            "exact_branch_count": len(states),
            "unique_coarse_unlabelled_tree_topologies": len(coarse_topo),
            "unique_caps7_states": len(coarse_caps),
            "expected_caps7": list(expected_caps),
            "unique_flattened_g1_tree_topologies": len(fine_topo_count),
            "flattened_grade_profiles": dict(sorted(fine_grade_hist.items())),
            "coarse_topologies": [coarse_topo[k] for k in sorted(coarse_topo)],
        }
        rows.append(row)
        if topology_witness is None and len(coarse_topo) > 1:
            ex = [coarse_topo[k] for k in sorted(coarse_topo)[:2]]
            topology_witness = {
                "g2_unit_count": n,
                "shared_caps7": ex[0]["caps7"],
                "shared_g3_bridge_operator": [a, b],
                "shared_g3_bridge_multiset_count": n - 1,
                "topology_A": ex[0],
                "topology_B": ex[1],
                "meaning": "Same graduated G3 CAPS7 state and same typed G3 bridge-consumption multiset, but nonisomorphic intrinsic G2-unit incidence trees.",
            }

    if topology_witness is None:
        raise G3R0Error("G3:R0 failed to produce a same-CAPS7/different-coarse-topology witness")

    shape_rows = unlabeled_tree_shape_census(int(spec["shape_census"]["max_g2_units"]))
    expected_counts = [int(x) for x in spec["shape_census"]["expected_unlabelled_tree_counts_n1_to_nmax"]]
    observed_counts = [int(r["unlabelled_tree_topology_count"]) for r in shape_rows]
    if observed_counts != expected_counts:
        raise G3R0Error(f"G3:R0 unlabeled tree census mismatch: {observed_counts}")

    shape_certificate = {
        "schema_id": "IG_G3_R0_BOUNDED_TREE_SHAPE_REALISABILITY_CERTIFICATE_V1",
        "status": "PASS",
        "method": "EXHAUSTIVE_PRUEFER_ENUMERATION_OF_UNLABELLED_TREE_SHAPES_PLUS_LEAF_ADDITION_REALISABILITY_UNDER_FROZEN_ENDPOINT_HEADROOM",
        "max_g2_units": int(spec["shape_census"]["max_g2_units"]),
        "chosen_seed_ref": seed_ref,
        "chosen_operator": [a, b],
        "seed_endpoint_headroom": [seed_caps[a], seed_caps[b]],
        "required_max_tree_degree": int(spec["shape_census"]["max_g2_units"]) - 1,
        "all_shapes_realisable_by_sequential_leaf_attachment": True,
        "fixed_rank_caps7_formula": "n*f_seed - (n-1)e_a - (n-1)e_b",
        "rows": shape_rows,
        "exact_materialisation_boundary": f"Exact recursive relation branches materialised only through n={max_exact_n}; n>{max_exact_n} shape coverage is theorem/certificate, not an exact branch census.",
    }
    shape_certificate["science_sha256"] = canonical_sha256(shape_certificate)

    result = {
        "schema_id": "IG_G3_R0_STRUCTURAL_RECON_RESULT_V1",
        "status": "PASS",
        "classification": "INTRINSIC_G3_TWO_SCALE_GRADED_TREE_PREGEOMETRY_PRESENT_CAPS7_TOPOLOGY_BLIND",
        "promotion": False,
        "g3_graduated_before_r0": True,
        "g3_graduation_preserved": True,
        "g4_started": False,
        "authority": auth,
        "frozen_question_sha256": spec["science_sha256"],
        "s0_exact_challenge_reproduction": reproduction,
        "hierarchical_tree_theorem": theorem,
        "exact_growth_probe": {
            "seed_g2_carrier_ref": seed_ref,
            "seed_caps7": seed_caps,
            "repeated_g3_bridge_operator": [a, b],
            "rows": rows,
        },
        "caps7_topology_separation_witness": topology_witness,
        "bounded_tree_shape_realisability": shape_certificate,
        "intrinsic_structure_status": "NONTRIVIAL_TWO_SCALE_EDGE_GRADED_TREE_STRUCTURE_PRESENT",
        "coarse_distance_status": "INTRINSIC_G2_UNIT_GRAPH_DISTANCE_PRESENT",
        "fine_distance_status": "INTRINSIC_FLATTENED_G1_GRAPH_DISTANCE_PRESENT",
        "contraction_status": "CONTRACTING_G2_INTERNAL_EDGE_COMPONENTS_RECOVERS_COARSE_G3_TREE",
        "cycle_status": "NO_COARSE_G3_OR_FLATTENED_G1_INCIDENCE_CYCLES_UNDER_FROZEN_ONE_CROSS_BINARY_GRAMMAR",
        "topology_visibility": "CAPS7_DOES_NOT_DETERMINE_INTRINSIC_G3_INCIDENCE_TOPOLOGY",
        "topology_promoted": False,
        "dimension_status": "NOT_EARNED",
        "geometry_status": "PHYSICAL_GEOMETRY_NOT_EARNED",
        "next_recommendation": "REVIEW_G3_R0_BEFORE_DESIGNING_G4_PHASE0; DO_NOT_PROMOTE_GRADED_TREE_STRUCTURE_TO_G3_PUBLIC_STATE_WITHOUT_A_NEW_OBSERVER/READ_THEOREM",
        "nonclaims": spec["nonclaims"],
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
