from __future__ import annotations

"""Registered non-promoting G2:R0 post-graduation structural reconnaissance.

R0 deliberately reads exact G1-unit incidence topology as a *reconnaissance observer*.
It does not modify the graduated CAPS7 transition grammar, does not promote topology into
G2 state, and does not launch G3.  Its purpose is to determine which intrinsic many-body
structures exist behind the graduated CAPS7 quotient.
"""

from collections import Counter, deque
from importlib.resources import files
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g2_relation import compose_binary_relation, is_g2_composite, relation_spec
from . import regime_scanner as rs


class UpliftR0Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G2_R0_STRUCTURAL_RECON_SPEC_V1.json"


def r0_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if expected != observed:
        raise UpliftR0Error(f"R0 spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def verify_r0_authority(graduation_certificate: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if graduation_certificate.get("schema_id") != "IG_G2_GRADUATION_CERTIFICATE_V1":
        failures.append("CERT_SCHEMA")
    if graduation_certificate.get("status") != "PASS" or graduation_certificate.get("g2_graduated") is not True:
        failures.append("G2_NOT_GRADUATED")
    if graduation_certificate.get("r0_unlocked") is not True:
        failures.append("R0_NOT_UNLOCKED")
    if graduation_certificate.get("authorizes") != "G2:R0_POST_GRADUATION_RECURSIVE_DEPTH_AXIS":
        failures.append("R0_AUTHORIZATION")
    if graduation_certificate.get("graduated_descriptor") != "IG_G2_CAPS7_STATE_V1":
        failures.append("CAPS7_IDENTITY")
    grammar = graduation_certificate.get("graduated_grammar", {})
    if int(grammar.get("binary_operator_count", -1)) != 31 or int(grammar.get("reservation_actions", -1)) != 7:
        failures.append("GRAMMAR_BASIS")
    out = {
        "schema_id": "IG_G2_R0_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "graduation_certificate_science_sha256": graduation_certificate.get("science_sha256"),
        "g2_relation_spec_sha256": relation_spec()["spec_sha256"],
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise UpliftR0Error("R0 authority failed: " + ",".join(failures))
    return out


def _witness_leaf_path(state: Any, witness: Any) -> tuple[int, ...]:
    """Resolve only the G2-owner portion of a reservation witness to one G1-unit path."""
    if not is_g2_composite(state):
        return tuple()
    if not isinstance(witness, tuple) or len(witness) != 3 or not isinstance(witness[0], int):
        raise UpliftR0Error("malformed G2 owner witness")
    i = int(witness[0])
    if i < 0 or i >= len(state.children):
        raise UpliftR0Error("G2 owner witness child outside range")
    return (i,) + _witness_leaf_path(state.children[i], witness[1])


def _leaf_paths(state: Any, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    if not is_g2_composite(state):
        return [prefix]
    out: list[tuple[int, ...]] = []
    for i, child in enumerate(state.children):
        out.extend(_leaf_paths(child, prefix + (i,)))
    return out


def flattened_g1_incidence(state: Any) -> dict[str, Any]:
    """Flatten a recursive G2 carrier to its G1-unit incidence tree.

    G1 internals remain atomic.  G2 owner witnesses are read only by this explicitly
    preregistered non-promoting reconnaissance observer.
    """
    paths = _leaf_paths(state)
    path_to_id = {p: i for i, p in enumerate(paths)}
    edges: list[tuple[int, int, int, int]] = []

    def rec(st: Any, prefix: tuple[int, ...]) -> None:
        if not is_g2_composite(st):
            return
        for i, child in enumerate(st.children):
            rec(child, prefix + (i,))
        for e in st.top_edges_full:
            if len(e) != 6:
                raise UpliftR0Error("unexpected G2 edge shape")
            u, v, a, b, wu, wv = e
            u = int(u); v = int(v); a = int(a); b = int(b)
            lp = prefix + (u,) + _witness_leaf_path(st.children[u], wu)
            rp = prefix + (v,) + _witness_leaf_path(st.children[v], wv)
            if lp not in path_to_id or rp not in path_to_id:
                raise UpliftR0Error("G2 edge witness did not resolve to a G1-unit leaf")
            edges.append((path_to_id[lp], path_to_id[rp], a, b))

    rec(state, tuple())
    return {
        "leaf_paths": [list(p) for p in paths],
        "edges": [list(e) for e in edges],
        "node_count": len(paths),
        "edge_count": len(edges),
    }


def _tree_canon(n: int, pairs: Sequence[tuple[int, int]]) -> str:
    """AHU canonical form for an unlabelled finite tree."""
    if n == 0:
        return "()"
    if n == 1:
        return "(())"
    adj = [set() for _ in range(n)]
    for u, v in pairs:
        u = int(u); v = int(v)
        if u == v:
            raise UpliftR0Error("self-loop in G2 incidence graph")
        adj[u].add(v); adj[v].add(u)
    # Connectedness and tree condition are checked by caller; centers via leaf peeling.
    deg = [len(x) for x in adj]
    leaves = deque(i for i, d in enumerate(deg) if d <= 1)
    remaining = n
    while remaining > 2:
        k = len(leaves)
        remaining -= k
        for _ in range(k):
            u = leaves.popleft()
            for v in adj[u]:
                deg[v] -= 1
                if deg[v] == 1:
                    leaves.append(v)
    centers = list(leaves)

    def rooted(u: int, parent: int) -> str:
        sub = sorted(rooted(v, u) for v in adj[u] if v != parent)
        return "(" + "".join(sub) + ")"

    forms = [rooted(c, -1) for c in centers]
    return min(forms)


def _graph_metrics(n: int, pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    gm = rs._graph_basic(n, pairs)
    shell_profiles = [] if gm.get("shells") is None else [list(x) for x in gm["shells"]]
    return {
        "connected": bool(gm.get("connected")),
        "beta": gm.get("beta"),
        "diameter": gm.get("diameter"),
        "radius": gm.get("radius"),
        "degree_sequence": list(gm.get("degree", [])),
        "articulations": int(gm.get("articulations", 0)),
        "bridges": int(gm.get("bridges", 0)),
        "triangles": int(gm.get("triangles", 0)),
        "distance_sum": gm.get("distance_sum"),
        "shell_profiles": shell_profiles,
    }


def _choose_growth_carrier_and_operator(carriers: Sequence[Any], bridge_pairs: Sequence[tuple[int, int]]) -> tuple[Any, tuple[int, int]]:
    """Outcome-blind choice maximizing reusable endpoint headroom for a homogeneous tree-growth probe."""
    best = None
    for st in carriers:
        caps = tuple(int(x) for x in st.total_caps)
        for a, b in bridge_pairs:
            # headroom for repeated attachment to old nodes and one reservation on each fresh leaf
            score = (min(caps[a], caps[b]), caps[a] + caps[b], sum(caps), -int(a), -int(b), str(st.construction_digest))
            if best is None or score[:-1] > best[0][:-1] or (score[:-1] == best[0][:-1] and score[-1] < best[0][-1]):
                best = (score, st, (int(a), int(b)))
    if best is None or best[0][0] < 2:
        raise UpliftR0Error("no homogeneous R0 growth seed with sufficient endpoint headroom")
    return best[1], best[2]


def incidence_tree_theorem() -> dict[str, Any]:
    out = {
        "schema_id": "IG_G2_R0_INCIDENCE_TREE_THEOREM_V1",
        "status": "PASS",
        "proof_method": "STRUCTURAL_INDUCTION_ON_GRADUATED_G2_BINARY_CONSTRUCTION_SYNTAX",
        "base": "One G1 unit is one vertex and has no G2 incidence edge.",
        "step": "Every admitted G2 binary constructor takes two already-connected disjoint recursive units and adds exactly one typed bridge between one owner leaf on the left and one owner leaf on the right. It performs no deletion, rewiring, or second cross-edge.",
        "conclusion": "Every connected recursively constructed G2 incidence graph on atomic G1 units is a finite tree: m=n-1, beta=0, and every incidence edge is a graph bridge.",
        "scope": "graduated frozen G2 binary grammar only; exact G1 internals are atomic",
        "nonclaims": ["NO_PHYSICAL_SPACE", "NO_DIMENSION", "NO_METRIC_SPACETIME", "NO_G3_PROMOTION"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def run_r0_recon(*, engine: Any, graduation_certificate: Mapping[str, Any]) -> dict[str, Any]:
    spec = r0_spec()
    auth = verify_r0_authority(graduation_certificate)
    theorem = incidence_tree_theorem()
    carriers = engine.ensure_g1_r100_population()
    bridge_pairs = sorted({tuple(map(int, x)) for x in engine.session.bridge_pairs})
    if len(carriers) != 193 or len(bridge_pairs) != 31:
        raise UpliftR0Error("R0 requires complete graduated 193-carrier/31-operator basis")
    seed, op = _choose_growth_carrier_and_operator(carriers, bridge_pairs)
    a, b = op

    states = [seed]
    by_n: list[dict[str, Any]] = []
    topology_witness = None
    all_checks = []
    for n in range(2, int(spec["growth_probe"]["max_g1_units"]) + 1):
        nxt: dict[str, Any] = {}
        for s in states:
            for out in compose_binary_relation(
                engine.session.engine, 100 + n, s, seed, a, b,
                lane="G2_R0_TREE_GROWTH", motif_id=f"G2:R0:TREE:{n}:{a}>{b}",
            ):
                nxt.setdefault(str(out.construction_digest), out)
        if not nxt:
            raise UpliftR0Error(f"R0 tree-growth probe exhausted at n={n}")
        states = [nxt[k] for k in sorted(nxt)]
        topo: dict[str, dict[str, Any]] = {}
        caps_set = set()
        diameter_hist = Counter()
        shell_set = set()
        for st in states:
            inc = flattened_g1_incidence(st)
            pairs = [(int(e[0]), int(e[1])) for e in inc["edges"]]
            gm = _graph_metrics(inc["node_count"], pairs)
            checks = {
                "n": inc["node_count"], "m": inc["edge_count"], "connected": gm["connected"],
                "beta": gm["beta"], "all_edges_bridges": gm["bridges"] == inc["edge_count"],
            }
            all_checks.append(checks)
            if not (checks["n"] == n and checks["m"] == n - 1 and checks["connected"] and checks["beta"] == 0 and checks["all_edges_bridges"]):
                raise UpliftR0Error("R0 incidence-tree invariant failed")
            canon = _tree_canon(n, pairs)
            caps = tuple(int(x) for x in st.total_caps)
            caps_set.add(caps)
            diameter_hist[int(gm["diameter"])] += 1
            shell_key = json.dumps(gm["shell_profiles"], sort_keys=True, separators=(",", ":"))
            shell_set.add(shell_key)
            topo.setdefault(canon, {
                "topology_canon": canon,
                "example_construction_digest": str(st.construction_digest),
                "metrics": gm,
                "caps7": list(caps),
                "typed_edges": inc["edges"],
            })
        row = {
            "g1_unit_count": n,
            "exact_branch_count": len(states),
            "unique_unlabelled_tree_topologies": len(topo),
            "unique_caps7_states": len(caps_set),
            "diameter_histogram": {str(k): v for k, v in sorted(diameter_hist.items())},
            "unique_shell_profile_families": len(shell_set),
            "topologies": [topo[k] for k in sorted(topo)],
        }
        by_n.append(row)
        if topology_witness is None and len(topo) > 1 and len(caps_set) == 1:
            ex = [topo[k] for k in sorted(topo)[:2]]
            topology_witness = {
                "g1_unit_count": n,
                "shared_caps7": ex[0]["caps7"],
                "shared_bridge_operator": [a, b],
                "shared_bridge_multiset_count": n - 1,
                "topology_A": ex[0],
                "topology_B": ex[1],
                "meaning": "Same graduated CAPS7 state and same toric bridge-consumption monomial, but different intrinsic G1-unit incidence topology.",
            }

    if topology_witness is None:
        raise UpliftR0Error("R0 failed to produce a same-CAPS7/different-topology witness")

    max_row = by_n[-1]
    diameter_values = sorted(int(k) for k in max_row["diameter_histogram"])
    result = {
        "schema_id": "IG_G2_R0_STRUCTURAL_RECON_RESULT_V1",
        "status": "PASS",
        "classification": "INTRINSIC_G2_TREE_PREGEOMETRY_PRESENT_CAPS7_AND_TORIC_RESOURCE_GEOMETRY_TOPOLOGY_BLIND",
        "promotion": False,
        "g2_graduated_before_r0": True,
        "g3_started": False,
        "authority": auth,
        "frozen_question_sha256": spec["science_sha256"],
        "incidence_tree_theorem": theorem,
        "growth_probe": {
            "seed_g1_construction_digest": str(seed.construction_digest),
            "seed_caps7": [int(x) for x in seed.total_caps],
            "repeated_bridge_operator": [a, b],
            "rows": by_n,
        },
        "caps7_topology_separation_witness": topology_witness,
        "intrinsic_distance_status": "NONTRIVIAL_GRAPH_DISTANCE_PRESENT",
        "largest_probe_diameter_range": [min(diameter_values), max(diameter_values)],
        "largest_probe_shell_profile_family_count": int(max_row["unique_shell_profile_families"]),
        "cycle_status": "NO_G2_INCIDENCE_CYCLES_UNDER_FROZEN_ONE_CROSS_BINARY_GRAMMAR",
        "toric_comparison": {
            "status": "RESOURCE_TORIC_GEOMETRY_DOES_NOT_DETERMINE_INCIDENCE_TOPOLOGY",
            "basis": "The explicit witness uses the same G1 leaf multiset and the same repeated typed bridge operator, hence identical CAPS7 and identical toric bridge-consumption monomial, while exact G1-unit incidence trees differ.",
            "toric_audit_promotion": False,
        },
        "dimension_status": "NOT_EARNED",
        "geometry_status": "PHYSICAL_GEOMETRY_NOT_EARNED",
        "next_recommendation": "DESIGN_G3_PHASE0_ONLY_AFTER_INTERPRETING_R0; DO_NOT_PROMOTE_INCIDENCE_TO_G2_STATE WITHOUT_A_NEW_OBSERVER/READ_THEOREM",
        "nonclaims": spec["nonclaims"],
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
