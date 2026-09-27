from __future__ import annotations
"""Independent bounded verifier for G5:R7 using manual graft + independent canon."""
from collections import defaultdict
from itertools import combinations, product
from typing import Any, Mapping

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g5_r3 import independent_unrooted_canon


class G5R7VerificationError(RuntimeError):
    pass


def _ican(ad: G4AcceptedAdapter, t: DecoratedG4Tree):
    return independent_unrooted_canon(t.n, t.edges, [ad._vertex_key(x) for x in t.H_classes], t.edge_operators)


def _q(pub):
    return (tuple(map(int, pub["caps7"])), tuple(sorted((str(k), int(v)) for k, v in pub["H_class_bag"].items())))


def _manual_rel(ad: G4AcceptedAdapter, t: DecoratedG4Tree):
    op = (0, 0)
    local = [list(map(int, ad._rows[k]["caps7"])) for k in t.H_classes]
    for (u, v), (a, b) in zip(t.edges, t.edge_operators):
        local[u][a] -= 1
        local[v][b] -= 1
    nc = list(map(int, ad._rows["C"]["caps7"]))
    out = set()
    for r in range(t.n):
        if local[r][0] <= 0 or nc[0] <= 0:
            continue
        ch = DecoratedG4Tree(t.n + 1, t.edges + ((r, t.n),), t.H_classes + ("C",), t.edge_operators + (op,))
        out.add(_ican(ad, ch))
    return tuple(sorted(out))


def _colors(n: int, c: int):
    for idxs in combinations(range(n), c):
        ss = set(idxs)
        yield tuple("C" if i in ss else "D" for i in range(n))


def _audit(rows):
    ad = G4AcceptedAdapter()
    exact = {}
    for t in rows:
        pub = ad.public_read(t)
        if pub.get("legal"):
            exact.setdefault(_ican(ad, t), t)
    d = defaultdict(dict)
    collisions = 0
    for p, t in sorted(exact.items(), key=lambda x: x[0]):
        q = _q(ad.public_read(t)); rel = _manual_rel(ad, t)
        if rel in d[q] and d[q][rel] != p:
            collisions += 1
        else:
            d[q][rel] = p
    return len(exact), len(d), collisions


def _primary_sample():
    shapes = list(sorted(_generate_tree_shapes(10)[10].items()))
    chosen = [shapes[i][1] for i in (0, 17, 35, 53, 71, 89, 105)]
    rows = []
    for edges in chosen:
        for c in (2, 5, 8):
            for colors in _colors(10, c):
                rows.append(DecoratedG4Tree(10, tuple(edges), tuple(colors), tuple((0, 0) for _ in range(9))))
    return _audit(rows)


def _endpoint_sample():
    shapes = list(_generate_tree_shapes(6)[6].values())
    non = ((0, 1), (1, 0))
    rows = []
    for pos in combinations(range(5), 2):
        for chosen in product(non, repeat=2):
            eo = [(0, 0)] * 5
            eo[pos[0]] = chosen[0]; eo[pos[1]] = chosen[1]
            for edges in shapes:
                for c in (2, 3, 4):
                    for colors in _colors(6, c):
                        rows.append(DecoratedG4Tree(6, tuple(edges), tuple(colors), tuple(eo)))
    return _audit(rows)


def _verify_witness(primary: Mapping[str, Any]):
    w = primary.get("first_collision_witness")
    if w is None:
        return None
    ad = G4AcceptedAdapter()
    def mk(x):
        return DecoratedG4Tree(int(x["n"]), tuple(tuple(e) for e in x["edges"]), tuple(x["H_classes"]), tuple(tuple(o) for o in x["edge_operators"]))
    a = mk(w["parent_a"]); b = mk(w["parent_b"])
    return bool(_ican(ad, a) != _ican(ad, b) and _q(ad.public_read(a)) == _q(ad.public_read(b)) and _manual_rel(ad, a) == _manual_rel(ad, b))


def verify(primary: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    if primary.get("schema_id") != "IG_G5_R7_SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION_RESULT_V1" or primary.get("status") != "PASS":
        failures.append("PRIMARY_SCHEMA_STATUS")
    if primary.get("promotion") is not False or primary.get("g5_graduation_preserved") is not True:
        failures.append("FIREWALL")
    pn, pq, pc = _primary_sample()
    en, eq, ec = _endpoint_sample()
    injective = bool(primary.get("single_payload_separation_survives_larger_fresh_scope"))
    if injective and (pc or ec):
        failures.append("INDEPENDENT_COLLISION_UNDER_INJECTIVE_PRIMARY")
    wv = _verify_witness(primary)
    if primary.get("first_collision_witness") is not None and not wv:
        failures.append("COLLISION_WITNESS_NOT_REPRODUCED")
    out = {
        "schema_id": "IG_G5_R7_INDEPENDENT_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "primary_science_sha256": primary.get("science_sha256"),
        "independent_primary_sample_exact_parent_count": pn,
        "independent_primary_sample_Q_fiber_count": pq,
        "independent_primary_sample_collision_count": pc,
        "independent_endpoint_sample_exact_parent_count": en,
        "independent_endpoint_sample_Q_fiber_count": eq,
        "independent_endpoint_sample_collision_count": ec,
        "primary_collision_witness_verified": wv,
        "manual_single_payload_graft_path_used": True,
        "independent_unrooted_canon_used": True,
        "structural_equality_only": True,
        "failures": failures,
    }
    out["verification_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R7VerificationError("verification failed: " + ",".join(failures))
    return out
