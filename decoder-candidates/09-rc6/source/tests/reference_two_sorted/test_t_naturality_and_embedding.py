#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ig_two_sorted_tree_wiring import (
    LabeledRelation,
    Row,
    TreeEdge,
    TreeWiring,
    compose_relations,
    evaluate_tree_flat,
    evaluate_tree_recursive,
    rowwise_multiset_projection,
    outer_halfports,
    substitute_box,
    relation_as_host_box,
)

OUT = Path(__file__).parents[1] / "05_RESULTS" / "T_NATURALITY_AND_TREE_WIRING_TESTS_v2.1.json"
TESTS: list[dict] = []


def check(name: str, condition: bool, detail: dict | None = None) -> None:
    if not condition:
        raise AssertionError(name)
    TESTS.append({"test": name, "status": "PASS", "detail": detail or {}})


# 1. T is a second globally anonymous interface, not a per-row bag.
R = LabeledRelation.from_rows(
    ("p0", "p1"),
    ("t0", "t1", "t2"),
    [
        Row(1, ((1, 4), (4, 0)), 2, (5, 290, 1518), True),
        Row(2, ((4, 0), (1, 4)), 1, (290, 5, 1518), False),
    ],
)
for qp in permutations(range(2)):
    for qt in permutations(range(3)):
        Rp = R.reorder(qp, qt)
        check(f"whole_relation_global_orbit_qp{qp}_qt{qt}", R.isomorphic_to(Rp))

# T-specific correlation witness: independent row-bag projection collapses
# two non-isomorphic relations.
T_sym = LabeledRelation.from_rows(
    (), ("u0", "u1"),
    [Row(0, (), 1, (5, 290), True), Row(0, (), 1, (290, 5), True)],
)
T_asym = LabeledRelation.from_rows(
    (), ("u0", "u1"),
    [Row(0, (), 1, (5, 290), True)],
)
check(
    "rowwise_T_multiset_projection_is_lossy",
    rowwise_multiset_projection(T_sym) == rowwise_multiset_projection(T_asym)
    and not T_sym.isomorphic_to(T_asym),
    {"sym_rows": len(T_sym.rows), "asym_rows": len(T_asym.rows)},
)

# 2. Binary T naturality under independent global port/T bijections.
A = LabeledRelation.from_rows(
    ("a0", "a1", "a2"), ("au0", "au1"),
    [
        Row(1, ((1, 4), (8, 0), (0, 0)), 2, (5, 290), True),
        Row(2, ((1, 4), (10, 0), (1, 0)), 1, (290, 5), False),
    ],
)
B = LabeledRelation.from_rows(
    ("b0", "b1"), ("bu0",),
    [
        Row(4, ((4, 0), (1, 24)), 2, (1518,), True),
        Row(8, ((4, 0), (0, 0)), 1, (1524,), True),
    ],
)
base = compose_relations(A, "a0", B, "b0")
for qpa in permutations(range(3)):
    for qta in permutations(range(2)):
        for qpb in permutations(range(2)):
            Ap = A.reorder(qpa, qta)
            Bp = B.reorder(qpb, (0,))
            out = compose_relations(Ap, "a0", Bp, "b0")
            check(f"binary_T_naturality_{qpa}_{qta}_{qpb}", base.isomorphic_to(out))

# 3. Exact flat tree wiring = every binary contraction schedule.
C = LabeledRelation.from_rows(
    ("c0", "c1", "c2"), ("cu0", "cu1"),
    [
        Row(16, ((8, 0), (1, 24), (0, 0)), 2, (1761, 3255), True),
        Row(32, ((10, 0), (1, 24), (1, 0)), 1, (3255, 1761), False),
    ],
)
D = LabeledRelation.from_rows(
    ("d0", "d1"), ("du0",),
    [Row(64, ((8, 0), (0, 0)), 2, (3269,), True)],
)
relations = (A, B, C, D)
wiring = TreeWiring((
    TreeEdge(0, "a0", 1, "b0"),
    TreeEdge(0, "a1", 2, "c0"),
    TreeEdge(2, "c1", 3, "d0"),
))
flat = evaluate_tree_flat(relations, wiring)
for order in permutations(range(3)):
    rec = evaluate_tree_recursive(relations, wiring, order)
    check(f"tree_wiring_flat_equals_recursive_{order}", flat.isomorphic_to(rec), {"rows": len(flat.rows)})

# 4. Random small trees and random global serializations.
rng = random.Random(20260827)
random_cases = 0
for case in range(40):
    rels = []
    for b in range(3):
        if b == 0:
            ps = ("x0", "x1")
            rows = [
                Row(rng.randrange(1, 16), ((1, 4), rng.choice([(0,0),(1,0),(8,0)])), rng.randrange(1,3), (rng.choice([0,5,290]),), bool(rng.randrange(2)))
                for _ in range(2)
            ]
        elif b == 1:
            ps = ("y0", "y1", "y2")
            rows = [
                Row(rng.randrange(1, 16), ((4, 0), (1, 4), rng.choice([(0,0),(1,0),(10,0)])), rng.randrange(1,3), (rng.choice([1518,1524]), rng.choice([1761,3255])), bool(rng.randrange(2)))
                for _ in range(2)
            ]
        else:
            ps = ("z0", "z1")
            rows = [
                Row(rng.randrange(1, 16), ((4, 0), rng.choice([(0,0),(1,0),(8,0)])), rng.randrange(1,3), (rng.choice([3269,24232]),), bool(rng.randrange(2)))
                for _ in range(2)
            ]
        rel = LabeledRelation.from_rows(
            ps,
            tuple(f"u{b}_{i}" for i in range(len(rows[0].destination_values))),
            rows,
        )
        qp = list(range(len(rel.port_slots))); rng.shuffle(qp)
        qt = list(range(len(rel.destination_slots))); rng.shuffle(qt)
        rels.append(rel.reorder(qp, qt))
    w = TreeWiring((TreeEdge(0,"x0",1,"y0"), TreeEdge(1,"y1",2,"z0")))
    f = evaluate_tree_flat(rels, w)
    for order in ((0,1),(1,0)):
        r = evaluate_tree_recursive(rels, w, order)
        if not f.isomorphic_to(r):
            raise AssertionError((case, order))
    random_cases += 1
check("random_tree_wiring_factorization", True, {"cases": random_cases, "schedules_each": 2})


# 5. Restricted operadic substitution: nested evaluation = flattened tree.
H0 = LabeledRelation.from_rows(
    ("h00", "h01"), ("h0t",),
    [Row(1, ((4,0),(0,0)), 2, (5,), True), Row(2, ((4,0),(1,0)), 1, (290,), True)],
)
H1_placeholder = LabeledRelation.from_rows(
    ("q0", "q1", "q2"), (),
    [Row(0, ((1,4),(1,24),(0,0)), 2, (), True)],
)
H2 = LabeledRelation.from_rows(
    ("h20", "h21"), ("h2t",),
    [Row(4, ((8,0),(0,0)), 2, (1518,), True), Row(8, ((10,0),(1,0)), 1, (1524,), False)],
)
host_relations = (H0, H1_placeholder, H2)
host_wiring = TreeWiring((
    TreeEdge(0, "h00", 1, "q0"),
    TreeEdge(1, "q1", 2, "h20"),
))
I0 = LabeledRelation.from_rows(
    ("i00", "i01"), ("i0t",),
    [Row(16, ((1,4),(0,0)), 2, (1761,), True), Row(32, ((1,4),(1,0)), 1, (3255,), False)],
)
I1 = LabeledRelation.from_rows(
    ("i10", "i11", "i12"), ("i1t0", "i1t1"),
    [Row(1, ((4,0),(1,24),(0,0)), 2, (3269,24232), True), Row(2, ((4,0),(1,24),(1,0)), 1, (24232,3269), True)],
)
inner_relations = (I0, I1)
inner_wiring = TreeWiring((TreeEdge(0, "i00", 1, "i10"),))
inner_outer = outer_halfports(inner_relations, inner_wiring)
mapping = dict(zip(H1_placeholder.port_slots, inner_outer))
inner_eval = evaluate_tree_flat(inner_relations, inner_wiring)
nested_box = relation_as_host_box(
    inner_eval,
    H1_placeholder.port_slots,
    tuple(mapping[s] for s in H1_placeholder.port_slots),
)
nested_eval = evaluate_tree_flat((H0, nested_box, H2), host_wiring)
flat_relations, flat_wiring = substitute_box(
    host_relations, host_wiring, 1,
    inner_relations, inner_wiring, mapping,
)
flattened_eval = evaluate_tree_flat(flat_relations, flat_wiring)
check(
    "tree_wiring_operadic_substitution",
    nested_eval.isomorphic_to(flattened_eval),
    {
        "nested_rows": len(nested_eval.rows),
        "flattened_rows": len(flattened_eval.rows),
        "flattened_boxes": len(flat_relations),
        "flattened_edges": len(flat_wiring.edges),
    },
)

result = {
    "suite": "IG_V2.1_T_NATURALITY_AND_TREE_WIRING",
    "status": "PASS",
    "tests": TESTS,
    "summary": {
        "tests_passed": len(TESTS),
        "T_model": "one global anonymous destination interface shared by all rows",
        "tree_model": "connected pairwise acyclic undirected wiring",
    },
    "limitations": [
        "Clean-room theorem kernel; not a second enumeration of J3 microscopic populations.",
        "Exact orbit canonicalization is brute-force and intended for small proof/test interfaces.",
        "The theorem covers frozen binary single-bridge composition trees, not cycles, self-contraction, copy/delete, arbitrary junctions, or a full UWD operad.",
        "Full abstraction/minimality remains observer-relative and open.",
    ],
}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
