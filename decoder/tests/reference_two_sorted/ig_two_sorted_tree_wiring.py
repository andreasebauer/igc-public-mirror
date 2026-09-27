#!/usr/bin/env python3
"""Clean-room formal kernel for the frozen IG tree-composition algebra.

Version 1.21 resolves destination field T as a second anonymous finite
interface shared globally across every row of a relation.

A relation is carried by two finite slot sets:
  I - exposed, connectable port slots;
  U - persistent destination slots.

Rows assign a P/M type to every port slot and a destination atom to every T
slot. A single pair of bijections on I and U acts on the whole relation.
This is deliberately not per-row sorting.

The tree-wiring evaluator is the exact relational semantics of the frozen
binary single-bridge compositor on composition trees:
  product rows; enforce the fixed bridge relation on internal edges; hide
  consumed port slots; join/min/AND scalar fields; carry all destination
  slots by disjoint union; deduplicate.

This kernel is language-independent and clean-room. It is not an independent
re-enumeration of the historical J3 microscopic population.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Hashable, Iterable, Sequence

PortType = tuple[int, int]
Slot = Hashable


def bridge_types(a: PortType, b: PortType) -> bool:
    pa, ma = a
    pb, mb = b
    return ((ma == 0 or (ma & pb) != 0) and
            (mb == 0 or (mb & pa) != 0))


@dataclass(frozen=True, order=True)
class Row:
    label: int
    port_values: tuple[PortType, ...]
    score: int
    destination_values: tuple[int, ...]
    stay: bool

    def serial(self) -> tuple:
        return (
            int(self.label),
            tuple((int(p), int(m)) for p, m in self.port_values),
            int(self.score),
            tuple(int(x) for x in self.destination_values),
            bool(self.stay),
        )


@dataclass(frozen=True)
class LabeledRelation:
    port_slots: tuple[Slot, ...]
    destination_slots: tuple[Slot, ...]
    rows: frozenset[Row]

    def __post_init__(self) -> None:
        if len(set(self.port_slots)) != len(self.port_slots):
            raise ValueError("port slots must be distinct")
        if len(set(self.destination_slots)) != len(self.destination_slots):
            raise ValueError("destination slots must be distinct")
        for row in self.rows:
            if len(row.port_values) != len(self.port_slots):
                raise ValueError("row port arity differs from relation interface")
            if len(row.destination_values) != len(self.destination_slots):
                raise ValueError("row destination arity differs from T interface")

    @classmethod
    def from_rows(
        cls,
        port_slots: Sequence[Slot],
        destination_slots: Sequence[Slot],
        rows: Iterable[Row],
    ) -> "LabeledRelation":
        return cls(tuple(port_slots), tuple(destination_slots), frozenset(rows))

    def port_index(self, slot: Slot) -> int:
        try:
            return self.port_slots.index(slot)
        except ValueError as exc:
            raise KeyError(f"unknown port slot: {slot!r}") from exc

    def reorder(
        self,
        port_order: Sequence[int] | None = None,
        destination_order: Sequence[int] | None = None,
    ) -> "LabeledRelation":
        """Apply one global serialization permutation to every row."""
        qp = tuple(range(len(self.port_slots))) if port_order is None else tuple(port_order)
        qt = tuple(range(len(self.destination_slots))) if destination_order is None else tuple(destination_order)
        if sorted(qp) != list(range(len(self.port_slots))):
            raise ValueError("port_order is not a complete permutation")
        if sorted(qt) != list(range(len(self.destination_slots))):
            raise ValueError("destination_order is not a complete permutation")
        return LabeledRelation(
            tuple(self.port_slots[i] for i in qp),
            tuple(self.destination_slots[i] for i in qt),
            frozenset(
                Row(
                    row.label,
                    tuple(row.port_values[i] for i in qp),
                    row.score,
                    tuple(row.destination_values[i] for i in qt),
                    row.stay,
                )
                for row in self.rows
            ),
        )

    def raw_key(self) -> tuple:
        return (
            len(self.port_slots),
            len(self.destination_slots),
            tuple(sorted(row.serial() for row in self.rows)),
        )

    def orbit_key(self, *, max_exact_slots: int = 8) -> tuple:
        """Canonical whole-relation orbit under Sym(I) x Sym(U).

        This exact brute-force canonicalizer is intended for theorem tests and
        small interfaces. It applies the same port and T permutation to every
        row, preserving cross-row slot correlations.
        """
        n, m = len(self.port_slots), len(self.destination_slots)
        if n > max_exact_slots or m > max_exact_slots:
            raise ValueError(
                f"exact orbit canonicalization limited to {max_exact_slots} slots per sort"
            )
        best: tuple | None = None
        for qp in permutations(range(n)):
            for qt in permutations(range(m)):
                rows = tuple(sorted(
                    (
                        int(row.label),
                        tuple(row.port_values[i] for i in qp),
                        int(row.score),
                        tuple(row.destination_values[i] for i in qt),
                        bool(row.stay),
                    )
                    for row in self.rows
                ))
                key = (n, m, rows)
                if best is None or key < best:
                    best = key
        assert best is not None
        return best

    def isomorphic_to(self, other: "LabeledRelation") -> bool:
        return (
            len(self.port_slots) == len(other.port_slots)
            and len(self.destination_slots) == len(other.destination_slots)
            and len(self.rows) == len(other.rows)
            and self.orbit_key() == other.orbit_key()
        )


def compose_relations(
    left: LabeledRelation,
    left_port: Slot,
    right: LabeledRelation,
    right_port: Slot,
) -> LabeledRelation:
    """Exact frozen binary relation compositor on globally aligned slots."""
    ia = left.port_index(left_port)
    ib = right.port_index(right_port)
    if set(left.port_slots) & set(right.port_slots):
        raise ValueError("input port slot handles must be disjoint")
    if set(left.destination_slots) & set(right.destination_slots):
        raise ValueError("input destination slot handles must be disjoint")

    out_ports = tuple(x for x in left.port_slots if x != left_port) + \
                tuple(x for x in right.port_slots if x != right_port)
    out_destinations = left.destination_slots + right.destination_slots
    out_rows: set[Row] = set()

    for a, b in product(left.rows, right.rows):
        if not bridge_types(a.port_values[ia], b.port_values[ib]):
            continue
        ports = tuple(v for k, v in enumerate(a.port_values) if k != ia) + \
                tuple(v for k, v in enumerate(b.port_values) if k != ib)
        out_rows.add(Row(
            label=int(a.label) | int(b.label),
            port_values=ports,
            score=min(int(a.score), int(b.score)),
            destination_values=a.destination_values + b.destination_values,
            stay=bool(a.stay and b.stay),
        ))
    return LabeledRelation(out_ports, out_destinations, frozenset(out_rows))


@dataclass(frozen=True)
class TreeEdge:
    left_box: int
    left_port: Slot
    right_box: int
    right_port: Slot


@dataclass(frozen=True)
class TreeWiring:
    """Connected pairwise acyclic wiring on boxes.

    Every port participates in at most one internal edge. Unmatched ports are
    the exposed outer interface. This is the tree-generated subsyntax of
    undirected wiring diagrams used by frozen IG composition trees.
    """
    edges: tuple[TreeEdge, ...]

    def validate(self, relations: Sequence[LabeledRelation]) -> None:
        n = len(relations)
        if n == 0:
            raise ValueError("tree wiring needs at least one box")
        if len(self.edges) != n - 1:
            raise ValueError("connected tree wiring must have n-1 internal edges")
        used: set[tuple[int, Slot]] = set()
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        for e in self.edges:
            if not (0 <= e.left_box < n and 0 <= e.right_box < n):
                raise ValueError("edge box index outside range")
            if e.left_box == e.right_box:
                raise ValueError("self-contraction is outside frozen tree syntax")
            if e.left_port not in relations[e.left_box].port_slots:
                raise ValueError("unknown left edge port")
            if e.right_port not in relations[e.right_box].port_slots:
                raise ValueError("unknown right edge port")
            for endpoint in ((e.left_box, e.left_port), (e.right_box, e.right_port)):
                if endpoint in used:
                    raise ValueError("one exposed port cannot be consumed twice")
                used.add(endpoint)
            if not union(e.left_box, e.right_box):
                raise ValueError("box adjacency graph contains a cycle")
        if len({find(i) for i in range(n)}) != 1:
            raise ValueError("box adjacency graph is disconnected")


def _globalize_relation(box: int, rel: LabeledRelation) -> LabeledRelation:
    return LabeledRelation(
        tuple((box, "P", x) for x in rel.port_slots),
        tuple((box, "T", x) for x in rel.destination_slots),
        rel.rows,
    )


def _join_labels(values: Iterable[int]) -> int:
    out = 0
    for x in values:
        out |= int(x)
    return out


def evaluate_tree_flat(
    relations: Sequence[LabeledRelation],
    wiring: TreeWiring,
) -> LabeledRelation:
    """Flat Rel/CSP evaluation of a whole tree wiring."""
    wiring.validate(relations)
    used: set[tuple[int, Slot]] = set()
    edge_indices: list[tuple[int, int, int, int]] = []
    for e in wiring.edges:
        li = relations[e.left_box].port_index(e.left_port)
        ri = relations[e.right_box].port_index(e.right_port)
        edge_indices.append((e.left_box, li, e.right_box, ri))
        used.add((e.left_box, e.left_port))
        used.add((e.right_box, e.right_port))

    outer_origins: list[tuple[int, Slot]] = []
    for b, rel in enumerate(relations):
        for slot in rel.port_slots:
            if (b, slot) not in used:
                outer_origins.append((b, slot))
    out_ports = tuple((b, "P", slot) for b, slot in outer_origins)
    out_destinations = tuple(
        (b, "T", slot)
        for b, rel in enumerate(relations)
        for slot in rel.destination_slots
    )

    out_rows: set[Row] = set()
    for chosen in product(*(rel.rows for rel in relations)):
        if any(
            not bridge_types(chosen[lb].port_values[li], chosen[rb].port_values[ri])
            for lb, li, rb, ri in edge_indices
        ):
            continue
        pvals = tuple(
            chosen[b].port_values[relations[b].port_index(slot)]
            for b, slot in outer_origins
        )
        tvals = tuple(value for row in chosen for value in row.destination_values)
        out_rows.add(Row(
            label=_join_labels(row.label for row in chosen),
            port_values=pvals,
            score=min(row.score for row in chosen),
            destination_values=tvals,
            stay=all(row.stay for row in chosen),
        ))
    return LabeledRelation(out_ports, out_destinations, frozenset(out_rows))


def evaluate_tree_recursive(
    relations: Sequence[LabeledRelation],
    wiring: TreeWiring,
    edge_order: Sequence[int],
) -> LabeledRelation:
    """Evaluate by a declared binary contraction schedule."""
    wiring.validate(relations)
    if sorted(edge_order) != list(range(len(wiring.edges))):
        raise ValueError("edge_order is not a complete edge permutation")

    components: dict[frozenset[int], LabeledRelation] = {
        frozenset({i}): _globalize_relation(i, rel)
        for i, rel in enumerate(relations)
    }

    def component_for(box: int) -> frozenset[int]:
        hits = [key for key in components if box in key]
        if len(hits) != 1:
            raise RuntimeError("box does not belong to exactly one current component")
        return hits[0]

    for ei in edge_order:
        e = wiring.edges[ei]
        ca = component_for(e.left_box)
        cb = component_for(e.right_box)
        if ca == cb:
            raise ValueError("contraction schedule attempted an internal self-edge")
        ra, rb = components.pop(ca), components.pop(cb)
        pa = (e.left_box, "P", e.left_port)
        pb = (e.right_box, "P", e.right_port)
        components[ca | cb] = compose_relations(ra, pa, rb, pb)

    if len(components) != 1:
        raise RuntimeError("tree contraction did not produce one component")
    return next(iter(components.values()))


def rowwise_multiset_projection(rel: LabeledRelation) -> frozenset[tuple]:
    """Deliberately lossy projection used only for correlation kill tests."""
    return frozenset(
        (
            row.label,
            tuple(sorted(row.port_values)),
            row.score,
            tuple(sorted(row.destination_values)),
            row.stay,
        )
        for row in rel.rows
    )


def outer_halfports(relations: Sequence[LabeledRelation], wiring: TreeWiring) -> tuple[tuple[int, Slot], ...]:
    """Return exposed input half-ports in deterministic box/slot order."""
    wiring.validate(relations)
    used = {
        endpoint
        for e in wiring.edges
        for endpoint in ((e.left_box, e.left_port), (e.right_box, e.right_port))
    }
    return tuple(
        (b, slot)
        for b, rel in enumerate(relations)
        for slot in rel.port_slots
        if (b, slot) not in used
    )


def substitute_box(
    host_relations: Sequence[LabeledRelation],
    host_wiring: TreeWiring,
    box: int,
    inner_relations: Sequence[LabeledRelation],
    inner_wiring: TreeWiring,
    host_to_inner_outer: dict[Slot, tuple[int, Slot]],
) -> tuple[tuple[LabeledRelation, ...], TreeWiring]:
    """Graph-substitute an inner tree for one host box.

    ``host_to_inner_outer`` must biject the replaced host box's port slots to
    the exposed half-ports of the inner tree.  The output box order is all
    surviving host boxes, in their original order, followed by all inner
    boxes.  This realizes the restricted operadic substitution used by the
    v1.21 embedding theorem.
    """
    host_wiring.validate(host_relations)
    inner_wiring.validate(inner_relations)
    if not (0 <= box < len(host_relations)):
        raise IndexError("host box outside range")
    host_slots = set(host_relations[box].port_slots)
    inner_outer = set(outer_halfports(inner_relations, inner_wiring))
    if set(host_to_inner_outer) != host_slots:
        raise ValueError("mapping must cover exactly the replaced host interface")
    if set(host_to_inner_outer.values()) != inner_outer or len(host_to_inner_outer) != len(inner_outer):
        raise ValueError("mapping must be a bijection onto the inner outer interface")

    survivors = [i for i in range(len(host_relations)) if i != box]
    host_new = {old: new for new, old in enumerate(survivors)}
    inner_offset = len(survivors)
    inner_new = {old: inner_offset + old for old in range(len(inner_relations))}

    new_relations = tuple(host_relations[i] for i in survivors) + tuple(inner_relations)
    new_edges: list[TreeEdge] = []

    # Inner tree edges survive with shifted box indices.
    for e in inner_wiring.edges:
        new_edges.append(TreeEdge(
            inner_new[e.left_box], e.left_port,
            inner_new[e.right_box], e.right_port,
        ))

    def translate(endpoint: tuple[int, Slot]) -> tuple[int, Slot]:
        b, slot = endpoint
        if b != box:
            return host_new[b], slot
        ib, islot = host_to_inner_outer[slot]
        return inner_new[ib], islot

    # Host edges survive, with incidences at the replaced box redirected to
    # the corresponding exposed inner half-port.
    for e in host_wiring.edges:
        lb, lp = translate((e.left_box, e.left_port))
        rb, rp = translate((e.right_box, e.right_port))
        new_edges.append(TreeEdge(lb, lp, rb, rp))

    result = TreeWiring(tuple(new_edges))
    result.validate(new_relations)
    return new_relations, result


def relation_as_host_box(
    evaluated_inner: LabeledRelation,
    host_port_slots: Sequence[Slot],
    inner_outer_order: Sequence[tuple[int, Slot]],
) -> LabeledRelation:
    """Rename/reorder an evaluated inner boundary to a host-box interface."""
    if len(host_port_slots) != len(inner_outer_order):
        raise ValueError("host and inner outer interfaces have different arity")
    expected = tuple((b, "P", slot) for b, slot in inner_outer_order)
    if set(expected) != set(evaluated_inner.port_slots):
        raise ValueError("inner evaluated outer interface mismatch")
    order = tuple(evaluated_inner.port_slots.index(x) for x in expected)
    aligned = evaluated_inner.reorder(port_order=order)
    return LabeledRelation(
        tuple(host_port_slots),
        tuple(("nestedT", i) for i in range(len(aligned.destination_slots))),
        aligned.rows,
    )
