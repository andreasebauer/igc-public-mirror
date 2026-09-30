#!/usr/bin/env python3
"""Finite-set cospan model for the restricted TreeMatch syntax.

The module formalizes the exact syntax embedding used by the Infinity Grid
Formal Foundation v2.1 theorem.  It contains no J3 enumeration and no physics.

A UWD-style operation from inner interfaces (I_v) to outer interface O is a
finite-set cospan

    O -> C <- disjoint_union_v I_v,

where C is the cable set.  The TreeMatch image is the restricted class in
which every cable has one of exactly two fibre shapes:

* two inner half-ports and no outer port (an internal matched pair);
* one inner half-port and one outer port (an exposed leg).

The box adjacency graph of the first kind of cables must be a tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from ig_two_sorted_tree_wiring import (
    LabeledRelation,
    TreeEdge,
    TreeWiring,
    outer_halfports,
    substitute_box,
)

HalfPort = tuple[int, Hashable]
Cable = Hashable
OuterSlot = Hashable


@dataclass(frozen=True)
class FiniteSetCospan:
    """A concrete finite-set cospan presentation of an undirected wiring."""

    input_interfaces: tuple[tuple[Hashable, ...], ...]
    outer_slots: tuple[OuterSlot, ...]
    cables: tuple[Cable, ...]
    input_to_cable: tuple[tuple[HalfPort, Cable], ...]
    outer_to_cable: tuple[tuple[OuterSlot, Cable], ...]

    def __post_init__(self) -> None:
        if len(set(self.cables)) != len(self.cables):
            raise ValueError("cable handles must be distinct")
        all_inputs = {
            (box, slot)
            for box, interface in enumerate(self.input_interfaces)
            for slot in interface
        }
        input_map = dict(self.input_to_cable)
        outer_map = dict(self.outer_to_cable)
        if set(input_map) != all_inputs:
            raise ValueError("input leg map must cover every inner half-port exactly once")
        if set(outer_map) != set(self.outer_slots):
            raise ValueError("outer map must cover every outer slot exactly once")
        cable_set = set(self.cables)
        if not set(input_map.values()) <= cable_set:
            raise ValueError("input map references unknown cable")
        if not set(outer_map.values()) <= cable_set:
            raise ValueError("outer map references unknown cable")
        # Every declared cable must be used by at least one leg in this restricted model.
        if set(input_map.values()) | set(outer_map.values()) != cable_set:
            raise ValueError("wasted/empty cables are outside the TreeMatch image")

    @property
    def input_map(self) -> dict[HalfPort, Cable]:
        return dict(self.input_to_cable)

    @property
    def outer_map(self) -> dict[OuterSlot, Cable]:
        return dict(self.outer_to_cable)

    def fibre_signature(self) -> tuple:
        """Canonical restricted-image signature up to cable/outer renaming.

        Input box and input-slot handles are retained.  Outer slots are anonymous,
        so only the number of outer incidences on each cable is recorded.
        """
        im = self.input_map
        om = self.outer_map
        items = []
        for cable in self.cables:
            inner = tuple(sorted((repr(h[0]), repr(h[1])) for h, c in im.items() if c == cable))
            outer_count = sum(1 for _, c in om.items() if c == cable)
            items.append((inner, outer_count))
        return tuple(sorted(items))

    def isomorphic_restricted(self, other: "FiniteSetCospan") -> bool:
        return (
            tuple(tuple(map(repr, x)) for x in self.input_interfaces)
            == tuple(tuple(map(repr, x)) for x in other.input_interfaces)
            and self.fibre_signature() == other.fibre_signature()
        )


def tree_to_cospan(
    relations: Sequence[LabeledRelation],
    wiring: TreeWiring,
) -> FiniteSetCospan:
    """Faithful inclusion of TreeMatch into finite-set cospan/UWD syntax."""
    wiring.validate(relations)
    input_interfaces = tuple(tuple(r.port_slots) for r in relations)
    input_map: dict[HalfPort, Cable] = {}
    outer_map: dict[OuterSlot, Cable] = {}
    cables: list[Cable] = []

    for index, edge in enumerate(wiring.edges):
        cable = ("internal", index)
        cables.append(cable)
        input_map[(edge.left_box, edge.left_port)] = cable
        input_map[(edge.right_box, edge.right_port)] = cable

    exposed = outer_halfports(relations, wiring)
    for index, halfport in enumerate(exposed):
        cable = ("outer", index)
        outer_slot = ("O", index, halfport)
        cables.append(cable)
        input_map[halfport] = cable
        outer_map[outer_slot] = cable

    return FiniteSetCospan(
        input_interfaces=input_interfaces,
        outer_slots=tuple(outer_map),
        cables=tuple(cables),
        input_to_cable=tuple(input_map.items()),
        outer_to_cable=tuple(outer_map.items()),
    )


def cospan_to_tree(cospan: FiniteSetCospan) -> TreeWiring:
    """Inverse on the restricted TreeMatch image.

    Raises ValueError when a cable has a fibre shape not generated by TreeMatch,
    or when the recovered internal box graph is not a tree.
    """
    im = cospan.input_map
    om = cospan.outer_map
    edges: list[TreeEdge] = []
    for cable in cospan.cables:
        inner = [h for h, c in im.items() if c == cable]
        outer = [o for o, c in om.items() if c == cable]
        if len(inner) == 2 and len(outer) == 0:
            (a, sa), (b, sb) = inner
            if a == b:
                raise ValueError("self-contraction is outside TreeMatch")
            edges.append(TreeEdge(a, sa, b, sb))
        elif len(inner) == 1 and len(outer) == 1:
            # Exposed leg: no internal edge to add.
            pass
        else:
            raise ValueError(
                "cospan is outside TreeMatch image: expected (2 inner,0 outer) "
                "or (1 inner,1 outer) cable fibre"
            )

    # Build dummy relations carrying the declared input interfaces, solely for
    # the existing TreeWiring validator.
    from ig_two_sorted_tree_wiring import Row

    dummies = tuple(
        LabeledRelation.from_rows(
            interface,
            (),
            [Row(0, tuple((0, 0) for _ in interface), 0, (), True)],
        )
        for interface in cospan.input_interfaces
    )
    tree = TreeWiring(tuple(edges))
    tree.validate(dummies)
    return tree


def tree_edge_signature(wiring: TreeWiring) -> frozenset:
    return frozenset(
        frozenset(((e.left_box, e.left_port), (e.right_box, e.right_port)))
        for e in wiring.edges
    )


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def substitute_cospans_by_pushout(
    host_relations: Sequence[LabeledRelation],
    host_wiring: TreeWiring,
    box: int,
    inner_relations: Sequence[LabeledRelation],
    inner_wiring: TreeWiring,
    host_to_inner_outer: Mapping[Hashable, HalfPort],
) -> FiniteSetCospan:
    """UWD cospan substitution by the finite-set pushout of cable sets.

    The host outer interface remains abstractly the same.  Comparison with the
    directly substituted TreeMatch diagram is therefore made up to outer-slot
    renaming via ``fibre_signature``.
    """
    host_wiring.validate(host_relations)
    inner_wiring.validate(inner_relations)
    host = tree_to_cospan(host_relations, host_wiring)
    inner = tree_to_cospan(inner_relations, inner_wiring)

    if set(host_to_inner_outer) != set(host_relations[box].port_slots):
        raise ValueError("mapping must cover exactly the replaced host interface")
    if set(host_to_inner_outer.values()) != set(outer_halfports(inner_relations, inner_wiring)):
        raise ValueError("mapping must biject to the inner outer interface")

    tagged_host = [("H", c) for c in host.cables]
    tagged_inner = [("I", c) for c in inner.cables]
    uf = _UnionFind(tagged_host + tagged_inner)
    him = host.input_map
    iim = inner.input_map
    iom = inner.outer_map

    # Identify host cables incident with replaced box slots with the inner
    # outer cables corresponding to the declared interface bijection.
    inner_outer_cable_by_halfport: dict[HalfPort, Cable] = {}
    for outer_slot, cable in iom.items():
        # In tree_to_cospan, the third component records the exposed half-port.
        halfport = outer_slot[2]
        inner_outer_cable_by_halfport[halfport] = cable

    for host_slot, inner_halfport in host_to_inner_outer.items():
        uf.union(
            ("H", him[(box, host_slot)]),
            ("I", inner_outer_cable_by_halfport[inner_halfport]),
        )

    survivors = [i for i in range(len(host_relations)) if i != box]
    host_new = {old: new for new, old in enumerate(survivors)}
    inner_offset = len(survivors)
    inner_new = {old: inner_offset + old for old in range(len(inner_relations))}

    classes = {}
    for tagged in tagged_host + tagged_inner:
        root = uf.find(tagged)
        classes.setdefault(root, len(classes))

    def cable_class(tagged):
        return ("P", classes[uf.find(tagged)])

    input_interfaces = tuple(tuple(host_relations[i].port_slots) for i in survivors) + \
                       tuple(tuple(r.port_slots) for r in inner_relations)
    input_map: dict[HalfPort, Cable] = {}

    for old in survivors:
        for slot in host_relations[old].port_slots:
            input_map[(host_new[old], slot)] = cable_class(("H", him[(old, slot)]))
    for old, rel in enumerate(inner_relations):
        for slot in rel.port_slots:
            input_map[(inner_new[old], slot)] = cable_class(("I", iim[(old, slot)]))

    # Host outer slots remain the output interface under operadic substitution.
    outer_map = {
        outer: cable_class(("H", cable))
        for outer, cable in host.outer_map.items()
    }
    cables = tuple(sorted(set(input_map.values()) | set(outer_map.values()), key=repr))

    return FiniteSetCospan(
        input_interfaces=input_interfaces,
        outer_slots=tuple(outer_map),
        cables=cables,
        input_to_cable=tuple(input_map.items()),
        outer_to_cable=tuple(outer_map.items()),
    )


def direct_substitution_cospan(
    host_relations: Sequence[LabeledRelation],
    host_wiring: TreeWiring,
    box: int,
    inner_relations: Sequence[LabeledRelation],
    inner_wiring: TreeWiring,
    host_to_inner_outer: Mapping[Hashable, HalfPort],
) -> FiniteSetCospan:
    relations, wiring = substitute_box(
        host_relations,
        host_wiring,
        box,
        inner_relations,
        inner_wiring,
        dict(host_to_inner_outer),
    )
    return tree_to_cospan(relations, wiring)


# ---------------------------------------------------------------------------
# Backward-compatible public aliases
# ---------------------------------------------------------------------------
# An earlier v2.1 test driver used the longer ``treematch_*`` names and a
# ``canonical_key`` helper.  Keep those names as thin wrappers so every
# scientific script shipped in the compact bundle remains executable.

def _finite_set_cospan_canonical_key(self: FiniteSetCospan) -> tuple:
    return (
        tuple(tuple(map(repr, x)) for x in self.input_interfaces),
        self.fibre_signature(),
    )


# Attach as a method without changing the frozen dataclass fields.
FiniteSetCospan.canonical_key = _finite_set_cospan_canonical_key  # type: ignore[attr-defined]


def _finite_set_cospan_fibres(self: FiniteSetCospan) -> dict:
    """Return cable fibres for compatibility with the earlier v2.1 driver."""
    im = self.input_map
    om = self.outer_map
    return {
        cable: {
            "inner": tuple(h for h, c in im.items() if c == cable),
            "outer": tuple(o for o, c in om.items() if c == cable),
        }
        for cable in self.cables
    }


FiniteSetCospan.fibres = _finite_set_cospan_fibres  # type: ignore[attr-defined]


def treematch_to_cospan(
    relations: Sequence[LabeledRelation],
    wiring: TreeWiring,
) -> FiniteSetCospan:
    """Backward-compatible alias for :func:`tree_to_cospan`."""
    return tree_to_cospan(relations, wiring)


def cospan_to_treematch(
    relations: Sequence[LabeledRelation],
    cospan: FiniteSetCospan,
) -> TreeWiring:
    """Backward-compatible inverse with an explicit interface check."""
    expected = tuple(tuple(r.port_slots) for r in relations)
    if expected != cospan.input_interfaces:
        raise ValueError("relation interfaces do not match cospan inputs")
    return cospan_to_tree(cospan)


def substitute_cospan_pushout(
    host_relations: Sequence[LabeledRelation],
    host_wiring: TreeWiring,
    box: int,
    inner_relations: Sequence[LabeledRelation],
    inner_wiring: TreeWiring,
    host_to_inner_outer: Mapping[Hashable, HalfPort],
) -> FiniteSetCospan:
    """Backward-compatible alias for finite-set pushout substitution."""
    return substitute_cospans_by_pushout(
        host_relations, host_wiring, box,
        inner_relations, inner_wiring, host_to_inner_outer,
    )
