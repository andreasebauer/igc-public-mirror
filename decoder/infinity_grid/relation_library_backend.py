"""Bounded two-sorted finite relations; BLISS witnesses replay on exact rows.

Separate from owner-graph automorphisms. No behavioral quotient, scalar/value
relabeling, canonical row naming, or production activation is implied.
"""
from dataclasses import dataclass
from itertools import product

MAX_SLOTS = 8
MAX_ROWS = 4096
MAX_VERTICES = 50000
MAX_PAIRS = 1000000


def _nat(x):
    return type(x) is int and x >= 0


@dataclass(frozen=True)
class Relation:
    n_ports: int
    n_destinations: int
    rows: frozenset

    def __post_init__(self):
        if not all(_nat(n) and n <= MAX_SLOTS for n in (self.n_ports, self.n_destinations)):
            raise ValueError('Outside relation slot bounds')
        rows = frozenset(self.rows)
        if len(rows) > MAX_ROWS:
            raise ValueError('Outside relation row bound')
        for r in rows:
            if (type(r) is not tuple or len(r) != 5 or not _nat(r[0])
                    or type(r[1]) is not tuple or len(r[1]) != self.n_ports
                    or not _nat(r[2]) or type(r[3]) is not tuple
                    or len(r[3]) != self.n_destinations or type(r[4]) is not bool
                    or any(type(p) is not tuple or len(p) != 2 or not all(_nat(v) for v in p) for p in r[1])
                    or not all(_nat(v) for v in r[3])):
                raise ValueError('Invalid relation row')
        if self.n_ports + self.n_destinations + len(rows)*(1+self.n_ports+self.n_destinations) > MAX_VERTICES:
            raise ValueError('Outside relation gadget bound')
        object.__setattr__(self, 'rows', rows)


def _permutation(order, n):
    p = tuple(order)
    if any(type(i) is not int for i in p) or sorted(p) != list(range(n)):
        raise ValueError('Not a complete slot permutation')
    return p


def transport(rel, ports, destinations):
    """new[k] = old[order[k]], applied globally to every row."""
    p = _permutation(ports, rel.n_ports)
    d = _permutation(destinations, rel.n_destinations)
    return Relation(rel.n_ports, rel.n_destinations, frozenset(
        (r[0], tuple(r[1][i] for i in p), r[2], tuple(r[3][i] for i in d), r[4]) for r in rel.rows))


def compose(left, right, left_slot, right_slot):
    """Disjoint copies; preserve all lawful rows, with destination concatenation."""
    if (type(left_slot) is not int or type(right_slot) is not int
            or not 0 <= left_slot < left.n_ports or not 0 <= right_slot < right.n_ports):
        raise ValueError('Invalid selected slot')
    np = left.n_ports + right.n_ports - 2
    nd = left.n_destinations + right.n_destinations
    if np > MAX_SLOTS or nd > MAX_SLOTS or len(left.rows)*len(right.rows) > MAX_PAIRS:
        raise ValueError('Outside composition bounds')
    out = set()
    for a, b in product(left.rows, right.rows):
        pa, ma = a[1][left_slot]; pb, mb = b[1][right_slot]
        if (ma and not ma & pb) or (mb and not mb & pa):
            continue
        ports = tuple(v for i, v in enumerate(a[1]) if i != left_slot) + tuple(v for i, v in enumerate(b[1]) if i != right_slot)
        out.add((a[0] | b[0], ports, min(a[2], b[2]), a[3] + b[3], a[4] and b[4]))
        if len(out) > MAX_ROWS:
            raise ValueError('Outside composition row bound')
    return Relation(np, nd, frozenset(out))


def _gadget(rel):
    labels = [('port',)]*rel.n_ports + [('destination',)]*rel.n_destinations
    edges = []
    for row in sorted(rel.rows):
        rv = len(labels); labels.append(('row', row[0], row[2], row[4]))
        for slot, (p, m) in enumerate(row[1]):
            cell = len(labels); labels.append(('port_value', p, m))
            edges.extend(((rv, cell), (cell, slot)))
        for slot, value in enumerate(row[3]):
            cell = len(labels); labels.append(('destination_value', value))
            edges.extend(((rv, cell), (cell, rel.n_ports + slot)))
    return labels, edges


def isomorphism(left, right):
    """Return new-right-slot -> old-left-slot maps, or None; replay every row.

    A common color vocabulary is essential: unrelated values may never acquire
    matching colors merely because each graph numbered its own vocabulary.
    Library failures propagate; there is no fallback to an unqualified method.
    """
    if (left.n_ports, left.n_destinations, len(left.rows)) != (right.n_ports, right.n_destinations, len(right.rows)):
        return None
    import igraph
    la, ea = _gadget(left); lb, eb = _gadget(right)
    vocab = {v: i for i, v in enumerate(sorted(set(la) | set(lb)))}
    ga = igraph.Graph(n=len(la), edges=ea, directed=False)
    gb = igraph.Graph(n=len(lb), edges=eb, directed=False)
    ok, mapping, _ = ga.isomorphic_bliss(gb, color1=[vocab[x] for x in la],
        color2=[vocab[x] for x in lb], return_mapping_12=True)
    if not ok:
        return None
    if (mapping is None or any(type(i) is not int for i in mapping)
            or sorted(mapping) != list(range(len(lb)))
            or any(la[i] != lb[mapping[i]] for i in range(len(la)))
            or {tuple(sorted((mapping[u], mapping[v]))) for u,v in ea} != {tuple(sorted(e)) for e in eb}):
        raise RuntimeError('BLISS graph witness replay failed')
    inverse = [0]*len(mapping)
    for i, j in enumerate(mapping): inverse[j] = i
    ports = tuple(inverse[i] for i in range(left.n_ports))
    destinations = tuple(inverse[left.n_ports+i]-left.n_ports for i in range(left.n_destinations))
    if transport(left, ports, destinations) != right:
        raise RuntimeError('BLISS relation witness replay failed')
    return {'ports': list(ports), 'destinations': list(destinations)}
