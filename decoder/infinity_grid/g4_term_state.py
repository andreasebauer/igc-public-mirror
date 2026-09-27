from __future__ import annotations

"""Compact exact-at-G3-boundary term carrier for G4.

The G-uplift firewall says complete previous-layer carriers are atomic relative to
that layer's earned public interface.  G4 therefore must not reopen G2/G1
implementation archaeology merely to instantiate whole G3 units.  This module
represents a complete graduated G3 construction term by:

* one CAPS7 resource vector on each atomic G2-unit occurrence;
* the retained typed G3 cross-edge incidence between those occurrences.

Relation-valued endpoint reservation is exact for this certified G3 construction
term: every eligible atomic G2-unit occurrence is retained as one lawful owner
branch, exact duplicate resulting *terms* are deduplicated modulo anonymous
G2-unit relabelling, and no hidden descriptor is used as a selector.

This is intentionally NOT a reconstruction of the historical lower-G implementation
carrier.  It is the next-uplift carrier mandated by the structural programme.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import permutations, product
from typing import Any, Iterable, Mapping, Sequence

from .canon import canonical_sha256


class G4TermError(RuntimeError):
    pass


def _caps7(v: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(x) for x in v)
    if len(out) != 7 or any(x < 0 for x in out):
        raise G4TermError("CAPS7 must contain seven nonnegative integers")
    return out


def _edge4(e: Sequence[int]) -> tuple[int, int, int, int]:
    if len(e) != 4:
        raise G4TermError("typed G3 edge must be [u,v,a,b]")
    u, v, a, b = map(int, e)
    if u == v or min(u, v) < 0 or not (0 <= a < 7 and 0 <= b < 7):
        raise G4TermError(f"malformed typed G3 edge {e}")
    return u, v, a, b


def _canonical_edge(u: int, v: int, a: int, b: int) -> tuple[int, int, int, int]:
    return (u, v, a, b) if u < v else (v, u, b, a)


def _colored_term_canon(node_caps: Sequence[Sequence[int]], typed_edges: Sequence[Sequence[int]]) -> tuple[Any, ...]:
    """Exact anonymous-node canonical representation, optimized by node colors.

    The legacy implementation enumerated all ``n!`` node relabelings.  Its
    lexicographic objective compares the node-color tuple *before* the edge
    tuple, so every minimizer must first place node colors in sorted order.
    Therefore only permutations *within equal-color blocks* can possibly win.
    Enumerating just those block permutations is mathematically identical to
    the legacy exhaustive canonicalizer while avoiding the factorial blow-up
    exposed by the n=9 G4 rebase challenge.
    """
    caps = tuple(_caps7(c) for c in node_caps)
    edges = tuple(_edge4(e) for e in typed_edges)
    return _colored_term_canon_cached(caps, edges)


@lru_cache(maxsize=65536)
def _colored_term_canon_cached(
    caps: tuple[tuple[int, ...], ...],
    edges: tuple[tuple[int, int, int, int], ...],
) -> tuple[Any, ...]:
    n = len(caps)
    if n == 0:
        raise G4TermError("G3 term must contain at least one atomic G2 unit")
    if any(max(u, v) >= n for u, v, _a, _b in edges):
        raise G4TermError("typed edge references missing G2-unit occurrence")

    # The first component of the legacy lexicographic objective is minimized
    # uniquely up to permutations among vertices with identical CAPS7 colors.
    sorted_colors = tuple(sorted(caps))
    color_to_old: dict[tuple[int, ...], list[int]] = {}
    for old, color in enumerate(caps):
        color_to_old.setdefault(color, []).append(old)

    blocks: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    pos = 0
    for color in sorted(color_to_old):
        olds = tuple(color_to_old[color])
        positions = tuple(range(pos, pos + len(olds)))
        pos += len(olds)
        blocks.append((color, olds, positions))

    # For each color block, a tuple perm_old gives the old vertices occupying
    # that block's increasing new-label positions.
    perm_iters = [permutations(olds) for _color, olds, _positions in blocks]
    best_edges: tuple[tuple[int, int, int, int], ...] | None = None
    for choices in product(*perm_iters):
        old_to_new = [0] * n
        for (_color, _olds, positions), perm_old in zip(blocks, choices):
            for new, old in zip(positions, perm_old):
                old_to_new[old] = new
        remapped = tuple(sorted(
            _canonical_edge(old_to_new[u], old_to_new[v], a, b)
            for u, v, a, b in edges
        ))
        if best_edges is None or remapped < best_edges:
            best_edges = remapped

    assert best_edges is not None
    return (sorted_colors, best_edges)


def _topology_canon_uncolored(n: int, typed_edges: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    edges = tuple((min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))) for e in typed_edges)
    best = None
    for p in permutations(range(n)):
        rr = tuple(sorted((min(p[u], p[v]), max(p[u], p[v])) for u, v in edges))
        if best is None or rr < best:
            best = rr
    return best or tuple()


@dataclass(frozen=True)
class G3TermState:
    """Complete G3 construction term with G2 units atomic at their CAPS7 interface."""

    node_caps: tuple[tuple[int, ...], ...]
    typed_edges: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_caps", tuple(_caps7(c) for c in self.node_caps))
        object.__setattr__(self, "typed_edges", tuple(_edge4(e) for e in self.typed_edges))
        n = len(self.node_caps)
        if n == 0 or any(max(u, v) >= n for u, v, _a, _b in self.typed_edges):
            raise G4TermError("malformed G3 term")

    @classmethod
    def from_homogeneous_seed(
        cls,
        seed_caps: Sequence[int],
        typed_edges: Sequence[Sequence[int]],
        *,
        unit_count: int,
    ) -> "G3TermState":
        seed = _caps7(seed_caps)
        n = int(unit_count)
        if n <= 0:
            raise G4TermError("unit_count must be positive")
        work = [list(seed) for _ in range(n)]
        edges = tuple(_edge4(e) for e in typed_edges)
        for u, v, a, b in edges:
            if max(u, v) >= n:
                raise G4TermError("typed edge references absent unit")
            if work[u][a] <= 0 or work[v][b] <= 0:
                raise G4TermError("G3 construction term consumes unavailable capacity")
            work[u][a] -= 1
            work[v][b] -= 1
        return cls(tuple(tuple(x) for x in work), edges)

    @cached_property
    def total_caps(self) -> tuple[int, ...]:
        return tuple(sum(c[t] for c in self.node_caps) for t in range(7))

    @cached_property
    def canonical_term(self) -> tuple[Any, ...]:
        return _colored_term_canon(self.node_caps, self.typed_edges)

    @cached_property
    def construction_digest(self) -> str:
        return canonical_sha256({"schema_id": "IG_CERTIFIED_G3_TERM_CANON_V1", "canon": self.canonical_term})

    @cached_property
    def topology_canon_sha256(self) -> str:
        return canonical_sha256({
            "schema_id": "IG_G3_TERM_UNCOLORED_TOPOLOGY_CANON_V1",
            "canon": _topology_canon_uncolored(len(self.node_caps), self.typed_edges),
        })

    def reserve_external_relation(self, endpoint_type: int) -> tuple["G3TermState", ...]:
        t = int(endpoint_type)
        if not (0 <= t < 7):
            raise G4TermError("endpoint type outside seven-type alphabet")
        out: dict[str, G3TermState] = {}
        for i, caps in enumerate(self.node_caps):
            if caps[t] <= 0:
                continue
            nc = [list(x) for x in self.node_caps]
            nc[i][t] -= 1
            st = G3TermState(tuple(tuple(x) for x in nc), self.typed_edges)
            out.setdefault(st.construction_digest, st)
        return tuple(out[k] for k in sorted(out))

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_id": "IG_CERTIFIED_G3_TERM_STATE_V1",
            "node_caps": [list(c) for c in self.node_caps],
            "typed_edges": [list(e) for e in self.typed_edges],
            "total_caps": list(self.total_caps),
            "term_ref": self.construction_digest,
            "topology_canon_sha256": self.topology_canon_sha256,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "G3TermState":
        if wire.get("schema_id") != "IG_CERTIFIED_G3_TERM_STATE_V1":
            raise G4TermError("bad G3 term wire schema")
        st = cls(tuple(tuple(int(x) for x in c) for c in wire["node_caps"]), tuple(tuple(int(x) for x in e) for e in wire["typed_edges"]))
        if list(st.total_caps) != [int(x) for x in wire.get("total_caps", [])]:
            raise G4TermError("G3 term wire total CAPS7 mismatch")
        if st.construction_digest != wire.get("term_ref"):
            raise G4TermError("G3 term wire digest mismatch")
        return st


@dataclass(frozen=True)
class G4PairTermState:
    """One G4 pair connection between two complete G3 construction terms."""

    left: G3TermState
    right: G3TermState
    operator: tuple[int, int]

    def __post_init__(self) -> None:
        a, b = map(int, self.operator)
        if not (0 <= a < 7 and 0 <= b < 7):
            raise G4TermError("bad G4 pair operator")
        object.__setattr__(self, "operator", (a, b))

    @cached_property
    def total_caps(self) -> tuple[int, ...]:
        return tuple(self.left.total_caps[t] + self.right.total_caps[t] for t in range(7))

    @cached_property
    def construction_digest(self) -> str:
        return canonical_sha256({
            "schema_id": "IG_G4_PAIR_TERM_CANON_V1",
            "left": self.left.construction_digest,
            "right": self.right.construction_digest,
            "operator": list(self.operator),
        })

    def reserve_external_relation(self, endpoint_type: int) -> tuple["G4PairTermState", ...]:
        t = int(endpoint_type)
        out: dict[str, G4PairTermState] = {}
        for ls in self.left.reserve_external_relation(t):
            st = G4PairTermState(ls, self.right, self.operator)
            out.setdefault(st.construction_digest, st)
        for rs in self.right.reserve_external_relation(t):
            st = G4PairTermState(self.left, rs, self.operator)
            out.setdefault(st.construction_digest, st)
        return tuple(out[k] for k in sorted(out))


def compose_g4_pair_relation(left: G3TermState, right: G3TermState, a: int, b: int) -> tuple[G4PairTermState, ...]:
    """Complete one-cross G4 pair relation over atomic previous-layer G3 units."""
    a, b = int(a), int(b)
    if not (0 <= a < 7 and 0 <= b < 7):
        raise G4TermError("operator outside seven-type alphabet")
    if left.total_caps[a] <= 0 or right.total_caps[b] <= 0:
        return tuple()
    out: dict[str, G4PairTermState] = {}
    for ls in left.reserve_external_relation(a):
        for rs in right.reserve_external_relation(b):
            st = G4PairTermState(ls, rs, (a, b))
            out.setdefault(st.construction_digest, st)
    return tuple(out[k] for k in sorted(out))


def derive_homogeneous_seed_caps(*, final_caps: Sequence[int], unit_count: int, typed_edges: Iterable[Sequence[int]]) -> tuple[int, ...]:
    """Invert the known homogeneous G3 one-cross construction resource accounting."""
    final = _caps7(final_caps)
    n = int(unit_count)
    if n <= 0:
        raise G4TermError("unit_count must be positive")
    consumed = [0] * 7
    for u, v, a, b in (_edge4(e) for e in typed_edges):
        consumed[a] += 1
        consumed[b] += 1
    seed = []
    for t in range(7):
        num = final[t] + consumed[t]
        if num % n:
            raise G4TermError(f"homogeneous seed CAPS7 not integral in coordinate {t}")
        seed.append(num // n)
    return _caps7(seed)


@dataclass(frozen=True)
class G4TripleTermState:
    """One recursive G4 P3 assembly: complete pair branch plus third G3 term."""

    pair: G4PairTermState
    third: G3TermState
    operator: tuple[int, int]

    def __post_init__(self) -> None:
        a, b = map(int, self.operator)
        if not (0 <= a < 7 and 0 <= b < 7):
            raise G4TermError("bad G4 recursive operator")
        object.__setattr__(self, "operator", (a, b))

    @cached_property
    def total_caps(self) -> tuple[int, ...]:
        return tuple(self.pair.total_caps[t] + self.third.total_caps[t] for t in range(7))

    @cached_property
    def construction_digest(self) -> str:
        return canonical_sha256({
            "schema_id": "IG_G4_TRIPLE_TERM_CANON_V1",
            "pair": self.pair.construction_digest,
            "third": self.third.construction_digest,
            "operator": list(self.operator),
        })

    def reserve_external_relation(self, endpoint_type: int) -> tuple["G4TripleTermState", ...]:
        t = int(endpoint_type)
        out: dict[str, G4TripleTermState] = {}
        for ps in self.pair.reserve_external_relation(t):
            st = G4TripleTermState(ps, self.third, self.operator)
            out.setdefault(st.construction_digest, st)
        for ts in self.third.reserve_external_relation(t):
            st = G4TripleTermState(self.pair, ts, self.operator)
            out.setdefault(st.construction_digest, st)
        return tuple(out[k] for k in sorted(out))


def compose_g4_recursive_triple_relation(
    pair_relation: Sequence[G4PairTermState],
    third: G3TermState,
    a: int,
    b: int,
) -> tuple[G4TripleTermState, ...]:
    """Compose every exact pair branch with one third G3 term, retaining all owner branches."""
    a, b = int(a), int(b)
    if not (0 <= a < 7 and 0 <= b < 7):
        raise G4TermError("operator outside seven-type alphabet")
    if third.total_caps[b] <= 0:
        return tuple()
    third_rel = third.reserve_external_relation(b)
    out: dict[str, G4TripleTermState] = {}
    for pair in pair_relation:
        if pair.total_caps[a] <= 0:
            continue
        for ps in pair.reserve_external_relation(a):
            for ts in third_rel:
                st = G4TripleTermState(ps, ts, (a, b))
                out.setdefault(st.construction_digest, st)
    return tuple(out[k] for k in sorted(out))
