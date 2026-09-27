from __future__ import annotations

"""Exact recursive messages for finite vertex/edge-decorated trees.

This module is intentionally G-level agnostic.  It provides a fixed local recursion
whose *iteration depth* may grow with a finite tree, without changing the message
constructor itself.  Exactness never relies on cryptographic hash collision
assumptions: canonical messages are structural Python tuples.  Hashes may be used
only as audit handles outside this module.
"""

from collections import deque
from functools import lru_cache
from typing import Any, Mapping, Sequence


class DecoratedTreeMessageError(ValueError):
    pass


def freeze_label(value: Any) -> tuple[Any, ...]:
    """Convert a JSON-like decoration to a deterministic, totally comparable tuple.

    Type tags avoid Python's bool/int equality collision.  The supported surface is
    deliberately small and sufficient for the Decoder's algebraic decorations.
    """
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", bool(value))
    if isinstance(value, int):
        return ("int", int(value))
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Mapping):
        return (
            "map",
            tuple(sorted((freeze_label(k), freeze_label(v)) for k, v in value.items())),
        )
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(freeze_label(x) for x in value))
    raise DecoratedTreeMessageError(f"unsupported decoration type: {type(value).__name__}")


def _validated_adjacency(
    n: int,
    edges: Sequence[Sequence[int]],
    edge_labels: Sequence[Any],
) -> list[list[tuple[int, tuple[Any, ...]]]]:
    n = int(n)
    if n < 1:
        raise DecoratedTreeMessageError("tree must contain at least one vertex")
    if len(edges) != len(edge_labels):
        raise DecoratedTreeMessageError("edge/edge-label arity mismatch")
    if len(edges) != n - 1:
        raise DecoratedTreeMessageError("finite tree must have n-1 edges")
    adj: list[list[tuple[int, tuple[Any, ...]]]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for e, lab in zip(edges, edge_labels):
        if len(e) != 2:
            raise DecoratedTreeMessageError("edge must have two endpoints")
        a, b = int(e[0]), int(e[1])
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise DecoratedTreeMessageError("invalid tree edge")
        x, y = (a, b) if a < b else (b, a)
        if (x, y) in seen:
            raise DecoratedTreeMessageError("duplicate tree edge")
        seen.add((x, y))
        flab = freeze_label(lab)
        adj[a].append((b, flab))
        adj[b].append((a, flab))
    # n-1 distinct edges + connectedness is exactly the finite-tree condition.
    q = deque([0])
    reached = {0}
    while q:
        v = q.popleft()
        for u, _ in adj[v]:
            if u not in reached:
                reached.add(u)
                q.append(u)
    if len(reached) != n:
        raise DecoratedTreeMessageError("graph is disconnected")
    return adj



class PreparedDecoratedTree:
    """Validated exact decorated-tree view with shared structural message caches.

    The cache stores the full structural Python tuples, never cryptographic hashes.
    Therefore cache lookup cannot turn a hash collision into scientific equality:
    Python mappings always confirm full key equality, and returned messages remain
    the exact canonical tuples used by the historical implementation.
    """

    def __init__(
        self,
        n: int,
        edges: Sequence[Sequence[int]],
        vertex_labels: Sequence[Any],
        edge_labels: Sequence[Any],
    ) -> None:
        self.n = int(n)
        if len(vertex_labels) != self.n:
            raise DecoratedTreeMessageError("vertex-label arity mismatch")
        self.adj = _validated_adjacency(self.n, edges, edge_labels)
        self.vlabs = tuple(freeze_label(x) for x in vertex_labels)

        @lru_cache(maxsize=None)
        def truncated(v: int, parent: int, remaining: int) -> tuple[Any, ...]:
            if remaining == 0:
                return ("V", self.vlabs[v], tuple())
            children = tuple(
                sorted(
                    (elab, truncated(u, v, remaining - 1))
                    for u, elab in self.adj[v]
                    if u != parent
                )
            )
            return ("V", self.vlabs[v], children)

        @lru_cache(maxsize=None)
        def full(v: int, parent: int) -> tuple[Any, ...]:
            children = tuple(
                sorted(
                    (elab, full(u, v))
                    for u, elab in self.adj[v]
                    if u != parent
                )
            )
            return ("V", self.vlabs[v], children)

        self._truncated = truncated
        self._full = full

    def rooted_truncated(self, root: int, depth: int) -> tuple[Any, ...]:
        root = int(root)
        depth = int(depth)
        if not (0 <= root < self.n) or depth < 0:
            raise DecoratedTreeMessageError("invalid root/depth")
        # A finite n-vertex tree has no simple branch longer than n-1 edges.
        # Beyond that depth the exact non-backtracking message is already stable,
        # so normalization keeps the per-tree cache mathematically bounded.
        depth = min(depth, self.n - 1)
        return self._truncated(root, -1, depth)

    def rooted_canon(self, root: int) -> tuple[Any, ...]:
        root = int(root)
        if not (0 <= root < self.n):
            raise DecoratedTreeMessageError("invalid root")
        return self._full(root, -1)

    def all_rooted_canons(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(self.rooted_canon(v) for v in range(self.n))

    def rooted_canon_bag(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(self.all_rooted_canons()))

    def unrooted_canon(self) -> tuple[Any, ...]:
        return min(self.all_rooted_canons())


def prepare_decorated_tree(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
) -> PreparedDecoratedTree:
    """Prepare one exact tree when several roots/depths will be queried."""
    return PreparedDecoratedTree(n, edges, vertex_labels, edge_labels)

def eccentricities(n: int, edges: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return the ordinary graph eccentricity of every vertex of a finite tree.

    This is an intrinsic combinatorial quantity only; no physical/metric meaning is
    attached to it by this module.
    """
    adj = _validated_adjacency(n, edges, [0] * len(edges))
    out: list[int] = []
    for s in range(int(n)):
        dist = [-1] * int(n)
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for u, _ in adj[v]:
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    q.append(u)
        out.append(max(dist))
    return tuple(out)


def rooted_truncated_cavity_message(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
    root: int,
    depth: int,
) -> tuple[Any, ...]:
    """Exact non-backtracking rooted message through ``depth`` edges.

    The local constructor is fixed:
      local vertex label + multiset(edge label, child message).
    Only the number of recursive rounds changes.
    """
    n = int(n)
    root = int(root)
    depth = int(depth)
    if len(vertex_labels) != n:
        raise DecoratedTreeMessageError("vertex-label arity mismatch")
    if not (0 <= root < n) or depth < 0:
        raise DecoratedTreeMessageError("invalid root/depth")
    adj = _validated_adjacency(n, edges, edge_labels)
    vlabs = tuple(freeze_label(x) for x in vertex_labels)

    def rec(v: int, parent: int, remaining: int) -> tuple[Any, ...]:
        if remaining == 0:
            return ("V", vlabs[v], tuple())
        children = tuple(
            sorted(
                (elab, rec(u, v, remaining - 1))
                for u, elab in adj[v]
                if u != parent
            )
        )
        return ("V", vlabs[v], children)

    return rec(root, -1, depth)


def rooted_tree_canon(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
    root: int,
) -> tuple[Any, ...]:
    """Complete rooted-isomorphism invariant for a finite decorated tree.

    This is the fixed-point/cavity message obtained when all branches visible from
    ``root`` have arrived.  Equality is exact structural equality of tuples.
    """
    n = int(n)
    root = int(root)
    if len(vertex_labels) != n:
        raise DecoratedTreeMessageError("vertex-label arity mismatch")
    if not (0 <= root < n):
        raise DecoratedTreeMessageError("invalid root")
    adj = _validated_adjacency(n, edges, edge_labels)
    vlabs = tuple(freeze_label(x) for x in vertex_labels)

    @lru_cache(maxsize=None)
    def directed(v: int, parent: int) -> tuple[Any, ...]:
        children = tuple(
            sorted(
                (elab, directed(u, v))
                for u, elab in adj[v]
                if u != parent
            )
        )
        return ("V", vlabs[v], children)

    return directed(root, -1)


def all_rooted_tree_canons(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
) -> tuple[tuple[Any, ...], ...]:
    """Compute every rooted canon with one shared directed-message cache."""
    n = int(n)
    if len(vertex_labels) != n:
        raise DecoratedTreeMessageError("vertex-label arity mismatch")
    adj = _validated_adjacency(n, edges, edge_labels)
    vlabs = tuple(freeze_label(x) for x in vertex_labels)

    @lru_cache(maxsize=None)
    def directed(v: int, parent: int) -> tuple[Any, ...]:
        children = tuple(
            sorted(
                (elab, directed(u, v))
                for u, elab in adj[v]
                if u != parent
            )
        )
        return ("V", vlabs[v], children)

    return tuple(directed(v, -1) for v in range(n))


def rooted_canon_bag(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
) -> tuple[tuple[Any, ...], ...]:
    """Isomorphism-invariant multiset of exact rooted action contexts."""
    return tuple(sorted(all_rooted_tree_canons(n, edges, vertex_labels, edge_labels)))


def unrooted_tree_canon(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
) -> tuple[Any, ...]:
    """Complete unrooted decorated-tree canon via minimum rooted canon."""
    return min(all_rooted_tree_canons(n, edges, vertex_labels, edge_labels))


def graft_fixed_leaf(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    edge_labels: Sequence[Any],
    root: int,
    *,
    new_vertex_label: Any,
    new_edge_label: Any,
) -> tuple[int, tuple[tuple[int, int], ...], tuple[Any, ...], tuple[Any, ...]]:
    """Return the exact decorated tree after adjoining one new leaf at ``root``."""
    # Validate parent before constructing child.
    _validated_adjacency(n, edges, edge_labels)
    if len(vertex_labels) != int(n):
        raise DecoratedTreeMessageError("vertex-label arity mismatch")
    if not (0 <= int(root) < int(n)):
        raise DecoratedTreeMessageError("invalid graft root")
    child_edges = tuple((int(e[0]), int(e[1])) for e in edges) + ((int(root), int(n)),)
    child_vertex_labels = tuple(vertex_labels) + (new_vertex_label,)
    child_edge_labels = tuple(edge_labels) + (new_edge_label,)
    return int(n) + 1, child_edges, child_vertex_labels, child_edge_labels
