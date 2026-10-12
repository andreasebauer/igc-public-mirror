from __future__ import annotations

"""Exact finite-tree messages with *endpoint-typed* edge decorations.

Unlike :mod:`decorated_tree_messages`, an edge label here is an ordered pair
``(type_at_first_endpoint, type_at_second_endpoint)`` bound to the stored edge
``(u, v)``.  Traversing the same edge from ``v`` to ``u`` therefore presents the
swapped local/remote pair.  This module exists so typed endpoint semantics are
not silently collapsed into an ordinary undirected whole-edge label.

Canonical equality remains exact structural tuple equality; hashes are audit
handles only.
"""

from collections import deque
from functools import lru_cache
from typing import Any, Sequence

from .decorated_tree_messages import DecoratedTreeMessageError, freeze_label


def _validated_endpoint_adjacency(
    n: int,
    edges: Sequence[Sequence[int]],
    endpoint_labels: Sequence[Sequence[Any]],
) -> list[list[tuple[int, tuple[Any, ...]]]]:
    n = int(n)
    if n < 1:
        raise DecoratedTreeMessageError("tree must contain at least one vertex")
    if len(edges) != len(endpoint_labels):
        raise DecoratedTreeMessageError("edge/endpoint-label arity mismatch")
    if len(edges) != n - 1:
        raise DecoratedTreeMessageError("finite tree must have n-1 edges")
    adj: list[list[tuple[int, tuple[Any, ...]]]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for e, lab in zip(edges, endpoint_labels):
        if len(e) != 2:
            raise DecoratedTreeMessageError("edge must have two endpoints")
        if len(lab) != 2:
            raise DecoratedTreeMessageError("endpoint decoration must have two entries")
        a, b = int(e[0]), int(e[1])
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise DecoratedTreeMessageError("invalid tree edge")
        x, y = (a, b) if a < b else (b, a)
        if (x, y) in seen:
            raise DecoratedTreeMessageError("duplicate tree edge")
        seen.add((x, y))
        # Preserve the historical whole-edge tuple encoding in the forward
        # direction, but swap its endpoint entries when traversing backwards.
        # For symmetric operators (a,a) this is byte-for-byte the old message
        # decoration, preserving the frozen R7 projection exactly.
        forward = freeze_label([lab[0], lab[1]])
        reverse = freeze_label([lab[1], lab[0]])
        adj[a].append((b, forward))
        adj[b].append((a, reverse))
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


class PreparedEndpointDecoratedTree:
    """Validated endpoint-typed tree with exact shared structural caches."""

    def __init__(
        self,
        n: int,
        edges: Sequence[Sequence[int]],
        vertex_labels: Sequence[Any],
        endpoint_labels: Sequence[Sequence[Any]],
    ) -> None:
        self.n = int(n)
        if len(vertex_labels) != self.n:
            raise DecoratedTreeMessageError("vertex-label arity mismatch")
        self.adj = _validated_endpoint_adjacency(self.n, edges, endpoint_labels)
        self.vlabs = tuple(freeze_label(x) for x in vertex_labels)

        @lru_cache(maxsize=None)
        def full(v: int, parent: int) -> tuple[Any, ...]:
            children = tuple(sorted((elab, full(u, v)) for u, elab in self.adj[v] if u != parent))
            return ("V", self.vlabs[v], children)

        @lru_cache(maxsize=None)
        def truncated(v: int, parent: int, remaining: int) -> tuple[Any, ...]:
            if remaining == 0:
                return ("V", self.vlabs[v], tuple())
            children = tuple(
                sorted((elab, truncated(u, v, remaining - 1)) for u, elab in self.adj[v] if u != parent)
            )
            return ("V", self.vlabs[v], children)

        self._full = full
        self._truncated = truncated

    def rooted_canon(self, root: int) -> tuple[Any, ...]:
        root = int(root)
        if not (0 <= root < self.n):
            raise DecoratedTreeMessageError("invalid root")
        return self._full(root, -1)

    def cavity_canon(self, vertex: int, excluded_neighbor: int) -> tuple[Any, ...]:
        """Exact directed cavity already used by the frozen full-message recursion."""
        v, parent = int(vertex), int(excluded_neighbor)
        if not (0 <= v < self.n) or not any(u == parent for u, _ in self.adj[v]):
            raise DecoratedTreeMessageError("invalid directed cavity edge")
        return self._full(v, parent)

    def all_rooted_canons(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(self.rooted_canon(v) for v in range(self.n))

    def unrooted_canon(self) -> tuple[Any, ...]:
        return min(self.all_rooted_canons())

    def rooted_truncated(self, root: int, depth: int) -> tuple[Any, ...]:
        root, depth = int(root), int(depth)
        if not (0 <= root < self.n) or depth < 0:
            raise DecoratedTreeMessageError("invalid root/depth")
        depth = min(depth, self.n - 1)
        return self._truncated(root, -1, depth)


def prepare_endpoint_decorated_tree(
    n: int,
    edges: Sequence[Sequence[int]],
    vertex_labels: Sequence[Any],
    endpoint_labels: Sequence[Sequence[Any]],
) -> PreparedEndpointDecoratedTree:
    return PreparedEndpointDecoratedTree(n, edges, vertex_labels, endpoint_labels)


def endpoint_rooted_tree_canon(n, edges, vertex_labels, endpoint_labels, root):
    return prepare_endpoint_decorated_tree(n, edges, vertex_labels, endpoint_labels).rooted_canon(root)


def endpoint_unrooted_tree_canon(n, edges, vertex_labels, endpoint_labels):
    return prepare_endpoint_decorated_tree(n, edges, vertex_labels, endpoint_labels).unrooted_canon()


def endpoint_rooted_truncated_cavity_message(n, edges, vertex_labels, endpoint_labels, root, depth):
    return prepare_endpoint_decorated_tree(n, edges, vertex_labels, endpoint_labels).rooted_truncated(root, depth)
