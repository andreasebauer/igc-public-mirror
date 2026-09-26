from __future__ import annotations

"""Decoder-owned exact tree-relation kernel, engineering checkpoint E1.

Preserves the frozen nested canonical tuples and relation SET semantics. The
engine validates and prepares parent/probe carriers once, groups attachment owners
by COMPLETE rooted canon, and reuses unaffected cavity messages when rerooting a
graft. No subprocesses or durable I/O exist in this kernel.

Caches are execution-only, bounded and scope-local. Python hash-table lookup is
always followed by exact key equality; digests never establish tree equality.
"""

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, fields
from typing import Any, Mapping

from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from .decorated_tree_messages import freeze_label
from .typed_tree_messages import prepare_endpoint_decorated_tree
from .v05_kernel_cache import ExactBoundedLRU, accounted_bytes

# Backward-compatible engineering-test alias; implementation is centralized in v05_kernel_cache.
_accounted_bytes = accounted_bytes


class ExactTreeRelationKernelError(RuntimeError):
    pass


def _integer(value: Any, name: str) -> int:
    # Reject lossy coercion and bool/int aliasing at the new execution boundary.
    if type(value) is not int:
        raise ExactTreeRelationKernelError(f"{name} must be an integer, not {type(value).__name__}")
    return value


def _operator(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ExactTreeRelationKernelError("operator must contain exactly two endpoint types")
    return (_integer(value[0], "operator[0]"), _integer(value[1], "operator[1]"))


def tree_from_record(obj: Mapping[str, Any]) -> DecoratedG4Tree:
    """Decode an immutable carrier without silently truncating or coercing data."""
    try:
        n = _integer(obj["n"], "tree.n")
        edges = tuple(_operator(e) for e in obj["edges"])
        klasses = tuple(obj["H_classes"])
        ops = tuple(_operator(o) for o in obj["edge_operators"])
    except (KeyError, TypeError) as exc:
        raise ExactTreeRelationKernelError("malformed tree record") from exc
    if any(type(k) is not str for k in klasses):
        raise ExactTreeRelationKernelError("H-class labels must be strings")
    return DecoratedG4Tree(n, edges, klasses, ops)



def tree_to_record(tree: DecoratedG4Tree) -> dict[str, Any]:
    """Encode an exact immutable carrier as the canonical service-boundary record."""
    _check_immutable_carrier(tree)
    return {
        "n": int(tree.n),
        "edges": [list(map(int, e)) for e in tree.edges],
        "H_classes": list(tree.H_classes),
        "edge_operators": [list(map(int, op)) for op in tree.edge_operators],
    }


def _thaw_frozen_decoration(value: Any) -> Any:
    if not isinstance(value, tuple) or not value:
        raise ExactTreeRelationKernelError("malformed frozen decoration")
    tag=value[0]
    if tag=="str" and len(value)==2: return str(value[1])
    if tag=="int" and len(value)==2: return int(value[1])
    if tag=="seq" and len(value)==2: return tuple(_thaw_frozen_decoration(x) for x in value[1])
    raise ExactTreeRelationKernelError(f"unsupported frozen decoration tag {tag!r}")


def reconstruct_tree_from_rooted_canon(canon: tuple[Any, ...]) -> DecoratedG4Tree:
    """Kernel-internal exact reconstruction of one representative from a rooted canon."""
    ad=G4AcceptedAdapter()
    table=ad.relation_class_table()
    reverse={key:klass for klass,key,_caps in table}
    if len(reverse)!=len(table):
        raise ExactTreeRelationKernelError("non-unique G4 vertex key table")
    edges:list[tuple[int,int]]=[]; ops:list[tuple[int,int]]=[]; classes:list[str]=[]
    def walk(node: Any) -> int:
        if not isinstance(node,tuple) or len(node)!=3 or node[0]!="V":
            raise ExactTreeRelationKernelError("malformed rooted tree canon")
        key=_thaw_frozen_decoration(node[1])
        if key not in reverse:
            raise ExactTreeRelationKernelError("unknown canonical vertex key")
        idx=len(classes); classes.append(reverse[key])
        children=node[2]
        if not isinstance(children,tuple):
            raise ExactTreeRelationKernelError("malformed rooted children")
        for entry in children:
            if not isinstance(entry,tuple) or len(entry)!=2:
                raise ExactTreeRelationKernelError("malformed rooted branch")
            op=_thaw_frozen_decoration(entry[0])
            if not isinstance(op,tuple) or len(op)!=2:
                raise ExactTreeRelationKernelError("malformed endpoint operator")
            child=walk(entry[1])
            edges.append((idx,child)); ops.append((int(op[0]),int(op[1])))
        return idx
    walk(canon)
    return DecoratedG4Tree(len(classes),tuple(edges),tuple(classes),tuple(ops))


def _component_tree(tree: DecoratedG4Tree, keep: set[int]) -> DecoratedG4Tree:
    _check_immutable_carrier(tree)
    order=sorted(keep); remap={old:i for i,old in enumerate(order)}
    edges=[]; ops=[]
    for (u,v),op in zip(tree.edges,tree.edge_operators):
        if u in keep and v in keep:
            edges.append((remap[u],remap[v])); ops.append(tuple(op))
    return DecoratedG4Tree(len(order),tuple(edges),tuple(tree.H_classes[i] for i in order),tuple(ops))


def split_tree_components(tree: DecoratedG4Tree, edge_index: int) -> tuple[DecoratedG4Tree,DecoratedG4Tree]:
    """Kernel-internal exact connected components after deleting one stored tree edge."""
    _check_immutable_carrier(tree)
    if not 0<=edge_index<len(tree.edges):
        raise ExactTreeRelationKernelError("edge index")
    a,b=tree.edges[edge_index]
    adj=[[] for _ in range(tree.n)]
    for i,(u,v) in enumerate(tree.edges):
        if i==edge_index: continue
        adj[u].append(v); adj[v].append(u)
    seen={a}; q=deque([a])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if u not in seen:
                seen.add(u); q.append(u)
    other=set(range(tree.n))-seen
    if not other or b not in other:
        raise ExactTreeRelationKernelError("split failed")
    return _component_tree(tree,seen),_component_tree(tree,other)


def edge_side_sizes(tree: DecoratedG4Tree) -> tuple[tuple[int,int], ...]:
    """Kernel-internal exact component sizes for every edge deletion."""
    _check_immutable_carrier(tree)
    if tree.n==1: return tuple()
    adj=[[] for _ in range(tree.n)]
    for i,(u,v) in enumerate(tree.edges):
        adj[u].append((v,i)); adj[v].append((u,i))
    parent=[-2]*tree.n; parent_edge=[-1]*tree.n; parent[0]=-1; order=[0]
    for v in order:
        for u,i in adj[v]:
            if parent[u]!=-2: continue
            parent[u]=v; parent_edge[u]=i; order.append(u)
    if len(order)!=tree.n:
        raise ExactTreeRelationKernelError("edge-size traversal disconnected")
    subtree=[1]*tree.n
    for v in reversed(order[1:]): subtree[parent[v]]+=subtree[v]
    out=[None]*len(tree.edges)
    for v in order[1:]:
        i=parent_edge[v]; sv=subtree[v]; a,b=tree.edges[i]
        if v==a: out[i]=(sv,tree.n-sv)
        elif v==b: out[i]=(tree.n-sv,sv)
        else: raise ExactTreeRelationKernelError("edge-size endpoint mismatch")
    if any(x is None for x in out):
        raise ExactTreeRelationKernelError("edge-size coverage mismatch")
    return tuple(out)  # type: ignore[arg-type]


def _check_immutable_carrier(tree: DecoratedG4Tree) -> None:
    if type(tree) is not DecoratedG4Tree:
        raise ExactTreeRelationKernelError("expected immutable DecoratedG4Tree")
    n = _integer(tree.n, "tree.n")
    if n < 1 or type(tree.H_classes) is not tuple or len(tree.H_classes) != n:
        raise ExactTreeRelationKernelError("G4 tree H-class arity/immutability mismatch")
    if any(type(k) is not str for k in tree.H_classes):
        raise ExactTreeRelationKernelError("H-class labels must be strings")
    if type(tree.edges) is not tuple or type(tree.edge_operators) is not tuple:
        raise ExactTreeRelationKernelError("tree edges/operators must be immutable tuples")
    if len(tree.edges) != len(tree.edge_operators):
        raise ExactTreeRelationKernelError("G4 tree edge/operator arity mismatch")
    for e in tree.edges:
        if type(e) is not tuple:
            raise ExactTreeRelationKernelError("tree edges must be immutable pairs")
        _operator(e)
    for op in tree.edge_operators:
        if type(op) is not tuple:
            raise ExactTreeRelationKernelError("tree operators must be immutable pairs")
        _operator(op)


@dataclass(frozen=True, slots=True)
class RelationAuthority:
    # Full frozen values are retained for equality; a digest is not used here.
    adapter_identity: tuple[str, str]
    operators: tuple[tuple[int, int], ...]
    classes: tuple[tuple[str, str, tuple[int, ...]], ...]

    @classmethod
    def from_adapter(cls, adapter: G4AcceptedAdapter) -> RelationAuthority:
        return cls(
            (adapter.DESCRIPTOR.adapter_id, adapter.DESCRIPTOR.version),
            tuple(adapter.operator_basis()),
            adapter.relation_class_table(),
        )


@dataclass(frozen=True, slots=True, init=False)
class PreparedG4RelationTree:
    tree: DecoratedG4Tree
    local_remaining: tuple[tuple[int, ...], ...]
    rooted_canons: tuple[tuple[Any, ...], ...]
    unrooted_canon: tuple[Any, ...]
    legal: bool
    authority: RelationAuthority
    vertex_labels: tuple[tuple[Any, ...], ...]
    # Per vertex: (neighbor, outward endpoint label, reverse label,
    #              exact neighbor->vertex cavity). All nested values immutable.
    branches: tuple[tuple[tuple[Any, ...], ...], ...]
    owners_by_type: tuple[tuple[int, ...], ...]
    representatives_by_type: tuple[tuple[int, ...], ...]
    # Exact component sizes after deleting each stored edge, prepared once.
    # Execution-only acceleration: structural values, never digests.
    edge_side_sizes: tuple[tuple[int, int], ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ExactTreeRelationKernelError("prepared records must be issued by engine preparation")

    def legal_owners(self, endpoint_type: int) -> tuple[int, ...]:
        t = _integer(endpoint_type, "endpoint_type")
        if not 0 <= t < 7:
            raise ExactTreeRelationKernelError(f"endpoint type outside CAPS7: {t}")
        return self.owners_by_type[t]

    def rooted_owner_representatives(self, endpoint_type: int) -> tuple[int, ...]:
        t = _integer(endpoint_type, "endpoint_type")
        if not 0 <= t < 7:
            raise ExactTreeRelationKernelError(f"endpoint type outside CAPS7: {t}")
        return self.representatives_by_type[t]


def _prepared_record(**values: Any) -> PreparedG4RelationTree:
    obj = object.__new__(PreparedG4RelationTree)
    for f in fields(PreparedG4RelationTree):
        object.__setattr__(obj, f.name, values[f.name])
    return obj


@dataclass(frozen=True)
class ExactTreeRelationResult:
    children: tuple[DecoratedG4Tree, ...]
    canons: tuple[tuple[Any, ...], ...]
    attempted_owner_pairs: int
    legal_owner_pairs: int
    rooted_owner_pair_candidates: int
    child_canon_constructions: int
    reused_cavity_branches: int = 0
    reroot_vertices_visited: int = 0


@dataclass(frozen=True)
class ExactTreeRelationProfile:
    exact_outcome_count: int
    attempted_owner_pairs: int
    legal_owner_pairs: int
    rooted_owner_pair_candidates: int
    child_canon_constructions: int
    reused_cavity_branches: int = 0
    reroot_vertices_visited: int = 0


class _ExactStructuralInterner:
    """Execution-local handles for exact recursive structural messages.

    Integer handles are never an equality authority.  A handle is issued only
    after ordinary Python mapping lookup has compared the complete immutable
    structural key, and one handle names exactly one such key for this family
    call.  The handles only keep repeated recursive tuple comparisons shallow.
    """

    __slots__ = ("_by_key", "_next")

    def __init__(self) -> None:
        self._by_key: dict[tuple[Any, ...], int] = {}
        self._next = 1

    def intern(self, key: tuple[Any, ...]) -> int:
        found = self._by_key.get(key)
        if found is not None:
            return found
        token = self._next
        self._next += 1
        self._by_key[key] = token
        return token


@dataclass(slots=True)
class _CompactPreparedTree:
    tree: DecoratedG4Tree
    adjacency: tuple[tuple[tuple[int, tuple[Any, ...]], ...], ...]
    edge_labels: dict[tuple[int, int], tuple[Any, ...]]
    message_ids: dict[tuple[int, int], int]
    root_ids: tuple[int, ...]
    side_sizes: dict[tuple[int, int], int]
    owners_by_type: tuple[tuple[int, ...], ...] | None
    representatives_by_type: tuple[tuple[int, ...], ...] | None


def _compact_prepare_tree(
    tree: DecoratedG4Tree,
    interner: _ExactStructuralInterner,
    frozen_vertex_labels: Mapping[str, tuple[Any, ...]],
    owner_capacities: Mapping[str, tuple[int, ...]] | None = None,
) -> _CompactPreparedTree:
    """Build exact directed messages and, when requested, owner classes once."""
    _check_immutable_carrier(tree)
    adjacency: list[list[tuple[int, tuple[Any, ...]]]] = [[] for _ in range(tree.n)]
    edge_labels: dict[tuple[int, int], tuple[Any, ...]] = {}
    for (u, v), op in zip(tree.edges, tree.edge_operators):
        forward = freeze_label(op)
        reverse = freeze_label((op[1], op[0]))
        adjacency[u].append((v, forward))
        adjacency[v].append((u, reverse))
        edge_labels[(u, v)] = forward
        edge_labels[(v, u)] = reverse

    parent = [-2] * tree.n
    parent[0] = -1
    order = [0]
    for vertex in order:
        for neighbor, _label in adjacency[vertex]:
            if parent[neighbor] != -2:
                continue
            parent[neighbor] = vertex
            order.append(neighbor)
    if len(order) != tree.n:
        raise ExactTreeRelationKernelError("compact message traversal disconnected")

    messages: dict[tuple[int, int], int] = {}
    for vertex in reversed(order):
        entries = []
        for neighbor, label in adjacency[vertex]:
            if parent[neighbor] == vertex:
                entries.append((label, messages[(neighbor, vertex)]))
        if vertex != 0:
            messages[(vertex, parent[vertex])] = interner.intern((
                "V", frozen_vertex_labels[tree.H_classes[vertex]], tuple(sorted(entries)),
            ))

    for vertex in order:
        for child, _child_label in adjacency[vertex]:
            if parent[child] != vertex:
                continue
            entries = []
            for neighbor, label in adjacency[vertex]:
                if neighbor == child:
                    continue
                if parent[neighbor] == vertex or parent[vertex] == neighbor:
                    entries.append((label, messages[(neighbor, vertex)]))
                else:
                    raise ExactTreeRelationKernelError("compact message parent mismatch")
            messages[(vertex, child)] = interner.intern((
                "V", frozen_vertex_labels[tree.H_classes[vertex]], tuple(sorted(entries)),
            ))

    roots = []
    for vertex in range(tree.n):
        entries = tuple(sorted(
            (label, messages[(neighbor, vertex)])
            for neighbor, label in adjacency[vertex]
        ))
        roots.append(interner.intern(("V", frozen_vertex_labels[tree.H_classes[vertex]], entries)))

    subtree = [1] * tree.n
    for vertex in reversed(order[1:]):
        subtree[parent[vertex]] += subtree[vertex]
    side_sizes: dict[tuple[int, int], int] = {}
    for vertex in order[1:]:
        p = parent[vertex]
        side_sizes[(vertex, p)] = subtree[vertex]
        side_sizes[(p, vertex)] = tree.n - subtree[vertex]

    owners_by_type = None
    representatives_by_type = None
    if owner_capacities is not None:
        local = [list(owner_capacities[klass]) for klass in tree.H_classes]
        for (u, v), (a, b) in zip(tree.edges, tree.edge_operators):
            local[u][a] -= 1
            local[v][b] -= 1
        legal = all(value >= 0 for row in local for value in row)
        owners_by_type = tuple(
            tuple(vertex for vertex in range(tree.n) if legal and local[vertex][endpoint_type] > 0)
            for endpoint_type in range(7)
        )
        representatives = []
        for allowed in owners_by_type:
            by_root: dict[int, int] = {}
            for vertex in allowed:
                by_root.setdefault(roots[vertex], vertex)
            representatives.append(tuple(by_root[token] for token in sorted(by_root)))
        representatives_by_type = tuple(representatives)
    return _CompactPreparedTree(
        tree=tree,
        adjacency=tuple(tuple(row) for row in adjacency),
        edge_labels=edge_labels,
        message_ids=messages,
        root_ids=tuple(roots),
        side_sizes=side_sizes,
        owners_by_type=owners_by_type,
        representatives_by_type=representatives_by_type,
    )


def _compact_path(prepared: _CompactPreparedTree, start: int, target: int) -> tuple[int, ...]:
    if start == target:
        return (start,)
    parent = {start: -1}
    queue = deque([start])
    while queue and target not in parent:
        vertex = queue.popleft()
        for neighbor, _label in prepared.adjacency[vertex]:
            if neighbor in parent:
                continue
            parent[neighbor] = vertex
            queue.append(neighbor)
    if target not in parent:
        raise ExactTreeRelationKernelError("compact path disconnected")
    out = [target]
    while out[-1] != start:
        out.append(parent[out[-1]])
    return tuple(reversed(out))


def _compact_modified_residual_root_id(
    prepared: _CompactPreparedTree,
    interner: _ExactStructuralInterner,
    frozen_vertex_labels: Mapping[str, tuple[Any, ...]],
    *, residual_root: int, detached_neighbor: int, attachment: int,
    added_label: tuple[Any, ...], added_root_id: int,
) -> int:
    """Exact rooted message after one cut and one graft, updating only its path."""
    route = _compact_path(prepared, attachment, residual_root)
    previous: int | None = None
    incoming: int | None = None
    for index, vertex in enumerate(route):
        next_vertex = route[index + 1] if index + 1 < len(route) else None
        entries = []
        for neighbor, label in prepared.adjacency[vertex]:
            if next_vertex is not None and neighbor == next_vertex:
                continue
            if vertex == residual_root and neighbor == detached_neighbor:
                continue
            if previous is not None and neighbor == previous:
                if incoming is None:
                    raise ExactTreeRelationKernelError("compact path update missing message")
                message_id = incoming
            else:
                message_id = prepared.message_ids[(neighbor, vertex)]
            entries.append((label, message_id))
        if vertex == attachment:
            entries.append((added_label, added_root_id))
        incoming = interner.intern((
            "V", frozen_vertex_labels[prepared.tree.H_classes[vertex]], tuple(sorted(entries)),
        ))
        previous = vertex
    if incoming is None:
        raise ExactTreeRelationKernelError("compact path update empty")
    return incoming


def _compact_centers(prepared: _CompactPreparedTree) -> tuple[int, ...]:
    if prepared.tree.n <= 2:
        return tuple(range(prepared.tree.n))
    degree = [len(row) for row in prepared.adjacency]
    leaves = [i for i, value in enumerate(degree) if value <= 1]
    remaining = prepared.tree.n
    while remaining > 2:
        remaining -= len(leaves)
        next_leaves = []
        for vertex in leaves:
            for neighbor, _label in prepared.adjacency[vertex]:
                if degree[neighbor] > 0:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        next_leaves.append(neighbor)
            degree[vertex] = 0
        leaves = next_leaves
    return tuple(sorted(leaves))


def _compact_unrooted_signature(
    tree: DecoratedG4Tree,
    interner: _ExactStructuralInterner,
    frozen_vertex_labels: Mapping[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Complete unrooted tree invariant in one shared exact-interner scope."""
    prepared = _compact_prepare_tree(tree, interner, frozen_vertex_labels)
    return ("CENTER_ROOT_IDS", tree.n, tuple(sorted(prepared.root_ids[v] for v in _compact_centers(prepared))))


class ExactTreeRelationKernel:
    """One execution-local authority snapshot and bounded parent/probe cache."""

    def __init__(
        self, *, adapter: G4AcceptedAdapter | None = None,
        scope_identity: str = "STANDALONE_ENGINEERING",
        max_cache_entries: int = 128, max_cache_bytes: int = 32 * 1024 * 1024,
        max_relation_cache_entries: int = 128, max_relation_cache_bytes: int = 32 * 1024 * 1024,
        max_observer_q_cache_entries: int = 256, max_observer_q_cache_bytes: int = 32 * 1024 * 1024,
        max_observer_decode_cache_entries: int = 256, max_observer_decode_cache_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(scope_identity, str) or not scope_identity:
            raise ExactTreeRelationKernelError("scope_identity must be a nonempty string")
        for name, value in (
            ("max_cache_entries", max_cache_entries), ("max_cache_bytes", max_cache_bytes),
            ("max_relation_cache_entries", max_relation_cache_entries), ("max_relation_cache_bytes", max_relation_cache_bytes),
            ("max_observer_q_cache_entries", max_observer_q_cache_entries), ("max_observer_q_cache_bytes", max_observer_q_cache_bytes),
            ("max_observer_decode_cache_entries", max_observer_decode_cache_entries), ("max_observer_decode_cache_bytes", max_observer_decode_cache_bytes),
        ):
            if _integer(value, name) < 0:
                raise ExactTreeRelationKernelError(f"{name} must be nonnegative")
        ad = adapter if adapter is not None else G4AcceptedAdapter()
        self.authority = RelationAuthority.from_adapter(ad)
        self.scope_identity = scope_identity
        self._ops = frozenset(self.authority.operators)
        self._keys = {k: key for k, key, _caps in self.authority.classes}
        self._caps = {k: caps for k, _key, caps in self.authority.classes}
        self._frozen_keys = {k: freeze_label(key) for k, key in self._keys.items()}
        self._endpoint_labels = {op: freeze_label(op) for op in self._ops}
        self.max_cache_entries, self.max_cache_bytes = max_cache_entries, max_cache_bytes
        self._cache: OrderedDict[DecoratedG4Tree, tuple[PreparedG4RelationTree, int]] = OrderedDict()
        self._cache_bytes = 0
        self._relation_cache=ExactBoundedLRU('relation_cache',max_entries=max_relation_cache_entries,max_bytes=max_relation_cache_bytes)
        self._observer_q_cache=ExactBoundedLRU('observer_q_cache',max_entries=max_observer_q_cache_entries,max_bytes=max_observer_q_cache_bytes)
        self._observer_decode_cache=ExactBoundedLRU('observer_decode_cache',max_entries=max_observer_decode_cache_entries,max_bytes=max_observer_decode_cache_bytes)
        self._stats = dict(preparation_hits=0, preparation_misses=0, preparation_evictions=0,
                           preparation_uncached_oversize=0, prepared_parent_constructions=0,
                           relation_profile_zero_fast_hits=0,
                           relation_profile_bridge_fingerprint_fast_hits=0,
                           relation_profile_bridge_fingerprint_fallbacks=0,
                           relation_profile_bridge_fingerprint_canon_avoided=0)

    def metrics(self) -> dict[str, int]:
        out=dict(self._stats, prepared_cache_entries=len(self._cache),
                 prepared_cache_accounted_bytes=self._cache_bytes,
                 prepared_cache_byte_limit=self.max_cache_bytes,
                 prepared_cache_entry_limit=self.max_cache_entries)
        out.update(self._relation_cache.metrics('relation_cache'))
        out.update(self._observer_q_cache.metrics('observer_q_cache'))
        out.update(self._observer_decode_cache.metrics('observer_decode_cache'))
        return out

    def clear(self) -> None:
        self._cache.clear(); self._cache_bytes = 0
        self._relation_cache.clear(); self._observer_q_cache.clear(); self._observer_decode_cache.clear()

    def observer_q_cache_lookup(self, key: Any) -> tuple[bool, Any]:
        return self._observer_q_cache.lookup(key)

    def observer_q_cache_store(self, key: Any, value: Any) -> None:
        self._observer_q_cache.put(key,value)

    def observer_decode_cache_lookup(self, key: Any) -> tuple[bool, Any]:
        return self._observer_decode_cache.lookup(key)

    def observer_decode_cache_store(self, key: Any, value: Any) -> None:
        self._observer_decode_cache.put(key,value)

    def prepare(self, tree: DecoratedG4Tree, *, cache: bool = True) -> PreparedG4RelationTree:
        _check_immutable_carrier(tree)
        if cache and tree in self._cache:
            self._stats["preparation_hits"] += 1
            self._cache.move_to_end(tree)
            return self._cache[tree][0]
        self._stats["preparation_misses"] += 1
        for klass in tree.H_classes:
            if klass not in self._keys:
                raise ExactTreeRelationKernelError(f"unsupported repaired-H class {klass}")
        if any(op not in self._ops for op in tree.edge_operators):
            raise ExactTreeRelationKernelError("operator outside frozen G4 basis")
        prepared = prepare_endpoint_decorated_tree(
            tree.n, tree.edges, [self._keys[k] for k in tree.H_classes], tree.edge_operators,
        )
        local = [list(self._caps[k]) for k in tree.H_classes]
        for (u, v), (a, b) in zip(tree.edges, tree.edge_operators):
            local[u][a] -= 1; local[v][b] -= 1
        legal = all(x >= 0 for row in local for x in row)
        rooted = prepared.all_rooted_canons()
        owners = tuple(tuple(v for v in range(tree.n) if legal and local[v][a] > 0) for a in range(7))
        representatives = []
        for allowed in owners:
            by_root: dict[tuple[Any, ...], int] = {}
            for v in allowed:
                by_root.setdefault(rooted[v], v)
            representatives.append(tuple(by_root[c] for c in sorted(by_root, key=repr)))
        branches = tuple(tuple(
            (u, label, ("seq", (label[1][1], label[1][0])), prepared.cavity_canon(u, v))
            for u, label in prepared.adj[v]
        ) for v in range(tree.n))
        result = _prepared_record(
            tree=tree, local_remaining=tuple(tuple(r) for r in local), rooted_canons=rooted,
            unrooted_canon=min(rooted), legal=legal, authority=self.authority,
            vertex_labels=tuple(self._frozen_keys[k] for k in tree.H_classes), branches=branches,
            owners_by_type=owners, representatives_by_type=tuple(representatives),
            edge_side_sizes=edge_side_sizes(tree),
        )
        self._stats["prepared_parent_constructions"] += 1
        if cache and self.max_cache_entries and self.max_cache_bytes:
            size = accounted_bytes(result)
            if size <= self.max_cache_bytes:
                while self._cache and (len(self._cache) >= self.max_cache_entries or self._cache_bytes + size > self.max_cache_bytes):
                    _, (_, old_size) = self._cache.popitem(last=False)
                    self._cache_bytes -= old_size; self._stats["preparation_evictions"] += 1
                self._cache[tree] = (result, size); self._cache_bytes += size
            else:
                self._stats["preparation_uncached_oversize"] += 1
        return result

    def _coerce(self, value: DecoratedG4Tree | PreparedG4RelationTree) -> PreparedG4RelationTree:
        if isinstance(value, PreparedG4RelationTree):
            if value.authority != self.authority:
                raise ExactTreeRelationKernelError("prepared carrier authority mismatch")
            return value
        return self.prepare(value)

    def enabled(
        self, left: DecoratedG4Tree | PreparedG4RelationTree,
        right: DecoratedG4Tree | PreparedG4RelationTree, operator: tuple[int, int],
    ) -> bool:
        """Exact legality/enabledness without constructing retained children."""
        op=_operator(operator)
        if op not in self._ops:
            raise ExactTreeRelationKernelError(f"operator {op} outside frozen G4 basis")
        lp,rp=self._coerce(left),self._coerce(right)
        return bool(lp.legal_owners(op[0]) and rp.legal_owners(op[1]))

    def _relation_canon_core(
        self, lp: PreparedG4RelationTree, rp: PreparedG4RelationTree, op: tuple[int, int],
    ) -> tuple[tuple[tuple[Any, ...], ...], dict[tuple[Any, ...], tuple[int, int]], int, int, int, int, int]:
        """Compute the exact retained canon set once, without constructing child carriers."""
        attempted = lp.tree.n * rp.tree.n
        legal_pairs = len(lp.legal_owners(op[0])) * len(rp.legal_owners(op[1]))
        if not legal_pairs:
            return (), {}, attempted, 0, 0, 0, 0
        lreps, rreps = lp.rooted_owner_representatives(op[0]), rp.rooted_owner_representatives(op[1])
        candidates = len(lreps) * len(rreps)
        by_canon: dict[tuple[Any, ...], tuple[int, int]] = {}
        forward, reverse = self._endpoint_labels[op], freeze_label((op[1], op[0]))
        for lu in lreps:
            for rv in rreps:
                can = _grafted_canon(lp, rp, lu, rv, forward, reverse)
                by_canon.setdefault(can, (lu, rv))
        canons = tuple(sorted(by_canon, key=repr))
        reused = candidates * (lp.tree.n + rp.tree.n - 2)
        rerooted = candidates * (lp.tree.n + rp.tree.n)
        return canons, by_canon, attempted, legal_pairs, candidates, reused, rerooted

    @staticmethod
    def _directed_branch(
        prepared: PreparedG4RelationTree, residual_vertex: int, detached_vertex: int,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return exact residual->detached edge label and detached rooted cavity canon."""
        for neighbor, label, _reverse, cavity in prepared.branches[int(residual_vertex)]:
            if int(neighbor) == int(detached_vertex):
                return label, cavity
        raise ExactTreeRelationKernelError("prepared branch endpoint mismatch")

    def _new_bridge_fingerprint_is_unique(
        self, lp: PreparedG4RelationTree, rp: PreparedG4RelationTree, op: tuple[int, int],
    ) -> bool:
        """Conservative exact certificate that a graft's new bridge cannot be mistaken for an old edge.

        Equal-size inputs deliberately fall back because an isomorphism may exchange the
        two complete input components.  For unequal sizes, only an internal edge of the
        larger input can have the same cut-side sizes as the new bridge.  Such an edge is
        potentially ambiguous only if, when oriented from the large residual side into
        the detached small side, (i) its endpoint label equals the new bridge label in
        that orientation and (ii) the exact rooted canon of the detached side equals an
        actually legal rooted-owner canon of the smaller input.  If no such edge exists,
        every output isomorphism must map the new bridge to the new bridge.

        The predicate is sufficient only: False always means use the retained exact
        canonical graft path.  Structural tuple equality, never a digest, is used here.
        """
        ln, rn = int(lp.tree.n), int(rp.tree.n)
        if ln == rn:
            return False
        if ln > rn:
            large, small = lp, rp
            small_n = rn
            small_endpoint_type = int(op[1])
            target_label = self._endpoint_labels[op]
        else:
            large, small = rp, lp
            small_n = ln
            small_endpoint_type = int(op[0])
            target_label = freeze_label((int(op[1]), int(op[0])))

        small_rooted_canons = {
            small.rooted_canons[v]
            for v in small.rooted_owner_representatives(small_endpoint_type)
        }
        if not small_rooted_canons:
            # relation_profile handles zero legal pairs before calling this predicate.
            return False

        for (u, v), (side_u, side_v) in zip(large.tree.edges, large.edge_side_sizes):
            if int(side_u) == small_n:
                label, cavity = self._directed_branch(large, int(v), int(u))
                if label == target_label and cavity in small_rooted_canons:
                    return False
            if int(side_v) == small_n:
                label, cavity = self._directed_branch(large, int(u), int(v))
                if label == target_label and cavity in small_rooted_canons:
                    return False
        return True

    def relation_profile_family(
        self,
        left: DecoratedG4Tree | PreparedG4RelationTree,
        rights,
        operators,
    ) -> tuple[ExactTreeRelationProfile, ...]:
        """Exact right-major profile family with compact structural transforms.

        The optimized branch is used when the common left carrier is strictly
        larger than every right carrier, which is the registered S7D2 P124
        shape.  Every directed rooted/cavity message is interned from its full
        immutable structural tuple.  A candidate graft is counted directly only
        when exact cut, label, and rooted-component comparisons prove that its
        new bridge cannot be exchanged with an old left edge.

        For the remaining candidate pairs, one path-local exact message update
        decides whether such an exchange is possible.  Only those genuinely
        ambiguous pairs construct a complete center-rooted graft signature.
        Center-rooted signatures are complete invariants for finite decorated
        trees; they are compared only inside the shared exact-interner scope.

        If the size precondition is absent, the already accepted exact batch
        profile path is used.  Thus this method changes execution only and
        preserves relation SET cardinality for every returned coordinate.
        """
        ops = tuple(_operator(op) for op in operators)
        if not ops:
            return tuple()
        for op in ops:
            if op not in self._ops:
                raise ExactTreeRelationKernelError(f"operator {op} outside frozen G4 basis")
        raw_rights = tuple(rights)
        if not raw_rights:
            return tuple()

        def raw_tree(value: DecoratedG4Tree | PreparedG4RelationTree) -> DecoratedG4Tree:
            if isinstance(value, PreparedG4RelationTree):
                if value.authority != self.authority:
                    raise ExactTreeRelationKernelError("prepared carrier authority mismatch")
                return value.tree
            _check_immutable_carrier(value)
            return value

        left_tree = raw_tree(left)
        right_trees = tuple(raw_tree(right) for right in raw_rights)
        self._stats["relation_profile_family_calls"] = int(self._stats.get("relation_profile_family_calls", 0)) + 1
        self._stats["relation_profile_family_right_count"] = int(self._stats.get("relation_profile_family_right_count", 0)) + len(right_trees)
        self._stats["relation_profile_family_operator_count"] = int(self._stats.get("relation_profile_family_operator_count", 0)) + len(ops)
        self._stats["relation_profile_family_profile_count"] = int(self._stats.get("relation_profile_family_profile_count", 0)) + len(right_trees) * len(ops)

        if any(int(left_tree.n) <= int(right_tree.n) for right_tree in right_trees):
            self._stats["relation_profile_family_general_fallbacks"] = int(self._stats.get("relation_profile_family_general_fallbacks", 0)) + 1
            lp = self._coerce(left)
            rps = tuple(self._coerce(right) for right in raw_rights)
            return tuple(
                profile
                for rp in rps
                for profile in self.relation_profile_batch(lp, rp, ops)
            )

        # The optimized common-large-left path intentionally avoids the accepted
        # full PreparedG4RelationTree construction.  That path computes every
        # rooted/cavity canon and cache-accounting traversal before the compact
        # messages are built, duplicating the dominant work once per S7D2 child.
        # Validate the same frozen carrier/authority boundary directly, then
        # derive legal owners from the compact root messages below.
        for tree in (left_tree, *right_trees):
            for klass in tree.H_classes:
                if klass not in self._frozen_keys:
                    raise ExactTreeRelationKernelError(f"unsupported repaired-H class {klass}")
            if any(operator not in self._ops for operator in tree.edge_operators):
                raise ExactTreeRelationKernelError("operator outside frozen G4 basis")

        interner = _ExactStructuralInterner()
        left_compact = _compact_prepare_tree(
            left_tree, interner, self._frozen_keys, self._caps,
        )
        right_compact = tuple(
            _compact_prepare_tree(right_tree, interner, self._frozen_keys, self._caps)
            for right_tree in right_trees
        )
        right_sizes = {int(right_tree.n) for right_tree in right_trees}

        # An old left edge is relevant only when its detached side has exactly a
        # registered right-component size.  The key uses exact directed label
        # and exact rooted detached-component structure (via an interned handle).
        old_edges: dict[tuple[int, tuple[Any, ...], int], list[tuple[int, int]]] = defaultdict(list)
        for u, v in left_tree.edges:
            for residual, detached in ((u, v), (v, u)):
                detached_size = left_compact.side_sizes[(detached, residual)]
                if detached_size in right_sizes:
                    old_edges[(
                        detached_size,
                        left_compact.edge_labels[(residual, detached)],
                        left_compact.message_ids[(detached, residual)],
                    )].append((residual, detached))
        self._stats["relation_profile_family_old_bridge_index_builds"] = int(self._stats.get("relation_profile_family_old_bridge_index_builds", 0)) + 1

        profiles: list[ExactTreeRelationProfile] = []
        certified_unique_total = 0
        ambiguous_total = 0
        exact_signature_total = 0
        path_update_total = 0
        for right_tree, compact_right in zip(right_trees, right_compact):
            for op in ops:
                assert left_compact.owners_by_type is not None
                assert left_compact.representatives_by_type is not None
                assert compact_right.owners_by_type is not None
                assert compact_right.representatives_by_type is not None
                left_owners = left_compact.owners_by_type[op[0]]
                right_owners = compact_right.owners_by_type[op[1]]
                attempted = int(left_tree.n) * int(right_tree.n)
                legal_pairs = len(left_owners) * len(right_owners)
                if not legal_pairs:
                    self._stats["relation_profile_zero_fast_hits"] += 1
                    profiles.append(ExactTreeRelationProfile(0, attempted, 0, 0, 0, 0, 0))
                    continue
                left_reps = left_compact.representatives_by_type[op[0]]
                right_reps = compact_right.representatives_by_type[op[1]]
                candidate_count = len(left_reps) * len(right_reps)
                label = self._endpoint_labels[op]
                possible = any(old_edges.get((
                    int(right_tree.n), label, compact_right.root_ids[right_owner],
                )) for right_owner in right_reps)
                if not possible:
                    certified_unique_total += candidate_count
                    profiles.append(ExactTreeRelationProfile(
                        candidate_count, attempted, legal_pairs, candidate_count, 0, 0, 0,
                    ))
                    continue

                certified_unique = 0
                ambiguous: list[tuple[int, int]] = []
                for left_owner in left_reps:
                    for right_owner in right_reps:
                        swappable = False
                        matches = old_edges.get((
                            int(right_tree.n), label, compact_right.root_ids[right_owner],
                        ), ())
                        for residual, detached in matches:
                            route = _compact_path(left_compact, left_owner, residual)
                            # The new seed must remain on the large side of the old
                            # cut; otherwise that cut has the wrong component sizes.
                            if len(route) > 1 and route[-2] == detached:
                                continue
                            path_update_total += 1
                            modified = _compact_modified_residual_root_id(
                                left_compact, interner, self._frozen_keys,
                                residual_root=residual,
                                detached_neighbor=detached,
                                attachment=left_owner,
                                added_label=label,
                                added_root_id=compact_right.root_ids[right_owner],
                            )
                            if modified == left_compact.root_ids[left_owner]:
                                swappable = True
                                break
                        if swappable:
                            ambiguous.append((left_owner, right_owner))
                        else:
                            certified_unique += 1

                signatures = set()
                for left_owner, right_owner in ambiguous:
                    signatures.add(_compact_unrooted_signature(
                        graft_pair_raw(left_tree, right_tree, left_owner, right_owner, op),
                        interner,
                        self._frozen_keys,
                    ))
                certified_unique_total += certified_unique
                ambiguous_total += len(ambiguous)
                exact_signature_total += len(ambiguous)
                profiles.append(ExactTreeRelationProfile(
                    certified_unique + len(signatures),
                    attempted,
                    legal_pairs,
                    candidate_count,
                    len(ambiguous),
                    0,
                    len(ambiguous) * (int(left_tree.n) + int(right_tree.n)),
                ))

        self._stats["relation_profile_family_certified_unique_candidates"] = int(self._stats.get("relation_profile_family_certified_unique_candidates", 0)) + certified_unique_total
        self._stats["relation_profile_family_ambiguous_candidates"] = int(self._stats.get("relation_profile_family_ambiguous_candidates", 0)) + ambiguous_total
        self._stats["relation_profile_family_exact_graft_signatures"] = int(self._stats.get("relation_profile_family_exact_graft_signatures", 0)) + exact_signature_total
        self._stats["relation_profile_family_path_updates"] = int(self._stats.get("relation_profile_family_path_updates", 0)) + path_update_total
        self._stats["relation_profile_family_full_canon_avoided"] = int(self._stats.get("relation_profile_family_full_canon_avoided", 0)) + certified_unique_total
        return tuple(profiles)

    def relation_profile_batch(
        self, left: DecoratedG4Tree | PreparedG4RelationTree,
        right: DecoratedG4Tree | PreparedG4RelationTree, operators,
    ) -> tuple[ExactTreeRelationProfile, ...]:
        """Exact profiles for many operators with one prepared left/right pair.

        This is an execution-only vectorization of :meth:`relation_profile`.  It
        preserves operator order and returns exactly the same profile record for
        every operator.  For unequal-size inputs, the conservative A24 bridge
        certificate is indexed once from exact edge/cavity tuples and then reused
        across the operator vector.  Any ambiguous entry still takes the retained
        exact canonical-graft fallback.
        """
        ops = tuple(_operator(op) for op in operators)
        if not ops:
            return tuple()
        for op in ops:
            if op not in self._ops:
                raise ExactTreeRelationKernelError(f"operator {op} outside frozen G4 basis")
        lp, rp = self._coerce(left), self._coerce(right)
        self._stats["relation_profile_batch_calls"] = int(self._stats.get("relation_profile_batch_calls", 0)) + 1
        self._stats["relation_profile_batch_operator_count"] = int(self._stats.get("relation_profile_batch_operator_count", 0)) + len(ops)

        # Equal-size inputs remain deliberately conservative: component exchange
        # may identify the new bridge with an old edge, so use the ordinary exact
        # profile path for every operator.
        if int(lp.tree.n) == int(rp.tree.n):
            return tuple(self.relation_profile(lp, rp, op) for op in ops)

        if int(lp.tree.n) > int(rp.tree.n):
            large, small = lp, rp
            small_n = int(rp.tree.n)
            small_is_right = True
        else:
            large, small = rp, lp
            small_n = int(lp.tree.n)
            small_is_right = False

        # Exact directed fingerprints of only those old edges whose detached side
        # has the same size as the smaller graft component.  Tuple equality is the
        # authority; hash tables are lookup acceleration only.
        cavities_by_label: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
        for (u, v), (side_u, side_v) in zip(large.tree.edges, large.edge_side_sizes):
            if int(side_u) == small_n:
                label, cavity = self._directed_branch(large, int(v), int(u))
                cavities_by_label.setdefault(label, set()).add(cavity)
            if int(side_v) == small_n:
                label, cavity = self._directed_branch(large, int(u), int(v))
                cavities_by_label.setdefault(label, set()).add(cavity)
        self._stats["relation_profile_batch_bridge_index_builds"] = int(self._stats.get("relation_profile_batch_bridge_index_builds", 0)) + 1
        small_canons_by_type: dict[int, frozenset[tuple[Any, ...]]] = {}

        out: list[ExactTreeRelationProfile] = []
        for op in ops:
            cache_key = (lp.tree, rp.tree, op)
            cache_hit, cached = self._relation_cache.lookup(cache_key)
            if cache_hit:
                out.append(ExactTreeRelationProfile(
                    len(cached.canons), cached.attempted_owner_pairs, cached.legal_owner_pairs,
                    cached.rooted_owner_pair_candidates, cached.child_canon_constructions,
                    cached.reused_cavity_branches, cached.reroot_vertices_visited,
                ))
                continue
            attempted = lp.tree.n * rp.tree.n
            legal_pairs = len(lp.legal_owners(op[0])) * len(rp.legal_owners(op[1]))
            if not legal_pairs:
                self._stats["relation_profile_zero_fast_hits"] += 1
                out.append(ExactTreeRelationProfile(0, attempted, 0, 0, 0, 0, 0))
                continue
            lreps = lp.rooted_owner_representatives(op[0])
            rreps = rp.rooted_owner_representatives(op[1])
            candidates = len(lreps) * len(rreps)
            small_endpoint_type = int(op[1] if small_is_right else op[0])
            small_canons = small_canons_by_type.get(small_endpoint_type)
            if small_canons is None:
                small_canons = frozenset(
                    small.rooted_canons[v]
                    for v in small.rooted_owner_representatives(small_endpoint_type)
                )
                small_canons_by_type[small_endpoint_type] = small_canons
            target_label = self._endpoint_labels[op] if small_is_right else freeze_label((int(op[1]), int(op[0])))
            old_cavities = cavities_by_label.get(target_label)
            unique = bool(small_canons) and (not old_cavities or small_canons.isdisjoint(old_cavities))
            if unique:
                self._stats["relation_profile_bridge_fingerprint_fast_hits"] += 1
                self._stats["relation_profile_bridge_fingerprint_canon_avoided"] += candidates
                out.append(ExactTreeRelationProfile(candidates, attempted, legal_pairs, candidates, 0, 0, 0))
                continue
            self._stats["relation_profile_bridge_fingerprint_fallbacks"] += 1
            canons, _by_canon, attempted2, legal_pairs2, candidates2, reused, rerooted = self._relation_canon_core(lp, rp, op)
            out.append(ExactTreeRelationProfile(
                len(canons), attempted2, legal_pairs2, candidates2, candidates2, reused, rerooted,
            ))
        return tuple(out)

    def relation_profile(
        self, left: DecoratedG4Tree | PreparedG4RelationTree,
        right: DecoratedG4Tree | PreparedG4RelationTree, operator: tuple[int, int],
    ) -> ExactTreeRelationProfile:
        """Exact relation cardinality/operational profile without materializing child carriers."""
        op = _operator(operator)
        if op not in self._ops:
            raise ExactTreeRelationKernelError(f"operator {op} outside frozen G4 basis")
        lp, rp = self._coerce(left), self._coerce(right)
        cache_key=(lp.tree,rp.tree,op)
        cache_hit,cached=self._relation_cache.lookup(cache_key)
        if cache_hit:
            return ExactTreeRelationProfile(
                len(cached.canons), cached.attempted_owner_pairs, cached.legal_owner_pairs,
                cached.rooted_owner_pair_candidates, cached.child_canon_constructions,
                cached.reused_cavity_branches, cached.reroot_vertices_visited,
            )
        attempted = lp.tree.n * rp.tree.n
        legal_pairs = len(lp.legal_owners(op[0])) * len(rp.legal_owners(op[1]))
        if not legal_pairs:
            self._stats["relation_profile_zero_fast_hits"] += 1
            return ExactTreeRelationProfile(0, attempted, 0, 0, 0, 0, 0)
        lreps = lp.rooted_owner_representatives(op[0])
        rreps = rp.rooted_owner_representatives(op[1])
        candidates = len(lreps) * len(rreps)
        if self._new_bridge_fingerprint_is_unique(lp, rp, op):
            self._stats["relation_profile_bridge_fingerprint_fast_hits"] += 1
            self._stats["relation_profile_bridge_fingerprint_canon_avoided"] += candidates
            return ExactTreeRelationProfile(
                candidates, attempted, legal_pairs, candidates, 0, 0, 0,
            )
        self._stats["relation_profile_bridge_fingerprint_fallbacks"] += 1
        canons, _by_canon, attempted, legal_pairs, candidates, reused, rerooted = self._relation_canon_core(lp, rp, op)
        return ExactTreeRelationProfile(
            len(canons), attempted, legal_pairs, candidates, candidates, reused, rerooted,
        )

    def relation(
        self, left: DecoratedG4Tree | PreparedG4RelationTree,
        right: DecoratedG4Tree | PreparedG4RelationTree, operator: tuple[int, int],
    ) -> ExactTreeRelationResult:
        op = _operator(operator)
        if op not in self._ops:
            raise ExactTreeRelationKernelError(f"operator {op} outside frozen G4 basis")
        lp, rp = self._coerce(left), self._coerce(right)
        cache_key=(lp.tree,rp.tree,op)
        cache_hit,cached=self._relation_cache.lookup(cache_key)
        if cache_hit:
            return cached
        canons, by_canon, attempted, legal_pairs, candidates, reused, rerooted = self._relation_canon_core(lp, rp, op)
        if not legal_pairs:
            result=ExactTreeRelationResult((), (), attempted, 0, 0, 0)
            self._relation_cache.put(cache_key,result)
            return result
        # Construct only retained child carriers. The complete canon was already
        # computed once per representative graft, and is not reconstructed here.
        children = tuple(graft_pair_raw(lp.tree, rp.tree, *by_canon[c], op) for c in canons)
        result=ExactTreeRelationResult(
            children, canons, attempted, legal_pairs, candidates, candidates, reused, rerooted,
        )
        self._relation_cache.put(cache_key,result)
        return result


def _component_grafted_min(
    prepared: PreparedG4RelationTree, anchor: int, incoming: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Reroot the graft along one original component, reusing untouched cavities.

    At v, every neighbor branch away from anchor is unchanged. Only the incoming
    branch contains the new cross edge. Removing the target neighbor's occurrence
    from the sorted child tuple yields the exact v->neighbor cavity. Occurrence
    removal (not set subtraction) preserves equal repeated branches.
    """
    stack = [(anchor, -1, incoming)]
    least = None
    while stack:
        v, parent, inc = stack.pop()
        entries = [(u, reverse, (label, cavity)) for u, label, reverse, cavity in prepared.branches[v] if u != parent]
        entries.append((-1, None, inc))
        entries.sort(key=lambda x: x[2])
        children = tuple(x[2] for x in entries)
        full = ("V", prepared.vertex_labels[v], children)
        if least is None or full < least:
            least = full
        for i, (u, reverse, _branch) in enumerate(entries):
            if u >= 0:
                cavity = ("V", prepared.vertex_labels[v], children[:i] + children[i+1:])
                stack.append((u, v, (reverse, cavity)))
    assert least is not None
    return least


def _grafted_canon(lp, rp, lu, rv, forward, reverse) -> tuple[Any, ...]:
    return min(
        _component_grafted_min(lp, lu, (forward, rp.rooted_canons[rv])),
        _component_grafted_min(rp, rv, (reverse, lp.rooted_canons[lu])),
    )


def graft_pair_raw(left, right, left_owner, right_owner, operator) -> DecoratedG4Tree:
    """Internal constructor; caller must establish valid disjoint parents/owners."""
    off = left.n
    return DecoratedG4Tree(
        left.n + right.n,
        left.edges + tuple((a + off, b + off) for a, b in right.edges) + ((left_owner, right_owner + off),),
        left.H_classes + right.H_classes,
        left.edge_operators + right.edge_operators + (tuple(operator),),
    )


_DEFAULT_KERNEL: ExactTreeRelationKernel | None = None
_DEFAULT_CONFIGURATION: dict[str, Any] = {"scope_identity": "STANDALONE_ENGINEERING"}


def configure_relation_kernel(*, scope_identity: str, max_cache_entries: int = 128,
                              max_cache_bytes: int = 32 * 1024 * 1024,
                              max_relation_cache_entries: int = 128, max_relation_cache_bytes: int = 32 * 1024 * 1024,
                              max_observer_q_cache_entries: int = 256, max_observer_q_cache_bytes: int = 32 * 1024 * 1024,
                              max_observer_decode_cache_entries: int = 256, max_observer_decode_cache_bytes: int = 32 * 1024 * 1024) -> None:
    """Decoder worker scope reset. Authority is loaded lazily for tree tasks only."""
    global _DEFAULT_KERNEL, _DEFAULT_CONFIGURATION
    if not isinstance(scope_identity, str) or not scope_identity:
        raise ExactTreeRelationKernelError("scope_identity must be a nonempty string")
    for name, value in (
        ("max_cache_entries", max_cache_entries), ("max_cache_bytes", max_cache_bytes),
        ("max_relation_cache_entries", max_relation_cache_entries), ("max_relation_cache_bytes", max_relation_cache_bytes),
        ("max_observer_q_cache_entries", max_observer_q_cache_entries), ("max_observer_q_cache_bytes", max_observer_q_cache_bytes),
        ("max_observer_decode_cache_entries", max_observer_decode_cache_entries), ("max_observer_decode_cache_bytes", max_observer_decode_cache_bytes),
    ):
        if _integer(value, name) < 0:
            raise ExactTreeRelationKernelError(f"{name} must be nonnegative")
    _DEFAULT_KERNEL = None
    _DEFAULT_CONFIGURATION = dict(scope_identity=scope_identity,
        max_cache_entries=max_cache_entries, max_cache_bytes=max_cache_bytes,
        max_relation_cache_entries=max_relation_cache_entries,max_relation_cache_bytes=max_relation_cache_bytes,
        max_observer_q_cache_entries=max_observer_q_cache_entries,max_observer_q_cache_bytes=max_observer_q_cache_bytes,
        max_observer_decode_cache_entries=max_observer_decode_cache_entries,max_observer_decode_cache_bytes=max_observer_decode_cache_bytes)


def get_relation_kernel() -> ExactTreeRelationKernel:
    global _DEFAULT_KERNEL
    if _DEFAULT_KERNEL is None:
        _DEFAULT_KERNEL = ExactTreeRelationKernel(**_DEFAULT_CONFIGURATION)
    return _DEFAULT_KERNEL


def prepare_g4_relation_tree(tree: DecoratedG4Tree, *, adapter=None) -> PreparedG4RelationTree:
    kernel = ExactTreeRelationKernel(adapter=adapter, max_cache_entries=0) if adapter is not None else get_relation_kernel()
    return kernel.prepare(tree, cache=False)


def prepare_g4_relation_tree_cached(tree: DecoratedG4Tree) -> PreparedG4RelationTree:
    return get_relation_kernel().prepare(tree)


def exact_graft_relation(left, right, operator, *, adapter=None) -> ExactTreeRelationResult:
    kernel = ExactTreeRelationKernel(adapter=adapter, max_cache_entries=0) if adapter is not None else get_relation_kernel()
    return kernel.relation(left, right, operator)
