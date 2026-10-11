"""Exact E7 labeling under joint, typed nested-parent automorphisms.

The BLISS graph contains every bag/site and distinct typed endpoint gadgets for
all edge occurrences. A canonical graph ordering is transported back into the
fixed parent slots by whole-tree isomorphisms; individual site orbits are never
permuted independently. V2 intentionally does not preserve legacy edge digests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json

from .canon import canonical_sha256

CANONICALIZATION_VERSION = 'IG_E7_NESTED_BLISS_V2'
IDENTITY_SCHEMA = 'IG_E7_GLOBAL_INCIDENCE_IDENTITY_V3'
IDENTITY_CANONICALIZATION_VERSION = 'IG_E7_COLORED_INCIDENCE_BLISS_V3'
LEGACY_IDENTITY_SCHEMA = 'IG_E7_CONTEXT_IDENTITY_V2'


def _parts(e):
    return e[0], tuple(e[1:6]), e[6], e[7], tuple(e[8:13]), e[13]


def _edge(c, p, t, d, q, u):
    if c > d:
        c, p, t, d, q, u = d, q, u, c, p, t
    return (c, *p, t, d, *q, u)


def _tree(value, depth=0):
    if depth == 5:
        port, free = value
        port, free = tuple(port), tuple(free)
        if (len(port) != 7 or len(free) != 7 or
                any(type(x) is not int or x < 0 for x in port + free) or
                any(f > p for p, f in zip(port, free))):
            raise ValueError('invalid typed parent resource site')
        return ('SITE', port, free)
    return (('R6', 'O5', 'O4', 'O3', 'O2')[depth],
            tuple(sorted(_tree(x, depth + 1) for x in value)))


def _build(ctx, edges):
    if type(ctx.n) is not int or ctx.n < 0 or ctx.n != len(ctx.parents):
        raise ValueError('owner count mismatch')
    if any(not isinstance(p.exact_color, str) or not p.exact_color for p in ctx.parents):
        raise ValueError('parent exact colors must be nonempty strings')
    nodes, links, colors, children, sites, roots = [], [], [], {}, {}, []
    bases = [_tree(p.h6) for p in ctx.parents]

    def add(owner, path, value, depth):
        v = len(nodes)
        base = _tree(value, depth)
        nodes.append((owner, path, base))
        color = ('OWNER', str(ctx.parents[owner].exact_color), base) if depth == 0 else base[:1]
        if depth == 5:
            color = base
        colors.append(json.dumps(color, separators=(',', ':')))
        descendants = []
        if depth == 5:
            sites[(owner, path)] = v
        else:
            for i, item in enumerate(value):
                w = add(owner, path + (i,), item, depth + 1)
                descendants.append(w)
                links.append((v, w))
        children[v] = descendants
        return v

    for o, p in enumerate(ctx.parents):
        roots.append(add(o, (), p.h6, 0))
    used = Counter()
    for e in edges:
        if len(e) != 14 or any(type(x) is not int or x < 0 for x in e):
            raise ValueError('E7 edges require fourteen nonnegative integers')
        c, p, t, d, q, u = _parts(e)
        if c == d or c >= ctx.n or d >= ctx.n or t >= 7 or u >= 7:
            raise ValueError('invalid E7 owner or endpoint type')
        if (c, p) not in sites or (d, q) not in sites:
            raise ValueError('invalid E7 site path')
        # Separate gadgets preserve parallel edges and correlate their two types.
        center = len(colors)
        colors.extend(['["EDGE"]', json.dumps(['END', t]), json.dumps(['END', u])])
        links.extend([(center, center + 1), (center, center + 2),
                      (center + 1, sites[(c, p)]), (center + 2, sites[(d, q)])])
        used[(c, p, t)] += 1
        used[(d, q, u)] += 1
    for (o, p, t), count in used.items():
        if count > nodes[sites[(o, p)]][2][2][t]:
            raise ValueError('E7 oversubscribed')
    return nodes, links, colors, children, sites, roots, bases


def canonicalize_with_witness(ctx, edges):
    """Return a canonical edge multiset and a complete hierarchical bijection."""
    import igraph
    if igraph.__version__ != '1.0.0':
        raise RuntimeError('IG_E7_NESTED_BLISS_V2 requires igraph==1.0.0')
    edges = tuple(tuple(e) for e in edges)
    nodes, links, colors, children, sites, roots, bases = _build(ctx, edges)
    palette = {color: i for i, color in enumerate(sorted(set(colors)))}
    graph = igraph.Graph(n=len(colors), edges=links, directed=False)
    permutation = graph.canonical_permutation(color=[palette[c] for c in colors])
    # In the pinned igraph 1.0.0 Python API this vector indexes original
    # vertices in canonical order (as consumed by permute_vertices). Invert
    # it to label each original vertex. Metamorphic tests pin this convention.
    labels = [0] * len(permutation)
    for canonical, original in enumerate(permutation):
        labels[original] = canonical
    classes = defaultdict(list)
    for o, p in enumerate(ctx.parents):
        classes[(str(p.exact_color), bases[o])].append(o)
    owner_map = list(range(ctx.n))
    for owners in classes.values():
        for old, new in zip(sorted(owners, key=lambda o: labels[roots[o]]), sorted(owners)):
            owner_map[old] = new
    node_map = {}

    def transport(src, dst):
        if nodes[src][2] != nodes[dst][2]:
            raise ValueError('canonical transport changed parent structure')
        node_map[src] = dst
        groups = defaultdict(list)
        targets = defaultdict(list)
        for ch in children[src]:
            groups[nodes[ch][2]].append(ch)
        for ch in children[dst]:
            targets[nodes[ch][2]].append(ch)
        for base, group in groups.items():
            for a, b in zip(sorted(group, key=lambda v: labels[v]),
                            sorted(targets[base], key=lambda v: nodes[v][1])):
                transport(a, b)

    for o, new in enumerate(owner_map):
        transport(roots[o], roots[new])
    site_map = {}
    for key, v in sites.items():
        target = nodes[node_map[v]]
        site_map[key] = (target[0], target[1])
    result = tuple(sorted(_edge(*site_map[(c, p)], t, *site_map[(d, q)], u)
                          for c, p, t, d, q, u in map(_parts, edges)))
    witness = {'schema_id': 'IG_E7_HIERARCHICAL_RELABELING_V2',
               'owner_map': owner_map,
               'node_map': [[o, list(p), nodes[node_map[v]][0], list(nodes[node_map[v]][1])]
                            for v, (o, p, _) in enumerate(nodes)]}
    return {'canonicalization_version': CANONICALIZATION_VERSION,
            'backend': 'igraph-BLISS', 'edges': result, 'witness': witness}


def verify_relabeling_witness(ctx, original, canonical, witness):
    """Verify a supplied whole-parent isomorphism without relying on BLISS."""
    nodes, _, _, children, sites, roots, _ = _build(ctx, tuple(tuple(e) for e in original))
    if witness.get('schema_id') != 'IG_E7_HIERARCHICAL_RELABELING_V2':
        raise ValueError('invalid relabeling witness schema')
    lookup = {(o, p): v for v, (o, p, _) in enumerate(nodes)}
    mapping = {}
    for o, p, d, q in witness['node_map']:
        key, target = (o, tuple(p)), (d, tuple(q))
        if key in mapping or key not in lookup or target not in lookup:
            raise ValueError('invalid witness node')
        mapping[key] = target
    if set(mapping) != set(lookup) or set(mapping.values()) != set(lookup):
        raise ValueError('witness must be a complete bijection')
    owner_map = witness['owner_map']
    if sorted(owner_map) != list(range(ctx.n)):
        raise ValueError('invalid witness owner permutation')
    for o in range(ctx.n):
        if mapping[(o, ())] != (owner_map[o], ()) or ctx.parents[o].exact_color != ctx.parents[owner_map[o]].exact_color:
            raise ValueError('witness changed owner color')
    for (o, p), target in mapping.items():
        v, w = lookup[(o, p)], lookup[target]
        if nodes[v][2] != nodes[w][2]:
            raise ValueError('witness changed parent resources')
        if p and mapping[(o, p[:-1])] != (target[0], target[1][:-1]):
            raise ValueError('witness broke nested ownership')
    transformed = tuple(sorted(_edge(*mapping[(c, p)], t, *mapping[(d, q)], u)
                               for c, p, t, d, q, u in map(_parts, original)))
    if transformed != tuple(tuple(e) for e in canonical):
        raise ValueError('witness does not reproduce canonical edge multiset')
    _build(ctx, transformed)  # independently reject invalid/oversubscribed output
    return True


def canonicalize_edges(ctx, edges):
    return canonicalize_with_witness(ctx, edges)['edges']


def _legacy_identity_payload(ctx, result):
    """Historical V2 alias only: its coordinates are not a global identity."""
    return {'schema_id': LEGACY_IDENTITY_SCHEMA,
               'canonicalization_version': CANONICALIZATION_VERSION,
               'carrier': 'E7_OVER_TYPED_NESTED_R6_BAGS',
               'parent_slots': [{'exact_color': str(p.exact_color), 'base': _tree(p.h6)}
                                for p in ctx.parents],
               'edges': result['edges']}


def state_identity(ctx, edges):
    """V3 global identity with V2 fixed-context edges/witness and historical alias.

    Parent slots retain the existing ordered context contract. Nested children
    are unordered. Graph colors include the actual typed port/free values, not
    just per-call palette numbers. Coordinates and occurrence labels never
    enter the canonical graph serialization.
    """
    import igraph
    edges = tuple(tuple(e) for e in edges)
    result = canonicalize_with_witness(ctx, edges)
    _, links, colors, _, _, _, _ = _build(ctx, edges)
    palette = {color: i for i, color in enumerate(sorted(set(colors)))}
    graph = igraph.Graph(n=len(colors), edges=links, directed=False)
    graph.vs['semantic_color'] = colors
    permutation = graph.canonical_permutation(color=[palette[c] for c in colors])
    canonical = graph.permute_vertices(permutation)
    # Carry vertex attributes through igraph's own permutation API, avoiding
    # an independently assumed permutation-vector direction.
    payload = {
        'schema_id': IDENTITY_SCHEMA,
        'canonicalization_version': IDENTITY_CANONICALIZATION_VERSION,
        'carrier': 'E7_OVER_TYPED_NESTED_R6_BAGS',
        'parent_slots': [{'exact_color': str(p.exact_color), 'base': _tree(p.h6)}
                         for p in ctx.parents],
        'graph': {'directed': False,
                  'vertex_colors': canonical.vs['semantic_color'],
                  'edges': sorted(tuple(sorted(pair)) for pair in canonical.get_edgelist())},
    }
    return {'object_identity': canonical_sha256(payload), 'identity_payload': payload,
            'identity_schema': IDENTITY_SCHEMA,
            'identity_canonicalization_version': IDENTITY_CANONICALIZATION_VERSION,
            'legacy_context_identity': canonical_sha256(_legacy_identity_payload(ctx, result)),
            'legacy_context_identity_schema': LEGACY_IDENTITY_SCHEMA,
            'canonicalization_version': CANONICALIZATION_VERSION,
            'edges': result['edges'], 'witness': result['witness']}


def bind_engine(engine):
    """Bind every internal call of a loaded science engine to V2, including workers."""
    engine.canonicalize_edges = canonicalize_edges
    engine.CANONICALIZATION_VERSION = CANONICALIZATION_VERSION
    engine.IDENTITY_SCHEMA = IDENTITY_SCHEMA
    engine.context_state_identity = state_identity
    return engine
