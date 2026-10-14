"""Small bounded reference using plain incidence lists; no Decoder imports."""
from independent_owner_reference import formations, remaining


def freeze(value):
    if type(value) in (list, tuple):
        return tuple(freeze(x) for x in value)
    return value


def canon(record, keys):
    n = record['n']
    assert type(n) is int and n > 0
    assert len(record['H_classes']) == n
    assert len(record['edges']) == len(record['edge_operators']) == n - 1
    adjacency = [[] for _ in range(n)]
    seen = set()
    for (u, v), (a, b) in zip(record['edges'], record['edge_operators']):
        assert all(type(x) is int for x in (u, v, a, b))
        assert 0 <= u < n and 0 <= v < n and u != v
        assert 0 <= a < 7 and 0 <= b < 7
        edge = tuple(sorted((u, v)))
        assert edge not in seen
        seen.add(edge)
        adjacency[u].append((v, (a, b)))
        adjacency[v].append((u, (b, a)))
    reached = {0}; pending = [0]
    for v in pending:
        for w, _ in adjacency[v]:
            if w not in reached:
                reached.add(w); pending.append(w)
    assert len(reached) == n

    def rooted(v, parent):
        branches = [(('seq', (('int', a), ('int', b))), rooted(w, v))
                    for w, (a, b) in adjacency[v] if w != parent]
        return ('V', ('str', keys[record['H_classes'][v]]), tuple(sorted(branches)))
    return min(rooted(v, -1) for v in range(n))


def reconstruct(value, keys):
    """Separate strict recursive parser for the frozen V/str/seq/int grammar."""
    reverse = {v: k for k, v in keys.items()}
    assert len(reverse) == len(keys)
    classes = []; edges = []; ops = []

    def visit(node):
        assert len(node) == 3 and node[0] == 'V'
        label = node[1]
        assert len(label) == 2 and label[0] == 'str' and label[1] in reverse
        here = len(classes); classes.append(reverse[label[1]])
        for decoration, subtree in node[2]:
            assert len(decoration) == 2 and decoration[0] == 'seq'
            pair = decoration[1]; assert len(pair) == 2
            assert all(len(x) == 2 and x[0] == 'int' and type(x[1]) is int and 0 <= x[1] < 7 for x in pair)
            child = visit(subtree)
            edges.append([here, child]); ops.append([pair[0][1], pair[1][1]])
        return here
    visit(value)
    record = {'n': len(classes), 'edges': edges, 'H_classes': classes, 'edge_operators': ops}
    canon(record, keys)
    return record


def delete_marker(record):
    indices = [i for i, k in enumerate(record['H_classes']) if k == 'D']
    assert len(indices) == 1
    d = indices[0]
    incident = [i for i, edge in enumerate(record['edges']) if d in edge]
    assert len(incident) == 1
    keep = [i for i in range(record['n']) if i != d]
    remap = {old: i for i, old in enumerate(keep)}
    edges = []; ops = []
    for i, ((u, v), op) in enumerate(zip(record['edges'], record['edge_operators'])):
        if i != incident[0]:
            edges.append([remap[u], remap[v]]); ops.append(list(op))
    return {'n': len(keep), 'edges': edges,
            'H_classes': [record['H_classes'][i] for i in keep], 'edge_operators': ops}


def marker_reference(record, keys, caps):
    assert 'D' not in record['H_classes']
    marker = {'n': 1, 'edges': [], 'H_classes': ['D'], 'edge_operators': []}
    lo, ro, rows = formations(record, marker, [0, 0], caps)
    assert ro == [0]
    q = tuple(sorted({canon(r['full_raw_child'], keys) for r in rows}, key=repr))
    identity = canon(record, keys)
    recoveries = []
    for c in q:
        child = reconstruct(c, keys)
        remaining(child, caps)
        parent = delete_marker(child)
        assert canon(parent, keys) == identity
        remaining(parent, caps)
        recoveries.append(parent)
    return q, rows, recoveries
