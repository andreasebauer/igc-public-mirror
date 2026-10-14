"""Independent small-tree reference. No Decoder scientific imports."""
def exact_canon(record, class_keys):
    adjacency = [[] for _ in range(record['n'])]
    for (u, v), (a, b) in zip(record['edges'], record['edge_operators']):
        adjacency[u].append((v, (a, b)))
        adjacency[v].append((u, (b, a)))
    def rooted(v, parent):
        return (class_keys[record['H_classes'][v]], tuple(sorted(
            (op, rooted(w, v)) for w, op in adjacency[v] if w != parent)))
    return min(rooted(v, -1) for v in range(record['n']))

def remaining(record, caps):
    rows = [list(caps[k]) for k in record['H_classes']]
    for (u, v), (a, b) in zip(record['edges'], record['edge_operators']):
        rows[u][a] -= 1
        rows[v][b] -= 1
    assert all(type(x) is int and x >= 0 for row in rows for x in row)
    return rows

def formations(left, right, op, caps):
    lr, rr = remaining(left, caps), remaining(right, caps)
    lo = [v for v in range(left['n']) if lr[v][op[0]] > 0]
    ro = [v for v in range(right['n']) if rr[v][op[1]] > 0]
    offset = left['n']; rows = []
    for u in lo:
        for v in ro:
            child = {'n': offset + right['n'],
                     'edges': [list(e) for e in left['edges']] +
                              [[a+offset, b+offset] for a,b in right['edges']] + [[u,v+offset]],
                     'H_classes': left['H_classes'] + right['H_classes'],
                     'edge_operators': left['edge_operators'] + right['edge_operators'] + [list(op)]}
            rows.append({'ordinal': len(rows), 'left_owner': u, 'right_owner': v,
                         'operator': list(op), 'left_owner_remaining': lr[u],
                         'right_owner_remaining': rr[v],
                         'left_component_map': list(range(offset)),
                         'right_component_map': list(range(offset, child['n'])),
                         'full_raw_child': child})
    return lo, ro, rows
