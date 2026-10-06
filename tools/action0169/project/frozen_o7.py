import collections
PN=7

def nested_h6(payload):
    return tuple((tuple((tuple((tuple((tuple(((tuple(site[0]), tuple(site[1])) for site in block)) for block in o3)) for o3 in o4)) for o4 in o5)) for o5 in payload))

def iter_sites(h6):
    for a, o5 in enumerate(h6):
        for j, o4 in enumerate(o5):
            for k, o3 in enumerate(o4):
                for v, block in enumerate(o3):
                    for i, (p, f) in enumerate(block):
                        yield (a, j, k, v, i, p, f)

def site_at(h6, path):
    a, j, k, v, i = path
    return h6[a][j][k][v][i]

def materialize_owner(base_h6, edges, owner):
    used = collections.Counter()
    for e in edges:
        c, *rest = e
        p1 = tuple(e[1:6])
        t1 = e[6]
        d = e[7]
        p2 = tuple(e[8:13])
        t2 = e[13]
        if c == owner:
            used[p1, t1] += 1
        if d == owner:
            used[p2, t2] += 1
    out = []
    for a, o5 in enumerate(base_h6):
        oo5 = []
        for j, o4 in enumerate(o5):
            oo4 = []
            for k, o3 in enumerate(o4):
                oo3 = []
                for v, block in enumerate(o3):
                    ob = []
                    for i, (p, f) in enumerate(block):
                        nf = tuple((f[t] - used[(a, j, k, v, i), t] for t in range(PN)))
                        if min(nf) < 0:
                            raise ValueError('E7 oversubscribed')
                        ob.append((p, nf))
                    oo3.append(tuple(ob))
                oo4.append(tuple(oo3))
            oo5.append(tuple(oo4))
        out.append(tuple(oo5))
    return tuple(out)

def owner_free(h6):
    return sum((sum(f) for *_, p, f in iter_sites(h6)))

def eparts(e):
    return (e[0], tuple(e[1:6]), e[6], e[7], tuple(e[8:13]), e[13])

def accounting(ctx, edges):
    K6 = ctx.n
    m7 = len(edges)
    N = sum((int(p.accounting['N']) for p in ctx.parents))
    d = sum((int(p.accounting['d']) for p in ctx.parents))
    r = sum((int(p.accounting['r']) for p in ctx.parents)) + m7
    beta7 = m7 - K6 + 1
    bf = r - N + 1
    P = d + 2 * N
    F = sum((int(p.accounting['F']) for p in ctx.parents)) - 2 * m7
    g = d + r
    deg = [0] * K6
    for e in edges:
        c, _, _, dd, _, _ = eparts(e)
        deg[c] += 1
        deg[dd] += 1
    loads = [2 * int(ctx.parents[i].accounting['r']) + deg[i] for i in range(K6)]
    ok = F == sum((owner_free(materialize_owner(p.h6, edges, i)) for i, p in enumerate(ctx.parents))) and all(((loads[i] - deg[i]) % 2 == 0 for i in range(K6)))
    return {'K6': K6, 'm7': m7, 'N': N, 'd': d, 'r': r, 'beta7': beta7, 'beta_flat': bf, 'P': P, 'F': F, 'g': g, 'degrees': deg, 'visible_loads': loads, 'ok': ok}
