import collections,functools,itertools,math,hashlib
def sha_repr(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()

@functools.lru_cache(maxsize=None)
def _site_maps_cached(qhash, sites_tuple):
    sites = list(sites_tuple)
    groups = collections.defaultdict(list)
    for i, c in enumerate(sites):
        groups[c].append(i)
    group_opts = []
    for inds in [groups[k] for k in sorted(groups, key=repr)]:
        if len(inds) <= 1:
            group_opts.append([tuple(inds)])
        else:
            group_opts.append(list(itertools.permutations(inds)))
    out = []
    for choice in itertools.product(*group_opts):
        mp = list(range(len(sites)))
        for inds, perm in zip([groups[k] for k in sorted(groups, key=repr)], choice):
            for newpos, oldidx in zip(inds, perm):
                mp[oldidx] = newpos
        out.append(tuple(mp))
    return tuple(out)

def _site_maps(qhash, qmap):
    return _site_maps_cached(qhash, tuple(qmap[qhash]))

def _entity_maps(entities):
    groups = collections.defaultdict(list)
    for i, q in enumerate(entities):
        groups[q].append(i)
    ordered = [groups[k] for k in sorted(groups)]
    opts = [list(itertools.permutations(g)) if len(g) > 1 else [tuple(g)] for g in ordered]
    for choice in itertools.product(*opts):
        mp = list(range(len(entities)))
        for inds, perm in zip(ordered, choice):
            for newpos, oldidx in zip(inds, perm):
                mp[oldidx] = newpos
        yield tuple(mp)

def base_automorphism_search_space(h, qmap):
    ec = collections.Counter(h['entities'])
    space = math.prod((math.factorial(x) for x in ec.values()))
    for q in h['entities']:
        space *= len(_site_maps(q, qmap))
    return space

def exact_canonical_base_key(h, qmap, threshold=8192):
    if base_automorphism_search_space(h, qmap) > threshold:
        return None
    smopts = [_site_maps(q, qmap) for q in h['entities']]
    best = None
    for em in _entity_maps(tuple(h['entities'])):
        for sms in itertools.product(*smopts):
            zz = []
            for v, i, a, w, j, b in h['edges']:
                nv, nw = (em[v], em[w])
                ni = sms[v][i]
                nj = sms[w][j]
                if nv < nw:
                    zz.append((nv, ni, a, nw, nj, b))
                else:
                    zz.append((nw, nj, b, nv, ni, a))
            key = (tuple(h['entities']), tuple(sorted(zz)))
            if best is None or key < best:
                best = key
    return best
