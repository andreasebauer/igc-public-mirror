"""Plain source-coordinate and resource reference; no Decoder imports."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile


def container_hash(tag, children):
    return hashlib.sha256((tag + '|' + '|'.join(sorted(children))).encode()).hexdigest()


def resource_hash(value, depth=0):
    if depth == 5:
        assert len(value) == 2
        p, f = value
        assert len(p) == len(f) == 7
        assert all(type(x) is int and x >= 0 for x in p + f)
        assert all(b <= a for a, b in zip(p, f))
        return hashlib.sha256(b'SITE|' + json.dumps([p, f], separators=(',', ':')).encode()).hexdigest()
    return container_hash(('R6', 'O5', 'O4', 'O3', 'O2')[depth],
                          [resource_hash(x, depth + 1) for x in value])


def sites(value, path=()):
    if len(path) == 5:
        assert len(value) == 2 and len(value[0]) == len(value[1]) == 7
        return {path: value}
    out = {}
    for i, child in enumerate(value):
        out.update(sites(child, path + (i,)))
    return out


def lower_sources(seed_bytes, selection):
    with zipfile.ZipFile(io.BytesIO(seed_bytes)) as outer:
        name = next(n for n in outer.namelist() if n.endswith('graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'))
        raw = outer.read(name)
        assert hashlib.sha256(raw).hexdigest() == selection['authority_member_sha256']
        authority = json.loads(raw)['records']
        selected = {}
        for row in selection['records']:
            matches = [x for x in authority if x['state_digest'] == row['state_digest']]
            assert matches == [row]
            selected[row['state_digest']] = row
        nested = outer.read(next(n for n in outer.namelist() if '/replay/' in n and n.endswith('.zip')))
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        names = inner.namelist()
        prototypes = json.loads(inner.read(next(n for n in names if n.endswith('/OSCOUT_O7_SELECTED_O6_PROTOTYPES.json'))))['prototypes']
        twins = json.loads(inner.read(next(n for n in names if n.endswith('/OSCOUT_O7_O6_TOPOLOGY_TWINS.json'))))['twins']
        parents = {r['prototype_id']: r for r in prototypes}
        parents.update({r['twin_id']: r for r in twins})
    return selected, parents


def reference_for_dag(dag, selected, parents):
    nodes = dag['nodes']; memo = {}
    def build(key):
        if key in memo:
            return memo[key]
        row = nodes[key]
        if row['kind'] == 'O7':
            source = selected[row['source_id']]
            originals = [parents[pid]['ordered_resource_O5_carriers'] for pid in source['parent_ids']]
            colors = [parents[pid]['exact_representative_sha256'] for pid in source['parent_ids']]
            assert colors == source['parent_exact_colors']
            materialized = deepcopy(originals)
            maps = [sites(x) for x in materialized]
            # Consume the actual original fourteen-coordinate coupled E7 edges.
            for edge in source['edges']:
                assert len(edge) == 14 and all(type(x) is int and x >= 0 for x in edge)
                for owner, path, t in [(edge[0], tuple(edge[1:6]), edge[6]),
                                       (edge[7], tuple(edge[8:13]), edge[13])]:
                    assert 0 <= owner < len(maps) and 0 <= t < 7
                    site = maps[owner][path]
                    assert site[1][t] > 0
                    site[1][t] -= 1
            base_skin = container_hash('R7', [resource_hash(x) for x in materialized])
            assert base_skin == source['R7_skin_sha256']
            reservation_sites = [[] for _ in range(7)]
            for t, count in enumerate(row['reserve_counts']):
                assert type(count) is int and count >= 0
                rem = count
                for owner, mapping in enumerate(maps):
                    for path, site in sorted(mapping.items()):
                        take = min(rem, site[1][t])
                        for _ in range(take):
                            reservation_sites[t].append({'owner': owner, 'site_path': list(path)})
                        site[1][t] -= take; rem -= take
                assert rem == 0
            owner_caps = [[sum(site[1][t] for site in mapping.values()) for t in range(7)] for mapping in maps]
            total_caps = [sum(c[t] for c in owner_caps) for t in range(7)]
            record = {'kind': 'O7', 'level': 7, 'source_id': row['source_id'],
                'parent_ids': source['parent_ids'], 'parent_colors': colors, 'original_parent_resources': originals,
                'original_coupled_edges': source['edges'], 'materialized_parent_resources': materialized,
                'reserve_counts': row['reserve_counts'], 'reservation_sites': reservation_sites,
                'owner_caps': owner_caps, 'total_caps': total_caps,
                'skin': container_hash('R7', [resource_hash(x) for x in materialized])}
        else:
            children = [build(x) for x in row['children']]
            record = {'kind': 'LIFT', 'level': row['level'], 'ordered_children': row['children'],
                'top_edges_full': row['top_edges_full'], 'owner_caps': [x['total_caps'] for x in children],
                'total_caps': [sum(x['total_caps'][t] for x in children) for t in range(7)],
                'skin': container_hash('R' + str(row['level']), [x['skin'] for x in children])}
        memo[key] = record
        return record
    for root in dag['roots']:
        build(root)

    # Occurrence paths retain every root and child slot despite compressed node storage.
    occurrences = []; claims = []; expected = set()
    def endpoint(key, path, witness, endpoint_type):
        row = nodes[key]
        if row['kind'] == 'O7':
            t, ordinal = witness
            assert t == endpoint_type and type(ordinal) is int
            location = memo[key]['reservation_sites'][t][ordinal]
            claim = (tuple(path), t, ordinal)
            assert claim not in expected_claims
            expected_claims.add(claim)
            claims.append({'occurrence_path': path, 'endpoint_type': t, 'reservation_ordinal': ordinal,
                           'source_id': row['source_id'], 'original_resource_location': location})
        else:
            slot, local, t = witness
            assert t == endpoint_type
            endpoint(row['children'][slot], path + [slot], local, t)
    expected_claims = set()
    def visit(key, path):
        row = nodes[key]; occurrences.append({'occurrence_path': path, 'node_alias': key})
        if row['kind'] == 'O7':
            for t, count in enumerate(row['reserve_counts']):
                for ordinal in range(count):expected.add((tuple(path), t, ordinal))
            return
        for edge in row['top_edges_full']:
            u, v, a, b, wu, wv = edge
            endpoint(row['children'][u], path + [u], wu, a)
            endpoint(row['children'][v], path + [v], wv, b)
        for slot, child in enumerate(row['children']):visit(child, path + [slot])
    for ordinal, root in enumerate(dag['roots']):visit(root, [ordinal])
    assert expected_claims == expected
    return {'schema_id': 'IG_MASTER_G3_INDEPENDENT_LOWER_CORRESPONDENCE_V1',
            'node_materializations': memo, 'logical_occurrences': occurrences,
            'complete_recursive_endpoint_claims': claims,
            'logical_occurrence_count': len(occurrences), 'endpoint_claim_count': len(claims)}
