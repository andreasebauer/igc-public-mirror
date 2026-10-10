"""Non-promoting historical public observer; no construction or ancestry fields."""
import hashlib
import json

def public_assembly(st):
    reservations = []
    for t in range(7):
        if st.total_caps[t] <= 0:
            reservations.append({'endpoint_type': t, 'available': False,
                                 'successor_skin': None, 'successor_total_caps': None})
        else:
            nx, _ = st.reserve_external(t)
            reservations.append({'endpoint_type': t, 'available': True,
                                 'successor_skin': str(nx.skin),
                                 'successor_total_caps': [int(x) for x in nx.total_caps]})
    return {'schema_id': 'IG_G2_S1_REALIZED_PUBLIC_ASSEMBLY_V1',
            'assembly_skin': str(st.skin), 'total_caps': [int(x) for x in st.total_caps],
            'typed_top_edge_multiset': sorted([list(map(int, e)) for e in st.typed_edges]),
            'one_endpoint_reservations': reservations}

def observer_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()
