from __future__ import annotations
import json,hashlib,math,collections,struct
from typing import Iterable
PN=7
STATE_HASH_PREFIX=b"IG-E6-STATE-v1|"

def shaj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def nested_o5(payload) -> tuple:
    return tuple(
        tuple(
            tuple(
                tuple((tuple(site[0]), tuple(site[1])) for site in block)
                for block in o3
            )
            for o3 in o4
        )
        for o4 in payload
    )

def canon_block(block: Iterable) -> tuple:
    return tuple(sorted((tuple(p), tuple(f)) for p, f in block))

def canon_o3(o3: Iterable) -> tuple:
    return tuple(sorted(canon_block(block) for block in o3))

def canon_o4(o4: Iterable) -> tuple:
    return tuple(sorted(canon_o3(o3) for o3 in o4))

def canon_o5(o5: Iterable) -> tuple:
    return tuple(sorted(canon_o4(o4) for o4 in o5))

def usage(edges):
    result = collections.Counter()
    for c, j, k, v, i, a, d, J, K, V, I, b in edges:
        result[(c, j, k, v, i, a)] += 1
        result[(d, J, K, V, I, b)] += 1
    return result

def base_free(proto_ids, prototypes, c, j, k, v, i, a):
    return prototypes[proto_ids[c]]["o5"][j][k][v][i][1][a]

def edge_bytes(edge):
    return struct.pack(">12I", *edge)

def state_payload(edges):
    return STATE_HASH_PREFIX + b"".join(edge_bytes(edge) for edge in edges)

def state_digest(edges):
    return hashlib.sha256(state_payload(edges)).hexdigest()

def graph_components(owner_count, edges):
    adjacency = [set() for _ in range(owner_count)]
    for edge in edges:
        adjacency[edge[0]].add(edge[6])
        adjacency[edge[6]].add(edge[0])
    seen = set()
    result = []
    for start in range(owner_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        result.append(tuple(sorted(component)))
    return tuple(sorted(result))

def materialize(proto_ids, prototypes, edges):
    used = usage(edges)
    result = []
    for c, pid in enumerate(proto_ids):
        out_o5 = []
        for j, o4 in enumerate(prototypes[pid]["o5"]):
            out_o4 = []
            for k, o3 in enumerate(o4):
                out_o3 = []
                for v, block in enumerate(o3):
                    out_block = []
                    for i, (p, f) in enumerate(block):
                        next_f = tuple(
                            f[a] - used[(c, j, k, v, i, a)] for a in range(PN)
                        )
                        out_block.append((p, next_f))
                    out_o3.append(tuple(out_block))
                out_o4.append(tuple(out_o3))
            out_o5.append(tuple(out_o4))
        result.append(tuple(out_o5))
    return tuple(result)

def component_invariants(proto_ids, prototypes, edges, component):
    members = set(component)
    internal_edges = tuple(
        edge for edge in edges if edge[0] in members and edge[6] in members
    )
    K5 = len(component)
    m6 = len(internal_edges)
    N = sum(prototypes[proto_ids[c]]["N"] for c in component)
    U2 = sum(prototypes[proto_ids[c]]["U2"] for c in component)
    m3 = sum(prototypes[proto_ids[c]]["m3"] for c in component)
    m4 = sum(prototypes[proto_ids[c]]["m4"] for c in component)
    m5 = sum(prototypes[proto_ids[c]]["m5"] for c in component)
    d = sum(prototypes[proto_ids[c]]["d"] for c in component)
    r = U2 + m3 + m4 + m5 + m6
    beta6 = m6 - K5 + 1
    carriers = materialize(proto_ids, prototypes, edges)
    F = sum(
        sum(f)
        for c in component
        for o4 in carriers[c]
        for o3 in o4
        for block in o3
        for _, f in block
    )
    P_total = sum(
        sum(p)
        for c in component
        for o4 in carriers[c]
        for o3 in o4
        for block in o3
        for p, _ in block
    )
    direct_used = sum(
        sum(p[a] - f[a] for a in range(PN))
        for c in component
        for o4 in carriers[c]
        for o3 in o4
        for block in o3
        for p, f in block
    )
    beta_flat = r - N + 1
    expected_beta_flat = (
        sum(prototypes[proto_ids[c]]["beta_flat5"] for c in component) + beta6
    )
    degree = collections.Counter()
    for edge in internal_edges:
        degree[edge[0]] += 1
        degree[edge[6]] += 1
    parity = []
    load_checks = []
    for c in component:
        load = sum(
            sum(p[a] - f[a] for a in range(PN))
            for o4 in carriers[c]
            for o3 in o4
            for block in o3
            for p, f in block
        )
        expected_load = 2 * prototypes[proto_ids[c]]["r"] + degree[c]
        parity.append((c, load % 2, degree[c] % 2))
        load_checks.append((c, load, expected_load))
    grade = d + r
    expected_grade = sum(prototypes[proto_ids[c]]["g"] for c in component) + m6
    ok = (
        direct_used == 2 * r
        and beta6 >= 0
        and beta_flat == expected_beta_flat
        and P_total == d + 2 * N
        and F == P_total - 2 * r
        and F == d + 2 - 2 * beta_flat
        and all(left == right for _, left, right in parity)
        and all(load == expected for _, load, expected in load_checks)
        and grade == expected_grade
    )
    return {
        "K5": K5,
        "m6": m6,
        "N": N,
        "U2": U2,
        "m3": m3,
        "m4": m4,
        "m5": m5,
        "d": d,
        "r": r,
        "beta6": beta6,
        "beta_flat": beta_flat,
        "expected_beta_flat": expected_beta_flat,
        "F": F,
        "P": P_total,
        "direct_used": direct_used,
        "parity": parity,
        "load_checks": load_checks,
        "g": grade,
        "expected_g": expected_grade,
        "ok": ok,
    }

def validate_exact_state(proto_ids, prototypes, edges, compatible_pairs):
    pair_set = set(compatible_pairs)
    used = usage(edges)
    failures = []
    for edge in edges:
        c, j, k, v, i, a, d, J, K, V, I, b = edge
        if c >= d:
            failures.append({"kind": "owner_order", "edge": list(edge)})
        if (a, b) not in pair_set:
            failures.append({"kind": "bridge", "edge": list(edge)})
    for key, amount in used.items():
        c, j, k, v, i, a = key
        capacity = base_free(proto_ids, prototypes, c, j, k, v, i, a)
        if amount > capacity:
            failures.append({"kind": "capacity", "endpoint": list(key), "used": amount, "capacity": capacity})
    carriers = materialize(proto_ids, prototypes, edges)
    for c, o5 in enumerate(carriers):
        for j, o4 in enumerate(o5):
            for k, o3 in enumerate(o4):
                for v, block in enumerate(o3):
                    for i, (p, f) in enumerate(block):
                        if any(x < 0 for x in f) or any(f[a] > p[a] for a in range(PN)):
                            failures.append({"kind": "negative_or_overfree", "path": [c, j, k, v, i]})
    return failures

def normalize_component(proto_ids, edges, component):
    old = list(component)
    mapping = {owner: index for index, owner in enumerate(old)}
    next_ids = tuple(proto_ids[owner] for owner in old)
    next_edges = tuple(
        sorted(
            (
                mapping[edge[0]],
                edge[1], edge[2], edge[3], edge[4], edge[5],
                mapping[edge[6]],
                edge[7], edge[8], edge[9], edge[10], edge[11],
            )
            for edge in edges
            if edge[0] in mapping and edge[6] in mapping
        )
    )
    return next_ids, next_edges

def verify_prototype_selection(pool, selection):
    rows = [(row["prototype_id"], list(row["features"])) for row in pool["prototypes"]]
    width = len(rows[0][1])
    mins = [min(row[1][i] for row in rows) for i in range(width)]
    maxs = [max(row[1][i] for row in rows) for i in range(width)]
    normalized = {
        pid: tuple(
            0.0 if maxs[i] == mins[i] else (features[i] - mins[i]) / (maxs[i] - mins[i])
            for i in range(width)
        )
        for pid, features in rows
    }
    centroid = tuple(sum(normalized[pid][i] for pid, _ in rows) / len(rows) for i in range(width))
    first = min((pid for pid, _ in rows), key=lambda pid: (dist(normalized[pid], centroid), pid))
    selected = [first]
    while len(selected) < 6:
        best = None
        for pid, _ in rows:
            if pid in selected:
                continue
            minimum_distance = min(dist(normalized[pid], normalized[other]) for other in selected)
            if (
                best is None
                or minimum_distance > best[0] + 1e-15
                or (abs(minimum_distance - best[0]) <= 1e-15 and pid < best[1])
            ):
                best = (minimum_distance, pid)
        selected.append(best[1])
    return selected == selection["selected_prototype_ids"], selected
