from __future__ import annotations

"""G3:R3 structural-algebra rebase after the delayed R18 shell-profile break.

This module does not change the graduated CAPS7 public G3 quotient.  It replaces the
superseded downstream hidden-tree challenge authority used by G4.  The repaired hidden
state is

    H(T) = SHELL_PROFILE_MULTISET(T)
           + PAIRED_SUPPORT_LOAD_DISTANCE_SIGNATURE_MULTISET(T).

For a binary tree graft W = T *_{{u,v}} U (one new cross edge u--v), the module derives
H(W) from H(T), H(U) and one full rooted incidence read at u and v.  The formula is
independent of construction history and does not inspect the child graph.  Direct graph
recomputation exists only as a verification oracle.
"""

from collections import deque
from importlib.resources import files
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .uplift_g3_r0 import _tree_canon


class G3R3Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G3_R3_STRUCTURAL_ALGEBRA_REBASE_SPEC_V1.json"
_AUTH_RESOURCE = "resources/uplift/G3_R18_R22_REBASE_AUTHORITY_V1.json"


def _load_hashed_resource(name: str, schema: str) -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(name).read_text(encoding="utf-8"))
    if obj.get("schema_id") != schema:
        raise G3R3Error(f"bad resource schema for {name}")
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if observed != expected:
        raise G3R3Error(f"resource hash mismatch for {name}: expected {expected}, observed {observed}")
    return obj


def r3_spec() -> dict[str, Any]:
    return _load_hashed_resource(_SPEC_RESOURCE, "IG_G3_R3_STRUCTURAL_ALGEBRA_REBASE_SPEC_V1")


def rebase_authority() -> dict[str, Any]:
    return _load_hashed_resource(_AUTH_RESOURCE, "IG_G3_R18_R22_REBASE_AUTHORITY_V1")


def verify_rebase_authority() -> dict[str, Any]:
    auth = rebase_authority()
    failures: list[str] = []
    required = {
        "r18_shell_break_science_sha256": "68877db178d919b72bfa40642ce4a35fc2c642573c81bbde3bb8eae071e626b2",
        "r22_repaired_state_science_sha256": "7b152ab5adb701332c6787dfb9392750848c64e599f4f38dc303182d5efd320b",
        "one_leaf_update_law_science_sha256": "f3f38988852c3339ffaa22bfc41efb35b3147cacb0ae3f07ff0b77c8ad447bc4",
        "action_read_v2_combined_science_sha256": "3bbe6486bdf8f92f0126428df93a2293908f0573bac6d01dcac84f8a6431822d",
        "action_read_v2_registry_science_sha256": "dcdd9c56c37986553701098db2292e35c57f618d28549dc01d22a65018ef3db4",
    }
    for key, expected in required.items():
        if str(auth.get(key)) != expected:
            failures.append(key)
    if auth.get("mainline_integration_status") != "DEFERRED_BEFORE_G3_R3":
        failures.append("MAINLINE_STATUS")
    if auth.get("g4_authority_consequence") != "G4_S0_S6_R0_REQUIRE_REBASE_FROM_G3_R3":
        failures.append("G4_REBASE_CONSEQUENCE")
    out = {
        "schema_id": "IG_G3_R3_REBASE_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "authority_science_sha256": auth["science_sha256"],
        "g3_public_caps7_graduation_preserved": True,
        "old_g4_certification_authority_status": "SUPERSEDED_PENDING_REBASE",
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G3R3Error("G3:R3 rebase authority failed: " + ",".join(failures))
    return out


def _adjacency(n: int, edges: Sequence[Sequence[int]]) -> list[list[int]]:
    if n < 1:
        raise G3R3Error("tree must have at least one vertex")
    adj = [[] for _ in range(n)]
    for e in edges:
        if len(e) != 2:
            raise G3R3Error("bad edge arity")
        u, v = map(int, e)
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            raise G3R3Error("bad tree edge")
        adj[u].append(v)
        adj[v].append(u)
    if len(edges) != n - 1:
        raise G3R3Error("expected tree edge count n-1")
    # Connectedness check.
    seen = {0}
    q = deque([0])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                q.append(y)
    if len(seen) != n:
        raise G3R3Error("disconnected tree")
    return adj


def _distance_matrix(n: int, edges: Sequence[Sequence[int]]) -> list[list[int]]:
    adj = _adjacency(n, edges)
    out: list[list[int]] = []
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        if any(d < 0 for d in dist):
            raise G3R3Error("distance matrix received disconnected graph")
        out.append(dist)
    return out


def _profiles_from_distances(dist: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    profiles: list[tuple[int, ...]] = []
    for row in dist:
        counts = [0] * (max(int(d) for d in row) + 1)
        for d in row:
            counts[int(d)] += 1
        profiles.append(tuple(counts))
    return profiles


def hidden_state(n: int, edges: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Direct graph oracle for the repaired R18-R22 hidden state H(T)."""
    adj = _adjacency(n, edges)
    dist = _distance_matrix(n, edges)
    profiles = _profiles_from_distances(dist)
    leaves = {i for i in range(n) if len(adj[i]) == 1} if n > 1 else set()
    loads: dict[int, int] = {}
    for s in range(n):
        load = sum(1 for z in adj[s] if z in leaves)
        if load:
            loads[s] = load
    support_set = set(loads)
    support_rows: list[tuple[int, tuple[int, ...]]] = []
    for s in sorted(support_set):
        signature = tuple(sorted(int(dist[s][t]) for t in support_set if t != s))
        support_rows.append((int(loads[s]), signature))
    state = {
        "shell_profile_multiset": [list(p) for p in sorted(profiles)],
        "paired_support_load_distance_signature_multiset": [
            {"terminal_leaf_load": int(load), "support_distance_signature": list(sig)}
            for load, sig in sorted(support_rows)
        ],
    }
    state["science_sha256"] = canonical_sha256(state)
    return state


def rooted_full_incidence_read(n: int, edges: Sequence[Sequence[int]], root: int) -> dict[str, Any]:
    """Full rooted read used by the first exact binary graft theorem.

    This is intentionally not claimed minimal.  It contains only intrinsic tree data:
    shell-profile class + distance for every vertex, and repaired support row + distance
    for every leaf-supporting vertex.  No construction identity or ancestry is present.
    """
    root = int(root)
    if root < 0 or root >= n:
        raise G3R3Error("root outside tree")
    adj = _adjacency(n, edges)
    dist = _distance_matrix(n, edges)
    profiles = _profiles_from_distances(dist)
    leaves = {i for i in range(n) if len(adj[i]) == 1} if n > 1 else set()
    loads: dict[int, int] = {}
    for s in range(n):
        load = sum(1 for z in adj[s] if z in leaves)
        if load:
            loads[s] = load
    support_set = set(loads)
    shell_inc = sorted((profiles[x], int(dist[root][x])) for x in range(n))
    support_inc = []
    for s in sorted(support_set):
        sig = tuple(sorted(int(dist[s][t]) for t in support_set if t != s))
        support_inc.append((int(loads[s]), sig, int(dist[root][s])))
    read = {
        "vertex_count": int(n),
        "root_degree": int(len(adj[root])),
        "root_shell_profile": list(profiles[root]),
        "shell_distance_incidence": [
            {"shell_profile": list(p), "root_distance": int(d)} for p, d in shell_inc
        ],
        "support_distance_incidence": [
            {
                "terminal_leaf_load": int(load),
                "support_distance_signature": list(sig),
                "root_distance": int(d),
            }
            for load, sig, d in sorted(support_inc)
        ],
        "construction_identity_read": False,
        "ancestry_read": False,
        "topology_canon_read": False,
    }
    read["science_sha256"] = canonical_sha256(read)
    return read


def _extend_profile(profile: Sequence[int], shift: int, counts: Sequence[int]) -> tuple[int, ...]:
    size = max(len(profile), int(shift) + len(counts))
    out = [0] * size
    for i, x in enumerate(profile):
        out[i] += int(x)
    for i, x in enumerate(counts):
        out[int(shift) + i] += int(x)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def _remove_one(values: Sequence[int], value: int) -> tuple[int, ...]:
    buf = [int(x) for x in values]
    try:
        buf.remove(int(value))
    except ValueError as exc:
        raise G3R3Error(f"support signature missing required distance {value}") from exc
    return tuple(sorted(buf))


def _support_records_after_local_root_change(read: Mapping[str, Any], other_vertex_count: int) -> list[list[Any]]:
    n = int(read["vertex_count"])
    root_degree = int(read["root_degree"])
    records = [
        [
            int(r["terminal_leaf_load"]),
            tuple(int(x) for x in r["support_distance_signature"]),
            int(r["root_distance"]),
        ]
        for r in read["support_distance_incidence"]
    ]

    root_was_leaf = n > 1 and root_degree == 1
    if root_was_leaf:
        # The root's unique old neighbour is the unique support at root distance one.
        candidates = [i for i, r in enumerate(records) if int(r[2]) == 1]
        if len(candidates) != 1:
            raise G3R3Error("leaf-root support neighbour is not uniquely recoverable")
        qi = candidates[0]
        records[qi][0] -= 1
        if records[qi][0] < 0:
            raise G3R3Error("negative support load")
        if records[qi][0] == 0:
            # q leaves the support set.  For every other support s, the tree identity
            # d(s,q)=d(root,s)-1 holds, except the n=2 root-support corner where
            # d(root,q)=1 directly.
            updated: list[list[Any]] = []
            for i, r in enumerate(records):
                if i == qi:
                    continue
                load, sig, droot = r
                dq = 1 if int(droot) == 0 else int(droot) - 1
                updated.append([int(load), _remove_one(sig, dq), int(droot)])
            records = updated

    # The opposite endpoint is a leaf after grafting iff its entire parent was a
    # singleton.  In that case this root gains one terminal leaf neighbour and hence
    # either increases its support load or becomes a new support vertex.
    if int(other_vertex_count) == 1:
        root_rows = [i for i, r in enumerate(records) if int(r[2]) == 0]
        if len(root_rows) > 1:
            raise G3R3Error("multiple rooted support rows")
        if root_rows:
            records[root_rows[0]][0] += 1
        else:
            for r in records:
                droot = int(r[2])
                r[1] = tuple(sorted(tuple(r[1]) + (droot,)))
            new_sig = tuple(sorted(int(r[2]) for r in records))
            records.append([1, new_sig, 0])

    return records


def binary_graft_state_from_parent_reads(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Exact formula H(T *_{{u,v}} U) from two rooted parent reads.

    No child adjacency, child topology canon, construction identity, or ancestry is read.
    """
    n_left = int(left["vertex_count"])
    n_right = int(right["vertex_count"])

    # Shell law.  For x in T and y in U, the unique cross-edge path gives
    # d_W(x,y)=d_T(x,u)+1+d_U(v,y), and symmetrically for U.
    right_root_counts = tuple(int(x) for x in right["root_shell_profile"])
    left_root_counts = tuple(int(x) for x in left["root_shell_profile"])
    child_profiles: list[tuple[int, ...]] = []
    for row in left["shell_distance_incidence"]:
        child_profiles.append(
            _extend_profile(row["shell_profile"], int(row["root_distance"]) + 1, right_root_counts)
        )
    for row in right["shell_distance_incidence"]:
        child_profiles.append(
            _extend_profile(row["shell_profile"], int(row["root_distance"]) + 1, left_root_counts)
        )

    # Local support changes are entirely endpoint-local (leaf removal and singleton
    # creation).  Once those are applied, all new cross-support distances again use
    # the unique cross-edge formula.
    left_support = _support_records_after_local_root_change(left, n_right)
    right_support = _support_records_after_local_root_change(right, n_left)
    support_rows: list[tuple[int, tuple[int, ...]]] = []
    for load, sig, droot in left_support:
        cross = tuple(int(droot) + 1 + int(r[2]) for r in right_support)
        support_rows.append((int(load), tuple(sorted(tuple(sig) + cross))))
    for load, sig, droot in right_support:
        cross = tuple(int(droot) + 1 + int(r[2]) for r in left_support)
        support_rows.append((int(load), tuple(sorted(tuple(sig) + cross))))

    state = {
        "shell_profile_multiset": [list(p) for p in sorted(child_profiles)],
        "paired_support_load_distance_signature_multiset": [
            {"terminal_leaf_load": int(load), "support_distance_signature": list(sig)}
            for load, sig in sorted(support_rows)
        ],
    }
    state["science_sha256"] = canonical_sha256(state)
    return state


def graft_graph(
    left_n: int,
    left_edges: Sequence[Sequence[int]],
    left_root: int,
    right_n: int,
    right_edges: Sequence[Sequence[int]],
    right_root: int,
) -> tuple[int, list[tuple[int, int]]]:
    edges = [(int(a), int(b)) for a, b in left_edges]
    edges.extend((int(a) + int(left_n), int(b) + int(left_n)) for a, b in right_edges)
    edges.append((int(left_root), int(left_n) + int(right_root)))
    return int(left_n) + int(right_n), edges


def _generate_tree_shapes(max_n: int) -> dict[int, dict[str, list[tuple[int, int]]]]:
    if max_n < 1:
        raise G3R3Error("max_n must be positive")
    shapes: dict[int, dict[str, list[tuple[int, int]]]] = {1: {_tree_canon(1, []): []}}
    for n in range(1, max_n):
        nxt: dict[str, list[tuple[int, int]]] = {}
        for canon in sorted(shapes[n]):
            edges = shapes[n][canon]
            for v in range(n):
                e2 = list(edges) + [(int(v), int(n))]
                c2 = _tree_canon(n + 1, e2)
                nxt.setdefault(c2, e2)
        shapes[n + 1] = nxt
    return shapes


def binary_graft_exhaustive_total_rank_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Independent deterministic exhaustive worker for one total child rank."""
    total = int(payload["total_vertex_count"])
    spec_hash = str(payload["r3_spec_science_sha256"])
    if spec_hash != str(r3_spec()["science_sha256"]):
        raise G3R3Error("worker/spec binding mismatch")
    if total < 2:
        raise G3R3Error("total rank must be >=2")
    shapes = _generate_tree_shapes(total - 1)
    rooted_cache: dict[tuple[int, str, int], dict[str, Any]] = {}
    actions = 0
    mismatches: list[dict[str, Any]] = []
    component_pair_count = 0
    topology_pair_count = 0
    for nl in range(1, total):
        nr = total - nl
        lefts = shapes[nl]
        rights = shapes[nr]
        component_pair_count += 1
        topology_pair_count += len(lefts) * len(rights)
        for lcanon in sorted(lefts):
            ledges = lefts[lcanon]
            for u in range(nl):
                key = (nl, lcanon, u)
                rooted_cache.setdefault(key, rooted_full_incidence_read(nl, ledges, u))
            for rcanon in sorted(rights):
                redges = rights[rcanon]
                for v in range(nr):
                    key = (nr, rcanon, v)
                    rooted_cache.setdefault(key, rooted_full_incidence_read(nr, redges, v))
                for u in range(nl):
                    lread = rooted_cache[(nl, lcanon, u)]
                    for v in range(nr):
                        rread = rooted_cache[(nr, rcanon, v)]
                        predicted = binary_graft_state_from_parent_reads(lread, rread)
                        child_n, child_edges = graft_graph(nl, ledges, u, nr, redges, v)
                        direct = hidden_state(child_n, child_edges)
                        actions += 1
                        if predicted["science_sha256"] != direct["science_sha256"]:
                            mismatches.append({
                                "left_n": nl,
                                "left_topology_canon": lcanon,
                                "left_root": u,
                                "right_n": nr,
                                "right_topology_canon": rcanon,
                                "right_root": v,
                                "predicted_sha256": predicted["science_sha256"],
                                "direct_sha256": direct["science_sha256"],
                            })
                            # Fail-closed first witness: no reason to continue a rank after a theorem break.
                            break
                    if mismatches:
                        break
                if mismatches:
                    break
            if mismatches:
                break
        if mismatches:
            break
    row = {
        "schema_id": "IG_G3_R3_BINARY_GRAFT_TOTAL_RANK_AUDIT_V1",
        "total_vertex_count": total,
        "ordered_component_size_pair_count": component_pair_count,
        "ordered_topology_pair_count": topology_pair_count,
        "ordered_rooted_binary_graft_action_count": actions,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "status": "PASS" if not mismatches else "FAIL",
    }
    row["science_sha256"] = canonical_sha256(row)
    return row


def structural_theorem_certificate() -> dict[str, Any]:
    theorem = {
        "schema_id": "IG_G3_R3_BINARY_GRAFT_STRUCTURAL_THEOREM_V1",
        "status": "PROVED_FROM_TREE_IDENTITIES",
        "hidden_state": "SHELL_PROFILE_MULTISET_PLUS_PAIRED_SUPPORT_LOAD_DISTANCE_SIGNATURE_MULTISET",
        "rooted_read": "FULL_ROOTED_SHELL_AND_SUPPORT_DISTANCE_INCIDENCE_V1",
        "shell_cross_distance_identity": "d_W(x,y)=d_T(x,u)+1+d_U(v,y)",
        "support_locality_identity": "Only endpoint degree changes can alter leaf status; support-load changes therefore occur only at old endpoint neighbours and at endpoints adjacent to a newly created singleton-side leaf.",
        "support_cross_distance_identity": "For surviving/new supports s in T and r in U, d_W(s,r)=d_T(s,u)+1+d_U(v,r).",
        "leaf_root_neighbour_reconstruction_identity": "If u is an old leaf, its unique neighbour q is the unique support at rooted distance 1; for every non-root support s, d_T(s,q)=d_T(s,u)-1; the n=2 rooted-support corner has d_T(u,q)=1.",
        "consequence": "H(T *_{{u,v}} U) is a deterministic function of the two parent repaired states plus the two full rooted incidence reads; child topology, ancestry and construction identity are not required.",
        "all_finite_tree_scope": True,
        "minimality_claim": False,
        "public_g3_caps7_state_changed": False,
        "g4_rebase_required": True,
    }
    theorem["science_sha256"] = canonical_sha256(theorem)
    return theorem


def finalize_r3_result(*, authority: Mapping[str, Any], theorem: Mapping[str, Any], rank_rows: Sequence[Mapping[str, Any]], execution_science: Mapping[str, Any] | None = None) -> dict[str, Any]:
    max_total = int(r3_spec()["exhaustive_binary_graft_validation"]["max_total_vertex_count"])
    mismatches = sum(int(r.get("mismatch_count", 0)) for r in rank_rows)
    actions = sum(int(r.get("ordered_rooted_binary_graft_action_count", 0)) for r in rank_rows)
    expected_totals = list(range(2, max_total + 1))
    observed_totals = [int(r["total_vertex_count"]) for r in rank_rows]
    pass_gate = (
        authority.get("status") == "PASS"
        and theorem.get("status") == "PROVED_FROM_TREE_IDENTITIES"
        and observed_totals == expected_totals
        and mismatches == 0
        and all(r.get("status") == "PASS" for r in rank_rows)
    )
    classification = (
        "G3_R3_REPAIRED_HIDDEN_TREE_BINARY_GRAFT_ALGEBRA_EARNED_G3_CAPS7_GRADUATION_PRESERVED_G4_REBASE_REQUIRED"
        if pass_gate
        else "G3_R3_STRUCTURAL_ALGEBRA_REBASE_REVIEW_REQUIRED_G4_REMAINS_STALE"
    )
    result = {
        "schema_id": "IG_G3_R3_STRUCTURAL_ALGEBRA_REBASE_RESULT_V1",
        "status": "PASS" if pass_gate else "REVIEW_REQUIRED",
        "classification": classification,
        "authority": dict(authority),
        "structural_theorem": dict(theorem),
        "exhaustive_binary_graft_validation": {
            "max_total_vertex_count": max_total,
            "rank_rows": [dict(r) for r in rank_rows],
            "ordered_rooted_binary_graft_action_count": actions,
            "mismatch_count": mismatches,
        },
        "execution_science": dict(execution_science) if execution_science is not None else None,
        "repaired_hidden_state": "SHELL_PROFILE_MULTISET_PLUS_PAIRED_SUPPORT_LOAD_DISTANCE_SIGNATURE_MULTISET",
        "rooted_read": "FULL_ROOTED_SHELL_AND_SUPPORT_DISTANCE_INCIDENCE_V1",
        "g3_public_descriptor": "CAPS7",
        "g3_public_caps7_graduation_preserved": True,
        "hidden_state_promoted_to_public_g3": False,
        "old_g4_s0_s6_r0_authority_status": "SUPERSEDED_REBASE_REQUIRED",
        "next_authorized_stage": "G4:S0.REBASE" if pass_gate else None,
        "nonclaims": list(r3_spec()["nonclaims"]),
    }
    stable = {k: v for k, v in result.items() if k not in {"execution_science"}}
    result["science_sha256"] = canonical_sha256(stable)
    return result
