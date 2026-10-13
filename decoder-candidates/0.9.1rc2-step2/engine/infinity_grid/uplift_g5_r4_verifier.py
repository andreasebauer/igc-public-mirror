from __future__ import annotations

"""Independent G5:R4 verification.

Does not import uplift_g5_r4.  It independently checks the unique-marker
reconstruction logic and replays a small structurally separate panel.
"""
from itertools import product
from typing import Any, Mapping

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes


class G5R4VerificationError(RuntimeError):
    pass


def _marker_relation(ad: G4AcceptedAdapter, t: DecoratedG4Tree, op=(0,0)):
    out = set()
    for r in range(t.n):
        for ch in ad.graft_relation(t, r, new_H_class="D", operator=op):
            if sum(x == "D" for x in ch.H_classes) != 1:
                raise G5R4VerificationError("marker not unique")
            # Delete the unique appended marker vertex/edge by construction, then
            # compare exact parent canons. This verifier does not use R4 theorem code.
            rec = DecoratedG4Tree(ch.n-1, tuple(ch.edges[:-1]), tuple(ch.H_classes[:-1]), tuple(ch.edge_operators[:-1]))
            if ad.unrooted_canon(rec) != ad.unrooted_canon(t):
                raise G5R4VerificationError("deletion recovery mismatch")
            out.add(ad.unrooted_canon(ch))
    return tuple(sorted(out))


def verify(primary: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    if primary.get("schema_id") != "IG_G5_R4_MARKED_LEAF_HIDDEN_MINIMALITY_RESULT_V1" or primary.get("status") != "PASS": failures.append("PRIMARY_SCHEMA_STATUS")
    if primary.get("promotion") is not False or primary.get("g5_graduation_preserved") is not True: failures.append("FIREWALL")
    if primary.get("scoped_hidden_minimality_earned") is not True or primary.get("global_hidden_minimality_claim") is not False: failures.append("MINIMALITY_SCOPE")
    th = primary.get("marked_leaf_separation_theorem", {})
    tids = {x.get("theorem_id") for x in th.get("theorems", [])}
    for i in range(1,5):
        if not any(str(t).startswith(f"R4-T{i}-") for t in tids): failures.append(f"MISSING_T{i}")
    hom = primary.get("homogeneous_topology_support", {})
    if hom.get("status") != "PASS" or hom.get("relation_signature_injective") is not True or hom.get("child_sets_pairwise_disjoint") is not True: failures.append("HOMOGENEOUS_SUPPORT")
    dec = primary.get("all_31_existing_operator_small_support", {})
    if dec.get("status") != "PASS" or dec.get("all_31_existing_operators_covered") is not True: failures.append("DECORATED_SUPPORT")
    sent = primary.get("all_31_marker_operator_sentinel", {})
    if sent.get("status") != "PASS" or sent.get("operator_count") != 31: failures.append("MARKER_SENTINEL")

    ad = G4AcceptedAdapter(); shapes = _generate_tree_shapes(6); child_owner = {}; exact_parents = 0; relation_children = 0
    # Separate panel: C-only all-(0,0) unlabeled trees through n=6 plus a small
    # endpoint-typed n=3 sample from three asymmetric/symmetric operators.
    for n in range(1,7):
        for edges in shapes[n].values():
            t = DecoratedG4Tree(n, tuple(edges), tuple("C" for _ in range(n)), tuple((0,0) for _ in range(max(0,n-1))))
            p = ad.unrooted_canon(t); rel = _marker_relation(ad,t)
            exact_parents += 1; relation_children += len(rel)
            if not rel: failures.append(f"EMPTY_N{n}")
            for c in rel:
                old = child_owner.get(c)
                if old is None: child_owner[c] = p
                elif old != p: failures.append(f"OVERLAP_N{n}")
    sample_ops = ((0,1),(1,0),(4,5))
    for edges in _generate_tree_shapes(3)[3].values():
        for ops in product(sample_ops, repeat=2):
            t = DecoratedG4Tree(3,tuple(edges),("C","C","C"),tuple(ops)); p=ad.unrooted_canon(t); rel=_marker_relation(ad,t)
            for c in rel:
                old=child_owner.get(c)
                if old is None: child_owner[c]=p
                elif old!=p: failures.append("SAMPLE_OVERLAP")
    out = {"schema_id":"IG_G5_R4_INDEPENDENT_VERIFICATION_V1","status":"PASS" if not failures else "FAIL","primary_science_sha256":primary.get("science_sha256"),"independent_homogeneous_parent_count":exact_parents,"independent_relation_child_total":relation_children,"independent_child_canon_owner_count":len(child_owner),"independent_marker_deletion_recovery":True,"structural_equality_only":True,"failures":failures[:50]}
    out["verification_sha256"] = canonical_sha256(out)
    if failures: raise G5R4VerificationError("verification failed: "+",".join(failures[:10]))
    return out
