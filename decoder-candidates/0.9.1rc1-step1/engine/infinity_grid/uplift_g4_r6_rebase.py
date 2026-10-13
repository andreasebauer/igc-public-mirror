from __future__ import annotations

"""G4:R6.REBASE: generic recursive decorated-tree message law.

R6 replaces the per-size message-depth ladder by one fixed recursive tree-message
constructor.  The scientific lower-bound theorem is scoped to the registered
abstract decorated-tree challenge grammar; it is deliberately not promoted to a
resource-realizability theorem for the full G4 hidden subcarrier.
"""

from collections import defaultdict
from importlib.resources import files
from itertools import product
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .decorated_tree_messages import (
    eccentricities,
    graft_fixed_leaf,
    rooted_canon_bag,
    rooted_tree_canon,
    rooted_truncated_cavity_message,
)
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g4_r4_rebase import action_signature as legacy_action_signature
from .uplift_g4_r4_rebase import state_read as legacy_state_read


class G4R6RebaseError(RuntimeError):
    pass


_SPEC = "resources/uplift/G4_R6_REBASE_GENERIC_DECORATED_MESSAGE_LAW_SPEC_V1.json"


def spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC).read_text(encoding="utf-8"))
    expected = obj.get("science_sha256")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if obj.get("schema_id") != "IG_G4_R6_REBASE_GENERIC_DECORATED_MESSAGE_LAW_SPEC_V1":
        raise G4R6RebaseError("bad spec schema")
    if canonical_sha256(payload) != expected:
        raise G4R6RebaseError("spec hash mismatch")
    return obj


def verify_authority(
    r5: Mapping[str, Any],
    replay: Mapping[str, Any],
    closeout: Mapping[str, Any],
) -> dict[str, Any]:
    s = spec()
    failures: list[str] = []
    if r5.get("schema_id") != "IG_G4_R5_REBASE_DEPTH_STABILIZATION_RESULT_V1" or r5.get("status") != "PASS":
        failures.append("R5_RESULT")
    if r5.get("science_sha256") != s["authority"]["g4_r5_primary_science_sha256"]:
        failures.append("R5_IDENTITY")
    if r5.get("classification") != s["authority"]["required_r5_classification"]:
        failures.append("R5_CLASSIFICATION")
    if r5.get("depth4_stable_at_n6_on_registered_scope") is not False:
        failures.append("R5_DEPTH4_BOUNDARY")
    if r5.get("depth5_repairs_registered_p6_scope") is not True:
        failures.append("R5_DEPTH5_REPAIR")
    if replay.get("schema_id") != "IG_G4_R5_REBASE_REPLAY_COMPARISON_V1" or replay.get("status") != "PASS":
        failures.append("R5_REPLAY")
    if replay.get("science_sha256") != s["authority"]["g4_r5_replay_science_sha256"]:
        failures.append("R5_REPLAY_IDENTITY")
    if closeout.get("schema_id") != "IG_G4_R5_REBASE_CERTIFIED_CLOSEOUT_V1" or closeout.get("status") != "CERTIFIED_PASS":
        failures.append("R5_CLOSEOUT")
    if closeout.get("science_sha256") != s["authority"]["g4_r5_closeout_science_sha256"]:
        failures.append("R5_CLOSEOUT_IDENTITY")
    if closeout.get("public_descriptor") != s["authority"]["required_public_descriptor"]:
        failures.append("PUBLIC_DESCRIPTOR")
    out = {
        "schema_id": "IG_G4_R6_REBASE_AUTHORITY_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "r5_science_sha256": r5.get("science_sha256"),
        "r5_replay_science_sha256": replay.get("science_sha256"),
        "r5_closeout_science_sha256": closeout.get("science_sha256"),
        "public_descriptor": "CAPS7_PLUS_H_CLASS_BAG",
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G4R6RebaseError("authority failed: " + ",".join(failures))
    return out


def _path(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(int(n) - 1)]


def theorem_certificate() -> dict[str, Any]:
    """Return the preregistered mathematical theorem chain.

    The general statements are mathematical consequences of the recursive
    definitions, not extrapolations from the finite support panel.
    """
    theorems = [
        {
            "theorem_id": "R6-T1-CAVITY-CANON-COMPLETENESS",
            "scope": "ALL_FINITE_VERTEX_AND_EDGE_DECORATED_TREES",
            "statement": (
                "The rooted cavity canon C(T,r)=(H(r), multiset[(g(r,u),C(T_{u|r},u))]) "
                "is a complete invariant of finite rooted decorated-tree isomorphism."
            ),
            "proof": (
                "Structural induction on the number of vertices. The singleton case is immediate. "
                "At a root, deleting the incident edges yields disjoint smaller rooted decorated trees. "
                "Equality of root labels and of the multiset of edge-label/child-canon pairs gives a "
                "bijection of incident branches. By the induction hypothesis each matched child canon "
                "gives a rooted decorated isomorphism; the branch isomorphisms assemble uniquely at the root. "
                "The converse follows directly from isomorphism invariance of the constructor."
            ),
            "status": "PROVED_FROM_DEFINITIONS",
        },
        {
            "theorem_id": "R6-T2-ECCENTRICITY-SUFFICIENCY",
            "scope": "ALL_FINITE_VERTEX_AND_EDGE_DECORATED_TREES",
            "statement": (
                "The non-backtracking truncated cavity message at root r and depth ecc_T(r) equals the full "
                "rooted cavity canon C(T,r)."
            ),
            "proof": (
                "A cavity round crosses at most one new edge away from the root and never backtracks across "
                "the recipient edge. Every vertex and every incident edge of a finite tree lies on a unique "
                "root path of length at most ecc_T(r). Therefore after ecc_T(r) rounds every branch is complete, "
                "and the truncated recursion is identical to the full structural recursion."
            ),
            "status": "PROVED_FROM_DEFINITIONS",
        },
        {
            "theorem_id": "R6-T3-NO-FIXED-FINITE-DEPTH",
            "scope": "REGISTERED_ABSTRACT_BINARY_H_DECORATED_TREE_CHALLENGE_GRAMMAR",
            "statement": (
                "For every fixed finite depth d, there are endpoint-rooted decorated paths P_(d+2) with equal "
                "legacy R2+depth-d action signatures but different complete decorated child action contexts after "
                "the same fixed one-leaf graft. Hence no fixed finite decoration-message depth is globally sufficient "
                "on this abstract grammar."
            ),
            "proof": (
                "Take the same path topology and edge grades in both parents, with the same H class everywhere "
                "except the far endpoint at distance d+1 from the action root. The depth-d local recursion cannot "
                "read that endpoint, and the R2 topology term is identical because topology is unchanged. After the "
                "same leaf graft at the action root, the far endpoint remains part of the child and its local H label "
                "is visible in the complete child action-context bag. Thus the children differ."
            ),
            "status": "PROVED_ON_REGISTERED_ABSTRACT_GRAMMAR",
        },
        {
            "theorem_id": "R6-T4-PATH-SHARPNESS",
            "scope": "ENDPOINT_ROOTED_DECORATED_PATH_FAMILY",
            "statement": (
                "For the witness path P_(d+2), the root eccentricity is d+1: depth d can fail while depth d+1 "
                "is sufficient. The eccentricity upper bound is therefore worst-case sharp on the registered grammar."
            ),
            "proof": "Combine R6-T2 with the explicit R6-T3 path family.",
            "status": "PROVED_ON_REGISTERED_ABSTRACT_GRAMMAR",
        },
        {
            "theorem_id": "R6-T5-FIXED-LEAF-GRAFT-PREDICTIVITY",
            "scope": "ALL_FINITE_DECORATED_TREES_FOR_THE_FIXED_ONE_LEAF_GRAFT_OBSERVER",
            "statement": (
                "The exact rooted cavity canon at the chosen owner is sufficient to determine the isomorphism class "
                "of the parent rooted decorated tree and therefore the isomorphism class, rooted-canon bag, and any "
                "isomorphism-invariant read of the child produced by adjoining a fixed decorated leaf at that owner."
            ),
            "proof": (
                "R6-T1 reconstructs the rooted decorated parent up to rooted isomorphism. The registered graft is a "
                "deterministic rooted operation that appends one known vertex and one known edge. Root-preserving "
                "isomorphisms extend over that new leaf, so the child is determined up to decorated-tree isomorphism."
            ),
            "status": "PROVED_FROM_R6_T1",
        },
    ]
    out = {
        "schema_id": "IG_G4_R6_REBASE_GENERIC_MESSAGE_THEOREM_CERTIFICATE_V1",
        "status": "PASS",
        "operator": {
            "name": "NONBACKTRACKING_DECORATED_CAVITY_MESSAGE_CANON",
            "directed_rule": "Q(u->v)=(H(u), multiset[(grade(u,w),Q(w->u)) for w adjacent u, w!=v])",
            "root_rule": "C(T,r)=(H(r), multiset[(grade(r,u),Q(u->r)) for u adjacent r])",
            "local_constructor_fixed_across_sizes": True,
            "iteration_count_may_grow": True,
            "cryptographic_hash_required_for_exactness": False,
        },
        "theorems": theorems,
        "interpretation": (
            "R4/R5 depth growth is explained by finite propagation of decoration context. A single local recursive "
            "rule suffices on every finite decorated tree, while a size-independent truncation depth does not suffice "
            "on the registered abstract challenge grammar. Eccentricity is a combinatorial propagation bound only."
        ),
        "resource_realizability_lower_bound_claim": False,
        "global_minimality_claim": False,
        "physical_geometry_claim": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _path_lower_bound_support(max_depth: int = 8) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for d in range(int(max_depth) + 1):
        n = d + 2
        edges = _path(n)
        grades = (0,) * (n - 1)
        colors_a = (0,) * n
        colors_b = (0,) * (n - 1) + (1,)
        sig_a = legacy_action_signature(n, edges, colors_a, grades, 0, d)
        sig_b = legacy_action_signature(n, edges, colors_b, grades, 0, d)
        if sig_a != sig_b:
            raise G4R6RebaseError(f"path lower-bound parent signature split at depth {d}")
        child_edges = edges + [(0, n)]
        child_a = legacy_state_read(n + 1, child_edges, colors_a + (0,), grades + (0,), d)
        child_b = legacy_state_read(n + 1, child_edges, colors_b + (0,), grades + (0,), d)
        if child_a == child_b:
            raise G4R6RebaseError(f"path lower-bound child did not split at depth {d}")
        ecc = eccentricities(n, edges)[0]
        if ecc != d + 1:
            raise G4R6RebaseError("unexpected path eccentricity")
        full_a = rooted_tree_canon(n, edges, colors_a, grades, 0)
        trunc_a = rooted_truncated_cavity_message(n, edges, colors_a, grades, 0, ecc)
        if full_a != trunc_a:
            raise G4R6RebaseError("eccentricity cavity bound failed on path witness")
        rows.append(
            {
                "depth_d": d,
                "parent_g3_units": n,
                "root_eccentricity": ecc,
                "same_legacy_depth_d_action_signature": True,
                "distinct_legacy_depth_d_child_state_reads": True,
                "parent_action_signature_sha256": canonical_sha256(sig_a),
                "child_a_read_sha256": canonical_sha256(child_a),
                "child_b_read_sha256": canonical_sha256(child_b),
                "eccentricity_depth_equals_full_cavity_canon": True,
            }
        )
    out = {
        "schema_id": "IG_G4_R6_REBASE_PATH_LOWER_BOUND_SUPPORT_V1",
        "status": "PASS",
        "support_depths": [0, int(max_depth)],
        "rows": rows,
        "theorem_dependence": "SUPPORT_ONLY_GENERAL_RESULT_COMES_FROM_R6_T3_PROOF",
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _class_bridge_for_scope(
    n: int,
    edges_list: Sequence[Sequence[Sequence[int]]],
    *,
    legacy_depth: int,
) -> dict[str, Any]:
    sig_to_canon: dict[Any, set[Any]] = defaultdict(set)
    canon_to_sig: dict[Any, set[Any]] = defaultdict(set)
    total = 0
    for edges in edges_list:
        for colors in product(range(2), repeat=n):
            for grades in product(range(2), repeat=max(0, n - 1)):
                for v in range(n):
                    sig = legacy_action_signature(n, edges, colors, grades, v, legacy_depth)
                    canon = rooted_tree_canon(n, edges, colors, grades, v)
                    sig_to_canon[sig].add(canon)
                    canon_to_sig[canon].add(sig)
                    total += 1
    sig_collision_sizes = [len(x) for x in sig_to_canon.values() if len(x) > 1]
    canon_split_sizes = [len(x) for x in canon_to_sig.values() if len(x) > 1]
    return {
        "g3_unit_count": n,
        "legacy_depth": legacy_depth,
        "attachment_instances": total,
        "legacy_action_class_count": len(sig_to_canon),
        "exact_rooted_canon_class_count": len(canon_to_sig),
        "legacy_classes_merging_multiple_exact_rooted_canons": len(sig_collision_sizes),
        "max_exact_rooted_canons_in_one_legacy_class": max(sig_collision_sizes, default=1),
        "exact_rooted_canons_split_across_multiple_legacy_classes": len(canon_split_sizes),
        "max_legacy_classes_for_one_exact_rooted_canon": max(canon_split_sizes, default=1),
        "equivalence_class_bijection": not sig_collision_sizes and not canon_split_sizes,
    }


def _legacy_bridge_audit(r4_reference: Mapping[str, Any], r5: Mapping[str, Any]) -> dict[str, Any]:
    if r4_reference.get("schema_id") != "IG_G4_R4_REBASE_DECORATION_AWARE_PREDICTIVE_READ_RESULT_V1":
        raise G4R6RebaseError("bad R4 legacy reference schema")
    if r4_reference.get("status") != "PASS":
        raise G4R6RebaseError("bad R4 legacy reference status")
    if r4_reference.get("science_sha256") != r5.get("authority", {}).get("r4_science_sha256"):
        raise G4R6RebaseError("R4 legacy reference does not match R5 authority")

    shapes = _generate_tree_shapes(6)
    r4_rows: list[dict[str, Any]] = []
    certified = {
        int(x["g3_unit_count"]): x
        for x in r4_reference.get("predictive_audit", {}).get("selected_rows", [])
    }
    for n in range(1, 6):
        row = _class_bridge_for_scope(n, list(shapes[n].values()), legacy_depth=4)
        ref = certified.get(n)
        if ref is None:
            raise G4R6RebaseError(f"missing R4 certified row n={n}")
        row["r4_certified_action_class_count"] = int(ref["action_read_class_count"])
        row["r4_certified_attachment_instances"] = int(ref["attachment_instances"])
        row["matches_r4_certified_counts"] = (
            row["legacy_action_class_count"] == int(ref["action_read_class_count"])
            and row["attachment_instances"] == int(ref["attachment_instances"])
        )
        if not row["equivalence_class_bijection"] or not row["matches_r4_certified_counts"]:
            raise G4R6RebaseError(f"R4/rooted-canon bridge failed n={n}")
        r4_rows.append(row)

    p6 = [_path(6)]
    p6_d4 = _class_bridge_for_scope(6, p6, legacy_depth=4)
    p6_d5 = _class_bridge_for_scope(6, p6, legacy_depth=5)
    r5d4 = r5["stabilization_audit"]["depth4"]
    r5d5 = r5["stabilization_audit"]["depth5"]
    p6_d4["matches_r5_certified_counts"] = (
        p6_d4["attachment_instances"] == int(r5d4["attachment_instances"])
        and p6_d4["legacy_action_class_count"] == int(r5d4["action_read_class_count"])
        and p6_d4["legacy_classes_merging_multiple_exact_rooted_canons"]
        == int(r5d4["nonpredictive_action_class_count"])
        and p6_d4["max_exact_rooted_canons_in_one_legacy_class"]
        == int(r5d4["max_distinct_child_reads_in_one_action_class"])
    )
    p6_d5["matches_r5_certified_counts"] = (
        p6_d5["attachment_instances"] == int(r5d5["attachment_instances"])
        and p6_d5["legacy_action_class_count"] == int(r5d5["action_read_class_count"])
        and p6_d5["legacy_classes_merging_multiple_exact_rooted_canons"] == 0
    )
    if p6_d4["equivalence_class_bijection"]:
        raise G4R6RebaseError("R5 depth4 unexpectedly equals exact rooted canon")
    if not p6_d4["matches_r5_certified_counts"]:
        raise G4R6RebaseError("R5 depth4 collision anatomy did not reproduce")
    if not p6_d5["equivalence_class_bijection"] or not p6_d5["matches_r5_certified_counts"]:
        raise G4R6RebaseError("R5 depth5/rooted-canon bridge failed")

    # Supporting, deliberately bounded all-tree n=6 bridge.  The general theorem does
    # not depend on this census.
    n6_all = _class_bridge_for_scope(6, list(shapes[6].values()), legacy_depth=5)
    if not n6_all["equivalence_class_bijection"]:
        raise G4R6RebaseError("supporting all-tree n6 depth5 bridge failed")

    out = {
        "schema_id": "IG_G4_R6_REBASE_LEGACY_TO_GENERIC_MESSAGE_BRIDGE_V1",
        "status": "PASS",
        "r4_reference_science_sha256": r4_reference.get("science_sha256"),
        "r5_reference_science_sha256": r5.get("science_sha256"),
        "r4_depth4_through_n5": r4_rows,
        "r5_p6_depth4": p6_d4,
        "r5_p6_depth5": p6_d5,
        "supporting_all_binary_decorated_trees_n6_depth5": n6_all,
        "interpretation": (
            "On every certified R4 binary decorated scope through n=5, the legacy depth-4 action signature "
            "induces exactly the same equivalence classes as the complete rooted decorated-tree canon. On the "
            "R5 P6 scope, the 512 depth-4 nonpredictive classes are exactly the 512 legacy classes that merge "
            "multiple exact rooted canons; depth 5 restores a 6144-to-6144 bijection. The same depth-5/rooted-canon "
            "bijection also holds on the supporting exhaustive all-tree binary n=6 panel."
        ),
        "all_ranks_evidence_claim": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _generic_graft_sanity() -> dict[str, Any]:
    """Small exact, palette-generic checks of the reusable operator surface."""
    cases = [
        (1, [], ["A"], [], 0, "B", "x"),
        (4, [(0, 1), (1, 2), (1, 3)], ["A", "B", "A", "C"], ["x", "y", "x"], 2, "D", "z"),
        (5, [(0, 1), (1, 2), (2, 3), (2, 4)], [0, 1, 2, 1, 0], [3, 4, 3, 5], 0, 2, 4),
    ]
    rows = []
    for n, edges, colors, grades, root, new_c, new_g in cases:
        canon = rooted_tree_canon(n, edges, colors, grades, root)
        ecc = eccentricities(n, edges)[root]
        trunc = rooted_truncated_cavity_message(n, edges, colors, grades, root, ecc)
        if canon != trunc:
            raise G4R6RebaseError("generic eccentricity sanity failed")
        cn, ce, cc, cg = graft_fixed_leaf(
            n,
            edges,
            colors,
            grades,
            root,
            new_vertex_label=new_c,
            new_edge_label=new_g,
        )
        child_bag = rooted_canon_bag(cn, ce, cc, cg)
        rows.append(
            {
                "parent_root_canon_sha256": canonical_sha256(canon),
                "root_eccentricity": ecc,
                "child_rooted_canon_bag_sha256": canonical_sha256(child_bag),
                "child_g3_units": cn,
            }
        )
    out = {
        "schema_id": "IG_G4_R6_REBASE_GENERIC_OPERATOR_SANITY_V1",
        "status": "PASS",
        "cases": rows,
        "binary_palette_required_by_operator": False,
        "fixed_local_rule": True,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def finalize(
    *,
    r5: Mapping[str, Any],
    r5_replay: Mapping[str, Any],
    r5_closeout: Mapping[str, Any],
    r4_reference: Mapping[str, Any],
) -> dict[str, Any]:
    s = spec()
    auth = verify_authority(r5, r5_replay, r5_closeout)
    theorem = theorem_certificate()
    lower = _path_lower_bound_support(8)
    bridge = _legacy_bridge_audit(r4_reference, r5)
    sanity = _generic_graft_sanity()

    out = {
        "schema_id": "IG_G4_R6_REBASE_GENERIC_DECORATED_MESSAGE_LAW_RESULT_V1",
        "status": "PASS",
        "stage_ref": "G4:R6.REBASE",
        "classification": s["pass_classification"],
        "authority": auth,
        "theorem_scope": s["theorem_scope"],
        "generic_message_theorem_certificate": theorem,
        "path_lower_bound_support": lower,
        "legacy_to_generic_bridge": bridge,
        "generic_operator_sanity": sanity,
        "selected_generic_hidden_action_read": {
            "name": "ROOTED_DECORATED_TREE_CAVITY_CANON",
            "definition": theorem["operator"],
            "legacy_r2_term_needed_after_full_cavity_canon": False,
            "reason": "The complete rooted decorated-tree canon already determines the underlying rooted topology and therefore every R2 topology action signature.",
            "storage_note": "Exactness is structural. Implementations may intern/collision-check canonical records and carry compact IDs instead of materializing every depth-k expansion at every action site.",
        },
        "no_fixed_finite_depth_on_registered_abstract_grammar": True,
        "rooted_eccentricity_universal_sufficiency_bound": True,
        "rooted_eccentricity_worst_case_sharp_on_path_family": True,
        "all_finite_decorated_tree_upper_bound_theorem": True,
        "resource_realizable_g4_unbounded_lower_bound_earned": False,
        "global_minimality_claim": False,
        "promotion": False,
        "topology_promoted": False,
        "public_descriptor_changed": False,
        "raw_h_promoted": False,
        "g4_graduation_preserved": True,
        "g5_started": False,
        "next_authorized_stage": "G4:R7.REBASE_DESIGN",
        "r7_recommended_question": (
            "Bridge the generic exact rooted-message law back to the resource-realizable G4 hidden carrier: test "
            "unbounded witness-family realizability and/or compute the observer-relative quotient of rooted canons "
            "before considering any stronger hidden-state promotion."
        ),
        "cost_control": s["cost_control"],
        "nonclaims": s["nonclaims"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def compare(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "primary_pass": primary.get("status") == "PASS",
        "cold_pass": cold.get("status") == "PASS",
        "classification_equal": primary.get("classification") == cold.get("classification"),
        "science_sha_equal": primary.get("science_sha256") == cold.get("science_sha256"),
        "theorem_equal": primary.get("generic_message_theorem_certificate") == cold.get("generic_message_theorem_certificate"),
        "lower_bound_equal": primary.get("path_lower_bound_support") == cold.get("path_lower_bound_support"),
        "legacy_bridge_equal": primary.get("legacy_to_generic_bridge") == cold.get("legacy_to_generic_bridge"),
    }
    ok = all(checks.values())
    out = {
        "schema_id": "IG_G4_R6_REBASE_REPLAY_COMPARISON_V1",
        "status": "PASS" if ok else "FAIL",
        "certification": "CERTIFIED_PASS" if ok else "NOT_CERTIFIED",
        "checks": checks,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def closeout(
    primary: Mapping[str, Any],
    cold: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    ok = replay.get("status") == "PASS" and replay.get("certification") == "CERTIFIED_PASS"
    out = {
        "schema_id": "IG_G4_R6_REBASE_CERTIFIED_CLOSEOUT_V1",
        "status": "CERTIFIED_PASS" if ok else "CERTIFICATION_FAIL",
        "classification": primary.get("classification") if ok else None,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
        "replay_science_sha256": replay.get("science_sha256"),
        "selected_generic_hidden_action_read": (
            primary.get("selected_generic_hidden_action_read", {}).get("name") if ok else None
        ),
        "no_fixed_finite_depth_on_registered_abstract_grammar": True if ok else None,
        "rooted_eccentricity_universal_sufficiency_bound": True if ok else None,
        "resource_realizable_g4_unbounded_lower_bound_earned": False,
        "g4_graduation_preserved": ok,
        "public_descriptor": "CAPS7_PLUS_H_CLASS_BAG" if ok else None,
        "topology_promoted": False,
        "public_descriptor_changed": False,
        "g5_started": False,
        "next_authorized_stage": "G4:R7.REBASE_DESIGN" if ok else None,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out
