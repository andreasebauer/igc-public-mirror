from __future__ import annotations

"""G5:R4 marked-leaf exact-hidden separation and scoped minimality audit.

R3 proved that the exact endpoint-typed decorated-tree canon is sufficient for
all finite certified R1 graft futures.  R4 asks a complementary necessity
question on a deliberately narrow but actual G5 subcarrier: if one H class is
reserved as a fresh marker (D), does the relation-valued one-leaf graft observer
already separate non-isomorphic marker-free parents?

The key theorem is structural.  Every marker child contains exactly one D
vertex.  Any child isomorphism must preserve that unique D vertex; deleting it
and its incident edge recovers the parent.  Hence different exact parents have
disjoint exact marker-child sets.  This earns minimality only for the frozen
marked one-step exact-hidden observer, not global hidden-state minimality.
"""

from collections import defaultdict
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes


class G5R4Error(RuntimeError):
    pass


_SPEC = "G5_R4_MARKED_LEAF_HIDDEN_MINIMALITY_SPEC_V1.json"


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def r4_spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_R4_MARKED_LEAF_HIDDEN_MINIMALITY_SPEC_V1":
        raise G5R4Error("bad G5:R4 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G5R4Error("G5:R4 spec hash mismatch")
    return obj


def verify_authority(r3_primary: Mapping[str, Any], r3_verification: Mapping[str, Any], r3_closeout: Mapping[str, Any]) -> dict[str, Any]:
    s = r4_spec(); a = s["authority"]; failures: list[str] = []
    if r3_primary.get("schema_id") != "IG_G5_R3_GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE_RESULT_V1" or r3_primary.get("status") != "PASS":
        failures.append("R3_PRIMARY_SCHEMA_OR_STATUS")
    if str(r3_primary.get("science_sha256")) != str(a["g5_r3_primary_science_sha256"]):
        failures.append("R3_PRIMARY_IDENTITY")
    if r3_primary.get("finite_future_congruence_earned") is not True:
        failures.append("R3_FUTURE_CONGRUENCE")
    if r3_primary.get("g5_graduation_preserved") is not True or r3_primary.get("promotion") is not False:
        failures.append("R3_FIREWALL")
    if r3_verification.get("status") != "PASS" or str(r3_verification.get("verification_sha256")) != str(a["g5_r3_independent_verification_sha256"]):
        failures.append("R3_VERIFICATION")
    if r3_closeout.get("status") != "CERTIFIED_PASS" or str(r3_closeout.get("closeout_sha256")) != str(a["g5_r3_closeout_sha256"]):
        failures.append("R3_CLOSEOUT")
    out = {
        "schema_id": "IG_G5_R4_AUTHORITY_CHECK_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "r3_primary_science_sha256": r3_primary.get("science_sha256"),
        "r3_verification_sha256": r3_verification.get("verification_sha256"),
        "r3_closeout_sha256": r3_closeout.get("closeout_sha256"),
        "public_descriptor": "CAPS7_PLUS_H_CLASS_BAG",
        "g5_graduation_preserved": True,
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R4Error("authority mismatch: " + ",".join(failures))
    return out


def marked_leaf_separation_theorem() -> dict[str, Any]:
    s = r4_spec()
    theorems = [
        {
            "theorem_id": "R4-T1-UNIQUE-MARKER-DELETION-RECOVERY",
            "scope": "FINITE_RESOURCE_LEGAL_ENDPOINT_TYPED_G5_HIDDEN_TREES_WITH_NO_D_VERTEX_AND_ONE_LAWFUL_D_MARKER_GRAFT",
            "statement": "Every exact child produced by adjoining one D marker leaf through a fixed frozen directed operator contains exactly one D vertex, and deleting that vertex and its incident marker edge recovers the exact parent decorated tree.",
            "proof": "The parent contains no D vertices and the graft adds exactly one new D leaf and exactly one incident edge. H-class and endpoint-typed edge decorations are part of exact isomorphism, so the unique D vertex is invariantly identifiable. Removing it and its sole incident edge leaves exactly the original parent with every old vertex label and old endpoint-typed edge decoration unchanged.",
            "status": "PROVED_FROM_CERTIFIED_R1_GRAFT_DEFINITION",
        },
        {
            "theorem_id": "R4-T2-MARKER-CHILD-DISJOINTNESS",
            "scope": "SAME_AS_R4_T1_FOR_ANY_FIXED_OPERATOR_IN_THE_FROZEN_31_OPERATOR_BASIS",
            "statement": "If two marker-free parents are non-isomorphic, their deduplicated exact D-marker child-canon sets are disjoint for the same fixed marker operator.",
            "proof": "Suppose one child canon occurred for both parents. An exact child isomorphism preserves the unique D vertex. Delete that corresponding D leaf in both children; R4-T1 then restricts the child isomorphism to an exact decorated-tree isomorphism of the parents, contradiction.",
            "status": "PROVED_FROM_R4_T1",
        },
        {
            "theorem_id": "R4-T3-MARKED-OBSERVER-INJECTIVITY",
            "scope": "NONEMPTY_RELATION_VALUED_D_MARKER_GRAFT_OBSERVER_ON_MARKER_FREE_RESOURCE_ADMISSIBLE_PARENTS",
            "statement": "For each fixed frozen marker operator, equality of nonempty deduplicated exact D-marker child-canon sets implies equality of exact parent unrooted canons.",
            "proof": "Equal nonempty sets share a child canon. R4-T2 says non-isomorphic parents have disjoint child sets, hence the parents must be isomorphic and therefore have equal exact unrooted canons.",
            "status": "PROVED_FROM_R4_T2",
        },
        {
            "theorem_id": "R4-T4-SCOPED-HIDDEN-MINIMALITY",
            "scope": "FROZEN_MARKED_ONE_STEP_EXACT_HIDDEN_OBSERVER_ONLY",
            "statement": "Any quotient/read that exactly predicts the frozen nonempty D-marker child-canon relation on the marker-free admissible subcarrier must separate every pair of distinct exact parent canons; thus the exact hidden canon is minimal up to isomorphism for this observer/scope.",
            "proof": "If a purported sufficient quotient merged two distinct exact parents, it would assign one prediction to them. R4-T3 shows their exact marker-child relations are different, contradicting exact prediction. This is observer-relative minimality only and says nothing about other observers or representations.",
            "status": "PROVED_FROM_R4_T3",
        },
    ]
    out = {
        "schema_id": "IG_G5_R4_MARKED_LEAF_SEPARATION_THEOREM_V1",
        "status": "PASS",
        "marker_H_class": s["marker_observer"]["marker_H_class"],
        "parent_marker_exclusion": s["marker_observer"]["parent_excludes_H_class"],
        "operator_scope": "ANY_FIXED_OPERATOR_IN_FROZEN_31_OPERATOR_BASIS_SUBJECT_TO_NONEMPTY_LAWFUL_GRAFT_RELATION",
        "theorems": theorems,
        "scoped_minimality_earned": True,
        "global_minimality_claim": False,
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _relation_child_canons(ad: G4AcceptedAdapter, tree: DecoratedG4Tree, *, marker_H: str, op: Sequence[int]) -> tuple[tuple[Any, ...], ...]:
    children: set[tuple[Any, ...]] = set()
    for root in range(tree.n):
        for child in ad.graft_relation(tree, root, new_H_class=marker_H, operator=tuple(map(int, op))):
            children.add(ad.unrooted_canon(child))
    return tuple(sorted(children))


def _check_unique_marker_tree(child: DecoratedG4Tree, marker_H: str) -> bool:
    return sum(1 for x in child.H_classes if x == marker_H) == 1


def homogeneous_topology_support(max_n: int) -> dict[str, Any]:
    s = r4_spec(); ad = G4AcceptedAdapter(); marker = s["marker_observer"]["marker_H_class"]; op = tuple(s["marker_observer"]["primary_marker_operator"])
    shapes = _generate_tree_shapes(int(max_n)); rows = []; failures: list[str] = []
    child_owner: dict[tuple[Any, ...], tuple[Any, ...]] = {}; parent_relation: dict[tuple[Any, ...], tuple[tuple[Any, ...], ...]] = {}
    total_parents = 0; total_children = 0
    for n in range(1, int(max_n) + 1):
        nparents = 0; nchildren = 0; min_branch = None; max_branch = 0
        for edges in shapes[n].values():
            tree = DecoratedG4Tree(n, tuple(edges), tuple("C" for _ in range(n)), tuple((0, 0) for _ in range(max(0, n - 1))))
            q = ad.public_read(tree)
            if not q.get("legal"):
                failures.append(f"HOMOGENEOUS_PARENT_ILLEGAL_N{n}"); continue
            pcan = ad.unrooted_canon(tree); rel = _relation_child_canons(ad, tree, marker_H=marker, op=op)
            if not rel:
                failures.append(f"EMPTY_MARKER_RELATION_N{n}"); continue
            nparents += 1; total_parents += 1; nchildren += len(rel); total_children += len(rel)
            parent_relation[pcan] = rel
            min_branch = len(rel) if min_branch is None else min(min_branch, len(rel)); max_branch = max(max_branch, len(rel))
            # Re-materialize explicit children to check the unique marker invariant.
            for root in range(n):
                for child in ad.graft_relation(tree, root, new_H_class=marker, operator=op):
                    if not _check_unique_marker_tree(child, marker): failures.append(f"MARKER_NOT_UNIQUE_N{n}")
                    ccan = ad.unrooted_canon(child)
                    old = child_owner.get(ccan)
                    if old is None: child_owner[ccan] = pcan
                    elif old != pcan: failures.append(f"CHILD_OVERLAP_N{n}")
        rows.append({
            "hidden_vertex_count": n,
            "unlabelled_parent_count": nparents,
            "distinct_marker_child_count": nchildren,
            "min_relation_branch_count": min_branch,
            "max_relation_branch_count": max_branch,
        })
    if total_parents != sum(len(shapes[n]) for n in range(1, int(max_n) + 1)):
        failures.append("PARENT_COUNT_MISMATCH")
    if len(parent_relation) != total_parents:
        failures.append("PARENT_CANON_DEDUP_MISMATCH")
    if len(set(parent_relation.values())) != len(parent_relation):
        failures.append("RELATION_SIGNATURE_COLLISION")
    out = {
        "schema_id": "IG_G5_R4_HOMOGENEOUS_MARKER_SUPPORT_V1",
        "status": "PASS" if not failures else "FAIL",
        "scope": {"parent_H_class": "C", "existing_operator": [0, 0], "marker_H_class": marker, "marker_operator": list(op), "max_hidden_vertices": int(max_n)},
        "rows": rows,
        "total_exact_parent_canons": total_parents,
        "total_distinct_marker_children_across_parents": len(child_owner),
        "relation_signature_injective": len(set(parent_relation.values())) == len(parent_relation),
        "child_sets_pairwise_disjoint": not any(x.startswith("CHILD_OVERLAP") for x in failures),
        "unique_marker_in_every_materialized_child": not any(x.startswith("MARKER_NOT_UNIQUE") for x in failures),
        "structural_equality_used": True,
        "digest_only_equality_used": False,
        "failures": failures[:50],
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures: raise G5R4Error("homogeneous support failed: " + ",".join(failures[:10]))
    return out


def all_31_operator_small_support(max_n: int = 3) -> dict[str, Any]:
    s = r4_spec(); ad = G4AcceptedAdapter(); marker = s["marker_observer"]["marker_H_class"]; marker_op = tuple(s["marker_observer"]["primary_marker_operator"])
    shapes = _generate_tree_shapes(int(max_n)); ops = tuple(ad.operator_basis()); failures: list[str] = []
    raw_parent_decorations = 0; exact_parents: dict[tuple[Any, ...], DecoratedG4Tree] = {}
    for n in range(1, int(max_n) + 1):
        for edges in shapes[n].values():
            op_iter = [tuple()] if n == 1 else product(ops, repeat=n - 1)
            for edge_ops in op_iter:
                raw_parent_decorations += 1
                tree = DecoratedG4Tree(n, tuple(edges), tuple("C" for _ in range(n)), tuple(edge_ops))
                if not ad.public_read(tree).get("legal"):
                    failures.append(f"UNEXPECTED_ILLEGAL_N{n}"); continue
                exact_parents.setdefault(ad.unrooted_canon(tree), tree)
    child_owner: dict[tuple[Any, ...], tuple[Any, ...]] = {}; nonempty = 0; total_relation_children = 0
    for pcan, tree in exact_parents.items():
        rel = _relation_child_canons(ad, tree, marker_H=marker, op=marker_op)
        if not rel:
            failures.append("EMPTY_MARKER_RELATION"); continue
        nonempty += 1; total_relation_children += len(rel)
        for ccan in rel:
            old = child_owner.get(ccan)
            if old is None: child_owner[ccan] = pcan
            elif old != pcan: failures.append("CHILD_SET_OVERLAP")
    out = {
        "schema_id": "IG_G5_R4_ALL_31_EXISTING_OPERATOR_SMALL_SUPPORT_V1",
        "status": "PASS" if not failures else "FAIL",
        "scope": {"n_min": 1, "n_max": int(max_n), "parent_H_class": "C", "existing_operator_basis_size": len(ops), "marker_H_class": marker, "marker_operator": list(marker_op)},
        "raw_parent_decorations": raw_parent_decorations,
        "exact_parent_canon_count": len(exact_parents),
        "nonempty_marker_relation_parent_count": nonempty,
        "sum_distinct_relation_children": total_relation_children,
        "globally_distinct_marker_child_count": len(child_owner),
        "child_sets_pairwise_disjoint": not any(x == "CHILD_SET_OVERLAP" for x in failures),
        "all_31_existing_operators_covered": len(ops) == 31,
        "failures": failures[:50],
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures: raise G5R4Error("31-operator support failed: " + ",".join(failures[:10]))
    return out


def marker_operator_sentinel() -> dict[str, Any]:
    ad = G4AcceptedAdapter(); ops = tuple(ad.operator_basis()); failures: list[str] = []; rows = []
    # One asymmetric C-C path is enough to exercise each marker operator as an
    # endpoint-typed edge to the unique D leaf.  The theorem itself is operator-generic.
    parent = DecoratedG4Tree(2, ((0, 1),), ("C", "C"), ((0, 0),))
    pcan = ad.unrooted_canon(parent)
    for op in ops:
        rel = _relation_child_canons(ad, parent, marker_H="D", op=op)
        if not rel: failures.append(f"EMPTY:{op}")
        for root in range(parent.n):
            for child in ad.graft_relation(parent, root, new_H_class="D", operator=op):
                if not _check_unique_marker_tree(child, "D"): failures.append(f"UNIQUE:{op}")
                # Structural deletion property: the old parent bytes/fields are a literal prefix of construction.
                recovered = DecoratedG4Tree(child.n - 1, tuple(child.edges[:-1]), tuple(child.H_classes[:-1]), tuple(child.edge_operators[:-1]))
                if ad.unrooted_canon(recovered) != pcan: failures.append(f"RECOVERY:{op}")
        rows.append({"marker_operator": list(op), "relation_child_count": len(rel), "deletion_recovers_parent": True})
    out = {"schema_id": "IG_G5_R4_ALL_31_MARKER_OPERATOR_SENTINEL_V1", "status": "PASS" if not failures else "FAIL", "operator_count": len(ops), "rows": rows, "failures": failures, "unique_marker_deletion_recovery_all_31": not failures}
    out["science_sha256"] = canonical_sha256(out)
    if failures: raise G5R4Error("marker operator sentinel failed: " + ",".join(failures[:10]))
    return out


def run_g5_r4_marked_hidden_minimality(*, engine: Any, r3_primary: Mapping[str, Any], r3_verification: Mapping[str, Any], r3_closeout: Mapping[str, Any]) -> dict[str, Any]:
    s = r4_spec(); auth = verify_authority(r3_primary, r3_verification, r3_closeout)
    theorem = marked_leaf_separation_theorem()
    homogeneous = homogeneous_topology_support(int(s["support_panel"]["homogeneous_max_n"]))
    decorated = all_31_operator_small_support(int(s["support_panel"]["all_31_existing_operator_max_n"]))
    sentinel = marker_operator_sentinel()
    classification = s["outcomes"]["pass"]
    out = {
        "schema_id": "IG_G5_R4_MARKED_LEAF_HIDDEN_MINIMALITY_RESULT_V1",
        "status": "PASS", "stage_ref": "G5:R4", "classification": classification,
        "promotion": False, "g5_graduation_preserved": True, "g6_started": False,
        "authority": auth, "marked_leaf_separation_theorem": theorem,
        "homogeneous_topology_support": homogeneous,
        "all_31_existing_operator_small_support": decorated,
        "all_31_marker_operator_sentinel": sentinel,
        "scoped_hidden_minimality_earned": True,
        "minimality_scope": "MARKER_FREE_PARENTS_UNDER_NONEMPTY_RELATION_VALUED_FRESH_D_LEAF_EXACT_HIDDEN_OBSERVER",
        "global_hidden_minimality_claim": False,
        "exact_hidden_canon_promoted_to_public": False,
        "topology_promoted": False, "public_descriptor_changed": False,
        "next_authorized_stage": "G5:R5_DESIGN_AFTER_HUMAN_REVIEW",
        "cost_control": s["cost_control"], "nonclaims": s["nonclaims"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def stable_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k not in {"source_sha256", "source_version", "registry_sha256", "execution_metadata"}}


def compare_cold_replay(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    ps = canonical_sha256(stable_payload(primary)); cs = canonical_sha256(stable_payload(cold)); failures = []
    if primary.get("science_sha256") != cold.get("science_sha256"): failures.append("SCIENCE_SHA")
    if primary.get("classification") != cold.get("classification"): failures.append("CLASSIFICATION")
    if ps != cs: failures.append("STABLE_PAYLOAD")
    out = {"schema_id": "IG_G5_R4_COLD_REPLAY_COMPARISON_V1", "status": "PASS" if not failures else "FAIL", "certification": "CERTIFIED_PASS" if not failures else "CERTIFICATION_FAILED", "failures": failures, "primary_science_sha256": primary.get("science_sha256"), "cold_science_sha256": cold.get("science_sha256"), "stable_scientific_payload_exact_equal": ps == cs, "stable_scientific_payload_sha256": ps}
    out["comparison_sha256"] = canonical_sha256(out); return out


def certified_closeout(primary: Mapping[str, Any], cold: Mapping[str, Any], comparison: Mapping[str, Any], independent: Mapping[str, Any]) -> dict[str, Any]:
    ok = primary.get("status") == "PASS" and cold.get("status") == "PASS" and comparison.get("certification") == "CERTIFIED_PASS" and independent.get("status") == "PASS"
    out = {"schema_id": "IG_G5_R4_CERTIFIED_CLOSEOUT_V1", "status": "CERTIFIED_PASS" if ok else "CERTIFICATION_FAILED", "experiment_id": "G5:R4.MARKED_LEAF_HIDDEN_MINIMALITY", "classification": primary.get("classification") if ok else "G5_R4_REVIEW_REQUIRED_NO_PROMOTION", "science_sha256": primary.get("science_sha256"), "stable_science_payload_sha256": comparison.get("stable_scientific_payload_sha256"), "comparison_sha256": comparison.get("comparison_sha256"), "independent_verification_sha256": independent.get("verification_sha256"), "source_sha256": primary.get("source_sha256"), "registry_sha256": primary.get("registry_sha256"), "promotion": False, "g5_graduation_preserved": True, "scoped_hidden_minimality_earned": bool(primary.get("scoped_hidden_minimality_earned")) if ok else False, "global_hidden_minimality_claim": False, "topology_promoted": False, "g6_started": False, "next_authorized_stage": "G5:R5_DESIGN_AFTER_HUMAN_REVIEW" if ok else None, "failures": comparison.get("failures", [])}
    out["closeout_sha256"] = canonical_sha256(out); return out
