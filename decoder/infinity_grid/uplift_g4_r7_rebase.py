from __future__ import annotations

"""G4:R7.REBASE — actual-G4 realizability bridge and observer-relative quotient challenge.

This stage closes the specific caveats left by the Drive-authoritative G4:R6:
1) transfer the fixed-depth lower bound from the abstract decorated-tree grammar to a
   constructively resource-realizable G4 subcarrier, and
2) evaluate a preregistered parent-conditioned finite-depth action-read family on a
   fresh n=7 challenge without changing the public G4 descriptor.
"""

from collections import defaultdict
from importlib.resources import files
from itertools import combinations
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .decorated_tree_messages import (
    graft_fixed_leaf,
    rooted_tree_canon,
    rooted_truncated_cavity_message,
    prepare_decorated_tree,
    unrooted_tree_canon,
)
from .uplift_g3_r1 import _generate_tree_shapes


class G4R7RebaseError(RuntimeError):
    pass


_SPEC = "resources/uplift/G4_R7_REBASE_REALIZABILITY_OBSERVER_QUOTIENT_SPEC_V1.json"
_S3_FIXTURE = "resources/uplift/authority_fixtures/G4_S3_REBASE_V03050/RESULT.json"
_S6_SPEC = "resources/uplift/G4_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json"


def _resource(name: str):
    return files("infinity_grid").joinpath(name)


def spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G4_R7_REBASE_REALIZABILITY_OBSERVER_QUOTIENT_SPEC_V1":
        raise G4R7RebaseError("bad R7 spec schema")
    expected = obj.get("science_sha256")
    if canonical_sha256({k: v for k, v in obj.items() if k != "science_sha256"}) != expected:
        raise G4R7RebaseError("R7 spec hash mismatch")
    return obj


def _load_embedded_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    s3 = json.loads(_resource(_S3_FIXTURE).read_text(encoding="utf-8"))
    s6 = json.loads(_resource(_S6_SPEC).read_text(encoding="utf-8"))
    return s3, s6


def verify_authority(
    *,
    r6: Mapping[str, Any],
    r6_replay: Mapping[str, Any],
    r6_closeout: Mapping[str, Any],
    branch_reconciliation: Mapping[str, Any],
    r4_r5_correction: Mapping[str, Any],
) -> dict[str, Any]:
    sp = spec()
    a = sp["authority"]
    s3, s6 = _load_embedded_authority()
    failures: list[str] = []

    if r6.get("schema_id") != "IG_G4_R6_REBASE_GENERIC_DECORATED_MESSAGE_LAW_RESULT_V1" or r6.get("status") != "PASS":
        failures.append("R6_RESULT")
    if r6.get("science_sha256") != a["drive_authoritative_r6_primary_science_sha256"]:
        failures.append("R6_PRIMARY_IDENTITY")
    if r6.get("classification") != a["required_r6_classification"]:
        failures.append("R6_CLASSIFICATION")
    if r6.get("source_sha256") != a["drive_authoritative_r6_source_sha256"]:
        failures.append("R6_SOURCE_IDENTITY")
    if r6.get("public_descriptor_changed") is not False or r6.get("g4_graduation_preserved") is not True:
        failures.append("R6_PUBLIC_G4_STATUS")
    if r6.get("resource_realizable_g4_unbounded_lower_bound_earned") is not False:
        failures.append("R6_REALIZABILITY_CAVEAT")

    if r6_replay.get("schema_id") != "IG_G4_R6_REBASE_REPLAY_COMPARISON_V1" or r6_replay.get("status") != "PASS":
        failures.append("R6_REPLAY")
    if r6_replay.get("science_sha256") != a["drive_authoritative_r6_replay_science_sha256"]:
        failures.append("R6_REPLAY_IDENTITY")
    if r6_closeout.get("schema_id") != "IG_G4_R6_REBASE_CERTIFIED_CLOSEOUT_V1" or r6_closeout.get("status") != "CERTIFIED_PASS":
        failures.append("R6_CLOSEOUT")
    if r6_closeout.get("science_sha256") != a["drive_authoritative_r6_closeout_science_sha256"]:
        failures.append("R6_CLOSEOUT_IDENTITY")
    if r6_closeout.get("public_descriptor") != a["required_public_descriptor"]:
        failures.append("PUBLIC_DESCRIPTOR")

    if branch_reconciliation.get("schema_id") != "IG_G4_R6_BRANCH_RECONCILIATION_V1" or branch_reconciliation.get("status") != "PASS":
        failures.append("R6_BRANCH_RECONCILIATION")
    if branch_reconciliation.get("science_sha256") != a["r6_branch_reconciliation_science_sha256"]:
        failures.append("R6_BRANCH_RECONCILIATION_IDENTITY")
    if branch_reconciliation.get("reconciliation_decision", {}).get("frontier_authority") != "DRIVE_BRANCH":
        failures.append("R6_FRONTIER_AUTHORITY")

    if r4_r5_correction.get("schema_id") != "IG_G4_R4_R5_INTERPRETATION_CORRECTION_V1" or r4_r5_correction.get("status") != "PASS":
        failures.append("R4_R5_CORRECTION")
    if r4_r5_correction.get("science_sha256") != a["r4_r5_interpretation_correction_science_sha256"]:
        failures.append("R4_R5_CORRECTION_IDENTITY")
    if r4_r5_correction.get("historical_results_edited") is not False:
        failures.append("R4_R5_HISTORY_MUTATION")

    if s3.get("schema_id") != "IG_G4_S3_REBASE_HIGHER_ORDER_RESULT_V1" or s3.get("status") != "PASS":
        failures.append("S3_FIXTURE")
    if s3.get("science_sha256") != a["s3_rebase_result_science_sha256"]:
        failures.append("S3_IDENTITY")
    if s3.get("candidate_interface", {}).get("science_sha256") != a["s3_candidate_interface_science_sha256"]:
        failures.append("S3_CANDIDATE_INDEX")

    if s6.get("science_sha256") != a["s6_operator_basis_spec_science_sha256"]:
        failures.append("S6_OPERATOR_SPEC_IDENTITY")
    ops = [tuple(map(int, x)) for x in s6.get("frozen_grammar", {}).get("operator_basis", [])]
    if len(set(ops)) != 31 or (0, 0) not in set(ops):
        failures.append("S6_OPERATOR_00")

    out = {
        "schema_id": "IG_G4_R7_REBASE_AUTHORITY_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "r6_science_sha256": r6.get("science_sha256"),
        "r6_replay_science_sha256": r6_replay.get("science_sha256"),
        "r6_closeout_science_sha256": r6_closeout.get("science_sha256"),
        "branch_reconciliation_science_sha256": branch_reconciliation.get("science_sha256"),
        "r4_r5_correction_science_sha256": r4_r5_correction.get("science_sha256"),
        "s3_science_sha256": s3.get("science_sha256"),
        "s3_candidate_interface_science_sha256": s3.get("candidate_interface", {}).get("science_sha256"),
        "operator_00_certified": (0, 0) in set(ops),
        "public_descriptor": "CAPS7_PLUS_H_CLASS_BAG",
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G4R7RebaseError("authority failed: " + ",".join(failures))
    return out


def _candidate_rows() -> dict[str, dict[str, Any]]:
    s3, _ = _load_embedded_authority()
    rows = s3["candidate_interface"]["rows"]
    out = {str(r["term_key"]): dict(r) for r in rows}
    if set(out) != {"A", "B", "C", "D"}:
        raise G4R7RebaseError("unexpected repaired-H candidate alphabet")
    return out


def _resource_realizability_certificate(max_support_depth: int = 12) -> dict[str, Any]:
    """Constructive transfer of the R6 path/broom lower bound to actual G4.

    The general resource lemma follows directly from the certified G3TermState
    reservation semantics: if total_caps[t] > 0, at least one local owner has
    positive t-capacity; a successor exists and lowers the total by exactly one.
    Hence k sequential reservations exist whenever the initial total is >= k.
    """
    rows = _candidate_rows()
    c = rows["C"]
    drow = rows["D"]
    ccaps = tuple(map(int, c["caps7"]))
    dcaps = tuple(map(int, drow["caps7"]))
    if ccaps != dcaps:
        raise G4R7RebaseError("registered C/D same-CAPS pair disappeared")
    if ccaps[0] < 3:
        raise G4R7RebaseError("C type-0 resource does not support degree-three witness")

    support = []
    edge_label = [0, 0]
    c_label = str(c["candidate_key_sha256"])
    for depth in range(int(max_support_depth) + 1):
        n = depth + 3
        path_edges = [(i, i + 1) for i in range(n - 1)]
        broom_edges = [(i, i + 1) for i in range(depth)] + [(depth, depth + 1), (depth, depth + 2)]
        colors = [c_label] * n
        grades_path = [edge_label] * len(path_edges)
        grades_broom = [edge_label] * len(broom_edges)
        mp = rooted_truncated_cavity_message(n, path_edges, colors, grades_path, 0, depth)
        mb = rooted_truncated_cavity_message(n, broom_edges, colors, grades_broom, 0, depth)
        if mp != mb:
            raise G4R7RebaseError(f"same-Q witness local messages split at depth {depth}")
        cp = graft_fixed_leaf(n, path_edges, colors, grades_path, 0, new_vertex_label=c_label, new_edge_label=edge_label)
        cb = graft_fixed_leaf(n, broom_edges, colors, grades_broom, 0, new_vertex_label=c_label, new_edge_label=edge_label)
        ocp = unrooted_tree_canon(*cp)
        ocb = unrooted_tree_canon(*cb)
        if ocp == ocb:
            raise G4R7RebaseError(f"same-Q witness children failed to separate at depth {depth}")
        # Public Q is identical: same C-bag cardinality and same number of (0,0) edges.
        final_caps = list(ccaps[i] * n for i in range(7))
        final_caps[0] -= 2 * (n - 1)
        support.append({
            "depth": depth,
            "g3_unit_count": n,
            "path_max_degree": 2 if n > 2 else 1,
            "broom_max_degree": 2 if depth == 0 else 3,
            "shared_public_caps7": final_caps,
            "shared_H_class_bag": {c_label: n},
            "truncated_action_read_sha256": canonical_sha256(mp),
            "path_child_canon_sha256": canonical_sha256(ocp),
            "broom_child_canon_sha256": canonical_sha256(ocb),
            "child_canons_distinct": True,
        })

    out = {
        "schema_id": "IG_G4_R7_REBASE_ACTUAL_RESOURCE_REALIZABILITY_THEOREM_V1",
        "status": "PASS",
        "actual_certified_g3_seed": {
            "term_key": "C",
            "term_ref": c["term_ref"],
            "H_class": c_label,
            "caps7": list(ccaps),
            "type0_capacity": ccaps[0],
        },
        "same_caps_distinct_H_partner": {
            "term_key": "D",
            "term_ref": drow["term_ref"],
            "H_class": drow["candidate_key_sha256"],
            "same_caps7_as_C": True,
        },
        "certified_g4_operator": [0, 0],
        "reservation_lemma": {
            "statement": "For any exact G3TermState and endpoint type t, total_caps[t]>0 implies reserve_external_relation(t) is nonempty; every returned successor lowers total_caps[t] by exactly one and changes no typed edge.",
            "proof": "total_caps[t] is the finite sum of nonnegative local node_caps[t]. Positive sum implies a positive local owner. The frozen reservation constructor decrements exactly one such local entry. Induction gives k sequential reservations whenever initial total_caps[t]>=k.",
            "implementation_reference": "infinity_grid.g4_term_state:G3TermState.reserve_external_relation",
            "status": "PROVED_FROM_FROZEN_EXACT_RESOURCE_SEMANTICS",
        },
        "sequential_leaf_attachment_lemma": {
            "statement": "Every finite tree of maximum degree at most 3 on copies of certified C is resource-realizable using only (0,0) G4 leaf attachments.",
            "proof": "Reverse a leaf-pruning order. Each final incident G4 edge consumes exactly one type-0 reservation from that vertex. No vertex needs more than degree<=3 reservations, while certified C has type-0 capacity >=3. Apply the reservation lemma at every attachment.",
            "status": "PROVED_FROM_RESERVATION_LEMMA_AND_TREE_LEAF_ORDER",
        },
        "same_Q_parent_conditioned_lower_bound": {
            "statement": "For every fixed finite d, the endpoint-rooted C-only path/broom pair on n=d+3 actual G4 units has equal public Q and equal depth-d R6 cavity action read, but the same fixed C/(0,0) leaf graft produces distinct exact hidden child canons.",
            "public_Q": "CAPS7_PLUS_H_CLASS_BAG",
            "action_read": "R6_NONBACKTRACKING_CAVITY_MESSAGE_AT_FIXED_DEPTH_d",
            "child_observer": "EXACT_UNROOTED_DECORATED_CHILD_CANON",
            "status": "PROVED_FOR_AN_ACTUAL_RESOURCE_REALIZABLE_G4_SUBCARRIER",
        },
        "support_checks_depth_0_through": int(max_support_depth),
        "support_rows": support,
        "unbounded_claim_scope": "THE_SPECIFIED_C_ONLY_MAX_DEGREE_3_(0,0)_G4_SUBCARRIER_AND_THE_FROZEN_PARENT_CONDITIONED_EXACT_CHILD_OBSERVER",
        "global_fixed_depth_sufficient": False,
        "public_descriptor_changed": False,
        "topology_promoted": False,
        "physical_geometry_claim": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _public_q_wire(caps_c: tuple[int, ...], caps_d: tuple[int, ...], c_count: int, d_count: int) -> dict[str, Any]:
    n = int(c_count) + int(d_count)
    caps = [int(c_count) * caps_c[i] + int(d_count) * caps_d[i] for i in range(7)]
    caps[0] -= 2 * (n - 1)
    return {
        "schema_id": "IG_G4_PUBLIC_Q_CAPS7_H_BAG_V1",
        "caps7": caps,
        "H_class_bag": {"C": int(c_count), "D": int(d_count)},
    }


def _fresh_n7_observer_challenge() -> dict[str, Any]:
    sp = spec()["fresh_challenge"]
    if int(sp["n"]) != 7 or list(sp["candidate_depths"]) != list(range(7)):
        raise G4R7RebaseError("fresh challenge registration changed")
    shapes = _generate_tree_shapes(7)[7]
    if len(shapes) != int(sp["unlabelled_tree_shapes_expected"]):
        raise G4R7RebaseError("n7 tree shape count mismatch")

    rows = _candidate_rows()
    c = rows["C"]
    drow = rows["D"]
    caps_c = tuple(map(int, c["caps7"]))
    caps_d = tuple(map(int, drow["caps7"]))
    if caps_c != caps_d or caps_c[0] < 7:
        raise G4R7RebaseError("C/D resource basis is not fit for n7 challenge")
    C = str(c["candidate_key_sha256"])
    D = str(drow["candidate_key_sha256"])
    edge_label = [0, 0]

    raw_instances = 0
    exact_cases: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    fiber_raw: dict[str, int] = defaultdict(int)

    for fiber in sp["public_Q_fibers"]:
        cc, dd = int(fiber["C_count"]), int(fiber["D_count"])
        if cc + dd != 7:
            raise G4R7RebaseError("bad Q fiber arity")
        q_wire = _public_q_wire(caps_c, caps_d, cc, dd)
        q_hash = canonical_sha256(q_wire)
        for edges in shapes.values():
            grades = [edge_label] * 6
            for dpos in combinations(range(7), dd):
                colors = [C] * 7
                for j in dpos:
                    colors[j] = D
                # P8 exact cost optimization: validate/freeze this decorated parent once,
                # then share directed structural messages across every root and depth.
                # The prepared cache stores full tuples; no digest participates in equality.
                prepared = prepare_decorated_tree(7, edges, colors, grades)
                rooted_canons = prepared.all_rooted_canons()
                messages_by_root = tuple(
                    tuple(prepared.rooted_truncated(root, depth) for depth in range(7))
                    for root in range(7)
                )
                for root in range(7):
                    raw_instances += 1
                    fiber_raw[q_hash] += 1
                    rc = rooted_canons[root]
                    key = (q_hash, rc)
                    cn, ce, cv, cg = graft_fixed_leaf(
                        7, edges, colors, grades, root,
                        new_vertex_label=C, new_edge_label=edge_label,
                    )
                    child = unrooted_tree_canon(cn, ce, cv, cg)
                    msgs = messages_by_root[root]
                    prior = exact_cases.get(key)
                    rec = {
                        "q_hash": q_hash,
                        "q_wire": q_wire,
                        "rooted_parent_canon": rc,
                        "child_canon": child,
                        "messages": msgs,
                    }
                    if prior is None:
                        exact_cases[key] = rec
                    elif prior["child_canon"] != child or prior["messages"] != msgs:
                        raise G4R7RebaseError("rooted-canon dedup failed invariance")

    # Registered raw count is 11*(C(7,1)+C(7,2))*7 = 2156.
    expected_raw = 11 * (7 + 21) * 7
    if raw_instances != expected_raw:
        raise G4R7RebaseError("fresh raw action count mismatch")

    # P8 exact structural interning.  The child canon tuple itself is the key.
    # Python dict lookup confirms full tuple equality even if internal hash values
    # collide, so no digest collision can merge scientific states.  Compact integer
    # ids are used only inside this census and never become public scientific data.
    child_canon_ids: dict[tuple[Any, ...], int] = {}
    def intern_child_canon(canon: tuple[Any, ...]) -> int:
        found = child_canon_ids.get(canon)
        if found is not None:
            return found
        ident = len(child_canon_ids) + 1
        child_canon_ids[canon] = ident
        return ident
    for rec in exact_cases.values():
        rec["_p8_child_canon_id"] = intern_child_canon(rec["child_canon"])

    per_depth = []
    fiber_rows = []
    first_predictive = None
    for depth in range(7):
        groups: dict[tuple[str, tuple[Any, ...]], set[int]] = defaultdict(set)
        for rec in exact_cases.values():
            groups[(rec["q_hash"], rec["messages"][depth])].add(rec["_p8_child_canon_id"])
        conflicts = [vals for vals in groups.values() if len(vals) > 1]
        row = {
            "depth": depth,
            "action_read_class_count": len(groups),
            "nonpredictive_class_count": len(conflicts),
            "max_distinct_child_canons_in_one_QA_class": max((len(x) for x in groups.values()), default=0),
            "predictive_on_registered_fresh_panel": not conflicts,
            "class_count_ratio_to_exact_rooted_action_cases": len(groups) / len(exact_cases),
        }
        if not conflicts and first_predictive is None:
            first_predictive = depth
        per_depth.append(row)

    q_hashes = sorted({rec["q_hash"] for rec in exact_cases.values()})
    for qh in q_hashes:
        cases = [r for r in exact_cases.values() if r["q_hash"] == qh]
        fr = {"q_hash": qh, "public_Q": cases[0]["q_wire"], "raw_action_instances": fiber_raw[qh], "exact_rooted_action_cases": len(cases), "depths": []}
        for depth in range(7):
            groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
            for rec in cases:
                groups[rec["messages"][depth]].add(rec["_p8_child_canon_id"])
            conflicts = [v for v in groups.values() if len(v) > 1]
            fr["depths"].append({
                "depth": depth,
                "action_read_class_count": len(groups),
                "nonpredictive_class_count": len(conflicts),
                "max_distinct_child_canons_in_one_action_class": max((len(v) for v in groups.values()), default=0),
                "predictive": not conflicts,
                "class_count_ratio_to_exact_rooted_action_cases": len(groups) / len(cases),
            })
        fiber_rows.append(fr)

    baseline_groups = {(rec["q_hash"], rec["rooted_parent_canon"]) for rec in exact_cases.values()}
    if len(baseline_groups) != len(exact_cases):
        raise G4R7RebaseError("exact baseline dedup collision")
    # At depth 6 every n7 rooted tree has fully arrived, so it must equal rooted canon.
    if not per_depth[6]["predictive_on_registered_fresh_panel"]:
        raise G4R7RebaseError("depth6 exact-rooted baseline failed on n7")
    for rec in exact_cases.values():
        if rec["messages"][6] != rec["rooted_parent_canon"]:
            raise G4R7RebaseError("depth6 message does not equal exact rooted canon")

    first_predictive_row = per_depth[first_predictive] if first_predictive is not None else None
    nontrivial = bool(first_predictive_row and first_predictive_row["class_count_ratio_to_exact_rooted_action_cases"] < 1.0)
    out = {
        "schema_id": "IG_G4_R7_REBASE_FRESH_N7_PARENT_CONDITIONED_OBSERVER_CHALLENGE_V1",
        "status": "PASS",
        "observer_system": spec()["frozen_observer_system"],
        "raw_action_instances": raw_instances,
        "unlabelled_tree_shapes": len(shapes),
        "exact_rooted_action_cases_after_dedup": len(exact_cases),
        "exact_child_canon_count": len(child_canon_ids),
        "depth_rows": per_depth,
        "first_predictive_depth_on_fresh_n7_panel": first_predictive,
        "nontrivial_reduction_candidate_survived": nontrivial,
        "candidate_scope": "FRESH_N7_C_D_H_CLASS_(0,0)_EDGE_ACTUAL_RESOURCE_REALIZABLE_PANEL_ONLY",
        "global_minimality_claim": False,
        "recursive_future_congruence_claim": False,
        "fiber_rows": fiber_rows,
        "stopping_rule_honored": True,
        "n8_or_larger_executed": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def finalize(
    *,
    r6: Mapping[str, Any],
    r6_replay: Mapping[str, Any],
    r6_closeout: Mapping[str, Any],
    branch_reconciliation: Mapping[str, Any],
    r4_r5_correction: Mapping[str, Any],
) -> dict[str, Any]:
    auth = verify_authority(
        r6=r6,
        r6_replay=r6_replay,
        r6_closeout=r6_closeout,
        branch_reconciliation=branch_reconciliation,
        r4_r5_correction=r4_r5_correction,
    )
    realizability = _resource_realizability_certificate(12)
    challenge = _fresh_n7_observer_challenge()
    first = challenge["first_predictive_depth_on_fresh_n7_panel"]
    if challenge["nontrivial_reduction_candidate_survived"]:
        finite_status = f"BOUNDED_N7_DEPTH{first}_PARENT_CONDITIONED_COMPRESSION_CANDIDATE_SURVIVED"
    else:
        finite_status = f"FRESH_N7_FIRST_PREDICTIVE_DEPTH{first}_BUT_PARTITION_EQUALS_EXACT_ROOTED_CANON_NO_NONTRIVIAL_INFORMATION_REDUCTION"
    classification = (
        "G4_R7_REBASE_ACTUAL_G4_RESOURCE_REALIZABILITY_BRIDGE_EARNED_"
        "PARENT_CONDITIONED_NO_FIXED_FINITE_LOCAL_DEPTH_LOWER_BOUND_EARNED_"
        + finite_status
        + "_G4_INVESTIGATION_COMPLETION_READY_G5_UNLAUNCHED"
    )
    out = {
        "schema_id": "IG_G4_R7_REBASE_REALIZABILITY_OBSERVER_QUOTIENT_RESULT_V1",
        "status": "PASS",
        "stage_ref": "G4:R7.REBASE",
        "classification": classification,
        "authority": auth,
        "frozen_observer_system": spec()["frozen_observer_system"],
        "actual_g4_resource_realizability": realizability,
        "fresh_n7_parent_conditioned_challenge": challenge,
        "scientific_endpoint": {
            "public_g4_law_graduation_preserved": True,
            "public_descriptor": "CAPS7_PLUS_H_CLASS_BAG",
            "r4_r5_interpretation_corrected_without_history_edit": True,
            "r6_drive_branch_authoritative": True,
            "r6_abstract_to_actual_g4_realizability_gap_closed_for_registered_witness_subcarrier": True,
            "parent_conditioned_fixed_depth_global_sufficiency": False,
            "finite_fresh_candidate_status": finite_status,
            "global_exact_rooted_canon_minimality_earned": False,
            "recursive_future_congruence_earned": False,
            "g5_started": False,
        },
        "promotion": False,
        "public_descriptor_changed": False,
        "topology_promoted": False,
        "raw_h_promoted": False,
        "g4_graduation_preserved": True,
        "g5_started": False,
        "next_authorized_work": "DECODER_ARCHITECTURE_PROGRAM_AFTER_G4_COMPLETION_RECORD; G5_REQUIRES_SEPARATE_AUTHORIZATION",
        "cost_control": spec()["cost_control"],
        "nonclaims": spec()["nonclaims"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _stable_science_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in result.items()
        if k not in {"source_sha256", "source_version", "registry_sha256", "execution_metadata"}
    }


def compare(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    p = _stable_science_payload(primary)
    c = _stable_science_payload(cold)
    checks = {
        "primary_pass": primary.get("status") == "PASS",
        "cold_pass": cold.get("status") == "PASS",
        "full_stable_science_payload_equal": p == c,
        "primary_science_hash_recomputes": canonical_sha256({k: v for k, v in p.items() if k != "science_sha256"}) == primary.get("science_sha256"),
        "cold_science_hash_recomputes": canonical_sha256({k: v for k, v in c.items() if k != "science_sha256"}) == cold.get("science_sha256"),
        "science_sha_equal": primary.get("science_sha256") == cold.get("science_sha256"),
        "source_sha_equal": primary.get("source_sha256") == cold.get("source_sha256"),
        "registry_sha_equal": primary.get("registry_sha256") == cold.get("registry_sha256"),
        "source_version_equal": primary.get("source_version") == cold.get("source_version"),
    }
    ok = all(checks.values())
    out = {
        "schema_id": "IG_G4_R7_REBASE_REPLAY_COMPARISON_V1",
        "status": "PASS" if ok else "FAIL",
        "certification": "CERTIFIED_PASS" if ok else "NOT_CERTIFIED",
        "checks": checks,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
        "stable_payload_sha256": canonical_sha256(p) if p == c else None,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def closeout(primary: Mapping[str, Any], cold: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    ok = replay.get("status") == "PASS" and replay.get("certification") == "CERTIFIED_PASS"
    out = {
        "schema_id": "IG_G4_R7_REBASE_CERTIFIED_CLOSEOUT_V1",
        "status": "CERTIFIED_PASS" if ok else "CERTIFICATION_FAIL",
        "classification": primary.get("classification") if ok else None,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
        "replay_science_sha256": replay.get("science_sha256"),
        "full_stable_science_payload_equal": replay.get("checks", {}).get("full_stable_science_payload_equal") if ok else False,
        "actual_g4_resource_realizability_bridge_earned": True if ok else None,
        "parent_conditioned_no_fixed_finite_depth_lower_bound_earned": True if ok else None,
        "fresh_n7_first_predictive_depth": primary.get("fresh_n7_parent_conditioned_challenge", {}).get("first_predictive_depth_on_fresh_n7_panel") if ok else None,
        "g4_investigation_completion_ready": bool(ok),
        "g4_graduation_preserved": bool(ok),
        "public_descriptor": "CAPS7_PLUS_H_CLASS_BAG" if ok else None,
        "public_descriptor_changed": False,
        "topology_promoted": False,
        "g5_started": False,
        "next_authorized_work": "DECODER_ARCHITECTURE_PROGRAM_AFTER_G4_COMPLETION_RECORD" if ok else None,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out
