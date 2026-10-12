from __future__ import annotations

"""G2:S6 recursive CAPS7 closure and graduation audit.

This module does not change the exact transition semantics.  It consumes the v0.30.3
S5 result and proves/attacks the recursive consequence of the already-earned CAPS7
read/write factorisation.  The exact holdout family is deliberately new at G2: every
holdout state contains at least three frozen G1:R100 leaves and therefore was not a
member of the S5 pair census.

Primary S6 PASS is only a graduation *candidate*. A separate cold-source rerun of the
same registered Decoder-native G2:S6 experiment must reproduce the stable S6 science
payload before ``finalize_s6_graduation`` can emit a G2 graduation / R0 unlock certificate.
No standalone alternate S6 science implementation is accepted in the certification chain.
"""

from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import inspect
import json

from .canon import canonical_sha256
from .g2_relation import relation_spec, reserve_external_relation, compose_binary_relation
from .uplift_s5 import reserve_write, binary_write


class UpliftS6Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G_UPLIFT_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json"


def s6_implementation_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if observed != expected:
        raise UpliftS6Error(f"S6 implementation spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def _sha(v: Any, field: str) -> str:
    s = str(v)
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise UpliftS6Error(f"{field} must be lowercase sha256")
    return s


def _caps(v: Sequence[Any], field: str = "caps") -> tuple[int, ...]:
    out = tuple(int(x) for x in v)
    if len(out) != 7 or any(x < 0 for x in out):
        raise UpliftS6Error(f"{field} must be seven nonnegative integers")
    return out


def verify_s6_authority(*, unlock_certificate: Mapping[str, Any], s5_result: Mapping[str, Any]) -> dict[str, Any]:
    spec = s6_implementation_spec()
    failures: list[str] = []
    if unlock_certificate.get("schema_id") != "IG_G2_S6_UNLOCK_CERTIFICATE_V1":
        failures.append("UNLOCK_SCHEMA")
    if unlock_certificate.get("status") != "PASS":
        failures.append("UNLOCK_STATUS")
    if unlock_certificate.get("authorizes") != "G2:S6_RECURSIVE_CLOSURE_AND_GRADUATION_AUDIT_UNDER_CAPS7":
        failures.append("UNLOCK_AUTHORIZATION")
    if unlock_certificate.get("g2_graduated") is not False:
        failures.append("PREMATURE_GRADUATION")
    if _sha(unlock_certificate.get("science_sha256"), "S6 unlock science") != spec["parent_s6_unlock_science_sha256"]:
        failures.append("UNLOCK_SCIENCE")
    if _sha(unlock_certificate.get("source_sha256"), "S6 unlock parent source") != spec["parent_source_sha256"]:
        failures.append("UNLOCK_SOURCE")
    if s5_result.get("schema_id") != "IG_G2_S5_FINITE_STATE_DESCRIPTOR_RESULT_V1":
        failures.append("S5_SCHEMA")
    if s5_result.get("status") != "PASS" or s5_result.get("s6_unlocked") is not True:
        failures.append("S5_STATUS")
    if s5_result.get("g2_graduated") is not False:
        failures.append("S5_GRADUATION_FIREWALL")
    if _sha(s5_result.get("science_sha256"), "S5 science") != spec["parent_s5_science_sha256"]:
        failures.append("S5_SCIENCE")
    if _sha(s5_result.get("source_sha256"), "S5 source") != spec["parent_source_sha256"]:
        failures.append("S5_SOURCE")
    if s5_result.get("observer_scope", {}).get("name") != spec["observer"]["name"]:
        failures.append("OBSERVER_SCOPE")
    if relation_spec()["spec_sha256"] != spec["g2_relation_spec_sha256"]:
        failures.append("RELATION_SPEC")
    out = {
        "schema_id": "IG_G2_S6_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "s5_science_sha256": s5_result.get("science_sha256"),
        "s6_unlock_science_sha256": unlock_certificate.get("science_sha256"),
        "s6_spec_science_sha256": spec["science_sha256"],
        "g2_graduated_before_s6": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise UpliftS6Error("S6 authority verification failed: " + ",".join(failures))
    return out


def verify_regression_evidence(regression: Mapping[str, Any], *, current_source_sha256: str) -> dict[str, Any]:
    failures: list[str] = []
    if regression.get("schema_id") != "IG_G2_S6_REGRESSION_GATE_V1":
        failures.append("SCHEMA")
    if regression.get("status") != "PASS":
        failures.append("STATUS")
    if regression.get("return_code") != 0:
        failures.append("RETURN_CODE")
    if _sha(regression.get("source_sha256"), "regression source") != _sha(current_source_sha256, "current source"):
        failures.append("SOURCE_IDENTITY")
    if int(regression.get("passed", 0)) <= 0:
        failures.append("NO_PASS_COUNT")
    out = {
        "schema_id": "IG_G2_S6_REGRESSION_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "source_sha256": current_source_sha256,
        "passed": int(regression.get("passed", 0)),
        "skipped": int(regression.get("skipped", 0)),
        "return_code": int(regression.get("return_code", -1)),
        "suite_summary": str(regression.get("suite_summary", "")),
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise UpliftS6Error("S6 regression gate failed: " + ",".join(failures))
    return out


def recursive_caps7_factorisation_argument(s5_result: Mapping[str, Any]) -> dict[str, Any]:
    """Machine-readable structural-induction proof obligation discharge.

    The mathematical generality comes from the induction, not from the fresh holdout.
    We re-check that every S5 premise used by the induction is present and passed.
    """
    failures: list[str] = []
    required = {
        "authority": s5_result.get("authority", {}).get("status"),
        "s0_base_reservation_factorisation": s5_result.get("s0_base_reservation_factorisation", {}).get("status"),
        "s4_binary_write_factorisation": s5_result.get("s4_binary_write_factorisation", {}).get("status"),
        "implementation_read_surface_audit": s5_result.get("implementation_read_surface_audit", {}).get("status"),
        "factorisation_argument": s5_result.get("factorisation_argument", {}).get("status"),
    }
    for k, v in required.items():
        if v != "PASS":
            failures.append(k)
    if s5_result.get("candidate_descriptor", {}).get("coordinate_count") != 7:
        failures.append("CAPS7_DIMENSION")
    if s5_result.get("g2_relation_spec_sha256") != relation_spec()["spec_sha256"]:
        failures.append("RELATION_SPEC")

    proof = {
        "proof_method": "STRUCTURAL_INDUCTION_ON_FINITE_G2_CONSTRUCTION_AND_ACTION_SYNTAX",
        "base_case": "For every frozen G1 leaf x, q(x)=CAPS7(x) is in N^7; S5 checked all 1,351 available leaf reservation rows and q(R_t(x))=q(x)-e_t.",
        "reservation_inductive_step": "Let X be a recursively formed G2 state with q(X)=f. Relation-valued R_t enumerates every eligible immediate owner. Each recursive child branch loses exactly e_t by induction; all other child capacities are unchanged. Since q is the coordinatewise child sum, every exact successor branch has q=f-e_t. Positive f_t is equivalent to existence of at least one eligible child.",
        "binary_inductive_step": "For recursively formed X,Y with q(X)=f,q(Y)=g, C_ab enumerates the Cartesian product of their complete reservation relations. Every left branch has f-e_a and every right branch g-e_b; wrapping them sums capacities, so every exact output branch has q=f+g-e_a-e_b.",
        "finite_context_congruence": "Induction on any finite action/context tree therefore preserves equality of q. Exact owner alternatives may differ, but each admitted action has one quotient successor and the complete quotient future depends only on CAPS7.",
        "closure": "All enabled writes subtract only coordinates known positive before the write; therefore every recursive quotient successor stays in N^7.",
        "rebracketing_scope": "At the CAPS7 observer, any two finite construction trees with the same leaf capacity sum and same multiset of typed bridge consumptions have the same quotient output. No raw-topology associativity is claimed.",
        "finite_action_alphabet": {"reservation_actions": 7, "binary_bridge_actions": 31, "total_action_schemata": 38},
        "state_coordinate_dimension": 7,
        "state_coordinate_domain": "N",
        "finite_state_cardinality_claimed": False,
        "minimality_claimed": False,
    }
    out = {
        "schema_id": "IG_G2_S6_RECURSIVE_CAPS7_FACTORISATION_THEOREM_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "premises": required,
        "proof": proof,
        "consequence": "CAPS7_IS_A_RECURSIVE_BRANCH_QUOTIENT_CONGRUENCE_FOR_ALL_FINITE_TERMS_OF_THE_FROZEN_G2_GRAMMAR" if not failures else "PREMISES_NOT_MET",
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def deterministic_holdout_plan(*, carrier_count: int, bridge_pairs: Sequence[Sequence[int]]) -> dict[str, Any]:
    if int(carrier_count) != 193:
        raise UpliftS6Error(f"S6 holdout requires complete 193-carrier R100 cohort, got {carrier_count}")
    ops = sorted({(int(a), int(b)) for a, b in bridge_pairs})
    if len(ops) != 31:
        raise UpliftS6Error(f"S6 holdout requires frozen 31-operator basis, got {len(ops)}")
    selected = [0, 24, 48, 72, 96, 120, 144, 168, 192]
    depth2 = []
    for j in range(31):
        depth2.append({
            "case_id": f"D2_{j:02d}",
            "carrier_slots": [selected[j % 9], selected[(2*j + 1) % 9], selected[(3*j + 2) % 9]],
            "base_operator_index": (7*j + 5) % 31,
            "recursive_operator_index": j,
        })
    deep = []
    for c in range(7):
        deep.append({
            "case_id": f"D4_{c:02d}",
            "carrier_slots": [selected[(c + k) % 9] for k in range(5)],
            "operator_indices": [(5*c + 8*k) % 31 for k in range(4)],
        })
    rebracket = []
    for c in range(7):
        rebracket.append({
            "case_id": f"RB_{c:02d}",
            "carrier_slots": [selected[(c + 2*k) % 9] for k in range(4)],
            "operator_indices": [(3*c + 1) % 31, (3*c + 11) % 31, (3*c + 21) % 31],
        })
    out = {
        "schema_id": "IG_G2_S6_FRESH_RECURSIVE_HOLDOUT_PLAN_V1",
        "status": "FROZEN_BEFORE_HOLDOUT_EXECUTION",
        "carrier_sort": "(public total_caps, construction_digest tie-break used only for deterministic holdout sampling)",
        "selected_sorted_carrier_indices": selected,
        "operator_basis": [list(x) for x in ops],
        "depth2_cases": depth2,
        "deep_chain_cases": deep,
        "rebracketing_cases": rebracket,
        "freshness_scope": "EVERY_TESTED_RECURSIVE_STATE_HAS_AT_LEAST_THREE_G1_LEAVES_AND_IS_OUTSIDE_THE_S5_PAIR_CENSUS",
        "outcome_dependent_selection": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _state_key(state: Any) -> str:
    d = getattr(state, "construction_digest", None)
    if isinstance(d, str) and len(d) == 64:
        return d
    return f"obj:{id(state)}"


def _dedupe(states: Sequence[Any]) -> tuple[Any, ...]:
    d: dict[str, Any] = {}
    for s in states:
        d.setdefault(_state_key(s), s)
    return tuple(d[k] for k in sorted(d))


def _assert_all_caps(states: Sequence[Any], expected: Sequence[int], label: str) -> None:
    e = _caps(expected, label + " expected")
    if not states:
        raise UpliftS6Error(label + ": empty exact branch relation")
    bad = [tuple(int(x) for x in s.total_caps) for s in states if _caps(s.total_caps, label + " state") != e]
    if bad:
        raise UpliftS6Error(f"{label}: exact branch CAPS7 mismatch; first={bad[0]} expected={e}")


def _check_all_reservations(states: Sequence[Any]) -> dict[str, int]:
    state_checks = 0
    action_checks = 0
    exact_successor_branches = 0
    for s in states:
        caps = _caps(s.total_caps, "holdout state")
        state_checks += 1
        for t in range(7):
            expected = reserve_write(caps, t)
            rel = reserve_external_relation(s, t)
            action_checks += 1
            if expected is None:
                if rel:
                    raise UpliftS6Error("disabled reservation returned exact successors")
                continue
            if not rel:
                raise UpliftS6Error("enabled reservation returned no exact successors")
            exact_successor_branches += len(rel)
            _assert_all_caps(rel, expected, f"reserve type {t}")
    return {"states_checked": state_checks, "reservation_actions_checked": action_checks, "exact_reservation_successor_branches_checked": exact_successor_branches}


def execute_depth2_holdout_case(*, engine: Any, sorted_carriers: Sequence[Any], bridge_pairs: Sequence[Sequence[int]], case: Mapping[str, Any]) -> dict[str, Any]:
    from . import regime_scanner as rs
    ops = sorted({(int(a), int(b)) for a, b in bridge_pairs})
    i0, i1, i2 = map(int, case["carrier_slots"])
    L, R, C = sorted_carriers[i0], sorted_carriers[i1], sorted_carriers[i2]
    bop = ops[int(case["base_operator_index"])]
    rop = ops[int(case["recursive_operator_index"])]
    pair = rs._build_lift(engine, 101, [L, R], [(0, 1)], bridge_pairs, "G2_S6_HOLDOUT", str(case["case_id"]) + ":PAIR", force_pair=bop)
    if pair is None:
        raise UpliftS6Error(f"{case['case_id']}: base pair build failed")
    expected = binary_write(pair.total_caps, C.total_caps, rop[0], rop[1], ops)
    if expected is None:
        raise UpliftS6Error(f"{case['case_id']}: abstract recursive operator unexpectedly disabled")
    outs = compose_binary_relation(engine, 102, pair, C, rop[0], rop[1], lane="G2_S6_HOLDOUT", motif_id=str(case["case_id"]) + ":REC")
    _assert_all_caps(outs, expected, str(case["case_id"]))
    rr = _check_all_reservations(outs)
    return {
        "case_id": case["case_id"],
        "status": "PASS",
        "leaf_count": 3,
        "base_operator": list(bop),
        "recursive_operator": list(rop),
        "expected_caps7": list(expected),
        "exact_output_branch_count": len(outs),
        **rr,
    }


def execute_deep_chain_holdout_case(*, engine: Any, sorted_carriers: Sequence[Any], bridge_pairs: Sequence[Sequence[int]], case: Mapping[str, Any]) -> dict[str, Any]:
    from . import regime_scanner as rs
    ops = sorted({(int(a), int(b)) for a, b in bridge_pairs})
    slots = [int(x) for x in case["carrier_slots"]]
    leaves = [sorted_carriers[x] for x in slots]
    opseq = [ops[int(x)] for x in case["operator_indices"]]
    pair = rs._build_lift(engine, 101, [leaves[0], leaves[1]], [(0, 1)], bridge_pairs, "G2_S6_HOLDOUT", str(case["case_id"]) + ":PAIR", force_pair=opseq[0])
    if pair is None:
        raise UpliftS6Error(f"{case['case_id']}: deep base pair build failed")
    frontier: tuple[Any, ...] = (pair,)
    expected = _caps(pair.total_caps)
    frontier_sizes = [1]
    branch_checks = 0
    for step in range(1, 4):
        op = opseq[step]
        right = leaves[step + 1]
        exp = binary_write(expected, right.total_caps, op[0], op[1], ops)
        if exp is None:
            raise UpliftS6Error(f"{case['case_id']}: deep abstract operator disabled at step {step}")
        nxt = []
        for st in frontier:
            nxt.extend(compose_binary_relation(engine, 101 + step, st, right, op[0], op[1], lane="G2_S6_HOLDOUT", motif_id=f"{case['case_id']}:STEP:{step}"))
        frontier = _dedupe(nxt)
        _assert_all_caps(frontier, exp, f"{case['case_id']} step {step}")
        branch_checks += len(frontier)
        frontier_sizes.append(len(frontier))
        expected = exp
    rr = _check_all_reservations(frontier)
    return {
        "case_id": case["case_id"],
        "status": "PASS",
        "leaf_count": 5,
        "operator_sequence": [list(x) for x in opseq],
        "expected_caps7": list(expected),
        "frontier_sizes": frontier_sizes,
        "exact_composition_branches_checked": branch_checks,
        **rr,
    }


def execute_rebracket_holdout_case(*, engine: Any, sorted_carriers: Sequence[Any], bridge_pairs: Sequence[Sequence[int]], case: Mapping[str, Any]) -> dict[str, Any]:
    from . import regime_scanner as rs
    ops = sorted({(int(a), int(b)) for a, b in bridge_pairs})
    slots = [int(x) for x in case["carrier_slots"]]
    A, B, C, D = [sorted_carriers[x] for x in slots]
    op0, op1, op2 = [ops[int(x)] for x in case["operator_indices"]]

    # Left route: ((A op0 B) op1 C) op2 D.
    ab = rs._build_lift(engine, 101, [A, B], [(0, 1)], bridge_pairs, "G2_S6_HOLDOUT", str(case["case_id"]) + ":AB", force_pair=op0)
    if ab is None:
        raise UpliftS6Error(f"{case['case_id']}: AB build failed")
    l1 = _dedupe(compose_binary_relation(engine, 102, ab, C, op1[0], op1[1], lane="G2_S6_HOLDOUT", motif_id=str(case["case_id"]) + ":L1"))
    left = []
    for st in l1:
        left.extend(compose_binary_relation(engine, 103, st, D, op2[0], op2[1], lane="G2_S6_HOLDOUT", motif_id=str(case["case_id"]) + ":L2"))
    left = _dedupe(left)

    # Right route: (A op0 (B op1 C)) op2 D.  This intentionally changes raw topology;
    # only CAPS7 observer-level rebracketing is tested.
    bc = rs._build_lift(engine, 101, [B, C], [(0, 1)], bridge_pairs, "G2_S6_HOLDOUT", str(case["case_id"]) + ":BC", force_pair=op1)
    if bc is None:
        raise UpliftS6Error(f"{case['case_id']}: BC build failed")
    r1 = _dedupe(compose_binary_relation(engine, 102, A, bc, op0[0], op0[1], lane="G2_S6_HOLDOUT", motif_id=str(case["case_id"]) + ":R1"))
    right = []
    for st in r1:
        right.extend(compose_binary_relation(engine, 103, st, D, op2[0], op2[1], lane="G2_S6_HOLDOUT", motif_id=str(case["case_id"]) + ":R2"))
    right = _dedupe(right)

    total = [int(A.total_caps[i]) + int(B.total_caps[i]) + int(C.total_caps[i]) + int(D.total_caps[i]) for i in range(7)]
    for op in (op0, op1, op2):
        total[op[0]] -= 1
        total[op[1]] -= 1
    expected = tuple(total)
    if any(x < 0 for x in expected):
        raise UpliftS6Error(f"{case['case_id']}: rebracket abstract underflow")
    _assert_all_caps(left, expected, str(case["case_id"]) + " left")
    _assert_all_caps(right, expected, str(case["case_id"]) + " right")
    lset = {tuple(int(x) for x in s.total_caps) for s in left}
    rset = {tuple(int(x) for x in s.total_caps) for s in right}
    if lset != rset or lset != {expected}:
        raise UpliftS6Error(f"{case['case_id']}: observer-level rebracketing mismatch")
    rr_left = _check_all_reservations(left)
    rr_right = _check_all_reservations(right)
    return {
        "case_id": case["case_id"],
        "status": "PASS",
        "leaf_count": 4,
        "operator_sequence": [list(op0), list(op1), list(op2)],
        "expected_caps7": list(expected),
        "left_exact_branch_count": len(left),
        "right_exact_branch_count": len(right),
        "caps7_quotient_equal": True,
        "left_reservation_checks": rr_left,
        "right_reservation_checks": rr_right,
    }


def aggregate_fresh_holdout(*, plan: Mapping[str, Any], depth2_results: Sequence[Mapping[str, Any]], deep_results: Sequence[Mapping[str, Any]], rebracket_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    if len(depth2_results) != 31 or any(x.get("status") != "PASS" for x in depth2_results):
        failures.append("DEPTH2")
    if len(deep_results) != 7 or any(x.get("status") != "PASS" for x in deep_results):
        failures.append("DEEP_CHAIN")
    if len(rebracket_results) != 7 or any(x.get("status") != "PASS" or x.get("caps7_quotient_equal") is not True for x in rebracket_results):
        failures.append("REBRACKET")
    observed_recursive_ops = {tuple(x["recursive_operator"]) for x in depth2_results}
    expected_ops = {tuple(x) for x in plan["operator_basis"]}
    if observed_recursive_ops != expected_ops:
        failures.append("OPERATOR_COVERAGE")
    out = {
        "schema_id": "IG_G2_S6_FRESH_RECURSIVE_HOLDOUT_RESULT_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "plan_science_sha256": plan["science_sha256"],
        "freshness_scope": plan["freshness_scope"],
        "operator_depth2_cases": len(depth2_results),
        "operator_coverage": len(observed_recursive_ops),
        "depth2_exact_output_branches_checked": sum(int(x["exact_output_branch_count"]) for x in depth2_results),
        "depth2_reservation_actions_checked": sum(int(x["reservation_actions_checked"]) for x in depth2_results),
        "depth2_exact_reservation_successor_branches_checked": sum(int(x["exact_reservation_successor_branches_checked"]) for x in depth2_results),
        "deep_chain_cases": len(deep_results),
        "deep_chain_max_leaf_count": max((int(x["leaf_count"]) for x in deep_results), default=0),
        "deep_chain_max_frontier_size": max((max(x["frontier_sizes"]) for x in deep_results), default=0),
        "deep_chain_exact_composition_branches_checked": sum(int(x["exact_composition_branches_checked"]) for x in deep_results),
        "deep_chain_reservation_actions_checked": sum(int(x["reservation_actions_checked"]) for x in deep_results),
        "deep_chain_exact_reservation_successor_branches_checked": sum(int(x["exact_reservation_successor_branches_checked"]) for x in deep_results),
        "rebracketing_cases": len(rebracket_results),
        "rebracketing_caps7_equal_count": sum(1 for x in rebracket_results if x.get("caps7_quotient_equal") is True),
        "depth2_cases_sha256": canonical_sha256(list(depth2_results)),
        "deep_chain_cases_sha256": canonical_sha256(list(deep_results)),
        "rebracket_cases_sha256": canonical_sha256(list(rebracket_results)),
        "hidden_internal_reads_in_promotable_observer": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def implementation_no_hidden_selector_audit() -> dict[str, Any]:
    from . import g2_relation
    from . import uplift_s5

    reserve_src = inspect.getsource(g2_relation._reserve_external_relation_with_witness)
    compose_src = inspect.getsource(g2_relation.compose_binary_relation)
    dedupe_src = inspect.getsource(g2_relation._exact_successor_key)
    abs_reserve_src = inspect.getsource(uplift_s5.reserve_write) + inspect.getsource(uplift_s5.reserve_enabled)
    abs_compose_src = inspect.getsource(uplift_s5.binary_write) + inspect.getsource(uplift_s5.bridge_enabled)
    failures: list[str] = []
    if ".skin" in reserve_src or ".skin" in compose_src:
        failures.append("SKIN_READ_IN_EXACT_TRANSITION_CORE")
    if "child.total_caps" not in reserve_src or "state.total_caps" not in reserve_src:
        failures.append("RESERVATION_CAPS_READ_MISSING")
    if "_exact_successor_key" not in reserve_src or "_exact_successor_key" not in compose_src:
        failures.append("SET_DEDUPE_PATH_MISSING")
    for tok in ("skin", "construction_digest", "children", "owner", "witness"):
        if tok in abs_reserve_src or tok in abs_compose_src:
            failures.append("ABSTRACT_HIDDEN_READ_" + tok.upper())
    if "construction_digest" not in dedupe_src:
        failures.append("DEDUPE_IDENTITY_AUDIT_UNRESOLVED")
    out = {
        "schema_id": "IG_G2_S6_NO_HIDDEN_SELECTOR_READ_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "exact_reservation_selection_fields": ["state.total_caps", "child.total_caps", "complete child enumeration"],
        "exact_binary_selection_fields": ["left/right total_caps", "complete reservation relations"],
        "construction_identity_role": "EXACT_SET_DEDUPLICATION_AFTER_BRANCH_GENERATION_ONLY",
        "abstract_read_write_fields": ["CAPS7", "endpoint types", "frozen 31-operator basis"],
        "skin_used_as_selector": False,
        "owner_witness_used_as_selector": False,
        "source_snippet_hashes": {
            "reserve_relation": hashlib.sha256(reserve_src.encode()).hexdigest(),
            "compose_relation": hashlib.sha256(compose_src.encode()).hexdigest(),
            "exact_dedupe": hashlib.sha256(dedupe_src.encode()).hexdigest(),
            "abstract_reserve": hashlib.sha256(abs_reserve_src.encode()).hexdigest(),
            "abstract_compose": hashlib.sha256(abs_compose_src.encode()).hexdigest(),
        },
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def stable_s6_science_payload(*, authority: Mapping[str, Any], regression: Mapping[str, Any], theorem: Mapping[str, Any], holdout: Mapping[str, Any], hidden_read_audit: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "schema_id": "IG_G2_S6_STABLE_SCIENCE_PAYLOAD_V1",
        "s6_spec_science_sha256": s6_implementation_spec()["science_sha256"],
        "authority_science_sha256": authority["science_sha256"],
        "regression_gate_science_sha256": regression["science_sha256"],
        "recursive_theorem_science_sha256": theorem["science_sha256"],
        "fresh_holdout_science_sha256": holdout["science_sha256"],
        "no_hidden_read_audit_science_sha256": hidden_read_audit["science_sha256"],
        "observer": s6_implementation_spec()["observer"]["name"],
        "relation_spec_sha256": relation_spec()["spec_sha256"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def s6_primary_result(*, authority: Mapping[str, Any], regression: Mapping[str, Any], theorem: Mapping[str, Any], holdout: Mapping[str, Any], hidden_read_audit: Mapping[str, Any]) -> dict[str, Any]:
    passed = all(x.get("status") == "PASS" for x in (authority, regression, theorem, holdout, hidden_read_audit))
    stable = stable_s6_science_payload(authority=authority, regression=regression, theorem=theorem, holdout=holdout, hidden_read_audit=hidden_read_audit)
    out = {
        "schema_id": "IG_G2_S6_PRIMARY_RECURSIVE_CLOSURE_RESULT_V1",
        "schema_version": "1.0.0",
        "date": "2026-09-03",
        "stage_ref": "G2:S6",
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "classification": "CAPS7_RECURSIVE_CLOSURE_EARNED_GRADUATION_CANDIDATE_AWAITING_COLD_REPLAY" if passed else "S6_RECURSIVE_CLOSURE_GATE_FAILED_G2_NOT_GRADUATED",
        "authority": dict(authority),
        "regression_gate": dict(regression),
        "recursive_caps7_factorisation_theorem": dict(theorem),
        "fresh_recursive_holdout": dict(holdout),
        "no_hidden_selector_read_audit": dict(hidden_read_audit),
        "stable_science_payload": stable,
        "stable_science_payload_sha256": stable["science_sha256"],
        "cold_replay_required_for_graduation": True,
        "g2_graduation_candidate": passed,
        "g2_graduated": False,
        "r0_unlocked": False,
        "graduation_scope_if_cold_replay_passes": "FROZEN_CAPS7_TRANSITION_OBSERVER_WITH_7_RESERVATION_ACTIONS_AND_31_RELATION_VALUED_TYPED_BINARY_OPERATORS_OVER_ALL_FINITE_RECURSIVE_G2_TERMS",
        "nonclaims": [
            "NOT_CAPS7_MINIMALITY",
            "NOT_RAW_PUBLIC_SKIN_EQUIVALENCE",
            "NOT_EXACT_BRANCH_MULTIPLICITY_EQUIVALENCE",
            "NOT_RAW_TOPOLOGY_ASSOCIATIVITY",
            "NOT_NEW_LOWER_G_OR_G1_SEMANTICS",
        ],
    }
    # Stable primary science identity excludes packaging/runtime fields added by the campaign handler.
    science_core = {k: v for k, v in out.items() if k not in {"source_sha256", "source_version", "registry_sha256", "execution_metadata", "science_sha256"}}
    out["science_sha256"] = canonical_sha256(science_core)
    return out



def decoder_native_s6_replay_evidence(*, primary_result: Mapping[str, Any], replay_result: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two executions of the same registered Decoder-native G2:S6 experiment.

    The replay is independent in execution and cold-source state, not a second science
    implementation. This preserves the preregistered S6 question while satisfying the
    campaign rule that scientific experiments execute through Decoder registration.
    """
    failures: list[str] = []
    for label, result in (("PRIMARY", primary_result), ("REPLAY", replay_result)):
        if result.get("schema_id") != "IG_G2_S6_PRIMARY_RECURSIVE_CLOSURE_RESULT_V1":
            failures.append(label + "_SCHEMA")
        if result.get("status") != "PASS" or result.get("g2_graduation_candidate") is not True:
            failures.append(label + "_STATUS")
        md = result.get("execution_metadata", {})
        if md.get("registered_experiment_id") != "G2:S6":
            failures.append(label + "_NOT_REGISTERED_G2_S6")
        if md.get("execution_backend_owned_by_decoder") is not True:
            failures.append(label + "_BACKEND_NOT_DECODER")
        if md.get("stage_specific_external_science_runner") is not False:
            failures.append(label + "_EXTERNAL_SCIENCE_RUNNER")
        stable = result.get("stable_science_payload", {})
        if stable.get("science_sha256") != result.get("stable_science_payload_sha256"):
            failures.append(label + "_STABLE_PAYLOAD_SELF_CHECK")
        stable_no_hash = {k: v for k, v in stable.items() if k != "science_sha256"}
        if canonical_sha256(stable_no_hash) != stable.get("science_sha256"):
            failures.append(label + "_STABLE_PAYLOAD_HASH")
    psha = primary_result.get("stable_science_payload_sha256")
    rsha = replay_result.get("stable_science_payload_sha256")
    exact = psha == rsha and primary_result.get("stable_science_payload") == replay_result.get("stable_science_payload")
    if not exact:
        failures.append("SCIENCE_PAYLOAD_MISMATCH")
    out = {
        "schema_id": "IG_G2_S6_DECODER_NATIVE_COLD_REPLAY_V1",
        "date": "2026-09-03",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "registered_experiment_id": "G2:S6",
        "replay_mode": "COLD_SOURCE_RERUN_OF_SAME_REGISTERED_DECODER_NATIVE_EXPERIMENT",
        "primary_stable_science_payload_sha256": psha,
        "cold_stable_science_payload_sha256": rsha,
        "stable_science_payload_exact_match": exact,
        "primary_science_sha256": primary_result.get("science_sha256"),
        "cold_science_sha256": replay_result.get("science_sha256"),
        "execution_backend_owned_by_decoder": True,
        "stage_specific_external_science_runner": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out

def finalize_s6_graduation(*, primary_result: Mapping[str, Any], cold_replay: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if primary_result.get("status") != "PASS" or primary_result.get("g2_graduation_candidate") is not True:
        failures.append("PRIMARY_S6")
    if cold_replay.get("schema_id") != "IG_G2_S6_DECODER_NATIVE_COLD_REPLAY_V1" or cold_replay.get("status") != "PASS":
        failures.append("COLD_REPLAY")
    if cold_replay.get("stable_science_payload_exact_match") is not True:
        failures.append("SCIENCE_PAYLOAD_MISMATCH")
    if _sha(cold_replay.get("primary_stable_science_payload_sha256"), "primary stable payload") != _sha(primary_result.get("stable_science_payload_sha256"), "primary result stable payload"):
        failures.append("PRIMARY_PAYLOAD_IDENTITY")
    if _sha(cold_replay.get("cold_stable_science_payload_sha256"), "cold stable payload") != _sha(primary_result.get("stable_science_payload_sha256"), "primary result stable payload"):
        failures.append("COLD_PAYLOAD_IDENTITY")
    passed = not failures
    out = {
        "schema_id": "IG_G2_GRADUATION_CERTIFICATE_V1",
        "date": "2026-09-03",
        "status": "PASS" if passed else "FAIL",
        "classification": "G2_GRADUATED_CAPS7_RECURSIVE_RELATION_GRAMMAR_EARNED_R0_UNLOCKED" if passed else "G2_NOT_GRADUATED_R0_LOCKED",
        "failures": failures,
        "g2_graduated": passed,
        "r0_unlocked": passed,
        "graduated_observer": s6_implementation_spec()["observer"]["name"],
        "graduated_descriptor": "IG_G2_CAPS7_STATE_V1",
        "graduated_grammar": {
            "reservation_actions": 7,
            "binary_operator_count": 31,
            "routing_semantics": "RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1",
            "recursive_scope": "ALL_FINITE_G2_CONSTRUCTION_AND_ACTION_TERMS",
        },
        "primary_s6_science_sha256": primary_result.get("science_sha256"),
        "stable_s6_science_payload_sha256": primary_result.get("stable_science_payload_sha256"),
        "cold_replay_science_sha256": cold_replay.get("science_sha256"),
        "cold_replay_mode": cold_replay.get("replay_mode"),
        "authorizes": "G2:R0_POST_GRADUATION_RECURSIVE_DEPTH_AXIS" if passed else None,
        "nonclaims": [
            "CAPS7_MINIMALITY_NOT_CLAIMED",
            "RAW_SKIN_EQUIVALENCE_NOT_CLAIMED",
            "EXACT_BRANCH_MULTIPLICITY_EQUIVALENCE_NOT_CLAIMED",
            "RAW_TOPOLOGY_ASSOCIATIVITY_NOT_CLAIMED",
            "NO_GEOMETRY_OR_PHYSICS_CLAIM",
        ],
        "reopen_conditions": [
            "CAPS7 observer changes",
            "relation-valued G2 routing changes",
            "frozen lower-G/G1 reservation semantics change",
            "31 bridge-operator basis changes",
        ],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out
