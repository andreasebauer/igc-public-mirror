from __future__ import annotations

"""Registered G3:S4 one-step composition-closure audit.

S4 closes the implementation gap intentionally left by S3.  S3 proved that the declared
P3/K3 public observer factorises through the certified S2 edge observer plus inherited CAPS7
and complete ordered two-reservation local continuations.  S4 now materialises complete
relation-valued pair candidates and reuses them as atomic inputs for one further binary
whole-unit connection.

The reduction is exact for the frozen S0 collision domain, but deliberately finite.  It does not
claim pair+pair closure, arbitrary depth, a final G3 descriptor, or G3 graduation.
"""

from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g2_relation import compose_binary_relation, reserve_external_relation
from .uplift_g3_s0 import phase0_spec


class G3S4Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s4_spec() -> dict[str, Any]:
    obj = json.loads(_resource("G3_S4_COMPOSITION_CLOSURE_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G3_S4_COMPOSITION_CLOSURE_SPEC_V1":
        raise G3S4Error("bad G3:S4 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G3S4Error("G3:S4 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G3S4Error("G3:S4/Phase0 binding mismatch")
    return obj


def _caps7(st: Any) -> list[int]:
    caps = [int(x) for x in st.total_caps]
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise G3S4Error("malformed CAPS7 in G3:S4")
    return caps


def _exact_key(st: Any) -> str:
    d = getattr(st, "construction_digest", None)
    if not isinstance(d, str) or len(d) != 64:
        raise G3S4Error("G3:S4 exact state lacks canonical construction digest")
    return d


def verify_s4_authority(*, s0_result: Mapping[str, Any], s1_result: Mapping[str, Any], s2_result: Mapping[str, Any], s3_result: Mapping[str, Any], s3_replay_comparison: Mapping[str, Any]) -> dict[str, Any]:
    a = s4_spec()["authority"]
    checks = [
        (s0_result.get("science_sha256") == a["g3_s0_science_sha256"], "S0_SHA"),
        (s1_result.get("science_sha256") == a["g3_s1_science_sha256"], "S1_SHA"),
        (s2_result.get("science_sha256") == a["g3_s2_science_sha256"], "S2_SHA"),
        (s3_result.get("status") == a["g3_s3_status"], "S3_STATUS"),
        (s3_result.get("classification") == a["g3_s3_classification"], "S3_CLASS"),
        (s3_result.get("science_sha256") == a["g3_s3_science_sha256"], "S3_SHA"),
        (bool(s3_result.get("g3_s4_unlocked")) is bool(a["required_g3_s4_unlocked"]), "S4_UNLOCK"),
        (bool(s3_result.get("g3_graduated")) is bool(a["required_g3_graduated"]), "G3_GRADUATION"),
        (bool(s3_result.get("topology_promoted")) is bool(a["required_topology_promoted"]), "TOPOLOGY_PROMOTION"),
        (int(s3_result.get("triple_reconstruction", {}).get("public_context_counts", {}).get("P3_CONNECTED_PATH", -1)) == int(a["required_s3_p3_public_context_count"]), "S3_P3_PUBLIC_COUNT"),
        (int(s3_result.get("triple_reconstruction", {}).get("exact_representative_assignment_counts", {}).get("P3_CONNECTED_PATH", -1)) == int(a["required_s3_p3_exact_assignment_count"]), "S3_P3_EXACT_COUNT"),
        (int(s3_result.get("caps7_two_reservation_sufficiency", {}).get("continuation_conflict_count", -1)) == int(a["required_s3_local_two_reservation_conflicts"]), "S3_LOCAL2_CONFLICTS"),
        (s3_replay_comparison.get("schema_id") == a["g3_s3_replay_schema_id"], "S3_REPLAY_SCHEMA"),
        (s3_replay_comparison.get("status") == a["required_g3_s3_cold_replay_status"], "S3_REPLAY_STATUS"),
        (bool(s3_replay_comparison.get("science_sha256_equal")) is bool(a["required_g3_s3_science_sha256_equal"]), "S3_REPLAY_SCIENCE_EQUAL"),
        (bool(s3_replay_comparison.get("source_sha256_equal")) is bool(a["required_g3_s3_source_sha256_equal"]), "S3_REPLAY_SOURCE_EQUAL"),
        (bool(s3_replay_comparison.get("stable_scientific_payload_exact_equal")) is bool(a["required_g3_s3_stable_payload_exact_equal"]), "S3_REPLAY_PAYLOAD_EXACT"),
        (s3_replay_comparison.get("stable_scientific_payload_sha256") == a["g3_s3_stable_payload_sha256"], "S3_REPLAY_STABLE_SHA"),
    ]
    failures = [name for ok, name in checks if not ok]
    if failures:
        raise G3S4Error("G3:S4 authority verification failed: " + ",".join(failures))
    out = {
        "schema_id": "IG_G3_S4_AUTHORITY_V1",
        "status": "PASS",
        "g3_s0_science_sha256": s0_result["science_sha256"],
        "g3_s1_science_sha256": s1_result["science_sha256"],
        "g3_s2_science_sha256": s2_result["science_sha256"],
        "g3_s3_science_sha256": s3_result["science_sha256"],
        "g3_s3_replay_stable_payload_sha256": s3_replay_comparison["stable_scientific_payload_sha256"],
        "g3_s3_cold_replay_exact": True,
        "g3_phase0_science_sha256": phase0_spec()["science_sha256"],
        "g3_s4_spec_sha256": s4_spec()["science_sha256"],
        "g3_graduated": False,
        "topology_promoted": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def ordered_three_reservation_signature(state: Any, first_type: int, second_type: int, third_type: int) -> dict[str, Any]:
    """Complete branch-sensitive CAPS7-only signature for three ordered reservations."""
    a, b, c = int(first_type), int(second_type), int(third_type)
    if not all(0 <= x < 7 for x in (a, b, c)):
        raise G3S4Error("endpoint type outside frozen seven-type alphabet")

    firsts = reserve_external_relation(state, a)
    first_hist: Counter[str] = Counter()
    first_payload: dict[str, dict[str, Any]] = {}
    total_seconds = 0
    total_thirds = 0

    for first in firsts:
        seconds = reserve_external_relation(first, b)
        total_seconds += len(seconds)
        second_hist: Counter[str] = Counter()
        second_payload: dict[str, dict[str, Any]] = {}
        for second in seconds:
            thirds = reserve_external_relation(second, c)
            total_thirds += len(thirds)
            sp = {
                "second_caps7": _caps7(second),
                "third_relation_cardinality": len(thirds),
                "third_successor_caps7_set": [list(x) for x in sorted({tuple(_caps7(s)) for s in thirds})],
            }
            sh = canonical_sha256(sp)
            second_payload.setdefault(sh, sp)
            second_hist[sh] += 1
        fp = {
            "first_caps7": _caps7(first),
            "second_relation_cardinality": len(seconds),
            "second_successor_caps7_set": [list(x) for x in sorted({tuple(_caps7(s)) for s in seconds})],
            "second_branch_profile_histogram": [
                {"branch_profile_sha256": h, "multiplicity": int(second_hist[h]), "profile": second_payload[h]}
                for h in sorted(second_hist)
            ],
        }
        fh = canonical_sha256(fp)
        first_payload.setdefault(fh, fp)
        first_hist[fh] += 1

    out = {
        "schema_id": "IG_G3_S4_ORDERED_THREE_RESERVATION_SIGNATURE_V1",
        "first_reserved_type": a,
        "second_reserved_type": b,
        "third_reserved_type": c,
        "first_step": {
            "relation_cardinality": len(firsts),
            "successor_caps7_set": [list(x) for x in sorted({tuple(_caps7(s)) for s in firsts})],
        },
        "first_branch_profile_histogram": [
            {"branch_profile_sha256": h, "multiplicity": int(first_hist[h]), "profile": first_payload[h]}
            for h in sorted(first_hist)
        ],
        "second_relation_output_cardinality": int(total_seconds),
        "final_relation_output_cardinality": int(total_thirds),
        "joint_legal": bool(total_thirds > 0),
        "direct_topology_read": False,
        "construction_identity_recorded": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def certify_caps7_three_reservation_sufficiency(*, s0_result: Mapping[str, Any], task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    challenge = list(s0_result.get("challenge_corpus", {}).get("records", []))
    if len(challenge) != 6:
        raise G3S4Error("G3:S4 expects six exact S0 challenge carriers")
    caps_by_ref = {str(r["g2_carrier_ref"]): str(r["public_interface"]["science_sha256"]) for r in challenge}
    topo_by_ref = {str(r["g2_carrier_ref"]): str(r["hidden_challenge_diagnostic"]["topology_canon"]) for r in challenge}
    expected = len(caps_by_ref) * 343
    if len(task_rows) != expected:
        raise G3S4Error(f"G3:S4 expected {expected} local-three rows, got {len(task_rows)}")

    seen = set()
    grouped: dict[tuple[str, int, int, int], list[tuple[str, str]]] = defaultdict(list)
    vectors: dict[str, dict[tuple[int, int, int], str]] = defaultdict(dict)
    for row in task_rows:
        ref = str(row["carrier_ref"])
        a, b, c = int(row["first_type"]), int(row["second_type"]), int(row["third_type"])
        if ref not in caps_by_ref or not all(0 <= x < 7 for x in (a, b, c)):
            raise G3S4Error("bad G3:S4 local-three row")
        key = (ref, a, b, c)
        if key in seen:
            raise G3S4Error("duplicate G3:S4 local-three row")
        seen.add(key)
        sig = row["signature"]
        sh = str(sig.get("science_sha256"))
        if canonical_sha256({k: v for k, v in sig.items() if k != "science_sha256"}) != sh:
            raise G3S4Error("G3:S4 local-three signature hash mismatch")
        grouped[(caps_by_ref[ref], a, b, c)].append((ref, sh))
        vectors[ref][(a, b, c)] = sh

    conflicts = []
    for (caps, a, b, c), vals in sorted(grouped.items(), key=str):
        hashes = sorted({h for _, h in vals})
        if len(vals) != len(caps_by_ref) or len(hashes) != 1:
            conflicts.append({
                "caps7_sha256": caps,
                "first_type": a,
                "second_type": b,
                "third_type": c,
                "representative_count": len(vals),
                "signature_count": len(hashes),
                "signature_sha256s": hashes,
            })

    representative_vectors = []
    for ref in sorted(caps_by_ref):
        seq = [vectors[ref][(a, b, c)] for a in range(7) for b in range(7) for c in range(7)]
        representative_vectors.append({
            "carrier_ref": ref,
            "topology_label": topo_by_ref[ref],
            "continuation_vector_sha256": canonical_sha256(seq),
        })
    full_hashes = sorted({r["continuation_vector_sha256"] for r in representative_vectors})
    if len(full_hashes) != 1:
        conflicts.append({"failure": "FULL_THREE_RESERVATION_VECTOR_CONFLICT", "vector_hashes": full_hashes})

    out = {
        "schema_id": "IG_G3_S4_CAPS7_THREE_RESERVATION_SUFFICIENCY_V1",
        "status": "PASS" if not conflicts else "FAIL",
        "exact_carrier_count": len(caps_by_ref),
        "caps7_class_count": len(set(caps_by_ref.values())),
        "hidden_topology_class_count": len(set(topo_by_ref.values())),
        "ordered_endpoint_triple_count": 343,
        "task_row_count": len(task_rows),
        "continuation_conflict_count": len(conflicts),
        "conflict_examples": conflicts[:16],
        "all_equal_caps7_representatives_share_one_complete_ordered_three_reservation_table": not conflicts,
        "representative_vectors": representative_vectors,
        "topology_used_only_after_public_sufficiency_test": True,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def compose_complete_pair_relation(*, engine: Any, target: Any, context: Any, operator: Sequence[int], orientation: str, motif_id: str) -> tuple[Any, ...]:
    if orientation not in {"TARGET_LEFT_CONTEXT_RIGHT", "CONTEXT_LEFT_TARGET_RIGHT"}:
        raise G3S4Error("bad G3:S4 pair orientation")
    a, b = map(int, operator)
    left, right = (target, context) if orientation == "TARGET_LEFT_CONTEXT_RIGHT" else (context, target)
    level = max(int(left.level), int(right.level)) + 1
    return compose_binary_relation(engine, level, left, right, a, b, lane="G3_S4_PAIR_CANDIDATE", motif_id=motif_id)


def recursive_relation_immediate_signature(*, engine: Any, pair_relation: Sequence[Any], context: Any, operator: Sequence[int], orientation: str, motif_id: str) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Materialise one complete pair relation + one base unit recursive P3 step."""
    if orientation not in {"TARGET_LEFT_CONTEXT_RIGHT", "CONTEXT_LEFT_TARGET_RIGHT"}:
        raise G3S4Error("bad G3:S4 recursive orientation")
    if not pair_relation:
        raise G3S4Error("empty G3:S4 pair candidate relation")
    a, b = map(int, operator)
    pair_caps_set = {tuple(_caps7(st)) for st in pair_relation}
    if len(pair_caps_set) != 1:
        raise G3S4Error("pair candidate relation has multiple CAPS7 states before recursive step")
    pair_caps = list(next(iter(pair_caps_set)))
    context_caps = _caps7(context)
    expected = [pair_caps[i] + context_caps[i] - (1 if i == a else 0) - (1 if i == b else 0) for i in range(7)]
    if any(x < 0 for x in expected):
        expected_nonnegative = False
    else:
        expected_nonnegative = True

    out_map: dict[str, Any] = {}
    branch_counts = []
    for pair_state in pair_relation:
        left, right = (pair_state, context) if orientation == "TARGET_LEFT_CONTEXT_RIGHT" else (context, pair_state)
        level = max(int(left.level), int(right.level)) + 1
        outs = compose_binary_relation(engine, level, left, right, a, b, lane="G3_S4_RECURSIVE_P3", motif_id=motif_id)
        branch_counts.append(len(outs))
        for st in outs:
            out_map.setdefault(_exact_key(st), st)
    outputs = tuple(out_map[k] for k in sorted(out_map))
    caps_set = sorted({tuple(_caps7(st)) for st in outputs})
    mismatch_count = sum(1 for st in outputs if _caps7(st) != expected)
    sig = {
        "schema_id": "IG_G3_S4_RECURSIVE_RELATION_IMMEDIATE_SIGNATURE_V1",
        "operator": [a, b],
        "orientation": orientation,
        "pair_relation_cardinality": len(pair_relation),
        "per_pair_branch_relation_cardinality_histogram": [
            {"relation_cardinality": int(k), "pair_branch_count": int(v)}
            for k, v in sorted(Counter(branch_counts).items())
        ],
        "recursive_relation_cardinality": len(outputs),
        "public_output_caps7_set": [list(x) for x in caps_set],
        "expected_output_caps7": expected,
        "expected_caps7_nonnegative": expected_nonnegative,
        "caps7_write_mismatch_count": int(mismatch_count),
        "relation_nonempty": bool(outputs),
        "complete_pair_relation_used": True,
        "exact_branch_selected": False,
        "construction_identity_recorded": False,
        "direct_topology_read": False,
    }
    sig["science_sha256"] = canonical_sha256(sig)
    return sig, outputs


def recursive_reserve_capability_signature(outputs: Sequence[Any]) -> dict[str, Any]:
    """Deep implementation check that recursive outputs still expose relation-valued CAPS7 reserves."""
    failures = []
    cardinality_hist: dict[int, Counter[int]] = {t: Counter() for t in range(7)}
    checked = 0
    for st in outputs:
        caps = _caps7(st)
        for t in range(7):
            if caps[t] <= 0:
                continue
            rel = reserve_external_relation(st, t)
            checked += 1
            cardinality_hist[t][len(rel)] += 1
            if not rel:
                failures.append({"endpoint_type": t, "failure": "EMPTY_RESERVE_RELATION"})
                continue
            expected = list(caps); expected[t] -= 1
            if any(_caps7(s) != expected for s in rel):
                failures.append({"endpoint_type": t, "failure": "CAPS7_RESERVE_WRITE_MISMATCH"})
    out = {
        "schema_id": "IG_G3_S4_RECURSIVE_RESERVE_CAPABILITY_SIGNATURE_V1",
        "status": "PASS" if not failures else "FAIL",
        "recursive_output_count": len(outputs),
        "reserve_checks": checked,
        "failures": failures[:32],
        "failure_count": len(failures),
        "reserve_relation_cardinality_histograms": [
            {
                "endpoint_type": t,
                "histogram": [
                    {"relation_cardinality": int(k), "output_count": int(v)}
                    for k, v in sorted(cardinality_hist[t].items())
                ],
            }
            for t in range(7)
        ],
        "construction_identity_recorded": False,
        "direct_topology_read": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def finalize_s4_result(*, s0_result: Mapping[str, Any], s1_result: Mapping[str, Any], s2_result: Mapping[str, Any], s3_result: Mapping[str, Any], s3_replay_comparison: Mapping[str, Any], local_three_rows: Sequence[Mapping[str, Any]], pair_basis: Mapping[str, Any], recursive_rows: Sequence[Mapping[str, Any]], reserve_basis_rows: Sequence[Mapping[str, Any]], reproduction: Mapping[str, Any]) -> dict[str, Any]:
    spec = s4_spec()
    auth = verify_s4_authority(s0_result=s0_result, s1_result=s1_result, s2_result=s2_result, s3_result=s3_result, s3_replay_comparison=s3_replay_comparison)
    local3 = certify_caps7_three_reservation_sufficiency(s0_result=s0_result, task_rows=local_three_rows)

    if pair_basis.get("status") != "PASS" or int(pair_basis.get("public_pair_context_key_count", -1)) != 62:
        raise G3S4Error("G3:S4 pair candidate basis failed before recursive closure")
    if len(recursive_rows) != 3844:
        raise G3S4Error(f"G3:S4 expected 3844 recursive rows, got {len(recursive_rows)}")
    seen = set(); recursive_failures = []; relation_card_hist = Counter(); branch_histograms = Counter()
    for row in recursive_rows:
        key = (int(row["first_key_index"]), int(row["second_key_index"]))
        if key in seen:
            raise G3S4Error("duplicate G3:S4 recursive P3 key")
        seen.add(key)
        sig = row["signature"]
        sh = str(sig.get("science_sha256"))
        if canonical_sha256({k: v for k, v in sig.items() if k != "science_sha256"}) != sh:
            raise G3S4Error("G3:S4 recursive signature hash mismatch")
        relation_card_hist[int(sig["recursive_relation_cardinality"])] += 1
        branch_histograms[canonical_sha256(sig["per_pair_branch_relation_cardinality_histogram"])] += 1
        if (not sig.get("relation_nonempty")) or int(sig.get("caps7_write_mismatch_count", -1)) != 0 or not sig.get("expected_caps7_nonnegative"):
            recursive_failures.append({"first_key_index": key[0], "second_key_index": key[1], "signature": sig})

    reserve_failures = [r for r in reserve_basis_rows if r.get("reserve_capability", {}).get("status") != "PASS"]
    operator_seen = sorted({tuple(map(int, r["operator"])) for r in reserve_basis_rows})
    expected_ops = sorted({tuple(map(int, r["operator"])) for r in s2_result.get("quotient_table", [])})
    reserve_basis_pass = len(reserve_basis_rows) == 31 and operator_seen == expected_ops and not reserve_failures

    transport_pass = (
        local3["status"] == "PASS"
        and int(s3_result.get("caps7_two_reservation_sufficiency", {}).get("continuation_conflict_count", -1)) == 0
        and int(s3_result.get("triple_reconstruction", {}).get("public_context_counts", {}).get("P3_CONNECTED_PATH", -1)) == 3844
        and not recursive_failures
    )
    coverage = {
        "schema_id": "IG_G3_S4_RECURSIVE_P3_FACTORISATION_COVERAGE_V1",
        "status": "PASS" if transport_pass else "FAIL",
        "public_recursive_p3_context_count": 3844,
        "exact_base_triples_per_public_context": 216,
        "exact_recursive_base_assignment_count": 830304,
        "post_recursive_reservation_types": 7,
        "factorized_post_recursive_public_continuation_cases": 5812128,
        "executed_local_three_reservation_tasks": len(local_three_rows),
        "executed_recursive_public_materialization_tasks": len(recursive_rows),
        "transport_argument": "Certified S2 fixes every first/second public edge observer; certified S3 fixes all ordered two-reservation local effects; S4 fixes the complete ordered three-reservation local effect on all six equal-CAPS7 base representatives. The relation-valued binary operator is childwise and retains all owner branches. Therefore the canonical complete-pair recursive materialization transports to all 216 exact base-triple assignments per public P3 context without topology, owner, construction, skin, or ancestry reads.",
        "bruteforce_exact_triples_executed": False,
    }
    coverage["science_sha256"] = canonical_sha256(coverage)

    passed = (
        auth["status"] == "PASS"
        and local3["status"] == "PASS"
        and pair_basis.get("status") == "PASS"
        and not recursive_failures
        and reserve_basis_pass
        and coverage["status"] == "PASS"
    )
    recursive_audit = {
        "schema_id": "IG_G3_S4_RECURSIVE_MATERIALISATION_AUDIT_V1",
        "status": "PASS" if not recursive_failures else "FAIL",
        "public_context_count": len(recursive_rows),
        "failure_count": len(recursive_failures),
        "failure_examples": recursive_failures[:16],
        "recursive_relation_cardinality_histogram": [
            {"relation_cardinality": int(k), "public_context_count": int(v)}
            for k, v in sorted(relation_card_hist.items())
        ],
        "per_pair_branch_relation_histogram_shape_count": len(branch_histograms),
    }
    recursive_audit["science_sha256"] = canonical_sha256(recursive_audit)
    reserve_audit = {
        "schema_id": "IG_G3_S4_RECURSIVE_OPERATOR_RESERVE_BASIS_AUDIT_V1",
        "status": "PASS" if reserve_basis_pass else "FAIL",
        "operator_count": len(expected_ops),
        "operator_rows": len(reserve_basis_rows),
        "failure_count": len(reserve_failures),
        "failure_examples": reserve_failures[:8],
    }
    reserve_audit["science_sha256"] = canonical_sha256(reserve_audit)

    result = {
        "schema_id": "IG_G3_S4_COMPOSITION_CLOSURE_RESULT_V1",
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "stage_ref": "G3:S4",
        "classification": "G3_ONE_STEP_RELATION_VALUED_COMPOSITION_CLOSURE_ON_FROZEN_P3_CAPS7_SCOPE_S5_UNLOCKED" if passed else "G3_COMPOSITION_CLOSURE_PREMISE_OR_MATERIALISATION_FAILURE_S5_LOCKED",
        "authority": auth,
        "challenge_reproduction": dict(reproduction),
        "pair_candidate_basis": dict(pair_basis),
        "caps7_three_reservation_sufficiency": local3,
        "recursive_materialisation_audit": recursive_audit,
        "recursive_operator_reserve_basis_audit": reserve_audit,
        "recursive_p3_factorisation_coverage": coverage,
        "minimal_added_read_at_tested_composition_observer": "NONE" if passed else None,
        "topology_promoted": False,
        "g3_s5_unlocked": passed,
        "g3_graduated": False,
        "next_authorized_stage": "G3:S5" if passed else None,
        "scope": "EXACT_FROZEN_G3_S0_SIX_UNIT_COLLISION_DOMAIN__COMPLETE_62x62_RECURSIVE_P3_PUBLIC_BASIS__ONE_POST_RECURSIVE_RELATION_RESERVATION_LAYER_FACTORIZED_BY_ORDERED_THREE_RESERVATION_TABLE",
        "nonclaims": list(spec["nonclaims"]),
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
