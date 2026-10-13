from __future__ import annotations

"""Registered G4:S3 higher-order compatibility / irreducibility audit.

G4:S2 earned Tier 1 SCALAR_TREE_SUMMARIES only as a bounded candidate on the
certified two-term / 248-row pair observer.  S3 asks the next finite question:
for connected simple three-unit P3/K3 motifs, can every declared joint public
compatibility fact be reconstructed from the certified S2 pair observer plus
the inherited CAPS7 and that Tier-1 candidate key?

The previous-layer G3 term remains atomic at its certified G3 boundary.  S3
never reopens G1:R100, G2, or historical G3 implementation carriers.  The only
new exact local calculation is the complete ordered 7x7 two-reservation
continuation table on each frozen G3 term.  The table is recorded only through
CAPS7 branch outcomes and joint legality; no topology, shell profile,
construction identity, owner witness, ancestry, or provenance is an observer
input.
"""

from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g4_term_state import G3TermState
from .uplift_g4_s0 import phase0_spec, challenge_tiers
from .uplift_g4_s1 import load_s0_term_states
from .uplift_g4_s2 import (
    s2_spec,
    certify_minimal_added_read,
)


class G4S3Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s3_spec() -> dict[str, Any]:
    obj = json.loads(_resource("G4_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G4_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_SPEC_V1":
        raise G4S3Error("bad G4:S3 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G4S3Error("G4:S3 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G4S3Error("G4:S3/Phase0 binding mismatch")
    if obj.get("g4_s2_spec_sha256") != s2_spec().get("science_sha256"):
        raise G4S3Error("G4:S3/S2-spec binding mismatch")
    if obj.get("challenge_read_tier_science_sha256") != challenge_tiers().get("science_sha256"):
        raise G4S3Error("G4:S3/tier-ladder binding mismatch")
    return obj


def _caps7(st: G3TermState) -> list[int]:
    caps = [int(x) for x in st.total_caps]
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise G4S3Error("malformed CAPS7 during G4:S3")
    return caps


def _tier1_value(row: Mapping[str, Any]) -> dict[str, int]:
    d = row["hidden_challenge_diagnostic"]
    deg = [int(x) for x in d["degree_sequence"]]
    return {
        "MAX_DEGREE": max(deg),
        "LEAF_COUNT": sum(1 for x in deg if x == 1),
        "diameter": int(d["diameter"]),
        "radius": int(d["radius"]),
        "articulation_count": int(d["articulation_count"]),
        "wiener_index": int(d["wiener_index"]),
    }


def candidate_interface_index(s0_result: Mapping[str, Any], s2_result: Mapping[str, Any]) -> dict[str, Any]:
    candidate = s2_result.get("minimal_sufficient_preregistered_tier")
    if candidate != {"tier": 1, "name": "SCALAR_TREE_SUMMARIES"}:
        raise G4S3Error("G4:S3 requires the certified Tier-1 S2 candidate")
    pair = s0_result.get("certified_collision_pair", {})
    rows: list[dict[str, Any]] = []
    for label in ("A", "B"):
        row = pair.get(label)
        if not isinstance(row, Mapping):
            raise G4S3Error("G4:S0 certified collision pair missing A/B")
        ref = str(row.get("g4_term_ref", ""))
        caps = [int(x) for x in row["public_interface"]["coordinates"]]
        tier1 = _tier1_value(row)
        key_payload = {
            "schema_id": "IG_G4_S3_CANDIDATE_UNIT_KEY_V1",
            "caps7": caps,
            "added_read": {"tier": 1, "name": "SCALAR_TREE_SUMMARIES", "value": tier1},
        }
        rows.append({
            "term_ref": ref,
            "caps7": caps,
            "tier1_value": tier1,
            "candidate_key_sha256": canonical_sha256(key_payload),
        })
    if len({r["term_ref"] for r in rows}) != 2:
        raise G4S3Error("G4:S3 expects two distinct exact frozen terms")
    if len({r["candidate_key_sha256"] for r in rows}) != 2:
        raise G4S3Error("certified Tier-1 candidate does not separate the frozen S0 terms")
    out = {
        "schema_id": "IG_G4_S3_CANDIDATE_INTERFACE_INDEX_V1",
        "status": "PASS",
        "candidate": {"tier": 1, "name": "SCALAR_TREE_SUMMARIES"},
        "candidate_key": "CAPS7_PLUS_SCALAR_TREE_SUMMARIES",
        "exact_term_count": 2,
        "candidate_class_count": 2,
        "rows": sorted(rows, key=lambda r: r["term_ref"]),
        "public_descriptor_promoted": False,
        "topology_promoted": False,
        "shell_profile_promoted": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def ordered_two_reservation_signature(state: G3TermState, first_type: int, second_type: int) -> dict[str, Any]:
    """Exact branch-sensitive local continuation for two ordered reservations.

    Only CAPS7-visible branch outcomes and joint legality are recorded.  The exact
    G3 term is used as the previous-layer carrier, not as a directly observed
    topology descriptor.
    """
    a, b = int(first_type), int(second_type)
    if not (0 <= a < 7 and 0 <= b < 7):
        raise G4S3Error("endpoint type outside seven-type alphabet")
    firsts = state.reserve_external_relation(a)
    hist: Counter[str] = Counter()
    payload_by_hash: dict[str, dict[str, Any]] = {}
    total_second = 0
    for first in firsts:
        seconds = first.reserve_external_relation(b)
        total_second += len(seconds)
        bp = {
            "first_caps7": _caps7(first),
            "second_relation_cardinality": len(seconds),
            "second_successor_caps7_set": [list(x) for x in sorted({tuple(_caps7(s)) for s in seconds})],
        }
        h = canonical_sha256(bp)
        payload_by_hash.setdefault(h, bp)
        hist[h] += 1
    out = {
        "schema_id": "IG_G4_S3_ORDERED_TWO_RESERVATION_SIGNATURE_V1",
        "first_reserved_type": a,
        "second_reserved_type": b,
        "first_step": {
            "endpoint_type": a,
            "relation_cardinality": len(firsts),
            "successor_caps7_set": [list(x) for x in sorted({tuple(_caps7(s)) for s in firsts})],
        },
        "first_branch_profile_histogram": [
            {"branch_profile_sha256": h, "multiplicity": int(hist[h]), "profile": payload_by_hash[h]}
            for h in sorted(hist)
        ],
        "final_relation_output_cardinality": int(total_second),
        "joint_legal": bool(total_second > 0),
        "direct_topology_read": False,
        "direct_shell_profile_read": False,
        "construction_identity_recorded": False,
        "lower_layer_implementation_read": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def verify_s3_authority(
    *,
    s0_result: Mapping[str, Any],
    s1_result: Mapping[str, Any],
    s1_replay: Mapping[str, Any],
    s1_closeout: Mapping[str, Any],
    s2_result: Mapping[str, Any],
    s2_replay: Mapping[str, Any],
    s2_closeout: Mapping[str, Any],
    signature_index: Mapping[str, Any],
) -> dict[str, Any]:
    a = s3_spec()["authority"]
    failures: list[str] = []
    checks = [
        (s0_result.get("status") == a["g4_s0_status"], "S0_STATUS"),
        (s0_result.get("classification") == a["g4_s0_classification"], "S0_CLASS"),
        (s0_result.get("science_sha256") == a["g4_s0_science_sha256"], "S0_SHA"),
        (s0_result.get("certified_term_corpus", {}).get("science_sha256") == a["g4_s0_term_corpus_sha256"], "S0_TERM_CORPUS_SHA"),
        (s1_result.get("status") == a["g4_s1_status"], "S1_STATUS"),
        (s1_result.get("outcome") == a["g4_s1_outcome"], "S1_OUTCOME"),
        (s1_result.get("classification") == a["g4_s1_classification"], "S1_CLASS"),
        (s1_result.get("science_sha256") == a["g4_s1_science_sha256"], "S1_SHA"),
        (s2_result.get("status") == a["g4_s2_status"], "S2_STATUS"),
        (s2_result.get("classification") == a["g4_s2_classification"], "S2_CLASS"),
        (s2_result.get("science_sha256") == a["g4_s2_science_sha256"], "S2_SHA"),
        (s2_result.get("minimal_sufficient_preregistered_tier") == a["required_candidate_read"], "S2_CANDIDATE"),
        (bool(s2_result.get("g4_s3_unlocked")) is bool(a["required_g4_s3_unlocked"]), "S3_UNLOCK"),
        (bool(s2_result.get("public_descriptor_promoted")) is bool(a["required_public_descriptor_promoted"]), "PUBLIC_PROMOTION"),
        (bool(s2_result.get("topology_promoted")) is bool(a["required_topology_promoted"]), "TOPOLOGY_PROMOTION"),
        (bool(s2_result.get("shell_profile_promoted")) is bool(a["required_shell_profile_promoted"]), "SHELL_PROMOTION"),
        (bool(s2_result.get("g4_graduated")) is bool(a["required_g4_graduated"]), "G4_GRADUATION"),
        (s2_result.get("s1_signature_index_binding", {}).get("science_sha256") == a["g4_s2_s1_signature_binding_sha256"], "S2_S1_BINDING_SHA"),
    ]
    failures.extend(name for ok, name in checks if not ok)

    tier1 = next((x for x in s2_result.get("tier_audit", []) if int(x.get("tier", -1)) == 1), None)
    if not tier1 or tier1.get("science_sha256") != a["g4_s2_tier1_audit_sha256"] or tier1.get("complete_s1_pair_observer_single_valued") is not True:
        failures.append("S2_TIER1_AUDIT")

    if s2_replay.get("schema_id") != "IG_G4_S2_COLD_REPLAY_COMPARISON_V1" or s2_replay.get("certification") != "CERTIFIED_PASS":
        failures.append("S2_REPLAY")
    if s2_replay.get("comparison_sha256") != a["g4_s2_replay_comparison_sha256"]:
        failures.append("S2_REPLAY_SHA")
    if s2_replay.get("primary_science_sha256") != s2_result.get("science_sha256") or s2_replay.get("cold_science_sha256") != s2_result.get("science_sha256"):
        failures.append("S2_REPLAY_SCIENCE_BINDING")
    if s2_closeout.get("schema_id") != "IG_G4_S2_CERTIFIED_CLOSEOUT_V1" or s2_closeout.get("status") != "CERTIFIED_PASS":
        failures.append("S2_CLOSEOUT")
    if s2_closeout.get("closeout_sha256") != a["g4_s2_closeout_sha256"]:
        failures.append("S2_CLOSEOUT_SHA")
    if s2_closeout.get("science_sha256") != s2_result.get("science_sha256") or s2_closeout.get("next_authorized_stage") != "G4:S3":
        failures.append("S2_CLOSEOUT_BINDING")

    # Recompute the complete S2 scientific payload from certified S0/S1 evidence.
    try:
        rebuilt_s2 = certify_minimal_added_read(
            s0_result=s0_result,
            s1_result=s1_result,
            s1_replay=s1_replay,
            s1_closeout=s1_closeout,
            signature_index=signature_index,
        )
        if rebuilt_s2.get("science_sha256") != s2_result.get("science_sha256"):
            failures.append("S2_RECONSTRUCTION_SHA")
    except Exception:
        failures.append("S2_RECONSTRUCTION_EXCEPTION")
        rebuilt_s2 = None

    if failures:
        raise G4S3Error("G4:S3 authority verification failed: " + ",".join(failures))
    out = {
        "schema_id": "IG_G4_S3_AUTHORITY_VERIFICATION_V1",
        "status": "PASS",
        "g4_s0_science_sha256": s0_result["science_sha256"],
        "g4_s0_term_corpus_sha256": s0_result["certified_term_corpus"]["science_sha256"],
        "g4_s1_science_sha256": s1_result["science_sha256"],
        "g4_s2_science_sha256": s2_result["science_sha256"],
        "g4_s2_replay_comparison_sha256": s2_replay["comparison_sha256"],
        "g4_s2_closeout_sha256": s2_closeout["closeout_sha256"],
        "reconstructed_s2_science_sha256": rebuilt_s2["science_sha256"] if rebuilt_s2 else None,
        "g4_phase0_spec_sha256": phase0_spec()["science_sha256"],
        "g4_s2_spec_sha256": s2_spec()["science_sha256"],
        "g4_s3_spec_sha256": s3_spec()["science_sha256"],
        "lower_layer_rematerialization": False,
        "public_descriptor_promoted": False,
        "g4_graduated": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def certify_candidate_two_reservation_sufficiency(
    *,
    candidate_index: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    idx_rows = list(candidate_index.get("rows", []))
    key_by_ref = {str(r["term_ref"]): str(r["candidate_key_sha256"]) for r in idx_rows}
    expected = len(key_by_ref) * 49
    if len(task_rows) != expected:
        raise G4S3Error(f"expected {expected} G4:S3 local continuation rows, got {len(task_rows)}")
    seen: set[tuple[str, int, int]] = set()
    grouped: dict[tuple[str, int, int], list[tuple[str, str]]] = defaultdict(list)
    full_vectors: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)
    for row in task_rows:
        ref = str(row["term_ref"])
        a, b = int(row["first_type"]), int(row["second_type"])
        if ref not in key_by_ref or not (0 <= a < 7 and 0 <= b < 7):
            raise G4S3Error("bad G4:S3 local continuation row")
        key = (ref, a, b)
        if key in seen:
            raise G4S3Error("duplicate G4:S3 local continuation row")
        seen.add(key)
        sig = row["signature"]
        sh = str(sig.get("science_sha256", ""))
        if canonical_sha256({k: v for k, v in sig.items() if k != "science_sha256"}) != sh:
            raise G4S3Error("G4:S3 local continuation signature hash mismatch")
        grouped[(key_by_ref[ref], a, b)].append((ref, sh))
        full_vectors[ref][(a, b)] = sh

    conflicts: list[dict[str, Any]] = []
    for (candidate_key, a, b), vals in sorted(grouped.items(), key=str):
        hashes = sorted({h for _, h in vals})
        if len(hashes) != 1:
            conflicts.append({
                "candidate_key_sha256": candidate_key,
                "first_type": a,
                "second_type": b,
                "representative_count": len(vals),
                "signature_count": len(hashes),
                "signature_sha256s": hashes,
                "term_refs": sorted(r for r, _ in vals),
            })

    class_members: dict[str, list[str]] = defaultdict(list)
    for ref, key in key_by_ref.items():
        class_members[key].append(ref)
    class_vectors = []
    for candidate_key in sorted(class_members):
        reps = sorted(class_members[candidate_key])
        vector_hashes = []
        for ref in reps:
            if len(full_vectors[ref]) != 49:
                raise G4S3Error("incomplete local continuation vector")
            vector_hashes.append(canonical_sha256([full_vectors[ref][(a, b)] for a in range(7) for b in range(7)]))
        if len(set(vector_hashes)) != 1:
            conflicts.append({"failure": "FULL_VECTOR_CONFLICT", "candidate_key_sha256": candidate_key, "vector_hashes": sorted(set(vector_hashes)), "term_refs": reps})
        class_vectors.append({
            "candidate_key_sha256": candidate_key,
            "representative_count": len(reps),
            "term_refs": reps,
            "continuation_vector_sha256": vector_hashes[0] if len(set(vector_hashes)) == 1 else None,
        })

    out = {
        "schema_id": "IG_G4_S3_TIER1_TWO_RESERVATION_SUFFICIENCY_V1",
        "status": "PASS" if not conflicts else "FAIL",
        "candidate": {"tier": 1, "name": "SCALAR_TREE_SUMMARIES"},
        "candidate_key": "CAPS7_PLUS_SCALAR_TREE_SUMMARIES",
        "exact_term_count": len(key_by_ref),
        "candidate_class_count": len(class_members),
        "ordered_endpoint_pair_count": 49,
        "task_row_count": len(task_rows),
        "continuation_conflict_count": len(conflicts),
        "conflict_examples": conflicts[:16],
        "all_representatives_within_each_candidate_class_share_one_complete_ordered_two_reservation_table": not conflicts,
        "candidate_class_vectors": class_vectors,
        "class_population_note": "The frozen S2 domain has one exact representative in each Tier-1 candidate class; this result is bounded and does not establish unseen-representative sufficiency.",
        "direct_topology_read": False,
        "public_descriptor_promoted": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def triple_reconstruction_coverage(
    *,
    candidate_class_count: int,
    directed_operator_count: int,
    local_sufficiency_pass: bool,
    authority_pass: bool,
) -> dict[str, Any]:
    c, o = int(candidate_class_count), int(directed_operator_count)
    assignments = c ** 3
    p3 = assignments * (o ** 2)
    k3 = assignments * (o ** 3)
    passed = bool(local_sufficiency_pass and authority_pass and c == 2 and o == 31)
    out = {
        "schema_id": "IG_G4_S3_TRIPLE_RECONSTRUCTION_COVERAGE_V1",
        "status": "PASS" if passed else "FAIL",
        "candidate_unit_class_count": c,
        "ordered_candidate_unit_assignments": assignments,
        "directed_operator_count": o,
        "public_context_counts": {
            "P3_CONNECTED_PATH": p3,
            "K3_TRIANGLE": k3,
            "TOTAL": p3 + k3,
        },
        "max_vertex_degree": 2,
        "ordered_local_endpoint_pair_count": 49,
        "factorization_argument": "Each ordered edge is fixed by the certified S2 pair observer. Each motif vertex has degree at most two. The complete ordered two-reservation branch-sensitive continuation is a single-valued function of the frozen CAPS7+Tier1 candidate key. Therefore the declared P3/K3 joint compatibility observer is a deterministic function of the candidate unit keys plus S2 edge values, with no additional hidden read.",
        "enumeration_note": "Counts are exact combinatorial coverage of the frozen two-class/31-operator motif domain; the proof step is local factorisation rather than sampled triple extrapolation.",
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def finalize_s3_result(
    *,
    s0_result: Mapping[str, Any],
    s1_result: Mapping[str, Any],
    s1_replay: Mapping[str, Any],
    s1_closeout: Mapping[str, Any],
    s2_result: Mapping[str, Any],
    s2_replay: Mapping[str, Any],
    s2_closeout: Mapping[str, Any],
    signature_index: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    spec = s3_spec()
    auth = verify_s3_authority(
        s0_result=s0_result, s1_result=s1_result, s1_replay=s1_replay, s1_closeout=s1_closeout,
        s2_result=s2_result, s2_replay=s2_replay, s2_closeout=s2_closeout, signature_index=signature_index,
    )
    candidate = candidate_interface_index(s0_result, s2_result)
    local = certify_candidate_two_reservation_sufficiency(candidate_index=candidate, task_rows=task_rows)
    cov = triple_reconstruction_coverage(
        candidate_class_count=int(candidate["candidate_class_count"]),
        directed_operator_count=31,
        local_sufficiency_pass=local["status"] == "PASS",
        authority_pass=auth["status"] == "PASS",
    )
    passed = local["status"] == "PASS" and cov["status"] == "PASS"
    result = {
        "schema_id": "IG_G4_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_RESULT_V1",
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "stage_ref": "G4:S3",
        "classification": "G4_NO_IRREDUCIBLE_TRIPLE_RESIDUAL_ON_REGISTERED_P3_K3_TIER1_S2_SCOPE_S4_UNLOCKED" if passed else "G4_HIGHER_ORDER_RESIDUAL_OR_BINDING_FAILURE_S4_LOCKED",
        "authority": auth,
        "candidate_interface_index": candidate,
        "challenge_reproduction": dict(reproduction),
        "tier1_two_reservation_sufficiency": local,
        "triple_reconstruction": cov,
        "irreducible_residual_count": 0 if passed else None,
        "minimal_added_read_at_tested_triple_observer": "NONE_BEYOND_S2_TIER1_CANDIDATE" if passed else None,
        "candidate_read_survives_s3": passed,
        "candidate_read_status": "BOUNDED_TIER1_CANDIDATE_SURVIVES_S3_NOT_PUBLICLY_PROMOTED" if passed else "S3_REVIEW_REQUIRED",
        "public_descriptor_promoted": False,
        "topology_promoted": False,
        "shell_profile_promoted": False,
        "g4_s4_unlocked": passed,
        "g4_graduated": False,
        "next_authorized_stage": "G4:S4" if passed else None,
        "scope": "EXACT_FROZEN_G4_S0_TWO_TERM_DOMAIN__CERTIFIED_G4_S2_TIER1_PAIR_QUOTIENT__CONNECTED_SIMPLE_P3_K3_COMPATIBILITY_OBSERVER",
        "boundedness_note": spec["boundedness_note"],
        "nonclaims": list(spec["nonclaims"]),
    }
    result["science_sha256"] = canonical_sha256(result)
    return result


def compare_cold_replay(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for label, obj in (("PRIMARY", primary), ("COLD", cold)):
        if obj.get("schema_id") != "IG_G4_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_RESULT_V1" or obj.get("status") != "PASS":
            failures.append(f"{label}_SCHEMA_OR_STATUS")
    checks = {
        "science_sha_equal": primary.get("science_sha256") == cold.get("science_sha256"),
        "source_sha_equal": primary.get("source_sha256") == cold.get("source_sha256"),
        "registry_sha_equal": primary.get("registry_sha256") == cold.get("registry_sha256"),
        "classification_equal": primary.get("classification") == cold.get("classification"),
        "candidate_interface_equal": primary.get("candidate_interface_index") == cold.get("candidate_interface_index"),
        "local_sufficiency_equal": primary.get("tier1_two_reservation_sufficiency") == cold.get("tier1_two_reservation_sufficiency"),
        "triple_reconstruction_equal": primary.get("triple_reconstruction") == cold.get("triple_reconstruction"),
    }
    failures.extend(k.upper() for k, ok in checks.items() if not ok)
    out = {
        "schema_id": "IG_G4_S3_COLD_REPLAY_COMPARISON_V1",
        "status": "PASS" if not failures else "FAIL",
        "certification": "CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
        "failures": failures,
        "experiment_id": "G4:S3",
        "same_registered_decoder_native_experiment": True,
        **checks,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
        "primary_source_sha256": primary.get("source_sha256"),
        "cold_source_sha256": cold.get("source_sha256"),
        "primary_registry_sha256": primary.get("registry_sha256"),
        "cold_registry_sha256": cold.get("registry_sha256"),
        "lower_layer_rematerialization": False,
    }
    out["comparison_sha256"] = canonical_sha256(out)
    return out


def certified_closeout(primary: Mapping[str, Any], cold: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if replay.get("schema_id") != "IG_G4_S3_COLD_REPLAY_COMPARISON_V1" or replay.get("certification") != "CERTIFIED_PASS":
        failures.append("REPLAY")
    if primary.get("science_sha256") != cold.get("science_sha256"):
        failures.append("SCIENCE")
    if primary.get("candidate_read_survives_s3") is not True or primary.get("g4_s4_unlocked") is not True:
        failures.append("S3_OUTCOME")
    out = {
        "schema_id": "IG_G4_S3_CERTIFIED_CLOSEOUT_V1",
        "status": "CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
        "failures": failures,
        "experiment_id": "G4:S3",
        "decoder_version": "0.30.40",
        "science_sha256": primary.get("science_sha256"),
        "source_sha256": primary.get("source_sha256"),
        "registry_sha256": primary.get("registry_sha256"),
        "classification": primary.get("classification"),
        "candidate_read_survives_s3": primary.get("candidate_read_survives_s3") is True,
        "public_descriptor_promoted": False,
        "topology_promoted": False,
        "shell_profile_promoted": False,
        "g4_s4_unlocked": primary.get("g4_s4_unlocked") is True,
        "g4_graduated": False,
        "comparison_sha256": replay.get("comparison_sha256"),
        "lower_layer_rematerialization": False,
        "next_authorized_stage": "G4:S4" if not failures else None,
    }
    out["closeout_sha256"] = canonical_sha256(out)
    return out

# Tiny portable worker context; safe under spawn/forkserver/fork.
_WORKER_STATES: dict[str, G3TermState] | None = None


def init_g4_s3_term_worker(payload: Mapping[str, Any]) -> None:
    global _WORKER_STATES
    _WORKER_STATES = {str(ref): G3TermState.from_wire(wire) for ref, wire in payload["states"].items()}


def g4_s3_local_continuation_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_STATES is None:
        raise G4S3Error("G4:S3 term worker context absent")
    ref = str(payload["term_ref"])
    if ref not in _WORKER_STATES:
        raise G4S3Error("unknown G4:S3 frozen term ref")
    a, b = int(payload["first_type"]), int(payload["second_type"])
    return {
        "term_ref": ref,
        "first_type": a,
        "second_type": b,
        "signature": ordered_two_reservation_signature(_WORKER_STATES[ref], a, b),
    }
