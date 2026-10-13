from __future__ import annotations

"""Registered G4:S2 minimal added-read pair-observer quotient audit.

S2 consumes only the corrected reusable G4:S0 term base plus the certified G4:S1
248-row operational signature table.  It does not rematerialize lower layers.

For each preregistered challenge-read tier, exact target/context construction refs are
replaced by the tier's relabel-invariant descriptor values.  The tier is sufficient
on the frozen pair-observer domain iff every resulting public pair-context key is
single-valued in the complete certified S1 operational observer.  The first
sufficient eligible tier is the bounded S2 candidate read.  This is not G4 graduation
or global minimality.
"""

from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import gzip, json

from .canon import canonical_sha256
from .uplift_g4_s0 import phase0_spec, challenge_tiers


class G4S2Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s2_spec() -> dict[str, Any]:
    p = _resource("G4_S2_MINIMAL_ADDED_READ_PAIR_QUOTIENT_SPEC_V1.json")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G4_S2_MINIMAL_ADDED_READ_PAIR_QUOTIENT_SPEC_V1":
        raise G4S2Error("bad G4:S2 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G4S2Error("G4:S2 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G4S2Error("G4:S2/Phase0 binding mismatch")
    if obj.get("challenge_read_tier_science_sha256") != challenge_tiers().get("science_sha256"):
        raise G4S2Error("G4:S2/tier-ladder binding mismatch")
    return obj


def load_s1_signature_index(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise G4S2Error("G4:S1 signature index missing")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("schema_id") != "IG_G4_S1_PAIR_CONTEXT_SIGNATURE_INDEX_V2":
        raise G4S2Error("bad G4:S1 signature index schema")
    rows = obj.get("rows", [])
    if int(obj.get("task_count", -1)) != len(rows):
        raise G4S2Error("G4:S1 signature index count mismatch")
    return obj


def verify_s2_authority(*, s0_result: Mapping[str, Any], s1_result: Mapping[str, Any],
                        s1_replay: Mapping[str, Any], s1_closeout: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    spec = s2_spec()
    if s0_result.get("schema_id") != "IG_G4_S0_INTERFACE_EXTRACTION_RESULT_V2" or s0_result.get("status") != "PASS":
        failures.append("S0_SCHEMA_OR_STATUS")
    if s0_result.get("s0_materializes_reusable_base") is not True:
        failures.append("S0_REUSABLE_BASE")
    if s0_result.get("s1_lower_layer_rematerialization_forbidden") is not True:
        failures.append("S0_LOWER_LAYER_FIREWALL")
    if s1_result.get("schema_id") != "IG_G4_S1_HIDDEN_STRUCTURE_CONTEXT_READ_RESULT_V2" or s1_result.get("status") != "PASS":
        failures.append("S1_SCHEMA_OR_STATUS")
    if s1_result.get("outcome") != "STRUCTURE_READ_EARNED" or s1_result.get("g4_s2_unlocked") is not True:
        failures.append("S1_OUTCOME_OR_UNLOCK")
    if s1_result.get("topology_promoted") is not False or s1_result.get("shell_profile_promoted") is not False:
        failures.append("S1_PROMOTION_FIREWALL")
    if int(s1_result.get("context_basis", {}).get("task_count", -1)) != int(spec["input_evidence"]["expected_s1_task_count"]):
        failures.append("S1_TASK_COUNT")
    if s1_replay.get("schema_id") != "IG_G4_S1_COLD_REPLAY_COMPARISON_V1" or s1_replay.get("certification") != "CERTIFIED_PASS":
        failures.append("S1_REPLAY")
    if s1_closeout.get("schema_id") != "IG_G4_S1_CERTIFIED_CLOSEOUT_V1" or s1_closeout.get("status") != "CERTIFIED_PASS":
        failures.append("S1_CLOSEOUT")
    if s1_closeout.get("science_sha256") != s1_result.get("science_sha256"):
        failures.append("S1_CLOSEOUT_SCIENCE_BINDING")
    if s1_closeout.get("next_authorized_stage") != "G4:S2":
        failures.append("S1_CLOSEOUT_AUTHORIZATION")
    if failures:
        raise G4S2Error("G4:S2 authority failed: " + ",".join(failures))
    out = {
        "schema_id": "IG_G4_S2_AUTHORITY_VERIFICATION_V1",
        "status": "PASS",
        "g4_s0_science_sha256": s0_result.get("science_sha256"),
        "g4_s0_term_corpus_sha256": s0_result.get("certified_term_corpus", {}).get("science_sha256"),
        "g4_s1_science_sha256": s1_result.get("science_sha256"),
        "g4_s1_replay_comparison_sha256": s1_replay.get("comparison_sha256"),
        "g4_s1_closeout_sha256": s1_closeout.get("closeout_sha256"),
        "g4_phase0_spec_sha256": phase0_spec()["science_sha256"],
        "g4_s2_spec_sha256": spec["science_sha256"],
        "challenge_read_tier_science_sha256": challenge_tiers()["science_sha256"],
        "lower_layer_rematerialization": False,
        "g4_graduated": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _descriptor_values(s0_result: Mapping[str, Any]) -> tuple[dict[str, dict[int, Any]], dict[int, str]]:
    pair = s0_result["certified_collision_pair"]
    values: dict[str, dict[int, Any]] = {}
    names = {int(t["tier"]): str(t["name"]) for t in challenge_tiers()["ordered_tiers"]}
    for key in ("A", "B"):
        row = pair[key]
        ref = str(row["g4_term_ref"])
        d = row["hidden_challenge_diagnostic"]
        deg = [int(x) for x in d["degree_sequence"]]
        vals = {
            0: [int(x) for x in row["public_interface"]["coordinates"]],
            1: {
                "MAX_DEGREE": max(deg),
                "LEAF_COUNT": sum(1 for x in deg if x == 1),
                "diameter": int(d["diameter"]),
                "radius": int(d["radius"]),
                "articulation_count": int(d["articulation_count"]),
                "wiener_index": int(d["wiener_index"]),
            },
            2: deg,
            3: {"degree_sequence": deg, "wiener_index": int(d["wiener_index"])},
            4: d["shell_profile_multiset"],
            5: str(d["topology_canon"]),
        }
        values[ref] = vals
    if len(values) != 2:
        raise G4S2Error("G4:S2 expects exactly two certified S0 terms")
    return values, names


def _value_hash(v: Any) -> str:
    return canonical_sha256({"descriptor_value": v})


def _verify_s1_signature_index_binding(*, signature_index: Mapping[str, Any],
                                       s1_result: Mapping[str, Any],
                                       certified_refs: list[str]) -> dict[str, Any]:
    """Fail closed unless the supplied 248-row index is the certified S1 observer table.

    S1 certified per-target behavior hashes over rows containing context, orientation,
    operator and operational-signature hash.  Reconstructing those hashes here binds
    the standalone compressed signature index back to the certified S1 result without
    reopening any lower layer or adding a new scientific read.
    """
    if signature_index.get("schema_id") != "IG_G4_S1_PAIR_CONTEXT_SIGNATURE_INDEX_V2":
        raise G4S2Error("bad G4:S1 signature index schema")
    rows = list(signature_index.get("rows", []))
    expected_total = int(s1_result.get("context_basis", {}).get("task_count", -1))
    if int(signature_index.get("task_count", -1)) != len(rows) or len(rows) != expected_total:
        raise G4S2Error("G4:S1 signature index count/certified-task mismatch")

    refs = sorted(str(x) for x in certified_refs)
    if sorted(str(x) for x in s1_result.get("challenge_reproduction", {}).get("carrier_refs", [])) != refs:
        raise G4S2Error("certified S1 carrier refs mismatch")
    expected_orientations = {"CONTEXT_LEFT_TARGET_RIGHT", "TARGET_LEFT_CONTEXT_RIGHT"}
    expected_operator_count = int(s1_result.get("context_basis", {}).get("operator_count", -1))
    expected_orientation_count = int(s1_result.get("context_basis", {}).get("orientation_count", -1))
    expected_target_count = int(s1_result.get("context_basis", {}).get("target_count", -1))
    expected_context_count = int(s1_result.get("context_basis", {}).get("context_partner_count", -1))
    if (expected_target_count, expected_context_count, expected_orientation_count) != (len(refs), len(refs), 2):
        raise G4S2Error("certified S1 context-basis dimensions mismatch")

    seen: set[tuple[str, str, str, tuple[int, int]]] = set()
    operator_sets: dict[tuple[str, str, str], set[tuple[int, int]]] = defaultdict(set)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_ops: set[tuple[int, int]] = set()
    for row in rows:
        tr, cr = str(row.get("target_ref", "")), str(row.get("context_ref", ""))
        if tr not in refs or cr not in refs:
            raise G4S2Error("S1 row outside certified S0/S1 term base")
        ori = str(row.get("orientation", ""))
        if ori not in expected_orientations:
            raise G4S2Error("unexpected S1 orientation")
        op_raw = row.get("operator")
        if not isinstance(op_raw, list) or len(op_raw) != 2:
            raise G4S2Error("bad S1 operator")
        op = tuple(map(int, op_raw))
        key = (tr, cr, ori, op)
        if key in seen:
            raise G4S2Error("duplicate S1 pair-context task key")
        seen.add(key)
        operator_sets[(tr, cr, ori)].add(op)
        all_ops.add(op)

        sig = row.get("operational_signature", {})
        sh = str(sig.get("science_sha256", ""))
        payload = {k: v for k, v in sig.items() if k != "science_sha256"}
        if canonical_sha256(payload) != sh:
            raise G4S2Error("S1 operational signature hash mismatch")
        by_target[tr].append({
            "context_ref": cr,
            "orientation": ori,
            "operator": list(op),
            "operational_signature_sha256": sh,
        })

    if len(all_ops) != expected_operator_count:
        raise G4S2Error("S1 operator basis count mismatch")
    expected_combo_count = len(refs) * len(refs) * len(expected_orientations)
    if len(operator_sets) != expected_combo_count:
        raise G4S2Error("S1 pair-context-orientation coverage incomplete")
    for ops in operator_sets.values():
        if ops != all_ops:
            raise G4S2Error("S1 operator basis not complete on every pair-context orientation")

    summaries = {str(x.get("target_ref")): x for x in s1_result.get("target_behavior_summaries", [])}
    if set(summaries) != set(refs) or len(summaries) != len(refs):
        raise G4S2Error("certified S1 target behavior summaries mismatch")
    reconstructed: dict[str, str] = {}
    for ref in refs:
        rr = by_target.get(ref, [])
        rr.sort(key=lambda x: (x["context_ref"], x["orientation"], tuple(x["operator"])))
        expected_rows = int(summaries[ref].get("context_row_count", -1))
        if len(rr) != expected_rows:
            raise G4S2Error("S1 target behavior row-count mismatch")
        h = canonical_sha256(rr)
        if h != summaries[ref].get("behavior_signature_sha256"):
            raise G4S2Error("S1 signature index not bound to certified target behavior summary")
        reconstructed[ref] = h

    out = {
        "schema_id": "IG_G4_S2_S1_SIGNATURE_INDEX_BINDING_V1",
        "status": "PASS",
        "task_count": len(rows),
        "unique_task_key_count": len(seen),
        "operator_count": len(all_ops),
        "orientation_count": len(expected_orientations),
        "target_count": len(refs),
        "context_partner_count": len(refs),
        "certified_target_behavior_sha256": reconstructed,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def certify_minimal_added_read(*, s0_result: Mapping[str, Any], s1_result: Mapping[str, Any],
                               s1_replay: Mapping[str, Any], s1_closeout: Mapping[str, Any],
                               signature_index: Mapping[str, Any]) -> dict[str, Any]:
    spec = s2_spec()
    auth = verify_s2_authority(s0_result=s0_result, s1_result=s1_result, s1_replay=s1_replay, s1_closeout=s1_closeout)
    rows = list(signature_index.get("rows", []))
    expected = int(spec["input_evidence"]["expected_s1_task_count"])
    if len(rows) != expected:
        raise G4S2Error(f"expected {expected} S1 rows, got {len(rows)}")

    desc, names = _descriptor_values(s0_result)
    refs = sorted(desc)
    if set(refs) != {str(x) for x in s1_result.get("challenge_reproduction", {}).get("carrier_refs", [])}:
        raise G4S2Error("S0/S1 carrier ref mismatch")
    signature_binding = _verify_s1_signature_index_binding(
        signature_index=signature_index, s1_result=s1_result, certified_refs=refs
    )

    tier_results: list[dict[str, Any]] = []
    first_sufficient: dict[str, Any] | None = None
    for tier in range(0, 6):
        grouped: dict[tuple[str, str, tuple[int, int], str], list[tuple[str, str, str]]] = defaultdict(list)
        for row in rows:
            tr, cr = str(row["target_ref"]), str(row["context_ref"])
            if tr not in desc or cr not in desc:
                raise G4S2Error("S1 row outside certified S0 term base")
            op = tuple(map(int, row["operator"])); ori = str(row["orientation"])
            sig = row.get("operational_signature", {})
            sh = str(sig.get("science_sha256", ""))
            payload = {k: v for k, v in sig.items() if k != "science_sha256"}
            if canonical_sha256(payload) != sh:
                raise G4S2Error("S1 operational signature hash mismatch")
            key = (_value_hash(desc[tr][tier]), _value_hash(desc[cr][tier]), op, ori)
            grouped[key].append((tr, cr, sh))

        conflicts = []
        max_mult = 0
        for key, vals in grouped.items():
            max_mult = max(max_mult, len(vals))
            hashes = sorted({x[2] for x in vals})
            if len(hashes) != 1:
                conflicts.append({
                    "target_descriptor_sha256": key[0],
                    "context_descriptor_sha256": key[1],
                    "operator": list(key[2]),
                    "orientation": key[3],
                    "exact_representative_pair_count": len(vals),
                    "observer_value_count": len(hashes),
                    "observer_hashes": hashes,
                })
        sufficient = len(conflicts) == 0
        r = {
            "tier": tier,
            "name": names[tier],
            "descriptor_class_count": len({_value_hash(desc[r][tier]) for r in refs}),
            "quotient_pair_context_key_count": len(grouped),
            "max_exact_representatives_per_quotient_key": max_mult,
            "conflict_count": len(conflicts),
            "complete_s1_pair_observer_single_valued": sufficient,
            "conflicts": conflicts[:16],
        }
        r["science_sha256"] = canonical_sha256(r)
        tier_results.append(r)
        eligible = bool(next(x for x in challenge_tiers()["ordered_tiers"] if int(x["tier"]) == tier).get("eligible_for_promotion"))
        if tier > 0 and eligible and sufficient and first_sufficient is None:
            first_sufficient = r

    if tier_results[0]["complete_s1_pair_observer_single_valued"]:
        # This would contradict certified S1 STRUCTURE_READ_EARNED.
        raise G4S2Error("CAPS7 unexpectedly sufficient in S2 despite certified S1 separation")
    if first_sufficient is None:
        status = "REVIEW_REQUIRED"
        classification = "G4_S2_NO_PREREGISTERED_TIER_SUFFICIENT_S3_LOCKED"
        next_stage = None
        candidate = None
    else:
        status = "PASS"
        classification = f"G4_S2_TIER{first_sufficient['tier']}_{first_sufficient['name']}_PAIR_OBSERVER_QUOTIENT_CERTIFIED_S3_UNLOCKED"
        next_stage = "G4:S3"
        candidate = {"tier": first_sufficient["tier"], "name": first_sufficient["name"]}

    result = {
        "schema_id": "IG_G4_S2_MINIMAL_ADDED_READ_PAIR_QUOTIENT_RESULT_V1",
        "status": status,
        "stage_ref": "G4:S2",
        "classification": classification,
        "authority": auth,
        "s1_signature_index_binding": signature_binding,
        "frozen_domain": {
            "certified_g3_term_count": len(refs),
            "s1_exact_pair_context_rows": len(rows),
            "operator_count": 31,
            "orientation_count": 2,
            "lower_layer_rematerialization": False,
        },
        "tier_audit": tier_results,
        "caps7_sufficient": tier_results[0]["complete_s1_pair_observer_single_valued"],
        "minimal_sufficient_preregistered_tier": candidate,
        "candidate_read_earned_for_s3_testing": candidate is not None,
        "public_descriptor_promoted": False,
        "promotion_status": "BOUNDED_PAIR_OBSERVER_CANDIDATE_ONLY__S3_MUST_TEST_HIGHER_ORDER_IRREDUCIBILITY" if candidate else "NONE",
        "topology_promoted": False,
        "shell_profile_promoted": False,
        "g4_s3_unlocked": candidate is not None,
        "g4_graduated": False,
        "next_authorized_stage": next_stage,
        "scope": "EXACT_ONLY_ON_CERTIFIED_G4_S0_TWO_TERM_BASE_UNDER_COMPLETE_REGISTERED_G4_S1_PAIR_CONTEXT_OBSERVER",
        "nonclaims": list(spec["nonclaims"]),
    }
    result["science_sha256"] = canonical_sha256(result)
    return result


def compare_cold_replay(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for label, obj in (("PRIMARY", primary), ("COLD", cold)):
        if obj.get("schema_id") != "IG_G4_S2_MINIMAL_ADDED_READ_PAIR_QUOTIENT_RESULT_V1" or obj.get("status") != "PASS":
            failures.append(f"{label}_SCHEMA_OR_STATUS")
    checks = {
        "science_sha_equal": primary.get("science_sha256") == cold.get("science_sha256"),
        "source_sha_equal": primary.get("source_sha256") == cold.get("source_sha256"),
        "registry_sha_equal": primary.get("registry_sha256") == cold.get("registry_sha256"),
        "classification_equal": primary.get("classification") == cold.get("classification"),
        "tier_audit_equal": primary.get("tier_audit") == cold.get("tier_audit"),
        "minimal_tier_equal": primary.get("minimal_sufficient_preregistered_tier") == cold.get("minimal_sufficient_preregistered_tier"),
    }
    failures.extend(k.upper() for k, ok in checks.items() if not ok)
    out = {
        "schema_id": "IG_G4_S2_COLD_REPLAY_COMPARISON_V1",
        "status": "PASS" if not failures else "FAIL",
        "certification": "CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
        "failures": failures,
        "experiment_id": "G4:S2",
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
    if replay.get("schema_id") != "IG_G4_S2_COLD_REPLAY_COMPARISON_V1" or replay.get("certification") != "CERTIFIED_PASS":
        failures.append("REPLAY")
    if primary.get("science_sha256") != cold.get("science_sha256"):
        failures.append("SCIENCE")
    out = {
        "schema_id": "IG_G4_S2_CERTIFIED_CLOSEOUT_V1",
        "status": "CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
        "failures": failures,
        "experiment_id": "G4:S2",
        "decoder_version": "0.30.39",
        "science_sha256": primary.get("science_sha256"),
        "source_sha256": primary.get("source_sha256"),
        "registry_sha256": primary.get("registry_sha256"),
        "classification": primary.get("classification"),
        "minimal_sufficient_preregistered_tier": primary.get("minimal_sufficient_preregistered_tier"),
        "candidate_read_earned_for_s3_testing": primary.get("candidate_read_earned_for_s3_testing") is True,
        "public_descriptor_promoted": False,
        "g4_s3_unlocked": primary.get("g4_s3_unlocked") is True,
        "g4_graduated": False,
        "comparison_sha256": replay.get("comparison_sha256"),
        "lower_layer_rematerialization": False,
        "next_authorized_stage": "G4:S3" if not failures else None,
    }
    out["closeout_sha256"] = canonical_sha256(out)
    return out
