from __future__ import annotations

"""Registered G3:S5 finite read/write descriptor audit.

S4 certified one-step relation-valued composition closure for the frozen G3 P3 challenge
domain. S5 asks the next, narrower question: whether the complete exact candidate relations
can be quotiented to a finite-coordinate read/write state for the admitted *one-step* G3
transition grammar. The candidate is inherited CAPS7.

This stage does not prove arbitrary recursive closure, pair+pair closure, minimality, topology
erasure, or G3 graduation. PASS unlocks S6 only.
"""

from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip
import inspect
import json

from .canon import canonical_sha256
from .g2_relation import relation_spec
from .uplift_g3_s0 import phase0_spec


class G3S5Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s5_spec() -> dict[str, Any]:
    obj = json.loads(_resource("G3_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G3_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1":
        raise G3S5Error("bad G3:S5 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G3S5Error("G3:S5 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G3S5Error("G3:S5/Phase0 binding mismatch")
    return obj


def _caps(v: Sequence[Any], field: str = "caps7") -> tuple[int, ...]:
    out = tuple(int(x) for x in v)
    if len(out) != 7 or any(x < 0 for x in out):
        raise G3S5Error(f"{field} must contain seven nonnegative integers")
    return out


def descriptor_from_caps(v: Sequence[Any]) -> dict[str, Any]:
    caps = _caps(v)
    out = {
        "schema_id": "IG_G3_CAPS7_READ_WRITE_STATE_V1",
        "total_free_by_type": list(caps),
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def descriptor_from_exact_relation(relation: Sequence[Any]) -> dict[str, Any]:
    """Read CAPS7 from a complete exact relation without selecting a branch."""
    if not relation:
        raise G3S5Error("empty exact G3 candidate relation")
    states = {_caps(st.total_caps, "relation branch CAPS7") for st in relation}
    if len(states) != 1:
        raise G3S5Error("exact G3 candidate relation does not factor to one CAPS7 state")
    return descriptor_from_caps(next(iter(states)))


def reserve_enabled(caps: Sequence[Any], endpoint_type: int) -> bool:
    c = _caps(caps)
    t = int(endpoint_type)
    if t < 0 or t >= 7:
        raise G3S5Error("endpoint type outside seven-type alphabet")
    return c[t] > 0


def reserve_write(caps: Sequence[Any], endpoint_type: int) -> tuple[int, ...] | None:
    c = list(_caps(caps))
    t = int(endpoint_type)
    if t < 0 or t >= 7:
        raise G3S5Error("endpoint type outside seven-type alphabet")
    if c[t] <= 0:
        return None
    c[t] -= 1
    return tuple(c)


def binary_enabled(left_caps: Sequence[Any], right_caps: Sequence[Any], a: int, b: int, operator_basis: Sequence[Sequence[int]]) -> bool:
    left, right = _caps(left_caps, "left CAPS7"), _caps(right_caps, "right CAPS7")
    aa, bb = int(a), int(b)
    basis = {(int(x), int(y)) for x, y in operator_basis}
    return (aa, bb) in basis and left[aa] > 0 and right[bb] > 0


def binary_write(left_caps: Sequence[Any], right_caps: Sequence[Any], a: int, b: int, operator_basis: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    left, right = _caps(left_caps, "left CAPS7"), _caps(right_caps, "right CAPS7")
    aa, bb = int(a), int(b)
    if not binary_enabled(left, right, aa, bb, operator_basis):
        return None
    out = [left[i] + right[i] for i in range(7)]
    out[aa] -= 1
    out[bb] -= 1
    if any(x < 0 for x in out):
        raise G3S5Error("CAPS7 binary write underflow")
    return tuple(out)


def verify_s5_authority(*, s4_result: Mapping[str, Any], s4_replay_comparison: Mapping[str, Any]) -> dict[str, Any]:
    spec = s5_spec()
    failures: list[str] = []
    if s4_result.get("schema_id") != "IG_G3_S4_COMPOSITION_CLOSURE_RESULT_V1":
        failures.append("S4_SCHEMA")
    if s4_result.get("status") != "PASS":
        failures.append("S4_STATUS")
    if s4_result.get("classification") != "G3_ONE_STEP_RELATION_VALUED_COMPOSITION_CLOSURE_ON_FROZEN_P3_CAPS7_SCOPE_S5_UNLOCKED":
        failures.append("S4_CLASSIFICATION")
    if s4_result.get("science_sha256") != spec["authority"]["g3_s4_science_sha256"]:
        failures.append("S4_SCIENCE")
    if s4_result.get("g3_s5_unlocked") is not True:
        failures.append("S5_NOT_UNLOCKED")
    if s4_result.get("g3_graduated") is not False:
        failures.append("G3_GRADUATION_FIREWALL")
    if s4_result.get("topology_promoted") is not False:
        failures.append("TOPOLOGY_PROMOTION_FIREWALL")
    if s4_replay_comparison.get("schema_id") != "IG_G3_S4_CERTIFICATION_REPLAY_COMPARISON_V1":
        failures.append("S4_REPLAY_SCHEMA")
    if s4_replay_comparison.get("status") != "PASS" or s4_replay_comparison.get("certification") != "CERTIFIED_PASS":
        failures.append("S4_REPLAY_STATUS")
    if s4_replay_comparison.get("science_sha_equal") is not True:
        failures.append("S4_REPLAY_SCIENCE_MISMATCH")
    if s4_replay_comparison.get("stable_scientific_payload_equal_ignoring_declared_non_science_runtime_fields") is not True:
        failures.append("S4_REPLAY_STABLE_PAYLOAD_MISMATCH")
    if s4_replay_comparison.get("primary_science_sha256") != spec["authority"]["g3_s4_science_sha256"]:
        failures.append("S4_REPLAY_PRIMARY_SCIENCE")
    if s4_replay_comparison.get("cold_science_sha256") != spec["authority"]["g3_s4_science_sha256"]:
        failures.append("S4_REPLAY_COLD_SCIENCE")
    if relation_spec().get("spec_sha256") != spec["g2_relation_spec_sha256"]:
        failures.append("RELATION_SPEC")
    out = {
        "schema_id": "IG_G3_S5_AUTHORITY_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "g3_s4_science_sha256": s4_result.get("science_sha256"),
        "g3_s4_cold_replay_certified": not failures,
        "g3_graduated": False,
        "topology_promoted": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G3S5Error("G3:S5 authority verification failed: " + ",".join(failures))
    return out


def _caps_class_summary(values: Sequence[tuple[int, ...]]) -> dict[str, Any]:
    classes = Counter(values)
    size_hist = Counter(classes.values())
    return {
        "record_count": len(values),
        "caps7_class_count": len(classes),
        "multi_record_caps7_class_count": sum(1 for n in classes.values() if n > 1),
        "records_in_multi_record_caps7_classes": sum(n for n in classes.values() if n > 1),
        "largest_caps7_class_size": max(classes.values()) if classes else 0,
        "caps7_class_size_histogram": [
            {"record_count": int(size), "caps7_class_count": int(count)}
            for size, count in sorted(size_hist.items())
        ],
    }


def certify_candidate_caps7_census(*, pair_basis: Mapping[str, Any], recursive_rows_path: str | Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if pair_basis.get("schema_id") != "IG_G3_S4_COMPLETE_PAIR_CANDIDATE_BASIS_V1" or pair_basis.get("status") != "PASS":
        raise G3S5Error("G3:S5 requires certified S4 pair basis")
    pair_caps: list[tuple[int, ...]] = []
    for row in pair_basis.get("rows", []):
        states = row.get("public_output_caps7_set", [])
        if len(states) != 1:
            failures.append({"scope": "PAIR", "key_index": row.get("key_index"), "reason": "NOT_SINGLETON_CAPS7"})
            continue
        pair_caps.append(_caps(states[0], "pair CAPS7"))

    with gzip.open(Path(recursive_rows_path), "rt", encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("schema_id") != "IG_G3_S4_RECURSIVE_P3_MATERIALISATION_INDEX_V1":
        raise G3S5Error("bad S4 recursive row index schema")
    rows = obj.get("rows", [])
    recursive_caps: list[tuple[int, ...]] = []
    relation_cardinality_by_caps: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for row in rows:
        sig = row.get("signature", {})
        expected_hash = sig.get("science_sha256")
        if canonical_sha256({k: v for k, v in sig.items() if k != "science_sha256"}) != expected_hash:
            raise G3S5Error("S4 recursive signature hash mismatch inside S5 census")
        states = sig.get("public_output_caps7_set", [])
        if len(states) != 1:
            failures.append({"scope": "RECURSIVE", "first": row.get("first_key_index"), "second": row.get("second_key_index"), "reason": "NOT_SINGLETON_CAPS7"})
            continue
        caps = _caps(states[0], "recursive CAPS7")
        recursive_caps.append(caps)
        relation_cardinality_by_caps[caps][int(sig.get("recursive_relation_cardinality", -1))] += 1
        if not sig.get("relation_nonempty"):
            failures.append({"scope": "RECURSIVE", "reason": "EMPTY_RELATION"})
        if int(sig.get("caps7_write_mismatch_count", -1)) != 0:
            failures.append({"scope": "RECURSIVE", "reason": "CAPS7_WRITE_MISMATCH"})
        if _caps(sig.get("expected_output_caps7", []), "expected recursive CAPS7") != caps:
            failures.append({"scope": "RECURSIVE", "reason": "EXPECTED_CAPS7_MISMATCH"})

    pair_summary = _caps_class_summary(pair_caps)
    recursive_summary = _caps_class_summary(recursive_caps)
    out = {
        "schema_id": "IG_G3_S5_S4_CANDIDATE_CAPS7_CENSUS_V1",
        "status": "PASS" if not failures and len(pair_caps) == 62 and len(recursive_caps) == 3844 else "FAIL",
        "pair_candidate_summary": pair_summary,
        "recursive_p3_candidate_summary": recursive_summary,
        "failure_count": len(failures),
        "failure_examples": failures[:32],
        "complete_exact_relation_maps_to_single_caps7": not failures,
        "exact_relation_cardinality_is_not_part_of_descriptor": True,
        "relation_cardinality_histograms_by_caps7_sha256": canonical_sha256([
            [list(caps), [[int(card), int(count)] for card, count in sorted(hist.items())]]
            for caps, hist in sorted(relation_cardinality_by_caps.items())
        ]),
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def verify_recursive_reserve_basis(reserve_basis: Mapping[str, Any]) -> dict[str, Any]:
    if reserve_basis.get("schema_id") != "IG_G3_S4_RECURSIVE_RESERVE_OPERATOR_BASIS_V1":
        raise G3S5Error("bad S4 recursive reserve basis schema")
    rows = reserve_basis.get("rows", [])
    failures: list[dict[str, Any]] = []
    ops = []
    reserve_checks = 0
    exact_outputs = 0
    for row in rows:
        op = tuple(map(int, row.get("operator", [])))
        if len(op) != 2:
            failures.append({"reason": "BAD_OPERATOR"})
            continue
        ops.append(op)
        immediate = row.get("immediate_signature", {})
        cap = row.get("reserve_capability", {})
        if immediate.get("relation_nonempty") is not True or int(immediate.get("caps7_write_mismatch_count", -1)) != 0:
            failures.append({"operator": list(op), "reason": "IMMEDIATE_RELATION_FAILURE"})
        if len(immediate.get("public_output_caps7_set", [])) != 1:
            failures.append({"operator": list(op), "reason": "IMMEDIATE_NOT_SINGLETON_CAPS7"})
        if cap.get("status") != "PASS" or int(cap.get("failure_count", -1)) != 0:
            failures.append({"operator": list(op), "reason": "RESERVE_CAPABILITY_FAILURE"})
        reserve_checks += int(cap.get("reserve_checks", 0))
        exact_outputs += int(cap.get("recursive_output_count", 0))
    unique_ops = sorted(set(ops))
    out = {
        "schema_id": "IG_G3_S5_S4_RECURSIVE_RESERVE_FACTORISATION_V1",
        "status": "PASS" if not failures and len(rows) == 31 and len(unique_ops) == 31 else "FAIL",
        "operator_basis_count": len(unique_ops),
        "basis_row_count": len(rows),
        "exact_recursive_outputs_checked": exact_outputs,
        "exact_reservation_relations_checked": reserve_checks,
        "failure_count": len(failures),
        "failure_examples": failures[:32],
        "law": "For every exact branch checked, reserve_t maps CAPS7 f to f-e_t whenever f_t>0; exact owner/branch multiplicity is not observed.",
        "operator_basis": [list(x) for x in unique_ops],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def implementation_read_surface_audit() -> dict[str, Any]:
    """Static audit that the promoted abstract read/write law does not select by hidden G3 structure."""
    from . import g2_relation

    reserve_src = inspect.getsource(g2_relation._reserve_external_relation_with_witness)
    compose_src = inspect.getsource(g2_relation.compose_binary_relation)
    desc_src = inspect.getsource(descriptor_from_exact_relation)
    failures: list[str] = []
    if ".skin" in reserve_src or ".skin" in compose_src:
        failures.append("SKIN_READ_IN_TRANSITION_CORE")
    if "total_caps" not in reserve_src or "total_caps" not in compose_src:
        failures.append("TOTAL_CAPS_NOT_PRESENT_IN_TRANSITION_CORE")
    if "total_caps" not in desc_src:
        failures.append("DESCRIPTOR_NOT_TOTAL_CAPS_BASED")
    for token in ("skin", "children", "construction_digest", "top_edges", "owner", "topology"):
        if token in desc_src:
            failures.append("DESCRIPTOR_FORBIDDEN_READ:" + token)
    out = {
        "schema_id": "IG_G3_S5_IMPLEMENTATION_READ_SURFACE_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "descriptor_reads_complete_relation_branch_total_caps_only": not any(x.startswith("DESCRIPTOR_") for x in failures),
        "abstract_enabledness_fields": ["CAPS7", "frozen_31_operator_basis", "selected_endpoint_types"],
        "abstract_write_fields": ["left.CAPS7", "right.CAPS7", "selected_endpoint_types"],
        "construction_identity_role": "EXACT_SET_DEDUP_ONLY_NOT_ABSTRACT_SELECTOR",
        "hidden_topology_read": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def caps7_factorisation_argument(*, candidate_census: Mapping[str, Any], reserve_factorisation: Mapping[str, Any], implementation_audit: Mapping[str, Any]) -> dict[str, Any]:
    passed = all(x.get("status") == "PASS" for x in (candidate_census, reserve_factorisation, implementation_audit))
    out = {
        "schema_id": "IG_G3_S5_CAPS7_FACTORISATION_ARGUMENT_V1",
        "status": "PASS" if passed else "FAIL",
        "state": "CAPS7 f=(f0,...,f6) in N^7 attached to a complete exact G3 candidate relation only when every exact branch has the same f.",
        "reserve_read_write": "reserve_t is enabled iff f_t>0 and every exact relation-valued branch maps to f-e_t; hidden owner choice may change exact multiplicity but not the abstract successor.",
        "binary_read_write": "For a frozen earned bridge operator (a,b), one-step composition is enabled from public resource state when f_a>0 and g_b>0, and every exact output branch maps to f+g-e_a-e_b.",
        "s4_evidence": "S4 certified 62 complete pair relations and all 3,844 registered recursive P3 public materialisations as nonempty singleton-CAPS7 exact relations with zero CAPS7 write mismatches, plus a 31-operator actual recursive reserve-capability basis.",
        "collision_consequence": "Distinct construction/context histories that collapse to the same CAPS7 are intentionally one abstract state for this observer; exact relation cardinality, owner witnesses, construction identity and topology remain outside the observer.",
        "implementation_consequence": "The public transition core computes abstract legality/write from capacities and the frozen operator basis; hidden G3 topology is not a selector.",
        "scope": "FROZEN_ONE_STEP_RELATION_VALUED_G3_CANDIDATE_GRAMMAR_CERTIFIED_BY_S4; arbitrary recursive closure remains S6.",
        "not_claimed": ["minimality", "raw exact relation identity", "exact branch multiplicity identity", "pair+pair closure", "unbounded recursive closure", "topology erasure", "G3 graduation"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def finalize_s5_result(*, authority: Mapping[str, Any], candidate_census: Mapping[str, Any], reserve_factorisation: Mapping[str, Any], implementation_audit: Mapping[str, Any], factorisation_argument: Mapping[str, Any]) -> dict[str, Any]:
    spec = s5_spec()
    passed = all(x.get("status") == "PASS" for x in (authority, candidate_census, reserve_factorisation, implementation_audit, factorisation_argument))
    pair = candidate_census.get("pair_candidate_summary", {})
    rec = candidate_census.get("recursive_p3_candidate_summary", {})
    out = {
        "schema_id": "IG_G3_S5_FINITE_READ_WRITE_DESCRIPTOR_RESULT_V1",
        "schema_version": "1.0.0",
        "date": "2026-09-03",
        "stage_ref": "G3:S5",
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "classification": "G3_CAPS7_FINITE_READ_WRITE_DESCRIPTOR_EARNED_ON_FROZEN_RELATION_VALUED_GRAMMAR_S6_UNLOCKED" if passed else "G3_FINITE_DESCRIPTOR_GATE_FAILED_S6_LOCKED",
        "candidate_descriptor": {
            "schema_id": "IG_G3_CAPS7_READ_WRITE_STATE_V1",
            "name": "CAPS7",
            "coordinate_count": 7,
            "coordinate_domain": "N",
            "fields": ["total_free_by_type[0..6]"],
            "relation_semantics": "complete exact relation quotiented only when every branch has one common CAPS7 vector",
        },
        "observer_scope": spec["frozen_observer"],
        "authority": dict(authority),
        "candidate_caps7_census": dict(candidate_census),
        "recursive_reserve_factorisation": dict(reserve_factorisation),
        "implementation_read_surface_audit": dict(implementation_audit),
        "factorisation_argument": dict(factorisation_argument),
        "compression_summary": {
            "s4_pair_public_context_keys": pair.get("record_count"),
            "s4_pair_caps7_classes": pair.get("caps7_class_count"),
            "s4_recursive_p3_public_contexts": rec.get("record_count"),
            "s4_recursive_p3_caps7_classes": rec.get("caps7_class_count"),
            "same_caps7_recursive_collision_classes": rec.get("multi_record_caps7_class_count"),
            "minimality_claimed": False,
        },
        "g3_s6_unlocked": passed,
        "g3_graduated": False,
        "topology_promoted": False,
        "nonclaims": spec["nonclaims"],
        "g3_s5_spec_sha256": spec["science_sha256"],
        "g2_relation_spec_sha256": relation_spec()["spec_sha256"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def stable_science_core(result: Mapping[str, Any]) -> dict[str, Any]:
    volatile = {"source_sha256", "source_version", "registry_sha256", "execution_metadata", "native_engine_science_sha256"}
    return {k: v for k, v in result.items() if k not in volatile and k != "science_sha256"}
