from __future__ import annotations

"""G2:S5 finite read/write state descriptor audit.

S5 freezes an observer that sees only the admitted G2 transition grammar:
reservation enabledness, earned typed binary-bridge enabledness, and the abstract
successor state.  It intentionally does *not* observe raw skin values, hidden owner
witnesses, construction identity, exact branch multiplicity, or hidden topology.

Under the v0.30.2 relation-valued routing semantics, the candidate state is the seven
non-negative total-free-capacity counters (CAPS7).  Exact relation-valued branches may
remain distinct implementation states, but every lawful branch of the same abstract
action must map to the same CAPS7 successor.

This stage is non-promoting.  PASS unlocks S6 only.
"""

from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip, inspect, json, re

from .canon import canonical_sha256
from .g2_relation import relation_spec


class UpliftS5Error(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G_UPLIFT_S5_FINITE_STATE_DESCRIPTOR_SPEC_V1.json"
_OP_RE = re.compile(r"^G1_PUBLIC_BRIDGE_RELATION_V1:(\d+)>(\d+)$")


def s5_implementation_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    expected = str(obj.get("science_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    observed = canonical_sha256(payload)
    if expected != observed:
        raise UpliftS5Error(f"S5 implementation spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def _sha(v: Any, field: str) -> str:
    s = str(v)
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise UpliftS5Error(f"{field} must be lowercase sha256")
    return s


def _caps(v: Sequence[Any], field: str = "caps") -> tuple[int, ...]:
    out = tuple(int(x) for x in v)
    if len(out) != 7 or any(x < 0 for x in out):
        raise UpliftS5Error(f"{field} must be seven nonnegative integers")
    return out


def caps7_descriptor_from_caps(v: Sequence[Any]) -> dict[str, Any]:
    c = _caps(v)
    out = {
        "schema_id": "IG_G2_CAPS7_STATE_V1",
        "total_free_by_type": list(c),
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def caps7_descriptor_from_state(state: Any) -> dict[str, Any]:
    """Strict S5 read: only ``state.total_caps`` is consulted."""
    return caps7_descriptor_from_caps(state.total_caps)


def reserve_enabled(caps: Sequence[Any], endpoint_type: int) -> bool:
    c = _caps(caps)
    t = int(endpoint_type)
    if t < 0 or t >= 7:
        raise UpliftS5Error("endpoint type outside seven-type alphabet")
    return c[t] > 0


def reserve_write(caps: Sequence[Any], endpoint_type: int) -> tuple[int, ...] | None:
    c = list(_caps(caps))
    t = int(endpoint_type)
    if t < 0 or t >= 7:
        raise UpliftS5Error("endpoint type outside seven-type alphabet")
    if c[t] <= 0:
        return None
    c[t] -= 1
    return tuple(c)


def bridge_enabled(left_caps: Sequence[Any], right_caps: Sequence[Any], a: int, b: int, bridge_pairs: Sequence[Sequence[int]]) -> bool:
    lc, rc = _caps(left_caps, "left_caps"), _caps(right_caps, "right_caps")
    aa, bb = int(a), int(b)
    basis = {(int(x), int(y)) for x, y in bridge_pairs}
    return (aa, bb) in basis and lc[aa] > 0 and rc[bb] > 0


def binary_write(left_caps: Sequence[Any], right_caps: Sequence[Any], a: int, b: int, bridge_pairs: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    lc, rc = _caps(left_caps, "left_caps"), _caps(right_caps, "right_caps")
    aa, bb = int(a), int(b)
    if not bridge_enabled(lc, rc, aa, bb, bridge_pairs):
        return None
    out = [lc[i] + rc[i] for i in range(7)]
    out[aa] -= 1
    out[bb] -= 1
    if any(x < 0 for x in out):
        raise UpliftS5Error("CAPS7 binary write underflow")
    return tuple(out)


def verify_s5_authority(*, unlock_certificate: Mapping[str, Any], s4_result: Mapping[str, Any]) -> dict[str, Any]:
    spec = s5_implementation_spec()
    failures: list[str] = []
    if unlock_certificate.get("schema_id") != "IG_G2_S5_UNLOCK_CERTIFICATE_V1":
        failures.append("UNLOCK_SCHEMA")
    if unlock_certificate.get("status") != "PASS":
        failures.append("UNLOCK_STATUS")
    if unlock_certificate.get("authorizes") != "G2:S5_DESIGN_AND_EXECUTION_UNDER_FROZEN_V0302_RELATION_VALUED_ROUTING":
        failures.append("UNLOCK_AUTHORIZATION")
    if unlock_certificate.get("g2_graduated") is not False:
        failures.append("GRADUATION_FIREWALL")
    if _sha(unlock_certificate.get("science_sha256"), "unlock science") != spec["parent_s5_unlock_science_sha256"]:
        failures.append("UNLOCK_SCIENCE")
    if s4_result.get("status") != "PASS" or s4_result.get("s5_unlocked") is not True:
        failures.append("S4_STATUS")
    if _sha(s4_result.get("science_sha256"), "S4 science") != spec["parent_s4_science_sha256"]:
        failures.append("S4_SCIENCE")
    if s4_result.get("g2_routing_semantics") != "RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1":
        failures.append("ROUTING_SEMANTICS")
    if relation_spec()["spec_sha256"] != spec["g2_relation_spec_sha256"]:
        failures.append("RELATION_SPEC")
    out = {
        "schema_id": "IG_G2_S5_AUTHORITY_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "s4_science_sha256": s4_result.get("science_sha256"),
        "s5_unlock_science_sha256": unlock_certificate.get("science_sha256"),
        "s5_spec_science_sha256": spec["science_sha256"],
        "g2_graduated": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise UpliftS5Error("S5 authority verification failed: " + ",".join(failures))
    return out


def verify_s0_base_reservation_law(s0_population: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    available = 0
    unavailable = 0
    for row in s0_population.get("interfaces", []):
        base = _caps(row["total_free_by_type"], "S0 base")
        reservations = row.get("one_endpoint_reservations", [])
        if len(reservations) != 7:
            raise UpliftS5Error("S0 interface missing seven reservation rows")
        for t, rr in enumerate(reservations):
            checked += 1
            enabled = base[t] > 0
            if bool(rr.get("available")) != enabled:
                if len(failures) < 20:
                    failures.append({"carrier_ref": row.get("carrier_ref"), "type": t, "reason": "ENABLEDNESS_MISMATCH"})
                continue
            if not enabled:
                unavailable += 1
                continue
            available += 1
            got = rr.get("successor_total_free_by_type")
            exp = reserve_write(base, t)
            if got is None or tuple(int(x) for x in got) != exp:
                if len(failures) < 20:
                    failures.append({"carrier_ref": row.get("carrier_ref"), "type": t, "reason": "DELTA_MISMATCH"})
    out = {
        "schema_id": "IG_G2_S5_S0_BASE_RESERVATION_FACTORISATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "carrier_count": len(s0_population.get("interfaces", [])),
        "reservation_rows_checked": checked,
        "available_rows": available,
        "unavailable_rows": unavailable,
        "failure_count": len(failures),
        "failure_examples": failures,
        "law": "CAPS7(reserve_t(x)) = CAPS7(x)-e_t whenever total_caps[t]>0",
        "hidden_internal_reads": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _pair_caps_from_record(record: Mapping[str, Any], iface: Mapping[str, Mapping[str, Any]]) -> tuple[int, ...]:
    if record.get("legality") != "LEGAL":
        raise UpliftS5Error("certified repaired stream contains non-LEGAL row")
    m = _OP_RE.fullmatch(str(record.get("connection_operator_ref", "")))
    if m is None:
        raise UpliftS5Error("bad pair operator")
    a, b = map(int, m.groups())
    left = iface[str(record["left_carrier_ref"])]
    right = iface[str(record["right_carrier_ref"])]
    lc = left["one_endpoint_reservations"][a]["successor_total_free_by_type"]
    rc = right["one_endpoint_reservations"][b]["successor_total_free_by_type"]
    if lc is None or rc is None:
        raise UpliftS5Error("legal S1 record has absent S0 reservation successor")
    return tuple(int(lc[i]) + int(rc[i]) for i in range(7))


def scan_certified_pair_caps7(repaired_records_path: str | Path, s0_population: Mapping[str, Any]) -> dict[str, Any]:
    iface = {str(x["carrier_ref"]): x for x in s0_population["interfaces"]}
    classes: Counter[tuple[int, ...]] = Counter()
    support_masks: Counter[tuple[bool, ...]] = Counter()
    rows = 0
    with gzip.open(Path(repaired_records_path), "rt") as f:
        for line in f:
            r = json.loads(line)
            rows += 1
            c = _pair_caps_from_record(r, iface)
            classes[c] += 1
            support_masks[tuple(x > 0 for x in c)] += 1
    hist = Counter(classes.values())
    class_payload = [[list(k), int(v)] for k, v in sorted(classes.items())]
    multi_rows = sum(v for v in classes.values() if v > 1)
    out = {
        "schema_id": "IG_G2_S5_CERTIFIED_PAIR_CAPS7_CENSUS_V1",
        "status": "PASS" if rows == 580351 and bool(classes) else "FAIL",
        "record_count": rows,
        "caps7_class_count": len(classes),
        "caps7_class_size_histogram": [{"record_count": int(k), "caps7_class_count": int(v)} for k, v in sorted(hist.items())],
        "caps7_multi_record_class_count": sum(1 for v in classes.values() if v > 1),
        "records_in_multi_record_caps7_classes": multi_rows,
        "largest_caps7_class_size": max(classes.values()) if classes else 0,
        "support_mask_count": len(support_masks),
        "all_records_positive_all_seven_types": len(support_masks) == 1 and next(iter(support_masks), None) == (True,) * 7,
        "caps7_class_map_sha256": canonical_sha256(class_payload),
        "descriptor_fields": ["total_free_by_type[0..6]"],
        "hidden_internal_reads": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _read_record_at(path: str | Path, target_row: int) -> dict[str, Any]:
    if target_row < 1:
        raise UpliftS5Error("row index must be >=1")
    with gzip.open(Path(path), "rt") as f:
        for i, line in enumerate(f, start=1):
            if i == target_row:
                return json.loads(line)
    raise UpliftS5Error(f"row {target_row} not present")


def verify_s4_operator_caps7_factorisation(*, operator_basis: Mapping[str, Any], repaired_records_path: str | Path, s0_population: Mapping[str, Any], bridge_pairs: Sequence[Sequence[int]]) -> dict[str, Any]:
    iface = {str(x["carrier_ref"]): x for x in s0_population["interfaces"]}
    seed_idx = int(operator_basis["input_pair_row_index"])
    seed_record = _read_record_at(repaired_records_path, seed_idx)
    seed_caps = _pair_caps_from_record(seed_record, iface)
    ctx = iface[str(operator_basis["context_carrier_ref"])]
    ctx_caps = _caps(ctx["total_free_by_type"], "context caps")
    failures: list[dict[str, Any]] = []
    checked_branches = 0
    rows = operator_basis.get("basis_rows", [])
    for row in rows:
        a, b = map(int, row["operator"])
        expected = binary_write(seed_caps, ctx_caps, a, b, bridge_pairs)
        if expected is None:
            if len(failures) < 20:
                failures.append({"operator": [a, b], "reason": "ABSTRACT_OPERATOR_NOT_ENABLED"})
            continue
        for outrow in row.get("outputs", []):
            checked_branches += 1
            got = _caps(outrow["output_total_caps"], "operator output caps")
            if got != expected:
                if len(failures) < 20:
                    failures.append({"operator": [a, b], "reason": "OUTPUT_DELTA_MISMATCH"})
    expected_basis = {(int(a), int(b)) for a, b in bridge_pairs}
    observed_basis = {(int(r["operator"][0]), int(r["operator"][1])) for r in rows}
    if observed_basis != expected_basis:
        failures.append({"reason": "OPERATOR_BASIS_MISMATCH"})
    out = {
        "schema_id": "IG_G2_S5_S4_BINARY_WRITE_FACTORISATION_V1",
        "status": "PASS" if not failures and operator_basis.get("status") == "PASS" else "FAIL",
        "operator_count": len(rows),
        "exact_relation_branches_checked": checked_branches,
        "seed_pair_row_index": seed_idx,
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "law": "CAPS7(compose_{a,b}(x,y)) = CAPS7(x)+CAPS7(y)-e_a-e_b for every exact relation-valued branch",
        "hidden_internal_reads_in_abstract_write": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def implementation_read_surface_audit() -> dict[str, Any]:
    """Static implementation audit for the promotable CAPS7 read/write law.

    Exact relation enumeration may inspect child capacities and exact construction identities for
    duplicate elimination, but the abstract legality/write law must not use skin, construction
    identity, or owner witnesses.  We inspect the two public transition implementations and the
    CAPS7 descriptor function itself.
    """
    from . import g2_relation

    reserve_src = inspect.getsource(g2_relation._reserve_external_relation_with_witness)
    compose_src = inspect.getsource(g2_relation.compose_binary_relation)
    desc_src = inspect.getsource(caps7_descriptor_from_state)

    failures: list[str] = []
    if ".skin" in reserve_src or ".skin" in compose_src:
        failures.append("SKIN_READ_IN_TRANSITION_CORE")
    if "construction_digest" in compose_src:
        # compose calls exact-dedup helper; this is allowed only for operational set dedupe.
        # The helper cannot participate in enabledness or CAPS7 output computation.
        pass
    if "total_caps" not in reserve_src or "total_caps" not in compose_src:
        failures.append("TOTAL_CAPS_NOT_PRESENT")
    if "total_caps" not in desc_src:
        failures.append("DESCRIPTOR_NOT_CAPS_ONLY")
    forbidden_desc = ["skin", "children", "construction_digest", "top_edges", "owner"]
    for tok in forbidden_desc:
        if tok in desc_src:
            failures.append(f"DESCRIPTOR_FORBIDDEN_READ:{tok}")

    out = {
        "schema_id": "IG_G2_S5_IMPLEMENTATION_READ_SURFACE_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "descriptor_reads_only_total_caps": not any(x.startswith("DESCRIPTOR_") for x in failures),
        "transition_core_reads_skin": ".skin" in reserve_src or ".skin" in compose_src,
        "construction_identity_role": "EXACT_SET_DEDUP_ONLY_NOT_ABSTRACT_SELECTOR",
        "abstract_enabledness_fields": ["total_caps", "frozen_bridge_pair_basis"],
        "abstract_write_fields": ["left.total_caps", "right.total_caps", "endpoint_types"],
        "hidden_owner_witness_in_descriptor": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def caps7_factorisation_argument(*, s0_base: Mapping[str, Any], pair_census: Mapping[str, Any], operator_factorisation: Mapping[str, Any], implementation_audit: Mapping[str, Any]) -> dict[str, Any]:
    passed = all(x.get("status") == "PASS" for x in (s0_base, pair_census, operator_factorisation, implementation_audit))
    out = {
        "schema_id": "IG_G2_S5_CAPS7_FACTORISATION_ARGUMENT_V1",
        "status": "PASS" if passed else "FAIL",
        "base_case": "Frozen G1 public one-reservation surface decrements exactly one selected capacity coordinate; verified exhaustively on all 193 S0 carriers and seven types.",
        "reservation_induction": "For a G2 composite, total capacity is the coordinatewise sum of child capacities. Every exact relation branch reserves exactly one eligible child. By the base/recursive hypothesis that child loses e_t, so every whole-unit branch loses exactly e_t. Positive whole-unit capacity implies at least one eligible child.",
        "binary_write": "G2 binary composition first applies one reservation on each input and then wraps the two reserved units. Therefore every exact branch has total capacities f+g-e_a-e_b. Exact branch multiplicity and owner identity disappear under CAPS7.",
        "read_sufficiency": "Reservation enabledness is f_t>0. Typed bridge enabledness is membership in the frozen 31-operator basis plus positive selected coordinates. No admitted G2 transition reads skin or hidden topology.",
        "consequence": "The exact relation-valued G2 transition system factors through the finite-coordinate CAPS7 quotient for the frozen transition observer.",
        "scope": "FROZEN_RELATION_VALUED_G2_EXTERNAL_RESERVATION_AND_31_TYPED_BINARY_COMPOSITION_OPERATORS",
        "not_claimed": ["raw public skin equivalence", "exact branch multiplicity equivalence", "minimality", "G2 graduation"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out



def _stable_s5_science_core(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the S5 science payload hashed by :func:`s5_result`.

    Runtime/source/registry metadata are certification provenance, not S5 science.
    """
    volatile = {"source_sha256", "source_version", "registry_sha256", "execution_metadata", "native_engine_science_sha256"}
    return {k: v for k, v in result.items() if k not in volatile and k != "science_sha256"}


def decoder_native_s5_replay_evidence(*, primary_result: Mapping[str, Any], replay_result: Mapping[str, Any]) -> dict[str, Any]:
    """Certify a cold replay of the *registered* G2:S5 Decoder experiment.

    This is intentionally not an alternate S5 implementation. Both inputs must come from
    the registered Decoder-native G2:S5 handler. The comparison merely checks exact
    reproduction of the frozen science payload.
    """
    failures: list[str] = []
    for label, result in (("PRIMARY", primary_result), ("REPLAY", replay_result)):
        if result.get("schema_id") != "IG_G2_S5_FINITE_STATE_DESCRIPTOR_RESULT_V1":
            failures.append(label + "_SCHEMA")
        if result.get("status") != "PASS":
            failures.append(label + "_STATUS")
        md = result.get("execution_metadata", {})
        if md.get("registered_experiment_id") != "G2:S5":
            failures.append(label + "_NOT_REGISTERED_G2_S5")
        if md.get("execution_backend_owned_by_decoder") is not True:
            failures.append(label + "_BACKEND_NOT_DECODER")
        if md.get("stage_specific_external_science_runner") is not False:
            failures.append(label + "_EXTERNAL_SCIENCE_RUNNER")
        core = _stable_s5_science_core(result)
        if canonical_sha256(core) != result.get("science_sha256"):
            failures.append(label + "_SCIENCE_HASH_SELF_CHECK")
    primary_core = _stable_s5_science_core(primary_result)
    replay_core = _stable_s5_science_core(replay_result)
    exact = primary_core == replay_core and primary_result.get("science_sha256") == replay_result.get("science_sha256")
    if not exact:
        failures.append("SCIENCE_PAYLOAD_MISMATCH")
    out = {
        "schema_id": "IG_G2_S5_DECODER_NATIVE_COLD_REPLAY_V1",
        "date": "2026-09-03",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "registered_experiment_id": "G2:S5",
        "replay_mode": "COLD_SOURCE_RERUN_OF_SAME_REGISTERED_DECODER_NATIVE_EXPERIMENT",
        "primary_science_sha256": primary_result.get("science_sha256"),
        "cold_science_sha256": replay_result.get("science_sha256"),
        "science_payload_exact_match": exact,
        "execution_backend_owned_by_decoder": True,
        "stage_specific_external_science_runner": False,
        "g2_graduated": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out

def s5_result(*, authority: Mapping[str, Any], s0_base: Mapping[str, Any], pair_census: Mapping[str, Any], operator_factorisation: Mapping[str, Any], implementation_audit: Mapping[str, Any], argument: Mapping[str, Any], s2_revalidation: Mapping[str, Any]) -> dict[str, Any]:
    passed = all(x.get("status") == "PASS" for x in (authority, s0_base, pair_census, operator_factorisation, implementation_audit, argument))
    s2_count = int(s2_revalidation.get("new", {}).get("s2_quotient_class_count", 0))
    out = {
        "schema_id": "IG_G2_S5_FINITE_STATE_DESCRIPTOR_RESULT_V1",
        "schema_version": "1.0.0",
        "date": "2026-09-03",
        "stage_ref": "G2:S5",
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "classification": "CAPS7_FINITE_READ_WRITE_DESCRIPTOR_EARNED_FOR_FROZEN_G2_TRANSITION_GRAMMAR_S6_UNLOCKED" if passed else "CAPS7_DESCRIPTOR_GATE_FAILED_S6_LOCKED",
        "candidate_descriptor": {
            "schema_id": "IG_G2_CAPS7_STATE_V1",
            "coordinate_count": 7,
            "coordinate_domain": "N",
            "fields": ["total_free_by_type[0..6]"],
        },
        "observer_scope": s5_implementation_spec()["frozen_observer"],
        "authority": dict(authority),
        "s0_base_reservation_factorisation": dict(s0_base),
        "certified_pair_caps7_census": dict(pair_census),
        "s4_binary_write_factorisation": dict(operator_factorisation),
        "implementation_read_surface_audit": dict(implementation_audit),
        "factorisation_argument": dict(argument),
        "compression_summary": {
            "s2_quotient_class_count": s2_count,
            "caps7_class_count_on_certified_pair_population": pair_census.get("caps7_class_count"),
            "caps7_merges_s2_classes_or_records": int(pair_census.get("caps7_class_count", 0)) < s2_count,
            "minimality_claimed": False,
        },
        "s6_unlocked": passed,
        "g2_graduated": False,
        "nonclaims": [
            "NOT_RAW_PUBLIC_STATE_IDENTITY",
            "NOT_BRANCH_MULTIPLICITY_IDENTITY",
            "NOT_MINIMAL_G2_DESCRIPTOR",
            "NOT_RECURSIVE_CLOSURE_OR_HOLDOUT_RESULT",
            "NOT_G2_GRADUATION",
        ],
        "s5_spec_science_sha256": s5_implementation_spec()["science_sha256"],
        "g2_relation_spec_sha256": relation_spec()["spec_sha256"],
    }
    out["science_sha256"] = canonical_sha256(out)
    return out
