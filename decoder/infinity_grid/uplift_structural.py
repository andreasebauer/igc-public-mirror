from __future__ import annotations

"""First-class G-uplift structural primitives for Decoder v0.29.2.

This module implements only the frozen G2:S0 and G2:S1 stages from the
v0.29.0 Phase-0 architecture contract:

* S0 extracts the declared *public* one-endpoint G1 carrier interface.
* S1 performs a deterministic pair-connection census using only S0 data and
  the already-earned G1 bridge-type catalogue.

It deliberately does not inspect carrier internals (children, owner graphs,
internal edges, ancestry, topology, etc.) and cannot graduate G2.
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import gzip
import hashlib
import json

from .canon import canonical_sha256
from .uplift_architecture import uplift_contract, UpliftArchitectureError


class UpliftStructuralError(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G_UPLIFT_S0_S1_IMPLEMENTATION_SPEC_V2.json"
_REQUIRED_RECORD_FIELDS = tuple(uplift_contract()["pair_connection_record_required_fields"])


def _load_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    expected = obj.get("spec_sha256")
    payload = {k: v for k, v in obj.items() if k != "spec_sha256"}
    observed = canonical_sha256(payload)
    if expected != observed:
        raise UpliftStructuralError(f"S0/S1 implementation spec identity mismatch: expected {expected}, observed {observed}")
    if obj.get("architecture_contract_sha256") != uplift_contract()["contract_sha256"]:
        raise UpliftStructuralError("S0/S1 implementation spec is not bound to the frozen uplift architecture")
    return obj


def implementation_spec() -> dict[str, Any]:
    return _load_spec()


def _require_sha256(value: str, *, field: str) -> str:
    value = str(value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise UpliftStructuralError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _semantic_interface_payload(*, boundary_skin: str, total_caps: Sequence[int], reservations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_id": "IG_G1_PUBLIC_ONE_ENDPOINT_INTERFACE_SEMANTICS_V1",
        "boundary_resource_skin_sha256": str(boundary_skin),
        "total_free_by_type": [int(x) for x in total_caps],
        "one_endpoint_reservations": [
            {
                "endpoint_type": int(r["endpoint_type"]),
                "available": bool(r["available"]),
                "successor_boundary_resource_skin_sha256": r.get("successor_boundary_resource_skin_sha256"),
                "successor_total_free_by_type": r.get("successor_total_free_by_type"),
            }
            for r in reservations
        ],
        "scope": "ONE_EXTERNAL_ENDPOINT_RESERVATION_FOR_WHOLE_CARRIER_PAIR_CONNECTION",
    }


def extract_carrier_interface(
    carrier: Any,
    *,
    source_stage: str,
    source_authority_sha256: str,
) -> dict[str, Any]:
    """Extract the frozen S0 public interface from one complete G1 carrier.

    The only carrier reads allowed here are the earned public boundary/resource
    surface: ``construction_digest`` (provenance ref), ``skin``, ``total_caps``
    and the public ``reserve_external(type)`` operation.  The reservation witness
    is intentionally discarded so internal owner/path information never enters
    the uplift state.
    """
    if source_stage != "G1:R100":
        raise UpliftStructuralError(f"v1 S0 implementation is frozen to G1:R100, got {source_stage!r}")
    source_authority_sha256 = _require_sha256(source_authority_sha256, field="source_authority_sha256")
    carrier_ref = str(carrier.construction_digest)
    _require_sha256(carrier_ref, field="carrier construction_digest")
    boundary_skin = str(carrier.skin)
    _require_sha256(boundary_skin, field="carrier skin")
    total_caps = tuple(int(x) for x in carrier.total_caps)
    if len(total_caps) != 7 or any(x < 0 for x in total_caps):
        raise UpliftStructuralError("G1 public resource vector must contain seven nonnegative counters")

    reservations: list[dict[str, Any]] = []
    for t in range(7):
        if total_caps[t] <= 0:
            reservations.append({
                "endpoint_type": t,
                "available": False,
                "successor_boundary_resource_skin_sha256": None,
                "successor_total_free_by_type": None,
            })
            continue
        try:
            successor, _hidden_witness_discarded = carrier.reserve_external(t)
        except Exception as exc:  # public availability and public operation must agree fail-closed
            raise UpliftStructuralError(f"public reserve_external({t}) failed despite positive capacity") from exc
        succ_skin = str(successor.skin)
        _require_sha256(succ_skin, field=f"reserve_external({t}) successor skin")
        succ_caps = [int(x) for x in successor.total_caps]
        if len(succ_caps) != 7 or succ_caps[t] != total_caps[t] - 1:
            raise UpliftStructuralError(f"reserve_external({t}) does not consume exactly one public type-{t} endpoint")
        for j in range(7):
            expected = total_caps[j] - (1 if j == t else 0)
            if succ_caps[j] != expected:
                raise UpliftStructuralError(f"reserve_external({t}) changed public type-{j} counter unexpectedly")
        reservations.append({
            "endpoint_type": t,
            "available": True,
            "successor_boundary_resource_skin_sha256": succ_skin,
            "successor_total_free_by_type": succ_caps,
        })

    semantic = _semantic_interface_payload(boundary_skin=boundary_skin, total_caps=total_caps, reservations=reservations)
    interface_sha = canonical_sha256(semantic)
    row = {
        "schema_id": "IG_G_UPLIFT_CARRIER_INTERFACE_V1",
        "schema_version": "1.0.0",
        "uplift_layer": 2,
        "stage_ref": "G2:S0",
        "source_stage": source_stage,
        "carrier_ref": carrier_ref,
        "interface_sha256": interface_sha,
        "boundary_resource_skin_sha256": boundary_skin,
        "total_free_by_type": list(total_caps),
        "one_endpoint_reservations": reservations,
        "hidden_internal_reads": False,
        "public_operation_witnesses_retained": False,
        "source_authority_sha256": source_authority_sha256,
        "implementation_spec_sha256": implementation_spec()["spec_sha256"],
        "relabel_invariance_status": "PASS_BY_PREVIOUS_LAYER_CANONICAL_PUBLIC_BOUNDARY",
        "nonclaims": [
            "NOT_A_COMPLETE_UNBOUNDED_FUTURE_INTERFACE",
            "NOT_A_G2_STATE_DESCRIPTOR",
            "NOT_G2_GRADUATION",
        ],
    }
    row["science_sha256"] = canonical_sha256(row)
    return row


def extract_interface_population(
    carriers: Iterable[Any],
    *,
    source_stage: str,
    source_authority_sha256: str,
) -> dict[str, Any]:
    rows = [extract_carrier_interface(c, source_stage=source_stage, source_authority_sha256=source_authority_sha256) for c in carriers]
    rows.sort(key=lambda r: r["carrier_ref"])
    refs = [r["carrier_ref"] for r in rows]
    if len(refs) != len(set(refs)):
        raise UpliftStructuralError("S0 population requires unique carrier refs")
    classes: dict[str, int] = {}
    for r in rows:
        classes[r["interface_sha256"]] = classes.get(r["interface_sha256"], 0) + 1
    obj = {
        "schema_id": "IG_G_UPLIFT_S0_INTERFACE_POPULATION_V1",
        "schema_version": "1.0.0",
        "status": "PASS",
        "stage_ref": "G2:S0",
        "source_stage": source_stage,
        "carrier_count": len(rows),
        "interface_class_count": len(classes),
        "interface_class_histogram": [
            {"interface_sha256": k, "carrier_count": classes[k]} for k in sorted(classes)
        ],
        "interfaces": rows,
        "carrier_ref_set_sha256": canonical_sha256(refs),
        "source_authority_sha256": _require_sha256(source_authority_sha256, field="source_authority_sha256"),
        "implementation_spec_sha256": implementation_spec()["spec_sha256"],
        "g2_graduated": False,
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


def _reservation(interface: Mapping[str, Any], endpoint_type: int) -> Mapping[str, Any]:
    rows = list(interface["one_endpoint_reservations"])
    if len(rows) != 7:
        raise UpliftStructuralError("interface reservation surface is malformed")
    row = rows[int(endpoint_type)]
    if int(row["endpoint_type"]) != int(endpoint_type):
        raise UpliftStructuralError("interface reservation rows are not canonical type order")
    return row


def _canonical_slots(
    left: Mapping[str, Any], right: Mapping[str, Any], a: int, b: int
) -> tuple[Mapping[str, Any], Mapping[str, Any], int, int]:
    lk = (str(left["carrier_ref"]), str(left["interface_sha256"]))
    rk = (str(right["carrier_ref"]), str(right["interface_sha256"]))
    if rk < lk:
        return right, left, int(b), int(a)
    return left, right, int(a), int(b)


def _outcome_semantics(left: Mapping[str, Any], right: Mapping[str, Any], a: int, b: int, *, legal: bool, reason: str) -> dict[str, Any]:
    lr = _reservation(left, a)
    rr = _reservation(right, b)
    endpoints = [
        {
            "input_interface_sha256": str(left["interface_sha256"]),
            "reserved_type": int(a),
            "available": bool(lr["available"]),
            "successor_boundary_resource_skin_sha256": lr.get("successor_boundary_resource_skin_sha256"),
            "successor_total_free_by_type": lr.get("successor_total_free_by_type"),
        },
        {
            "input_interface_sha256": str(right["interface_sha256"]),
            "reserved_type": int(b),
            "available": bool(rr["available"]),
            "successor_boundary_resource_skin_sha256": rr.get("successor_boundary_resource_skin_sha256"),
            "successor_total_free_by_type": rr.get("successor_total_free_by_type"),
        },
    ]
    # Semantic assembly identity is invariant under swapping the two whole carriers.
    endpoints = sorted(endpoints, key=lambda x: canonical_sha256(x))
    payload: dict[str, Any] = {
        "schema_id": "IG_G_UPLIFT_S1_PAIR_OUTCOME_SEMANTICS_V1",
        "legal": bool(legal),
        "reason": str(reason),
        "endpoints": endpoints,
        "relation_kind": "G1_PUBLIC_TYPED_BRIDGE_RELATION" if legal else None,
    }
    if legal:
        left_caps = [int(x) for x in lr["successor_total_free_by_type"]]
        right_caps = [int(x) for x in rr["successor_total_free_by_type"]]
        payload["assembly_total_free_by_type"] = [left_caps[i] + right_caps[i] for i in range(7)]
    else:
        payload["assembly_total_free_by_type"] = None
    return payload


def pair_connection_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    endpoint_type_left: int,
    endpoint_type_right: int,
    *,
    bridge_pairs: Sequence[tuple[int, int]],
    source_authority_sha256: str,
) -> dict[str, Any]:
    source_authority_sha256 = _require_sha256(source_authority_sha256, field="source_authority_sha256")
    left, right, a, b = _canonical_slots(left, right, endpoint_type_left, endpoint_type_right)
    if (a, b) not in set((int(x), int(y)) for x, y in bridge_pairs):
        raise UpliftStructuralError(f"({a},{b}) is not an earned G1 bridge operator and is outside the S1 attempt catalogue")
    lr = _reservation(left, a)
    rr = _reservation(right, b)
    la, ra = bool(lr["available"]), bool(rr["available"])
    if la and ra:
        legality, reason = "LEGAL", "PUBLIC_RESOURCES_AVAILABLE"
    elif not la and not ra:
        legality, reason = "ILLEGAL_BOTH_NO_RESOURCE", "PUBLIC_RESOURCE_ABSENT"
    elif not la:
        legality, reason = "ILLEGAL_LEFT_NO_RESOURCE", "PUBLIC_RESOURCE_ABSENT"
    else:
        legality, reason = "ILLEGAL_RIGHT_NO_RESOURCE", "PUBLIC_RESOURCE_ABSENT"
    outcome_sem = _outcome_semantics(left, right, a, b, legal=(legality == "LEGAL"), reason=reason)
    outcome_sha = canonical_sha256(outcome_sem)
    row = {
        "schema_id": "IG_G_UPLIFT_PAIR_CONNECTION_RECORD_V1",
        "uplift_layer": 2,
        "stage_ref": "G2:S1",
        "left_carrier_ref": str(left["carrier_ref"]),
        "right_carrier_ref": str(right["carrier_ref"]),
        "left_interface_sha256": str(left["interface_sha256"]),
        "right_interface_sha256": str(right["interface_sha256"]),
        "connection_operator_ref": f"G1_PUBLIC_BRIDGE_RELATION_V1:{a}>{b}",
        "legality": legality,
        "outcome_ref": ("G2S1O:" if legality == "LEGAL" else "G2S1F:") + outcome_sha,
        "outcome_science_sha256": outcome_sha,
        "relabel_invariance_status": "PASS_CANONICAL_WHOLE_CARRIER_SWAP_INVARIANT_OUTCOME",
        "source_authority_sha256": source_authority_sha256,
    }
    missing = [x for x in _REQUIRED_RECORD_FIELDS if x not in row]
    if missing:
        raise UpliftStructuralError(f"pair connection record missing frozen fields: {missing}")
    return row




def post_reservation_public_continuation_projection(carrier: Any, first_reserved_type: int) -> dict[str, Any]:
    """Strictly public D4 continuation projection for one input carrier.

    This is the public continuation used by the preregistered D4 precursor:
    reserve one declared S1 endpoint, then expose only the successor skin/caps
    and the seven possible next public reserve_external outcomes. Hidden
    witnesses and all carrier internals are discarded.
    """
    t = int(first_reserved_type)
    caps0 = tuple(int(x) for x in carrier.total_caps)
    if len(caps0) != 7 or t < 0 or t >= 7 or caps0[t] <= 0:
        raise UpliftStructuralError("D4 continuation requires an available declared first reservation")
    try:
        first, _hidden_witness_discarded = carrier.reserve_external(t)
    except Exception as exc:
        raise UpliftStructuralError(f"public reserve_external({t}) failed during D4 continuation") from exc
    first_skin = str(first.skin)
    _require_sha256(first_skin, field="D4 first successor skin")
    first_caps = [int(x) for x in first.total_caps]
    if len(first_caps) != 7:
        raise UpliftStructuralError("D4 first successor resource vector must contain seven counters")
    next_rows: list[dict[str, Any]] = []
    for u in range(7):
        if first_caps[u] <= 0:
            next_rows.append({
                "endpoint_type": u,
                "available": False,
                "successor_boundary_resource_skin_sha256": None,
                "successor_total_free_by_type": None,
            })
            continue
        try:
            nxt, _hidden_witness_discarded2 = first.reserve_external(u)
        except Exception as exc:
            raise UpliftStructuralError(f"public second reserve_external({u}) failed during D4 continuation") from exc
        nxt_skin = str(nxt.skin)
        _require_sha256(nxt_skin, field=f"D4 second successor skin type {u}")
        nxt_caps = [int(x) for x in nxt.total_caps]
        next_rows.append({
            "endpoint_type": u,
            "available": True,
            "successor_boundary_resource_skin_sha256": nxt_skin,
            "successor_total_free_by_type": nxt_caps,
        })
    # Hash the exact D4 continuation semantics frozen in the earlier descriptor probe.
    semantic = {
        "schema_id": "IG_G1_POST_RESERVATION_PUBLIC_CONTINUATION_V1",
        "first_reserved_type": t,
        "successor_skin": first_skin,
        "successor_total_caps": first_caps,
        "next_reservations": [
            {
                "endpoint_type": r["endpoint_type"],
                "available": r["available"],
                "successor_skin": r["successor_boundary_resource_skin_sha256"],
                "successor_total_caps": r["successor_total_free_by_type"],
            }
            for r in next_rows
        ],
    }
    semantic["science_sha256"] = canonical_sha256(semantic)
    return semantic

def one_reservation_successor_skin_projection(realized_assembly: Any) -> dict[str, Any]:
    """Strictly public Q2 projection earned by the S1 repair probe.

    Reads only the realized assembly's previous-layer public resource vector and
    ``reserve_external(type)`` successor public skin.  Reservation witnesses are
    discarded.  Hidden children/owners/topology/ancestry are never read.
    """
    caps = tuple(int(x) for x in realized_assembly.total_caps)
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise UpliftStructuralError("realized assembly public resource vector must contain seven nonnegative counters")
    rows: list[dict[str, Any]] = []
    for t in range(7):
        if caps[t] <= 0:
            rows.append({
                "endpoint_type": t,
                "available": False,
                "successor_boundary_resource_skin_sha256": None,
            })
            continue
        try:
            successor, _hidden_witness_discarded = realized_assembly.reserve_external(t)
        except Exception as exc:
            raise UpliftStructuralError(f"realized public reserve_external({t}) failed despite positive capacity") from exc
        succ_skin = str(successor.skin)
        _require_sha256(succ_skin, field=f"realized reserve_external({t}) successor skin")
        rows.append({
            "endpoint_type": t,
            "available": True,
            "successor_boundary_resource_skin_sha256": succ_skin,
        })
    # Bind the science hash to the exact Q2 candidate semantics frozen in the
    # strict-public probe. The surrounding envelope is non-science metadata.
    frozen_rows = [
        {
            "t": int(r["endpoint_type"]),
            "available": bool(r["available"]),
            "successor_skin": r["successor_boundary_resource_skin_sha256"],
        }
        for r in rows
    ]
    projection = {
        "schema_id": "IG_G2_S1_Q2_ONE_RESERVATION_SUCCESSOR_SKINS_V1",
        "rows": rows,
        "scope": "STRICT_PREVIOUS_LAYER_PUBLIC_ONE_RESERVATION_SUCCESSOR_SKINS",
        "hidden_internal_reads": False,
        "reservation_witnesses_retained": False,
        "science_sha256": canonical_sha256(frozen_rows),
    }
    return projection


def refine_legal_pair_connection_record_with_q2(
    projected_record: Mapping[str, Any],
    realized_assembly: Any,
    *,
    source_authority_sha256: str,
) -> dict[str, Any]:
    """Refine one legal projected S1 record by the preregistered strict-public Q2 winner.

    This does not use construction identity or any hidden carrier read.  It binds
    the old projected outcome to the realized assembly's public one-reservation
    successor-skin projection.  The resulting outcome identity is the candidate
    S1 repair to be independently re-audited before S2 can unlock.
    """
    source_authority_sha256 = _require_sha256(source_authority_sha256, field="source_authority_sha256")
    if str(projected_record.get("legality")) != "LEGAL":
        raise UpliftStructuralError("Q2 realized refinement currently requires a legal projected S1 record")
    old_sha = _require_sha256(str(projected_record["outcome_science_sha256"]), field="projected outcome_science_sha256")
    q2 = one_reservation_successor_skin_projection(realized_assembly)
    semantics = {
        "schema_id": "IG_G_UPLIFT_S1_REPAIRED_PAIR_OUTCOME_SEMANTICS_V2",
        "projected_outcome_science_sha256": old_sha,
        "q2_one_reservation_successor_skins_sha256": q2["science_sha256"],
        "repair_basis": "STRICT_PUBLIC_REPAIR_PROBE_Q2_WINNER",
        "observer_scope": "ONE_STEP_REALIZED_PUBLIC_CONGRUENCE_CANDIDATE",
    }
    new_sha = canonical_sha256(semantics)
    row = dict(projected_record)
    row.update({
        "schema_id": "IG_G_UPLIFT_PAIR_CONNECTION_RECORD_V2",
        "projected_outcome_science_sha256": old_sha,
        "q2_one_reservation_successor_skins": q2,
        "q2_one_reservation_successor_skins_sha256": q2["science_sha256"],
        "outcome_ref": "G2S1O2:" + new_sha,
        "outcome_science_sha256": new_sha,
        "realization_status": "PASS",
        "strict_public_repair_status": "CANDIDATE_PENDING_FULL_580351_CONGRUENCE_REAUDIT",
        "relabel_invariance_status": "PASS_CANONICAL_WHOLE_CARRIER_SWAP_INVARIANT_PROJECTED_OUTCOME_PLUS_ASSEMBLY_PUBLIC_Q2",
        "source_authority_sha256": source_authority_sha256,
        "nonclaims": [
            "NOT_G2_STATE_DESCRIPTOR",
            "NOT_UNBOUNDED_CONTEXT_COMPLETE",
            "NOT_S2_UNLOCK_BY_IMPLEMENTATION_ALONE",
            "NO_G2_GRADUATION",
        ],
    })
    missing = [x for x in _REQUIRED_RECORD_FIELDS if x not in row]
    if missing:
        raise UpliftStructuralError(f"repaired pair connection record missing frozen fields: {missing}")
    return row



def repaired_pair_outcome_semantics_v2(
    projected_record: Mapping[str, Any],
    *,
    left_continuation_sha256: str,
    right_continuation_sha256: str,
    q2_sha256: str,
) -> dict[str, Any]:
    """Pure canonical D4+Q2 repaired outcome semantics from verified public hashes."""
    if str(projected_record.get("legality")) != "LEGAL":
        raise UpliftStructuralError("repaired V2 outcome semantics currently requires a legal projected S1 record")
    old_sha = _require_sha256(str(projected_record["outcome_science_sha256"]), field="projected outcome_science_sha256")
    left_continuation_sha256 = _require_sha256(left_continuation_sha256, field="left continuation sha256")
    right_continuation_sha256 = _require_sha256(right_continuation_sha256, field="right continuation sha256")
    q2_sha256 = _require_sha256(q2_sha256, field="Q2 sha256")
    return {
        "schema_id": "IG_G_UPLIFT_S1_REPAIRED_PAIR_OUTCOME_SEMANTICS_V2",
        "projected_outcome_science_sha256": old_sha,
        "connection_operator_ref": str(projected_record["connection_operator_ref"]),
        "left_post_reservation_public_continuation_sha256": left_continuation_sha256,
        "right_post_reservation_public_continuation_sha256": right_continuation_sha256,
        "q2_one_reservation_successor_skins_sha256": q2_sha256,
        "repair_basis": "D4_STRICT_PUBLIC_PRECURSOR_PLUS_Q2_STRICT_PUBLIC_RESIDUAL_WINNER",
        "observer_scope": "ONE_STEP_REALIZED_PUBLIC_CONGRUENCE_CANDIDATE",
    }


def repaired_pair_connection_record_v2_from_public_hashes(
    projected_record: Mapping[str, Any],
    *,
    left_continuation_sha256: str,
    right_continuation_sha256: str,
    q2_projection: Mapping[str, Any],
    source_authority_sha256: str,
) -> dict[str, Any]:
    """Build the repaired record from separately verified strictly-public projections."""
    source_authority_sha256 = _require_sha256(source_authority_sha256, field="source_authority_sha256")
    q2_sha = _require_sha256(str(q2_projection["science_sha256"]), field="Q2 projection science_sha256")
    semantics = repaired_pair_outcome_semantics_v2(
        projected_record,
        left_continuation_sha256=left_continuation_sha256,
        right_continuation_sha256=right_continuation_sha256,
        q2_sha256=q2_sha,
    )
    new_sha = canonical_sha256(semantics)
    row = dict(projected_record)
    row.update({
        "schema_id": "IG_G_UPLIFT_PAIR_CONNECTION_RECORD_V2",
        "projected_outcome_science_sha256": str(projected_record["outcome_science_sha256"]),
        "d4_left_post_reservation_public_continuation_sha256": left_continuation_sha256,
        "d4_right_post_reservation_public_continuation_sha256": right_continuation_sha256,
        "q2_one_reservation_successor_skins_sha256": q2_sha,
        "q2_projection_serialized": False,
        "outcome_ref": "G2S1O2:" + new_sha,
        "outcome_science_sha256": new_sha,
        "realization_status": "PASS",
        "strict_public_repair_status": "D4_PLUS_Q2_CANDIDATE_PENDING_FULL_580351_CONGRUENCE_REAUDIT",
        "relabel_invariance_status": "PASS_CANONICAL_INPUT_SLOT_ORDER_PLUS_REALIZED_ASSEMBLY_PUBLIC_Q2",
        "source_authority_sha256": source_authority_sha256,
        "nonclaims": [
            "NOT_G2_STATE_DESCRIPTOR",
            "NOT_UNBOUNDED_CONTEXT_COMPLETE",
            "NOT_S2_UNLOCK_BY_IMPLEMENTATION_ALONE",
            "NO_G2_GRADUATION",
        ],
    })
    missing = [x for x in _REQUIRED_RECORD_FIELDS if x not in row]
    if missing:
        raise UpliftStructuralError(f"repaired D4+Q2 pair connection record missing frozen fields: {missing}")
    return row

def refine_legal_pair_connection_record_with_d4_q2(
    projected_record: Mapping[str, Any],
    left_carrier: Any,
    right_carrier: Any,
    realized_assembly: Any,
    *,
    source_authority_sha256: str,
) -> dict[str, Any]:
    """Final v0.29.2 strict-public S1 repair candidate: D4 precursor + Q2 residual winner.

    Evidence order is preserved exactly: D4 was the strongest preregistered
    strictly-public precursor and left 1,952 residual split classes; Q2 was then
    frozen and tested only on those D4 residual classes and removed all of them
    on discovery and holdout.  Therefore the promotable candidate is D4+Q2,
    not Q2 alone.
    """
    source_authority_sha256 = _require_sha256(source_authority_sha256, field="source_authority_sha256")
    if str(projected_record.get("legality")) != "LEGAL":
        raise UpliftStructuralError("D4+Q2 realized refinement currently requires a legal projected S1 record")
    old_sha = _require_sha256(str(projected_record["outcome_science_sha256"]), field="projected outcome_science_sha256")
    op = str(projected_record["connection_operator_ref"])
    try:
        pair = op.rsplit(":", 1)[1]
        a_s, b_s = pair.split(">", 1)
        a, b = int(a_s), int(b_s)
    except Exception as exc:
        raise UpliftStructuralError("malformed G1 public bridge operator ref") from exc
    left_cont = post_reservation_public_continuation_projection(left_carrier, a)
    right_cont = post_reservation_public_continuation_projection(right_carrier, b)
    q2 = one_reservation_successor_skin_projection(realized_assembly)
    return repaired_pair_connection_record_v2_from_public_hashes(
        projected_record,
        left_continuation_sha256=left_cont["science_sha256"],
        right_continuation_sha256=right_cont["science_sha256"],
        q2_projection=q2,
        source_authority_sha256=source_authority_sha256,
    )

def _canonical_json_line(obj: Mapping[str, Any]) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def run_pair_connection_census(
    interface_population: Mapping[str, Any],
    *,
    bridge_pairs: Sequence[tuple[int, int]],
    source_authority_sha256: str,
    records_path: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen S1 unordered-with-replacement whole-carrier pair census.

    If ``records_path`` is supplied, every canonical connection record is retained
    as gzip-compressed JSONL.  The summary contains a deterministic stream hash and
    per-pair hashes, so the full census is independently replay-verifiable.
    """
    spec = implementation_spec()
    source_authority_sha256 = _require_sha256(source_authority_sha256, field="source_authority_sha256")
    if interface_population.get("stage_ref") != "G2:S0" or interface_population.get("status") != "PASS":
        raise UpliftStructuralError("S1 requires a passing G2:S0 interface population")
    if interface_population.get("source_authority_sha256") != source_authority_sha256:
        raise UpliftStructuralError("S0/S1 source authority mismatch")
    pairs_norm = [(int(a), int(b)) for a, b in bridge_pairs]
    if canonical_sha256([list(x) for x in pairs_norm]) != spec["pair_census"]["bridge_operator_catalog_sha256"]:
        raise UpliftStructuralError("live G1 bridge operator catalogue differs from frozen S0/S1 spec")
    if len(pairs_norm) != len(set(pairs_norm)):
        raise UpliftStructuralError("bridge operator catalogue contains duplicates")

    interfaces = sorted((dict(x) for x in interface_population["interfaces"]), key=lambda x: x["carrier_ref"])
    n = len(interfaces)
    pair_count = n * (n + 1) // 2
    attempted = pair_count * len(pairs_norm)
    counts: dict[str, int] = {}
    outcome_hashes: set[str] = set()
    stream_hash = hashlib.sha256()
    pair_summaries: list[dict[str, Any]] = []
    in_memory_records: list[dict[str, Any]] | None = [] if records_path is None else None

    gz = None
    if records_path is not None:
        records_path = Path(records_path)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        gz = gzip.GzipFile(filename=str(records_path), mode="wb", compresslevel=9, mtime=0)
    try:
        for i in range(n):
            for j in range(i, n):
                left, right = interfaces[i], interfaces[j]
                pair_hasher = hashlib.sha256()
                pair_counts: dict[str, int] = {}
                pair_outcomes: set[str] = set()
                for a, b in pairs_norm:
                    row = pair_connection_record(
                        left, right, a, b,
                        bridge_pairs=pairs_norm,
                        source_authority_sha256=source_authority_sha256,
                    )
                    line = _canonical_json_line(row)
                    stream_hash.update(line)
                    pair_hasher.update(line)
                    if gz is not None:
                        gz.write(line)
                    else:
                        assert in_memory_records is not None
                        in_memory_records.append(row)
                    leg = str(row["legality"])
                    counts[leg] = counts.get(leg, 0) + 1
                    pair_counts[leg] = pair_counts.get(leg, 0) + 1
                    outcome_hashes.add(str(row["outcome_science_sha256"]))
                    pair_outcomes.add(str(row["outcome_science_sha256"]))
                pair_summaries.append({
                    "left_carrier_ref": str(left["carrier_ref"]),
                    "right_carrier_ref": str(right["carrier_ref"]),
                    "left_interface_sha256": str(left["interface_sha256"]),
                    "right_interface_sha256": str(right["interface_sha256"]),
                    "attempt_count": len(pairs_norm),
                    "legality_counts": [{"legality": k, "count": pair_counts[k]} for k in sorted(pair_counts)],
                    "outcome_class_count": len(pair_outcomes),
                    "pair_record_stream_sha256": pair_hasher.hexdigest(),
                })
    finally:
        if gz is not None:
            gz.close()

    if sum(counts.values()) != attempted:
        raise UpliftStructuralError("S1 census attempted-record accounting mismatch")
    summary = {
        "schema_id": "IG_G_UPLIFT_S1_PAIR_CONNECTION_CENSUS_V1",
        "schema_version": "1.0.0",
        "status": "PASS",
        "stage_ref": "G2:S1",
        "source_stage": str(interface_population["source_stage"]),
        "pair_mode": "UNORDERED_WHOLE_CARRIER_PAIRS_WITH_REPLACEMENT",
        "carrier_count": n,
        "interface_class_count": int(interface_population["interface_class_count"]),
        "unordered_pair_count": pair_count,
        "bridge_operator_count": len(pairs_norm),
        "attempted_connection_count": attempted,
        "legality_counts": [{"legality": k, "count": counts[k]} for k in sorted(counts)],
        "outcome_class_count": len(outcome_hashes),
        "record_stream_sha256": stream_hash.hexdigest(),
        "records_encoding": "CANONICAL_JSONL_GZIP" if records_path is not None else "IN_MEMORY_CANONICAL_RECORDS",
        "records_file_name": None if records_path is None else Path(records_path).name,
        "pair_summaries": pair_summaries,
        "source_authority_sha256": source_authority_sha256,
        "implementation_spec_sha256": spec["spec_sha256"],
        "architecture_contract_sha256": spec["architecture_contract_sha256"],
        "failed_connections_retained": True,
        "hidden_internal_reads": False,
        "g2_graduated": False,
        "next_stage_if_verified": "G2:S2",
        "nonclaims": [
            "NO_PAIR_OUTCOME_QUOTIENT_YET",
            "NO_TRIPLE_IRREDUCIBILITY_RESULT",
            "NO_G2_GRADUATION",
        ],
    }
    if in_memory_records is not None:
        summary["records"] = in_memory_records
    summary["science_sha256"] = canonical_sha256(summary)
    return summary


def verify_census_record_stream(summary: Mapping[str, Any], records_path: Path) -> dict[str, Any]:
    expected_count = int(summary["attempted_connection_count"])
    expected_hash = str(summary["record_stream_sha256"])
    h = hashlib.sha256()
    count = 0
    missing_rows = 0
    with gzip.open(Path(records_path), "rb") as f:
        for line in f:
            h.update(line)
            row = json.loads(line)
            if any(field not in row for field in _REQUIRED_RECORD_FIELDS):
                missing_rows += 1
            count += 1
    obj = {
        "schema_id": "IG_G_UPLIFT_S1_RECORD_STREAM_VERIFICATION_V1",
        "status": "PASS" if (count == expected_count and h.hexdigest() == expected_hash and missing_rows == 0) else "FAIL",
        "observed_record_count": count,
        "expected_record_count": expected_count,
        "observed_stream_sha256": h.hexdigest(),
        "expected_stream_sha256": expected_hash,
        "rows_missing_required_fields": missing_rows,
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


def _write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_g2_s0_s1(output: Path, *, workers: int = 4) -> dict[str, Any]:
    """Execute the frozen first-class G2:S0/S1 experiment on the full G1:R100 cohort.

    This is intentionally synchronous; production/chat use should launch it through the
    project's detached checkpointed controller protocol.  The computation itself writes
    durable evidence incrementally into ``output`` and fails closed on any authority,
    population, stream-verification, or hidden-read contract mismatch.
    """
    from . import build_meta
    from .execution import ExecutionPolicy
    from .materialized_discovery import MaterializedDiscoverySession
    from .maturation_parallel import parallel_candidate_descriptors, rebuild_selected_states
    from .records import source_sha256
    from .semantic_sentinel import verify_native_semantics
    from .theorem_registry import verify_all_theorems
    import platform
    import time

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    spec = implementation_spec()
    arch = uplift_contract()
    meta = build_meta()

    expected_runtime = arch["compatibility"]["phase3_authorized_runtime"]
    observed_source = source_sha256()
    preflight = {
        "schema_id": "IG_G_UPLIFT_S0_S1_RUNTIME_PREFLIGHT_V1",
        "status": "PASS",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "expected_python": expected_runtime["version"],
        "expected_implementation": expected_runtime["implementation"],
        "source_sha256": observed_source,
        "build_meta_source_sha256": str(meta.get("source_sha256")),
        "native_semantics_status": verify_native_semantics(raise_on_change=False)["status"],
        "theorem_registry_status": verify_all_theorems(raise_on_stale=False)["status"],
        "architecture_contract_sha256": arch["contract_sha256"],
        "implementation_spec_sha256": spec["spec_sha256"],
    }
    checks = [
        preflight["python"] == preflight["expected_python"],
        preflight["implementation"] == preflight["expected_implementation"],
        preflight["source_sha256"] == preflight["build_meta_source_sha256"],
        preflight["native_semantics_status"] == "PASS",
        preflight["theorem_registry_status"] == "PASS",
    ]
    if not all(checks):
        preflight["status"] = "FAIL"
    preflight["science_sha256"] = canonical_sha256(preflight)
    _write_json_atomic(evidence / "RUNTIME_PREFLIGHT.json", preflight)
    if preflight["status"] != "PASS":
        raise UpliftStructuralError("G2:S0/S1 runtime/source preflight failed")

    prereg = {
        "schema_id": "IG_G2_S0_S1_PREREGISTRATION_V1",
        "date": "2026-09-02",
        "status": "FROZEN_BEFORE_SCIENCE",
        "architecture_contract_sha256": arch["contract_sha256"],
        "implementation_spec_sha256": spec["spec_sha256"],
        "source_authority_sha256": observed_source,
        "source_population": spec["source_population"],
        "source_stage": "G1:R100",
        "stages": ["G2:S0", "G2:S1"],
        "pair_mode": spec["pair_census"]["pair_mode"],
        "promotion": False,
        "stopping_rules": [
            "STOP_ON_RUNTIME_OR_SOURCE_AUTHORITY_MISMATCH",
            "STOP_IF_G1_R100_CANDIDATE_COUNT_IS_NOT_193",
            "STOP_ON_PUBLIC_RESERVATION_INCONSISTENCY",
            "STOP_ON_BRIDGE_CATALOG_MISMATCH",
            "STOP_ON_PAIR_RECORD_STREAM_VERIFICATION_FAILURE",
        ],
        "hard_nonclaims": spec["hard_nonclaims"],
    }
    prereg["science_sha256"] = canonical_sha256(prereg)
    _write_json_atomic(output / "PREREGISTRATION.json", prereg)

    t0 = time.time()
    policy = ExecutionPolicy(
        backend="AUTO",
        requested_workers=int(workers),
        scheduler="COST_WEIGHTED_SHARDS",
        owner="g2-s0-s1",
    )
    with MaterializedDiscoverySession(execution_policy=policy) as session:
        # Stop at R99, then explicitly reconstruct the complete deterministic R100
        # pre-selection cohort rather than the 24-member observation panel.
        session.advance_to(99)
        prev = session.level_states[99]
        candidates, selected_desc, exec_meta, center = parallel_candidate_descriptors(
            engine=session.engine,
            prev=prev,
            level=100,
            pairs=session.bridge_pairs,
            motifs=session.motifs,
            spec=session.spec,
            policy=policy,
        )
        if len(candidates) != 193 or len(selected_desc) != 24:
            raise UpliftStructuralError(
                f"unexpected G1:R100 complete/selected population {len(candidates)}/{len(selected_desc)}"
            )
        carriers = rebuild_selected_states(
            candidates,
            engine=session.engine,
            prev=prev,
            level=100,
            pairs=session.bridge_pairs,
            motifs=session.motifs,
            center=center,
        )
        carriers = sorted(carriers, key=lambda s: s.construction_digest)
        if len({s.construction_digest for s in carriers}) != 193:
            raise UpliftStructuralError("G1:R100 complete cohort carrier refs are not unique")

        population_meta = {
            "schema_id": "IG_G2_S0_INPUT_POPULATION_V1",
            "status": "PASS",
            "source_stage": "G1:R100",
            "candidate_count": len(carriers),
            "selected_reference_count": len(selected_desc),
            "carrier_refs_sha256": canonical_sha256([s.construction_digest for s in carriers]),
            "candidate_construction_metadata": exec_meta,
            "bridge_operator_catalog_sha256": canonical_sha256([list(x) for x in session.bridge_pairs]),
        }
        population_meta["science_sha256"] = canonical_sha256(population_meta)
        _write_json_atomic(evidence / "G1_R100_COMPLETE_COHORT.json", population_meta)

        s0 = extract_interface_population(
            carriers,
            source_stage="G1:R100",
            source_authority_sha256=observed_source,
        )
        _write_json_atomic(evidence / "G2_S0_INTERFACE_POPULATION.json", s0)

        records_path = evidence / "G2_S1_PAIR_CONNECTION_RECORDS.jsonl.gz"
        s1 = run_pair_connection_census(
            s0,
            bridge_pairs=session.bridge_pairs,
            source_authority_sha256=observed_source,
            records_path=records_path,
        )
        _write_json_atomic(evidence / "G2_S1_PAIR_CONNECTION_CENSUS.json", s1)
        verification = verify_census_record_stream(s1, records_path)
        _write_json_atomic(evidence / "G2_S1_RECORD_STREAM_VERIFICATION.json", verification)
        if verification["status"] != "PASS":
            raise UpliftStructuralError("G2:S1 record stream verification failed")

    result = {
        "schema_id": "IG_G2_S0_S1_RESULT_V1",
        "date": "2026-09-02",
        "status": "PASS",
        "classification": "G2_S0_S1_WHOLE_CARRIER_INTERFACE_AND_PAIR_CENSUS_COMPLETE",
        "stages_completed": ["G2:S0", "G2:S1"],
        "source_stage": "G1:R100",
        "carrier_count": s0["carrier_count"],
        "interface_class_count": s0["interface_class_count"],
        "unordered_pair_count": s1["unordered_pair_count"],
        "attempted_connection_count": s1["attempted_connection_count"],
        "legality_counts": s1["legality_counts"],
        "pair_outcome_class_count_pre_S2": s1["outcome_class_count"],
        "s0_science_sha256": s0["science_sha256"],
        "s1_science_sha256": s1["science_sha256"],
        "s1_record_stream_sha256": s1["record_stream_sha256"],
        "record_stream_file_sha256": _sha256_file(records_path),
        "elapsed_seconds": round(time.time() - t0, 6),
        "next_stage": "G2:S2",
        "g2_graduated": False,
        "nonclaims": [
            "NO_G2_GRADUATION",
            "NO_PAIR_QUOTIENT_CLAIM_BEFORE_S2",
            "NO_TRIPLE_IRREDUCIBILITY_CLAIM_BEFORE_S3",
        ],
    }
    # Timing is operational and excluded from the sealed scientific result identity.
    science_projection = {k: v for k, v in result.items() if k != "elapsed_seconds"}
    result["sealed_science_sha256"] = canonical_sha256(science_projection)
    _write_json_atomic(output / "RESULT.json", result)

    files_out = []
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256.json":
            files_out.append({
                "path": p.relative_to(output).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            })
    manifest = {
        "schema_id": "IG_G2_S0_S1_RUN_MANIFEST_V1",
        "file_count": len(files_out),
        "files": files_out,
    }
    manifest["science_sha256"] = canonical_sha256(manifest)
    _write_json_atomic(output / "MANIFEST_SHA256.json", manifest)
    return result
