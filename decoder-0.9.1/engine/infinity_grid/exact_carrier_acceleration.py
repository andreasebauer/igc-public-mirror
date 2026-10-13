from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from importlib.resources import files

from . import regime_scanner as rs
from .canon import write_json_atomic
from .exact_carrier_unblinding import ExactO7State, ExactProjectedActionOracle, _profile_digest
from .semantic_sentinel import current_action_contract, verify_native_semantics
from .theorem_registry import verify_theorem


class ExactCarrierAccelerationError(RuntimeError):
    pass


class AuthorityMismatch(ExactCarrierAccelerationError):
    pass


class ReopenEventRequired(ExactCarrierAccelerationError):
    pass


def load_exact_carrier_acceleration_spec() -> dict:
    p = files("infinity_grid").joinpath(
        "resources/decoder/O_EXACT_CARRIER_THEOREM_ACCELERATION_SPEC_v1.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


def _j(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _milestone(output: Path, stage: str, **extra: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": "IG_EXACT_CARRIER_FAST_CALIBRATION_STATUS_V1",
        "status": "RUNNING" if stage != "COMPLETE" else "COMPLETE",
        "stage": stage,
        "updated_unix": time.time(),
        **extra,
    }
    write_json_atomic(output / "CALIBRATION_STATUS.json", body)


def current_fixed_grammar_contract() -> dict:
    gate_path = files("infinity_grid").joinpath("resources/decoder/GRRL_APPLICATION_GATE_RECONCILED_v0.1.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    native = current_action_contract()
    return {
        "schema": "IG_GRRL_NORMALIZED_IMPLEMENTATION_CONTRACT_V2",
        "normalized_grammar_sha256": rs.GRAMMAR_EXPECTED,
        "grrl_application_gate_sha256": rs._sha(gate),
        "relation_arity": native["relation_arity"],
        "ownership_scope": "DISTINCT_TOP_OWNER_OCCURRENCES",
        "reservation_effect": "ONE_TYPED_ENDPOINT_PER_SIDE",
        "rank_lift_effect": "LOCAL_P_F_INCREMENT_OWNERSHIP_PRESERVED",
        "relation_effect": "MONOTONE_ADD_ONLY",
        "successor_semantics": native["successor_semantics"],
        "read_set": native["read_set"],
        "hidden_reads": native["hidden_reads"],
        "writes": native["writes"],
        "forbidden_features_present": native["forbidden_features_present"],
        "native_semantics_science_sha256": native["native_semantics_science_sha256"],
        "native_semantics_baseline_sha256": native["native_semantics_baseline_sha256"],
    }


def compare_fixed_grammar_contracts(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict:
    """Fail-open scientifically: any normalized contract change becomes a reopen event."""
    keys = [
        "normalized_grammar_sha256",
        "grrl_application_gate_sha256",
        "relation_arity",
        "ownership_scope",
        "reservation_effect",
        "rank_lift_effect",
        "relation_effect",
        "successor_semantics",
        "read_set",
        "hidden_reads",
        "writes",
        "forbidden_features_present",
        "native_semantics_science_sha256",
        "native_semantics_baseline_sha256",
    ]
    changes = []
    for key in keys:
        a = previous.get(key)
        b = current.get(key)
        if a != b:
            changes.append({"field": key, "previous": a, "current": b})
    return {
        "classification": "REOPEN_EVENT" if changes else "FIXED_GRAMMAR_UNCHANGED",
        "changes": changes,
        "previous_sha256": rs._sha(dict(previous)),
        "current_sha256": rs._sha(dict(current)),
    }


def audit_phase8_theorem_authority(phase8: Path) -> dict:
    spec = load_exact_carrier_acceleration_spec()
    expected = spec["authority"]

    generic = phase8 / "graduation_compact" / "07_INPUT_SNAPSHOTS" / "GENERIC_THEOREM_STATUS.txt"
    adapter_p = phase8 / "authority" / "O7_ADAPTER_AUDIT_RESULT.json"
    read_probe_p = phase8 / "authority" / "O7_TOPOLOGY_AWARE_READ_PROBE_RESULT.json"
    o8_p = phase8 / "authority" / "O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json"
    o9_p = phase8 / "authority" / "O9_BP0_RESULT.json"
    for p in (generic, adapter_p, read_probe_p, o8_p, o9_p):
        if not p.is_file():
            raise AuthorityMismatch(f"required authority missing: {p}")

    generic_text = generic.read_text(encoding="utf-8")
    adapter = _j(adapter_p)
    read_probe = _j(read_probe_p)
    o8 = _j(o8_p)
    o9 = _j(o9_p)
    current = current_fixed_grammar_contract()

    gates = {
        "A0_machine_theorem": (
            rs._sha_file(generic) == expected["generic_theorem_status_file_sha256"]
            and expected["generic_theorem_status"] in generic_text
        ),
        "A1_O7_adapter": (
            adapter.get("status") == expected["o7_adapter_status"]
            and adapter.get("science_sha256") == expected["o7_adapter_science_sha256"]
            and all(x.get("status") == "PASS" for x in adapter.get("B1_B4", []))
        ),
        "A2_O7_read_probe": (
            read_probe.get("status") == "PASS"
            and read_probe.get("science_sha256") == expected["o7_topology_read_probe_science_sha256"]
            and read_probe.get("operational_read_classification") == "TOPOLOGY_REMAINS_HIDDEN_UNDER_O7_ACTIONS"
        ),
        "A3_O8_graduation": (
            o8.get("status") == expected["o8_status"]
            and o8.get("science_sha256") == expected["o8_science_sha256"]
            and o8.get("gates", {}).get("G4_transition_closure_and_R8_factorisation", {}).get("status") == "PASS"
        ),
        "A4_O9_repeat": (
            o9.get("outcome") == expected["o9_outcome"]
            and o9.get("science_sha256") == expected["o9_science_sha256"]
            and o9.get("normalized_grammar", {}).get("candidate_sha256") == expected["normalized_grammar_sha256"]
            and o9.get("gates", {}).get("P2_B1_B4") == "PASS"
            and o9.get("representation_gate", {}).get("status") == "PASS"
        ),
        "A5_current_implementation_contract": (
            current["normalized_grammar_sha256"] == expected["normalized_grammar_sha256"]
            and current["grrl_application_gate_sha256"] == expected["grrl_application_gate_sha256"]
            and current["successor_semantics"] == "COUNTER"
            and not current["hidden_reads"]
            and not any(current["forbidden_features_present"].values())
        ),
    }
    failures = [k for k, ok in gates.items() if not ok]
    if failures:
        raise AuthorityMismatch(f"theorem authority/current contract gate failed: {failures}")
    out = {
        "schema": "IG_EXACT_CARRIER_THEOREM_AUTHORITY_AUDIT_V1",
        "status": "PASS",
        "gates": {k: "PASS" for k in gates},
        "generic_theorem_status_file_sha256": rs._sha_file(generic),
        "o7_adapter_science_sha256": adapter["science_sha256"],
        "o7_read_probe_science_sha256": read_probe["science_sha256"],
        "o8_science_sha256": o8["science_sha256"],
        "o9_science_sha256": o9["science_sha256"],
        "current_contract": current,
    }
    out["science_sha256"] = rs._sha({k: v for k, v in out.items() if k != "science_sha256"})
    return out


def theorem_transport_certificate(
    *,
    level: int,
    parent_certificate_sha256: str,
    material_instance_sha256: str,
    authority_audit: Mapping[str, Any],
    previous_contract: Mapping[str, Any] | None = None,
    current_contract: Mapping[str, Any] | None = None,
) -> dict:
    """Transport factorisation through one synthetic fixed-grammar material lift.

    This certifies factorisation for the supplied material instance. It is deliberately
    *not* a historical-level graduation or existence theorem.
    """
    spec = load_exact_carrier_acceleration_spec()
    verify_native_semantics(raise_on_change=True)
    verify_theorem("O_GENERIC_FIXED_GRAMMAR_FACTORISATION_V1", raise_on_stale=True)
    current_contract = dict(current_contract or current_fixed_grammar_contract())
    previous_contract = dict(previous_contract or current_contract)
    contract_cmp = compare_fixed_grammar_contracts(previous_contract, current_contract)
    gates = {
        "G0_parent_certificate": isinstance(parent_certificate_sha256, str) and len(parent_certificate_sha256) == 64,
        "G1_authority": authority_audit.get("status") == "PASS",
        "G2_contract_unchanged": contract_cmp["classification"] == "FIXED_GRAMMAR_UNCHANGED",
        "G3_counter_observer": current_contract.get("successor_semantics") == "COUNTER" and not current_contract.get("hidden_reads"),
        "G4_material_instance": isinstance(material_instance_sha256, str) and len(material_instance_sha256) == 64,
    }
    failed = [k for k, ok in gates.items() if not ok]
    if failed:
        raise ReopenEventRequired(f"theorem transport blocked; targeted exact audit required: {failed}; changes={contract_cmp['changes']}")
    out = {
        "schema": "IG_EXACT_CARRIER_FACTORISATION_TRANSPORT_CERTIFICATE_V1",
        "level": int(level),
        "status": "FACTORISATION_TRANSPORTED_BY_GENERIC_LIFT_THEOREM",
        "scientific_scope": "SYNTHETIC_MATERIAL_FIXED_GRAMMAR_INSTANCE_NOT_HISTORICAL_GRADUATION",
        "parent_certificate_sha256": parent_certificate_sha256,
        "material_instance_sha256": material_instance_sha256,
        "authority_science_sha256": authority_audit["science_sha256"],
        "contract_sha256": rs._sha(current_contract),
        "gates": {k: "PASS" for k in gates},
        "reopen_triggers": spec["reopen_triggers"],
        "nonclaims": spec["nonclaims"],
    }
    out["science_sha256"] = rs._sha({k: v for k, v in out.items() if k != "science_sha256"})
    return out


def _load_o7_sentinel_state(phase8: Path, o7root: Path, witness: Mapping[str, Any]):
    os.environ["OSCOUT_DATA_ROOT"] = str(o7root.resolve())
    engine = rs._load_module("ig_exact_carrier_fast_calibration_o7_engine", o7root / "02_CODE" / "o7_live_engine.py")
    engine.O6 = engine.import_o6()
    parent_map = engine.load_parent_records()
    _ports, bpairs = engine.O6.load_rules()
    bridge_pairs = sorted(tuple(map(int, x)) for x in bpairs)
    records = _j(
        phase8 / "graduation_compact" / "07_INPUT_SNAPSHOTS" / "O7_IMMUTABLE_SURVIVORS.json"
    )["records"]
    record = next((r for r in records if r.get("state_digest") == witness["state_digest"]), None)
    if record is None:
        raise AuthorityMismatch(f"fixed O7 sentinel missing from immutable authority: {witness['state_digest']}")
    if record.get("lane") != witness["lane"] or record.get("R7_skin_sha256") != witness["R7_skin"]:
        raise AuthorityMismatch("fixed O7 sentinel authority metadata mismatch")
    ctx = engine._profile_row_context(record, parent_map)
    state = ExactO7State(engine, ctx, tuple(tuple(x) for x in record["edges"]), record["state_digest"], record["lane"])
    return state, bridge_pairs

def run_theorem_accelerated_calibration(
    phase8_seed: Path,
    output: Path,
    *,
    verify_fast_kernel_against_reference: bool = False,
    keep_work: bool = False,
) -> dict:
    """Fast O7/O8/O9 calibration: one real O7 separator sentinel + theorem authority replay.

    No O8/O9 motif census is generated.  That work is theorem-redundant under the frozen
    fixed grammar and was the dominant performance defect in the previous calibration.
    """
    spec = load_exact_carrier_acceleration_spec()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _milestone(output, "SETUP")
    work = Path(tempfile.mkdtemp(prefix="ig_fast_exact_cal_", dir=str(output)))
    t0 = time.monotonic()
    try:
        phase8, o7root = rs._extract_seed(Path(phase8_seed), work)
        authority = audit_phase8_theorem_authority(phase8)
        write_json_atomic(output / "THEOREM_AUTHORITY_AUDIT.json", authority)
        _milestone(output, "AUTHORITY_PASS", authority_science_sha256=authority["science_sha256"])

        witness = spec["fixed_witness_suite"]["O7_current_level_profile_sentinel"]
        state, bridge_pairs = _load_o7_sentinel_state(phase8, o7root, witness)
        if state.skin != witness["R7_skin"]:
            raise AuthorityMismatch("O7 fixed sentinel R7 skin mismatch")
        oracle = ExactProjectedActionOracle(bridge_pairs)
        profile = oracle.current_level_profile(state)
        profile_sha = _profile_digest(profile)
        if profile_sha != witness["R7_relation_add_profile_sha256"]:
            raise AuthorityMismatch(f"O7 fast profile regression mismatch: expected {witness['R7_relation_add_profile_sha256']} got {profile_sha}")
        if len(profile) != int(witness["action_successor_pairs"]) or int(sum(profile.values())) != int(witness["total_action_copies"]):
            raise AuthorityMismatch("O7 fast profile aggregate regression mismatch")
        kernel_check = None
        if verify_fast_kernel_against_reference:
            reference = oracle.current_level_profile_materialized_reference(state)
            kernel_check = {"fast_equals_materialized_reference": profile == reference}
            if not kernel_check["fast_equals_materialized_reference"]:
                raise AuthorityMismatch("fast projected kernel differs from materialized reference")
        o7 = {
            "status": "PASS",
            "classification": "O7_FIXED_EXACT_RELATION_ADD_KERNEL_SENTINEL_PASS",
            "state_digest": witness["state_digest"],
            "lane": witness["lane"],
            "R7_skin": witness["R7_skin"],
            "H_struct_sha256": state.h_struct_canon,
            "profile_sha256": profile_sha,
            "action_successor_pairs": len(profile),
            "total_action_copies": int(sum(profile.values())),
            "fast_kernel_reference_check": kernel_check,
            "factorisation_authority": "O7_ADAPTER_AND_TOPOLOGY_READ_PROBE; sentinel is code-regression only",
        }
        o7["science_sha256"] = rs._sha({k: v for k, v in o7.items() if k != "science_sha256"})
        write_json_atomic(output / "O7_FIXED_EXACT_SENTINEL_RESULT.json", o7)
        _milestone(output, "O7_SENTINEL_PASS", o7_science_sha256=o7["science_sha256"])

        o8auth = _j(phase8 / "authority" / "O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json")
        o8 = {
            "status": "PASS",
            "classification": "O8_FACTORISATION_REPLAYED_FROM_THEOREM_ACCELERATED_GRADUATION_AUTHORITY",
            "science_sha256": o8auth["science_sha256"],
            "transition_closure_basis": o8auth["transition_closure_basis"],
            "broad_census": 0,
        }
        write_json_atomic(output / "O8_THEOREM_REPLAY_RESULT.json", o8)
        _milestone(output, "O8_THEOREM_PASS", o8_science_sha256=o8["science_sha256"])

        o9auth = _j(phase8 / "authority" / "O9_BP0_RESULT.json")
        o9 = {
            "status": "PASS",
            "classification": "O9_EXISTENCE_REPEAT_SENTINEL_REPLAYED_THEOREM_PARAMETRIC",
            "outcome": o9auth["outcome"],
            "science_sha256": o9auth["science_sha256"],
            "normalized_grammar_sha256": o9auth["normalized_grammar"]["candidate_sha256"],
            "new_operational_read_found": bool(o9auth["emergent_read_gate"]["R3_new_operational_read_found"]),
            "broad_census": 0,
        }
        if o9["new_operational_read_found"]:
            raise AuthorityMismatch("O9 authority unexpectedly reports a new operational read")
        write_json_atomic(output / "O9_THEOREM_REPEAT_REPLAY_RESULT.json", o9)
        _milestone(output, "O9_THEOREM_PASS", o9_science_sha256=o9["science_sha256"])

        result = {
            "schema": "IG_EXACT_CARRIER_THEOREM_ACCELERATED_CALIBRATION_RESULT_V1",
            "status": "PASS",
            "classification": "THEOREM_ACCELERATED_O7_O9_CALIBRATION_PASS",
            "phase8_seed_sha256": rs._sha_file(Path(phase8_seed)),
            "spec_sha256": rs._sha(spec),
            "authority_science_sha256": authority["science_sha256"],
            "O7": o7,
            "O8": o8,
            "O9": o9,
            "runtime_policy": spec["runtime_policy"],
            "elapsed_seconds": round(time.monotonic() - t0, 6),
            "nonclaims": spec["nonclaims"],
        }
        # Wall time is execution metadata, not science.
        science_body = {k: v for k, v in result.items() if k not in {"science_sha256", "elapsed_seconds"}}
        result["science_sha256"] = rs._sha(science_body)
        write_json_atomic(output / "O_EXACT_CARRIER_FAST_CALIBRATION_RESULT.json", result)
        _milestone(output, "COMPLETE", result_science_sha256=result["science_sha256"])
        return result
    finally:
        if keep_work:
            (output / "WORKDIR.txt").write_text(str(work) + "\n", encoding="utf-8")
        else:
            shutil.rmtree(work, ignore_errors=True)
