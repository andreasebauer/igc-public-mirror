from __future__ import annotations

"""Registered G4:S1 whole-G3-unit hidden-structure context-read audit (V2).

Scientific correction in v0.30.35:
G4 consumes the complete G3 construction-term corpus frozen by G4:S0.  It does
not reopen G2/G1 implementation carriers and never rematerializes G1:R100.
This is the structural-uplift firewall: complete previous-layer carriers are
atomic relative to their earned public interface.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

from .canon import canonical_sha256
from .g4_term_state import G3TermState, G4PairTermState, compose_g4_pair_relation
from .uplift_g4_s0 import phase0_spec, challenge_tiers, s1_placeholder_spec


class G4S1Error(RuntimeError):
    pass


def s1_spec() -> dict[str, Any]:
    return s1_placeholder_spec()


def _term_ref(row: Mapping[str, Any]) -> str:
    r = str(row.get("g4_term_ref", ""))
    if len(r) != 64:
        raise G4S1Error("G4:S0 V2 challenge row missing certified g4_term_ref")
    return r


def verify_s1_authority(
    s0_result: Mapping[str, Any],
    s0_replay_comparison: Mapping[str, Any],
    s0_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if s0_result.get("schema_id") != "IG_G4_S0_INTERFACE_EXTRACTION_RESULT_V2" or s0_result.get("status") != "PASS":
        failures.append("S0_V2_SCHEMA_OR_STATUS")
    if s0_result.get("g4_s1_unlocked") is not True or s0_result.get("g4_graduated") is not False:
        failures.append("S0_UNLOCK_OR_GRADUATION")
    if s0_result.get("s0_materializes_reusable_base") is not True or s0_result.get("s1_lower_layer_rematerialization_forbidden") is not True:
        failures.append("S0_BASE_MATERIALIZATION_CONTRACT")
    if s0_result.get("baseline_public_observer") != "CAPS7_ONLY_INHERITED_FROM_GRADUATED_G3":
        failures.append("S0_PUBLIC_OBSERVER")
    if s0_result.get("topology_visibility") != "HIDDEN_CHALLENGE_EVIDENCE_ONLY_NOT_PROMOTED":
        failures.append("S0_TOPOLOGY_FIREWALL")
    if s0_result.get("shell_profile_visibility") != "HIDDEN_CHALLENGE_EVIDENCE_ONLY_NOT_PROMOTED":
        failures.append("S0_SHELL_FIREWALL")

    corpus = s0_result.get("certified_term_corpus", {})
    if corpus.get("schema_id") != "IG_G4_S0_CERTIFIED_G3_TERM_CORPUS_V1" or corpus.get("status") != "PASS":
        failures.append("S0_TERM_CORPUS_SCHEMA_OR_STATUS")
    if corpus.get("lower_layer_rematerialization") is not False or corpus.get("semantics") != "PREVIOUS_LAYER_G2_UNITS_ATOMIC_AT_EARNED_CAPS7_INTERFACE":
        failures.append("S0_TERM_CORPUS_ATOMICITY")
    try:
        terms = {k: G3TermState.from_wire(v["term"]) for k, v in corpus.get("terms", {}).items()}
        if set(terms) != {"A", "B"}:
            failures.append("S0_TERM_CORPUS_CARDINALITY")
        elif terms["A"].total_caps != terms["B"].total_caps:
            failures.append("S0_TERM_CORPUS_CAPS7_COLLISION_LOST")
    except Exception:
        failures.append("S0_TERM_CORPUS_REHYDRATION")

    # Certification authority is dynamic: S1 binds to the exact S0 result/replay/closeout
    # supplied by the certified run instead of hardcoding one source release forever.
    if s0_replay_comparison.get("schema_id") != "IG_G4_S0_COLD_REPLAY_COMPARISON_V2" or s0_replay_comparison.get("status") != "PASS":
        failures.append("S0_REPLAY_SCHEMA_OR_STATUS")
    if s0_replay_comparison.get("certification") != "CERTIFIED_PASS" or s0_replay_comparison.get("same_registered_decoder_native_experiment") is not True:
        failures.append("S0_REPLAY_CERTIFICATION")
    if not all(s0_replay_comparison.get(k) is True for k in ("science_sha_equal", "source_sha_equal", "registry_sha_equal")):
        failures.append("S0_REPLAY_NOT_EXACT")
    if s0_replay_comparison.get("primary_science_sha256") != s0_result.get("science_sha256") or s0_replay_comparison.get("cold_science_sha256") != s0_result.get("science_sha256"):
        failures.append("S0_REPLAY_SCIENCE_BINDING")
    if s0_replay_comparison.get("primary_source_sha256") != s0_result.get("source_sha256") or s0_replay_comparison.get("cold_source_sha256") != s0_result.get("source_sha256"):
        failures.append("S0_REPLAY_SOURCE_BINDING")
    if s0_replay_comparison.get("primary_registry_sha256") != s0_result.get("registry_sha256") or s0_replay_comparison.get("cold_registry_sha256") != s0_result.get("registry_sha256"):
        failures.append("S0_REPLAY_REGISTRY_BINDING")

    if s0_closeout.get("schema_id") != "IG_G4_S0_CERTIFIED_CLOSEOUT_V2" or s0_closeout.get("status") != "CERTIFIED_PASS":
        failures.append("S0_CLOSEOUT_SCHEMA_OR_STATUS")
    if s0_closeout.get("experiment_id") != "G4:S0" or s0_closeout.get("next_authorized_stage") != "G4:S1":
        failures.append("S0_CLOSEOUT_AUTHORIZATION")
    if s0_closeout.get("science_sha256") != s0_result.get("science_sha256"):
        failures.append("S0_CLOSEOUT_SCIENCE_BINDING")
    if s0_closeout.get("source_sha256") != s0_result.get("source_sha256") or s0_closeout.get("registry_sha256") != s0_result.get("registry_sha256"):
        failures.append("S0_CLOSEOUT_CODE_BINDING")
    if s0_closeout.get("certified_term_corpus_sha256") != corpus.get("science_sha256"):
        failures.append("S0_CLOSEOUT_TERM_CORPUS_BINDING")

    out = {
        "schema_id": "IG_G4_S1_AUTHORITY_VERIFICATION_V2",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "g4_s0_science_sha256": s0_result.get("science_sha256"),
        "g4_s0_source_sha256": s0_result.get("source_sha256"),
        "g4_s0_registry_sha256": s0_result.get("registry_sha256"),
        "certified_term_corpus_sha256": corpus.get("science_sha256"),
        "g4_phase0_spec_sha256": phase0_spec()["science_sha256"],
        "g4_s1_preregistered_spec_sha256": s1_spec()["science_sha256"],
        "lower_layer_rematerialization": False,
        "promotion": False,
        "g4_graduated": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G4S1Error("G4:S1 authority failed: " + ",".join(failures))
    return out


def load_s0_term_states(s0_result: Mapping[str, Any]) -> tuple[dict[str, G3TermState], dict[str, Any]]:
    corpus = s0_result["certified_term_corpus"]
    pair = s0_result["certified_collision_pair"]
    states: dict[str, G3TermState] = {}
    topo = {}
    for key in ("A", "B"):
        row = pair[key]
        ref = _term_ref(row)
        term = G3TermState.from_wire(corpus["terms"][key]["term"])
        if term.construction_digest != ref:
            raise G4S1Error("S0 term ref/wire mismatch")
        if list(term.total_caps) != [int(x) for x in row["public_interface"]["coordinates"]]:
            raise G4S1Error("S0 term/public CAPS7 mismatch")
        states[ref] = term
        topo[ref] = str(row["hidden_challenge_diagnostic"]["topology_canon"])
    if len(states) != 2 or len({tuple(s.total_caps) for s in states.values()}) != 1:
        raise G4S1Error("S0 term challenge basis malformed")
    meta = {
        "schema_id": "IG_G4_S1_CERTIFIED_TERM_CORPUS_LOAD_V1",
        "status": "PASS",
        "carrier_refs": sorted(states),
        "exact_carrier_count": 2,
        "single_caps7_class": True,
        "carrier_input_mode": "CERTIFIED_G3_TERM_CORPUS_FROM_S0",
        "lower_layer_rematerialization": False,
        "g1_r100_materialization": False,
        "hidden_topology_used_as_context_input": False,
        "hidden_topology_labels": topo,
        "producer_experiment_id": "G4:S0",
        "producer_term_corpus_sha256": corpus["science_sha256"],
    }
    meta["science_sha256"] = canonical_sha256(meta)
    return states, meta


def _caps7(st: G3TermState | G4PairTermState) -> list[int]:
    caps = [int(x) for x in st.total_caps]
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise G4S1Error("malformed CAPS7 during G4:S1")
    return caps


def _reserve_relation(st: G3TermState | G4PairTermState, endpoint_type: int):
    return st.reserve_external_relation(int(endpoint_type))


def _reservation_profile_for_state(state: G4PairTermState) -> list[dict[str, Any]]:
    caps = _caps7(state)
    rows = []
    for t in range(7):
        rel = _reserve_relation(state, t) if caps[t] > 0 else tuple()
        succ_caps = sorted({tuple(_caps7(s)) for s in rel})
        rows.append({
            "endpoint_type": t,
            "available": caps[t] > 0,
            "relation_cardinality": len(rel),
            "successor_caps7_set": [list(x) for x in succ_caps],
        })
    return rows


def pair_context_kernel_measurement(*, left: G3TermState, right: G3TermState, operator: Sequence[int]) -> dict[str, Any]:
    a, b = map(int, operator)
    outputs = compose_g4_pair_relation(left, right, a, b)
    public_caps_set = sorted({tuple(_caps7(st)) for st in outputs})
    branch_payloads: dict[str, dict[str, Any]] = {}
    branch_hist: dict[str, int] = {}
    for st in outputs:
        bp = {"output_caps7": _caps7(st), "reservation_profile": _reservation_profile_for_state(st)}
        h = canonical_sha256(bp)
        branch_payloads.setdefault(h, bp)
        branch_hist[h] = branch_hist.get(h, 0) + 1
    return {
        "operator": [a, b],
        "relation_output_cardinality": len(outputs),
        "public_output_caps7_set": [list(x) for x in public_caps_set],
        "branch_profile_histogram": [
            {"branch_profile_sha256": h, "multiplicity": int(branch_hist[h]), "profile": branch_payloads[h]}
            for h in sorted(branch_hist)
        ],
    }


def expand_pair_context_kernel_signature(kernel_measurement: Mapping[str, Any], *, orientation: str) -> dict[str, Any]:
    if orientation not in {"TARGET_LEFT_CONTEXT_RIGHT", "CONTEXT_LEFT_TARGET_RIGHT"}:
        raise G4S1Error(f"bad G4:S1 orientation {orientation}")
    sig = {
        "schema_id": "IG_G4_S1_OPERATIONAL_PAIR_CONTEXT_SIGNATURE_V2",
        "operator": [int(x) for x in kernel_measurement["operator"]],
        "orientation": orientation,
        "relation_output_cardinality": int(kernel_measurement["relation_output_cardinality"]),
        "public_output_caps7_set": kernel_measurement["public_output_caps7_set"],
        "branch_profile_histogram": kernel_measurement["branch_profile_histogram"],
        "previous_layer_atomicity": "G3_TERM_BOUNDARY",
        "direct_hidden_tree_read": False,
        "direct_shell_profile_read": False,
        "lower_layer_implementation_read": False,
        "output_construction_identity_recorded": False,
    }
    sig["science_sha256"] = canonical_sha256(sig)
    return sig


def _tier_pair_analysis(s0_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    pair = s0_result["certified_collision_pair"]
    A = pair["A"]["hidden_challenge_diagnostic"]
    B = pair["B"]["hidden_challenge_diagnostic"]
    rows = []
    for tier in challenge_tiers()["ordered_tiers"]:
        n = int(tier["tier"])
        if n == 0:
            va, vb = pair["A"]["public_interface"]["coordinates"], pair["B"]["public_interface"]["coordinates"]
        elif n == 1:
            keys = ["diameter", "radius", "articulation_count", "wiener_index"]
            da, db = [int(x) for x in A["degree_sequence"]], [int(x) for x in B["degree_sequence"]]
            va = {"MAX_DEGREE": max(da), "LEAF_COUNT": sum(1 for x in da if x == 1), **{k: A[k] for k in keys}}
            vb = {"MAX_DEGREE": max(db), "LEAF_COUNT": sum(1 for x in db if x == 1), **{k: B[k] for k in keys}}
        elif n == 2:
            va, vb = A["degree_sequence"], B["degree_sequence"]
        elif n == 3:
            va, vb = {"degree_sequence": A["degree_sequence"], "wiener_index": A["wiener_index"]}, {"degree_sequence": B["degree_sequence"], "wiener_index": B["wiener_index"]}
        elif n == 4:
            va, vb = A["shell_profile_multiset"], B["shell_profile_multiset"]
        elif n == 5:
            va, vb = A["topology_canon"], B["topology_canon"]
        else:
            raise G4S1Error(f"unknown preregistered tier {n}")
        rows.append({"tier": n, "name": tier["name"], "pair_equal": va == vb, "A_value": va, "B_value": vb})
    return rows


def finalize_s1_result(*, s0_result: Mapping[str, Any], authority: Mapping[str, Any], reproduction: Mapping[str, Any], task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spec = s1_spec(); pair = s0_result["certified_collision_pair"]
    refs = sorted([_term_ref(pair["A"]), _term_ref(pair["B"])])
    expected_tasks = len(refs) * len(refs) * 31 * 2
    if len(task_rows) != expected_tasks:
        raise G4S1Error(f"G4:S1 expected {expected_tasks} task rows, got {len(task_rows)}")
    per_target = {r: [] for r in refs}; row_by_key = {}
    for row in task_rows:
        tr, cr, ori = str(row["target_ref"]), str(row["context_ref"]), str(row["orientation"])
        a, b = map(int, row["operator"]); key = (tr, cr, ori, a, b)
        if key in row_by_key: raise G4S1Error("duplicate G4:S1 task key")
        row_by_key[key] = row
        per_target[tr].append({"context_ref": cr, "orientation": ori, "operator": [a, b], "operational_signature_sha256": str(row["operational_signature"]["science_sha256"])})
    topo_by_ref = {_term_ref(pair["A"]): str(pair["A"]["hidden_challenge_diagnostic"]["topology_canon"]), _term_ref(pair["B"]): str(pair["B"]["hidden_challenge_diagnostic"]["topology_canon"])}
    behavior_hash = {}; summaries = []
    for tr in refs:
        rows = sorted(per_target[tr], key=lambda x: (x["context_ref"], x["orientation"], x["operator"]))
        bh = canonical_sha256(rows); behavior_hash[tr] = bh
        summaries.append({"target_ref": tr, "frozen_hidden_topology_label": topo_by_ref[tr], "behavior_signature_sha256": bh, "context_row_count": len(rows)})
    ra, rb = refs; sep = None
    for ka in sorted(k for k in row_by_key if k[0] == ra):
        _, ctx, ori, a, b = ka; kb = (rb, ctx, ori, a, b)
        xa, xb = row_by_key[ka], row_by_key[kb]
        if xa["operational_signature"]["science_sha256"] != xb["operational_signature"]["science_sha256"]:
            sep = {"target_A_ref": ra, "target_B_ref": rb, "context_ref": ctx, "orientation": ori, "operator": [a, b], "signature_A": xa["operational_signature"], "signature_B": xb["operational_signature"], "meaning": "Same certified CAPS7 G3 term, same admitted G4 context/operator, different topology-blind operational relation signature."}
            break
    tier_analysis = _tier_pair_analysis(s0_result) if sep else []
    lowest = next((r for r in tier_analysis if int(r["tier"]) > 0 and not r["pair_equal"]), None)
    outcome = "STRUCTURE_NOT_READ" if sep is None else "STRUCTURE_READ_EARNED"
    classification = "G4_STRUCTURE_NOT_READ_ON_CERTIFIED_G3_TERM_PAIR_CONTEXT_BASIS_CAPS7_SURVIVES_S2_UNLOCKED" if sep is None else "G4_STRUCTURE_READ_EARNED_ON_CERTIFIED_G3_TERM_PAIR_CONTEXT_BASIS_TIER_ANALYSIS_REQUIRED_S2_UNLOCKED"
    result = {
        "schema_id": "IG_G4_S1_HIDDEN_STRUCTURE_CONTEXT_READ_RESULT_V2", "status": "PASS", "classification": classification, "outcome": outcome,
        "g4_started": True, "g4_graduated": False, "g4_s2_unlocked": True, "promotion": False, "topology_promoted": False, "shell_profile_promoted": False,
        "authority": dict(authority), "phase0_spec_sha256": phase0_spec()["science_sha256"], "s1_preregistered_spec_sha256": spec["science_sha256"], "challenge_read_tier_science_sha256": challenge_tiers()["science_sha256"],
        "challenge_reproduction": dict(reproduction),
        "context_basis": {"target_count": len(refs), "context_partner_count": len(refs), "operator_count": 31, "orientation_count": 2, "task_count": len(task_rows), "all_targets_and_contexts_from_certified_s0_term_corpus": True, "lower_layer_rematerialization": False},
        "target_behavior_summaries": summaries, "same_caps7_behavior_equal": behavior_hash[ra] == behavior_hash[rb], "separation_witness": sep,
        "post_separation_preregistered_tier_pair_analysis": tier_analysis, "lowest_pair_separating_preregistered_tier": None if lowest is None else {"tier": lowest["tier"], "name": lowest["name"]},
        "tier_promotion_status": "NOT_PROMOTED_BY_S1; S2_MUST_TEST_SUFFICIENCY" if sep else "NOT_APPLICABLE_NO_SEPARATION",
        "operational_observer": {"allowed_outputs": list(spec["allowed_observer_outputs"]), "forbidden_direct_reads": list(spec["forbidden_direct_reads"])},
        "next_authorized_stage": "G4:S2", "nonclaims": list(spec["nonclaims"]),
    }
    result["science_sha256"] = canonical_sha256(result)
    return result


# Tiny serialized worker context; safe under spawn/forkserver/fork.
_WORKER_STATES: dict[str, G3TermState] | None = None


def init_g4_s1_term_worker(payload: Mapping[str, Any]) -> None:
    global _WORKER_STATES
    _WORKER_STATES = {str(ref): G3TermState.from_wire(wire) for ref, wire in payload["states"].items()}


def g4_s1_term_kernel_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_STATES is None: raise G4S1Error("G4:S1 term worker context absent")
    lr, rr = str(payload["left_ref"]), str(payload["right_ref"])
    if lr not in _WORKER_STATES or rr not in _WORKER_STATES: raise G4S1Error("unknown G4:S1 term ref")
    op = [int(x) for x in payload["operator"]]
    return {"left_ref": lr, "right_ref": rr, "operator": op, "kernel_measurement": pair_context_kernel_measurement(left=_WORKER_STATES[lr], right=_WORKER_STATES[rr], operator=op)}



def compare_cold_replay(primary: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    """Certify exact primary/cold replay identity for registered G4:S1 V2."""
    failures: list[str] = []
    for label, obj in (("PRIMARY", primary), ("COLD", cold)):
        if obj.get("schema_id") != "IG_G4_S1_HIDDEN_STRUCTURE_CONTEXT_READ_RESULT_V2" or obj.get("status") != "PASS":
            failures.append(f"{label}_SCHEMA_OR_STATUS")
    checks = {
        "science_sha_equal": primary.get("science_sha256") == cold.get("science_sha256"),
        "source_sha_equal": primary.get("source_sha256") == cold.get("source_sha256"),
        "registry_sha_equal": primary.get("registry_sha256") == cold.get("registry_sha256"),
        "classification_equal": primary.get("classification") == cold.get("classification"),
        "outcome_equal": primary.get("outcome") == cold.get("outcome"),
        "behavior_summaries_equal": primary.get("target_behavior_summaries") == cold.get("target_behavior_summaries"),
        "separation_witness_equal": primary.get("separation_witness") == cold.get("separation_witness"),
        "tier_analysis_equal": primary.get("post_separation_preregistered_tier_pair_analysis") == cold.get("post_separation_preregistered_tier_pair_analysis"),
    }
    for name, ok in checks.items():
        if not ok:
            failures.append(name.upper())
    out = {
        "schema_id": "IG_G4_S1_COLD_REPLAY_COMPARISON_V1",
        "status": "PASS" if not failures else "FAIL",
        "certification": "CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
        "failures": failures,
        "experiment_id": "G4:S1",
        "same_registered_decoder_native_experiment": True,
        **checks,
        "primary_science_sha256": primary.get("science_sha256"),
        "cold_science_sha256": cold.get("science_sha256"),
        "primary_source_sha256": primary.get("source_sha256"),
        "cold_source_sha256": cold.get("source_sha256"),
        "primary_registry_sha256": primary.get("registry_sha256"),
        "cold_registry_sha256": cold.get("registry_sha256"),
        "lower_layer_rematerialization": False,
        "stage_specific_external_science_runner": False,
    }
    out["comparison_sha256"] = canonical_sha256(out)
    return out


def certified_closeout(primary: Mapping[str, Any], cold: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    """Close G4:S1 only after exact replay certification."""
    failures: list[str] = []
    if replay.get("schema_id") != "IG_G4_S1_COLD_REPLAY_COMPARISON_V1" or replay.get("certification") != "CERTIFIED_PASS":
        failures.append("REPLAY")
    for k in (
        "science_sha_equal", "source_sha_equal", "registry_sha_equal", "classification_equal", "outcome_equal",
        "behavior_summaries_equal", "separation_witness_equal", "tier_analysis_equal",
    ):
        if replay.get(k) is not True:
            failures.append(k.upper())
    if primary.get("science_sha256") != cold.get("science_sha256"):
        failures.append("SCIENCE")
    out = {
        "schema_id": "IG_G4_S1_CERTIFIED_CLOSEOUT_V1",
        "status": "CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
        "failures": failures,
        "experiment_id": "G4:S1",
        "decoder_version": "0.30.37",
        "science_sha256": primary.get("science_sha256"),
        "source_sha256": primary.get("source_sha256"),
        "registry_sha256": primary.get("registry_sha256"),
        "classification": primary.get("classification"),
        "outcome": primary.get("outcome"),
        "comparison_sha256": replay.get("comparison_sha256"),
        "g4_s2_unlocked": primary.get("g4_s2_unlocked") is True,
        "g4_graduated": False,
        "topology_promoted": primary.get("topology_promoted") is True,
        "shell_profile_promoted": primary.get("shell_profile_promoted") is True,
        "carrier_input_mode": "CERTIFIED_G3_TERM_CORPUS_FROM_S0",
        "lower_layer_rematerialization": False,
        "next_authorized_stage": "G4:S2" if not failures else None,
        "stage_specific_external_science_runner": False,
    }
    out["closeout_sha256"] = canonical_sha256(out)
    return out
