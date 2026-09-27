from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .hashing import sha256_file


SPEC_MAP = {
    "pregeometry": "PREGEOMETRY_SCOUT_SPEC_v1.json",
    "meta-grammar": "O_CHAIN_META_GRAMMAR_SPEC_v1.json",
    "o10-plus": "O10_PLUS_TARGETED_PROBE_MATRIX_v1.json",
}


def load_pregeometry_spec(name: str) -> dict[str, Any]:
    if name not in SPEC_MAP:
        raise KeyError(name)
    p = files("infinity_grid").joinpath("resources/decoder/" + SPEC_MAP[name])
    return json.loads(p.read_text(encoding="utf-8"))


def _find(root: Path, name: str) -> Path:
    root = Path(root)
    if root.is_file():
        if root.name != name:
            raise RuntimeError(f"expected {name}, got file {root.name}")
        return root
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(hits)}")
    return hits[0]


def _read_json(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    p = _find(root, name)
    return p, json.loads(p.read_text(encoding="utf-8"))


def _science_pin(path: Path, obj: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    if obj is not None and obj.get("science_sha256"):
        out["science_sha256"] = obj["science_sha256"]
    return out


def _all_equal(values: list[str]) -> bool:
    return bool(values) and len(set(values)) == 1


def _grrl_has(spec: dict[str, Any], key: str, needle: str) -> bool:
    return needle.lower() in str(spec.get("hypotheses", {}).get(key, "")).lower()


def run_pregeometry_scout(
    *,
    o7_graduation_root: Path,
    post_o7_root: Path,
    grrl_root: Path,
    phase8_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Phase-B evidence-only pregeometry/meta-grammar audit.

    This runner intentionally generates no O10 carrier.  It reconciles already-earned
    O7/O8/O9 evidence with the frozen GRRL theorem, separates primitive hidden-read
    questions from same-grammar collective-service questions, and freezes the smallest
    next-level probe matrix.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    phase_spec = load_pregeometry_spec("pregeometry")
    meta_spec = load_pregeometry_spec("meta-grammar")
    probe_spec = load_pregeometry_spec("o10-plus")

    p_o7, o7 = _read_json(Path(o7_graduation_root), "O7_PHASE2_GRADUATION_AUDIT_RESULT.json")
    p_machine = _find(Path(o7_graduation_root), "GENERIC_THEOREM_STATUS.txt")
    machine_text = p_machine.read_text(encoding="utf-8", errors="replace")
    p_post, post = _read_json(Path(post_o7_root), "POST_O7_STRUCTURAL_AUDIT_RECONCILED_RESULT.json")
    p_theorem, theorem = _read_json(Path(grrl_root), "GRRL_THEOREM_SPEC_v1.json")
    p_c3, c3 = _read_json(Path(grrl_root), "C3_GRAMMAR_DELTA.json")
    p_o8bp0, o8bp0 = _read_json(Path(phase8_root), "O8_BP0_RESULT.json")
    p_o8, o8 = _read_json(Path(phase8_root), "O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json")
    p_o9, o9 = _read_json(Path(phase8_root), "O9_BP0_RESULT.json")
    p_stop, stop = _read_json(Path(phase8_root), "O_FRONTIER_AUTO_ADVANCE_RESULT.json")

    expected_grammar = "5a2b34df6572a10648d46936da81228c606406d09b5b4c7e9603b624527d5a07"
    normalized_hashes = dict(c3.get("hashes", {}))
    normalized_hashes["O8"] = o8bp0.get("normalized_grammar", {}).get("candidate_sha256")
    normalized_hashes["O9"] = o9.get("normalized_grammar", {}).get("candidate_sha256")

    machine_checked = (
        "GENERIC_O_HIERARCHY_LIFT_MACHINE_CHECKED_ABSTRACT_SCHEMA" in machine_text
        and "sorryAx: absent" in machine_text
        and "Fresh cold default build: PASS" in machine_text
    )
    authority_checks = {
        "O7_graduated": o7.get("status") == "O7_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V2_7_PHASE2",
        "post_O7_reconciled": post.get("status") == "PASS" and post.get("geometry_status") == "GEOMETRY_NOT_EARNED",
        "GRRL_spec": theorem.get("schema") == "IG_GRRL_THEOREM_SPEC_V1" and theorem.get("status") == "FROZEN_CONDITIONAL_THEOREM_SPEC",
        "GRRL_machine": machine_checked,
        "O4_O7_grammar_delta": c3.get("status") == "PASS" and c3.get("classification") == "PARAMETRIC_LEVEL_EXTENSION_ONLY",
        "O8_BP0_repeat": o8bp0.get("outcome") == "O8_EXISTS_GRRL_REPEAT_STOP" and o8bp0.get("normalized_grammar", {}).get("delta") == "EMPTY",
        "O8_graduated": o8.get("status") == "O8_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_THEOREM_ACCELERATED_V2_8",
        "O9_sentinel_repeat": o9.get("status") == "PASS" and o9.get("outcome") == "O9_EXISTS_GRRL_REPEAT_STOP" and o9.get("normalized_grammar", {}).get("delta") == "EMPTY",
        "fixed_grammar_stop": stop.get("status") == "PASS" and stop.get("classification") == "FIXED_GRAMMAR_INDUCTION_STOP" and stop.get("automatic_O10_execution") is False,
        "grammar_hash_O4_O9": _all_equal([v for v in normalized_hashes.values() if v]) and set(normalized_hashes) == {"O4", "O5", "O6", "O7", "O8", "O9"} and set(normalized_hashes.values()) == {expected_grammar},
    }
    authority_ok = all(authority_checks.values())
    if not authority_ok:
        raise RuntimeError(f"Phase-B authority gate failed: {[k for k,v in authority_checks.items() if not v]}")

    h9_no_hidden_read = _grrl_has(theorem, "H9", "no admitted action reads hidden topology")
    h10_monotone = _grrl_has(theorem, "H10", "no deletion") and _grrl_has(theorem, "H10", "feedback")
    c10_not_type_fixed = any("C10" in str(x) and "not a type fixed point" in str(x) for x in theorem.get("conclusions", []))
    o8_no_read = o8.get("gates", {}).get("G7_nontrivial_hiding_and_read_gate", {}).get("operational_read") is False
    o9_no_read = o9.get("emergent_read_gate", {}).get("R3_new_operational_read_found") is False
    o9_repr = o9.get("representation_gate", {})
    no_grammar_break = (
        o9_repr.get("status") == "PASS"
        and not o9_repr.get("new_hidden_read", True)
        and not o9_repr.get("new_state_fields")
        and not o9_repr.get("relation_arity_change", True)
        and not o9_repr.get("orientation_or_order_read", True)
        and not o9_repr.get("deletion_or_rewiring", True)
        and not o9_repr.get("counter_semantics_change", True)
    )

    function_rows = [
        {
            "id": "outer_adjacency",
            "current_rank": "R2_OBSERVER_SEPARATING_NOT_R3_READ",
            "basis": "intrinsic E7 relation + exact transport laws; same-skin/different-topology witnesses at O7/O8/O9; frozen resource action language does not consult adjacency",
        },
        {
            "id": "graph_distance_d_n",
            "current_rank": "R2_OBSERVER_SEPARATING_NOT_R3_READ",
            "basis": "intrinsic shortest E7-path distance and exact update laws under relation-add/one-cross; not an admitted control input",
        },
        {
            "id": "shell_growth",
            "current_rank": "R2_OBSERVER_SEPARATING_NOT_R3_READ",
            "basis": "intrinsic graph-distance shells and exact update consequences; observer-visible only in current scope",
        },
        {
            "id": "cycle_rank_beta_n",
            "current_rank": "R2_OBSERVER_SEPARATING_NOT_R3_READ",
            "basis": "exact pairwise-relation accounting and topology-aware separation; not read by frozen resource control",
        },
        {
            "id": "local_euler_defect_kappa_n",
            "current_rank": "R2_OBSERVER_SEPARATING_NOT_R3_READ",
            "basis": "intrinsic local defect with exact sum/update law; no control dependence earned",
        },
        {
            "id": "nested_ownership_separation_rank",
            "current_rank": "R0_DEFINABLE_CANDIDATE_NOT_PROMOTED",
            "basis": "hierarchical ownership gives a definable separation-depth candidate; no exact R1/R2/R3 promotion is presently sealed",
        },
    ]

    primitive_read_lane = {
        "status": "NO_PRIMITIVE_R3_R4_UNDER_FROZEN_GRRL",
        "H9_no_hidden_read": h9_no_hidden_read,
        "H10_fixed_monotone_pairwise_grammar": h10_monotone,
        "O8_operational_read": False if o8_no_read else None,
        "O9_new_operational_read": False if o9_no_read else None,
        "representation_break": False if no_grammar_break else None,
        "conclusion": "Within the frozen GRRL grammar, a primitive hidden-topology/role/direction read cannot arise merely by increasing O depth: H9 excludes it from admitted actions. Any exact R3 hit of that kind is therefore a theorem-reopen/grammar-break event, not an unnoticed continuation of the same grammar.",
        "scope": "frozen GRRL observer/action language only; a changed observer/action language may reveal new distinctions",
    }

    collective_service_lane = {
        "status": "OPEN_NOT_DECIDED_BY_EXISTING_O7_O9_FRONTIER_EVIDENCE",
        "why_not_closed_by_GRRL": "GRRL transports the fixed local resource grammar and same-q finite futures within each lifted level; it does not prove cross-depth service equivalence, depth stabilization, or a carrier/type fixed point.",
        "allowed_novelty_under_same_grammar": [
            "new finite branch capability visible in q-level futures",
            "stable selective-access or coordination service built from repeated local actions",
            "depth-dependent role/route service under a newly frozen service observer",
            "convergence/periodicity of a coarse service signature"
        ],
        "positive_evidence_required": "exact matched intervention or finite-context separator plus Local Sufficiency certificate; descriptor movement alone is rejected",
        "existing_evidence_reused": "O8/O9 topology twins establish hidden structure but not collective service emergence; the quarantined descriptor-only Phase9 side experiment is not used.",
    }

    local_sufficiency = {
        "status": "WAITING_FOR_EXACT_COLLECTIVE_SERVICE_HIT",
        "gate": "For every future positive service witness, find the smallest O depth, subcarrier and frozen context/action set that still separates the behavior. Promotion fails if the claimed service disappears under that exact local restriction or is reducible to inherited service data.",
    }
    spectroscope = {
        "status": "NOT_RUN_NO_NEW_BEHAVIORAL_HIT",
        "rule": "Recognition-only. It may classify an already-earned relation/service hit but may not select the discovery target or promote geometry.",
    }

    gates = {
        "P0_normalized_grammar_plateau": {
            "status": "EARNED",
            "levels": ["O4", "O5", "O6", "O7", "O8", "O9"],
            "normalized_grammar_sha256": expected_grammar,
            "authority_note": "O4-O7 normalized by the O7 C3 audit; O8 by exact BP0/theorem application and graduation; O9 by bounded existence/repeat sentinel. O9 is not graduated.",
        },
        "P1_generic_recursive_transport": {
            "status": "EARNED_CONDITIONALLY_MACHINE_CHECKED",
            "machine_status": "GENERIC_O_HIERARCHY_LIFT_MACHINE_CHECKED_ABSTRACT_SCHEMA",
            "scope": "synthetic fixed-grammar recursion for CertifiedOverlayModule/LiftPresentation hypotheses; not historical existence of every level",
        },
        "P2_material_instances": {
            "status": "EARNED_THROUGH_O9_WITH_MIXED_AUTHORITY",
            "O7": "FULL_GRADUATED_ANCHOR",
            "O8": "THEOREM_ACCELERATED_GRADUATED",
            "O9": "BOUNDED_EXISTENCE_REPEAT_SENTINEL_NOT_GRADUATED",
        },
        "P3_carrier_type_fixed_point": {
            "status": "NOT_EARNED",
            "GRRL_C10": "recurrence is stratified self-similarity, not a type fixed point" if c10_not_type_fixed else "C10_NOT_VERIFIED",
        },
        "P4_depth_independent_service_quotient": {"status": "OPEN"},
        "P5_meta_composition_substitution": {"status": "OPEN"},
        "P6_meta_full_abstraction_minimality": {"status": "OPEN"},
    }

    meta_result = {
        "schema": "IG_O_CHAIN_META_GRAMMAR_RESULT_V1",
        "date": "2026-08-30",
        "status": "PASS",
        "classification": "STRATIFIED_FIXED_GRAMMAR_PLATEAU_EARNED_TYPE_FIXED_POINT_NOT_EARNED",
        "regime_name": "O_GRRL_FIXED_GRAMMAR_REGIME",
        "normalized_grammar_hashes": normalized_hashes,
        "gates": gates,
        "raised_O_group_status": "DESCRIPTIVE_META_REGIME_ONLY_NOT_NEW_ALGEBRAIC_CARRIER",
        "interpretation": "The O4-O9 evidence supports a plateau of the normalized recursive grammar: level number is currently a stratification/depth parameter inside one repeated construction regime. This is not a proof that O_n and O_(n+1) are the same carrier type, and it does not erase the ownership layer. A genuine raised O-group requires at least a depth-independent exact service quotient and an exact meta-composition/substitution theorem.",
        "nonclaims": ["historical O10+ existence", "infinite O-tower", "type fixed point", "geometry"],
    }
    meta_result["science_sha256"] = canonical_sha256({k: v for k, v in meta_result.items() if k != "science_sha256"})

    probe_matrix = dict(probe_spec)
    probe_matrix["execution"] = {
        "O10_run": False,
        "O11_run": False,
        "O12_run": False,
        "reason": "Phase B first freezes what would constitute same-grammar collective-service novelty or a grammar break. Existing evidence is sufficient for the baseline classification and insufficient to justify blind level generation.",
        "first_recommended_probe": "DEPTH_SERVICE_SIGNATURE",
        "first_use_existing_data": "Freeze the service observer and exhaust what can be decided from O8/O9 retained evidence before constructing an O10 sentinel.",
    }
    probe_matrix["science_sha256"] = canonical_sha256({k: v for k, v in probe_matrix.items() if k != "science_sha256"})

    source_pins = {
        "schema": "IG_PHASE_B_SOURCE_PINS_V1",
        "files": [
            _science_pin(p_o7, o7), _science_pin(p_machine), _science_pin(p_post, post),
            _science_pin(p_theorem, theorem), _science_pin(p_c3, c3), _science_pin(p_o8bp0, o8bp0),
            _science_pin(p_o8, o8), _science_pin(p_o9, o9), _science_pin(p_stop, stop),
        ],
        "specs": {
            "pregeometry": canonical_sha256(phase_spec),
            "meta_grammar": canonical_sha256(meta_spec),
            "o10_plus": canonical_sha256(load_pregeometry_spec("o10-plus")),
        },
    }
    source_pins["science_sha256"] = canonical_sha256({k: v for k, v in source_pins.items() if k != "science_sha256"})

    result = {
        "schema": "IG_PREGEOMETRY_SCOUT_RESULT_V1",
        "date": "2026-08-30",
        "status": "PASS",
        "phase_status": "PHASE_B_PREGEOMETRY_META_GRAMMAR_BASELINE_COMPLETE",
        "authority_gate": {"status": "PASS", "checks": authority_checks},
        "source_pins_sha256": source_pins["science_sha256"],
        "method": {
            "scanner_template": "relation-first; theorem/reuse before generation; Local Sufficiency before promotion",
            "spectroscope": "recognition-only after exact behavioral hit",
            "broad_census_cases": 0,
            "new_O_levels_generated": 0,
            "quarantined_phase9_side_experiment_used": False,
        },
        "intrinsic_pregeometry": {
            "status": "PRESENT",
            "geometry_status": "GEOMETRY_NOT_EARNED",
            "registered_functions": function_rows,
        },
        "lane_A_primitive_read_break": primitive_read_lane,
        "lane_B_same_grammar_collective_service": collective_service_lane,
        "local_sufficiency_gate": local_sufficiency,
        "spectroscope_gate": spectroscope,
        "meta_grammar": {
            "classification": meta_result["classification"],
            "science_sha256": meta_result["science_sha256"],
            "raised_O_group_status": meta_result["raised_O_group_status"],
        },
        "O10_plus": {
            "status": "FROZEN_TARGETED_MATRIX_NOT_EXECUTED",
            "science_sha256": probe_matrix["science_sha256"],
            "O10_run": False,
        },
        "main_conclusion": "The O-chain has earned a stratified fixed-grammar plateau, not a carrier/type fixed point. Intrinsic pregeometry is already present, while primitive topology reads are excluded by the current GRRL action contract. The unresolved scientific frontier is same-grammar collective service emergence and, separately, explicit grammar-break events. O10 is not justified until one of those questions is frozen and cannot be decided from existing O8/O9 evidence.",
        "next": "FREEZE_DEPTH_SERVICE_OBSERVER_AND_TEST_EXISTING_O8_O9_EVIDENCE_FIRST; ONLY_IF_UNDECIDED_BUILD_MINIMAL_O10_SENTINEL",
        "nonclaims": phase_spec["forbidden_claims"],
    }
    result["science_sha256"] = canonical_sha256({k: v for k, v in result.items() if k != "science_sha256"})

    write_json_atomic(output / "PREGEOMETRY_SCOUT_RESULT.json", result)
    write_json_atomic(output / "O_CHAIN_META_GRAMMAR_RESULT.json", meta_result)
    write_json_atomic(output / "O10_PLUS_TARGETED_PROBE_MATRIX.json", probe_matrix)
    write_json_atomic(output / "SOURCE_PINS.json", source_pins)

    report = f"""Infinity Grid Algebra Decoder — Phase B Pregeometry / O-chain Meta-Grammar\nDate: 2026-08-30\n\nSTATUS\n{result['phase_status']}\n\nMAIN RESULT\n{result['main_conclusion']}\n\nMETA-GRAMMAR\n{meta_result['classification']}\nNormalized grammar O4-O9: {expected_grammar}\nRaised O-group: {meta_result['raised_O_group_status']}\n\nPREGEOMETRY\nIntrinsic pregeometry: PRESENT\nGeometry: GEOMETRY_NOT_EARNED\nPrimitive hidden-topology R3/R4 under frozen GRRL: ABSENT BY CONTRACT / NO HIT\nSame-grammar collective-service emergence: OPEN\n\nEXECUTION\nBroad census cases: 0\nNew O levels generated: 0\nO10 run: NO\n\nNEXT\n{result['next']}\n\nScience SHA-256: {result['science_sha256']}\n"""
    (output / "PHASE_B_HUMAN_REPORT.txt").write_text(report, encoding="utf-8")
    return result
