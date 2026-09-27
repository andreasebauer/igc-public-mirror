from __future__ import annotations

"""O7 compatibility adapter for the generic scientific protocol layer.

The adapter is intentionally non-authoritative for new science.  It decomposes the
v0.28.8 trust-repair corrected O7-O13 scanner evidence into independent TestSpec projections and
proves that those projections reconstruct the frozen legacy scientific evidence exactly.
The historical v0.26 fixture remains preserved as provenance; active compatibility replay uses the separately frozen v0.28.8 corrected observer fixture.
"""

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, Callable
import json

from .canon import canonical_sha256
from .scientific_architecture import (
    ObservationView,
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    TestExecution,
    build_finding,
    make_ref,
    thaw_json,
)


ENTITY_REF = "IG_O7_ORGANIZATION_CARRIER_COMPAT@1.0.0"
REGIME_REF = "IG_O7_PLUS_ORGANIZATIONAL_REGIME_COMPAT@1.0.0"
PACK_REF = "IG_TEST_PACK_O7_MATURATION_COMPAT@1.0.0"
PROVIDER_REF = "IG_O7_CAPABILITY_PROVIDER_COMPAT@1.0.0"

TEST_GRAMMAR = "IG_TEST_O7_GRAMMAR_GUARD_COMPAT@1.0.0"
TEST_DIVERSITY = "IG_TEST_O7_DIVERSITY_COMPAT@1.0.0"
TEST_BRANCHING = "IG_TEST_O7_BRANCHING_COMPAT@1.0.0"
TEST_SYMMETRY = "IG_TEST_O7_SYMMETRY_COMPAT@1.0.0"
TEST_OVERLAP = "IG_TEST_O7_OVERLAP_GLUING_COMPAT@1.0.0"
TEST_LINEAGE = "IG_TEST_O7_LINEAGE_COMPAT@1.0.0"
TEST_QUOTIENT = "IG_TEST_O7_QUOTIENT_OBSERVER_COMPAT@1.0.0"
TEST_TOPOLOGY = "IG_TEST_O7_TOPOLOGY_SERVICES_COMPAT@1.0.0"
TEST_OBSTRUCTION = "IG_TEST_O7_OBSTRUCTION_RELIEF_COMPAT@1.0.0"
TEST_SATURATION = "IG_TEST_O7_SATURATION_COMPAT@1.0.0"
TEST_REPRESENTATION = "IG_TEST_O7_REPRESENTATION_STABILITY_COMPAT@1.0.0"
TEST_BOUNDARY_SENTINEL = "IG_TEST_REGIME_BOUNDARY_SENTINEL_O7_COMPAT@1.0.0"

LEAF_TEST_REFS = (
    TEST_GRAMMAR,
    TEST_DIVERSITY,
    TEST_BRANCHING,
    TEST_SYMMETRY,
    TEST_OVERLAP,
    TEST_LINEAGE,
    TEST_QUOTIENT,
    TEST_TOPOLOGY,
    TEST_OBSTRUCTION,
    TEST_SATURATION,
    TEST_REPRESENTATION,
)


class O7CompatibilityError(ScientificArchitectureError):
    pass


def _legacy_fixture_path():
    return files("infinity_grid").joinpath(
        "resources/compatibility/fixtures/v0288/IG_O_REGIME_ADAPTIVE_SCANNER_RESULT_TRUST_REPAIR_V1.json"
    )


def load_frozen_legacy_scanner_fixture() -> dict[str, Any]:
    return json.loads(_legacy_fixture_path().read_text(encoding="utf-8"))


def legacy_level_science_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen legacy scientific evidence partition used for migration equivalence.

    Generation bookkeeping (build_meta) is deliberately not Test evidence.  Everything
    below is preserved exactly by the generic compatibility wrapper.
    """
    out = {
        "level": summary["level"],
        "states": summary["states"],
        "grammar_sha256": summary["grammar_sha256"],
        "diversity": summary["diversity"],
        "branching": summary["branching"],
        "symmetry": summary["symmetry"],
        "overlap_gluing": summary["overlap_gluing"],
        "lineage": summary["lineage"],
        "quotient_observer": summary["quotient_observer"],
        "topology_services": summary["topology_services"],
        "obstruction_relief": summary["obstruction_relief"],
        "raw_growth": summary["raw_growth"],
        "normalized_signature": summary["normalized_signature"],
        "normalized_signature_sha256": summary["normalized_signature_sha256"],
        "matched_backbone": summary["matched_backbone"],
        "genealogy": summary["genealogy"],
        "structural_shock_changed_families": summary["structural_shock_changed_families"],
    }
    if "inherited_law_events" in summary:
        out["inherited_law_events"] = summary["inherited_law_events"]
    if "exploratory_shock_events" in summary:
        out["exploratory_shock_events"] = summary["exploratory_shock_events"]
    return out


def _legacy_capability_payloads(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Typed compatibility projections from one frozen legacy level summary.

    The provider intentionally exposes only declared capability-shaped projections;
    a Test never receives the raw level summary.
    """
    normalized = summary["normalized_signature"]
    return {
        "EXACT_ENTITY_IDENTITY": {
            "level": summary["level"],
            "states": summary["states"],
            "normalized_signature_sha256": summary["normalized_signature_sha256"],
        },
        "BOUNDARY_INTERFACE": {
            "quotient_observer": summary["quotient_observer"],
            "matched_backbone": summary["matched_backbone"],
            "inherited_law_events": summary.get("inherited_law_events", []),
        },
        "RELATION_GRAPH": {
            "diversity": summary["diversity"],
            "overlap_gluing": summary["overlap_gluing"],
            "lineage": summary["lineage"],
            "topology_services": summary["topology_services"],
        },
        "TYPED_RELATIONS": {
            "grammar_sha256": summary["grammar_sha256"],
            "branching": summary["branching"],
            "topology_services": summary["topology_services"],
        },
        "TRANSITION_SYSTEM": {
            "grammar_sha256": summary["grammar_sha256"],
            "branching": summary["branching"],
            "quotient_observer": summary["quotient_observer"],
            "obstruction_relief": summary["obstruction_relief"],
            "matched_backbone": summary["matched_backbone"],
        },
        "ACTION_ENABLEDNESS": {
            "branching": summary["branching"],
            "obstruction_relief": summary["obstruction_relief"],
        },
        "ACTION_MULTIPLICITY": {"branching": summary["branching"]},
        "BLOCK_STRUCTURE": {"diversity": summary["diversity"]},
        "ANCESTRY": {"lineage": summary["lineage"]},
        "FACTORIZATION": {
            "lineage": summary["lineage"],
            "overlap_gluing": summary["overlap_gluing"],
        },
        "INTRINSIC_DISTANCE": {"topology_services": summary["topology_services"]},
        "NEIGHBORHOOD_SHELLS": {"topology_services": summary["topology_services"]},
        "CANONICAL_AUTOMORPHISMS": {"symmetry": summary["symmetry"]},
        "RESOURCE_COUNTERS": {
            "min_total_free_by_type": summary["diversity"]["min_total_free_by_type"],
            "raw_growth": summary["raw_growth"],
            "obstruction_relief": summary["obstruction_relief"],
            "matched_backbone": summary["matched_backbone"],
        },
        "OBSERVER_QUOTIENT": {
            "quotient_observer": summary["quotient_observer"],
            "normalized_signature": normalized,
            "normalized_signature_sha256": summary["normalized_signature_sha256"],
            "resource_skins": summary["diversity"]["resource_skins"],
            "organizational_classes": summary["diversity"]["organizational_classes"],
        },
    }


def build_legacy_observation_view(
    *,
    summary: Mapping[str, Any],
    test_ref: str,
    source_science_sha256: str,
    registry: ScientificProtocolRegistry | None = None,
) -> ObservationView:
    registry = registry or ScientificProtocolRegistry()
    spec = registry.test(test_ref)
    if ENTITY_REF not in spec["admissibility"]["entity_class_refs"]:
        raise O7CompatibilityError(f"{test_ref} is not admissible for {ENTITY_REF}")
    if REGIME_REF not in spec["admissibility"]["regime_refs"]:
        raise O7CompatibilityError(f"{test_ref} is not admissible for {REGIME_REF}")
    if spec["input_mode"] != "OBSERVATION_VIEW":
        raise O7CompatibilityError(f"{test_ref} is not an ObservationView Test")
    level = summary["level"]
    return ObservationView(
        entity_instance_ref=f"legacy:O{level}:scanner-panel",
        entity_class_ref=ENTITY_REF,
        regime_ref=REGIME_REF,
        regime_depth=level,
        test_ref=test_ref,
        required_capabilities=spec["required_capabilities"],
        capability_payloads=_legacy_capability_payloads(summary),
        source_science_sha256=source_science_sha256,
        provider_ref=PROVIDER_REF,
    )


def _grammar(view: ObservationView) -> dict[str, Any]:
    transition = view.read("TRANSITION_SYSTEM")
    out = {"grammar_sha256": transition["grammar_sha256"]}
    # v0.28.0 materialized discovery can add a bounded operational read sentinel without
    # changing the historical/scanner/transport grammar evidence shape.  The semantic
    # signature deliberately excludes depth-specific carrier hashes: it changes only when
    # the tested read semantics/classification changes.
    if transition.get("operational_read_signature_sha256") is not None:
        out["operational_read_signature_sha256"] = transition["operational_read_signature_sha256"]
        out["operational_read_classification"] = transition.get("operational_read_classification")
    return out


def _diversity(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("RELATION_GRAPH")["diversity"])


def _branching(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("ACTION_MULTIPLICITY")["branching"])


def _symmetry(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("CANONICAL_AUTOMORPHISMS")["symmetry"])


def _overlap(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("FACTORIZATION")["overlap_gluing"])


def _lineage(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("ANCESTRY")["lineage"])


def _quotient(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("OBSERVER_QUOTIENT")["quotient_observer"])


def _topology(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("RELATION_GRAPH")["topology_services"])


def _obstruction(view: ObservationView) -> dict[str, Any]:
    return thaw_json(view.read("ACTION_ENABLEDNESS")["obstruction_relief"])


def _saturation(view: ObservationView) -> dict[str, Any]:
    ident = view.read("EXACT_ENTITY_IDENTITY")
    res = view.read("RESOURCE_COUNTERS")
    q = view.read("OBSERVER_QUOTIENT")
    return {
        "states": ident["states"],
        "raw_growth": thaw_json(res["raw_growth"]),
        "normalized_signature": thaw_json(q["normalized_signature"]),
        "normalized_signature_sha256": ident["normalized_signature_sha256"],
    }


def _representation(view: ObservationView) -> dict[str, Any]:
    b = view.read("BOUNDARY_INTERFACE")
    matched = thaw_json(b["matched_backbone"])
    if b.get("materialized_discovery") is not None:
        # The scanner's matched longitudinal backbone explicitly declares its stable signature
        # as the depth-erased representation evidence.  Absolute construction/growth diagnostics
        # can change with depth and must not masquerade as a representation-law break.
        matched = {
            "status": matched.get("status"),
            "signature_sha256": matched.get("signature_sha256"),
            "inherited_earned_law": matched.get("inherited_earned_law"),
        }
    out = {"matched_backbone": matched}
    if b["inherited_law_events"]:
        events = thaw_json(b["inherited_law_events"])
        if b.get("materialized_discovery") is not None:
            events = [
                {"law_id": x.get("law_id"), "disposition": x.get("disposition")}
                for x in events
            ]
        out["inherited_law_events"] = events
    transport = b.get("theorem_transport_consistency")
    if transport is not None:
        out["theorem_transport_consistency"] = {
            "status": transport.get("status"),
            "classification": transport.get("classification"),
            "discovery_authority": bool(transport.get("discovery_authority", False)),
        }
    return out


_ADAPTERS: dict[str, Callable[[ObservationView], dict[str, Any]]] = {
    TEST_GRAMMAR: _grammar,
    TEST_DIVERSITY: _diversity,
    TEST_BRANCHING: _branching,
    TEST_SYMMETRY: _symmetry,
    TEST_OVERLAP: _overlap,
    TEST_LINEAGE: _lineage,
    TEST_QUOTIENT: _quotient,
    TEST_TOPOLOGY: _topology,
    TEST_OBSTRUCTION: _obstruction,
    TEST_SATURATION: _saturation,
    TEST_REPRESENTATION: _representation,
}


def _status_from_genealogy(value: str) -> str:
    if value == "PERSISTS":
        return "INVARIANT"
    if value in {"SCOUT_NEW", "EXPANDS", "CLOSES", "REORGANIZES", "REFINES", "COMBINES", "DESTROYS_OR_RELIEVES"}:
        return "OBSERVED_VARIATION"
    return "NULL"


def run_observation_test(
    *,
    view: ObservationView,
    test_ref: str,
    source_science_sha256: str,
    scientific_status: str,
    summary: str,
    historical_labels: list[str] | tuple[str, ...] = (),
    registry: ScientificProtocolRegistry | None = None,
) -> TestExecution:
    """Execute one existing O7-compatible leaf Test against any capability provider.

    This is the provider-neutral runtime extracted in v0.27.3.  The Test adapter sees
    only its ObservationView; it does not know whether capabilities came from a frozen
    compatibility fixture, a live exact scanner summary, or theorem transport.
    """
    registry = registry or ScientificProtocolRegistry()
    spec = registry.test(test_ref)
    if view.test_ref != test_ref:
        raise O7CompatibilityError(f"ObservationView/Test mismatch: {view.test_ref} != {test_ref}")
    try:
        adapter = _ADAPTERS[test_ref]
    except KeyError as exc:
        raise O7CompatibilityError(f"no scientific adapter for {test_ref}") from exc
    evidence = adapter(view)
    finding = build_finding(
        test_spec=spec,
        entity_instance_ref=view.entity_instance_ref,
        regime_ref=view.regime_ref,
        regime_depth=view.regime_depth,
        source_science_sha256=source_science_sha256,
        evidence_payload=evidence,
        scientific_status=scientific_status,
        summary=summary,
        historical_labels=historical_labels,
    )
    return TestExecution(
        test_ref=test_ref,
        evidence_payload=evidence,
        evidence_sha256=canonical_sha256(evidence),
        finding=finding,
        observation_view=view.artifact(),
    )


def run_leaf_test(
    *,
    summary: Mapping[str, Any],
    test_ref: str,
    source_science_sha256: str,
    genealogy_value: str | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> TestExecution:
    registry = registry or ScientificProtocolRegistry()
    spec = registry.test(test_ref)
    view = build_legacy_observation_view(
        summary=summary,
        test_ref=test_ref,
        source_science_sha256=source_science_sha256,
        registry=registry,
    )
    return run_observation_test(
        view=view,
        test_ref=test_ref,
        source_science_sha256=source_science_sha256,
        scientific_status=_status_from_genealogy(genealogy_value or "SCOUT_NEW"),
        summary=f"Legacy-compatible {spec['test_id']} evidence at historical O{summary['level']}; no new science claim.",
        historical_labels=[f"O{summary['level']}", spec["observer_definition"]],
        registry=registry,
    )


def _legacy_genealogy_from_evidence(
    prev: Mapping[str, TestExecution] | None,
    cur: Mapping[str, TestExecution],
) -> dict[str, str]:
    if prev is None:
        return {k: "SCOUT_NEW" for k in [
            "grammar", "diversity", "branching", "symmetry", "overlap_gluing", "lineage",
            "quotient", "topology_services", "obstruction_relief", "representation",
        ]}

    out: dict[str, str] = {}
    out["grammar"] = "PERSISTS" if prev[TEST_GRAMMAR].evidence_payload == cur[TEST_GRAMMAR].evidence_payload else "REORGANIZES"

    pd = prev[TEST_DIVERSITY].evidence_payload
    cd = cur[TEST_DIVERSITY].evidence_payload
    if cd["organizational_classes"] > pd["organizational_classes"]:
        out["diversity"] = "EXPANDS"
    elif cd["organizational_classes"] < pd["organizational_classes"]:
        out["diversity"] = "CLOSES"
    else:
        out["diversity"] = "PERSISTS"

    for lane, tref in [
        ("branching", TEST_BRANCHING),
        ("symmetry", TEST_SYMMETRY),
        ("overlap_gluing", TEST_OVERLAP),
        ("lineage", TEST_LINEAGE),
        ("quotient", TEST_QUOTIENT),
        ("topology_services", TEST_TOPOLOGY),
    ]:
        out[lane] = "PERSISTS" if canonical_sha256(prev[tref].evidence_payload) == canonical_sha256(cur[tref].evidence_payload) else "REORGANIZES"

    po = prev[TEST_OBSTRUCTION].evidence_payload
    co = cur[TEST_OBSTRUCTION].evidence_payload
    if (
        (not po["all_bridge_types_supported_everywhere"] and co["all_bridge_types_supported_everywhere"])
        or (po["zero_action_states"] > 0 and co["zero_action_states"] == 0)
    ):
        out["obstruction_relief"] = "DESTROYS_OR_RELIEVES"
    else:
        out["obstruction_relief"] = "PERSISTS" if canonical_sha256(po) == canonical_sha256(co) else "REORGANIZES"

    ps = prev[TEST_SATURATION].evidence_payload["normalized_signature_sha256"]
    cs = cur[TEST_SATURATION].evidence_payload["normalized_signature_sha256"]
    out["representation"] = "PERSISTS" if ps == cs else "REFINES"
    return out


def _legacy_changed_families(genealogy: Mapping[str, str]) -> list[str]:
    return [
        k for k, v in genealogy.items()
        if v in {"REFINES", "CLOSES", "REORGANIZES", "COMBINES", "DESTROYS_OR_RELIEVES", "SCOUT_NEW"}
        and k not in {"grammar", "representation"}
    ]


def run_regime_boundary_sentinel(
    *,
    current: Mapping[str, TestExecution],
    previous: Mapping[str, TestExecution] | None,
    source_science_sha256: str,
    level: int,
    registry: ScientificProtocolRegistry | None = None,
    entity_instance_ref: str | None = None,
    regime_ref: str = REGIME_REF,
) -> TestExecution:
    """Generic meta-Test over only the four dependencies declared by Spec v1.

    This does not replace the legacy scanner structural-shock rule.  It is emitted as a
    recognition-only migration artifact while the legacy scanner remains authoritative.
    """
    registry = registry or ScientificProtocolRegistry()
    spec = registry.test(TEST_BOUNDARY_SENTINEL)
    deps = spec["dependency_test_refs"]
    for dep in deps:
        if dep not in current:
            raise O7CompatibilityError(f"boundary sentinel missing dependency {dep}")

    if previous is None:
        changed = list(deps)
        persistent = False
        status = "NULL"
        summary = "Ignition/reference depth has no previous matched Test finding set; no boundary promotion is inferred."
    else:
        changed = [dep for dep in deps if current[dep].evidence_sha256 != previous[dep].evidence_sha256]
        # Persistence requires a subsequent depth and is therefore intentionally false at a
        # single-depth invocation.  The maturation auditor will be responsible for temporal
        # persistence in the later extraction step.
        persistent = False
        status = "CANDIDATE" if len(changed) >= 3 else "NULL"
        summary = (
            f"{len(changed)} of {len(deps)} independent declared families changed relative to the previous depth; "
            "next-depth persistence is not asserted by this single-depth compatibility sentinel."
        )

    evidence = {
        "schema_id": "IG_REGIME_BOUNDARY_SENTINEL_COMPAT_EVIDENCE_V1",
        "level": level,
        "dependency_test_refs": deps,
        "changed_dependency_test_refs": changed,
        "minimum_independent_family_threshold": 3,
        "single_depth_candidate": len(changed) >= 3,
        "next_depth_persistence_confirmed": persistent,
        "promotion_authorized": False,
        "legacy_scanner_replaced": False,
    }
    finding = build_finding(
        test_spec=spec,
        entity_instance_ref=entity_instance_ref or f"legacy:O{level}:scanner-panel",
        regime_ref=regime_ref,
        regime_depth=level,
        source_science_sha256=source_science_sha256,
        evidence_payload=evidence,
        scientific_status=status,
        summary=summary,
        historical_labels=[f"O{level}", "RegimeBoundarySentinel compatibility migration"],
        reopen_triggered=False,
    )
    return TestExecution(
        test_ref=TEST_BOUNDARY_SENTINEL,
        evidence_payload=evidence,
        evidence_sha256=canonical_sha256(evidence),
        finding=finding,
        observation_view=None,
    )


def _reconstruct_legacy_projection(
    *,
    summary: Mapping[str, Any],
    leaves: Mapping[str, TestExecution],
    genealogy: Mapping[str, str],
) -> dict[str, Any]:
    sat = leaves[TEST_SATURATION].evidence_payload
    rep = leaves[TEST_REPRESENTATION].evidence_payload
    out = {
        "level": summary["level"],
        "states": sat["states"],
        "grammar_sha256": leaves[TEST_GRAMMAR].evidence_payload["grammar_sha256"],
        "diversity": leaves[TEST_DIVERSITY].evidence_payload,
        "branching": leaves[TEST_BRANCHING].evidence_payload,
        "symmetry": leaves[TEST_SYMMETRY].evidence_payload,
        "overlap_gluing": leaves[TEST_OVERLAP].evidence_payload,
        "lineage": leaves[TEST_LINEAGE].evidence_payload,
        "quotient_observer": leaves[TEST_QUOTIENT].evidence_payload,
        "topology_services": leaves[TEST_TOPOLOGY].evidence_payload,
        "obstruction_relief": leaves[TEST_OBSTRUCTION].evidence_payload,
        "raw_growth": sat["raw_growth"],
        "normalized_signature": sat["normalized_signature"],
        "normalized_signature_sha256": sat["normalized_signature_sha256"],
        "matched_backbone": rep["matched_backbone"],
        "genealogy": dict(genealogy),
        "structural_shock_changed_families": _legacy_changed_families(genealogy),
    }
    if rep.get("inherited_law_events"):
        out["inherited_law_events"] = rep["inherited_law_events"]
    if "exploratory_shock_events" in summary:
        # No frozen fixture currently contains this, but preserve it in the compatibility
        # projection if a future frozen legacy fixture does.
        out["exploratory_shock_events"] = summary["exploratory_shock_events"]
    return out


def run_o7_compatibility_equivalence(
    legacy_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = ScientificProtocolRegistry()
    legacy = dict(legacy_result or load_frozen_legacy_scanner_fixture())
    if legacy.get("schema") != "IG_O_REGIME_ADAPTIVE_SCANNER_RESULT_V1":
        raise O7CompatibilityError("legacy scanner fixture has unexpected schema")
    source_sha = legacy["science_sha256"]
    previous_leaves: dict[str, TestExecution] | None = None
    level_rows = []
    all_equivalent = True

    for level_key in sorted(legacy["level_summaries"], key=int):
        summary = legacy["level_summaries"][level_key]
        legacy_genealogy = summary["genealogy"]
        leaves: dict[str, TestExecution] = {}
        # Each leaf Test is independently evaluated against a capability-limited view.
        lane_map = {
            TEST_GRAMMAR: legacy_genealogy["grammar"],
            TEST_DIVERSITY: legacy_genealogy["diversity"],
            TEST_BRANCHING: legacy_genealogy["branching"],
            TEST_SYMMETRY: legacy_genealogy["symmetry"],
            TEST_OVERLAP: legacy_genealogy["overlap_gluing"],
            TEST_LINEAGE: legacy_genealogy["lineage"],
            TEST_QUOTIENT: legacy_genealogy["quotient"],
            TEST_TOPOLOGY: legacy_genealogy["topology_services"],
            TEST_OBSTRUCTION: legacy_genealogy["obstruction_relief"],
            TEST_SATURATION: legacy_genealogy["representation"],
            TEST_REPRESENTATION: legacy_genealogy["representation"],
        }
        for tref in LEAF_TEST_REFS:
            leaves[tref] = run_leaf_test(
                summary=summary,
                test_ref=tref,
                source_science_sha256=source_sha,
                genealogy_value=lane_map[tref],
                registry=registry,
            )

        reconstructed_genealogy = _legacy_genealogy_from_evidence(previous_leaves, leaves)
        reconstructed = _reconstruct_legacy_projection(
            summary=summary,
            leaves=leaves,
            genealogy=reconstructed_genealogy,
        )
        expected = legacy_level_science_projection(summary)
        equivalent = reconstructed == expected
        all_equivalent = all_equivalent and equivalent
        sentinel = run_regime_boundary_sentinel(
            current=leaves,
            previous=previous_leaves,
            source_science_sha256=source_sha,
            level=int(level_key),
            registry=registry,
        )
        level_rows.append({
            "level": int(level_key),
            "legacy_science_projection_sha256": canonical_sha256(expected),
            "generic_reconstruction_sha256": canonical_sha256(reconstructed),
            "byte_semantic_equivalent": equivalent,
            "leaf_test_count": len(leaves),
            "leaf_test_evidence_sha256": {k: leaves[k].evidence_sha256 for k in sorted(leaves)},
            "observation_view_sha256": {
                k: canonical_sha256(dict(leaves[k].observation_view or {})) for k in sorted(leaves)
            },
            "finding_sha256": {k: canonical_sha256(dict(leaves[k].finding)) for k in sorted(leaves)},
            "regime_boundary_sentinel_evidence_sha256": sentinel.evidence_sha256,
            "regime_boundary_sentinel_scientific_status": sentinel.finding["scientific_status"],
        })
        previous_leaves = leaves

    result = {
        "schema_id": "IG_O7_GENERIC_SCIENTIFIC_PROTOCOL_EQUIVALENCE_V1",
        "schema_version": "1.0.0",
        "status": "PASS" if all_equivalent else "FAIL",
        "classification": "GENERIC_TEST_PROTOCOL_O7_COMPATIBILITY_EQUIVALENCE" if all_equivalent else "O7_COMPATIBILITY_MISMATCH",
        "entity_class_ref": ENTITY_REF,
        "regime_ref": REGIME_REF,
        "test_pack_ref": PACK_REF,
        "legacy_scanner_schema": legacy["schema"],
        "legacy_scanner_science_sha256": source_sha,
        "levels": level_rows,
        "levels_checked": len(level_rows),
        "leaf_tests_per_level": len(LEAF_TEST_REFS),
        "all_legacy_science_projections_exactly_reconstructed": all_equivalent,
        "legacy_scanner_source_modified": False,
        "production_science_routed_through_generic_layer": False,
        "new_scientific_claim": False,
        "notes": [
            "The legacy O-regime scanner remains the scientific oracle during this migration step.",
            "The generic Test layer reconstructs the frozen per-level scientific evidence exactly from capability-limited ObservationViews.",
            "RegimeBoundarySentinel is emitted recognition-only and does not replace the legacy structural-shock promotion rule in this step.",
        ],
    }
    result["science_sha256"] = canonical_sha256({k: v for k, v in result.items() if k != "science_sha256"})
    return result
