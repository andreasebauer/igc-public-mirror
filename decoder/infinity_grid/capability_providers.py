from __future__ import annotations

"""Live capability providers for the generic Decoder scientific protocol layer.

The providers are deliberately separate from Tests.  They translate an authoritative
scientific source into the typed capability vocabulary declared by a Regime.  Tests receive
only an ObservationView and therefore cannot tell how a capability was computed.

v0.28.0 provides three O-regime providers:

* ExactScannerSummaryProvider: projections of a live/bounded O-regime scanner level summary.
* MaterializedDiscoveryProvider: exact finite synthetic fixed-lift panels materialized depth by depth, calibrated against O7..O13, and used as the primary O14+ discovery observation.
* FixedGrammarTransportProvider: observer-relative, theorem-transported capabilities from an
  immutable fixed-grammar level delta.  It remains an auxiliary consistency/growth-bound lane and
  never pretends that an exact carrier population was materialized.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .scientific_architecture import (
    ObservationView,
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    validate_scientific_artifact,
)
from .o7_science_compat import ENTITY_REF, REGIME_REF
from . import fixed_grammar_transport as fgt


class CapabilityProviderError(ScientificArchitectureError):
    pass


SCANNER_PROVIDER_REF = "IG_O_REGIME_SCANNER_SUMMARY_CAPABILITY_PROVIDER@1.0.0"
TRANSPORT_PROVIDER_REF = "IG_O_REGIME_FIXED_GRAMMAR_CAPABILITY_PROVIDER@1.0.0"
DISCOVERY_PROVIDER_REF = "IG_O_REGIME_MATERIALIZED_DISCOVERY_CAPABILITY_PROVIDER@1.0.0"


@dataclass(frozen=True)
class CapabilitySnapshot:
    provider_ref: str
    provider_kind: str
    entity_instance_ref: str
    entity_class_ref: str
    regime_ref: str
    regime_depth: int
    source_science_sha256: str
    payloads: Mapping[str, Any]
    exactness: str

    @property
    def science_sha256(self) -> str:
        return canonical_sha256({
            "provider_ref": self.provider_ref,
            "provider_kind": self.provider_kind,
            "entity_instance_ref": self.entity_instance_ref,
            "entity_class_ref": self.entity_class_ref,
            "regime_ref": self.regime_ref,
            "regime_depth": self.regime_depth,
            "source_science_sha256": self.source_science_sha256,
            "payloads": dict(self.payloads),
            "exactness": self.exactness,
        })

    def observation_view(self, test_ref: str, *, registry: ScientificProtocolRegistry | None = None) -> ObservationView:
        registry = registry or ScientificProtocolRegistry()
        spec = registry.test(test_ref)
        if self.entity_class_ref not in spec["admissibility"]["entity_class_refs"]:
            raise CapabilityProviderError(f"{test_ref} is not admissible for {self.entity_class_ref}")
        if self.regime_ref not in spec["admissibility"]["regime_refs"]:
            raise CapabilityProviderError(f"{test_ref} is not admissible for {self.regime_ref}")
        return ObservationView(
            entity_instance_ref=self.entity_instance_ref,
            entity_class_ref=self.entity_class_ref,
            regime_ref=self.regime_ref,
            regime_depth=self.regime_depth,
            test_ref=test_ref,
            required_capabilities=spec["required_capabilities"],
            capability_payloads=self.payloads,
            source_science_sha256=self.source_science_sha256,
            provider_ref=self.provider_ref,
        )


def _scanner_summary_payloads(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Projection used for any live O-regime scanner summary, not only frozen O7..O13."""
    required = [
        "level", "states", "grammar_sha256", "diversity", "branching", "symmetry",
        "overlap_gluing", "lineage", "quotient_observer", "topology_services",
        "obstruction_relief", "raw_growth", "normalized_signature", "normalized_signature_sha256",
        "matched_backbone",
    ]
    missing = [k for k in required if k not in summary]
    if missing:
        raise CapabilityProviderError(f"scanner level summary missing required fields {missing}")
    normalized = summary["normalized_signature"]
    return {
        "EXACT_ENTITY_IDENTITY": {
            "level": summary["level"],
            "states": summary["states"],
            "entity_population_materialized": True,
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
        "ACTION_ENABLEDNESS": {"branching": summary["branching"], "obstruction_relief": summary["obstruction_relief"]},
        "ACTION_MULTIPLICITY": {"branching": summary["branching"]},
        "BLOCK_STRUCTURE": {"diversity": summary["diversity"]},
        "ANCESTRY": {"lineage": summary["lineage"]},
        "FACTORIZATION": {"lineage": summary["lineage"], "overlap_gluing": summary["overlap_gluing"]},
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


def from_scanner_level_summary(
    summary: Mapping[str, Any],
    *,
    source_science_sha256: str | None = None,
    entity_instance_ref: str | None = None,
) -> CapabilitySnapshot:
    level = int(summary["level"])
    source_sha = source_science_sha256 or canonical_sha256(dict(summary))
    return CapabilitySnapshot(
        provider_ref=SCANNER_PROVIDER_REF,
        provider_kind="LIVE_BOUNDED_SCANNER_SUMMARY",
        entity_instance_ref=entity_instance_ref or f"live:O{level}:scanner-panel",
        entity_class_ref=ENTITY_REF,
        regime_ref=REGIME_REF,
        regime_depth=level,
        source_science_sha256=source_sha,
        payloads=_scanner_summary_payloads(summary),
        exactness="CERTIFIED_DERIVED",
    )


def from_materialized_discovery(
    summary: Mapping[str, Any],
    materialization: Mapping[str, Any],
    *,
    transport_audit: Mapping[str, Any] | None = None,
    entity_instance_ref: str | None = None,
) -> CapabilitySnapshot:
    """Primary O14+ discovery provider over actual generated finite carriers.

    The generic Test layer receives the same frozen capability vocabulary used by the legacy
    scanner, but every O14+ payload is derived from the depth-specific materialized panel.  The
    fixed-grammar theorem transport audit is attached only as auxiliary consistency evidence.
    """
    level = int(summary["level"])
    if int(materialization.get("level", -1)) != level:
        raise CapabilityProviderError("materialized discovery level does not match scanner summary")
    if materialization.get("provider_ref") != DISCOVERY_PROVIDER_REF:
        raise CapabilityProviderError("materialized discovery provider pin mismatch")
    declared_materialization_sha = materialization.get("science_sha256")
    observed_materialization_sha = canonical_sha256({k: v for k, v in materialization.items() if k != "science_sha256"})
    if declared_materialization_sha != observed_materialization_sha:
        raise CapabilityProviderError("materialized discovery evidence science hash mismatch")
    panel_payload = {
        "level": level,
        "state_probes": materialization.get("state_probes", []),
        "read_probe": materialization.get("read_probe", {}),
    }
    if materialization.get("panel_science_sha256") != canonical_sha256(panel_payload):
        raise CapabilityProviderError("materialized discovery panel science hash mismatch")
    source_summary = dict(summary)
    source_summary.pop("materialized_discovery", None)
    if materialization.get("scanner_summary_sha256") != canonical_sha256(source_summary):
        raise CapabilityProviderError("materialized discovery scanner-summary binding mismatch")
    if materialization.get("entity_population_materialized") is not True:
        raise CapabilityProviderError("materialized discovery provider requires an actual finite state population")
    if int(materialization.get("materialized_state_count", 0)) <= 0:
        raise CapabilityProviderError("materialized discovery provider received an empty state population")
    calibration = materialization.get("calibration", {})
    if calibration.get("status") != "PASS":
        raise CapabilityProviderError("materialized discovery replay calibration is not PASS")
    read_probe = materialization.get("read_probe", {})
    if read_probe.get("status") not in {"PASS", "CANDIDATE"}:
        raise CapabilityProviderError("materialized discovery read probe is inconclusive")
    cohort = materialization.get("candidate_cohort")
    if cohort is not None:
        if cohort.get("schema_id") != "IG_O_REGIME_LONGITUDINAL_CANDIDATE_COHORT_V1":
            raise CapabilityProviderError("materialized discovery candidate cohort schema mismatch")
        cohort_observed = canonical_sha256({k: v for k, v in cohort.items() if k != "science_sha256"})
        if cohort.get("science_sha256") != cohort_observed:
            raise CapabilityProviderError("materialized discovery candidate cohort science hash mismatch")
        rows = cohort.get("motif_structural_signatures", [])
        if int(cohort.get("candidate_count", -1)) != len(rows):
            raise CapabilityProviderError("materialized discovery candidate cohort count mismatch")
    if transport_audit is not None and transport_audit.get("status") != "PASS":
        raise CapabilityProviderError("auxiliary theorem-transport consistency audit failed")

    payloads = _scanner_summary_payloads(summary)
    payloads["EXACT_ENTITY_IDENTITY"] = dict(payloads["EXACT_ENTITY_IDENTITY"])
    payloads["EXACT_ENTITY_IDENTITY"].update({
        "entity_population_materialized": True,
        "materialized_state_count": int(materialization["materialized_state_count"]),
        "materialized_panel_science_sha256": materialization["panel_science_sha256"],
        "materialized_state_probes": materialization["state_probes"],
        "longitudinal_candidate_cohort": None if cohort is None else dict(cohort),
        "discovery_scope": materialization.get("scope"),
    })
    payloads["TRANSITION_SYSTEM"] = dict(payloads["TRANSITION_SYSTEM"])
    payloads["TRANSITION_SYSTEM"].update({
        "operational_read_probe": read_probe,
        "operational_read_signature_sha256": read_probe["semantic_signature_sha256"],
        "operational_read_classification": read_probe["classification"],
        "materialized_panel_science_sha256": materialization["panel_science_sha256"],
    })
    payloads["RELATION_GRAPH"] = dict(payloads["RELATION_GRAPH"])
    payloads["RELATION_GRAPH"].update({
        "materialized_panel_science_sha256": materialization["panel_science_sha256"],
        "materialized_state_count": int(materialization["materialized_state_count"]),
    })
    payloads["BOUNDARY_INTERFACE"] = dict(payloads["BOUNDARY_INTERFACE"])
    payloads["BOUNDARY_INTERFACE"].update({
        "materialized_discovery": {
            "provider_ref": DISCOVERY_PROVIDER_REF,
            "panel_science_sha256": materialization["panel_science_sha256"],
            "scope": materialization.get("scope"),
            "calibration": calibration,
        },
        "theorem_transport_consistency": dict(transport_audit) if transport_audit is not None else None,
    })
    payloads["RESOURCE_COUNTERS"] = dict(payloads["RESOURCE_COUNTERS"])
    payloads["RESOURCE_COUNTERS"].update({
        "materialized_state_count": int(materialization["materialized_state_count"]),
        "materialized_panel_science_sha256": materialization["panel_science_sha256"],
    })

    source_packet = {
        "summary": dict(summary),
        "materialization_science_sha256": materialization.get("science_sha256"),
        "materialized_panel_science_sha256": materialization.get("panel_science_sha256"),
        "read_probe_semantic_signature_sha256": read_probe.get("semantic_signature_sha256"),
        "transport_audit_science_sha256": transport_audit.get("science_sha256") if transport_audit else None,
        "candidate_cohort_science_sha256": cohort.get("science_sha256") if cohort else None,
    }
    source_sha = canonical_sha256(source_packet)
    return CapabilitySnapshot(
        provider_ref=DISCOVERY_PROVIDER_REF,
        provider_kind="BOUNDED_MATERIALIZED_STRUCTURAL_DISCOVERY",
        entity_instance_ref=entity_instance_ref or f"live:O{level}:materialized-discovery-panel",
        entity_class_ref=ENTITY_REF,
        regime_ref=REGIME_REF,
        regime_depth=level,
        source_science_sha256=source_sha,
        payloads=payloads,
        exactness="CERTIFIED_DERIVED",
    )


def _transport_organization(delta: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if delta.get("schema") != "IG_O_REGIME_STREAMING_LEVEL_DELTA_V2" or delta.get("status") != "PASS":
        raise CapabilityProviderError("fixed-grammar provider requires a PASS streaming level delta")
    if not fgt._record_hash_ok(dict(delta)):
        raise CapabilityProviderError("fixed-grammar level delta science hash mismatch")
    org = delta.get("normalized_organization")
    if not isinstance(org, Mapping):
        raise CapabilityProviderError("fixed-grammar level delta has no normalized_organization")
    static = fgt.streaming_recipe_baseline()
    if org.get("static_signature_sha256") != static["static_signature_sha256"]:
        raise CapabilityProviderError("transport delta/static recipe baseline pin mismatch")
    return dict(org), static


def _transport_payloads(delta: Mapping[str, Any]) -> dict[str, Any]:
    org, static = _transport_organization(delta)
    level = int(delta["level"])
    contract = delta["contract"]
    growth = delta["growth_bounds"]
    branch = static["normalized_lane_payloads"]["branching"]
    symmetry = static["normalized_lane_payloads"]["symmetry"]
    lineage = static["normalized_lane_payloads"]["lineage"]
    topology = static["normalized_lane_payloads"]["topology_services"]
    diversity = dict(static["diversity"])
    diversity.update({
        "resource_skins": None,
        "same_skin_groups": None,
        "same_skin_topology_collision_groups": None,
        "min_total_free_by_type": growth["min_total_free_by_type"],
        "population_materialized": False,
        "transported_recipe_count": static["recipe_count"],
    })
    quotient = dict(static["quotient_observer"])
    quotient["provider_scope"] = "THEOREM_TRANSPORTED_OBSERVER_RELATIVE"
    obstruction = dict(static["obstruction_relief"])
    matched = {
        "status": "THEOREM_TRANSPORTED_STATIC_RECIPE_ORGANIZATION",
        "signature_sha256": static["static_signature_sha256"],
        "recipe_catalogue_sha256": static["recipe_catalogue_sha256"],
        "inherited_earned_law": "O_DEPTH_ERASED_NORMALIZED_ORGANIZATION_V1",
    }
    normalized = {
        "grammar": contract["grammar_sha256"],
        "diversity": static["diversity"],
        "branching": branch,
        "symmetry": symmetry,
        "overlap_gluing": static["overlap_gluing"],
        "lineage": lineage,
        "quotient": quotient,
        "topology_services": topology,
        "obstruction_relief": obstruction,
        "observer_scope": "FIXED_RECIPE_DEPTH_ERASED_THEOREM_TRANSPORT",
    }
    norm_sha = canonical_sha256(normalized)
    raw_growth = {
        "mode": "CERTIFIED_CONSERVATIVE_BOUNDS",
        "min_total_free_by_type": growth["min_total_free_by_type"],
        "max_total_free_by_type": growth["max_total_free_by_type"],
        "leaf_count_min": growth["leaf_count_min"],
        "leaf_count_max": growth["leaf_count_max"],
        "relation_count_min": growth["relation_count_min"],
        "relation_count_max": growth["relation_count_max"],
    }
    return {
        "EXACT_ENTITY_IDENTITY": {
            "level": level,
            "states": None,
            "entity_population_materialized": False,
            "transported_recipe_count": static["recipe_count"],
            "normalized_signature_sha256": norm_sha,
        },
        "BOUNDARY_INTERFACE": {
            "quotient_observer": quotient,
            "matched_backbone": matched,
            "inherited_law_events": [{
                "law_id": "O_DEPTH_ERASED_NORMALIZED_ORGANIZATION_V1",
                "disposition": "THEOREM_TRANSPORTED",
            }],
        },
        "RELATION_GRAPH": {
            "diversity": diversity,
            "overlap_gluing": static["overlap_gluing"],
            "lineage": lineage,
            "topology_services": topology,
        },
        "TYPED_RELATIONS": {"grammar_sha256": contract["grammar_sha256"], "branching": branch, "topology_services": topology},
        "TRANSITION_SYSTEM": {
            "grammar_sha256": contract["grammar_sha256"],
            "branching": branch,
            "quotient_observer": quotient,
            "obstruction_relief": obstruction,
            "matched_backbone": matched,
            "successor_semantics": contract["successor_semantics"],
        },
        "ACTION_ENABLEDNESS": {"branching": branch, "obstruction_relief": obstruction},
        "ACTION_MULTIPLICITY": {"branching": branch},
        "BLOCK_STRUCTURE": {"diversity": diversity},
        "ANCESTRY": {"lineage": lineage},
        "FACTORIZATION": {"lineage": lineage, "overlap_gluing": static["overlap_gluing"]},
        "INTRINSIC_DISTANCE": {"topology_services": topology},
        "NEIGHBORHOOD_SHELLS": {"topology_services": topology},
        "CANONICAL_AUTOMORPHISMS": {"symmetry": symmetry},
        "RESOURCE_COUNTERS": {
            "min_total_free_by_type": growth["min_total_free_by_type"],
            "raw_growth": raw_growth,
            "obstruction_relief": obstruction,
            "matched_backbone": matched,
        },
        "OBSERVER_QUOTIENT": {
            "quotient_observer": quotient,
            "normalized_signature": normalized,
            "normalized_signature_sha256": norm_sha,
            "resource_skins": None,
            "organizational_classes": static["diversity"]["organizational_classes"],
        },
    }


def from_fixed_grammar_delta(delta: Mapping[str, Any], *, entity_instance_ref: str | None = None) -> CapabilitySnapshot:
    level = int(delta["level"])
    payloads = _transport_payloads(delta)
    return CapabilitySnapshot(
        provider_ref=TRANSPORT_PROVIDER_REF,
        provider_kind="THEOREM_TRANSPORTED_FIXED_GRAMMAR",
        entity_instance_ref=entity_instance_ref or f"live:O{level}:fixed-grammar-frontier",
        entity_class_ref=ENTITY_REF,
        regime_ref=REGIME_REF,
        regime_depth=level,
        source_science_sha256=str(delta["science_sha256"]),
        payloads=payloads,
        exactness="OBSERVER_RELATIVE",
    )


def load_fixed_grammar_delta(output: str | Path, level: int) -> dict[str, Any]:
    path = Path(output) / "deltas" / f"O{int(level):05d}.json"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CapabilityProviderError(f"cannot load fixed-grammar delta {path}: {type(exc).__name__}: {exc}") from exc
    return obj


def fixed_grammar_snapshot(output: str | Path, level: int) -> CapabilitySnapshot:
    return from_fixed_grammar_delta(load_fixed_grammar_delta(output, level))


def provider_contracts(*, registry: ScientificProtocolRegistry | None = None) -> list[dict[str, Any]]:
    """Return validated provider contracts for every currently frozen capability."""
    registry = registry or ScientificProtocolRegistry()
    out: list[dict[str, Any]] = []
    for capability_id in sorted(registry.capability_ids):
        for provider_ref, exactness, read_set in (
            (SCANNER_PROVIDER_REF, "CERTIFIED_DERIVED", ["O_REGIME_SCANNER_LEVEL_SUMMARY"]),
            (DISCOVERY_PROVIDER_REF, "CERTIFIED_DERIVED", ["MATERIALIZED_O_REGIME_PANEL", "MATERIALIZED_RELATION_ADD_READ_PROBE", "THEOREM_TRANSPORT_AUXILIARY_AUDIT"]),
            (TRANSPORT_PROVIDER_REF, "OBSERVER_RELATIVE", ["IG_O_REGIME_STREAMING_LEVEL_DELTA_V2", "O_REGIME_STREAMING_STATIC_RECIPE_BASELINE_V2"]),
        ):
            obj = {
                "schema_id": "IG_CAPABILITY_PROVIDER_V1",
                "schema_version": "1.0.0",
                "capability_id": capability_id,
                "version": "1.0.0",
                "provider_ref": provider_ref,
                "output_schema_ref": f"ig://scientific-capability/{capability_id}/runtime-v1",
                "exactness": exactness,
                "read_set": read_set,
                "reopen_conditions": [
                    "Regime capability vocabulary changes",
                    "source schema or observer semantics changes",
                    "PrimitiveSemanticsSentinel or theorem freezewall reopens",
                ],
            }
            validate_scientific_artifact(obj)
            out.append(obj)
    return out


def verify_capability_providers() -> dict[str, Any]:
    registry = ScientificProtocolRegistry()
    contracts = provider_contracts(registry=registry)
    return {
        "schema_id": "IG_LIVE_CAPABILITY_PROVIDER_VERIFICATION_V1",
        "status": "PASS",
        "providers": [SCANNER_PROVIDER_REF, DISCOVERY_PROVIDER_REF, TRANSPORT_PROVIDER_REF],
        "capabilities": len(registry.capability_ids),
        "provider_contracts": len(contracts),
        "all_regime_capabilities_covered_by_all_providers": len(contracts) == 3 * len(registry.capability_ids),
        "materialized_discovery_provider_population_claim": "EXACT_FINITE_SYNTHETIC_PANEL_MATERIALIZED",
        "fixed_grammar_provider_population_claim": "NOT_MATERIALIZED",
        "science_sha256": canonical_sha256(contracts),
    }
