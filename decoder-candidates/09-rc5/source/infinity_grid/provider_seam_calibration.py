from __future__ import annotations

"""Phase-4 O13->O14 provider-seam calibration.

The historical O7..O13 scanner provider and the v0.28 materialized discovery provider expose
slightly different presentation envelopes.  The latter intentionally adds an operational read
probe and compact representation metadata.  A provider switch therefore cannot be compared by
raw provider identity alone.

Phase 4 solves that problem by using the *same O13 scientific carrier depth* as an overlap:
materialize O13 with the new provider, independently execute the frozen TestPack through both
providers, and prove equality under an explicit common projection.  O14 can then be compared to
the materialized O13 overlap baseline.  The provider switch itself is no longer novelty; any O14
variation is actual same-provider longitudinal evidence.
"""

from typing import Any, Mapping
import tempfile
from pathlib import Path

from .canon import canonical_sha256
from .capability_providers import (
    SCANNER_PROVIDER_REF,
    DISCOVERY_PROVIDER_REF,
    from_scanner_level_summary,
    from_materialized_discovery,
)
from .materialized_discovery import MaterializedDiscoverySession, theorem_transport_consistency
from .fixed_grammar_transport import run_fixed_grammar_transport
from .capability_providers import load_fixed_grammar_delta
from .maturation_auditor import LevelTestRecord, execute_snapshot_tests
from .o7_science_compat import (
    LEAF_TEST_REFS,
    TEST_GRAMMAR,
    TEST_REPRESENTATION,
    load_frozen_legacy_scanner_fixture,
)
from .scientific_architecture import ScientificProtocolRegistry


CALIBRATION_SCHEMA_ID = "IG_O13_O14_PROVIDER_SEAM_CALIBRATION_V1"
CALIBRATION_CLASSIFICATION = "MECHANICALLY_PROVED_OVERLAP_PROJECTION_EQUIVALENCE"
OVERLAP_DEPTH = 13


class ProviderSeamCalibrationError(RuntimeError):
    pass


def _representation_common_projection(ev: Mapping[str, Any]) -> dict[str, Any]:
    matched = ev.get("matched_backbone", {}) if isinstance(ev.get("matched_backbone"), Mapping) else {}
    inherited = matched.get("inherited_earned_law")
    if inherited is None:
        events = ev.get("inherited_law_events", [])
        if isinstance(events, (list, tuple)):
            ids = sorted({str(x.get("law_id")) for x in events if isinstance(x, Mapping) and x.get("law_id")})
            inherited = ids[0] if len(ids) == 1 else ids
    return {
        "status": matched.get("status"),
        "signature_sha256": matched.get("signature_sha256"),
        "inherited_earned_law": inherited,
    }


def common_leaf_projection(test_ref: str, evidence_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Common semantics visible in both O13 providers.

    New-provider-only fields are excluded only at the overlap equality step.  They remain live
    scientific evidence when O14+ is compared against the materialized O13 baseline.
    """
    if test_ref == TEST_GRAMMAR:
        return {"grammar_sha256": evidence_payload.get("grammar_sha256")}
    if test_ref == TEST_REPRESENTATION:
        return {"matched_backbone": _representation_common_projection(evidence_payload)}
    return dict(evidence_payload)


def _common_scanner_summary_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    # These two keys are provider-envelope additions. All legacy scanner scientific fields must
    # otherwise be byte-identical at O13.
    return {
        k: v for k, v in summary.items()
        if k not in {"materialized_discovery", "inherited_law_events"}
    }


def verify_seam_calibration_certificate(cert: Mapping[str, Any]) -> None:
    if cert.get("schema_id") != CALIBRATION_SCHEMA_ID:
        raise ProviderSeamCalibrationError("provider seam calibration schema mismatch")
    if cert.get("status") != "PASS":
        raise ProviderSeamCalibrationError("provider seam calibration is not PASS")
    if cert.get("classification") != CALIBRATION_CLASSIFICATION:
        raise ProviderSeamCalibrationError("provider seam calibration classification mismatch")
    if int(cert.get("overlap_depth", -1)) != OVERLAP_DEPTH:
        raise ProviderSeamCalibrationError("provider seam overlap depth mismatch")
    if cert.get("from_provider_ref") != SCANNER_PROVIDER_REF or cert.get("to_provider_ref") != DISCOVERY_PROVIDER_REF:
        raise ProviderSeamCalibrationError("provider seam endpoints mismatch")
    declared = cert.get("science_sha256")
    observed = canonical_sha256({k: v for k, v in cert.items() if k != "science_sha256"})
    if declared != observed:
        raise ProviderSeamCalibrationError("provider seam calibration science hash mismatch")
    checks = cert.get("leaf_projection_checks", [])
    if len(checks) != len(LEAF_TEST_REFS) or not all(x.get("equal") is True for x in checks):
        raise ProviderSeamCalibrationError("provider seam leaf projection equivalence incomplete")
    if cert.get("scanner_common_projection_equal") is not True:
        raise ProviderSeamCalibrationError("provider seam scanner projection equivalence failed")


def calibrate_o13_o14_provider_seam(
    *,
    discovery: MaterializedDiscoverySession | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> tuple[dict[str, Any], LevelTestRecord]:
    """Return a PASS certificate and the materialized O13 comparison baseline.

    If ``discovery`` is supplied it remains owned by the caller and is advanced to O13. Otherwise
    this function creates and closes a temporary discovery session.
    """
    registry = registry or ScientificProtocolRegistry()
    legacy = load_frozen_legacy_scanner_fixture()
    legacy_summary = legacy["level_summaries"].get(str(OVERLAP_DEPTH))
    if legacy_summary is None:
        raise ProviderSeamCalibrationError("legacy O13 overlap summary unavailable")

    owns = discovery is None
    session = discovery or MaterializedDiscoverySession()
    if owns:
        session.__enter__()
    try:
        row = session.advance_to(OVERLAP_DEPTH)
        cal = row.materialization.get("calibration", {})
        if cal.get("status") != "PASS" or not all(cal.get("checks", {}).values()):
            raise ProviderSeamCalibrationError("materialized O13 failed frozen O7..O13 replay calibration")

        legacy_common = _common_scanner_summary_projection(legacy_summary)
        materialized_common = _common_scanner_summary_projection(row.summary)
        scanner_equal = canonical_sha256(legacy_common) == canonical_sha256(materialized_common)
        if not scanner_equal:
            raise ProviderSeamCalibrationError("O13 common scanner summary projection differs across providers")

        legacy_snapshot = from_scanner_level_summary(
            legacy_summary,
            source_science_sha256=legacy["science_sha256"],
            entity_instance_ref="phase4-overlap:O13:legacy-scanner",
        )
        # Bind the O13 materialized overlap to the same auxiliary theorem-transport envelope
        # used at O14+.  Otherwise the representation Test would see the appearance of the
        # auxiliary audit itself as a false longitudinal variation at the seam.  The audit has
        # no discovery authority; it is included only to make the new-provider presentation
        # internally homogeneous from the O13 baseline onward.
        with tempfile.TemporaryDirectory(prefix="ig_phase4_o13_transport_") as td:
            troot = Path(td)
            run_fixed_grammar_transport(troot, through=OVERLAP_DEPTH, reset=True)
            delta13 = load_fixed_grammar_delta(troot, OVERLAP_DEPTH)
            transport_audit = theorem_transport_consistency(row.summary, row.materialization, delta13)
        if transport_audit.get("status") != "PASS":
            raise ProviderSeamCalibrationError("O13 theorem-transport auxiliary consistency audit failed")
        materialized_snapshot = from_materialized_discovery(
            row.summary,
            row.materialization,
            transport_audit=transport_audit,
            entity_instance_ref="phase4-overlap:O13:materialized-discovery",
        )
        legacy_record = execute_snapshot_tests(legacy_snapshot, registry=registry)
        materialized_record = execute_snapshot_tests(materialized_snapshot, registry=registry)

        checks: list[dict[str, Any]] = []
        for tref in LEAF_TEST_REFS:
            le = legacy_record.executions[tref]
            me = materialized_record.executions[tref]
            lp = common_leaf_projection(tref, le.evidence_payload)
            mp = common_leaf_projection(tref, me.evidence_payload)
            lsha = canonical_sha256(lp)
            msha = canonical_sha256(mp)
            checks.append({
                "test_ref": tref,
                "legacy_common_projection_sha256": lsha,
                "materialized_common_projection_sha256": msha,
                "equal": lsha == msha,
                "raw_evidence_equal": le.evidence_sha256 == me.evidence_sha256,
            })
        failed = [x["test_ref"] for x in checks if not x["equal"]]
        if failed:
            raise ProviderSeamCalibrationError(f"O13 common Test projections differ: {failed}")

        cert: dict[str, Any] = {
            "schema_id": CALIBRATION_SCHEMA_ID,
            "schema_version": "1.0.0",
            "status": "PASS",
            "classification": CALIBRATION_CLASSIFICATION,
            "overlap_depth": OVERLAP_DEPTH,
            "from_provider_ref": SCANNER_PROVIDER_REF,
            "to_provider_ref": DISCOVERY_PROVIDER_REF,
            "legacy_source_science_sha256": legacy["science_sha256"],
            "materialized_source_science_sha256": materialized_snapshot.source_science_sha256,
            "materialized_o13_evidence_sha256": row.materialization["science_sha256"],
            "o13_theorem_transport_auxiliary_audit_sha256": transport_audit["science_sha256"],
            "scanner_common_projection_sha256": canonical_sha256(legacy_common),
            "scanner_common_projection_equal": scanner_equal,
            "leaf_projection_checks": checks,
            "new_provider_extensions_not_used_to_prove_overlap_equivalence": [
                "grammar.operational_read_signature_sha256",
                "grammar.operational_read_classification",
                "representation.inherited_law_events presentation",
                "representation.theorem_transport_consistency",
            ],
            "comparison_rule": (
                "At the O13->O14 seam, O14 longitudinal Test status is computed against the "
                "materialized O13 overlap baseline. The physical provider change is recorded but "
                "is not itself scientific novelty. New-provider-only fields remain live evidence "
                "from the materialized O13 baseline onward."
            ),
            "nonclaims": [
                "NOT_PROVIDER_EQUIVALENCE_OUTSIDE_THE_FROZEN_O13_OVERLAP_PROJECTION",
                "NOT_A_HISTORICAL_O14_GRADUATION",
                "NOT_A_PLATEAU_OR_LIFT_CERTIFICATE",
                "NOT_AN_UNBOUNDED_O_TOWER_THEOREM",
            ],
        }
        cert["science_sha256"] = canonical_sha256(cert)
        verify_seam_calibration_certificate(cert)
        return cert, materialized_record
    finally:
        if owns:
            session.__exit__(None, None, None)
