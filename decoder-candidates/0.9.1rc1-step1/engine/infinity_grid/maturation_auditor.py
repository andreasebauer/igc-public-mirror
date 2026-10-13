from __future__ import annotations

"""Generic maturation and bounded-plateau auditor.

The auditor consumes read-only TestExecution records produced from capability providers.  It
never develops entities, never mutates the ResearchFrontier, and never performs a Lift.  Its
plateau certificate is deliberately bounded by the frozen TestPack/observer and carries the
Architecture Spec v1 nonclaim: not eternal closure and not automatic lift.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .canon import canonical_sha256
from .scientific_architecture import (
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    TestExecution,
    build_finding,
    validate_scientific_artifact,
)
from .capability_providers import CapabilitySnapshot
from .o7_science_compat import (
    ENTITY_REF,
    PACK_REF,
    TEST_BOUNDARY_SENTINEL,
    TEST_DIVERSITY,
    TEST_SATURATION,
    TEST_REPRESENTATION,
    TEST_GRAMMAR,
    LEAF_TEST_REFS,
    run_observation_test,
    run_regime_boundary_sentinel,
)


class MaturationAuditError(ScientificArchitectureError):
    pass


REVIEW_TRIGGER_STATUSES = frozenset({"OBSERVED_VARIATION", "CANDIDATE", "COUNTEREXAMPLE", "REOPEN_TRIGGER"})
# Diversity and saturation are intentionally allowed to vary as complexity diagnostics.
# Every other leaf Test is treated as a review-critical scientific lane.
NON_STOPPING_GROWTH_TEST_REFS = frozenset({TEST_DIVERSITY, TEST_SATURATION})


@dataclass(frozen=True)
class LevelTestRecord:
    depth: int
    provider_ref: str
    provider_kind: str
    source_science_sha256: str
    executions: Mapping[str, TestExecution]
    stabilization_signature_sha256: str
    complexity_projection: Mapping[str, Any]
    complexity_growth_from_previous: bool | None
    provider_seam: bool
    provider_seam_calibrated: bool = False
    provider_seam_calibration_sha256: str | None = None
    longitudinal_candidate_cohort: Mapping[str, Any] | None = None
    longitudinal_discriminator: Mapping[str, Any] | None = None
    stabilization_basis: str = "SELECTED_PANEL_TESTPACK"


def _needed_leaf_refs(selected_refs: Iterable[str], registry: ScientificProtocolRegistry) -> set[str]:
    selected = set(selected_refs)
    needed = set(selected & set(LEAF_TEST_REFS))
    if TEST_BOUNDARY_SENTINEL in selected:
        needed.update(registry.test(TEST_BOUNDARY_SENTINEL).get("dependency_test_refs", []))
    return needed


def _leaf_status(previous: TestExecution | None, current_evidence_sha: str, *, provider_seam: bool) -> str:
    if previous is None or provider_seam:
        return "NULL"
    return "INVARIANT" if previous.evidence_sha256 == current_evidence_sha else "OBSERVED_VARIATION"


def _plateau_projection(executions: Mapping[str, TestExecution]) -> dict[str, Any]:
    """Projection used only for bounded maturation/plateau recognition.

    Absolute resource magnitude growth is deliberately excluded.  The saturation Test contributes
    only its normalized observer signature; representation contributes only the matched structural
    signature and earned-law identities.  This mirrors the frozen scanner rule that raw growth is
    a complexity guard, not organizational novelty.
    """
    out: dict[str, Any] = {}
    for tref in sorted(set(LEAF_TEST_REFS) & set(executions)):
        ev = executions[tref].evidence_payload
        if tref == TEST_SATURATION:
            out[tref] = {"normalized_signature_sha256": ev["normalized_signature_sha256"]}
        elif tref == TEST_DIVERSITY:
            # Resource magnitudes are a growth guard, not organizational novelty.
            out[tref] = {k: v for k, v in ev.items() if k not in {"min_total_free_by_type", "population_materialized"}}
        elif tref == TEST_REPRESENTATION:
            matched = ev.get("matched_backbone", {})
            out[tref] = {
                "matched_signature_sha256": matched.get("signature_sha256"),
                "matched_status": matched.get("status"),
                "inherited_law_ids": sorted(x.get("law_id") for x in ev.get("inherited_law_events", []) if x.get("law_id")),
            }
        else:
            out[tref] = ev
    return out


def _complexity_projection(executions: Mapping[str, TestExecution]) -> dict[str, Any]:
    if TEST_SATURATION not in executions:
        return {"status": "UNAVAILABLE"}
    raw = executions[TEST_SATURATION].evidence_payload.get("raw_growth", {})
    if raw.get("mode") == "CERTIFIED_CONSERVATIVE_BOUNDS":
        return {
            "status": "CERTIFIED_BOUNDS",
            "leaf": raw.get("leaf_count_min"),
            "relations": raw.get("relation_count_min"),
            "free": sum(raw.get("min_total_free_by_type", [])),
        }
    # Exact/bounded scanner summary path.
    candidates = {
        "leaf": raw.get("median_leaf_count"),
        "relations": raw.get("median_total_relations"),
        "free": raw.get("median_total_free"),
    }
    return {"status": "SCANNER_DIAGNOSTICS", **candidates}


def _complexity_grew(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> bool:
    if prev.get("status") == "UNAVAILABLE" or cur.get("status") == "UNAVAILABLE":
        return False
    comparable = []
    for k in ("leaf", "relations", "free"):
        a, b = prev.get(k), cur.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            comparable.append((a, b))
    return bool(comparable) and any(b > a for a, b in comparable) and all(b >= a for a, b in comparable)



def _candidate_cohort_from_snapshot(snapshot: CapabilitySnapshot) -> Mapping[str, Any] | None:
    ident = snapshot.payloads.get("EXACT_ENTITY_IDENTITY", {})
    if not isinstance(ident, Mapping):
        return None
    cohort = ident.get("longitudinal_candidate_cohort")
    return cohort if isinstance(cohort, Mapping) else None


def _cohort_rows(cohort: Mapping[str, Any]) -> dict[str, str]:
    rows = cohort.get("motif_structural_signatures", [])
    out: dict[str, str] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            raise MaturationAuditError("candidate cohort row is not a mapping")
        motif = str(row.get("motif_id", ""))
        sig = str(row.get("structural_seed_sha256", ""))
        if not motif or not sig or motif in out:
            raise MaturationAuditError("candidate cohort contains invalid or duplicate motif identity")
        out[motif] = sig
    if int(cohort.get("candidate_count", -1)) != len(out):
        raise MaturationAuditError("candidate cohort count does not match unique motif rows")
    return out


def _longitudinal_discriminator(
    previous: LevelTestRecord | None,
    current_cohort: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Separate within-entity evolution from selected-panel composition turnover.

    This is intentionally fail-closed.  Suppression of aggregate selected-panel variation is
    authorized only when both adjacent depths expose complete candidate cohorts and every
    common motif carries an identical depth-erased structural seed.  New/lost motif identities
    or any changed motif seed remain scientific review events.
    """
    if previous is None or current_cohort is None or previous.longitudinal_candidate_cohort is None:
        return None
    prev_cohort = previous.longitudinal_candidate_cohort
    prev_rows = _cohort_rows(prev_cohort)
    cur_rows = _cohort_rows(current_cohort)
    prev_ids, cur_ids = set(prev_rows), set(cur_rows)
    entered = sorted(cur_ids - prev_ids)
    exited = sorted(prev_ids - cur_ids)
    changed = sorted(m for m in prev_ids & cur_ids if prev_rows[m] != cur_rows[m])
    prev_selected = set(map(str, prev_cohort.get("selected_motif_ids", [])))
    cur_selected = set(map(str, current_cohort.get("selected_motif_ids", [])))
    selected_entered = sorted(cur_selected - prev_selected)
    selected_exited = sorted(prev_selected - cur_selected)
    if entered or exited:
        classification = "CANDIDATE_POPULATION_MEMBERSHIP_CHANGE"
        aggregate_panel_variation_suppressible = False
    elif changed:
        classification = "WITHIN_ENTITY_STRUCTURAL_EVOLUTION"
        aggregate_panel_variation_suppressible = False
    elif selected_entered or selected_exited:
        classification = "PANEL_TURNOVER_ONLY_OVER_STRUCTURALLY_INVARIANT_COHORT"
        aggregate_panel_variation_suppressible = True
    else:
        classification = "COHORT_AND_PANEL_STRUCTURALLY_INVARIANT"
        aggregate_panel_variation_suppressible = True
    turnover_authority_payload = {
        "schema_id": "IG_PANEL_TURNOVER_AUTHORITY_V1",
        "previous_population_structural_signature_sha256": prev_cohort.get("population_structural_signature_sha256"),
        "current_population_structural_signature_sha256": current_cohort.get("population_structural_signature_sha256"),
        "previous_motif_id_set_sha256": prev_cohort.get("motif_id_set_sha256"),
        "current_motif_id_set_sha256": current_cohort.get("motif_id_set_sha256"),
        "candidate_motif_ids_entered": entered,
        "candidate_motif_ids_exited": exited,
        "within_motif_structural_changes": changed,
        "selected_panel_entered": selected_entered,
        "selected_panel_exited": selected_exited,
    }
    turnover_authority_verified = bool(
        not entered and not exited and not changed
        and prev_cohort.get("population_structural_signature_sha256") == current_cohort.get("population_structural_signature_sha256")
        and prev_cohort.get("motif_id_set_sha256") == current_cohort.get("motif_id_set_sha256")
        and isinstance(prev_cohort.get("science_sha256"), str)
        and isinstance(current_cohort.get("science_sha256"), str)
    )
    obj = {
        "schema_id": "IG_O_REGIME_LONGITUDINAL_MATURATION_DISCRIMINATOR_V1",
        "schema_version": "1.0.0",
        "classification": classification,
        "previous_candidate_count": len(prev_rows),
        "current_candidate_count": len(cur_rows),
        "candidate_motif_ids_entered": entered,
        "candidate_motif_ids_exited": exited,
        "within_motif_structural_changes": changed,
        "within_motif_structural_change_count": len(changed),
        "selected_panel_overlap": len(prev_selected & cur_selected),
        "selected_panel_entered": selected_entered,
        "selected_panel_exited": selected_exited,
        "aggregate_panel_variation_suppressible": aggregate_panel_variation_suppressible,
        "turnover_only_authority_verified": turnover_authority_verified,
        "turnover_only_authority_sha256": canonical_sha256(turnover_authority_payload) if turnover_authority_verified else None,
        "turnover_only_authority_payload": turnover_authority_payload if turnover_authority_verified else None,
        "scientific_rule": (
            "Selected-panel aggregate variation cannot establish a regime boundary when the complete "
            "deterministic candidate cohort is motif-for-motif structurally invariant after depth erasure."
        ),
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


def _cohort_stabilization_signature(snapshot: CapabilitySnapshot, cohort: Mapping[str, Any]) -> str:
    transition = snapshot.payloads.get("TRANSITION_SYSTEM", {})
    return canonical_sha256({
        "mode": "LONGITUDINAL_COMPLETE_CANDIDATE_COHORT_V1",
        "population_structural_signature_sha256": cohort.get("population_structural_signature_sha256"),
        "motif_id_set_sha256": cohort.get("motif_id_set_sha256"),
        "grammar_sha256": transition.get("grammar_sha256") if isinstance(transition, Mapping) else None,
        "operational_read_signature_sha256": transition.get("operational_read_signature_sha256") if isinstance(transition, Mapping) else None,
    })


def _panel_turnover_only(row: LevelTestRecord) -> bool:
    """Return true only for internally consistent, authority-bound turnover evidence.

    The v0.28.7 aggregate boolean was fail-open: later mutation of the
    discriminator could leave a stale ``turnover_only_authority_verified`` flag
    behind and still suppress unrelated variation.  Recompute both the full
    discriminator identity and the turnover authority from the fields they
    authenticate before permitting demotion.
    """
    d = row.longitudinal_discriminator
    if not isinstance(d, Mapping):
        return False
    if d.get("turnover_only_authority_verified") is not True:
        return False
    if d.get("classification") not in {
        "PANEL_TURNOVER_ONLY_OVER_STRUCTURALLY_INVARIANT_COHORT",
        "COHORT_AND_PANEL_STRUCTURALLY_INVARIANT",
    }:
        return False
    if d.get("candidate_motif_ids_entered") or d.get("candidate_motif_ids_exited"):
        return False
    if d.get("within_motif_structural_changes") or int(d.get("within_motif_structural_change_count", 0) or 0) != 0:
        return False

    # The discriminator itself is science-hashed.  Never trust a stale nested
    # authority after any post-construction mutation.
    declared_science = d.get("science_sha256")
    if not isinstance(declared_science, str) or len(declared_science) != 64:
        return False
    body = dict(d)
    body.pop("science_sha256", None)
    if canonical_sha256(body) != declared_science:
        return False

    payload = d.get("turnover_only_authority_payload")
    declared_authority = d.get("turnover_only_authority_sha256")
    if not isinstance(payload, Mapping) or not isinstance(declared_authority, str) or len(declared_authority) != 64:
        return False
    expected_payload = {
        "schema_id": "IG_PANEL_TURNOVER_AUTHORITY_V1",
        "previous_population_structural_signature_sha256": payload.get("previous_population_structural_signature_sha256"),
        "current_population_structural_signature_sha256": payload.get("current_population_structural_signature_sha256"),
        "previous_motif_id_set_sha256": payload.get("previous_motif_id_set_sha256"),
        "current_motif_id_set_sha256": payload.get("current_motif_id_set_sha256"),
        "candidate_motif_ids_entered": list(d.get("candidate_motif_ids_entered", [])),
        "candidate_motif_ids_exited": list(d.get("candidate_motif_ids_exited", [])),
        "within_motif_structural_changes": list(d.get("within_motif_structural_changes", [])),
        "selected_panel_entered": list(d.get("selected_panel_entered", [])),
        "selected_panel_exited": list(d.get("selected_panel_exited", [])),
    }
    return bool(dict(payload) == expected_payload and canonical_sha256(expected_payload) == declared_authority)


def execute_snapshot_tests(
    snapshot: CapabilitySnapshot,
    *,
    selected_test_refs: Iterable[str] | None = None,
    previous: LevelTestRecord | None = None,
    registry: ScientificProtocolRegistry | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    seam_baseline: LevelTestRecord | None = None,
    seam_calibration: Mapping[str, Any] | None = None,
) -> LevelTestRecord:
    registry = registry or ScientificProtocolRegistry()
    selected = tuple(selected_test_refs or [m["test_ref"] for m in registry.pack(PACK_REF)["members"]])
    provider_seam = previous is not None and previous.provider_ref != snapshot.provider_ref
    provider_seam_calibrated = False
    provider_seam_calibration_sha256 = None
    comparison_previous = previous
    if provider_seam and seam_baseline is not None:
        from .provider_seam_calibration import verify_seam_calibration_certificate
        if seam_calibration is None:
            raise MaturationAuditError("provider seam baseline supplied without calibration certificate")
        verify_seam_calibration_certificate(seam_calibration)
        if seam_baseline.depth != previous.depth or seam_baseline.provider_ref != snapshot.provider_ref:
            raise MaturationAuditError("provider seam calibration baseline does not bind previous depth/new provider")
        comparison_previous = seam_baseline
        provider_seam_calibrated = True
        provider_seam_calibration_sha256 = str(seam_calibration["science_sha256"])
    needed = _needed_leaf_refs(selected, registry)
    leaves: dict[str, TestExecution] = {}
    for tref in LEAF_TEST_REFS:
        if tref not in needed:
            continue
        if progress_callback is not None:
            progress_callback("TEST_START", {"depth": snapshot.regime_depth, "test_ref": tref})
        view = snapshot.observation_view(tref, registry=registry)
        # Compute evidence once, then determine status against the same-provider previous depth.
        provisional = run_observation_test(
            view=view,
            test_ref=tref,
            source_science_sha256=snapshot.source_science_sha256,
            scientific_status="NULL",
            summary=f"Generic live Test evidence at regime depth {snapshot.regime_depth}; provider={snapshot.provider_kind}.",
            historical_labels=[f"O{snapshot.regime_depth}", snapshot.provider_kind],
            registry=registry,
        )
        prev_exec = None if comparison_previous is None or (provider_seam and not provider_seam_calibrated) else comparison_previous.executions.get(tref)
        status = _leaf_status(prev_exec, provisional.evidence_sha256, provider_seam=(provider_seam and not provider_seam_calibrated))
        # A materialized read probe can itself produce a present-depth separator.  That is not
        # merely longitudinal variation: even the first sampled discovery depth must stop for
        # scientific review if the inherited resource quotient is demonstrably too coarse for
        # the admitted action future.
        if (
            tref == TEST_GRAMMAR
            and provisional.evidence_payload.get("operational_read_classification")
            == "INHERITED_RESOURCE_QUOTIENT_BREAK_CANDIDATE"
        ):
            status = "CANDIDATE"
        if status == "NULL":
            leaves[tref] = provisional
        else:
            # v0.28.5 performance-neutral repair: the scientific adapter has already
            # produced the complete evidence payload above.  Rebuild only the Finding
            # wrapper with the longitudinal status; never execute the adapter twice.
            # build_finding derives the same finding_id/evidence_ref from the unchanged
            # evidence payload, so this is byte-for-byte science-equivalent to the old
            # second run_observation_test call.
            spec = registry.test(tref)
            finding = build_finding(
                test_spec=spec,
                entity_instance_ref=view.entity_instance_ref,
                regime_ref=view.regime_ref,
                regime_depth=view.regime_depth,
                source_science_sha256=snapshot.source_science_sha256,
                evidence_payload=provisional.evidence_payload,
                scientific_status=status,
                summary=(
                    f"Generic live Test evidence at regime depth {snapshot.regime_depth}; "
                    f"{status.lower().replace('_', ' ')} relative to prior same-provider depth."
                ),
                historical_labels=[f"O{snapshot.regime_depth}", snapshot.provider_kind],
            )
            leaves[tref] = TestExecution(
                test_ref=tref,
                evidence_payload=provisional.evidence_payload,
                evidence_sha256=provisional.evidence_sha256,
                finding=finding,
                observation_view=provisional.observation_view,
            )
        if progress_callback is not None:
            progress_callback("TEST_COMPLETE", {
                "depth": snapshot.regime_depth,
                "test_ref": tref,
                "scientific_status": leaves[tref].finding["scientific_status"],
                "evidence_sha256": leaves[tref].evidence_sha256,
            })

    executions: dict[str, TestExecution] = dict(leaves)
    if TEST_BOUNDARY_SENTINEL in selected:
        deps = registry.test(TEST_BOUNDARY_SENTINEL)["dependency_test_refs"]
        cur_deps = {d: leaves[d] for d in deps}
        prev_deps = None
        if comparison_previous is not None and not (provider_seam and not provider_seam_calibrated):
            prev_deps = {d: comparison_previous.executions[d] for d in deps if d in comparison_previous.executions}
            if len(prev_deps) != len(deps):
                prev_deps = None
        if progress_callback is not None:
            progress_callback("TEST_START", {"depth": snapshot.regime_depth, "test_ref": TEST_BOUNDARY_SENTINEL})
        sentinel = run_regime_boundary_sentinel(
            current=cur_deps,
            previous=prev_deps,
            source_science_sha256=snapshot.source_science_sha256,
            level=snapshot.regime_depth,
            registry=registry,
            entity_instance_ref=snapshot.entity_instance_ref,
            regime_ref=snapshot.regime_ref,
        )
        executions[TEST_BOUNDARY_SENTINEL] = sentinel
        if progress_callback is not None:
            progress_callback("TEST_COMPLETE", {
                "depth": snapshot.regime_depth,
                "test_ref": TEST_BOUNDARY_SENTINEL,
                "scientific_status": sentinel.finding["scientific_status"],
                "evidence_sha256": sentinel.evidence_sha256,
            })

    plateau_projection = _plateau_projection(executions)
    complexity = _complexity_projection(executions)
    grew = None if comparison_previous is None or (provider_seam and not provider_seam_calibrated) else _complexity_grew(comparison_previous.complexity_projection, complexity)
    cohort = _candidate_cohort_from_snapshot(snapshot)
    discriminator_previous = None if (provider_seam and not provider_seam_calibrated) else comparison_previous
    discriminator = _longitudinal_discriminator(discriminator_previous, cohort)
    if cohort is not None:
        stabilization_signature = _cohort_stabilization_signature(snapshot, cohort)
        stabilization_basis = "LONGITUDINAL_COMPLETE_CANDIDATE_COHORT"
    else:
        stabilization_signature = canonical_sha256(plateau_projection)
        stabilization_basis = "SELECTED_PANEL_TESTPACK"
    return LevelTestRecord(
        depth=snapshot.regime_depth,
        provider_ref=snapshot.provider_ref,
        provider_kind=snapshot.provider_kind,
        source_science_sha256=snapshot.source_science_sha256,
        executions=executions,
        stabilization_signature_sha256=stabilization_signature,
        complexity_projection=complexity,
        complexity_growth_from_previous=grew,
        provider_seam=provider_seam,
        provider_seam_calibrated=provider_seam_calibrated,
        provider_seam_calibration_sha256=provider_seam_calibration_sha256,
        longitudinal_candidate_cohort=None if cohort is None else dict(cohort),
        longitudinal_discriminator=None if discriminator is None else dict(discriminator),
        stabilization_basis=stabilization_basis,
    )


def _confirmed_shock_events(levels: list[LevelTestRecord], registry: ScientificProtocolRegistry) -> list[dict[str, Any]]:
    """Return composite shock candidates, including a terminal candidate awaiting confirmation."""
    deps = registry.test(TEST_BOUNDARY_SENTINEL)["dependency_test_refs"]
    events: list[dict[str, Any]] = []
    for i in range(1, len(levels)):
        prev, cur = levels[i-1], levels[i]
        if (cur.provider_seam or prev.provider_ref != cur.provider_ref) and not cur.provider_seam_calibrated:
            continue
        # Aggregate selected-panel movement is not a structural shock when the complete
        # longitudinal cohort is motif-for-motif depth-erased invariant.
        if _panel_turnover_only(cur):
            continue
        changed = [d for d in deps if prev.executions[d].evidence_sha256 != cur.executions[d].evidence_sha256]
        if len(changed) < 3:
            continue
        if i + 1 >= len(levels):
            events.append({
                "depth": cur.depth,
                "changed_dependency_test_refs": changed,
                "held_one_depth_later": [],
                "confirmed": False,
                "confirmation_available": False,
                "confirmation_depth": None,
            })
            continue
        nxt = levels[i+1]
        if (nxt.provider_seam or cur.provider_ref != nxt.provider_ref) and not nxt.provider_seam_calibrated:
            events.append({
                "depth": cur.depth,
                "changed_dependency_test_refs": changed,
                "held_one_depth_later": [],
                "confirmed": False,
                "confirmation_available": False,
                "confirmation_depth": None,
                "confirmation_blocked_by_provider_seam": True,
            })
            continue
        held = [d for d in changed if cur.executions[d].evidence_sha256 == nxt.executions[d].evidence_sha256]
        events.append({
            "depth": cur.depth,
            "changed_dependency_test_refs": changed,
            "held_one_depth_later": held,
            "confirmed": len(held) >= 3,
            "confirmation_available": True,
            "confirmation_depth": nxt.depth,
        })
    return events


def _review_gate(levels: list[LevelTestRecord], *, registry: ScientificProtocolRegistry) -> dict[str, Any]:
    """Evaluate the fail-closed scientific stop boundary for the latest committed depth.

    The gate is intentionally stronger than the 3-of-4 composite shock rule: any variation in
    a review-critical leaf lane pauses execution. Provider seams are uncomparable until a seam
    calibration certificate exists, so they also require explicit review acknowledgement.
    """
    latest = levels[-1]
    reasons: list[dict[str, Any]] = []
    # O7..O13 scanner summaries are frozen historical compatibility replay, not a new live
    # maturation campaign. Do not stop on their already-known internal variation; the first
    # live governance boundary is the provider seam into O14+.
    historical_compat_replay = latest.provider_kind == "LIVE_BOUNDED_SCANNER_SUMMARY" and int(latest.depth) <= 13
    if latest.provider_seam and not latest.provider_seam_calibrated:
        reasons.append({
            "kind": "UNCOMPARABLE_PROVIDER_SEAM",
            "depth": latest.depth,
            "provider_ref": latest.provider_ref,
            "message": "Provider changed without a bound seam-calibration certificate.",
        })

    critical_variations: list[dict[str, Any]] = []
    for tref in LEAF_TEST_REFS:
        if historical_compat_replay:
            continue
        if tref in NON_STOPPING_GROWTH_TEST_REFS:
            continue
        exe = latest.executions.get(tref)
        if exe is None:
            continue
        status = exe.finding.get("scientific_status")
        # A present-depth CANDIDATE/COUNTEREXAMPLE/REOPEN is never suppressed.  Only
        # longitudinal OBSERVED_VARIATION caused by selected-panel composition may be demoted
        # when the complete candidate cohort proves structural invariance.
        if status == "OBSERVED_VARIATION" and _panel_turnover_only(latest):
            continue
        if status in REVIEW_TRIGGER_STATUSES:
            critical_variations.append({"test_ref": tref, "scientific_status": status, "evidence_sha256": exe.evidence_sha256})
    if critical_variations:
        reasons.append({
            "kind": "CRITICAL_LANE_VARIATION",
            "depth": latest.depth,
            "lanes": critical_variations,
        })

    sentinel = latest.executions.get(TEST_BOUNDARY_SENTINEL)
    if sentinel is not None and not (latest.provider_seam and not latest.provider_seam_calibrated) and not historical_compat_replay:
        sev = sentinel.evidence_payload if isinstance(sentinel.evidence_payload, Mapping) else {}
        if sentinel.finding.get("scientific_status") == "CANDIDATE" and not _panel_turnover_only(latest):
            reasons.append({
                "kind": "REGIME_BOUNDARY_CANDIDATE",
                "depth": latest.depth,
                "changed_dependency_test_refs": list(sev.get("changed_dependency_test_refs", [])),
                "confirmation_required": True,
            })

    shocks = [] if historical_compat_replay else _confirmed_shock_events(levels, registry)
    for event in shocks:
        if event.get("confirmed") and event.get("confirmation_depth") == latest.depth:
            reasons.append({
                "kind": "CONFIRMED_STRUCTURAL_SHOCK",
                "depth": event["depth"],
                "confirmation_depth": latest.depth,
                "changed_dependency_test_refs": list(event["changed_dependency_test_refs"]),
                "held_one_depth_later": list(event["held_one_depth_later"]),
            })
        elif event.get("depth") == latest.depth and not event.get("confirmation_available"):
            reasons.append({
                "kind": "UNRESOLVED_NEEDS_NEXT_DEPTH",
                "depth": latest.depth,
                "changed_dependency_test_refs": list(event["changed_dependency_test_refs"]),
            })

    diagnostics: list[dict[str, Any]] = []
    if _panel_turnover_only(latest):
        diagnostics.append({
            "kind": "SELECTED_PANEL_TURNOVER_DEMOTED_BY_COMPLETE_COHORT",
            "depth": latest.depth,
            "longitudinal_discriminator": dict(latest.longitudinal_discriminator or {}),
        })
    gate = {
        "schema_id": "IG_SCIENTIFIC_REVIEW_GATE_V1",
        "schema_version": "1.0.0",
        "status": "REVIEW_REQUIRED" if reasons else "CLEAR",
        "required": bool(reasons),
        "evaluated_through_depth": latest.depth,
        "reasons": reasons,
        "non_stopping_diagnostics": diagnostics,
        "automatic_advance_authorized": not bool(reasons),
        "acknowledgement_required_to_continue": bool(reasons),
        "frontier_mutation_authorized": False,
        "plateau_or_lift_promotion_authorized": False,
    }
    gate["science_sha256"] = canonical_sha256(gate)
    return gate


def _plateau_candidate(levels: list[LevelTestRecord], *, window: int, pack_ref: str, registry: ScientificProtocolRegistry) -> dict[str, Any] | None:
    if len(levels) < window:
        return None
    shocks = [x for x in _confirmed_shock_events(levels, registry) if x["confirmed"]]
    last_shock_depth = max((x["depth"] for x in shocks), default=None)
    for end_idx in range(window - 1, len(levels)):
        win = levels[end_idx-window+1:end_idx+1]
        if any(x.provider_seam for x in win[1:]):
            continue
        if len({x.provider_ref for x in win}) != 1:
            continue
        if last_shock_depth is not None and win[0].depth <= last_shock_depth:
            continue
        if len({x.stabilization_signature_sha256 for x in win}) != 1:
            continue
        if not all(x.complexity_growth_from_previous is True for x in win[1:]):
            continue
        ev_payload = {
            "depths": [x.depth for x in win],
            "provider_ref": win[0].provider_ref,
            "stabilization_signature_sha256": win[0].stabilization_signature_sha256,
            "complexity": [dict(x.complexity_projection) for x in win],
        }
        ev_sha = canonical_sha256(ev_payload)
        obj = {
            "schema_id": "IG_PLATEAU_CERTIFICATE_V1",
            "schema_version": "1.0.0",
            "plateau_id": canonical_sha256({"pack_ref": pack_ref, "depths": [x.depth for x in win], "signature": win[0].stabilization_signature_sha256}),
            "status": "CANDIDATE",
            "entity_class_ref": ENTITY_REF,
            "regime_ref": win[0].executions[next(iter(win[0].executions))].finding["regime_ref"],
            "observer_or_test_pack_ref": pack_ref,
            "scope_class": registry.pack(pack_ref)["claim_ceiling"],
            "depth_window": {"start": win[0].depth, "end": win[-1].depth, "width": window},
            "stabilization_definition": "Byte-identical normalized TestPack stabilization projection across the declared window after provider-seam and confirmed-shock exclusion.",
            "persistence_rule": f"Normalized TestPack signature persists for W={window} consecutive same-provider depths.",
            "complexity_growth_guard": "At least one declared complexity diagnostic strictly increases and none of the comparable guards decreases at every step in the window.",
            "evidence_refs": [{"id": f"maturation-window:{ev_sha}", "version": "1", "role": "bounded plateau support", "scope": f"depths {win[0].depth}..{win[-1].depth}", "sha256": ev_sha}],
            "outstanding_counterexamples": [],
            "reopen_conditions": [
                "TestPack membership/version changes",
                "observer/projection changes",
                "Regime grammar or provider theorem premise changes",
                "new counterexample appears",
            ],
            "nonclaim": "NOT_ETERNAL_CLOSURE_AND_NOT_AUTOMATIC_LIFT",
        }
        validate_scientific_artifact(obj)
        return obj
    return None


def audit_maturation(
    levels: Iterable[LevelTestRecord],
    *,
    test_pack_ref: str = PACK_REF,
    plateau_window: int = 4,
    registry: ScientificProtocolRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or ScientificProtocolRegistry()
    rows = sorted(list(levels), key=lambda x: x.depth)
    if not rows:
        raise MaturationAuditError("maturation audit requires at least one completed depth")
    if len({x.depth for x in rows}) != len(rows):
        raise MaturationAuditError("duplicate regime depth in maturation audit")

    shocks = _confirmed_shock_events(rows, registry)
    seams = [x.depth for x in rows if x.provider_seam]
    calibrated_seams = [x.depth for x in rows if x.provider_seam and x.provider_seam_calibrated]
    uncalibrated_seams = [x.depth for x in rows if x.provider_seam and not x.provider_seam_calibrated]
    finding_ids = [
        exe.finding["finding_id"]
        for row in rows
        for _, exe in sorted(row.executions.items())
    ]
    novelty_events = [f"CALIBRATED_PROVIDER_SEAM@O{d}" for d in calibrated_seams]
    novelty_events.extend(f"PROVIDER_SEAM@O{d}" for d in uncalibrated_seams)
    novelty_events.extend(f"CONFIRMED_STRUCTURAL_SHOCK@O{x['depth']}" for x in shocks if x["confirmed"])
    for row in rows:
        disc = row.longitudinal_discriminator
        if not isinstance(disc, Mapping):
            continue
        cls = disc.get("classification")
        if cls == "PANEL_TURNOVER_ONLY_OVER_STRUCTURALLY_INVARIANT_COHORT":
            novelty_events.append(f"PANEL_COMPOSITION_TURNOVER_ONLY@O{row.depth}")
        elif cls == "WITHIN_ENTITY_STRUCTURAL_EVOLUTION":
            novelty_events.append(f"WITHIN_ENTITY_STRUCTURAL_EVOLUTION@O{row.depth}")
        elif cls == "CANDIDATE_POPULATION_MEMBERSHIP_CHANGE":
            novelty_events.append(f"CANDIDATE_POPULATION_MEMBERSHIP_CHANGE@O{row.depth}")
    unresolved = [f"UNCOMPARABLE_PROVIDER_SEAM@O{d}" for d in uncalibrated_seams]
    for x in shocks:
        if x["confirmed"]:
            continue
        if not x.get("confirmation_available"):
            unresolved.append(f"UNRESOLVED_NEEDS_NEXT_DEPTH@O{x['depth']}")
        else:
            unresolved.append(f"UNCONFIRMED_STRUCTURAL_SHOCK@O{x['depth']}")
    # Grammar/reopen status is no longer inferred from one arbitrary leaf. A change in any
    # review-critical Test lane is enough to require review; diversity/saturation remain growth guards.
    grammar_stable = all(
        (
            exe.finding["scientific_status"] not in REVIEW_TRIGGER_STATUSES
            or (exe.finding["scientific_status"] == "OBSERVED_VARIATION" and _panel_turnover_only(row))
        )
        for row in rows
        for tref, exe in row.executions.items()
        if tref in set(LEAF_TEST_REFS) - set(NON_STOPPING_GROWTH_TEST_REFS)
    )
    maturation = {
        "schema_id": "IG_MATURATION_RECORD_V1",
        "schema_version": "1.0.0",
        "maturation_id": canonical_sha256({"depths": [x.depth for x in rows], "pack": test_pack_ref, "findings": finding_ids}),
        "entity_class_ref": ENTITY_REF,
        "regime_ref": rows[0].executions[next(iter(rows[0].executions))].finding["regime_ref"],
        "depth_start": rows[0].depth,
        "depth_end": rows[-1].depth,
        "test_pack_refs": [test_pack_ref],
        "finding_refs": finding_ids,
        "novelty_events": novelty_events,
        "unresolved_findings": unresolved,
        "grammar_status": "STABLE_IN_DECLARED_TEST_SCOPE" if grammar_stable else "VARIATION_OR_REOPEN_REQUIRES_REVIEW",
        "notes": [
            "Provider seams are physical provenance boundaries. Uncalibrated seams are UNCOMPARABLE; a bound overlap certificate permits longitudinal comparison without treating the provider change itself as novelty.",
            "Raw magnitude growth is used only as a complexity guard, not as plateau-breaking novelty.",
            "Maturation audit cannot perform Lift or mutate ResearchFrontier.",
            "For materialized discovery, complete pre-selection candidate-cohort invariance outranks selected-panel aggregate turnover for regime-boundary maturation. Present-depth candidates/counterexamples are never suppressed.",
        ],
    }
    validate_scientific_artifact(maturation)
    review_gate = _review_gate(rows, registry=registry)
    plateau = None
    if review_gate.get("status") == "CLEAR" and grammar_stable:
        plateau = _plateau_candidate(rows, window=int(plateau_window), pack_ref=test_pack_ref, registry=registry)
    result = {
        "schema_id": "IG_MATURATION_AUDIT_RESULT_V1",
        "schema_version": "1.0.0",
        "status": "PASS",
        "classification": "GENERIC_MATURATION_AUDIT",
        "maturation_record": maturation,
        "confirmed_structural_shocks": shocks,
        "provider_seams": seams,
        "calibrated_provider_seams": calibrated_seams,
        "uncalibrated_provider_seams": uncalibrated_seams,
        "review_gate": review_gate,
        "longitudinal_discriminators": {str(x.depth): x.longitudinal_discriminator for x in rows if x.longitudinal_discriminator is not None},
        "plateau_candidate": plateau,
        "plateau_certified": False,
        "lift_certified": False,
        "research_frontier_mutated": False,
        "new_scientific_claim": False,
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
