from __future__ import annotations

"""RunPlan V2 validator/compiler for the generic Entity -> Regime -> Test architecture.

RunPlan V2 is declarative scientific intent.  Compilation resolves all Test/TestPack
references and emits deterministic execution-task intents against existing Decoder
execution services.  Compilation never mutates the ResearchFrontier and never promotes a
plateau or lift.

In v0.28.1, O7..O13 remain compatibility replay while O14+ is supplied by a materialized
finite structural-discovery provider regenerated from the certified O7 compact replay root.
Fixed-grammar theorem transport remains active only as an auxiliary consistency/growth-bound
lane. The O13->O14 seam is calibrated by a mechanically checked O13 overlap projection.
MaturationAuditor, PlateauAuditor and LiftAuditor are three separate gates; none may mutate
ResearchFrontier automatically.
"""

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

from .canon import canonical_sha256, write_json_atomic
from .scientific_architecture import (
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    make_ref,
    validate_scientific_artifact,
)
from .research_frontier import load_research_frontier, verify_research_frontier
from .o7_science_compat import (
    ENTITY_REF as O7_ENTITY_REF,
    REGIME_REF as O7_REGIME_REF,
    PACK_REF as O7_PACK_REF,
    LEAF_TEST_REFS as O7_LEAF_TEST_REFS,
    TEST_BOUNDARY_SENTINEL as O7_BOUNDARY_SENTINEL,
    load_frozen_legacy_scanner_fixture,
    run_leaf_test,
    run_regime_boundary_sentinel,
)
from .capability_providers import from_scanner_level_summary, fixed_grammar_snapshot, load_fixed_grammar_delta, from_materialized_discovery
from .maturation_auditor import execute_snapshot_tests, audit_maturation
from .plateau_auditor import audit_plateau
from .lift_auditor import audit_lift
from .fixed_grammar_transport import run_fixed_grammar_transport
from .materialized_discovery import MaterializedDiscoverySession, theorem_transport_consistency
from .provider_seam_calibration import calibrate_o13_o14_provider_seam
from .execution import ExecutionPolicy


class RunPlanV2Error(ScientificArchitectureError):
    pass


@dataclass(frozen=True)
class RunPlanResolution:
    plan: Mapping[str, Any]
    frontier: Mapping[str, Any]
    selected_test_refs: tuple[str, ...]
    plan_science_sha256: str
    frontier_science_sha256: str
    start_depth: int
    max_depth: int | None
    fully_launchable_depth_max: int | None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RunPlanV2Error(f"cannot parse RunPlan V2 {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RunPlanV2Error("RunPlan V2 must be a JSON object")
    return obj


def load_run_plan_v2(path: str | Path) -> dict[str, Any]:
    obj = _load_json(Path(path))
    validate_scientific_artifact(obj)
    return obj


def _depth(value: Any, *, label: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise RunPlanV2Error(f"{label} cannot be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.upper().startswith("O") and raw[1:].isdigit():
            return int(raw[1:])
        if raw.isdigit():
            return int(raw)
    raise RunPlanV2Error(f"{label} must be integer or O<number>{' or null' if allow_none else ''}")


def _topological_test_order(test_refs: Iterable[str], registry: ScientificProtocolRegistry) -> tuple[str, ...]:
    refs = sorted(set(test_refs))
    outgoing = {r: [] for r in refs}
    indegree = {r: 0 for r in refs}
    for r in refs:
        spec = registry.test(r)
        for dep in spec.get("dependency_test_refs", []):
            if dep not in outgoing:
                raise RunPlanV2Error(f"selected Test {r} requires unselected dependency {dep}")
            outgoing[dep].append(r)
            indegree[r] += 1
    ready = sorted(r for r, d in indegree.items() if d == 0)
    out: list[str] = []
    while ready:
        r = ready.pop(0)
        out.append(r)
        for nxt in sorted(outgoing[r]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(out) != len(refs):
        raise RunPlanV2Error("selected Test dependency graph contains a cycle")
    return tuple(out)


def resolve_run_plan_v2(
    plan: Mapping[str, Any],
    *,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> RunPlanResolution:
    validate_scientific_artifact(plan)
    registry = registry or ScientificProtocolRegistry()
    frontier = dict(frontier or load_research_frontier())
    verify_research_frontier(frontier, registry=registry, raise_on_error=True)

    entity_ref = str(plan["entity_class_ref"])
    regime_ref = str(plan["regime_ref"])
    entity = registry.entity(entity_ref)
    regime = registry.regime(regime_ref)
    if regime["input_entity_class_ref"] != entity_ref:
        raise RunPlanV2Error(f"Regime {regime_ref} does not consume EntityClass {entity_ref}")
    if frontier["current_entity_class_ref"] != entity_ref:
        raise RunPlanV2Error("RunPlan EntityClass does not match the authoritative ResearchFrontier")
    if frontier["current_regime_ref"] != regime_ref:
        raise RunPlanV2Error("RunPlan Regime does not match the authoritative ResearchFrontier")

    start_depth = _depth(plan["start"]["regime_depth"], label="start.regime_depth")
    origin = _depth(regime.get("depth_coordinate", {}).get("origin"), label="Regime origin")
    assert start_depth is not None
    if origin is not None and start_depth < origin:
        raise RunPlanV2Error(f"RunPlan start depth {start_depth} precedes Regime origin {origin}")
    max_depth = _depth(plan["development_budget"].get("max_depth"), label="development_budget.max_depth", allow_none=True)
    if max_depth is not None and max_depth < start_depth:
        raise RunPlanV2Error("development_budget.max_depth precedes start depth")

    available = set(frontier.get("available_test_pack_refs", []))
    selected: set[str] = set()
    for pref in plan.get("selected_test_pack_refs", []):
        if pref not in available:
            raise RunPlanV2Error(f"selected TestPack {pref} is not available at the current ResearchFrontier")
        pack = registry.pack(pref)
        selected.update(m["test_ref"] for m in pack["members"])
    for tref in plan.get("selected_test_refs", []):
        registry.test(tref)
        selected.add(tref)
    if not selected:
        raise RunPlanV2Error("RunPlan V2 must select at least one Test or TestPack")

    # Admissibility and capability gates are checked again at plan-freeze time.
    for tref in sorted(selected):
        spec = registry.test(tref)
        if entity_ref not in spec["admissibility"]["entity_class_refs"]:
            raise RunPlanV2Error(f"Test {tref} is not admissible for EntityClass {entity_ref}")
        if regime_ref not in spec["admissibility"]["regime_refs"]:
            raise RunPlanV2Error(f"Test {tref} is not admissible for Regime {regime_ref}")
        missing = set(spec["required_capabilities"]) - set(regime["exposed_capabilities"])
        if missing:
            raise RunPlanV2Error(f"Test {tref} requires Regime capabilities not exposed: {sorted(missing)}")

    ordered = _topological_test_order(selected, registry)
    # Current exact generic-test executor is frozen over the O7..O13 compatibility oracle.
    fully_launchable_depth_max = 13 if entity_ref == O7_ENTITY_REF and regime_ref == O7_REGIME_REF else None
    return RunPlanResolution(
        plan=dict(plan),
        frontier=frontier,
        selected_test_refs=ordered,
        plan_science_sha256=canonical_sha256(dict(plan)),
        frontier_science_sha256=canonical_sha256(frontier),
        start_depth=start_depth,
        max_depth=max_depth,
        fully_launchable_depth_max=fully_launchable_depth_max,
    )


def validate_run_plan_v2(
    plan: Mapping[str, Any],
    *,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> dict[str, Any]:
    try:
        resolved = resolve_run_plan_v2(plan, frontier=frontier, registry=registry)
        errors: list[str] = []
    except Exception as exc:
        resolved = None
        errors = [f"{type(exc).__name__}: {exc}"]
    obj = {
        "schema_id": "IG_RUN_PLAN_V2_VALIDATION_RESULT_V1",
        "status": "PASS" if not errors else "FAIL",
        "plan_id": plan.get("plan_id") if isinstance(plan, Mapping) else None,
        "plan_science_sha256": canonical_sha256(dict(plan)) if isinstance(plan, Mapping) else None,
        "frontier_science_sha256": resolved.frontier_science_sha256 if resolved else None,
        "selected_test_refs": list(resolved.selected_test_refs) if resolved else [],
        "errors": errors,
        "frontier_mutation_authorized": False,
        "plateau_promotion_authorized": False,
        "lift_promotion_authorized": False,
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


def compile_run_plan_v2(
    plan: Mapping[str, Any],
    *,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or ScientificProtocolRegistry()
    r = resolve_run_plan_v2(plan, frontier=frontier, registry=registry)
    max_depth = r.max_depth
    preflight = [
        {
            "task_id": "primitive-semantics-sentinel",
            "binding": "infinity_grid.semantic_sentinel.verify_native_semantics",
            "task_kind": "THEOREM_SAFETY",
            "fail_closed": True,
        },
        {
            "task_id": "theorem-premise-status",
            "binding": "infinity_grid.theorem_registry.verify_all_theorems",
            "task_kind": "THEOREM_SAFETY",
            "fail_closed": True,
        },
        {
            "task_id": "research-frontier-verify",
            "binding": "infinity_grid.research_frontier.verify_research_frontier",
            "task_kind": "SCIENTIFIC_AUTHORITY",
            "fail_closed": True,
        },
    ]
    development = {
        "task_id": "develop-regime",
        "task_kind": "DEVELOPMENT",
        "binding": "infinity_grid.materialized_discovery.MaterializedDiscoverySession",
        "auxiliary_bindings": ["infinity_grid.fixed_grammar_transport.run_fixed_grammar_transport"],
        "start_depth": r.start_depth,
        "through_depth": max_depth,
        "semantic_role": "MATERIALIZE_BOUNDED_ENTITY_INSTANCES_UNDER_EARNED_REGIME_LAW_AND_OBSERVE_STRUCTURE",
        "auxiliary_semantic_role": "THEOREM_TRANSPORT_CONSISTENCY_AND_GROWTH_BOUND_ONLY",
        "science_mutation": False,
    }

    tests = []
    for tref in r.selected_test_refs:
        spec = registry.test(tref)
        tests.append({
            "task_id": "test-" + spec["test_id"].lower().replace("_", "-"),
            "task_kind": "READ_ONLY_TEST",
            "test_ref": tref,
            "input_mode": spec["input_mode"],
            "required_capabilities": list(spec["required_capabilities"]),
            "dependency_test_refs": list(spec.get("dependency_test_refs", [])),
            "default_cadence": spec["default_cadence"],
            "read_only": True,
            "feedback_mode": "RECOGNITION_ONLY",
        })

    launchability = "FULLY_LAUNCHABLE_COMPATIBILITY_RANGE"
    deferred_reason = None
    live_o_regime = plan["entity_class_ref"] == O7_ENTITY_REF and plan["regime_ref"] == O7_REGIME_REF
    if max_depth is None:
        launchability = "COMPILED_UNBOUNDED_STOP_DRIVER_DEFERRED"
        deferred_reason = "Finite-depth live providers are available; unbounded execution still requires an explicit stopping driver."
    elif live_o_regime and max_depth > 13:
        launchability = "FULLY_LAUNCHABLE_GENERIC_TESTS_WITH_EXPLICIT_PROVIDER_SEAM"
        deferred_reason = None
    elif r.fully_launchable_depth_max is None or max_depth > r.fully_launchable_depth_max:
        launchability = "NO_LIVE_CAPABILITY_PROVIDER_FOR_REGIME"
        deferred_reason = "No generic live capability provider is registered for this EntityClass/Regime."

    compiled = {
        "schema_id": "IG_RUN_PLAN_V2_COMPILED_V1",
        "schema_version": "1.0.0",
        "status": "PASS",
        "plan_id": plan["plan_id"],
        "plan_science_sha256": r.plan_science_sha256,
        "frontier_id": r.frontier["frontier_id"],
        "frontier_science_sha256": r.frontier_science_sha256,
        "entity_class_ref": plan["entity_class_ref"],
        "regime_ref": plan["regime_ref"],
        "start_depth": r.start_depth,
        "max_depth": max_depth,
        "resolved_test_refs": list(r.selected_test_refs),
        "preflight_tasks": preflight,
        "development_task": development,
        "test_tasks": tests,
        "post_observation_tasks": [
            {
                "task_id": "maturation-audit",
                "task_kind": "SCIENTIFIC_AUDIT",
                "binding": "infinity_grid.maturation_auditor.audit_maturation",
                "may_issue_plateau_candidate": True,
                "may_issue_plateau_certificate": False,
            },
            {
                "task_id": "plateau-audit",
                "task_kind": "SCIENTIFIC_AUDIT",
                "binding": "infinity_grid.plateau_auditor.audit_plateau",
                "requires_independent_confirmation": True,
                "automatic_promotion": False,
                "may_issue_plateau_certificate": True,
                "may_authorize_lift": False,
            },
            {
                "task_id": "lift-audit",
                "task_kind": "SCIENTIFIC_AUDIT",
                "binding": "infinity_grid.lift_auditor.audit_lift",
                "requires_certified_plateau": True,
                "requires_explicit_lift_candidate": True,
                "automatic_promotion": False,
            },
        ],
        "launchability": launchability,
        "deferred_reason": deferred_reason,
        "compile_target": plan["compile_target"],
        "legacy_execution_plan_preserved": True,
        "research_frontier_mutation": "FORBIDDEN_BY_COMPILER",
        "plateau_automatic_promotion": False,
        "lift_automatic_promotion": False,
        "new_scientific_claim": False,
    }
    compiled["compiled_science_sha256"] = canonical_sha256(compiled)
    return compiled



def _execute_declared_science_preflights() -> None:
    """Execute fail-closed scientific authority checks before live science.

    Both checks are always invoked so a failing earlier check cannot mask a
    second broken authority layer.
    """
    from . import semantic_sentinel as _semantic_sentinel
    from . import theorem_registry as _theorem_registry
    errors=[]
    for name, fn in (
        ("native_semantics", _semantic_sentinel.verify_native_semantics),
        ("theorem_premises", _theorem_registry.verify_all_theorems),
    ):
        try:
            out=fn()
            if isinstance(out, Mapping) and out.get("status") != "PASS":
                errors.append(f"{name}: {out.get('status')}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RunPlanV2Error("scientific preflight failed closed: " + " | ".join(errors))

def execute_o7_compatibility_run_plan(
    plan: Mapping[str, Any],
    *,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> dict[str, Any]:
    """Execute the generic Test layer only in its frozen exact-compatibility range O7..O13.

    This is intentionally a migration executor, not a new scientific run. It reads the
    frozen legacy scanner fixture and proves that a declarative RunPlan selects/executes the
    requested generic Tests deterministically.  It cannot be used beyond O13.
    """
    registry = registry or ScientificProtocolRegistry()
    r = resolve_run_plan_v2(plan, frontier=frontier, registry=registry)

    _execute_declared_science_preflights()
    if plan["entity_class_ref"] != O7_ENTITY_REF or plan["regime_ref"] != O7_REGIME_REF:
        raise RunPlanV2Error("v0.27.2 compatibility executor only supports the frozen O7 compatibility Regime")
    if r.max_depth is None or r.max_depth > 13 or r.start_depth < 7:
        raise RunPlanV2Error("v0.27.2 compatibility executor is deliberately limited to O7..O13")

    legacy = load_frozen_legacy_scanner_fixture()
    source_sha = legacy["science_sha256"]
    selected = set(r.selected_test_refs)
    level_rows: list[dict[str, Any]] = []
    previous_all = None

    for level in range(r.start_depth, r.max_depth + 1):
        summary = legacy["level_summaries"].get(str(level))
        if summary is None:
            raise RunPlanV2Error(f"frozen O7 compatibility oracle has no O{level} level summary")
        genealogy = summary["genealogy"]
        lane_map = {
            O7_LEAF_TEST_REFS[0]: genealogy["grammar"],
            O7_LEAF_TEST_REFS[1]: genealogy["diversity"],
            O7_LEAF_TEST_REFS[2]: genealogy["branching"],
            O7_LEAF_TEST_REFS[3]: genealogy["symmetry"],
            O7_LEAF_TEST_REFS[4]: genealogy["overlap_gluing"],
            O7_LEAF_TEST_REFS[5]: genealogy["lineage"],
            O7_LEAF_TEST_REFS[6]: genealogy["quotient"],
            O7_LEAF_TEST_REFS[7]: genealogy["topology_services"],
            O7_LEAF_TEST_REFS[8]: genealogy["obstruction_relief"],
            O7_LEAF_TEST_REFS[9]: genealogy["representation"],
            O7_LEAF_TEST_REFS[10]: genealogy["representation"],
        }
        # Boundary sentinel depends on four leaf findings; execute all dependencies even if
        # the operator selected only the sentinel, but emit only selected findings.
        needed = set(selected)
        if O7_BOUNDARY_SENTINEL in needed:
            needed.update(registry.test(O7_BOUNDARY_SENTINEL).get("dependency_test_refs", []))
        leaves = {}
        for tref in O7_LEAF_TEST_REFS:
            if tref not in needed:
                continue
            leaves[tref] = run_leaf_test(
                summary=summary,
                test_ref=tref,
                source_science_sha256=source_sha,
                genealogy_value=lane_map[tref],
                registry=registry,
            )
        all_for_sentinel = None
        sentinel = None
        if O7_BOUNDARY_SENTINEL in selected:
            deps = registry.test(O7_BOUNDARY_SENTINEL)["dependency_test_refs"]
            all_for_sentinel = {d: leaves[d] for d in deps}
            prev_deps = None if previous_all is None else {d: previous_all[d] for d in deps}
            sentinel = run_regime_boundary_sentinel(
                current=all_for_sentinel,
                previous=prev_deps,
                source_science_sha256=source_sha,
                level=level,
                registry=registry,
            )
        emitted = {t: leaves[t].finding for t in sorted(selected & set(leaves))}
        if sentinel is not None:
            emitted[O7_BOUNDARY_SENTINEL] = sentinel.finding
        level_rows.append({
            "regime_depth": level,
            "selected_findings": emitted,
            "selected_finding_sha256": canonical_sha256(emitted),
            "source_science_sha256": source_sha,
        })
        # Cache all four boundary dependencies across depths when available.
        if all_for_sentinel is not None:
            previous_all = all_for_sentinel

    result = {
        "schema_id": "IG_RUN_PLAN_V2_COMPAT_EXECUTION_RESULT_V1",
        "schema_version": "1.0.0",
        "status": "PASS",
        "classification": "RUNPLAN_V2_GENERIC_PROTOCOL_COMPATIBILITY_EXECUTION_ONLY",
        "plan_id": plan["plan_id"],
        "plan_science_sha256": r.plan_science_sha256,
        "frontier_science_sha256": r.frontier_science_sha256,
        "entity_class_ref": plan["entity_class_ref"],
        "regime_ref": plan["regime_ref"],
        "depths_executed": [x["regime_depth"] for x in level_rows],
        "resolved_test_refs": list(r.selected_test_refs),
        "levels": level_rows,
        "legacy_scanner_source_modified": False,
        "production_science_routed_through_runplan_v2": False,
        "frontier_mutated": False,
        "plateau_certified": False,
        "lift_certified": False,
        "new_scientific_claim": False,
    }
    result["science_sha256"] = canonical_sha256(result)
    return result


def _materialized_o_regime_source_and_snapshot(
    discovery: MaterializedDiscoverySession,
    development: Path,
    level: int,
) -> tuple[dict[str, Any], Any]:
    """Build the primary O14+ materialized discovery snapshot plus auxiliary transport audit."""
    run_fixed_grammar_transport(
        development,
        through=int(level),
        phase8_seed=None,
        reset=not (development / "CURRENT_FRONTIER.json").exists(),
    )
    transport_delta = load_fixed_grammar_delta(development, int(level))
    row = discovery.advance_to(int(level))
    transport_audit = theorem_transport_consistency(row.summary, row.materialization, transport_delta)
    if transport_audit.get("status") != "PASS":
        failed = [x.get("check_id") for x in transport_audit.get("checks", []) if x.get("status") != "PASS"]
        raise RunPlanV2Error(
            f"O{int(level)} theorem-transport auxiliary consistency audit failed closed: {failed}"
        )
    snapshot = from_materialized_discovery(
        row.summary,
        row.materialization,
        transport_audit=transport_audit,
        entity_instance_ref=f"live:O{int(level)}:materialized-discovery-panel",
    )
    source_input = {
        "schema_id": "IG_O_REGIME_MATERIALIZED_DISCOVERY_SOURCE_PACKET_V1",
        "schema_version": "1.0.0",
        "level": int(level),
        "primary_discovery_authority": "MATERIALIZED_FINITE_STRUCTURAL_PANEL",
        "materialized_scanner_summary": row.summary,
        "materialized_discovery_evidence": row.materialization,
        "theorem_transport_delta": transport_delta,
        "theorem_transport_auxiliary_audit": transport_audit,
        "theorem_transport_discovery_authority": False,
    }
    # RunOperations reserves science_sha256 as the pointer to the Test record's source
    # authority, not as a self-hash of SOURCE_INPUT.json. Keep both identities explicitly.
    source_input["source_packet_content_sha256"] = canonical_sha256(source_input)
    source_input["science_sha256"] = snapshot.source_science_sha256
    return source_input, snapshot


def execute_o7_live_run_plan(
    plan: Mapping[str, Any],
    *,
    output: str | Path,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
) -> dict[str, Any]:
    """Execute a finite O7-regime RunPlan through generic capability providers.

    O7..O13 use the frozen bounded scanner summaries only as compatibility source data.
    O14+ materialize exact finite synthetic fixed-lift panels from the certified O7 replay root.
    Fixed-grammar theorem transport is retained only as an auxiliary consistency/growth-bound
    lane. The O13->O14 provider seam is recorded and compared only through the Phase-4 O13 overlap calibration certificate.
    """
    registry = registry or ScientificProtocolRegistry()
    r = resolve_run_plan_v2(plan, frontier=frontier, registry=registry)
    _execute_declared_science_preflights()
    if plan["entity_class_ref"] != O7_ENTITY_REF or plan["regime_ref"] != O7_REGIME_REF:
        raise RunPlanV2Error("v0.28.1 live executor currently supports the O7 organizational Regime")
    if r.max_depth is None:
        raise RunPlanV2Error("live RunPlan execution currently requires a finite max_depth")
    output = Path(output)
    development = output / "development"

    legacy = load_frozen_legacy_scanner_fixture()
    previous = None
    level_records = []
    public_levels = []
    discovery_context = MaterializedDiscoverySession() if r.max_depth > 13 else nullcontext(None)
    with discovery_context as discovery:
        seam_calibration = None
        seam_baseline = None
        if discovery is not None and r.start_depth <= 13 and r.max_depth >= 14:
            seam_calibration, seam_baseline = calibrate_o13_o14_provider_seam(discovery=discovery, registry=registry)
            output.mkdir(parents=True, exist_ok=True)
            (output / "O13_O14_PROVIDER_SEAM_CALIBRATION.json").write_text(
                json.dumps(seam_calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        for level in range(r.start_depth, r.max_depth + 1):
            if level <= 13:
                summary = legacy["level_summaries"].get(str(level))
                if summary is None:
                    raise RunPlanV2Error(f"bounded compatibility source has no O{level} summary")
                snapshot = from_scanner_level_summary(
                    summary,
                    source_science_sha256=legacy["science_sha256"],
                    entity_instance_ref=f"compat-source:O{level}:scanner-panel",
                )
            else:
                if discovery is None:
                    raise RunPlanV2Error("materialized discovery session unavailable for O14+")
                _, snapshot = _materialized_o_regime_source_and_snapshot(discovery, development, level)
            record = execute_snapshot_tests(
                snapshot,
                selected_test_refs=r.selected_test_refs,
                previous=previous,
                registry=registry,
                seam_baseline=seam_baseline if level == 14 else None,
                seam_calibration=seam_calibration if level == 14 else None,
            )
            level_records.append(record)
            public_levels.append({
                "regime_depth": level,
                "provider_ref": record.provider_ref,
                "provider_kind": record.provider_kind,
                "provider_seam": record.provider_seam,
                "provider_seam_calibrated": record.provider_seam_calibrated,
                "provider_seam_calibration_sha256": record.provider_seam_calibration_sha256,
                "stabilization_signature_sha256": record.stabilization_signature_sha256,
                "complexity_projection": dict(record.complexity_projection),
                "complexity_growth_from_previous": record.complexity_growth_from_previous,
                "findings": {k: v.finding for k, v in sorted(record.executions.items()) if k in set(r.selected_test_refs)},
            })
            previous = record
            maturation_so_far = audit_maturation(level_records, test_pack_ref=O7_PACK_REF, plateau_window=4, registry=registry)
            if maturation_so_far.get("review_gate", {}).get("required"):
                result = {
                    "schema_id": "IG_RUN_PLAN_V2_LIVE_EXECUTION_RESULT_V1",
                    "schema_version": "1.0.0",
                    "status": "REVIEW_REQUIRED",
                    "classification": "RUNPLAN_V2_GENERIC_LIVE_CAPABILITY_EXECUTION_REVIEW_STOP",
                    "plan_id": plan["plan_id"],
                    "plan_science_sha256": r.plan_science_sha256,
                    "frontier_science_sha256": r.frontier_science_sha256,
                    "entity_class_ref": plan["entity_class_ref"],
                    "regime_ref": plan["regime_ref"],
                    "depths_executed": [x.depth for x in level_records],
                    "provider_seam_depths": [x.depth for x in level_records if x.provider_seam],
                    "provider_seam_calibration": seam_calibration,
                    "levels": public_levels,
                    "maturation_audit": maturation_so_far,
                    "review_gate": maturation_so_far.get("review_gate"),
                    "plateau_audit": None,
                    "lift_audit": None,
                    "historical_scanner_used_after_O13": False,
                    "materialized_discovery_used_after_O13": any(x.depth > 13 for x in level_records),
                    "fixed_grammar_transport_used_after_O13": any(x.depth > 13 for x in level_records),
                    "fixed_grammar_transport_role_after_O13": "AUXILIARY_CONSISTENCY_AND_GROWTH_BOUND_ONLY",
                    "frontier_mutated": False,
                    "plateau_certified": False,
                    "lift_certified": False,
                    "new_scientific_claim": False,
                    "scientific_result_finalized": False,
                }
                result["science_sha256"] = canonical_sha256(result)
                output.mkdir(parents=True, exist_ok=True)
                (output / "RUNPLAN_V2_LIVE_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return result

    maturation = audit_maturation(level_records, test_pack_ref=O7_PACK_REF, plateau_window=4, registry=registry)
    # A live RunPlan may recognize a candidate, but it supplies no independent confirmation
    # authority and no LiftCandidate. Therefore these separate auditors must remain non-promoting.
    plateau_audit = audit_plateau(
        level_records,
        maturation_audit=maturation,
        candidate=maturation.get("plateau_candidate"),
        independent_evidence=(),
        test_pack_ref=O7_PACK_REF,
        plateau_window=4,
        registry=registry,
    )
    lift_audit = audit_lift(
        None,
        plateau_certificates=tuple(
            x for x in [plateau_audit.get("plateau_certificate")] if x is not None
        ),
        audit_evidence=(),
        certification_input=None,
        registry=registry,
    )
    result = {
        "schema_id": "IG_RUN_PLAN_V2_LIVE_EXECUTION_RESULT_V1",
        "schema_version": "1.0.0",
        "status": "PASS",
        "classification": "RUNPLAN_V2_GENERIC_LIVE_CAPABILITY_EXECUTION",
        "plan_id": plan["plan_id"],
        "plan_science_sha256": r.plan_science_sha256,
        "frontier_science_sha256": r.frontier_science_sha256,
        "entity_class_ref": plan["entity_class_ref"],
        "regime_ref": plan["regime_ref"],
        "depths_executed": list(range(r.start_depth, r.max_depth + 1)),
        "provider_seam_depths": [x["regime_depth"] for x in public_levels if x["provider_seam"]],
        "provider_seam_calibration": seam_calibration,
        "levels": public_levels,
        "maturation_audit": maturation,
        "review_gate": maturation.get("review_gate"),
        "plateau_audit": plateau_audit,
        "lift_audit": lift_audit,
        "historical_scanner_used_after_O13": False,
        "materialized_discovery_used_after_O13": r.max_depth > 13,
        "fixed_grammar_transport_used_after_O13": r.max_depth > 13,
        "fixed_grammar_transport_role_after_O13": "AUXILIARY_CONSISTENCY_AND_GROWTH_BOUND_ONLY",
        "frontier_mutated": False,
        "plateau_certified": plateau_audit.get("decision") == "CERTIFIED_IN_DECLARED_SCOPE",
        "lift_certified": lift_audit.get("decision") == "CERTIFIED_IN_DECLARED_SCOPE",
        "new_scientific_claim": bool(plateau_audit.get("new_scientific_claim") or lift_audit.get("new_scientific_claim")),
    }
    result["science_sha256"] = canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "RUNPLAN_V2_LIVE_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _execute_o7_resumable_maturation_run_plan_locked(
    plan: Mapping[str, Any],
    *,
    output: str | Path,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
    stop_after_depth: int | None = None,
    acknowledge_review: bool = False,
    _controller_lock_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Crash-safe finite O7-regime maturation execution with per-depth checkpoints.

    This is the operationally hardened successor to ``execute_o7_live_run_plan``.  It preserves
    identical scientific Test/Maturation/Plateau/Lift semantics while adding only status,
    timings, checkpoint and strict-resume behavior.  Committed checkpoints are authoritative;
    uncommitted transport/test work may be replayed after a crash.
    """
    import time as _time
    from . import __version__ as _decoder_version
    from .run_operations import RunOperations, level_record_from_json

    registry = registry or ScientificProtocolRegistry()
    r = resolve_run_plan_v2(plan, frontier=frontier, registry=registry)
    _execute_declared_science_preflights()
    if plan["entity_class_ref"] != O7_ENTITY_REF or plan["regime_ref"] != O7_REGIME_REF:
        raise RunPlanV2Error("resumable maturation executor currently supports the O7 organizational Regime")
    if r.max_depth is None:
        raise RunPlanV2Error("resumable maturation execution requires a finite max_depth")

    output = Path(output)
    ops = RunOperations(
        output,
        run_id=str(plan["plan_id"]),
        plan=plan,
        plan_science_sha256=r.plan_science_sha256,
        frontier_science_sha256=r.frontier_science_sha256,
        entity_class_ref=str(plan["entity_class_ref"]),
        regime_ref=str(plan["regime_ref"]),
        selected_test_refs=r.selected_test_refs,
        start_depth=r.start_depth,
        max_depth=r.max_depth,
    )
    ops.bind_controller_lock(_controller_lock_payload)
    ops.initialize_or_verify()
    pending_external = ops.pending_external_ack_depths()
    if pending_external:
        pending_depth = pending_external[0]
        status = ops.write_status(
            state="EXTERNAL_ACK_REQUIRED",
            current_depth=(pending_depth + 1 if pending_depth < r.max_depth else None),
        )
        return {
            "schema_id": "IG_RESUMABLE_MATURATION_OPERATION_RESULT_V1",
            "schema_version": "1.0.0",
            "status": "EXTERNAL_ACK_REQUIRED",
            "run_id": plan["plan_id"],
            "decoder_version": _decoder_version,
            "completed_depths": ops.valid_checkpoint_depths(),
            "pending_external_ack_depth": pending_depth,
            "external_request_path": str(ops.external_request_path(pending_depth)),
            "run_status": status,
            "scientific_result_finalized": False,
            "frontier_mutated": False,
        }
    segment_records = ops.load_committed_records()
    segment_completed = [x.depth for x in segment_records]
    continuation_seed = _load_continuation_seed(output, plan, start_depth=r.start_depth)
    history_records = []
    seed_prior_audit = None
    if continuation_seed is not None:
        history_records = [level_record_from_json(x) for x in continuation_seed["history_level_records"]]
        depths = [x.depth for x in history_records]
        expected_depths = list(range(depths[0], int(continuation_seed["prior_last_depth"]) + 1))
        if depths != expected_depths or depths[-1] != r.start_depth - 1:
            raise RunPlanV2Error("continuation seed historical LevelTestRecords are non-contiguous")
        seed_prior_audit = continuation_seed.get("prior_maturation_audit")
    records = history_records + segment_records
    previous = records[-1] if records else None
    next_depth = segment_completed[-1] + 1 if segment_completed else r.start_depth
    if stop_after_depth is not None and next_depth > int(stop_after_depth):
        status = ops.write_status(state="PAUSED", current_depth=next_depth if next_depth <= r.max_depth else None)
        return {
            "schema_id": "IG_RESUMABLE_MATURATION_OPERATION_RESULT_V1",
            "schema_version": "1.0.0",
            "status": "PAUSED",
            "run_id": plan["plan_id"],
            "decoder_version": _decoder_version,
            "completed_depths": segment_completed,
            "next_depth": next_depth if next_depth <= r.max_depth else None,
            "run_status": status,
            "scientific_result_finalized": False,
            "frontier_mutated": False,
        }
    prior_audit = ops.latest_maturation_audit()
    if prior_audit is None and seed_prior_audit is not None:
        prior_audit = seed_prior_audit
    prior_gate = None if prior_audit is None else prior_audit.get("review_gate")
    if isinstance(prior_gate, Mapping) and prior_gate.get("required"):
        if not acknowledge_review:
            status = ops.write_status(
                state="REVIEW_REQUIRED",
                current_depth=next_depth if next_depth <= r.max_depth else None,
                review_gate=prior_gate,
            )
            return {
                "schema_id": "IG_RESUMABLE_MATURATION_OPERATION_RESULT_V1",
                "schema_version": "1.0.0",
                "status": "REVIEW_REQUIRED",
                "run_id": plan["plan_id"],
                "decoder_version": _decoder_version,
                "completed_depths": segment_completed,
                "continuation_history_through_depth": (history_records[-1].depth if history_records else None),
                "next_depth": next_depth if next_depth <= r.max_depth else None,
                "review_gate": prior_gate,
                "run_status": status,
                "scientific_result_finalized": False,
                "frontier_mutated": False,
            }
        ops.record_review_acknowledgement(prior_gate)
    ops.write_status(state="RUNNING", current_depth=next_depth if next_depth <= r.max_depth else None)

    development = output / "development"
    legacy = load_frozen_legacy_scanner_fixture()
    operational = dict(plan.get("operational_integrity") or {})
    budget = dict(operational.get("resource_budget") or {})
    max_workers = budget.get("max_effective_workers")
    discovery_policy = None
    if max_workers is not None:
        discovery_policy = ExecutionPolicy(
            backend="AUTO",
            requested_workers=int(max_workers),
            scheduler="COST_WEIGHTED_SHARDS",
            owner="o-regime-maturation-certified",
        )
    if next_depth <= r.max_depth and r.max_depth > 13:
        # Preserve the historical zero-argument construction surface when no
        # certified worker ceiling is configured.  Several frozen regression
        # harnesses substitute a minimal discovery test double with that exact
        # interface; operational hardening must not change scientific adapters.
        discovery = (
            MaterializedDiscoverySession(execution_policy=discovery_policy)
            if discovery_policy is not None
            else MaterializedDiscoverySession()
        )
    else:
        discovery = None
    seam_calibration = None
    seam_baseline = None
    try:
        if discovery is not None and next_depth <= 14 and r.start_depth <= 13 and r.max_depth >= 14:
            seam_calibration, seam_baseline = calibrate_o13_o14_provider_seam(discovery=discovery, registry=registry)
            write_json_atomic(output / "O13_O14_PROVIDER_SEAM_CALIBRATION.json", seam_calibration)
        for level in range(next_depth, r.max_depth + 1):
            ops.check_resource_budget(depth=level)
            ops.begin_depth_attempt(level)
            depth_t0 = _time.perf_counter()
            depth_started_utc = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
            ops.write_status(state="RUNNING", current_depth=level)
            if level <= 13:
                summary = legacy["level_summaries"].get(str(level))
                if summary is None:
                    raise RunPlanV2Error(f"bounded compatibility source has no O{level} summary")
                source_input = dict(summary)
                snapshot = from_scanner_level_summary(
                    summary,
                    source_science_sha256=legacy["science_sha256"],
                    entity_instance_ref=f"compat-source:O{level}:scanner-panel",
                )
            else:
                # The materialized finite panel is primary scientific evidence. The existing
                # theorem transport is still advanced exactly once, but only as an auxiliary
                # consistency / conservative-growth-bound lane inside the source packet.
                if discovery is None:
                    raise RunPlanV2Error("materialized discovery session unavailable for O14+")
                source_input, snapshot = _materialized_o_regime_source_and_snapshot(
                    discovery, development, level
                )

            snapshot_json = {
                "provider_ref": snapshot.provider_ref,
                "provider_kind": snapshot.provider_kind,
                "entity_instance_ref": snapshot.entity_instance_ref,
                "entity_class_ref": snapshot.entity_class_ref,
                "regime_ref": snapshot.regime_ref,
                "regime_depth": snapshot.regime_depth,
                "source_science_sha256": snapshot.source_science_sha256,
                "payloads": dict(snapshot.payloads),
                "exactness": snapshot.exactness,
                "snapshot_science_sha256": snapshot.science_sha256,
            }
            callback = ops.progress_callback(depth=level, provider_ref=snapshot.provider_ref)
            record = execute_snapshot_tests(
                snapshot,
                selected_test_refs=r.selected_test_refs,
                previous=previous,
                registry=registry,
                progress_callback=callback,
                seam_baseline=seam_baseline if level == 14 else None,
                seam_calibration=seam_calibration if level == 14 else None,
            )
            candidate_records = records + [record]
            maturation = audit_maturation(candidate_records, test_pack_ref=O7_PACK_REF, plateau_window=4, registry=registry)
            depth_elapsed = _time.perf_counter() - depth_t0
            ops.commit_depth(
                depth=level,
                source_input=source_input,
                capability_snapshot=snapshot_json,
                level_record=record,
                maturation_audit=maturation,
                depth_started_utc=depth_started_utc,
                depth_elapsed_seconds=depth_elapsed,
            )
            ops.finish_depth_attempt(level, "COMMITTED")
            records.append(record)
            previous = record
            if ops.external_ack_required:
                try:
                    ops.verify_external_ack(level)
                except Exception as ack_exc:
                    from .run_operations import ExternalDurabilityAckRequired
                    if isinstance(ack_exc, ExternalDurabilityAckRequired):
                        status = ops.write_status(
                            state="EXTERNAL_ACK_REQUIRED",
                            current_depth=(level + 1 if level < r.max_depth else None),
                        )
                        return {
                            "schema_id": "IG_RESUMABLE_MATURATION_OPERATION_RESULT_V1",
                            "schema_version": "1.0.0",
                            "status": "EXTERNAL_ACK_REQUIRED",
                            "run_id": plan["plan_id"],
                            "decoder_version": _decoder_version,
                            "completed_depths": [x.depth for x in records],
                            "pending_external_ack_depth": level,
                            "external_request_path": str(ops.external_request_path(level)),
                            "run_status": status,
                            "scientific_result_finalized": False,
                            "frontier_mutated": False,
                        }
                    raise
            review_gate = maturation.get("review_gate", {})
            if review_gate.get("required"):
                next_after_review = level + 1 if level < r.max_depth else None
                status = ops.write_status(
                    state="REVIEW_REQUIRED",
                    current_depth=next_after_review,
                    review_gate=review_gate,
                )
                return {
                    "schema_id": "IG_RESUMABLE_MATURATION_OPERATION_RESULT_V1",
                    "schema_version": "1.0.0",
                    "status": "REVIEW_REQUIRED",
                    "run_id": plan["plan_id"],
                    "decoder_version": _decoder_version,
                    "completed_depths": [x.depth for x in records],
                    "next_depth": next_after_review,
                    "review_gate": review_gate,
                    "provider_seam_calibration": seam_calibration,
                    "maturation_audit": maturation,
                    "run_status": status,
                    "scientific_result_finalized": False,
                    "frontier_mutated": False,
                }
            if stop_after_depth is not None and level >= int(stop_after_depth) and level < r.max_depth:
                status = ops.write_status(state="PAUSED", current_depth=level + 1)
                return {
                    "schema_id": "IG_RESUMABLE_MATURATION_OPERATION_RESULT_V1",
                    "schema_version": "1.0.0",
                    "status": "PAUSED",
                    "run_id": plan["plan_id"],
                    "decoder_version": _decoder_version,
                    "completed_depths": [x.depth for x in records],
                    "next_depth": level + 1,
                    "run_status": status,
                    "scientific_result_finalized": False,
                    "frontier_mutated": False,
                }
            ops.write_status(state="RUNNING", current_depth=level + 1 if level < r.max_depth else None)

        maturation = audit_maturation(records, test_pack_ref=O7_PACK_REF, plateau_window=4, registry=registry)
        plateau_audit = audit_plateau(
            records,
            maturation_audit=maturation,
            candidate=maturation.get("plateau_candidate"),
            independent_evidence=(),
            test_pack_ref=O7_PACK_REF,
            plateau_window=4,
            registry=registry,
        )
        lift_audit = audit_lift(
            None,
            plateau_certificates=tuple(x for x in [plateau_audit.get("plateau_certificate")] if x is not None),
            audit_evidence=(),
            certification_input=None,
            registry=registry,
        )
        public_levels = []
        for record in records:
            public_levels.append({
                "regime_depth": record.depth,
                "provider_ref": record.provider_ref,
                "provider_kind": record.provider_kind,
                "provider_seam": record.provider_seam,
                "provider_seam_calibrated": record.provider_seam_calibrated,
                "provider_seam_calibration_sha256": record.provider_seam_calibration_sha256,
                "stabilization_signature_sha256": record.stabilization_signature_sha256,
                "complexity_projection": dict(record.complexity_projection),
                "complexity_growth_from_previous": record.complexity_growth_from_previous,
                "findings": {k: v.finding for k, v in sorted(record.executions.items()) if k in set(r.selected_test_refs)},
            })
        result = {
            "schema_id": "IG_RUN_PLAN_V2_LIVE_EXECUTION_RESULT_V1",
            "schema_version": "1.0.0",
            "status": "PASS",
            "classification": "RUNPLAN_V2_GENERIC_LIVE_CAPABILITY_EXECUTION_RESUMABLE",
            "plan_id": plan["plan_id"],
            "plan_science_sha256": r.plan_science_sha256,
            "frontier_science_sha256": r.frontier_science_sha256,
            "entity_class_ref": plan["entity_class_ref"],
            "regime_ref": plan["regime_ref"],
            "depths_executed": [x.depth for x in records],
            "provider_seam_depths": [x.depth for x in records if x.provider_seam],
            "levels": public_levels,
            "maturation_audit": maturation,
            "review_gate": maturation.get("review_gate"),
            "plateau_audit": plateau_audit,
            "lift_audit": lift_audit,
            "run_operations": {
                "status_file": "RUN_STATUS.json",
                "timing_ledger": "RUN_TIMINGS.csv",
                "checkpoint_root": "checkpoints",
                "checkpoint_count": len(records),
                "strict_resume": True,
            },
            "historical_scanner_used_after_O13": False,
            "materialized_discovery_used_after_O13": r.max_depth > 13,
            "fixed_grammar_transport_used_after_O13": r.max_depth > 13,
            "fixed_grammar_transport_role_after_O13": "AUXILIARY_CONSISTENCY_AND_GROWTH_BOUND_ONLY",
            "frontier_mutated": False,
            "plateau_certified": plateau_audit.get("decision") == "CERTIFIED_IN_DECLARED_SCOPE",
            "lift_certified": lift_audit.get("decision") == "CERTIFIED_IN_DECLARED_SCOPE",
            "new_scientific_claim": bool(plateau_audit.get("new_scientific_claim") or lift_audit.get("new_scientific_claim")),
        }
        result["science_sha256"] = canonical_sha256(result)
        write_json_atomic(output / "RUNPLAN_V2_LIVE_RESULT.json", result)
        ops.write_status(state="COMPLETE", current_depth=None)
        return result
    except Exception as exc:
        try:
            ops.abort_active_attempts()
            ops.write_status(
                state="FAILED_SAFE",
                current_depth=(records[-1].depth + 1 if records else r.start_depth),
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
        finally:
            raise
    finally:
        if discovery is not None:
            discovery.close()

def _load_continuation_seed(output: Path, plan: Mapping[str, Any], *, start_depth: int):
    """Load a content-addressed historical maturation prefix for a new-source continuation.

    The seed is operational provenance, not a rewritten checkpoint chain.  Its SHA is bound
    through start.entity_or_state_ref, which is itself covered by the RunPlan science hash.
    Historical LevelTestRecords are used only as read-only longitudinal context; newly committed
    checkpoints begin at start_depth under the new run identity.
    """
    ref = str(plan.get("start", {}).get("entity_or_state_ref", ""))
    prefix = "CONTINUATION_SEED_SHA256:"
    if not ref.startswith(prefix):
        return None
    expected = ref[len(prefix):]
    if len(expected) != 64:
        raise RunPlanV2Error("continuation seed reference must carry a 64-hex SHA-256")
    seed_path = output / "CONTINUATION_SEED.json"
    if not seed_path.is_file():
        raise RunPlanV2Error(f"continuation seed file missing: {seed_path}")
    seed = _load_json(seed_path)
    raw = dict(seed); got = raw.pop("seed_sha256", None)
    calc = canonical_sha256(raw)
    if got != calc or calc != expected:
        raise RunPlanV2Error("continuation seed content hash mismatch")
    if seed.get("schema_id") != "IG_MATURATION_CONTINUATION_SEED_V1":
        raise RunPlanV2Error("unsupported continuation seed schema")
    if int(seed.get("resume_depth", -1)) != int(start_depth):
        raise RunPlanV2Error("continuation seed resume depth does not match RunPlan start depth")
    if int(seed.get("prior_last_depth", -1)) != int(start_depth) - 1:
        raise RunPlanV2Error("continuation seed prior depth is not start_depth-1")
    rows = seed.get("history_level_records")
    if not isinstance(rows, list) or not rows:
        raise RunPlanV2Error("continuation seed has no historical LevelTestRecords")
    return seed


def execute_o7_resumable_maturation_run_plan(
    plan: Mapping[str, Any],
    *,
    output: str | Path,
    frontier: Mapping[str, Any] | None = None,
    registry: ScientificProtocolRegistry | None = None,
    stop_after_depth: int | None = None,
    acknowledge_review: bool = False,
) -> dict[str, Any]:
    """Single-writer wrapper for crash-safe O7-regime maturation execution."""
    from .checkpoints import RunLock, recover_stale_lock

    output_path = Path(output)
    # Same-host dead-process locks are recovered with an auditable record.  A live owner or
    # unresolved foreign-host owner still fails closed with DuplicateController.
    recover_stale_lock(output_path)
    with RunLock(output_path) as lock:
        return _execute_o7_resumable_maturation_run_plan_locked(
            plan, output=output_path, frontier=frontier, registry=registry,
            stop_after_depth=stop_after_depth, acknowledge_review=acknowledge_review,
            _controller_lock_payload=lock.payload,
        )

