from __future__ import annotations

"""Controller-only G6:S3R recovery handler.

This module contains scientific interpretation only.  It does not create workers,
write checkpoints, reduce shards, or own resume state.  Historical S3R L1 shards
are migrated by StageScienceRuntime, which is Decoder-owned execution machinery.
"""

from typing import Any, Mapping
import hashlib

from .canon import canonical_sha256
from .execution import TaskSpec
from .g6_s1_repair import _basis
from .adapters.g4_accepted import G4AcceptedAdapter

from .v05_chain import ChainExecutionResult
from .v05_stage_runtime import StageRuntimeError


def g6_s3r_l1_recovery_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    params = dict(stage["execution"].get("parameters") or {})
    legacy_root = str(params["legacy_l1_shard_root"])
    expected_count = int(params["expected_task_count"])
    part = runtime.import_legacy_partition_shards(
        phase_id=str(params.get("phase_id", "L1_S2R_CONTEXT_LEGACY_RECOVERY")),
        shard_root=legacy_root,
        expected_schema_id=str(params.get("legacy_schema_id", "IG_G6_S3R_OBSERVER_SHARD_V1")),
        expected_level=str(params.get("legacy_level", "L1_S2R_CONTEXT")),
    )
    summary = part.summary
    if int(summary["task_count"]) != expected_count:
        raise StageRuntimeError(
            f"G6:S3R legacy L1 coverage mismatch {summary['task_count']} != {expected_count}"
        )
    if int(summary["class_count"]) == expected_count and int(summary["multi_class_count"]) == 0:
        outcome = "PASS_HIGHER_DISCRETE_KERNEL_L1"
    else:
        # Conservative engineering migration stop.  This does not alter the frozen
        # S3R rule that real L1 collisions require L2; it prevents the recovery
        # handler from inventing a private L2 execution path.
        outcome = "L1_COLLISIONS_REQUIRE_CONTROLLER_ONLY_L2"
    return ChainExecutionResult(result={
        "schema_id": "IG_G6_S3R_CONTROLLER_ONLY_L1_RECOVERY_RESULT_V1",
        "outcome": outcome,
        "original_question_sha256": str(stage["question_sha256"]),
        "legacy_l1_observer_level": "L1_S2R_CONTEXT",
        "partition": summary,
        "scientific_observer_unchanged": True,
        "legacy_science_rerun": False,
        "execution_architecture_changed_only": True,
        "promotion_effect": "NONE",
    })


def register_g6_s3_controller_handlers(controller) -> None:
    controller.register_stage_handler("g6.s3r.l1.legacy_recovery", g6_s3r_l1_recovery_handler)
    controller.register_stage_handler("g6.s3r.l1.recompute", g6_s3r_l1_recompute_handler)



def _identity_set_sha256(ids) -> str:
    return canonical_sha256(sorted(str(x) for x in ids))


def g6_s3r_l1_recompute_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    """Rebuild frozen S3R state space and L1 through Decoder-owned common runtime.

    This is the recovery route when historical L1 evidence bytes are unavailable.  The
    scientific question, generator axes, probe, operator and position remain frozen.  All
    multiprocessing, caching, durable state storage, resume, reduction and telemetry are
    supplied by StageScienceRuntime.
    """
    params = dict(stage["execution"].get("parameters") or {})
    expected_s1_count = int(params["expected_s1_count"])
    expected_s1_identity_set_sha = str(params["expected_s1_identity_set_sha256"])
    expected_higher_count = int(params["expected_higher_count"])
    expected_higher_identity_set_sha = str(params["expected_higher_identity_set_sha256"])
    expected_axis_b_parent_sha = str(params["expected_axis_b_selected_parent_ids_sha256"])
    expected_partition_binding = str(params["expected_l1_partition_bindings_sha256"])
    workers = int(params.get("workers", runtime.default_workers))

    basis = _basis(); refs = sorted(basis); ops = tuple(G4AcceptedAdapter().operator_basis())
    s1_tasks = []
    idx = 0
    for left_ref in refs:
        for right_ref in refs:
            for op in ops:
                payload = {"left_ref": left_ref, "right_ref": right_ref, "operator": list(op)}
                s1_tasks.append(TaskSpec(
                    task_id=f"S1{idx:04d}", task_kind="G6_S1_UNIVERSE_REBUILD",
                    binding_sha256=canonical_sha256({"question_sha256": str(stage["question_sha256"]), "payload": payload}),
                    payload=payload, cost_weight=1.0,
                )); idx += 1
    s1_gen = runtime.run_content_indexed_generation(
        phase_id=str(params.get("s1_phase_id", "S1_AUTHORITY_REBUILD")),
        tasks=s1_tasks,
        evaluator_ref="infinity_grid.g6_controller_evaluators:g6_s1_universe_generation_evaluator",
        requested_workers=workers,
        max_tasks=len(s1_tasks),
        max_generated_occurrences=int(params.get("max_s1_generated_occurrences", 100000)),
    )
    s1_rows = list(runtime.iter_generated_states(phase_id=str(params.get("s1_phase_id", "S1_AUTHORITY_REBUILD"))))
    s1_ids = sorted(str(r["identity_sha256"]) for r in s1_rows)
    if len(s1_rows) != expected_s1_count:
        raise StageRuntimeError(f"G6:S3R S1 exact-state count mismatch {len(s1_rows)} != {expected_s1_count}")
    observed_s1_set_sha = _identity_set_sha256(s1_ids)
    if observed_s1_set_sha != expected_s1_identity_set_sha:
        raise StageRuntimeError("G6:S3R S1 exact identity-set binding mismatch")
    if any(str(r["state_token"]) != f"{r['identity_sha256']}:0" for r in s1_rows):
        raise StageRuntimeError("G6:S3R S1 identity digest collision encountered; historical state-id route cannot proceed")
    s1_by_id = {str(r["identity_sha256"]): r["state"] for r in s1_rows}

    higher_tasks = []; idx = 0
    for a in refs:
        for b in refs:
            for c in refs:
                payload = {"axis": "A", "triple": [a, b, c]}
                higher_tasks.append(TaskSpec(
                    task_id=f"A{idx:03d}", task_kind="G6_S3R_AXIS_A",
                    binding_sha256=canonical_sha256({"question_sha256": str(stage["question_sha256"]), "payload": payload}),
                    payload=payload, cost_weight=1.0,
                )); idx += 1
    axis_b_n = int(params.get("axis_b_sample_size", 128))
    selected = sorted(s1_by_id, reverse=True)[:axis_b_n]
    if canonical_sha256(selected) != expected_axis_b_parent_sha:
        raise StageRuntimeError("G6:S3R axis-B selected-parent binding mismatch")
    for i, sid in enumerate(selected):
        payload = {"axis": "B", "parent_state_id": sid, "state_tree": s1_by_id[sid]}
        higher_tasks.append(TaskSpec(
            task_id=f"B{i:03d}", task_kind="G6_S3R_AXIS_B",
            binding_sha256=canonical_sha256({"question_sha256": str(stage["question_sha256"]), "payload": payload}),
            payload=payload, cost_weight=max(1.0, float(s1_by_id[sid]["n"])),
        ))
    higher_phase = str(params.get("higher_phase_id", "S3R_HIGHER_REBUILD"))
    higher_gen = runtime.run_content_indexed_generation(
        phase_id=higher_phase,
        tasks=higher_tasks,
        evaluator_ref="infinity_grid.g6_controller_evaluators:g6_s3_higher_generation_evaluator",
        requested_workers=workers,
        max_tasks=len(higher_tasks),
        max_generated_occurrences=int(params.get("max_higher_generated_occurrences", 100000)),
    )
    higher_rows = list(runtime.iter_generated_states(phase_id=higher_phase))
    higher_ids = sorted(str(r["identity_sha256"]) for r in higher_rows)
    if len(higher_rows) != expected_higher_count:
        raise StageRuntimeError(f"G6:S3R higher exact-state count mismatch {len(higher_rows)} != {expected_higher_count}")
    observed_higher_set_sha = _identity_set_sha256(higher_ids)
    if observed_higher_set_sha != expected_higher_identity_set_sha:
        raise StageRuntimeError("G6:S3R higher exact identity-set binding mismatch")
    overlap = sorted(set(higher_ids) & set(s1_ids))
    if overlap:
        raise StageRuntimeError(f"G6:S3R rebuilt higher panel overlaps S1 at {overlap[0]}")
    if any(str(r["state_token"]) != f"{r['identity_sha256']}:0" for r in higher_rows):
        raise StageRuntimeError("G6:S3R higher identity digest collision encountered; historical state-id route cannot proceed")

    l1_tasks = []
    for r in higher_rows:
        payload = {"state_tree": r["state"], "probe_ref": "D2_PATH", "operator": [0, 0], "position": "LEFT"}
        l1_tasks.append(TaskSpec(
            task_id=str(r["identity_sha256"]), task_kind="G6_S3R_L1_RECOMPUTE",
            binding_sha256=canonical_sha256({"question_sha256": str(stage["question_sha256"]), "payload": payload}),
            payload=payload, cost_weight=max(1.0, float(r["state"]["n"])),
        ))
    part = runtime.run_structural_partition(
        phase_id=str(params.get("l1_phase_id", "L1_S2R_CONTEXT_RECOMPUTE")),
        tasks=l1_tasks,
        evaluator_ref="infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator",
        requested_workers=workers,
        max_tasks=expected_higher_count,
    )
    summary = part.summary; emeta = part.execution_metadata
    if str(summary["partition_bindings_sha256"]) != expected_partition_binding:
        raise StageRuntimeError("G6:S3R recomputed L1 partition binding differs from historical migrated binding")
    observer_workspace_limit = int(params.get("observer_workspace_acceptance_bytes", 100 * 1024 * 1024))
    observer_rss_limit = int(params.get("observer_rss_acceptance_bytes", 1024 * 1024 * 1024))
    if int(emeta.get("evidence_bytes_peak", 2**63-1)) > observer_workspace_limit:
        raise StageRuntimeError("G6:S3R observer checkpoint footprint exceeds accepted limit")
    if int(emeta.get("reducer_rss_bytes_peak", 2**63-1)) > observer_rss_limit:
        raise StageRuntimeError("G6:S3R reducer RSS exceeds accepted limit")
    discrete = int(summary["class_count"]) == expected_higher_count and int(summary["multi_class_count"]) == 0
    outcome = "PASS_HIGHER_DISCRETE_KERNEL_L1" if discrete else "L1_COLLISIONS_REQUIRE_CONTROLLER_ONLY_L2"
    return ChainExecutionResult(result={
        "schema_id": "IG_G6_S3R_CONTROLLER_ONLY_L1_RECOMPUTE_RESULT_V1",
        "outcome": outcome,
        "original_question_sha256": str(stage["question_sha256"]),
        "scientific_observer_unchanged": True,
        "legacy_science_rerun": True,
        "execution_architecture": "DECODER_COMMON_RUNTIME_ONLY",
        "s1_rebuild": s1_gen.summary,
        "s1_identity_set_sha256": observed_s1_set_sha,
        "higher_rebuild": higher_gen.summary,
        "higher_identity_set_sha256": observed_higher_set_sha,
        "higher_overlap_with_s1_count": 0,
        "partition": summary,
        "recovery_execution": {
            "s1_generation_wall_seconds": float(s1_gen.execution_metadata.get("wall_seconds", 0.0)),
            "higher_generation_wall_seconds": float(higher_gen.execution_metadata.get("wall_seconds", 0.0)),
            "l1_observer_wall_seconds": float(emeta.get("wall_seconds", 0.0)),
            "l1_evidence_bytes_peak": int(emeta.get("evidence_bytes_peak", 0)),
            "l1_reducer_rss_bytes_peak": int(emeta.get("reducer_rss_bytes_peak", 0)),
            "workers": int(emeta.get("workers", 0)),
        },
        "historical_partition_binding_reproduced": True,
        "promotion_effect": "NONE",
    })
