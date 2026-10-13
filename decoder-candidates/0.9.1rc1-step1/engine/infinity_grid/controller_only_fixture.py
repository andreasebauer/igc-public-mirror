from __future__ import annotations

"""Small controller-only scientific-stage fixture used by architecture gates.

This module intentionally contains no execution infrastructure and no durable I/O.
It is a release-test fixture for the v0.5 V2 DECODER_STAGE contract.
"""

from typing import Any, Mapping

from .canon import canonical_sha256
from .execution import TaskSpec
from .v05_chain import ChainExecutionResult


def partition_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = int(payload["value"])
    modulus = int(payload.get("modulus", 3))
    return {
        "signature": {"residue": value % modulus},
        "outcome_count": 1,
        "metrics": {"value": value},
    }


def collision_partition_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = int(payload["value"])
    return {
        "signature": {"exact_value": value},
        "outcome_count": 1,
        "metrics": {},
        "_test_signature_digest_override": "f" * 64,
    }


def stage_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    params = dict(stage["execution"].get("parameters") or {})
    values = [int(x) for x in params.get("values", list(range(12)))]
    modulus = int(params.get("modulus", 3))
    collision_mode = bool(params.get("collision_mode", False))
    evaluator_ref = (
        "infinity_grid.controller_only_fixture:collision_partition_evaluator"
        if collision_mode
        else "infinity_grid.controller_only_fixture:partition_evaluator"
    )
    tasks = []
    for value in values:
        payload = {"value": value, "modulus": modulus}
        binding = canonical_sha256({
            "question_sha256": str(stage["question_sha256"]),
            "task_kind": "CONTROLLER_ONLY_FIXTURE",
            "payload": payload,
        })
        tasks.append(TaskSpec(
            task_id=f"v{value:06d}",
            task_kind="CONTROLLER_ONLY_FIXTURE",
            binding_sha256=binding,
            payload=payload,
            cost_weight=1.0,
        ))
    part = runtime.run_structural_partition(
        phase_id=str(params.get("phase_id", "PARTITION")),
        tasks=tasks,
        evaluator_ref=evaluator_ref,
        requested_workers=int(params.get("workers", runtime.default_workers)),
        max_tasks=int(params.get("max_tasks", max(1, len(tasks)))),
    )
    return ChainExecutionResult(result={
        "outcome": "PASS",
        "partition": part.summary,
    })


def generation_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = int(payload["value"])
    # Two occurrences, with one identity shared across neighboring tasks.
    return {
        "states": [
            {"identity": {"kind": "shared", "value": value // 2}, "state": {"value": value, "slot": 0}},
            {"identity": {"kind": "unique", "value": value}, "state": {"value": value, "slot": 1}},
        ],
        "metrics": {"fixture_generation_calls": 1},
    }


def collision_generation_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = int(payload["value"])
    return {
        "states": [
            {"identity": {"kind": "collision", "value": value}, "state": {"value": value},
             "_test_identity_digest_override": "e" * 64}
        ],
        "metrics": {},
    }
