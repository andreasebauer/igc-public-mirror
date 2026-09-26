from __future__ import annotations

"""Central registry for Decoder v0.5 controller-only scientific stage handlers.

O3A adds a semantic service contract to the existing execution registry.  The
legacy evaluator tuple remains derived for compatibility; EvaluatorSpec is now
the source of truth for what each evaluator may ask the Decoder kernel to do.
"""

from typing import Any
from pathlib import Path
import importlib
import inspect

from .v05_stage_architecture import audit_callable
from .v05_kernel_services import (
    EvaluatorSpec, PRIMARY, REFERENCE_ORACLE,
    audit_evaluator_semantics, require_evaluator_semantics,
)


CONTROLLER_ONLY_STAGE_HANDLERS: dict[str, str] = {
    "g6.s3r.l1.legacy_recovery": "infinity_grid.g6_s3_controller_stage:g6_s3r_l1_recovery_handler",
    "g6.s3r.l1.recompute": "infinity_grid.g6_s3_controller_stage:g6_s3r_l1_recompute_handler",
}

# O3B: every production evaluator is service-only; no exact-byte legacy exemptions remain.
CONTROLLER_ONLY_EVALUATOR_SPECS: tuple[EvaluatorSpec, ...] = (
    EvaluatorSpec('infinity_grid.g6_controller_evaluators:exact_one_step_relation_evaluator',PRIMARY,('EXACT_RELATION',)),
    EvaluatorSpec('infinity_grid.g6_controller_evaluators:g6_s1_universe_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_controller_evaluators:g6_s3_axis_a_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_controller_evaluators:g6_s3_axis_b_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_controller_evaluators:g6_s3_higher_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_s7_evaluators:s7_public_state_evaluator',PRIMARY,('PUBLIC_READ',)),
    EvaluatorSpec('infinity_grid.g6_s7_evaluators:s7_ordinary_branch_count_component_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION_PROFILE')),
    EvaluatorSpec('infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_BATCH','EXACT_RELATION_PROFILE_FAMILY')),
    EvaluatorSpec('infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION_PROFILE')),
    EvaluatorSpec('infinity_grid.g6_s7_evaluators:s7_ordinary_branch_relation_evaluator',PRIMARY,('AUTHORITY_BASIS','PUBLIC_READ','EXACT_RELATION','EXACT_RELATION_PROFILE')),
    EvaluatorSpec('infinity_grid.g6_s8_evaluators:s8_recursive_outer_component_evaluator',PRIMARY,('AUTHORITY_BASIS','PUBLIC_READ','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_FAMILY')),
    EvaluatorSpec('infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator',PRIMARY,('AUTHORITY_BASIS','PUBLIC_READ','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_FAMILY')),
    EvaluatorSpec('infinity_grid.g6_s8_evaluators:s8_recursive_ordinary_signature_evaluator',PRIMARY,('AUTHORITY_BASIS','PUBLIC_READ','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_FAMILY')),
    EvaluatorSpec('infinity_grid.g6_s8_evaluators:s8_v3_relation_generation_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION')),
    EvaluatorSpec('infinity_grid.g6_s8_evaluators:s8_v3_profile_extension_evaluator',PRIMARY,('AUTHORITY_BASIS','EXACT_RELATION_PROFILE_BATCH')),
    EvaluatorSpec('infinity_grid.g6_s5r_crw_evaluators:observer_inversion_descriptor_evaluator',PRIMARY,('EXACT_IDENTITY','OBSERVER_Q','OBSERVER_DECODE','ATTACHMENT_RESPONSE_DESCRIPTOR')),
    EvaluatorSpec('infinity_grid.g6_s5r_crw_evaluators:observer_state_write_law_evaluator',PRIMARY,('EXACT_IDENTITY','EXACT_RELATION','OBSERVER_Q','OBSERVER_WRITE')),
    EvaluatorSpec('infinity_grid.g6_s6r_evaluators:recursive_closure_holdout_evaluator',PRIMARY,('EXACT_IDENTITY','EXACT_RELATION','OBSERVER_Q','OBSERVER_DECODE','OBSERVER_WRITE')),
    EvaluatorSpec('infinity_grid.g6_s6r_evaluators:l2_common_parent_uniqueness_evaluator',PRIMARY,('EXACT_IDENTITY','OBSERVER_Q','OBSERVER_DECODE')),
    EvaluatorSpec('infinity_grid.g6_s6r_evaluators:l2_sibling_q_evaluator',PRIMARY,('OBSERVER_Q',)),
    EvaluatorSpec('infinity_grid.g6_s6r_evaluators:recursive_closure_independent_evaluator',REFERENCE_ORACLE,
                  ('EXACT_IDENTITY','EXACT_RELATION','OBSERVER_Q','LEGACY_OBSERVER_DECODE_ORACLE'),
                  reference_oracle_for=('LEGACY_OBSERVER_DECODE_ORACLE',)),
    EvaluatorSpec('infinity_grid.g6_marker_evaluators:marker_write_holdout_evaluator',PRIMARY,('EXACT_IDENTITY','EXACT_RELATION','MARKER_Q','MARKER_DECODE','MARKER_WRITE')),
    EvaluatorSpec('infinity_grid.g6_marker_evaluators:marker_write_independent_evaluator',REFERENCE_ORACLE,
                  ('EXACT_IDENTITY','EXACT_RELATION','MARKER_Q','LEGACY_MARKER_DECODE_ORACLE'),
                  reference_oracle_for=('LEGACY_MARKER_DECODE_ORACLE',)),
)

EVALUATOR_SPEC_BY_REF: dict[str, EvaluatorSpec] = {s.ref: s for s in CONTROLLER_ONLY_EVALUATOR_SPECS}
if len(EVALUATOR_SPEC_BY_REF) != len(CONTROLLER_ONLY_EVALUATOR_SPECS):
    raise RuntimeError("DUPLICATE_EVALUATOR_SPEC")

# Compatibility tuple is derived, never hand-maintained.
CONTROLLER_ONLY_WORKER_EVALUATORS: tuple[str, ...] = tuple(s.ref for s in CONTROLLER_ONLY_EVALUATOR_SPECS)


def get_evaluator_spec(ref: str) -> EvaluatorSpec:
    try:
        return EVALUATOR_SPEC_BY_REF[str(ref)]
    except KeyError as exc:
        raise ValueError("EVALUATOR_NOT_IN_ACCEPTED_SPEC_REGISTRY:" + str(ref)) from exc


def _resolve(ref: str):
    mod, sep, name = str(ref).partition(":")
    if not sep:
        raise ValueError(f"bad callable ref {ref!r}")
    fn = getattr(importlib.import_module(mod), name)
    if not callable(fn):
        raise TypeError(f"not callable: {ref}")
    return fn


def semantic_audit_for_ref(ref: str) -> dict[str, Any]:
    spec = get_evaluator_spec(ref)
    fn = _resolve(ref)
    path = inspect.getsourcefile(fn)
    if not path:
        raise RuntimeError("EVALUATOR_SOURCE_UNRESOLVED:" + ref)
    return audit_evaluator_semantics(path, spec)


def require_registered_evaluator_semantics(ref: str, module_path: str | Path) -> dict[str, Any]:
    return require_evaluator_semantics(module_path, get_evaluator_spec(ref))


def register_controller_only_stage_handlers(controller) -> None:
    for key, ref in sorted(CONTROLLER_ONLY_STAGE_HANDLERS.items()):
        controller.register_stage_handler(key, _resolve(ref))


def controller_only_registry_audit() -> dict[str, Any]:
    handlers = []
    for key, ref in sorted(CONTROLLER_ONLY_STAGE_HANDLERS.items()):
        row = audit_callable(_resolve(ref)); row["registry_key"] = key; handlers.append(row)
    evaluators = []
    for spec in CONTROLLER_ONLY_EVALUATOR_SPECS:
        row = audit_callable(_resolve(spec.ref)); row["registry_ref"] = spec.ref
        semantic = semantic_audit_for_ref(spec.ref)
        row["semantic_gate"] = semantic
        row["evaluator_spec"] = spec.as_dict()
        if semantic["status"] not in {"PASS", "PASS_LEGACY_FROZEN"}:
            row["status"] = "FAIL"
        evaluators.append(row)
    rows = handlers + evaluators
    return {
        "schema_id": "IG_DECODER_V05_CONTROLLER_ONLY_STAGE_REGISTRY_AUDIT_V3_O3B",
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        "handler_count": len(handlers),
        "evaluator_count": len(evaluators),
        "legacy_frozen_evaluator_count": sum(1 for r in evaluators if r["semantic_gate"]["status"] == "PASS_LEGACY_FROZEN"),
        "handlers": handlers,
        "evaluators": evaluators,
    }

# These entries exist only for explicit ENGINEERING fixture sessions. They cannot
# be registered as an official science handler or emit official execution receipts.
ENGINEERING_ONLY_STAGE_HANDLERS = {
    "fixture": "infinity_grid.controller_only_fixture:stage_handler",
    "engineering.validate": "infinity_grid.v05_engineering_stage:engineering_job_handler",
    "engineering.accept": "infinity_grid.v05_engineering_stage:engineering_acceptance_handler",
}
ENGINEERING_ONLY_WORKER_EVALUATORS = (
    "infinity_grid.controller_only_fixture:partition_evaluator",
    "infinity_grid.controller_only_fixture:collision_partition_evaluator",
    "infinity_grid.controller_only_fixture:generation_evaluator",
    "infinity_grid.controller_only_fixture:collision_generation_evaluator",
)
