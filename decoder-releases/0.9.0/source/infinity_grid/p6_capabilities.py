from __future__ import annotations

"""Registered P6 specialist science capabilities.

These functions are intentionally small.  The generic workflow engine owns the
branching/stopping semantics; these capabilities only expose family-specific
mathematical observations through P5 adapters.
"""

from typing import Any, Mapping

from .canon import canonical_sha256
from .v05_workflow import register_workflow_capability, register_workflow_semantic_contract
from .adapters.g4_accepted import DecoratedG4Tree


@register_workflow_capability("p6.history.reference")
def historical_stage_reference(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    """Transport an already-earned stage result through the generic S shell.

    This is not a rerun of the historical stage.  It checks that the registered
    adapter family/scope and frozen reference identity are the ones declared by
    the transport registration and returns a non-promoting history record.
    """
    if execution_context is not None:
        execution_context.reserve_cases(1)
    family = adapter.descriptor()["family"]
    if parameters.get("expected_adapter_family") != family:
        return {"outcome": "BLOCKED", "observed_alternative": "ADAPTER_FAMILY_MISMATCH", "family": family}
    if parameters.get("stage") != stage:
        return {"outcome": "BLOCKED", "observed_alternative": "STAGE_BINDING_MISMATCH"}
    spec_sha = parameters.get("spec_sha256")
    if not isinstance(spec_sha, str) or len(spec_sha) != 64:
        return {"outcome": "BLOCKED", "observed_alternative": "SPEC_IDENTITY_MISSING"}
    classification = str(parameters.get("classification", ""))
    if not classification:
        return {"outcome": "BLOCKED", "observed_alternative": "REFERENCE_CLASSIFICATION_MISSING"}
    scope = adapter.scope_contract()
    out = {
        "history_mode": "ALREADY_EARNED_LAW_TRANSPORT",
        "historical_stage_ref": parameters.get("historical_stage_ref"),
        "reference_status": parameters.get("reference_status", "PASS"),
        "reference_classification": classification,
        "reference_spec_sha256": spec_sha,
        "adapter_descriptor": adapter.descriptor(),
        "adapter_scope_sha256": canonical_sha256(scope),
        "full_historical_stage_rerun": False,
        "new_science": False,
        "promotion": False,
    }
    return {"outcome": "PASS", "observed_alternative": classification, **out}


def _g4_path_broom(depth: int) -> tuple[DecoratedG4Tree, DecoratedG4Tree]:
    # R7 actual-G4 witness uses d+3 C-labelled vertices and root 0.
    n = int(depth) + 3
    path_edges = tuple((i, i + 1) for i in range(n - 1))
    broom_edges = tuple(
        [(i, i + 1) for i in range(int(depth))]
        + [(int(depth), int(depth) + 1), (int(depth), int(depth) + 2)]
    )
    h = tuple("C" for _ in range(n))
    return (
        DecoratedG4Tree(n, path_edges, h, tuple((0, 0) for _ in path_edges)),
        DecoratedG4Tree(n, broom_edges, h, tuple((0, 0) for _ in broom_edges)),
    )


@register_workflow_capability("p6.g4.r7-path-broom")
def g4_r7_path_broom_rows(*, adapter: Any, parameters: Mapping[str, Any], registration, execution_context=None) -> dict[str, Any]:
    """Return exact Q/A/O rows for the certified actual-G4 path/broom family.

    Changing the registered ``message_depth`` is a new bounded probe and needs no
    dispatcher/worker/checkpoint/certificate code.  The capability is unchanged.
    """
    depth = int(parameters.get("message_depth", 2))
    witness_depth = int(parameters.get("witness_depth", 2))
    if depth < 0 or witness_depth < 0:
        raise ValueError("depths must be nonnegative")
    path, broom = _g4_path_broom(witness_depth)
    support = adapter.realizability_witness(witness_depth)
    if execution_context is not None:
        execution_context.reserve_cases(2)
    rows = []
    for parent_id, tree in (("PATH", path), ("BROOM", broom)):
        q = adapter.public_read(tree)
        a = adapter.action_read(tree, 0, depth)
        children = adapter.graft_relation(tree, 0, new_H_class="C", operator=(0, 0))
        if len(children) != 1:
            rows.append({
                "parent_id": parent_id, "action_id": "ROOT0_C_00", "legal": False,
                "Q": q, "A": a, "O": None, "exact_action": adapter.rooted_canon(tree, 0),
            })
            continue
        child = children[0]
        rows.append({
            "parent_id": parent_id,
            "action_id": "ROOT0_C_00",
            "legal": True,
            "Q": q,
            "A": a,
            "O": adapter.unrooted_canon(child),
            "exact_action": adapter.rooted_canon(tree, 0),
        })
    return {
        "observed_alternative": "ROWS_RETURNED",
        "rows": rows,
        "support_certificate": support,
        "message_depth": depth,
        "witness_depth": witness_depth,
        "fresh_holdout": False,
        "role": "R7_REGRESSION_OR_BOUNDED_PROBE",
        # The capability reports which preregistered proof obligations this exact
        # construction discharges.  The workflow engine compares this set against
        # the registration rather than trusting an empty "unmet" list.
        "satisfied_proof_obligations": [
            "ACTUAL_G4_REALIZABILITY",
            "FIXED_CHILD_OBSERVER",
            "NO_GLOBAL_MINIMALITY_TRANSFER",
        ],
        "unmet_proof_obligations": [],
        "changed_assumptions": [],
    }


register_workflow_semantic_contract("p6.g4.r7-path-broom", {
    "contract_id":"IG_P6_G4_R7_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"R_INVESTIGATION", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG",
    "parameter_keys":["message_depth","witness_depth"],
})
