from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canon import canonical_sha256
from .graph_adapters import (
    register_current_environment,
    register_current_source,
    register_protocol,
    register_question,
)
from .graph_store import GraphStore, build_node, build_recipe
from .protocols import ProtocolRegistry


def register_runtime_backbone(
    graph: GraphStore,
    *,
    source_manifest_path: str | Path,
    wheel_path: str | Path | None = None,
) -> dict[str, Any]:
    """Register the exact current source/environment/protocol foundation."""
    source = register_current_source(graph, source_manifest_path=source_manifest_path, wheel_path=wheel_path)
    environment = register_current_environment(graph)
    protocols = {}
    for descriptor in ProtocolRegistry().list():
        node = register_protocol(graph, descriptor)
        protocols[(descriptor["protocol_id"], str(descriptor["version"]))] = node["node_id"]
    return {
        "schema_id": "IG_GRAPH_RUNTIME_BACKBONE_REGISTRATION_V0_1",
        "status": "PASS",
        "source_node_id": source["node_id"],
        "environment_node_id": environment["node_id"],
        "protocol_nodes": {f"{k[0]}@{k[1]}": v for k, v in sorted(protocols.items())},
    }


def _artifact_node(graph: GraphStore, path: Path, *, role: str, retention: str = "PINNED") -> dict[str, Any]:
    record = graph.artifacts.put_file(path, logical_role=role, source_name=path.name)
    return graph.register_artifact_node(record, semantic_role=role, retention_class=retention)


def register_principal_run_replay(
    graph: GraphStore,
    *,
    run_id: str,
    principal_logical_name: str,
    alias: str,
    alias_version: str,
    source_node_id: str,
    environment_node_id: str,
    retention_class: str = "CACHE",
) -> dict[str, Any]:
    """Register one existing run's principal artifact as an exact replay target.

    The original run and artifacts are never rewritten. The resulting recipe is path-independent
    and resolves only exact graph identities.
    """
    run_dir = graph.paths.runs / run_id
    run_path = run_dir / "run.json"
    plan_path = run_dir / "plan.json"
    question_path = run_dir / "question.json"
    if not all(p.is_file() for p in (run_path, plan_path, question_path)):
        raise FileNotFoundError(f"incomplete run record for {run_id}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    question = json.loads(question_path.read_text(encoding="utf-8"))
    if run.get("lifecycle") != "COMPLETE_VALID":
        raise RuntimeError(f"run {run_id} is not COMPLETE_VALID")

    source = graph.get_node(source_node_id)
    environment = graph.get_node(environment_node_id)
    if source["node_type"] != "SOURCE_SNAPSHOT" or environment["node_type"] != "ENVIRONMENT":
        raise RuntimeError("source/environment node type mismatch")

    descriptor = ProtocolRegistry().get(
        plan["protocol_id"],
        version=plan.get("protocol_version"),
        descriptor_sha=plan.get("descriptor_sha256"),
    )
    protocol_node = register_protocol(graph, descriptor)
    question_node = register_question(graph, question)
    if question_node["semantic_identity"]["question_sha256"] != run["question_sha256"]:
        raise RuntimeError("question identity mismatch")

    dsid = plan["input_datasets"][0]["dataset_sha256"]
    dataset_manifest = graph.datasets.load(dsid)
    dataset_node = graph.register_dataset_node(dataset_manifest, semantic_role=dataset_manifest["logical_role"], retention_class="PINNED")
    dataset_manifest_path = graph.paths.store / "datasets" / f"{dsid}.json"
    dataset_manifest_node = _artifact_node(graph, dataset_manifest_path, role="DATASET_MANIFEST")
    plan_node = _artifact_node(graph, plan_path, role="EXECUTION_PLAN")

    mode = run.get("execution_mode") or {
        "schema_id": "IG_EXECUTION_CAPABILITY_V0_17_1",
        "mode": "PHASE1_HOLD",
        "new_science_enabled": False,
        "reason": "GRAPH_REPLAY_COMPATIBILITY",
    }
    mode_record = graph.artifacts.put_bytes(
        (json.dumps(mode, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
        logical_role="EXECUTION_MODE",
        source_name=f"{run_id}_EXECUTION_MODE.json",
        media_type="application/json",
    )
    mode_node = graph.register_artifact_node(mode_record, semantic_role="EXECUTION_MODE", retention_class="PINNED")

    principal = next((a for a in run.get("result_artifacts", []) if a.get("logical_name") == principal_logical_name), None)
    if principal is None:
        raise RuntimeError(f"principal artifact {principal_logical_name} missing from {run_id}")
    target_node = graph.register_artifact_node(
        principal,
        semantic_role=principal_logical_name,
        retention_class=retention_class,
        status="SEALED",
        provenance_refs=[{"run_id": run_id, "run_core_sha256": run["run_core_sha256"]}],
    )

    input_role = plan["input_datasets"][0]["logical_role"]
    output_role = principal_logical_name
    recipe = build_recipe(
        recipe_id=f"replay:{run_id}:{principal_logical_name}",
        recipe_version="1",
        producer={
            "source_snapshot_ref": source_node_id,
            "entrypoint": "infinity_grid.controller.Controller.run",
            "runtime_family": "UNIFIED_RUNTIME",
        },
        protocol_ref=protocol_node["node_id"],
        question_ref=question_node["node_id"],
        input_requirements=[
            {
                "semantic_role": input_role,
                "accepted_node_types": ["DATASET"],
                "exact_identity_or_constraint": {"node_id": dataset_node["node_id"]},
                "resolution_policy": "EXACT_IDENTITY",
                "optional": False,
                "materialize_as": f"inputs/{input_role}",
            },
            {
                "semantic_role": "EXECUTION_PLAN",
                "accepted_node_types": ["ARTIFACT"],
                "exact_identity_or_constraint": {"node_id": plan_node["node_id"]},
                "resolution_policy": "EXACT_IDENTITY",
                "optional": False,
                "materialize_as": "inputs/plan.json",
            },
            {
                "semantic_role": "DATASET_MANIFEST",
                "accepted_node_types": ["ARTIFACT"],
                "exact_identity_or_constraint": {"node_id": dataset_manifest_node["node_id"]},
                "resolution_policy": "EXACT_IDENTITY",
                "optional": False,
                "materialize_as": "inputs/dataset_manifest.json",
            },
            {
                "semantic_role": "EXECUTION_MODE",
                "accepted_node_types": ["ARTIFACT"],
                "exact_identity_or_constraint": {"node_id": mode_node["node_id"]},
                "resolution_policy": "EXACT_IDENTITY",
                "optional": False,
                "materialize_as": "inputs/execution_mode.json",
            },
        ],
        environment_ref=environment_node_id,
        stages=[
            {
                "stage_id": "execute_unified_plan",
                "dependencies": [],
                "action": {
                    "type": "UNIFIED_RUNTIME_PLAN",
                    "plan_input_role": "EXECUTION_PLAN",
                    "dataset_input_role": input_role,
                    "dataset_manifest_input_role": "DATASET_MANIFEST",
                    "execution_mode_input_role": "EXECUTION_MODE",
                    "output_logical_names": {output_role: principal_logical_name},
                },
                "output_roles": [output_role],
                "checkpoint_policy": "UNIFIED_RUNTIME_NATIVE",
                "determinism_contract": "BYTE_IDENTICAL_DECLARED_OUTPUT",
            }
        ],
        outputs=[
            {
                "semantic_role": output_role,
                "relative_path": f"outputs/{principal_logical_name}",
                "node_id": target_node["node_id"],
                "expected_sha256": principal["sha256"],
            }
        ],
        canonicalization={"version": "IG_CANONICAL_JSON_V1", "dataset": dataset_manifest.get("canonicalization_version")},
        acceptance={
            "expected_hashes": {output_role: principal["sha256"]},
            "expected_invariants": {"lifecycle": "COMPLETE_VALID"},
            "evidence_scope": run.get("evidence", {}),
            "fail_closed_conditions": [
                "HASH_MISMATCH", "SOURCE_MISMATCH", "ENVIRONMENT_MISMATCH",
                "CANONICALIZATION_MISMATCH", "UNRESOLVED_INPUT",
            ],
        },
        resource_policy={"network": "FORBIDDEN", "clean_root": True},
        replay_class="EXACT_REPLAY",
    )
    recipe = graph.put_recipe(recipe)
    recipe_node_id = f"recipe:sha256:{recipe['recipe_sha256']}"
    graph.add_edge(edge_type="PRODUCED_BY", source_node_id=target_node["node_id"], target_node_id=recipe_node_id, scope="EXACT_REPLAY")
    for dep, scope in [
        (source_node_id, "SOURCE"),
        (environment_node_id, "ENVIRONMENT"),
        (protocol_node["node_id"], "PROTOCOL"),
        (question_node["node_id"], "QUESTION"),
        (dataset_node["node_id"], "INPUT_DATASET"),
        (plan_node["node_id"], "EXECUTION_PLAN"),
        (dataset_manifest_node["node_id"], "DATASET_MANIFEST"),
        (mode_node["node_id"], "EXECUTION_MODE"),
    ]:
        graph.add_edge(edge_type="REQUIRES", source_node_id=recipe_node_id, target_node_id=dep, scope=scope)
    graph.register_alias(alias, alias_version, target_node["node_id"])
    return {
        "schema_id": "IG_PRINCIPAL_RUN_REPLAY_REGISTRATION_V0_1",
        "status": "PASS",
        "run_id": run_id,
        "target_node_id": target_node["node_id"],
        "target_sha256": principal["sha256"],
        "recipe_sha256": recipe["recipe_sha256"],
        "alias": alias,
        "alias_version": alias_version,
    }
