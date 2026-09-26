from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canon import canonical_bytes, canonical_sha256
from .graph_store import GraphStore, build_node
from .hashing import sha256_file
from .protocols import ProtocolRegistry
from .records import runtime_descriptor, runtime_sha256, source_sha256


def _artifact_for_json(graph: GraphStore, obj: dict[str, Any], *, role: str, source_name: str) -> dict:
    rec = graph.artifacts.put_bytes(canonical_bytes(obj), logical_role=role, source_name=source_name, media_type="application/json")
    return rec


def register_current_source(graph: GraphStore, *, source_manifest_path: str | Path, wheel_path: str | Path | None = None) -> dict:
    manifest_path = Path(source_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    src_sha = manifest["source_sha256"]
    if src_sha != source_sha256():
        raise RuntimeError("source manifest does not match live executable source")
    mrec = graph.artifacts.put_file(manifest_path, logical_role="SOURCE_TREE_MANIFEST", source_name=manifest_path.name)
    refs = [{"kind": "ARTIFACT", "sha256": mrec["sha256"], "size_bytes": mrec["size_bytes"]}]
    metadata = {"manifest_sha256": mrec["sha256"], "record_is_content": False}
    if wheel_path is not None:
        wrec = graph.artifacts.put_file(wheel_path, logical_role="RUNTIME_WHEEL", source_name=Path(wheel_path).name)
        refs.append({"kind": "ARTIFACT", "sha256": wrec["sha256"], "size_bytes": wrec["size_bytes"]})
        metadata["wheel_sha256"] = wrec["sha256"]
    node = build_node(
        node_type="SOURCE_SNAPSHOT", semantic_identity={"source_sha256": src_sha},
        semantic_role="ALGEBRA_DECODER_RUNTIME_SOURCE", retention_class="PINNED",
        status="VERIFIED", content_refs=refs, metadata=metadata,
    )
    return graph.put_node(node)


def register_current_environment(graph: GraphStore) -> dict:
    descriptor = runtime_descriptor()
    env_sha = runtime_sha256()
    obj = {"schema_id": "IG_DECODER_ENVIRONMENT_LOCK_V0_1", "environment_sha256": env_sha, "descriptor": descriptor}
    rec = _artifact_for_json(graph, obj, role="ENVIRONMENT_LOCK", source_name="ENVIRONMENT_LOCK.json")
    node = build_node(
        node_type="ENVIRONMENT", semantic_identity={"environment_sha256": env_sha},
        semantic_role="CURRENT_ALGEBRA_DECODER_ENVIRONMENT", retention_class="PINNED",
        content_refs=[{"kind": "ARTIFACT", "sha256": rec["sha256"], "size_bytes": rec["size_bytes"]}],
        metadata={"descriptor": descriptor},
    )
    return graph.put_node(node)


def register_protocol(graph: GraphStore, descriptor: dict[str, Any]) -> dict:
    rec = _artifact_for_json(graph, descriptor, role="PROTOCOL_DESCRIPTOR", source_name=f"{descriptor['protocol_id']}_{descriptor['version']}.json")
    node = build_node(
        node_type="PROTOCOL",
        semantic_identity={
            "protocol_id": descriptor["protocol_id"], "version": str(descriptor["version"]),
            "descriptor_sha256": descriptor["descriptor_sha256"],
        },
        semantic_role=f"PROTOCOL_{descriptor['protocol_id']}", retention_class="PINNED",
        content_refs=[{"kind": "ARTIFACT", "sha256": rec["sha256"], "size_bytes": rec["size_bytes"]}],
        metadata={"purpose": descriptor.get("purpose")},
    )
    out = graph.put_node(node)
    graph.register_alias(f"ig:protocol/{descriptor['protocol_id']}", str(descriptor["version"]), out["node_id"])
    return out


def register_question(graph: GraphStore, question: dict[str, Any], *, semantic_role: str = "FROZEN_QUESTION") -> dict:
    qsha = canonical_sha256(question)
    rec = _artifact_for_json(graph, question, role="QUESTION", source_name="question.json")
    node = build_node(
        node_type="QUESTION", semantic_identity={"question_sha256": qsha}, semantic_role=semantic_role,
        retention_class="PINNED", content_refs=[{"kind": "ARTIFACT", "sha256": rec["sha256"], "size_bytes": rec["size_bytes"]}],
        metadata={"protocol_id": question.get("protocol_id"), "subject": question.get("subject")},
    )
    return graph.put_node(node)


def register_review_decision(graph: GraphStore, decision: dict[str, Any], *, semantic_role: str, retention_class: str = "PINNED") -> dict:
    dsha = canonical_sha256(decision)
    rec = _artifact_for_json(graph, decision, role="REVIEW_DECISION", source_name=f"{semantic_role}.json")
    node = build_node(
        node_type="REVIEW_DECISION", semantic_identity={"decision_sha256": dsha}, semantic_role=semantic_role,
        retention_class=retention_class, status=decision.get("status", "FROZEN"),
        content_refs=[{"kind": "ARTIFACT", "sha256": rec["sha256"], "size_bytes": rec["size_bytes"]}],
        metadata={"scope": decision.get("scope"), "record_is_content": False},
    )
    return graph.put_node(node)


def ingest_existing_records(paths) -> dict[str, Any]:
    """Add graph mirrors for existing runtime records without rewriting originals."""
    graph = GraphStore(paths)
    counts = {k: 0 for k in ["artifacts", "datasets", "protocols", "runs", "stages", "claims", "releases"]}
    # Existing artifact records.
    for p in sorted((paths.store / "artifacts").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        graph.register_artifact_node(rec, semantic_role=rec.get("logical_role") or "ARTIFACT", retention_class="PINNED")
        counts["artifacts"] += 1
    # Existing datasets.
    for p in sorted((paths.store / "datasets").glob("*.json")):
        man = json.loads(p.read_text(encoding="utf-8"))
        graph.register_dataset_node(man, semantic_role=man.get("logical_role") or "DATASET", retention_class="PINNED")
        counts["datasets"] += 1
    # Protocols are explicit executable records.
    reg = ProtocolRegistry()
    for desc in reg.list():
        register_protocol(graph, desc); counts["protocols"] += 1
    # Runs and stages are adapted by immutable record artifacts.
    for rp in sorted(paths.runs.glob("*/run.json")):
        rr = json.loads(rp.read_text(encoding="utf-8"))
        core_path = rp.parent / "run_core.json"
        if not core_path.is_file():
            continue
        core = json.loads(core_path.read_text(encoding="utf-8"))
        core_rec = graph.artifacts.put_file(core_path, logical_role="RUN_CORE", source_name=f"{rr['run_id']}_run_core.json")
        run_node = build_node(
            node_type="RUN", semantic_identity={"run_core_sha256": core["run_core_sha256"], "run_id": rr["run_id"]},
            semantic_role=f"RUN_{rr['protocol']['protocol_id']}", retention_class="PINNED",
            status=rr.get("lifecycle", "UNKNOWN"),
            content_refs=[{"kind": "ARTIFACT", "sha256": core_rec["sha256"], "size_bytes": core_rec["size_bytes"]}],
            metadata={"protocol": rr.get("protocol"), "evidence": rr.get("evidence"), "record_is_content": False},
        )
        run_node = graph.put_node(run_node); counts["runs"] += 1
        # Explicit protocol binding.
        try:
            pnode = graph.resolve(f"protocol:sha256:{rr['protocol']['descriptor_sha256']}")
            graph.add_edge(edge_type="REQUIRES", source_node_id=run_node["node_id"], target_node_id=pnode["node_id"], scope="PROTOCOL")
        except Exception:
            pass
        for d in rr.get("input_artifacts", []):
            dsid = d.get("dataset_sha256")
            if dsid:
                try: graph.add_edge(edge_type="REQUIRES", source_node_id=run_node["node_id"], target_node_id=f"dataset:sha256:{dsid}", scope="INPUT_DATASET")
                except Exception: pass
        for a in rr.get("result_artifacts", []):
            try:
                an = graph.get_node(f"artifact:sha256:{a['sha256']}")
                graph.add_edge(edge_type="PRODUCED_BY", source_node_id=an["node_id"], target_node_id=run_node["node_id"], scope="RUN_OUTPUT")
            except Exception:
                pass
        for cp in sorted((rp.parent / "stages").glob("*/current.json")):
            ptr = json.loads(cp.read_text(encoding="utf-8")); ap = cp.parent / "attempts" / f"{int(ptr['attempt']):06d}.json"
            if not ap.is_file(): continue
            co = json.loads(ap.read_text(encoding="utf-8")); arec = graph.artifacts.put_file(ap, logical_role="STAGE_CHECKPOINT", source_name=f"{rr['run_id']}_{co['stage_id']}.json")
            snode = build_node(
                node_type="STAGE", semantic_identity={
                    "run_node_id": run_node["node_id"], "stage_id": co["stage_id"],
                    "stage_spec_sha256": co.get("identity", {}).get("stage_spec_sha256") or canonical_sha256(co.get("stage_result", {})),
                    "dependency_binding_sha256": canonical_sha256(co.get("dependencies", [])),
                }, semantic_role="RUN_STAGE", retention_class="CACHE", status=co.get("status", "UNKNOWN"),
                content_refs=[{"kind": "ARTIFACT", "sha256": arec["sha256"], "size_bytes": arec["size_bytes"]}], producer_ref=run_node["node_id"],
            )
            snode = graph.put_node(snode)
            graph.add_edge(edge_type="CHECKPOINT_OF", source_node_id=snode["node_id"], target_node_id=run_node["node_id"], mandatory=False, scope="STAGE")
            counts["stages"] += 1
    # Claims: SUPPORTS edges are created only from explicit evidence_refs already accepted by the firewall.
    for p in sorted((paths.store / "claims").glob("*/*.json")):
        c = json.loads(p.read_text(encoding="utf-8")); crec = graph.artifacts.put_file(p, logical_role="CLAIM_RECORD", source_name=p.name)
        cnode = build_node(
            node_type="CLAIM", semantic_identity={"claim_id": c["claim_id"], "version": int(c["version"])},
            semantic_role="SCIENTIFIC_CLAIM", retention_class="PINNED", status=c.get("status", "UNKNOWN"),
            content_refs=[{"kind": "ARTIFACT", "sha256": crec["sha256"], "size_bytes": crec["size_bytes"]}], metadata={"statement": c.get("statement")},
        )
        cnode = graph.put_node(cnode); counts["claims"] += 1
        for rid in c.get("evidence_refs", []):
            matches = [n for n in graph.list_nodes() if n["node_type"] == "RUN" and n["semantic_identity"].get("run_id") == rid]
            if len(matches) == 1:
                graph.add_edge(edge_type="SUPPORTS", source_node_id=matches[0]["node_id"], target_node_id=cnode["node_id"], scope="EXPLICIT_CLAIM_EVIDENCE")
    # Releases.
    for p in sorted(paths.releases.glob("*/release_manifest.json")):
        r = json.loads(p.read_text(encoding="utf-8")); rrec = graph.artifacts.put_file(p, logical_role="RELEASE_MANIFEST", source_name=p.name)
        release_sha = r.get("archive_sha256") or rrec["sha256"]
        rnode = build_node(
            node_type="RELEASE", semantic_identity={"release_sha256": release_sha}, semantic_role=f"RELEASE_{r.get('mode')}", retention_class="PINNED",
            content_refs=[{"kind": "ARTIFACT", "sha256": rrec["sha256"], "size_bytes": rrec["size_bytes"]}], metadata={"release_id": r.get("release_id"), "run_id": r.get("run_id")},
        )
        rnode = graph.put_node(rnode); counts["releases"] += 1
        matches = [n for n in graph.list_nodes() if n["node_type"] == "RUN" and n["semantic_identity"].get("run_id") == r.get("run_id")]
        if len(matches) == 1:
            graph.add_edge(edge_type="DERIVED_FROM", source_node_id=rnode["node_id"], target_node_id=matches[0]["node_id"], scope="RELEASE_RUN")
    return {"schema_id": "IG_GRAPH_EXISTING_RECORD_INGEST_V0_1", "status": "PASS", "counts": counts}


def ingest_o7_unresolved_case(
    graph: GraphStore, *, preregistration_bundle: str | Path, qualified_engine_bundle: str | Path,
    repair_bundle: str | Path, postrun_analysis_bundle: str | Path,
) -> dict[str, Any]:
    roots = {}
    for role, path in [
        ("O7_PREREGISTRATION", preregistration_bundle),
        ("O7_QUALIFIED_ENGINE", qualified_engine_bundle),
        ("O7_CONTEXT_REPAIR", repair_bundle),
        ("O7_POSTRUN_ANALYSIS", postrun_analysis_bundle),
    ]:
        rec = graph.artifacts.put_file(path, logical_role=role, source_name=Path(path).name)
        roots[role] = graph.register_artifact_node(rec, semantic_role=role, retention_class="PINNED")
    status = {
        "schema_id": "IG_O7_RECONSTRUCTION_STATUS_V0_1",
        "status": "RECONSTRUCTABILITY_NOT_YET_PROVED",
        "scientific_scope": "STRONG_BOUNDED_DIRECT_OBSERVER_REPEAT_EVIDENCE_WITH_CERTIFICATION_DEBT_NOT_O7_GRADUATION",
        "known_missing_roles": [
            "O7_SELECTED_COMPONENT_PROFILES.json", "O7_IMMUTABLE_SURVIVORS.json",
            "COMPLETE_LATE_LANE_RANK_CHECKPOINTS", "C03_PROFILE_CHUNKS", "COMPLETE_C04_C05_CHECKPOINT_FAMILY",
        ],
        "forbidden_claim": "O7_GRADUATED",
    }
    srec = _artifact_for_json(graph, status, role="O7_RECONSTRUCTION_STATUS", source_name="O7_RECONSTRUCTION_STATUS.json")
    result_sha = canonical_sha256(status)
    rnode = build_node(
        node_type="RESULT", semantic_identity={"result_sha256": result_sha}, semantic_role="O7_RECONSTRUCTION_CASE_STATUS",
        retention_class="PINNED", status=status["status"],
        content_refs=[{"kind": "ARTIFACT", "sha256": srec["sha256"], "size_bytes": srec["size_bytes"]}],
        metadata={"known_missing_roles": status["known_missing_roles"], "forbidden_inferences": ["O7_GRADUATED"]},
    )
    rnode = graph.put_node(rnode)
    for n in roots.values():
        graph.add_edge(edge_type="REQUIRES", source_node_id=rnode["node_id"], target_node_id=n["node_id"], scope="O7_AVAILABLE_ROOT")
    review = {
        "schema_id": "IG_O7_POSTRUN_REVIEW_DECISION_V0_1",
        "status": "QUALIFIED_WITH_CERTIFICATION_DEBT",
        "scope": status["scientific_scope"],
        "forbidden_inferences": ["O7_GRADUATED", "AUTOMATIC_O8"],
    }
    review_node = register_review_decision(graph, review, semantic_role="O7_POSTRUN_REVIEW")
    graph.add_edge(edge_type="QUALIFIES", source_node_id=review_node["node_id"], target_node_id=rnode["node_id"], scope="O7_SCOPE_QUALIFICATION")
    graph.register_alias("ig:oscout/O7/reconstruction-status", "0.1", rnode["node_id"])
    return {
        "schema_id": "IG_O7_GRAPH_INGEST_RESULT_V0_1", "status": "PASS",
        "result_node_id": rnode["node_id"], "review_node_id": review_node["node_id"],
        "root_nodes": {k: v["node_id"] for k, v in roots.items()},
        "o7_graduated_claim_created": False,
    }
