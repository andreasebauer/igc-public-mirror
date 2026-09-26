from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .canon import canonical_sha256, write_json_atomic
from .datasets import DatasetStore
from .records import utc_now
from .store import ArtifactStore


NODE_TYPES = {
    "ARTIFACT", "DATASET", "SOURCE_SNAPSHOT", "ENVIRONMENT", "PROTOCOL",
    "QUESTION", "RECIPE", "RUN", "STAGE", "RESULT", "REVIEW_DECISION",
    "CLAIM", "RELEASE", "EXTERNAL_ROOT",
}
RETENTION_CLASSES = {"PINNED", "CACHE_EXPENSIVE", "CACHE", "EPHEMERAL"}
COMPUTATION_EDGE_TYPES = {
    "REQUIRES", "PRODUCED_BY", "MATERIALIZES", "DERIVED_FROM",
    "CHECKPOINT_OF", "SUPERSEDES", "REPLAYS",
}
EVIDENCE_EDGE_TYPES = {
    "SUPPORTS", "FALSIFIES", "QUALIFIES", "REQUIRES_CLAIM", "SEPARATES", "AUTHORIZES",
}
EDGE_TYPES = COMPUTATION_EDGE_TYPES | EVIDENCE_EDGE_TYPES
TRANSITIVE_COMPUTATION_EDGE_TYPES = {"REQUIRES", "PRODUCED_BY", "MATERIALIZES", "DERIVED_FROM", "REPLAYS"}


class GraphError(RuntimeError):
    pass


class GraphValidationError(GraphError):
    pass


class IdentityConflict(GraphError):
    pass


class AmbiguousAlias(GraphError):
    pass


class UnresolvedNode(GraphError):
    pass


class UnresolvedRoot(GraphError):
    pass


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _record_filename(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"


def _semantic_hash(node_type: str, schema_version: str, semantic_identity: dict[str, Any]) -> str:
    return canonical_sha256({
        "node_type": node_type,
        "schema_version": schema_version,
        "semantic_identity": semantic_identity,
    })


# Semantic identity is the immutable part of a graph node.  Keep its schema
# closed: adding an ignored key must never allow two different immutable
# identities to share the same node_id.  EXTERNAL_ROOT supports the two
# historical names for its content digest, but exactly one may be present.
_SEMANTIC_IDENTITY_KEYS: dict[str, frozenset[str]] = {
    "ARTIFACT": frozenset({"sha256"}),
    "DATASET": frozenset({"dataset_sha256"}),
    "SOURCE_SNAPSHOT": frozenset({"source_sha256"}),
    "ENVIRONMENT": frozenset({"environment_sha256"}),
    "PROTOCOL": frozenset({"protocol_id", "version", "descriptor_sha256"}),
    "QUESTION": frozenset({"question_sha256"}),
    "RECIPE": frozenset({"recipe_sha256"}),
    "RUN": frozenset({"run_id", "run_core_sha256"}),
    "STAGE": frozenset({"run_node_id", "stage_id", "stage_spec_sha256", "dependency_binding_sha256"}),
    "RESULT": frozenset({"result_sha256"}),
    "REVIEW_DECISION": frozenset({"decision_sha256"}),
    "CLAIM": frozenset({"claim_id", "version"}),
    "RELEASE": frozenset({"release_sha256"}),
}


def _validate_semantic_identity_shape(node_type: str, semantic_identity: dict[str, Any]) -> None:
    if node_type == "EXTERNAL_ROOT":
        keys = frozenset(semantic_identity)
        allowed = (
            frozenset({"content_sha256", "origin", "role"}),
            frozenset({"root_sha256", "origin", "role"}),
        )
        if keys not in allowed:
            raise GraphValidationError(
                "EXTERNAL_ROOT semantic_identity must contain exactly one content digest plus origin/role"
            )
        return
    expected = _SEMANTIC_IDENTITY_KEYS.get(node_type)
    if expected is None:
        raise GraphValidationError(f"no semantic identity schema for node type {node_type}")
    observed = frozenset(semantic_identity)
    if observed != expected:
        raise GraphValidationError(
            f"{node_type} semantic_identity fields mismatch: expected {sorted(expected)}, observed {sorted(observed)}"
        )


def compute_node_id(node_type: str, schema_version: str, semantic_identity: dict[str, Any]) -> str:
    if node_type not in NODE_TYPES:
        raise GraphValidationError(f"unknown node type {node_type}")
    if not isinstance(semantic_identity, dict) or not semantic_identity:
        raise GraphValidationError("semantic_identity must be a non-empty object")
    _validate_semantic_identity_shape(node_type, semantic_identity)
    if node_type == "ARTIFACT":
        sha = semantic_identity.get("sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("ARTIFACT semantic_identity.sha256 invalid")
        return f"artifact:sha256:{sha}"
    if node_type == "DATASET":
        sha = semantic_identity.get("dataset_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("DATASET semantic_identity.dataset_sha256 invalid")
        return f"dataset:sha256:{sha}"
    if node_type == "PROTOCOL":
        sha = semantic_identity.get("descriptor_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("PROTOCOL descriptor_sha256 invalid")
        return f"protocol:sha256:{sha}"
    if node_type == "QUESTION":
        sha = semantic_identity.get("question_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("QUESTION question_sha256 invalid")
        return f"question:sha256:{sha}"
    if node_type == "RECIPE":
        sha = semantic_identity.get("recipe_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("RECIPE recipe_sha256 invalid")
        return f"recipe:sha256:{sha}"
    if node_type == "SOURCE_SNAPSHOT":
        sha = semantic_identity.get("source_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("SOURCE_SNAPSHOT source_sha256 invalid")
        return f"source:sha256:{sha}"
    if node_type == "ENVIRONMENT":
        sha = semantic_identity.get("environment_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("ENVIRONMENT environment_sha256 invalid")
        return f"environment:sha256:{sha}"
    if node_type == "RESULT":
        sha = semantic_identity.get("result_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("RESULT result_sha256 invalid")
        return f"result:sha256:{sha}"
    if node_type == "REVIEW_DECISION":
        sha = semantic_identity.get("decision_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("REVIEW_DECISION decision_sha256 invalid")
        return f"review:sha256:{sha}"
    if node_type == "RELEASE":
        sha = semantic_identity.get("release_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("RELEASE release_sha256 invalid")
        return f"release:sha256:{sha}"
    if node_type == "CLAIM":
        cid = semantic_identity.get("claim_id")
        ver = semantic_identity.get("version")
        if not isinstance(cid, str) or not cid or not isinstance(ver, int) or ver < 1:
            raise GraphValidationError("CLAIM claim_id/version invalid")
        return f"claim:{cid}:v{ver:04d}"
    if node_type == "RUN":
        sha = semantic_identity.get("run_core_sha256")
        rid = semantic_identity.get("run_id")
        if not _is_sha256(sha) or not isinstance(rid, str) or not rid:
            raise GraphValidationError("RUN run identity invalid")
        return f"run:{rid}:sha256:{sha}"
    if node_type == "STAGE":
        required = ("run_node_id", "stage_id", "stage_spec_sha256", "dependency_binding_sha256")
        if any(k not in semantic_identity for k in required):
            raise GraphValidationError("STAGE semantic identity missing required field")
        if not isinstance(semantic_identity.get("run_node_id"), str) or not semantic_identity["run_node_id"]:
            raise GraphValidationError("STAGE run_node_id invalid")
        if not isinstance(semantic_identity.get("stage_id"), str) or not semantic_identity["stage_id"]:
            raise GraphValidationError("STAGE stage_id invalid")
        if not _is_sha256(semantic_identity.get("stage_spec_sha256")) or not _is_sha256(semantic_identity.get("dependency_binding_sha256")):
            raise GraphValidationError("STAGE digest invalid")
    if node_type == "EXTERNAL_ROOT":
        sha = semantic_identity.get("content_sha256") or semantic_identity.get("root_sha256")
        if not _is_sha256(sha):
            raise GraphValidationError("EXTERNAL_ROOT content hash invalid")
        if not isinstance(semantic_identity.get("origin"), str) or not semantic_identity.get("origin"):
            raise GraphValidationError("EXTERNAL_ROOT origin invalid")
        if not isinstance(semantic_identity.get("role"), str) or not semantic_identity.get("role"):
            raise GraphValidationError("EXTERNAL_ROOT role invalid")
    return f"{node_type.lower()}:sha256:{_semantic_hash(node_type, schema_version, semantic_identity)}"


def build_node(
    *, node_type: str, semantic_identity: dict[str, Any], semantic_role: str,
    retention_class: str, status: str = "VERIFIED", content_refs: list[dict] | None = None,
    producer_ref: str | None = None, provenance_refs: list[Any] | None = None,
    metadata: dict[str, Any] | None = None, schema_version: str = "0.1",
    created_utc: str | None = None,
) -> dict[str, Any]:
    node_id = compute_node_id(node_type, schema_version, semantic_identity)
    return {
        "schema_id": "IG_REPRO_GRAPH_NODE_V0_1",
        "node_id": node_id,
        "node_type": node_type,
        "schema_version": schema_version,
        "status": status,
        "created_utc": created_utc or utc_now(),
        "retention_class": retention_class,
        "semantic_role": semantic_role,
        "semantic_identity": semantic_identity,
        "content_refs": content_refs or [],
        "producer_ref": producer_ref,
        "provenance_refs": provenance_refs or [],
        "metadata": metadata or {},
    }


def _validate_semantic_identity_shape(node_type: str, semantic_identity: dict[str, Any]) -> None:
    """Require the exact frozen identity payload for every typed node.

    Node ids for content-addressed types intentionally use a readable primary
    digest (for example ``artifact:sha256:<digest>``).  Without an exact-key
    check, an archive could add identity-looking fields that are ignored by
    the readable id constructor.  That would create two different semantic
    payloads under one immutable node id.
    """
    exact = {
        "ARTIFACT": {"sha256"},
        "DATASET": {"dataset_sha256"},
        "SOURCE_SNAPSHOT": {"source_sha256"},
        "ENVIRONMENT": {"environment_sha256"},
        "PROTOCOL": {"protocol_id", "version", "descriptor_sha256"},
        "QUESTION": {"question_sha256"},
        "RECIPE": {"recipe_sha256"},
        "RUN": {"run_core_sha256", "run_id"},
        "STAGE": {"run_node_id", "stage_id", "stage_spec_sha256", "dependency_binding_sha256"},
        "RESULT": {"result_sha256"},
        "REVIEW_DECISION": {"decision_sha256"},
        "CLAIM": {"claim_id", "version"},
        "RELEASE": {"release_sha256"},
    }
    keys = set(semantic_identity)
    if node_type == "EXTERNAL_ROOT":
        allowed_a = {"content_sha256", "origin", "role"}
        allowed_b = {"root_sha256", "origin", "role"}
        if keys not in (allowed_a, allowed_b):
            raise GraphValidationError(
                f"EXTERNAL_ROOT semantic_identity fields invalid: {sorted(keys)}"
            )
        return
    expected = exact.get(node_type)
    if expected is None:
        raise GraphValidationError(f"unknown node type {node_type}")
    if keys != expected:
        raise GraphValidationError(
            f"{node_type} semantic_identity fields invalid: "
            f"expected {sorted(expected)}, observed {sorted(keys)}"
        )


def validate_node(node: dict[str, Any]) -> None:
    allowed = {
        "schema_id", "node_id", "node_type", "schema_version", "status", "created_utc",
        "retention_class", "semantic_role", "semantic_identity", "content_refs",
        "producer_ref", "provenance_refs", "metadata",
    }
    required = allowed - {"metadata"}
    unknown = set(node) - allowed
    missing = required - set(node)
    if unknown:
        raise GraphValidationError(f"unknown node fields: {sorted(unknown)}")
    if missing:
        raise GraphValidationError(f"missing node fields: {sorted(missing)}")
    if node.get("schema_id") != "IG_REPRO_GRAPH_NODE_V0_1":
        raise GraphValidationError("bad node schema_id")
    if node.get("node_type") not in NODE_TYPES:
        raise GraphValidationError("bad node_type")
    if node.get("retention_class") not in RETENTION_CLASSES:
        raise GraphValidationError("bad retention_class")
    if not isinstance(node.get("semantic_role"), str) or not node["semantic_role"]:
        raise GraphValidationError("semantic_role required")
    if not isinstance(node.get("status"), str) or not node["status"]:
        raise GraphValidationError("status required")
    if not isinstance(node.get("created_utc"), str) or not node["created_utc"]:
        raise GraphValidationError("created_utc required")
    if not isinstance(node.get("semantic_identity"), dict):
        raise GraphValidationError("semantic_identity must be object")
    _validate_semantic_identity_shape(node["node_type"], node["semantic_identity"])
    if not isinstance(node.get("content_refs"), list):
        raise GraphValidationError("content_refs must be list")
    if not isinstance(node.get("provenance_refs"), list):
        raise GraphValidationError("provenance_refs must be list")
    if node.get("producer_ref") is not None and not isinstance(node.get("producer_ref"), str):
        raise GraphValidationError("producer_ref must be node id or null")
    expected = compute_node_id(node["node_type"], str(node["schema_version"]), node["semantic_identity"])
    if node.get("node_id") != expected:
        raise GraphValidationError(f"node identity mismatch: {node.get('node_id')} != {expected}")
    for ref in node["content_refs"]:
        if not isinstance(ref, dict) or "kind" not in ref:
            raise GraphValidationError("invalid content_ref")
        if ref["kind"] == "ARTIFACT" and not _is_sha256(ref.get("sha256")):
            raise GraphValidationError("invalid ARTIFACT content_ref")
        if ref["kind"] == "DATASET" and not _is_sha256(ref.get("dataset_sha256")):
            raise GraphValidationError("invalid DATASET content_ref")
        if ref["kind"] not in {"ARTIFACT", "DATASET"}:
            raise GraphValidationError(f"unknown content_ref kind: {ref['kind']!r}")


def build_edge(
    *, edge_type: str, source_node_id: str, target_node_id: str,
    mandatory: bool = True, scope: str = "GLOBAL", metadata: dict[str, Any] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    if edge_type not in EDGE_TYPES:
        raise GraphValidationError(f"unknown edge type {edge_type}")
    identity = {
        "edge_type": edge_type,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "mandatory": bool(mandatory),
        "scope": scope,
        "metadata": metadata or {},
    }
    return {
        "schema_id": "IG_GRAPH_EDGE_V0_1",
        "edge_id": f"edge:sha256:{canonical_sha256(identity)}",
        **identity,
        "created_utc": created_utc or utc_now(),
    }


def validate_edge(edge: dict[str, Any]) -> None:
    allowed = {
        "schema_id", "edge_id", "edge_type", "source_node_id", "target_node_id",
        "mandatory", "scope", "metadata", "created_utc",
    }
    if set(edge) - allowed:
        raise GraphValidationError(f"unknown edge fields: {sorted(set(edge)-allowed)}")
    if allowed - set(edge):
        raise GraphValidationError(f"missing edge fields: {sorted(allowed-set(edge))}")
    if edge.get("schema_id") != "IG_GRAPH_EDGE_V0_1" or edge.get("edge_type") not in EDGE_TYPES:
        raise GraphValidationError("bad edge schema/type")
    expected = build_edge(
        edge_type=edge["edge_type"], source_node_id=edge["source_node_id"],
        target_node_id=edge["target_node_id"], mandatory=edge["mandatory"],
        scope=edge["scope"], metadata=edge.get("metadata") or {}, created_utc=edge["created_utc"],
    )["edge_id"]
    if edge.get("edge_id") != expected:
        raise GraphValidationError("edge identity mismatch")


def recipe_identity(recipe: dict[str, Any]) -> str:
    base = {k: v for k, v in recipe.items() if k not in {"recipe_sha256", "created_utc"}}
    return canonical_sha256(base)


def validate_recipe(recipe: dict[str, Any]) -> None:
    required = {
        "schema_id", "recipe_id", "recipe_version", "producer", "protocol_ref", "question_ref",
        "input_requirements", "environment_ref", "stages", "outputs", "canonicalization",
        "acceptance", "resource_policy", "replay_class", "recipe_sha256", "created_utc",
    }
    missing = required - set(recipe)
    if missing:
        raise GraphValidationError(f"missing recipe fields: {sorted(missing)}")
    if recipe.get("schema_id") != "IG_REPLAY_RECIPE_V0_1":
        raise GraphValidationError("bad recipe schema")
    if recipe.get("replay_class") not in {"EXACT_REPLAY", "INVARIANT_REPLAY", "DOCUMENTARY_ONLY"}:
        raise GraphValidationError("bad replay_class")
    if recipe_identity(recipe) != recipe.get("recipe_sha256"):
        raise GraphValidationError("recipe hash mismatch")
    forbidden = ("absolute_path", "hostname", "temporary_directory", "chat_id", "process_id")
    text = json.dumps(recipe, sort_keys=True)
    for key in forbidden:
        if f'"{key}"' in text:
            raise GraphValidationError(f"forbidden recipe identity field {key}")
    for req in recipe.get("input_requirements", []):
        if not isinstance(req, dict) or not req.get("semantic_role"):
            raise GraphValidationError("bad input requirement")
        exact = req.get("exact_identity_or_constraint", {})
        if not isinstance(exact, dict) or not exact.get("node_id"):
            raise GraphValidationError("Phase-1 exact recipe inputs require node_id")
        if "path" in exact or "absolute_path" in exact:
            raise GraphValidationError("path cannot satisfy recipe dependency")
    seen = set()
    for stage in recipe.get("stages", []):
        sid = stage.get("stage_id")
        if not sid or sid in seen:
            raise GraphValidationError("duplicate/missing recipe stage")
        if any(d not in seen for d in stage.get("dependencies", [])):
            raise GraphValidationError("recipe stages not topological")
        seen.add(sid)


def build_recipe(**kwargs: Any) -> dict[str, Any]:
    rec = {
        "schema_id": "IG_REPLAY_RECIPE_V0_1",
        "recipe_id": kwargs["recipe_id"],
        "recipe_version": kwargs.get("recipe_version", "1"),
        "producer": kwargs["producer"],
        "protocol_ref": kwargs["protocol_ref"],
        "question_ref": kwargs["question_ref"],
        "input_requirements": kwargs.get("input_requirements", []),
        "environment_ref": kwargs["environment_ref"],
        "stages": kwargs["stages"],
        "outputs": kwargs["outputs"],
        "canonicalization": kwargs.get("canonicalization", {"version": "IG_CANONICAL_JSON_V1"}),
        "acceptance": kwargs["acceptance"],
        "resource_policy": kwargs.get("resource_policy", {}),
        "replay_class": kwargs.get("replay_class", "EXACT_REPLAY"),
        "created_utc": kwargs.get("created_utc") or utc_now(),
    }
    rec["recipe_sha256"] = recipe_identity(rec)
    validate_recipe(rec)
    return rec


class GraphStore:
    def __init__(self, paths):
        self.paths = paths
        self.artifacts = ArtifactStore(paths.store)
        self.datasets = DatasetStore(self.artifacts)
        self.root = paths.store / "graph"
        self.nodes_dir = self.root / "nodes"
        self.edges_dir = self.root / "edges"
        self.recipes_dir = self.root / "recipes"
        self.aliases_dir = self.root / "aliases"
        self.proofs_dir = self.root / "reconstruction_proofs"
        self.imports_dir = self.root / "imports"
        for p in (self.nodes_dir, self.edges_dir, self.recipes_dir, self.aliases_dir, self.proofs_dir, self.imports_dir):
            p.mkdir(parents=True, exist_ok=True)

    def _node_path(self, node_id: str) -> Path:
        return self.nodes_dir / _record_filename(node_id)

    def _edge_path(self, edge_id: str) -> Path:
        return self.edges_dir / _record_filename(edge_id)

    def put_node(self, node: dict[str, Any]) -> dict[str, Any]:
        validate_node(node)
        p = self._node_path(node["node_id"])
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            validate_node(old)
            # Node ids intentionally authenticate semantic identity only, so
            # every other material field must be conflict-checked explicitly.
            material_fields=("node_id","node_type","schema_version","status","retention_class","semantic_role","semantic_identity","content_refs","producer_ref","provenance_refs","metadata")
            conflicts=[k for k in material_fields if old.get(k) != node.get(k)]
            if conflicts:
                raise IdentityConflict(f"node record conflict {node['node_id']}: {conflicts}")
            return old
        write_json_atomic(p, node)
        return node

    def get_node(self, node_id: str) -> dict[str, Any]:
        p = self._node_path(node_id)
        if not p.is_file():
            raise UnresolvedNode(node_id)
        node = json.loads(p.read_text(encoding="utf-8"))
        validate_node(node)
        if node["node_id"] != node_id:
            raise IdentityConflict("node path/identity mismatch")
        return node

    def list_nodes(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.nodes_dir.glob("*.json")):
            node = json.loads(p.read_text(encoding="utf-8"))
            validate_node(node)
            out.append(node)
        return out

    def put_edge(self, edge: dict[str, Any], *, require_endpoints: bool = True) -> dict[str, Any]:
        validate_edge(edge)
        if require_endpoints:
            self.get_node(edge["source_node_id"])
            self.get_node(edge["target_node_id"])
        p = self._edge_path(edge["edge_id"])
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            validate_edge(old)
            if old != edge:
                # Ignore created time only when semantically identical.
                a = {k: v for k, v in old.items() if k != "created_utc"}
                b = {k: v for k, v in edge.items() if k != "created_utc"}
                if a != b:
                    raise IdentityConflict(f"edge identity conflict {edge['edge_id']}")
            return old
        write_json_atomic(p, edge)
        return edge

    def add_edge(self, **kwargs: Any) -> dict[str, Any]:
        return self.put_edge(build_edge(**kwargs))

    def list_edges(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.edges_dir.glob("*.json")):
            edge = json.loads(p.read_text(encoding="utf-8"))
            validate_edge(edge)
            out.append(edge)
        return out

    def outgoing(self, node_id: str, edge_types: Iterable[str] | None = None) -> list[dict[str, Any]]:
        kinds = set(edge_types) if edge_types is not None else None
        return [e for e in self.list_edges() if e["source_node_id"] == node_id and (kinds is None or e["edge_type"] in kinds)]

    def incoming(self, node_id: str, edge_types: Iterable[str] | None = None) -> list[dict[str, Any]]:
        kinds = set(edge_types) if edge_types is not None else None
        return [e for e in self.list_edges() if e["target_node_id"] == node_id and (kinds is None or e["edge_type"] in kinds)]

    def put_recipe(self, recipe: dict[str, Any], *, retention_class: str = "PINNED") -> dict[str, Any]:
        validate_recipe(recipe)
        p = self.recipes_dir / f"{recipe['recipe_sha256']}.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if old != recipe:
                # created_utc can differ for independently constructed identical recipe.
                a = {k: v for k, v in old.items() if k != "created_utc"}
                b = {k: v for k, v in recipe.items() if k != "created_utc"}
                if a != b:
                    raise IdentityConflict("recipe identity conflict")
            recipe = old
        else:
            write_json_atomic(p, recipe)
        node = build_node(
            node_type="RECIPE", semantic_identity={"recipe_sha256": recipe["recipe_sha256"]},
            semantic_role=recipe["recipe_id"], retention_class=retention_class,
            status="VALIDATED", content_refs=[], metadata={"record_path_hint": f"graph/recipes/{p.name}"},
        )
        self.put_node(node)
        return recipe

    def get_recipe(self, recipe_sha256: str) -> dict[str, Any]:
        if not _is_sha256(recipe_sha256):
            raise ValueError("invalid recipe SHA")
        p = self.recipes_dir / f"{recipe_sha256}.json"
        if not p.is_file():
            raise UnresolvedNode(f"recipe:sha256:{recipe_sha256}")
        recipe = json.loads(p.read_text(encoding="utf-8"))
        validate_recipe(recipe)
        return recipe

    def register_alias(self, alias: str, version: str, node_id: str, *, status: str = "ACTIVE") -> dict[str, Any]:
        self.get_node(node_id)
        if not isinstance(alias, str) or not alias or alias.startswith("/") or "\\" in alias:
            raise GraphValidationError("invalid semantic alias")
        rec = {
            "schema_id": "IG_GRAPH_ALIAS_V0_1", "alias": alias, "version": str(version),
            "node_id": node_id, "status": status, "created_utc": utc_now(),
        }
        aid = canonical_sha256({k: rec[k] for k in ("alias", "version", "node_id", "status")})
        rec["alias_id"] = f"alias:sha256:{aid}"
        p = self.aliases_dir / _record_filename(rec["alias_id"])
        existing = [x for x in self.list_aliases() if x["alias"] == alias and x["version"] == str(version) and x["status"] == "ACTIVE"]
        if existing and any(x["node_id"] != node_id for x in existing):
            raise AmbiguousAlias(f"alias {alias}@{version} already resolves to another node")
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            return old
        write_json_atomic(p, rec)
        return rec

    def list_aliases(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.aliases_dir.glob("*.json")):
            o = json.loads(p.read_text(encoding="utf-8"))
            if o.get("schema_id") != "IG_GRAPH_ALIAS_V0_1":
                raise GraphValidationError("bad alias schema")
            expected = canonical_sha256({k:o[k] for k in ("alias","version","node_id","status")})
            if o.get("alias_id") != f"alias:sha256:{expected}" or p.name != _record_filename(o["alias_id"]):
                raise GraphValidationError("alias identity mismatch")
            out.append(o)
        return out

    def resolve_alias(self, alias: str, version: str | None = None) -> str:
        matches = [x for x in self.list_aliases() if x["alias"] == alias and x.get("status") == "ACTIVE"]
        if version is not None:
            matches = [x for x in matches if x["version"] == str(version)]
        if len(matches) != 1:
            if not matches:
                raise UnresolvedNode(f"alias:{alias}@{version or '*'}")
            raise AmbiguousAlias(f"alias {alias}@{version or '*'} has {len(matches)} active resolutions")
        self.get_node(matches[0]["node_id"])
        return matches[0]["node_id"]

    def resolve(self, reference: str, *, version: str | None = None) -> dict[str, Any]:
        # Paths are never dependency identities.
        if reference.startswith("/") or reference.startswith("./") or reference.startswith("../"):
            raise GraphValidationError("filesystem path cannot satisfy graph dependency; import by content first")
        if reference.startswith("sha256:"):
            sha = reference.split(":", 1)[1]
            candidates = [n for n in self.list_nodes() if any(r.get("sha256") == sha for r in n.get("content_refs", []))]
            if len(candidates) != 1:
                if not candidates:
                    raise UnresolvedNode(reference)
                raise AmbiguousAlias(f"content sha resolves to {len(candidates)} graph nodes")
            return candidates[0]
        try:
            return self.get_node(reference)
        except UnresolvedNode:
            return self.get_node(self.resolve_alias(reference, version=version))

    def register_artifact_node(
        self, artifact_record: dict[str, Any], *, semantic_role: str | None = None,
        retention_class: str = "PINNED", status: str = "VERIFIED",
        provenance_refs: list[Any] | None = None,
    ) -> dict[str, Any]:
        sha = artifact_record.get("sha256")
        v = self.artifacts.verify(sha, artifact_record.get("size_bytes"))
        if v.get("status") != "PASS":
            raise GraphValidationError(f"artifact blob not verified: {v}")
        node = build_node(
            node_type="ARTIFACT", semantic_identity={"sha256": sha},
            semantic_role=semantic_role or artifact_record.get("logical_role") or "ARTIFACT",
            retention_class=retention_class, status=status,
            content_refs=[{"kind": "ARTIFACT", "sha256": sha, "size_bytes": artifact_record.get("size_bytes")}],
            provenance_refs=provenance_refs or [],
            metadata={"media_type": artifact_record.get("media_type"), "source_name_hint": artifact_record.get("source_name")},
        )
        return self.put_node(node)

    def register_dataset_node(
        self, manifest: dict[str, Any], *, semantic_role: str | None = None,
        retention_class: str = "PINNED", status: str = "VERIFIED",
        provenance_refs: list[Any] | None = None,
    ) -> dict[str, Any]:
        dsid = manifest.get("dataset_sha256")
        v = self.datasets.verify(dsid)
        if v.get("status") != "PASS":
            raise GraphValidationError(f"dataset not verified: {v}")
        node = build_node(
            node_type="DATASET", semantic_identity={"dataset_sha256": dsid},
            semantic_role=semantic_role or manifest.get("logical_role") or "DATASET",
            retention_class=retention_class, status=status,
            content_refs=[{"kind": "DATASET", "dataset_sha256": dsid}],
            provenance_refs=provenance_refs or [],
            metadata={"canonicalization_version": manifest.get("canonicalization_version"), "format": manifest.get("format")},
        )
        out = self.put_node(node)
        # Explicit dataset -> shard artifact dependencies.
        for shard in manifest.get("shards", []):
            arec_path = self.paths.store / "artifacts" / f"{shard['sha256']}.json"
            if arec_path.is_file():
                arec = json.loads(arec_path.read_text(encoding="utf-8"))
            else:
                arec = {
                    "sha256": shard["sha256"], "size_bytes": shard["size_bytes"],
                    "logical_role": manifest.get("logical_role", "DATASET_SHARD"),
                    "source_name": shard.get("relative_path"), "media_type": "application/octet-stream",
                }
            an = self.register_artifact_node(arec, semantic_role="DATASET_SHARD", retention_class=retention_class)
            self.add_edge(edge_type="REQUIRES", source_node_id=out["node_id"], target_node_id=an["node_id"], mandatory=True, scope="DATASET_SHARD")
        return out

    def node_available(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        refs = node.get("content_refs", [])
        if node.get("node_type") == "RECIPE":
            recipe_sha = node.get("semantic_identity", {}).get("recipe_sha256")
            if not isinstance(recipe_sha, str) or not (self.recipes_dir / f"{recipe_sha}.json").is_file():
                return False
        # Content-bearing scientific nodes are not materialized merely because
        # their immutable graph record exists.  Without a verified content ref
        # a RESULT (or raw artifact/dataset/release) must be rebuilt from an
        # exact recipe or reported unresolved.  Metadata/control nodes may be
        # satisfied by their immutable graph record alone.
        if not refs and node.get("node_type") in {"ARTIFACT", "DATASET", "RESULT", "RELEASE"}:
            return False
        for ref in refs:
            kind = ref.get("kind")
            if kind == "ARTIFACT":
                if self.artifacts.verify(ref["sha256"], ref.get("size_bytes")).get("status") != "PASS":
                    return False
                # Content-bearing ARTIFACT semantic identity must name the same bytes.
                if node.get("node_type") == "ARTIFACT" and node.get("semantic_identity", {}).get("sha256") != ref.get("sha256"):
                    return False
            elif kind == "DATASET":
                if self.datasets.verify(ref["dataset_sha256"]).get("status") != "PASS":
                    return False
                if node.get("node_type") == "DATASET" and node.get("semantic_identity", {}).get("dataset_sha256") != ref.get("dataset_sha256"):
                    return False
            else:
                return False
        return True

    def validate_graph(self, *, targets: list[str] | None = None) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        nodes: dict[str, dict[str, Any]] = {}
        try:
            for n in self.list_nodes():
                if n["node_id"] in nodes:
                    failures.append({"reason": "duplicate_node", "node_id": n["node_id"]})
                nodes[n["node_id"]] = n
        except Exception as e:
            failures.append({"reason": "node_validation", "error": str(e)})
        edges = []
        try:
            edges = self.list_edges()
        except Exception as e:
            failures.append({"reason": "edge_validation", "error": str(e)})
        for e in edges:
            if e["source_node_id"] not in nodes or e["target_node_id"] not in nodes:
                failures.append({"reason": "dangling_edge", "edge_id": e["edge_id"]})
        # DAG check on replay-relevant computation dependencies.
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for e in edges:
            if e["edge_type"] in TRANSITIVE_COMPUTATION_EDGE_TYPES and e.get("mandatory", True):
                adj.setdefault(e["source_node_id"], []).append(e["target_node_id"])
        state: dict[str, int] = {}
        stack: list[str] = []
        def visit(v: str):
            state[v] = 1; stack.append(v)
            for w in adj.get(v, []):
                if state.get(w) == 1:
                    failures.append({"reason": "computation_cycle", "cycle_at": w, "stack": list(stack)})
                    continue
                if state.get(w, 0) == 0:
                    visit(w)
            stack.pop(); state[v] = 2
        for n in sorted(nodes):
            if state.get(n, 0) == 0:
                visit(n)
        # Alias ambiguity.
        aliases: dict[tuple[str, str], set[str]] = {}
        try:
            for a in self.list_aliases():
                if a.get("status") == "ACTIVE":
                    aliases.setdefault((a["alias"], a["version"]), set()).add(a["node_id"])
            for key, vals in aliases.items():
                if len(vals) != 1:
                    failures.append({"reason": "ambiguous_alias", "alias": key[0], "version": key[1], "nodes": sorted(vals)})
        except Exception as e:
            failures.append({"reason": "alias_validation", "error": str(e)})
        # Content availability/hash checks always cover every pinned root, plus
        # any explicit targets.  Supplying a target must not narrow doctor/graph
        # validation enough to hide corruption in another irreducible root.
        inspect = sorted(
            set(targets or [])
            | {n for n, o in nodes.items() if o.get("retention_class") == "PINNED"}
        )
        for nid in inspect:
            if nid not in nodes:
                failures.append({"reason": "target_missing", "node_id": nid}); continue
            try:
                available = self.node_available(nid)
            except Exception as e:
                failures.append({"reason": "content_validation", "node_id": nid, "error": str(e)}); continue
            if nodes[nid]["retention_class"] == "PINNED" and not available:
                failures.append({"reason": "pinned_content_missing", "node_id": nid})
        return {
            "schema_id": "IG_GRAPH_VALIDATION_RESULT_V0_1",
            "status": "PASS" if not failures else "FAIL",
            "nodes": len(nodes), "edges": len(edges), "aliases": sum(len(v) for v in aliases.values()),
            "failures": failures,
        }
