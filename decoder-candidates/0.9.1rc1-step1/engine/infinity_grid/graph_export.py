from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .graph_replay import computation_closure, evidence_closure, minimum_preservation_set
from .graph_store import GraphStore, IdentityConflict, build_edge
from .hashing import sha256_file
from .records import utc_now


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class GraphExportError(RuntimeError):
    pass


def _zip_deterministic(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            rel = p.relative_to(root).as_posix()
            info = zipfile.ZipInfo(rel, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())


def _write_internal_manifest(root: Path) -> dict[str, Any]:
    files = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "INTERNAL_MANIFEST.json"):
        files.append({"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    man = {"schema_id": "IG_GRAPH_EXPORT_INTERNAL_MANIFEST_V0_1", "files": files}
    write_json_atomic(root / "INTERNAL_MANIFEST.json", man)
    return man


def verify_graph_export_archive(archive: str | Path) -> dict[str, Any]:
    archive = Path(archive)
    failures = []
    try:
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                failures.append({"reason": "zip_crc", "member": bad})
            names = set(z.namelist())
            if "INTERNAL_MANIFEST.json" not in names or "GRAPH_EXPORT_MANIFEST.json" not in names:
                failures.append({"reason": "required_member_missing"})
                return {"status": "FAIL", "archive": str(archive), "failures": failures}
            man = json.loads(z.read("INTERNAL_MANIFEST.json"))
            expected = {r["path"] for r in man["files"]} | {"INTERNAL_MANIFEST.json"}
            if names != expected:
                failures.append({"reason": "inventory", "missing": sorted(expected - names), "extra": sorted(names - expected)})
            for rec in man["files"]:
                data = z.read(rec["path"])
                import hashlib
                sha = hashlib.sha256(data).hexdigest()
                if sha != rec["sha256"] or len(data) != rec["size_bytes"]:
                    failures.append({"reason": "member_mismatch", "path": rec["path"]})
            export = json.loads(z.read("GRAPH_EXPORT_MANIFEST.json"))
            base = {k: v for k, v in export.items() if k != "export_manifest_sha256"}
            if canonical_sha256(base) != export.get("export_manifest_sha256"):
                failures.append({"reason": "export_manifest_hash"})
    except Exception as e:
        failures.append({"reason": "exception", "error": str(e)})
    return {"status": "PASS" if not failures else "FAIL", "archive": str(archive), "sha256": sha256_file(archive) if archive.is_file() else None, "failures": failures}


class GraphExporter:
    def __init__(self, paths):
        self.paths = paths
        self.graph = GraphStore(paths)

    def _collect(self, targets: list[str], mode: str) -> dict[str, Any]:
        comp = computation_closure(self.graph, targets)
        if comp["status"] != "PASS":
            raise GraphExportError(f"computation closure unresolved: {comp['unresolved']}")
        nodes = set(comp["nodes"]); edges = set(comp["edges"])
        evidence = {"nodes": [], "edges": []}
        if mode == "PUBLICATION":
            for ref in targets:
                ev = evidence_closure(self.graph, ref)
                nodes.update(ev["nodes"]); edges.update(ev["edges"])
            evidence = {"nodes": sorted(set(nodes) - set(comp["nodes"])), "edges": sorted(set(edges) - set(comp["edges"]))}
        roots = minimum_preservation_set(self.graph, targets)
        return {"computation": comp, "evidence": evidence, "nodes": sorted(nodes), "edges": sorted(edges), "roots": roots}

    def create(self, targets: list[str], mode: str, destination: str | Path, *, scope: dict | None = None, forbidden_inferences: list[str] | None = None) -> dict[str, Any]:
        mode = mode.upper()
        if mode not in {"COMPACT", "STANDALONE", "PUBLICATION"}:
            raise ValueError("mode must be COMPACT, STANDALONE, or PUBLICATION")
        coll = self._collect(targets, mode)
        if mode == "STANDALONE" and coll["roots"]["status"] != "PASS":
            raise GraphExportError("standalone export requires a resolved minimum root set")
        tmp = Path(tempfile.mkdtemp(prefix="ig-graph-export-", dir=self.paths.workspace))
        try:
            (tmp / "graph/nodes").mkdir(parents=True)
            (tmp / "graph/edges").mkdir(parents=True)
            (tmp / "graph/recipes").mkdir(parents=True)
            (tmp / "graph/aliases").mkdir(parents=True)
            (tmp / "objects/sha256").mkdir(parents=True)
            (tmp / "datasets").mkdir(parents=True)
            node_records = []
            external_refs = []
            include_objects_for = set(coll["nodes"])
            if mode in {"COMPACT", "PUBLICATION"}:
                # Pinned roots may be referenced externally; target bytes and rebuildable/cache bytes stay included when present.
                target_ids = {self.graph.resolve(t)["node_id"] for t in targets}
                include_objects_for = {
                    nid for nid in coll["nodes"]
                    if nid in target_ids or self.graph.get_node(nid)["retention_class"] != "PINNED"
                }
            for nid in coll["nodes"]:
                node = self.graph.get_node(nid)
                p = self.graph._node_path(nid)
                shutil.copy2(p, tmp / "graph/nodes" / p.name)
                node_records.append(nid)
                if nid not in include_objects_for:
                    for ref in node.get("content_refs", []):
                        external_refs.append({"node_id": nid, **ref})
                    continue
                for ref in node.get("content_refs", []):
                    if ref["kind"] == "ARTIFACT":
                        src = self.graph.artifacts.blob_path(ref["sha256"])
                        if not src.is_file():
                            if mode == "STANDALONE": raise GraphExportError(f"missing standalone object {ref['sha256']}")
                            external_refs.append({"node_id": nid, **ref}); continue
                        dst = tmp / "objects/sha256" / ref["sha256"][:2] / ref["sha256"]
                        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
                    elif ref["kind"] == "DATASET":
                        manifest = self.graph.datasets.load(ref["dataset_sha256"])
                        mp = self.paths.store / "datasets" / f"{ref['dataset_sha256']}.json"
                        shutil.copy2(mp, tmp / "datasets" / mp.name)
                        for shard in manifest["shards"]:
                            src = self.graph.artifacts.blob_path(shard["sha256"])
                            if not src.is_file():
                                if mode == "STANDALONE": raise GraphExportError(f"missing dataset shard {shard['sha256']}")
                                external_refs.append({"node_id": nid, "kind": "ARTIFACT", "sha256": shard["sha256"], "size_bytes": shard["size_bytes"]}); continue
                            dst = tmp / "objects/sha256" / shard["sha256"][:2] / shard["sha256"]
                            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            for eid in coll["edges"]:
                p = self.graph._edge_path(eid); shutil.copy2(p, tmp / "graph/edges" / p.name)
            # Include every recipe referenced by included recipe nodes.
            recipe_shas = []
            for nid in coll["nodes"]:
                n = self.graph.get_node(nid)
                if n["node_type"] == "RECIPE":
                    sha = n["semantic_identity"]["recipe_sha256"]; recipe_shas.append(sha)
                    shutil.copy2(self.graph.recipes_dir / f"{sha}.json", tmp / "graph/recipes" / f"{sha}.json")
            # Alias snapshot for included nodes.
            aliases = [a for a in self.graph.list_aliases() if a["node_id"] in set(coll["nodes"])]
            for a in aliases:
                src = self.graph.aliases_dir / (self.graph.aliases_dir / "x").name  # placeholder overwritten below
                # Find exact stored record by alias_id.
                candidates = [p for p in self.graph.aliases_dir.glob("*.json") if json.loads(p.read_text()).get("alias_id") == a["alias_id"]]
                if len(candidates) == 1: shutil.copy2(candidates[0], tmp / "graph/aliases" / candidates[0].name)
            replay = {
                "schema_id": "IG_GRAPH_EXPORT_REPLAY_INSTRUCTIONS_V0_1",
                "targets": [self.graph.resolve(t)["node_id"] for t in targets],
                "commands": [f"ig --root <ROOT> import graph <ARCHIVE>", *[f"ig --root <ROOT> replay '{self.graph.resolve(t)['node_id']}' --clean" for t in targets]],
                "offline": mode == "STANDALONE",
            }
            write_json_atomic(tmp / "REPLAY_INSTRUCTIONS.json", replay)
            evidence_scope = {
                "schema_id": "IG_GRAPH_EXPORT_EVIDENCE_SCOPE_V0_1", "mode": mode,
                "scope": scope or {"authority": "UNCHANGED_FROM_SOURCE_GRAPH"},
                "forbidden_inferences": forbidden_inferences or [],
                "computation_edges_do_not_imply_support": True,
            }
            write_json_atomic(tmp / "EVIDENCE_SCOPE.json", evidence_scope)
            export = {
                "schema_id": "IG_GRAPH_EXPORT_MANIFEST_V0_1", "mode": mode,
                "created_utc": utc_now(), "targets": [self.graph.resolve(t)["node_id"] for t in targets],
                "nodes": node_records, "edges": coll["edges"], "recipes": recipe_shas,
                "aliases": [{"alias": a["alias"], "version": a["version"], "node_id": a["node_id"]} for a in aliases],
                "external_root_references": sorted(external_refs, key=lambda x: (x["node_id"], x.get("sha256", x.get("dataset_sha256", "")))),
                "minimum_preservation": coll["roots"], "computation_closure_status": coll["computation"]["status"],
                "evidence_closure": coll["evidence"], "standalone": mode == "STANDALONE",
            }
            export["export_manifest_sha256"] = canonical_sha256(export)
            write_json_atomic(tmp / "GRAPH_EXPORT_MANIFEST.json", export)
            _write_internal_manifest(tmp)
            destination = Path(destination)
            _zip_deterministic(tmp, destination)
            verify = verify_graph_export_archive(destination)
            if verify["status"] != "PASS":
                raise GraphExportError(f"created export failed verification {verify}")
            return {"schema_id": "IG_GRAPH_EXPORT_RESULT_V0_1", "status": "PASS", "mode": mode, "archive": str(destination), "archive_sha256": sha256_file(destination), "manifest_sha256": export["export_manifest_sha256"], "nodes": len(node_records), "edges": len(coll["edges"]), "external_root_references": len(external_refs)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class GraphImporter:
    def __init__(self, paths):
        self.paths = paths; self.graph = GraphStore(paths)

    def import_archive(self, archive: str | Path) -> dict[str, Any]:
        archive = Path(archive)
        verify = verify_graph_export_archive(archive)
        if verify["status"] != "PASS":
            raise GraphExportError(f"archive verification failed: {verify}")
        tmp = Path(tempfile.mkdtemp(prefix="ig-graph-import-", dir=self.paths.workspace))
        try:
            with zipfile.ZipFile(archive) as z: z.extractall(tmp)
            manifest = json.loads((tmp / "GRAPH_EXPORT_MANIFEST.json").read_text())
            # Publish raw objects first.
            for p in sorted((tmp / "objects/sha256").glob("*/*")):
                expected = p.name
                if sha256_file(p) != expected: raise GraphExportError("object path/hash mismatch")
                self.graph.artifacts.put_file(p, logical_role="GRAPH_IMPORT_OBJECT", source_name=expected)
            # Dataset manifests are immutable and validated against imported shards.
            for p in sorted((tmp / "datasets").glob("*.json")):
                obj = json.loads(p.read_text()); dsid = obj["dataset_sha256"]
                from .canon import canonical_sha256
                if canonical_sha256({k:v for k,v in obj.items() if k!="dataset_sha256"}) != dsid: raise GraphExportError("dataset manifest hash mismatch")
                for s in obj.get("shards", []):
                    if self.graph.artifacts.verify(s["sha256"], s["size_bytes"])["status"] != "PASS": raise GraphExportError("dataset shard unavailable")
                dest = self.paths.store / "datasets" / p.name
                if dest.exists() and json.loads(dest.read_text()) != obj: raise IdentityConflict("dataset immutable conflict")
                if not dest.exists(): write_json_atomic(dest, obj)
            # Nodes, recipes, aliases, edges through strict stores.
            for p in sorted((tmp / "graph/nodes").glob("*.json")):
                self.graph.put_node(json.loads(p.read_text()))
            for p in sorted((tmp / "graph/recipes").glob("*.json")):
                self.graph.put_recipe(json.loads(p.read_text()))
            for p in sorted((tmp / "graph/aliases").glob("*.json")):
                o=json.loads(p.read_text()); self.graph.register_alias(o["alias"],o["version"],o["node_id"],status=o.get("status","ACTIVE"))
            for p in sorted((tmp / "graph/edges").glob("*.json")):
                self.graph.put_edge(json.loads(p.read_text()))
            validation = self.graph.validate_graph(targets=manifest.get("targets"))
            # Compact imports can have unavailable externally referenced pinned roots; referential graph must still validate.
            if validation["status"] != "PASS":
                non_external = [f for f in validation["failures"] if f.get("reason") not in {"pinned_content_missing"}]
                external_nodes = {x["node_id"] for x in manifest.get("external_root_references", [])}
                non_external += [f for f in validation["failures"] if f.get("reason") == "pinned_content_missing" and f.get("node_id") not in external_nodes]
                if non_external: raise GraphExportError(f"imported graph invalid: {non_external}")
            rec = {"schema_id":"IG_GRAPH_IMPORT_RECORD_V0_1","archive_sha256":sha256_file(archive),"manifest_sha256":manifest["export_manifest_sha256"],"mode":manifest["mode"],"status":"PASS","created_utc":utc_now()}
            write_json_atomic(self.graph.imports_dir/f"{rec['archive_sha256']}.json",rec)
            return {"status":"PASS","archive_sha256":rec["archive_sha256"],"mode":manifest["mode"],"nodes":len(manifest.get("nodes",[])),"edges":len(manifest.get("edges",[])),"external_refs":len(manifest.get("external_root_references",[]))}
        finally:
            shutil.rmtree(tmp,ignore_errors=True)
