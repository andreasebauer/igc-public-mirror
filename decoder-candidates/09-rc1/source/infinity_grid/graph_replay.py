from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from . import __version__
from .canon import canonical_sha256, write_json_atomic
from .controller import Controller
from .datasets import DatasetStore
from .graph_store import (
    GraphError, GraphStore, GraphValidationError, UnresolvedNode, UnresolvedRoot,
    TRANSITIVE_COMPUTATION_EDGE_TYPES, build_node,
)
from .hashing import sha256_file
from .paths import resolve_root
from .records import runtime_descriptor, runtime_sha256, source_sha256, utc_now
from .store import ArtifactStore
import infinity_grid.adapters  # noqa: F401 -- register existing runtime runners


SUPPORTED_CANONICALIZATION = {"IG_CANONICAL_JSON_V1", "FILESET_V1"}


class ReplayError(GraphError):
    pass


class EnvironmentMismatch(ReplayError):
    pass


class CanonicalizationMismatch(ReplayError):
    pass


class ReplayPlanner:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def _recipe_for_target(self, node_id: str) -> dict[str, Any] | None:
        edges = self.graph.outgoing(node_id, {"PRODUCED_BY"})
        recipes = []
        for e in edges:
            n = self.graph.get_node(e["target_node_id"])
            if n["node_type"] == "RECIPE":
                recipes.append(self.graph.get_recipe(n["semantic_identity"]["recipe_sha256"]))
        if len(recipes) > 1:
            raise ReplayError(f"ambiguous exact replay recipes for {node_id}")
        return recipes[0] if recipes else None

    def plan(self, target_node_id: str, *, force_rebuild: bool = False) -> dict[str, Any]:
        with self.graph.live_read() as graph:
            return ReplayPlanner(graph)._plan_live(target_node_id, force_rebuild=force_rebuild)

    def _plan_live(self, target_node_id: str, *, force_rebuild: bool = False) -> dict[str, Any]:
        self.graph.get_node(target_node_id)
        steps: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def walk(nid: str, force: bool = False):
            if nid in done:
                return
            if nid in visiting:
                raise ReplayError(f"recipe dependency cycle at {nid}")
            visiting.add(nid)
            node = self.graph.get_node(nid)
            available = self.graph.node_available(nid)
            if available and not force:
                steps.append({"action": "USE_LOCAL", "node_id": nid})
                done.add(nid); visiting.remove(nid); return
            recipe = self._recipe_for_target(nid)
            if recipe is None:
                if node["retention_class"] == "PINNED":
                    unresolved.append({"node_id": nid, "reason": "MISSING_IRREDUCIBLE_ROOT"})
                else:
                    unresolved.append({"node_id": nid, "reason": "NO_VALIDATED_REPLAY_RECIPE"})
                visiting.remove(nid); return
            if recipe.get("replay_class") != "EXACT_REPLAY":
                unresolved.append({"node_id": nid, "reason": "RECIPE_NOT_EXACT", "recipe_sha256": recipe["recipe_sha256"]})
                visiting.remove(nid); return
            for req in recipe.get("input_requirements", []):
                walk(req["exact_identity_or_constraint"]["node_id"], False)
            # Producer/protocol/question/environment are explicit graph dependencies too.
            for dep in [recipe["producer"]["source_snapshot_ref"], recipe["protocol_ref"], recipe["question_ref"], recipe["environment_ref"]]:
                walk(dep, False)
            steps.append({"action": "REPLAY", "node_id": nid, "recipe_sha256": recipe["recipe_sha256"]})
            done.add(nid); visiting.remove(nid)

        walk(target_node_id, force_rebuild)
        return {
            "schema_id": "IG_GRAPH_REPLAY_PLAN_V0_1",
            "target_node_id": target_node_id,
            "force_rebuild": force_rebuild,
            "steps": steps,
            "unresolved": unresolved,
            "status": "PASS" if not unresolved else "UNRESOLVED",
            "plan_sha256": canonical_sha256({"target_node_id": target_node_id, "force_rebuild": force_rebuild, "steps": steps, "unresolved": unresolved}),
        }


class GraphReplayEngine:
    def __init__(self, paths):
        self.paths = paths
        self.graph = GraphStore(paths)
        self.planner = ReplayPlanner(self.graph)

    def _check_recipe_context(self, recipe: dict[str, Any]) -> None:
        can = recipe.get("canonicalization", {}).get("version")
        if can not in SUPPORTED_CANONICALIZATION:
            raise CanonicalizationMismatch(f"unsupported canonicalization {can}")
        env_node = self.graph.get_node(recipe["environment_ref"])
        if env_node["node_type"] != "ENVIRONMENT":
            raise EnvironmentMismatch("environment_ref is not ENVIRONMENT node")
        expected = env_node["semantic_identity"]["environment_sha256"]
        observed = runtime_sha256()
        if expected != observed:
            raise EnvironmentMismatch(f"environment mismatch: expected {expected}, observed {observed}")
        src_node = self.graph.get_node(recipe["producer"]["source_snapshot_ref"])
        if src_node["node_type"] != "SOURCE_SNAPSHOT":
            raise EnvironmentMismatch("source_snapshot_ref is not SOURCE_SNAPSHOT node")
        expected_src = src_node["semantic_identity"]["source_sha256"]
        observed_src = source_sha256()
        if expected_src != observed_src:
            raise EnvironmentMismatch(f"source mismatch: expected {expected_src}, observed {observed_src}")

    def _materialize_node(self, node_id: str, destination: Path) -> Path:
        node = self.graph.get_node(node_id)
        refs = node.get("content_refs", [])
        if not refs:
            destination.mkdir(parents=True, exist_ok=True)
            write_json_atomic(destination / "NODE_RECORD.json", node)
            return destination
        if len(refs) != 1:
            raise ReplayError(f"Phase-1 materialization requires one primary content ref for {node_id}")
        ref = refs[0]
        if ref["kind"] == "ARTIFACT":
            destination.parent.mkdir(parents=True, exist_ok=True)
            return self.graph.artifacts.materialize(ref["sha256"], destination, expected_size=ref.get("size_bytes"))
        if ref["kind"] == "DATASET":
            return self.graph.datasets.materialize(ref["dataset_sha256"], destination)
        raise ReplayError(f"unsupported content ref {ref}")

    def _prepare_inputs(self, recipe: dict[str, Any], workspace: Path) -> dict[str, Path]:
        inputs: dict[str, Path] = {}
        for idx, req in enumerate(recipe.get("input_requirements", [])):
            role = req["semantic_role"]
            nid = req["exact_identity_or_constraint"]["node_id"]
            node = self.graph.get_node(nid)
            suffix = ".bin" if node["node_type"] == "ARTIFACT" else ""
            rel = req.get("materialize_as") or f"inputs/{idx:02d}_{role}{suffix}"
            dest = workspace / rel
            inputs[role] = self._materialize_node(nid, dest)
        return inputs

    @staticmethod
    def _expand_arg(value: str, *, workspace: Path, inputs: dict[str, Path], outputs: dict[str, Path]) -> str:
        out = value.replace("{workspace}", str(workspace))
        for k, v in inputs.items():
            out = out.replace("{input:" + k + "}", str(v))
        for k, v in outputs.items():
            out = out.replace("{output:" + k + "}", str(v))
        return out

    def _run_python_script(self, action: dict, *, workspace: Path, inputs: dict[str, Path], outputs: dict[str, Path]) -> dict:
        role = action["script_input_role"]
        script = inputs[role]
        args = [self._expand_arg(str(x), workspace=workspace, inputs=inputs, outputs=outputs) for x in action.get("args", [])]
        env = dict(os.environ); env.update({str(k): self._expand_arg(str(v), workspace=workspace, inputs=inputs, outputs=outputs) for k, v in action.get("env", {}).items()})
        proc = subprocess.run([sys.executable, str(script), *args], cwd=str(workspace), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=int(action.get("timeout_seconds", 600)), check=False)
        (workspace / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (workspace / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise ReplayError(f"python stage failed rc={proc.returncode}: {proc.stderr[-2000:]}")
        return {"returncode": proc.returncode, "stdout_sha256": canonical_sha256(proc.stdout), "stderr_sha256": canonical_sha256(proc.stderr)}

    def _run_unified_plan(self, action: dict, *, workspace: Path, inputs: dict[str, Path], outputs: dict[str, Path]) -> dict:
        plan = json.loads(inputs[action["plan_input_role"]].read_text(encoding="utf-8"))
        dataset_role = action["dataset_input_role"]
        dataset_dir = inputs[dataset_role]
        rt = resolve_root(workspace / "runtime_root").ensure()
        # Install executable protocol descriptors into the clean root.
        from .protocols import ProtocolRegistry
        ProtocolRegistry().install_into_store(rt)
        frozen_manifest = json.loads(inputs[action["dataset_manifest_input_role"]].read_text(encoding="utf-8"))
        observed = DatasetStore(ArtifactStore(rt.store)).import_directory(
            dataset_dir,
            logical_role=frozen_manifest["logical_role"],
            canonicalization_version=frozen_manifest.get("canonicalization_version", "FILESET_V1"),
        )
        expected_dsid = plan["input_datasets"][0]["dataset_sha256"]
        if observed["dataset_sha256"] != expected_dsid:
            raise ReplayError(f"dataset identity mismatch {observed['dataset_sha256']} != {expected_dsid}")
        mode = json.loads(inputs[action["execution_mode_input_role"]].read_text(encoding="utf-8"))
        rr = Controller(rt).run(plan, execution_mode=mode)
        by_name = {a.get("logical_name"): a for a in rr.get("result_artifacts", [])}
        for role, out_path in outputs.items():
            logical = action.get("output_logical_names", {}).get(role, role)
            if logical not in by_name:
                raise ReplayError(f"run did not produce {logical}")
            art = by_name[logical]
            ArtifactStore(rt.store).materialize(art["sha256"], out_path, expected_size=art.get("size_bytes"))
        return {"run_id": rr["run_id"], "lifecycle": rr["lifecycle"], "run_core_sha256": rr.get("run_core_sha256")}

    def _run_standalone_capsule(self, action: dict, *, workspace: Path, inputs: dict[str, Path], outputs: dict[str, Path]) -> dict:
        cap = inputs[action["capsule_input_role"]]
        extract = workspace / "capsule"
        with zipfile.ZipFile(cap) as z:
            z.testzip()
            z.extractall(extract)
        verify = subprocess.run([sys.executable, str(extract / "verify_release.py")], cwd=str(extract), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)
        if verify.returncode != 0:
            raise ReplayError(f"standalone capsule verification failed: {verify.stderr}")
        # The capsule carries the historical exact wheel. Create an isolated venv and replay offline.
        venv = workspace / "venv"
        py = Path("/usr/bin/python") if Path("/usr/bin/python").is_file() else Path(sys.executable)
        subprocess.run([str(py), "-m", "venv", str(venv)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True)
        vpy = venv / "bin" / "python"
        wheels = list((extract / "wheelhouse").glob("*.whl"))
        if len(wheels) != 1:
            raise ReplayError("standalone capsule wheelhouse is ambiguous")
        install = subprocess.run([str(vpy), "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])], cwd=str(extract), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
        if install.returncode != 0:
            raise ReplayError(f"offline wheel install failed: {install.stderr}")
        replay = subprocess.run([str(vpy), str(extract / "reproduce.py")], cwd=str(extract), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=int(action.get("timeout_seconds", 1200)), check=False)
        if replay.returncode != 0:
            raise ReplayError(f"standalone reproduction failed: {replay.stderr[-4000:]}")
        try:
            parsed = json.loads(replay.stdout.strip().splitlines()[-1])
        except Exception as e:
            raise ReplayError(f"standalone replay output not JSON: {e}")
        if parsed.get("status") != "PASS":
            raise ReplayError(f"standalone replay reported {parsed}")
        att = {
            "schema_id": "IG_GRAPH_STANDALONE_REPLAY_ATTESTATION_V0_1",
            "status": "PASS", "capsule_sha256": sha256_file(cap),
            "verification": json.loads(verify.stdout.strip().splitlines()[-1]),
            "reproduction": parsed,
        }
        for role, out_path in outputs.items():
            write_json_atomic(out_path, att)
        return {"capsule_sha256": att["capsule_sha256"], "status": "PASS"}


    @staticmethod
    def _safe_extract_archive(archive: Path, destination: Path) -> Path:
        """Extract a frozen jump-start ZIP without accepting path traversal or symlinks."""
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad is not None:
                raise ReplayError(f"archive CRC failure: {bad}")
            for info in z.infolist():
                name = info.filename.replace("\\", "/")
                pp = Path(name)
                if pp.is_absolute() or ".." in pp.parts:
                    raise ReplayError(f"unsafe archive member {name}")
                # Unix symlink file type in external attributes.
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ReplayError(f"symlink forbidden in replay archive: {name}")
            z.extractall(destination)
        children = [x for x in destination.iterdir() if x.name != "__MACOSX"]
        dirs = [x for x in children if x.is_dir()]
        files = [x for x in children if x.is_file()]
        if len(dirs) == 1 and not files:
            return dirs[0]
        return destination

    def _run_archive_command(self, action: dict, *, workspace: Path, inputs: dict[str, Path], outputs: dict[str, Path]) -> dict:
        archive = inputs[action["archive_input_role"]]
        extract = workspace / "archive_payload"
        work_root = self._safe_extract_archive(archive, extract)
        relroot = action.get("work_root_relative_path")
        if relroot:
            work_root = (work_root / relroot).resolve()
            try:
                work_root.relative_to(extract.resolve())
            except ValueError:
                raise ReplayError("archive work root escaped extraction directory")
        raw_argv = list(action.get("argv", []))
        if not raw_argv:
            raise ReplayError("ARCHIVE_COMMAND_REPLAY requires argv")
        argv = []
        for token in raw_argv:
            token = self._expand_arg(str(token), workspace=workspace, inputs=inputs, outputs=outputs)
            argv.append(sys.executable if token == "{python}" else token)
        env = dict(os.environ)
        env.update({str(k): self._expand_arg(str(v), workspace=workspace, inputs=inputs, outputs=outputs).replace("{work_root}", str(work_root)) for k, v in action.get("env", {}).items()})
        stdout_path = workspace / "archive.stdout.txt"
        stderr_path = workspace / "archive.stderr.txt"
        # Write directly to files rather than PIPE. Historical replay scripts may spawn
        # helper processes that inherit stdout/stderr; PIPE capture can otherwise wait for
        # grandchildren after the authoritative parent has already finished.
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.Popen(argv, cwd=str(work_root), env=env, text=True, stdout=stdout_f, stderr=stderr_f, start_new_session=True)
            try:
                returncode = proc.wait(timeout=int(action.get("timeout_seconds", 1200)))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 15)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, 9)
                    except ProcessLookupError:
                        pass
                raise ReplayError(f"archive replay command timed out: {argv}")
            finally:
                # Some historical drivers leave helper processes alive after writing the
                # authoritative result. They are not part of the replay product and must not
                # escape the isolated replay stage or keep automation sessions open.
                try:
                    os.killpg(proc.pid, 15)
                except ProcessLookupError:
                    pass
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if returncode != 0:
            raise ReplayError(f"archive replay command failed rc={returncode}: {stderr_text[-4000:]}")
        for rel in action.get("required_files", []):
            if not (work_root / rel).is_file():
                raise ReplayError(f"archive replay required file missing: {rel}")
        src = work_root / action["result_relative_path"]
        if not src.is_file():
            raise ReplayError(f"archive replay result missing: {action['result_relative_path']}")
        dst = outputs[action["output_role"]]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {
            "returncode": returncode,
            "work_root_relative": str(work_root.relative_to(extract)),
            "result_relative_path": action["result_relative_path"],
            "result_sha256": sha256_file(src),
            "stdout_sha256": canonical_sha256(stdout_text),
            "stderr_sha256": canonical_sha256(stderr_text),
        }

    def _execute_recipe(self, recipe: dict[str, Any], *, workspace: Path) -> dict[str, Any]:
        self._check_recipe_context(recipe)
        inputs = self._prepare_inputs(recipe, workspace)
        output_paths: dict[str, Path] = {}
        for out in recipe.get("outputs", []):
            role = out["semantic_role"]
            rel = out.get("relative_path") or f"outputs/{role}.bin"
            p = workspace / rel; p.parent.mkdir(parents=True, exist_ok=True)
            output_paths[role] = p
        stage_results = []
        for stage in recipe.get("stages", []):
            action = stage["action"]
            kind = action.get("type")
            t0 = time.perf_counter()
            if kind == "COPY_INPUT":
                src = inputs[action["input_role"]]
                dst = output_paths[action["output_role"]]
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                res = {"copied": True}
            elif kind == "PYTHON_SCRIPT":
                res = self._run_python_script(action, workspace=workspace, inputs=inputs, outputs=output_paths)
            elif kind == "UNIFIED_RUNTIME_PLAN":
                res = self._run_unified_plan(action, workspace=workspace, inputs=inputs, outputs=output_paths)
            elif kind == "STANDALONE_CAPSULE_REPLAY":
                res = self._run_standalone_capsule(action, workspace=workspace, inputs=inputs, outputs=output_paths)
            elif kind == "ARCHIVE_COMMAND_REPLAY":
                res = self._run_archive_command(action, workspace=workspace, inputs=inputs, outputs=output_paths)
            else:
                raise ReplayError(f"unsupported recipe action type {kind}")
            stage_results.append({"stage_id": stage["stage_id"], "action_type": kind, "wall_seconds": time.perf_counter() - t0, "result": res})
        produced = {}
        expected = recipe.get("acceptance", {}).get("expected_hashes", {})
        for out in recipe.get("outputs", []):
            role = out["semantic_role"]; path = output_paths[role]
            if not path.is_file():
                raise ReplayError(f"expected output missing: {role}")
            sha = sha256_file(path)
            exp = expected.get(role) or out.get("expected_sha256")
            if exp and sha != exp:
                raise ReplayError(f"output hash mismatch for {role}: {sha} != {exp}")
            rec = self.graph.artifacts.put_file(path, logical_role=role, source_name=path.name)
            node_id = out["node_id"]
            node = self.graph.get_node(node_id)
            binding_ok = False
            if node["node_type"] == "ARTIFACT" and node["semantic_identity"].get("sha256") == sha:
                binding_ok = True
            elif node["node_type"] == "RESULT" and node["semantic_identity"].get("result_sha256") == sha:
                refs = [r for r in node.get("content_refs", []) if r.get("kind") == "ARTIFACT"]
                binding_ok = any(r.get("sha256") == sha for r in refs)
            if not binding_ok:
                raise ReplayError(f"output node binding mismatch for {role}")
            produced[role] = {"node_id": node_id, "sha256": sha, "size_bytes": rec["size_bytes"]}
        return {"stage_results": stage_results, "outputs": produced}

    def replay(self, target_reference: str, *, force_rebuild: bool = False, workspace: str | Path | None = None) -> dict[str, Any]:
        target = self.graph.resolve(target_reference)["node_id"]
        plan = self.planner.plan(target, force_rebuild=force_rebuild)
        if plan["status"] != "PASS":
            raise UnresolvedRoot(json.dumps(plan["unresolved"], sort_keys=True))
        ws = Path(workspace) if workspace is not None else self.paths.workspace / f"graph-replay-{canonical_sha256({'target': target, 'plan': plan['plan_sha256']})[:16]}"
        ws = ws.resolve(strict=False)
        designated = self.paths.workspace.resolve()
        project_root = Path(getattr(self.paths, "root", designated.parent)).resolve()
        explicit = workspace is not None
        if explicit:
            # Never allow a destructive replay workspace to equal/contain the
            # decoder root, or to live in an internal project directory other
            # than the dedicated workspace tree.
            if ws == project_root or project_root.is_relative_to(ws):
                raise ReplayError("workspace may not equal or contain the decoder root")
            if ws.is_relative_to(project_root) and not ws.is_relative_to(designated):
                raise ReplayError("workspace inside decoder root must be below dedicated workspace")
        protected_roots = [self.paths.store.resolve(), self.paths.runs.resolve(), self.paths.catalog.resolve()]
        if hasattr(self.paths, "releases"):
            protected_roots.append(self.paths.releases.resolve())
        for protected in protected_roots:
            if ws.is_relative_to(protected):
                raise ReplayError("workspace may not be inside immutable/project-record roots")
        shutil.rmtree(ws, ignore_errors=False); ws.mkdir(parents=True, exist_ok=True)
        executed = []
        for step in plan["steps"]:
            if step["action"] != "REPLAY":
                continue
            recipe = self.graph.get_recipe(step["recipe_sha256"])
            rws = ws / recipe["recipe_sha256"]
            rws.mkdir(parents=True, exist_ok=True)
            result = self._execute_recipe(recipe, workspace=rws)
            executed.append({"target_node_id": step["node_id"], "recipe_sha256": recipe["recipe_sha256"], "result": result})
        if not self.graph.node_available(target):
            raise ReplayError("target remains unavailable after replay")
        roots = minimum_preservation_set(self.graph, [target])
        proof_base = {
            "schema_id": "IG_RECONSTRUCTION_PROOF_V0_1",
            "target_node_id": target,
            "root_set_sha256": canonical_sha256(roots.get("required_roots", [])),
            "recipe_closure_sha256": canonical_sha256([x["recipe_sha256"] for x in executed]),
            "status": "PASS", "result_sha256": self.graph.get_node(target)["semantic_identity"].get("sha256") or self.graph.get_node(target)["semantic_identity"].get("result_sha256"),
            "plan_sha256": plan["plan_sha256"], "executed": executed,
            "created_utc": utc_now(),
        }
        proof_id = canonical_sha256({k: v for k, v in proof_base.items() if k != "created_utc"})
        proof = dict(proof_base, proof_id=f"proof:sha256:{proof_id}")
        write_json_atomic(self.graph.proofs_dir / f"{proof_id}.json", proof)
        return {
            "schema_id": "IG_GRAPH_REPLAY_RESULT_V0_1", "status": "PASS",
            "target_node_id": target, "plan": plan, "executed": executed,
            "proof_id": proof["proof_id"], "workspace": str(ws),
        }


def minimum_preservation_set(graph: GraphStore, targets: list[str]) -> dict[str, Any]:
    with graph.live_read() as view:
        return _minimum_preservation_set_live(view, targets)


def _minimum_preservation_set_live(graph: GraphStore, targets: list[str]) -> dict[str, Any]:
    required: set[str] = set()
    expensive: set[str] = set()
    cache: set[str] = set()
    ephemeral: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    planner = ReplayPlanner(graph)

    def recipe_for(nid: str):
        try:
            return planner._recipe_for_target(nid)
        except Exception as e:
            unresolved.append({"node_id": nid, "reason": "RECIPE_AMBIGUITY", "error": str(e)})
            return None

    def walk(nid: str):
        if nid in seen:
            return
        seen.add(nid)
        try:
            node = graph.get_node(nid)
        except Exception as e:
            unresolved.append({"node_id": nid, "reason": "NODE_UNRESOLVED", "error": str(e)}); return
        ret = node["retention_class"]
        rec = recipe_for(nid)
        if ret == "PINNED":
            required.add(nid)
            # PINNED means the node is part of the irreducible preservation set.
            # If it declares concrete content, that content must actually be
            # present and hash-valid; a missing pinned blob/dataset cannot be
            # treated as a successful minimum-root closure merely because its
            # identity record still exists.
            if node.get("content_refs") and not graph.node_available(nid):
                unresolved.append({"node_id": nid, "reason": "MISSING_IRREDUCIBLE_ROOT"})
            # Pinned composite records may still have mandatory content dependencies.
            for e in graph.outgoing(nid, TRANSITIVE_COMPUTATION_EDGE_TYPES):
                if e.get("mandatory", True):
                    walk(e["target_node_id"])
            return
        if rec is not None and rec.get("replay_class") == "EXACT_REPLAY":
            if ret == "CACHE_EXPENSIVE": expensive.add(nid)
            elif ret == "CACHE": cache.add(nid)
            else: ephemeral.add(nid)
            for req in rec.get("input_requirements", []):
                walk(req["exact_identity_or_constraint"]["node_id"])
            for dep in [rec["producer"]["source_snapshot_ref"], rec["protocol_ref"], rec["question_ref"], rec["environment_ref"]]:
                walk(dep)
            return
        # A non-pinned node without exact recipe is not safe for GC/minimum closure.
        unresolved.append({"node_id": nid, "reason": "NOT_EXACTLY_RECONSTRUCTABLE", "retention_class": ret})

    resolved_targets = []
    for ref in targets:
        try:
            nid = graph.resolve(ref)["node_id"]
            resolved_targets.append(nid); walk(nid)
        except Exception as e:
            unresolved.append({"target": ref, "reason": "TARGET_UNRESOLVED", "error": str(e)})
    out = {
        "schema_id": "IG_MINIMUM_PRESERVATION_RESULT_V0_1",
        "status": "PASS" if not unresolved else "UNRESOLVED",
        "targets": resolved_targets,
        "required_roots": sorted(required),
        "optional_expensive_cache": sorted(expensive),
        "optional_cache": sorted(cache),
        "ephemeral": sorted(ephemeral),
        "unresolved": unresolved,
    }
    out["root_set_sha256"] = canonical_sha256({k: out[k] for k in ("targets", "required_roots", "optional_expensive_cache", "optional_cache", "ephemeral", "unresolved")})
    return out


def computation_closure(graph: GraphStore, targets: list[str]) -> dict[str, Any]:
    with graph.live_read() as view:
        return _computation_closure_live(view, targets)


def _computation_closure_live(graph: GraphStore, targets: list[str]) -> dict[str, Any]:
    nodes: set[str] = set(); edges: set[str] = set(); unresolved = []
    def walk(nid: str):
        if nid in nodes: return
        try: graph.get_node(nid)
        except Exception as e: unresolved.append({"node_id": nid, "error": str(e)}); return
        nodes.add(nid)
        for edge in graph.outgoing(nid, TRANSITIVE_COMPUTATION_EDGE_TYPES):
            if edge.get("mandatory", True):
                edges.add(edge["edge_id"]); walk(edge["target_node_id"])
    resolved=[]
    for ref in targets:
        try: nid=graph.resolve(ref)["node_id"]; resolved.append(nid); walk(nid)
        except Exception as e: unresolved.append({"target":ref,"error":str(e)})
    return {"status":"PASS" if not unresolved else "UNRESOLVED","targets":resolved,"nodes":sorted(nodes),"edges":sorted(edges),"unresolved":unresolved}


def evidence_closure(graph: GraphStore, target_reference: str) -> dict[str, Any]:
    with graph.live_read() as view:
        return _evidence_closure_live(view, target_reference)


def _evidence_closure_live(graph: GraphStore, target_reference: str) -> dict[str, Any]:
    target = graph.resolve(target_reference)["node_id"]
    nodes={target}; edges=set(); queue=[target]
    # Evidence may point into a claim or review, so traverse both directions for explicit evidence edges only.
    while queue:
        n=queue.pop(0)
        evidence_types={"SUPPORTS","FALSIFIES","QUALIFIES","REQUIRES_CLAIM","SEPARATES","AUTHORIZES"}
        for e in graph.outgoing(n,evidence_types)+graph.incoming(n,evidence_types):
            if e["source_node_id"]==n:
                edges.add(e["edge_id"])
                if e["target_node_id"] not in nodes: nodes.add(e["target_node_id"]); queue.append(e["target_node_id"])
            elif e["target_node_id"]==n:
                edges.add(e["edge_id"])
                if e["source_node_id"] not in nodes: nodes.add(e["source_node_id"]); queue.append(e["source_node_id"])
    return {"status":"PASS","target":target,"nodes":sorted(nodes),"edges":sorted(edges)}
