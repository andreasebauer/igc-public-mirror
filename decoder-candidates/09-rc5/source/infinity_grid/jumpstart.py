from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .graph_store import GraphStore, TRANSITIVE_COMPUTATION_EDGE_TYPES
from .hashing import sha256_file
from .records import utc_now
from .replay_contracts import ReplayContractStore, ReplayContractError


class JumpstartError(RuntimeError):
    pass


PROFILE_SCHEMA = "IG_JUMPSTART_PROFILE_V0_1"


def _profile_identity(profile: dict[str, Any]) -> str:
    return canonical_sha256({k: v for k, v in profile.items() if k not in {"profile_sha256", "created_utc"}})


def validate_profile(profile: dict[str, Any]) -> None:
    required = {
        "schema_id", "profile_id", "profile_version", "target_node_id", "target_alias",
        "materialization", "launch", "smoke", "status_probe", "created_utc", "profile_sha256",
    }
    missing = required - set(profile)
    if missing:
        raise JumpstartError(f"jumpstart profile missing fields: {sorted(missing)}")
    if profile.get("schema_id") != PROFILE_SCHEMA:
        raise JumpstartError("bad jumpstart profile schema")
    if _profile_identity(profile) != profile.get("profile_sha256"):
        raise JumpstartError("jumpstart profile hash mismatch")
    m = profile.get("materialization", {})
    if m.get("mode") not in {"SINGLE_ZIP_PAYLOAD", "MATERIAL_ONLY"}:
        raise JumpstartError(f"unsupported materialization mode {m.get('mode')}")
    if not isinstance(m.get("expected_material_root_count"), int) or m["expected_material_root_count"] < 1:
        raise JumpstartError("expected_material_root_count must be positive")
    for section in ("launch", "smoke"):
        argv = profile.get(section, {}).get("argv", [])
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
            raise JumpstartError(f"{section}.argv must be a nonempty string list")
        for x in argv:
            if x.startswith("/") or "../" in x:
                raise JumpstartError(f"absolute/upward path forbidden in {section}.argv")


def build_profile(**kwargs: Any) -> dict[str, Any]:
    p = {
        "schema_id": PROFILE_SCHEMA,
        "profile_id": kwargs["profile_id"],
        "profile_version": str(kwargs.get("profile_version", "1")),
        "target_node_id": kwargs["target_node_id"],
        "target_alias": kwargs.get("target_alias"),
        "materialization": kwargs["materialization"],
        "launch": kwargs["launch"],
        "smoke": kwargs["smoke"],
        "status_probe": kwargs.get("status_probe", {}),
        "created_utc": kwargs.get("created_utc") or utc_now(),
    }
    p["profile_sha256"] = _profile_identity(p)
    validate_profile(p)
    return p


class JumpstartProfileStore:
    def __init__(self, paths):
        self.paths = paths
        self.root = paths.store / "graph" / "jumpstart_profiles"
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, profile: dict[str, Any], graph: GraphStore | None = None) -> dict[str, Any]:
        validate_profile(profile)
        graph = graph or GraphStore(self.paths)
        graph.get_node(profile["target_node_id"])
        if profile.get("target_alias"):
            resolved = graph.resolve(profile["target_alias"])["node_id"]
            if resolved != profile["target_node_id"]:
                raise JumpstartError("profile target_alias does not resolve to target_node_id")
        p = self.root / f"{profile['profile_sha256']}.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            validate_profile(old)
            if old != profile:
                raise JumpstartError("immutable jumpstart profile conflict")
            return old
        write_json_atomic(p, profile)
        return profile

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            o = json.loads(p.read_text(encoding="utf-8")); validate_profile(o); out.append(o)
        return out

    def resolve_for_target(self, target_node_id: str) -> dict[str, Any]:
        matches = [p for p in self.list() if p["target_node_id"] == target_node_id]
        if len(matches) != 1:
            if not matches:
                raise JumpstartError(f"no jumpstart profile registered for {target_node_id}")
            raise JumpstartError(f"ambiguous jumpstart profiles for {target_node_id}: {len(matches)}")
        return matches[0]


# PRODUCED_BY points to an alternate exact reconstruction recipe; it is not a material
# dependency of a jump-start payload. Following it here would incorrectly turn source,
# environment, question and protocol metadata into payload roots.
JUMPSTART_DEPENDENCY_EDGE_TYPES = set(TRANSITIVE_COMPUTATION_EDGE_TYPES) - {"PRODUCED_BY"}

def _mandatory_children(graph: GraphStore, node_id: str) -> list[str]:
    return sorted({
        e["target_node_id"] for e in graph.outgoing(node_id, JUMPSTART_DEPENDENCY_EDGE_TYPES)
        if e.get("mandatory", True)
    })


class JumpstartPlanner:
    """Target-scoped dependency resolver.

    It intentionally does not weaken the global graph doctor. A compact imported graph may have
    external pinned roots for unrelated targets; jump-start correctness is judged only on the exact
    mandatory closure of the requested target.
    """
    def __init__(self, paths):
        self.paths = paths
        self.graph = GraphStore(paths)
        self.profiles = JumpstartProfileStore(paths)
        self.replay_contracts = ReplayContractStore(paths)

    def _resolve_target(self, target_ref: str) -> dict[str, Any]:
        try:
            return self.graph.resolve(target_ref)
        except Exception as first:
            if ":" not in target_ref and "/" not in target_ref:
                try:
                    return self.graph.resolve(f"ig:test/{target_ref}")
                except Exception:
                    pass
            raise first

    def plan(self, target_ref: str) -> dict[str, Any]:
        with self.graph.live_read() as graph:
            return self._plan_live(target_ref, graph)

    def _plan_live(self, target_ref: str, graph) -> dict[str, Any]:
        target = self._resolve_target(target_ref)
        profile = self.profiles.resolve_for_target(target["node_id"])
        replay_contract = None
        try:
            replay_contract = self.replay_contracts.resolve_for_target(target["node_id"])
            if replay_contract["profile_id"] != profile["profile_id"] or replay_contract["profile_sha256"] != profile["profile_sha256"]:
                raise ReplayContractError("replay contract/profile identity mismatch")
        except ReplayContractError:
            if self.replay_contracts.required():
                raise
        nodes: set[str] = set(); edges: list[dict[str, Any]] = []; unresolved: list[dict[str, Any]] = []
        visiting: set[str] = set()

        def walk(nid: str):
            if nid in nodes: return
            if nid in visiting:
                unresolved.append({"node_id": nid, "reason": "DEPENDENCY_CYCLE"}); return
            visiting.add(nid)
            try: node = graph.get_node(nid)
            except Exception as e:
                unresolved.append({"node_id": nid, "reason": "NODE_UNRESOLVED", "error": str(e)}); visiting.remove(nid); return
            nodes.add(nid)
            for e in graph.outgoing(nid, JUMPSTART_DEPENDENCY_EDGE_TYPES):
                if not e.get("mandatory", True):
                    continue
                edges.append(e); walk(e["target_node_id"])
            visiting.remove(nid)

        walk(target["node_id"])
        children = {n: _mandatory_children(graph, n) for n in nodes}
        leaves = sorted(n for n in nodes if n != target["node_id"] and not children[n])
        material_roots = []
        for nid in leaves:
            node = graph.get_node(nid)
            refs = node.get("content_refs", [])
            if not refs:
                unresolved.append({"node_id": nid, "reason": "LEAF_HAS_NO_MATERIAL_CONTENT"}); continue
            if len(refs) != 1 or refs[0].get("kind") not in {"ARTIFACT", "DATASET"}:
                unresolved.append({"node_id": nid, "reason": "UNSUPPORTED_ROOT_CONTENT_REFS", "refs": refs}); continue
            if not graph.node_available(nid):
                unresolved.append({"node_id": nid, "reason": "MISSING_MATERIAL_ROOT"}); continue
            material_roots.append({
                "node_id": nid, "semantic_role": node.get("semantic_role"),
                "retention_class": node.get("retention_class"), "content_ref": refs[0],
            })
        expected = profile["materialization"]["expected_material_root_count"]
        if len(material_roots) != expected:
            unresolved.append({"reason": "MATERIAL_ROOT_COUNT_MISMATCH", "expected": expected, "observed": len(material_roots)})
        cache = []
        for nid in sorted(nodes):
            if nid == target["node_id"]: continue
            node = graph.get_node(nid)
            if node.get("retention_class") in {"CACHE_EXPENSIVE", "CACHE", "EPHEMERAL"}:
                cache.append({"node_id": nid, "retention_class": node["retention_class"], "available": graph.node_available(nid)})
        core = {
            "schema_id": "IG_JUMPSTART_PLAN_V0_1", "target_ref": target_ref,
            "target_node_id": target["node_id"], "profile_sha256": profile["profile_sha256"],
            "mandatory_nodes": sorted(nodes), "mandatory_edges": sorted(e["edge_id"] for e in edges),
            "material_roots": material_roots, "cache_candidates": cache,
            "replay_contract": None if replay_contract is None else {
                "contract_sha256": replay_contract["contract_sha256"],
                "replay_class": replay_contract["replay_class"],
                "authority_class": replay_contract["authority_class"],
                "historical_identity": replay_contract["historical_identity"],
                "recomputes": replay_contract["recomputes"],
                "does_not_recompute": replay_contract["does_not_recompute"],
                "limitations": list(replay_contract["limitations"]),
                "legacy_resume_semantics": replay_contract.get("legacy_resume_semantics", ""),
            },
            "unresolved": unresolved,
            "status": "PASS" if not unresolved else "UNRESOLVED",
        }
        core["plan_sha256"] = canonical_sha256({k: v for k, v in core.items() if k != "plan_sha256"})
        return core


def _safe_extract_zip(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad: raise JumpstartError(f"zip CRC failure: {bad}")
        dest = destination.resolve()
        for info in z.infolist():
            p = (destination / info.filename).resolve()
            try: p.relative_to(dest)
            except ValueError: raise JumpstartError(f"unsafe zip path: {info.filename}")
        z.extractall(destination)
    return destination


def _verify_fileset(root: Path, manifest_rel: str, expected_tree_sha256: str | None = None, mutable_globs: list[str] | None = None) -> dict[str, Any]:
    mp = root / manifest_rel
    if not mp.is_file(): raise JumpstartError(f"manifest missing: {manifest_rel}")
    man = json.loads(mp.read_text(encoding="utf-8"))
    failures = []
    rows = []
    ignored = []
    mutable_globs = mutable_globs or []
    for rec in man.get("files", []):
        rel = rec["path"]
        if any(fnmatch.fnmatch(rel, pat) for pat in mutable_globs):
            ignored.append(rel); continue
        p = root / rel
        if not p.is_file(): failures.append({"path": rel, "reason": "MISSING"}); continue
        sha = sha256_file(p); size = p.stat().st_size
        if sha != rec["sha256"] or size != rec["size_bytes"]:
            failures.append({"path": rel, "reason": "HASH_OR_SIZE", "observed_sha256": sha, "observed_size": size})
        rows.append({"path": rel, "sha256": sha, "size_bytes": size})
    # Frozen fileset tree hash is the manifest's own declared tree identity. Do not recompute
    # under a different canonicalization; verify the declaration against the profile pin.
    declared_tree = man.get("tree_sha256")
    if expected_tree_sha256 and declared_tree != expected_tree_sha256:
        failures.append({"reason": "TREE_IDENTITY_MISMATCH", "expected": expected_tree_sha256, "declared": declared_tree})
    return {"status": "PASS" if not failures else "FAIL", "manifest": str(mp), "files_checked": len(rows), "mutable_ignored": ignored, "declared_tree_sha256": declared_tree, "failures": failures}


def _expand_argv(argv: list[str], *, work_root: Path) -> list[str]:
    out = []
    for x in argv:
        x = x.replace("{python}", sys.executable).replace("{work_root}", str(work_root))
        out.append(x)
    return out


class JumpstartRuntime:
    def __init__(self, paths):
        self.paths = paths
        self.graph = GraphStore(paths)
        self.planner = JumpstartPlanner(paths)
        self.profiles = JumpstartProfileStore(paths)

    def _default_workspace(self, target_node_id: str) -> Path:
        tag = hashlib.sha256(target_node_id.encode()).hexdigest()[:16]
        return self.paths.workspace / "jumpstart" / tag

    def prepare(self, target_ref: str, *, workspace: Path | None = None, clean: bool = False) -> dict[str, Any]:
        plan = self.planner.plan(target_ref)
        if plan["status"] != "PASS": raise JumpstartError(f"jumpstart unresolved: {plan['unresolved']}")
        profile = self.profiles.resolve_for_target(plan["target_node_id"])
        ws = Path(workspace) if workspace else self._default_workspace(plan["target_node_id"])
        ws = ws.resolve()
        project_root = self.paths.root.resolve(); designated = self.paths.workspace.resolve()
        if workspace is not None:
            if ws == project_root or project_root.is_relative_to(ws):
                raise JumpstartError("workspace may not equal or contain the decoder root")
            if ws.is_relative_to(project_root) and not ws.is_relative_to(designated):
                raise JumpstartError("workspace inside decoder root must be below dedicated workspace")
        for protected in (self.paths.store.resolve(), self.paths.runs.resolve(), self.paths.catalog.resolve(), self.paths.releases.resolve()):
            if ws.is_relative_to(protected):
                raise JumpstartError("workspace may not live inside immutable record roots")
        if clean and ws.exists(): shutil.rmtree(ws, ignore_errors=False)
        ws.mkdir(parents=True, exist_ok=True)
        marker = ws / "JUMPSTART_PREPARED.json"
        if marker.is_file() and not clean:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("plan_sha256") != plan["plan_sha256"] or prior.get("profile_sha256") != profile["profile_sha256"]:
                raise JumpstartError("existing workspace belongs to a different jumpstart plan")
            work_root = Path(prior["work_root"])
            verify = self._verify_payload(profile, work_root)
            if verify["status"] != "PASS": raise JumpstartError(f"existing workspace integrity failed: {verify['failures']}")
            return {"schema_id":"IG_JUMPSTART_PREPARE_RESULT_V0_1","status":"PASS","reused":True,"target_node_id":plan["target_node_id"],"plan":plan,"workspace":str(ws),"work_root":str(work_root),"verification":verify,"launch_argv":_expand_argv(profile["launch"]["argv"],work_root=work_root),"smoke_argv":_expand_argv(profile["smoke"]["argv"],work_root=work_root)}

        roots_dir = ws / "roots"; roots_dir.mkdir(exist_ok=True)
        mats = []
        for i, rootrec in enumerate(plan["material_roots"]):
            ref = rootrec["content_ref"]; role = (rootrec.get("semantic_role") or "ROOT").replace("/", "_")
            if ref["kind"] == "ARTIFACT":
                ext = ".zip" if profile["materialization"]["mode"] == "SINGLE_ZIP_PAYLOAD" else ".bin"
                dest = roots_dir / f"{i:02d}_{role}_{ref['sha256'][:12]}{ext}"
                self.graph.artifacts.materialize(ref["sha256"], dest, expected_size=ref.get("size_bytes"))
                mats.append({"node_id":rootrec["node_id"],"path":str(dest),"sha256":sha256_file(dest),"size_bytes":dest.stat().st_size})
            else:
                dest = roots_dir / f"{i:02d}_{role}_{ref['dataset_sha256'][:12]}"
                self.graph.datasets.materialize(ref["dataset_sha256"], dest)
                mats.append({"node_id":rootrec["node_id"],"path":str(dest),"dataset_sha256":ref["dataset_sha256"]})
        mode = profile["materialization"]["mode"]
        if mode == "SINGLE_ZIP_PAYLOAD":
            payload = ws / "payload"; shutil.rmtree(payload, ignore_errors=True); payload.mkdir()
            _safe_extract_zip(Path(mats[0]["path"]), payload)
            tops = [p for p in payload.iterdir() if p.is_dir()]
            if len(tops) != 1: raise JumpstartError(f"single-zip payload must contain exactly one top-level directory, found {len(tops)}")
            work_root = tops[0]
        else:
            work_root = ws
        verify = self._verify_payload(profile, work_root)
        if verify["status"] != "PASS": raise JumpstartError(f"prepared payload verification failed: {verify['failures']}")
        marker_obj = {"schema_id":"IG_JUMPSTART_WORKSPACE_MARKER_V0_1","target_node_id":plan["target_node_id"],"plan_sha256":plan["plan_sha256"],"profile_sha256":profile["profile_sha256"],"material_roots":mats,"workspace":str(ws),"work_root":str(work_root),"created_utc":utc_now()}
        write_json_atomic(marker, marker_obj)
        return {"schema_id":"IG_JUMPSTART_PREPARE_RESULT_V0_1","status":"PASS","reused":False,"target_node_id":plan["target_node_id"],"plan":plan,"workspace":str(ws),"work_root":str(work_root),"verification":verify,"materialized":mats,"launch_argv":_expand_argv(profile["launch"]["argv"],work_root=work_root),"smoke_argv":_expand_argv(profile["smoke"]["argv"],work_root=work_root)}

    def _verify_payload(self, profile: dict[str, Any], work_root: Path) -> dict[str, Any]:
        m = profile["materialization"]
        manifest_rel = m.get("manifest_relative_path")
        if manifest_rel:
            return _verify_fileset(work_root, manifest_rel, m.get("expected_tree_sha256"), m.get("mutable_globs", []))
        return {"status":"PASS","files_checked":0,"failures":[]}

    def smoke(self, target_ref: str, *, workspace: Path | None = None, clean: bool = False, timeout_seconds: int | None = None) -> dict[str, Any]:
        prep = self.prepare(target_ref, workspace=workspace, clean=clean)
        profile = self.profiles.resolve_for_target(prep["target_node_id"])
        work_root = Path(prep["work_root"])
        argv = _expand_argv(profile["smoke"]["argv"], work_root=work_root)
        env = dict(os.environ); env.update({str(k): str(v).replace("{work_root}",str(work_root)) for k,v in profile["smoke"].get("env",{}).items()})
        outdir=Path(prep["workspace"])/"jumpstart_logs"; outdir.mkdir(exist_ok=True)
        stdout_path=outdir/"smoke.stdout"; stderr_path=outdir/"smoke.stderr"
        timeout=int(timeout_seconds or profile["smoke"].get("timeout_seconds",120))
        t0=time.time(); timed_out=False
        # Historical verification scripts occasionally launch helper processes that outlive
        # the authoritative parent.  Pipes can then remain open forever even though the
        # verifier itself has exited.  Use file-backed logs plus a dedicated process group so
        # smoke verification is bounded and fail-closed across all recovered profiles.
        with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
            proc=subprocess.Popen(argv,cwd=str(work_root),env=env,stdout=so,stderr=se,start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=10)
            else:
                # Reap any helper left in the verifier's private process group.  The parent
                # result is authoritative; detached helpers must not survive a smoke check.
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        stdout=stdout_path.read_text(encoding="utf-8",errors="replace")
        stderr=stderr_path.read_text(encoding="utf-8",errors="replace")
        ok=(not timed_out) and proc.returncode==0
        for rel in profile["smoke"].get("required_files",[]):
            if not (work_root/rel).is_file(): ok=False
        result={"schema_id":"IG_JUMPSTART_SMOKE_RESULT_V0_1","status":"PASS" if ok else "FAIL","target_node_id":prep["target_node_id"],"plan_sha256":prep["plan"]["plan_sha256"],"workspace":prep["workspace"],"work_root":prep["work_root"],"argv":argv,"returncode":proc.returncode,"timed_out":timed_out,"runtime_seconds":round(time.time()-t0,6),"stdout_sha256":hashlib.sha256(stdout.encode()).hexdigest(),"stderr_sha256":hashlib.sha256(stderr.encode()).hexdigest(),"required_files":profile["smoke"].get("required_files",[])}
        write_json_atomic(outdir/"SMOKE_RESULT.json",result)
        if not ok:
            why=f"timeout after {timeout}s" if timed_out else f"rc={proc.returncode}"
            raise JumpstartError(f"jumpstart smoke failed {why}: {stderr[-2000:]}")
        return result

    def launch(self, target_ref: str, *, workspace: Path | None = None, clean: bool = False, detach: bool = True) -> dict[str, Any]:
        prep=self.prepare(target_ref,workspace=workspace,clean=clean)
        profile=self.profiles.resolve_for_target(prep["target_node_id"]); work_root=Path(prep["work_root"])
        argv=_expand_argv(profile["launch"]["argv"],work_root=work_root)
        env=dict(os.environ); env.update({str(k):str(v).replace("{work_root}",str(work_root)) for k,v in profile["launch"].get("env",{}).items()})
        logs=Path(prep["workspace"])/"jumpstart_logs"; logs.mkdir(exist_ok=True); log=logs/"launch.log"
        if detach:
            f=log.open("ab",buffering=0)
            proc=subprocess.Popen(argv,cwd=str(work_root),env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            f.close(); pid=proc.pid
            # We intentionally relinquish process ownership to the OS session. Mark the local
            # Popen wrapper reaped so its destructor does not warn while the detached child runs.
            proc.returncode=0
            state={"schema_id":"IG_JUMPSTART_LAUNCH_STATE_V0_1","status":"RUNNING_DETACHED","target_node_id":prep["target_node_id"],"pid":pid,"argv":argv,"workspace":prep["workspace"],"work_root":prep["work_root"],"log":str(log),"started_utc":utc_now()}
            write_json_atomic(Path(prep["workspace"])/"JUMPSTART_LAUNCH_STATE.json",state); return state
        with log.open("ab",buffering=0) as f:
            rc=subprocess.call(argv,cwd=str(work_root),env=env,stdout=f,stderr=subprocess.STDOUT)
        state={"schema_id":"IG_JUMPSTART_LAUNCH_STATE_V0_1","status":"COMPLETE" if rc==0 else "FAIL_CLOSED","target_node_id":prep["target_node_id"],"returncode":rc,"argv":argv,"workspace":prep["workspace"],"work_root":prep["work_root"],"log":str(log),"finished_utc":utc_now()}
        write_json_atomic(Path(prep["workspace"])/"JUMPSTART_LAUNCH_STATE.json",state); return state

    def status(self, target_ref: str, *, workspace: Path | None = None) -> dict[str, Any]:
        target=self.planner._resolve_target(target_ref); ws=Path(workspace).resolve() if workspace else self._default_workspace(target["node_id"])
        statep=ws/"JUMPSTART_LAUNCH_STATE.json"; marker=ws/"JUMPSTART_PREPARED.json"
        out={"schema_id":"IG_JUMPSTART_STATUS_V0_1","target_node_id":target["node_id"],"workspace":str(ws),"prepared":marker.is_file(),"launch_state":None,"native_status":None}
        if statep.is_file():
            state=json.loads(statep.read_text()); out["launch_state"]=state
            pid=state.get("pid")
            if pid and state.get("status")=="RUNNING_DETACHED":
                try: os.kill(pid,0); out["process_alive"]=True
                except OSError: out["process_alive"]=False
        if marker.is_file():
            prior=json.loads(marker.read_text()); work_root=Path(prior["work_root"]); profile=self.profiles.resolve_for_target(target["node_id"])
            for rel in profile.get("status_probe",{}).get("json_files",[]):
                p=work_root/rel
                if p.is_file():
                    try: out.setdefault("native_json",{})[rel]=json.loads(p.read_text())
                    except Exception: pass
            for rel in profile.get("status_probe",{}).get("text_files",[]):
                p=work_root/rel
                if p.is_file(): out.setdefault("native_text",{})[rel]=p.read_text(errors="replace").strip()
        out["status"]="PASS"
        return out
