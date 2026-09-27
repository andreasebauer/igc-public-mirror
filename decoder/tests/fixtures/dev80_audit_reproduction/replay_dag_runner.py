from __future__ import annotations

"""Fail-closed state machine for the L0-upward replay DAG.

This module schedules obligations and seals evidence.  It deliberately does
not execute science and it cannot mint audit authority.  Scientific handlers
and controller registration belong to P2B.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .replay_obligation_compiler import ReplayObligationError, verify_compiled_manifest
from .replay_index_lock import replay_index_lock


RUNNER_SCHEMA = "IG_REPLAY_DAG_RUNNER_STATE_V1"
RESULT_SCHEMA = "IG_REPLAY_NODE_RESULT_V1"
CHECKPOINT_SCHEMA = "IG_REPLAY_NODE_CHECKPOINT_V1"
CAPSULE_SCHEMA = "IG_REPLAY_AUDIT_CAPSULE_V1"
DECISION_SCHEMA = "IG_REPLAY_EXTERNAL_AUDIT_DECISION_V1"
CORE_STATUS = "RUNNER_CORE_ONLY_NO_SCIENCE_HANDLER_BINDINGS"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class ReplayDagRunnerError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayDagRunnerError(message)


def _self_hash(record: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({key: value for key, value in record.items() if key != field})


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayDagRunnerError(f"cannot read canonical replay record {path}: {exc}") from exc
    _require(isinstance(value, dict), f"replay record is not an object: {path}")
    return value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        _require(_load_json(path) == value, f"immutable replay record collision: {path}")
        return
    write_json_atomic(path, value)


class ReplayDagRunner:
    """Deterministic, resumable scheduler over a verified compiled manifest."""

    def __init__(self, manifest: Mapping[str, Any], state_dir: Path, state: Mapping[str, Any]):
        self.manifest = deepcopy(dict(manifest))
        self.state_dir = Path(state_dir)
        self.state = deepcopy(dict(state))
        self.nodes = {row["canonical_id"]: row for row in self.manifest["nodes"]}
        self._persisted_state_sha256 = self.state.get("state_sha256")
        with replay_index_lock(self.state_dir, ReplayDagRunnerError):
            self._verify_all()

    @classmethod
    def create(
        cls,
        manifest: Mapping[str, Any],
        state_dir: Path,
        *,
        root_run_id: str,
        dataset_root: Mapping[str, Any],
    ) -> "ReplayDagRunner":
        try:
            verify_compiled_manifest(manifest)
        except ReplayObligationError as exc:
            raise ReplayDagRunnerError(f"compiled manifest rejected: {exc}") from exc
        _require(isinstance(root_run_id, str) and bool(root_run_id.strip()), "root_run_id must be nonempty")
        _require(dataset_root.get("state") == "EMPTY", "P2 replay must start from an empty dataset root")
        digest = dataset_root.get("sha256")
        _require(isinstance(digest, str) and HASH_RE.fullmatch(digest) is not None,
                 "dataset_root.sha256 must be a canonical SHA-256")
        root = Path(state_dir)
        _require(not (root / "runner_state.json").exists(), "runner state already exists; use resume")
        state: dict[str, Any] = {
            "schema_id": RUNNER_SCHEMA,
            "implementation_status": CORE_STATUS,
            "root_run_id": root_run_id,
            "dataset_root": dict(dataset_root),
            "manifest_sha256": manifest["dag_sha256"],
            "runner_status": "READY",
            "completed_node_ids": [],
            "accepted_checkpoint_sha256_by_node": {},
            "attempt_by_node": {},
            "externally_authorized_node_ids": [],
            "external_decision_sha256s": [],
            "active_audit_capsule_sha256": None,
        }
        state["state_sha256"] = _self_hash(state, "state_sha256")
        with replay_index_lock(root, ReplayDagRunnerError):
            _require(not (root / "runner_state.json").exists(), "runner state already exists; use resume")
            write_json_atomic(root / "runner_state.json", state)
        return cls(manifest, root, state)

    @classmethod
    def resume(cls, manifest: Mapping[str, Any], state_dir: Path) -> "ReplayDagRunner":
        return cls(manifest, state_dir, _load_json(Path(state_dir) / "runner_state.json"))

    def _verify_all(self) -> None:
        self._verify_current()
        try:
            verify_compiled_manifest(self.manifest)
        except ReplayObligationError as exc:
            raise ReplayDagRunnerError(f"compiled manifest rejected: {exc}") from exc
        _require(self.state.get("schema_id") == RUNNER_SCHEMA, "bad runner state schema")
        _require(self.state.get("implementation_status") == CORE_STATUS,
                 "runner implementation boundary was altered")
        _require(self.state.get("manifest_sha256") == self.manifest["dag_sha256"],
                 "runner state is bound to another manifest")
        _require(self.state.get("state_sha256") == _self_hash(self.state, "state_sha256"),
                 "runner state hash mismatch")
        completed = self.state.get("completed_node_ids")
        hashes = self.state.get("accepted_checkpoint_sha256_by_node")
        _require(isinstance(completed, list) and len(completed) == len(set(completed)),
                 "invalid completed-node frontier")
        _require(isinstance(hashes, dict) and set(hashes) == set(completed),
                 "checkpoint index disagrees with completed-node frontier")
        order = self.manifest["topological_order"]
        _require(completed == [node_id for node_id in order if node_id in set(completed)],
                 "completed-node frontier is not in canonical topological order")
        for node_id in completed:
            digest = hashes[node_id]
            _require(isinstance(digest, str) and HASH_RE.fullmatch(digest) is not None,
                     f"invalid checkpoint hash for {node_id}")
            path = self.state_dir / "checkpoints" / f"{digest}.json"
            checkpoint = _load_json(path)
            _require(checkpoint.get("checkpoint_sha256") == digest,
                     f"checkpoint identity mismatch for {node_id}")
            _require(_self_hash(checkpoint, "checkpoint_sha256") == digest,
                     f"checkpoint content hash mismatch for {node_id}")
            _require(checkpoint.get("node_id") == node_id, f"checkpoint node mismatch for {node_id}")
            _require(checkpoint.get("root_run_id") == self.state["root_run_id"]
                     and checkpoint.get("manifest_sha256") == self.manifest["dag_sha256"],
                     f"checkpoint run binding mismatch for {node_id}")
        retained = {path.stem for path in (self.state_dir / "checkpoints").glob("*.json")}
        _require(retained == set(hashes.values()),
                 "RUNNER_INDEX_ROLLBACK_OR_INTERRUPTED_ACCEPT: retained checkpoint missing from index")
        active = self.state.get("active_audit_capsule_sha256")
        if active is not None:
            capsule = self._load_capsule(active)
            _require(capsule["root_run_id"] == self.state["root_run_id"],
                     "active audit capsule belongs to another run")

    def _verify_current(self) -> None:
        disk = _load_json(self.state_dir / "runner_state.json")
        _require(disk.get("state_sha256") == _self_hash(disk, "state_sha256"),
                 "runner state hash mismatch")
        _require(disk.get("state_sha256") == self._persisted_state_sha256,
                 "STALE_RUNNER_STATE: reopen before continuing")

    def _write_state(self) -> None:
        self._verify_current()
        self.state["state_sha256"] = _self_hash(self.state, "state_sha256")
        write_json_atomic(self.state_dir / "runner_state.json", self.state)
        self._persisted_state_sha256 = self.state["state_sha256"]

    def _load_capsule(self, digest: str) -> dict[str, Any]:
        _require(isinstance(digest, str) and HASH_RE.fullmatch(digest) is not None,
                 "invalid audit capsule hash")
        capsule = _load_json(self.state_dir / "audit_capsules" / f"{digest}.json")
        _require(capsule.get("audit_capsule_sha256") == digest, "audit capsule identity mismatch")
        _require(_self_hash(capsule, "audit_capsule_sha256") == digest,
                 "audit capsule content hash mismatch")
        return capsule

    def _dependency_checkpoint_hashes(self, node: Mapping[str, Any]) -> dict[str, str]:
        accepted = self.state["accepted_checkpoint_sha256_by_node"]
        return {dep: accepted[dep] for dep in node["dependencies"]}

    def _accept(self, node: Mapping[str, Any], acceptance: Mapping[str, Any]) -> dict[str, Any]:
        node_id = node["canonical_id"]
        _require(node_id not in self.state["accepted_checkpoint_sha256_by_node"],
                 f"node already completed: {node_id}")
        checkpoint: dict[str, Any] = {
            "schema_id": CHECKPOINT_SCHEMA,
            "root_run_id": self.state["root_run_id"],
            "manifest_sha256": self.manifest["dag_sha256"],
            "node_id": node_id,
            "layer": node["layer"],
            "node_kind": node["node_kind"],
            "dependency_checkpoint_sha256s": self._dependency_checkpoint_hashes(node),
            "acceptance": deepcopy(dict(acceptance)),
        }
        checkpoint["checkpoint_sha256"] = _self_hash(checkpoint, "checkpoint_sha256")
        digest = checkpoint["checkpoint_sha256"]
        _write_immutable(self.state_dir / "checkpoints" / f"{digest}.json", checkpoint)
        self.state["completed_node_ids"].append(node_id)
        self.state["accepted_checkpoint_sha256_by_node"][node_id] = digest
        self.state["runner_status"] = "READY"
        self._write_state()
        return checkpoint

    def _stop(
        self,
        node: Mapping[str, Any],
        reason: str,
        *,
        node_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        allowed = set(self.manifest["audit_policy"]["stop_outcomes"])
        _require(reason in allowed, f"unknown or unauthorized stop outcome: {reason}")
        node_id = node["canonical_id"]
        capsule: dict[str, Any] = {
            "schema_id": CAPSULE_SCHEMA,
            "root_run_id": self.state["root_run_id"],
            "manifest_sha256": self.manifest["dag_sha256"],
            "stopped_node_id": node_id,
            "layer": node["layer"],
            "stop_outcome": reason,
            "attempt": self.state["attempt_by_node"].get(node_id, 1),
            "completed_node_ids": list(self.state["completed_node_ids"]),
            "dependency_checkpoint_sha256s": self._dependency_checkpoint_hashes(node),
            "node_result": deepcopy(dict(node_result)) if node_result is not None else None,
            "allowed_external_decisions": list(self.manifest["audit_policy"]["external_decisions"]),
        }
        capsule["audit_capsule_sha256"] = _self_hash(capsule, "audit_capsule_sha256")
        digest = capsule["audit_capsule_sha256"]
        _write_immutable(self.state_dir / "audit_capsules" / f"{digest}.json", capsule)
        self.state["active_audit_capsule_sha256"] = digest
        self.state["runner_status"] = "WAITING_FOR_EXTERNAL_AUDIT"
        self._write_state()
        return capsule

    def next_action(self) -> dict[str, Any]:
        """Return the next scientific node, auto-checkpointing workflow gates."""
        with replay_index_lock(self.state_dir, ReplayDagRunnerError):
            return self._next_action()

    def _next_action(self) -> dict[str, Any]:
        self._verify_all()
        if self.state["active_audit_capsule_sha256"] is not None:
            return {"action": "WAIT_FOR_EXTERNAL_AUDIT",
                    "audit_capsule": self._load_capsule(self.state["active_audit_capsule_sha256"])}
        completed = set(self.state["completed_node_ids"])
        for node_id in self.manifest["topological_order"]:
            if node_id in completed:
                continue
            node = self.nodes[node_id]
            _require(all(dep in completed for dep in node["dependencies"]),
                     f"topological dependency frontier is incomplete at {node_id}")
            if node["node_kind"] == "WORKFLOW_GATE":
                self._accept(node, {
                    "mode": "AUTOMATIC_WORKFLOW_GATE",
                    "outcome": "CERTIFIED_REPLAY_PASS",
                    "science_authority_effect": "NONE",
                })
                completed.add(node_id)
                continue
            authorized = node.get("audit_authorization_state") == "HISTORICALLY_AUTHORIZED"
            authorized = authorized or node_id in self.state["externally_authorized_node_ids"]
            if not authorized:
                capsule = self._stop(node, "MISSING_HISTORICAL_AUDIT_AUTHORIZATION")
                return {"action": "WAIT_FOR_EXTERNAL_AUDIT", "audit_capsule": capsule}
            self.state["attempt_by_node"].setdefault(node_id, 1)
            self._write_state()
            return {
                "action": "EXECUTE_SCIENTIFIC_NODE",
                "node_id": node_id,
                "attempt": self.state["attempt_by_node"][node_id],
                "contract": {
                    key: deepcopy(node[key]) for key in (
                        "layer", "series", "carrier_type", "producer", "consumer", "observer",
                        "equality_mode", "set_or_multiset_semantics", "exact_or_compressed",
                        "formation_provenance", "known_qualification_ids",
                    ) if key in node
                },
            }
        self.state["runner_status"] = "COMPLETE"
        self._write_state()
        return {"action": "COMPLETE", "completed_node_ids": list(self.state["completed_node_ids"])}

    def record_node_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Seal an executor-produced comparison result; never execute the node here."""
        with replay_index_lock(self.state_dir, ReplayDagRunnerError):
            return self._record_node_result(result)

    def _record_node_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        action = self._next_action()
        _require(action["action"] == "EXECUTE_SCIENTIFIC_NODE", "runner is not awaiting a node result")
        _require(result.get("schema_id") == RESULT_SCHEMA, "bad node result schema")
        node_id = result.get("node_id")
        _require(node_id == action["node_id"], "node result is not for the scheduled node")
        _require(result.get("attempt") == action["attempt"], "node result attempt mismatch")
        for field in ("result_sha256", "evidence_sha256"):
            _require(isinstance(result.get(field), str) and HASH_RE.fullmatch(result[field]) is not None,
                     f"invalid {field}")
        outcome = result.get("comparison_outcome")
        automatic = set(self.manifest["audit_policy"]["automatic_continue_outcomes"])
        stops = set(self.manifest["audit_policy"]["stop_outcomes"])
        node = self.nodes[node_id]
        if outcome in automatic:
            fresh = node.get('effective_execution_class') == 'FRESH_RECOMPUTE'
            if fresh:
                _require(outcome == 'REPRODUCED_WITH_DECLARED_EVIDENCE_MODE'
                         and result.get('execution_class') == 'FRESH_RECOMPUTE'
                         and result.get('counts_toward_empty_root_science_replay') is True,
                         'fresh result requires a declared fresh execution')
            checkpoint = self._accept(node, {
                "mode": "AUTOMATIC_FRESH_RECOMPUTATION" if fresh else "AUTOMATIC_HISTORICAL_REPLAY",
                "comparison_outcome": outcome,
                "node_result": deepcopy(dict(result)),
                "known_qualification_ids": list(node.get("known_qualification_ids", [])),
                "result_language": "REPRODUCED",
            })
            return {"action": "CONTINUE", "checkpoint": checkpoint}
        _require(outcome in stops, f"unrecognized comparison outcome: {outcome}")
        capsule = self._stop(node, outcome, node_result=result)
        return {"action": "WAIT_FOR_EXTERNAL_AUDIT", "audit_capsule": capsule}

    def apply_external_decision(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a user-owned, hash-bound decision without interpreting science."""
        with replay_index_lock(self.state_dir, ReplayDagRunnerError):
            return self._apply_external_decision(decision)

    def _apply_external_decision(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        self._verify_all()
        active = self.state.get("active_audit_capsule_sha256")
        _require(active is not None, "no active audit capsule")
        capsule = self._load_capsule(active)
        _require(decision.get("schema_id") == DECISION_SCHEMA, "bad external decision schema")
        _require(decision.get("decision_sha256") == _self_hash(decision, "decision_sha256"),
                 "external decision hash mismatch")
        _require(decision.get("audit_capsule_sha256") == active,
                 "external decision is bound to another audit capsule")
        _require(decision.get("root_run_id") == self.state["root_run_id"],
                 "external decision is bound to another run")
        _require(decision.get("stopped_node_id") == capsule["stopped_node_id"],
                 "external decision node mismatch")
        node_result = capsule["node_result"]
        for field in ("result_sha256", "evidence_sha256"):
            expected = node_result.get(field) if node_result else None
            _require(decision.get(f"bound_{field}") == expected,
                     f"external decision {field} binding mismatch")
        name = decision.get("decision")
        _require(name in capsule["allowed_external_decisions"], "unsupported external decision")
        if capsule["stop_outcome"] == "MISSING_HISTORICAL_AUDIT_AUTHORIZATION":
            _require(name == "CERTIFY_AND_ADVANCE",
                     "missing authority requires CERTIFY_AND_ADVANCE")
        digest = decision["decision_sha256"]
        _require(digest not in self.state["external_decision_sha256s"],
                 "external decision was already applied")
        _write_immutable(self.state_dir / "external_decisions" / f"{digest}.json", decision)
        self.state["external_decision_sha256s"].append(digest)
        node_id = capsule["stopped_node_id"]
        node = self.nodes[node_id]
        if name == "REPEAT":
            self.state["attempt_by_node"][node_id] = capsule["attempt"] + 1
            self.state["active_audit_capsule_sha256"] = None
            self.state["runner_status"] = "READY"
        elif name in {"PATCH_REQUIRED", "NEW_SCIENCE_REQUIRED"}:
            self.state["runner_status"] = "WAITING_FOR_EXTERNAL_AUDIT"
        elif capsule["stop_outcome"] == "MISSING_HISTORICAL_AUDIT_AUTHORIZATION":
            ids = self.state["externally_authorized_node_ids"]
            if node_id not in ids:
                ids.append(node_id)
            self.state["active_audit_capsule_sha256"] = None
            self.state["runner_status"] = "READY"
        else:
            _require(node_result is not None, "cannot advance a discrepancy without a result")
            self.state["active_audit_capsule_sha256"] = None
            self._accept(node, {
                "mode": "EXTERNAL_AUDIT_DECISION",
                "decision": name,
                "decision_sha256": digest,
                "node_result": node_result,
                "result_language": "EXTERNALLY_ADJUDICATED_REPRODUCTION",
            })
            return {"action": "CONTINUE", "decision_sha256": digest}
        self._write_state()
        return {"action": self.state["runner_status"], "decision_sha256": digest}


def seal_external_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical helper for the external chat/audit side of the boundary."""
    record = deepcopy(dict(payload))
    record["decision_sha256"] = _self_hash(record, "decision_sha256")
    return record
