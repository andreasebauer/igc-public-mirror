from __future__ import annotations

"""Decoder v0.5 controlled scientific chaining.

This module is orchestration-only.  It chains already-frozen scientific stages
without inventing new scientific semantics.  S-series transitions may auto-run
only for registered outcomes.  R-series transitions are adaptive but may choose
only among preregistered branches; novelty, changed assumptions, authority
mismatch, or an unregistered result stops fail-closed for human review.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import copy
import json
import os
import time
import hashlib
import inspect
import marshal

from .v05_execution_authority import (
    EngineeringAuthority, ExecutionAuthorityError, ENGINEERING_ROLE, digest,
    science_digest, new_run_id, source_tree_digest, verify_execution_receipt,
)
from .v05_origin_guard import require_controller_execution_origin

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .records import utc_now
from .safety import validate_identifier

CHAIN_SCHEMA_V1 = "IG_DECODER_V05_SCIENTIFIC_CHAIN_REGISTRATION_V1"
CHAIN_SCHEMA_V2 = "IG_DECODER_V05_SCIENTIFIC_CHAIN_REGISTRATION_V2"
# Default registration schema is V2. Historical V1 remains explicit-only.
CHAIN_SCHEMA = CHAIN_SCHEMA_V2
LEGACY_CHAIN_SCHEMA = CHAIN_SCHEMA_V1
CHAIN_STATE_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_CHAIN_STATE_V2_AUTHENTICATED"
CHAIN_STAGE_COMMIT_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_CHAIN_STAGE_COMMIT_V2_AUTHENTICATED"
CHAIN_EVENT_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_CHAIN_EVENT_V1"
CHAIN_MIRROR_RECEIPT_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_CHAIN_MIRROR_RECEIPT_V1"

_ALLOWED_SERIES = {"S", "R", "REVIEW"}
_ALLOWED_STAGE_KINDS = {"PREPARED_EXECUTOR", "DECODER_PLAN", "DECODER_STAGE", "HUMAN_REVIEW"}
_ALLOWED_TRANSITION_ACTIONS = {"NEXT", "END", "REVIEW", "FAIL"}


class ScientificChainError(RuntimeError):
    pass


class ScientificChainValidationError(ScientificChainError):
    pass


class ScientificChainReviewRequired(ScientificChainError):
    pass


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_text(dict(obj)) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(fd, "ab", closefd=True) as h:
            h.write(data)
            h.flush()
            os.fsync(h.fileno())
    finally:
        _fsync_dir(path.parent)


def _json_pointer(obj: Any, pointer: str) -> Any:
    if pointer in {"", "/"}:
        return obj
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ScientificChainValidationError("outcome_pointer must be JSON pointer")
    cur = obj
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            try:
                cur = cur[int(token)]
            except Exception as exc:
                raise ScientificChainError(f"JSON pointer list lookup failed: {pointer}") from exc
        elif isinstance(cur, Mapping):
            if token not in cur:
                raise ScientificChainError(f"JSON pointer key missing: {pointer}")
            cur = cur[token]
        else:
            raise ScientificChainError(f"JSON pointer traversed scalar: {pointer}")
    return cur


def _validate_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ScientificChainValidationError(f"{field} must be lowercase sha256")
    return value


def _base_without_hash(obj: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(obj))
    out.pop("registration_sha256", None)
    return out


def seal_chain_registration(record_without_hash: Mapping[str, Any]) -> dict[str, Any]:
    base = _base_without_hash(record_without_hash)
    base["registration_sha256"] = canonical_sha256(base)
    return validate_chain_registration(base)


def validate_chain_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    reg = copy.deepcopy(dict(registration))
    required = {
        "schema_id", "chain_id", "subject", "release_line", "mode", "parent_authority",
        "stages", "budgets", "durability", "authority_policy", "registration_sha256",
    }
    if set(reg) != required:
        raise ScientificChainValidationError(f"chain registration fields mismatch: {sorted(set(reg)^required)}")
    if reg["schema_id"] not in {CHAIN_SCHEMA_V1, CHAIN_SCHEMA_V2}:
        raise ScientificChainValidationError("bad chain schema")
    schema_v2 = reg["schema_id"] == CHAIN_SCHEMA_V2
    validate_identifier(str(reg["chain_id"]), field="chain_id")
    if reg["release_line"] not in {"v0.5", "0.50", "v0.5 / 0.50"}:
        raise ScientificChainValidationError("chain release_line must be Decoder v0.5 / 0.50")
    if reg["mode"] != "CONTROLLED_S_THEN_ADAPTIVE_R":
        raise ScientificChainValidationError("unsupported chain mode")

    subj = reg["subject"]
    if not isinstance(subj, Mapping) or set(subj) != {"hierarchy", "level", "parent_ref"}:
        raise ScientificChainValidationError("subject must contain hierarchy/level/parent_ref")
    if not isinstance(subj["hierarchy"], str) or not subj["hierarchy"]:
        raise ScientificChainValidationError("subject hierarchy missing")
    if not isinstance(subj["level"], int) or subj["level"] < 0:
        raise ScientificChainValidationError("subject level invalid")
    if not isinstance(subj["parent_ref"], str) or not subj["parent_ref"]:
        raise ScientificChainValidationError("subject parent_ref missing")

    pa = reg["parent_authority"]
    if not isinstance(pa, Mapping) or set(pa) != {"authority_ref", "science_sha256", "verification_sha256", "status"}:
        raise ScientificChainValidationError("parent_authority fields mismatch")
    _validate_sha(pa["science_sha256"], "parent_authority.science_sha256")
    _validate_sha(pa["verification_sha256"], "parent_authority.verification_sha256")
    if pa["status"] not in {"GRADUATED", "CERTIFIED_PASS"}:
        raise ScientificChainValidationError("parent authority must be verified graduated/certified")

    stages = reg["stages"]
    if not isinstance(stages, list) or not stages:
        raise ScientificChainValidationError("chain requires stages")
    ids: list[str] = []
    seen_r = False
    for idx, st in enumerate(stages):
        if not isinstance(st, Mapping):
            raise ScientificChainValidationError("stage must be object")
        expected = {
            "stage_id", "series", "stage_kind", "question_ref", "question_sha256", "depends_on",
            "execution", "result_contract", "transitions", "auto_run", "promotion_effect",
        }
        if set(st) != expected:
            raise ScientificChainValidationError(f"stage {idx} fields mismatch")
        sid = str(st["stage_id"]); validate_identifier(sid.replace(":", "-"), field="stage_id")
        if sid in ids:
            raise ScientificChainValidationError(f"duplicate stage_id {sid}")
        ids.append(sid)
        if st["series"] not in _ALLOWED_SERIES:
            raise ScientificChainValidationError(f"bad series {sid}")
        if st["series"] == "R":
            seen_r = True
        elif st["series"] == "S" and seen_r:
            raise ScientificChainValidationError("S stage cannot appear after R series begins")
        if st["stage_kind"] not in _ALLOWED_STAGE_KINDS:
            raise ScientificChainValidationError(f"bad stage_kind {sid}")
        if schema_v2 and st["stage_kind"] == "PREPARED_EXECUTOR":
            raise ScientificChainValidationError(f"V2 scientific chains forbid PREPARED_EXECUTOR: {sid}")
        if (not schema_v2) and st["stage_kind"] == "DECODER_STAGE":
            raise ScientificChainValidationError(f"DECODER_STAGE requires V2 chain schema: {sid}")
        if not isinstance(st["question_ref"], str) or not st["question_ref"]:
            raise ScientificChainValidationError(f"question_ref missing {sid}")
        _validate_sha(st["question_sha256"], f"{sid}.question_sha256")
        if not isinstance(st["depends_on"], list) or any(x not in ids[:-1] for x in st["depends_on"]):
            raise ScientificChainValidationError(f"depends_on must reference earlier stages: {sid}")
        if not isinstance(st["execution"], Mapping):
            raise ScientificChainValidationError(f"execution must be object: {sid}")
        if st["stage_kind"] == "DECODER_PLAN":
            if set(st["execution"]) != {"plan"} or not isinstance(st["execution"]["plan"], Mapping):
                raise ScientificChainValidationError(f"DECODER_PLAN requires inline plan: {sid}")
        elif st["stage_kind"] == "PREPARED_EXECUTOR":
            if set(st["execution"]) != {"executor_key", "parameters"}:
                raise ScientificChainValidationError(f"PREPARED_EXECUTOR execution fields mismatch: {sid}")
            if not isinstance(st["execution"]["executor_key"], str) or not st["execution"]["executor_key"]:
                raise ScientificChainValidationError(f"executor_key missing: {sid}")
            if not isinstance(st["execution"]["parameters"], Mapping):
                raise ScientificChainValidationError(f"executor parameters invalid: {sid}")
        elif st["stage_kind"] == "DECODER_STAGE":
            if set(st["execution"]) != {"handler_key", "parameters"}:
                raise ScientificChainValidationError(f"DECODER_STAGE execution fields mismatch: {sid}")
            if not isinstance(st["execution"]["handler_key"], str) or not st["execution"]["handler_key"]:
                raise ScientificChainValidationError(f"handler_key missing: {sid}")
            if not isinstance(st["execution"]["parameters"], Mapping):
                raise ScientificChainValidationError(f"handler parameters invalid: {sid}")
        else:
            if st["execution"] != {}:
                raise ScientificChainValidationError(f"HUMAN_REVIEW execution must be empty: {sid}")

        rc = st["result_contract"]
        if not isinstance(rc, Mapping) or set(rc) != {"artifact_logical_name", "outcome_pointer", "allowed_outcomes"}:
            raise ScientificChainValidationError(f"result_contract fields mismatch: {sid}")
        if not isinstance(rc["artifact_logical_name"], str) or not rc["artifact_logical_name"]:
            raise ScientificChainValidationError(f"artifact_logical_name missing: {sid}")
        if not isinstance(rc["allowed_outcomes"], list) or not rc["allowed_outcomes"]:
            raise ScientificChainValidationError(f"allowed_outcomes missing: {sid}")
        tr = st["transitions"]
        if not isinstance(tr, Mapping) or set(tr) != set(rc["allowed_outcomes"]):
            raise ScientificChainValidationError(f"transitions must exactly cover allowed outcomes: {sid}")
        for outcome, rule in tr.items():
            if not isinstance(rule, Mapping) or set(rule) != {"action", "next_stage", "reason"}:
                raise ScientificChainValidationError(f"transition shape mismatch: {sid}/{outcome}")
            if rule["action"] not in _ALLOWED_TRANSITION_ACTIONS:
                raise ScientificChainValidationError(f"bad transition action: {sid}/{outcome}")
            nxt = rule["next_stage"]
            if rule["action"] == "NEXT":
                if not isinstance(nxt, str) or not nxt:
                    raise ScientificChainValidationError(f"NEXT requires next_stage: {sid}/{outcome}")
            elif nxt is not None:
                raise ScientificChainValidationError(f"non-NEXT transition next_stage must be null: {sid}/{outcome}")
        if not isinstance(st["auto_run"], bool):
            raise ScientificChainValidationError(f"auto_run must be bool: {sid}")
        if st["series"] == "R" and st["auto_run"] is True:
            # R may auto-run only because the branch itself is preregistered.  Human-review
            # outcomes still stop below. This flag is allowed but explicit.
            pass
        if st["promotion_effect"] not in {"NONE", "CANDIDATE_ONLY", "EXPLICIT_REVIEW_ONLY"}:
            raise ScientificChainValidationError(f"bad promotion_effect: {sid}")

    stage_set = set(ids)
    for st in stages:
        for outcome, rule in st["transitions"].items():
            if rule["action"] == "NEXT" and rule["next_stage"] not in stage_set:
                raise ScientificChainValidationError(f"transition target unknown: {st['stage_id']}/{outcome}")

    budgets = reg["budgets"]
    if not isinstance(budgets, Mapping) or set(budgets) != {"max_stage_executions", "max_chain_wall_seconds", "default_workers"}:
        raise ScientificChainValidationError("budgets fields mismatch")
    if not isinstance(budgets["max_stage_executions"], int) or budgets["max_stage_executions"] < 1:
        raise ScientificChainValidationError("max_stage_executions invalid")
    if not isinstance(budgets["max_chain_wall_seconds"], (int, float)) or budgets["max_chain_wall_seconds"] <= 0:
        raise ScientificChainValidationError("max_chain_wall_seconds invalid")
    if not isinstance(budgets["default_workers"], int) or not (1 <= budgets["default_workers"] <= 64):
        raise ScientificChainValidationError("default_workers invalid")

    dur = reg["durability"]
    if not isinstance(dur, Mapping) or set(dur) != {"fsync_each_transition", "stage_commits", "external_mirror_required"}:
        raise ScientificChainValidationError("durability fields mismatch")
    if dur["fsync_each_transition"] is not True or dur["stage_commits"] != "APPEND_ONLY":
        raise ScientificChainValidationError("chain requires fsync + append-only stage commits")
    if not isinstance(dur["external_mirror_required"], bool):
        raise ScientificChainValidationError("external_mirror_required must be bool")

    ap = reg["authority_policy"]
    if not isinstance(ap, Mapping) or set(ap) != {
        "automatic_promotion", "require_verified_parent", "novelty_policy", "changed_assumptions_policy",
        "unregistered_outcome_policy", "r_science_policy",
    }:
        raise ScientificChainValidationError("authority_policy fields mismatch")
    if ap["automatic_promotion"] is not False or ap["require_verified_parent"] is not True:
        raise ScientificChainValidationError("chain cannot auto-promote and must require verified parent")
    for k in ("novelty_policy", "changed_assumptions_policy", "unregistered_outcome_policy"):
        if ap[k] != "REVIEW_REQUIRED":
            raise ScientificChainValidationError(f"{k} must be REVIEW_REQUIRED")
    if ap["r_science_policy"] != "ADAPTIVE_ONLY_WITHIN_PREREGISTERED_BRANCHES":
        raise ScientificChainValidationError("R science policy mismatch")

    declared = _validate_sha(reg["registration_sha256"], "registration_sha256")
    if declared != canonical_sha256(_base_without_hash(reg)):
        raise ScientificChainValidationError("chain registration hash mismatch")
    return reg


@dataclass
class ChainExecutionResult:
    result: dict[str, Any]
    run_record: dict[str, Any] | None = None


ExecutorFn = Callable[[dict[str, Any], Path], ChainExecutionResult]


class ScientificChainController:
    """Crash-safe finite scientific chain controller.

    The controller never invents a stage or outcome. It executes a frozen stage,
    commits its exact result, evaluates only the preregistered transition table,
    and either advances or stops. Re-entry reuses verified stage commits.
    """

    def __init__(self, root: str | Path, *, decoder_paths=None,
                 allow_legacy_new_registrations: bool = False, engineering_only: bool = False,
                 _service_session=None):
        self._service_session = _service_session
        if _service_session is not None:
            from .v05_registered_service import RegisteredServiceSession
            if type(_service_session) is not RegisteredServiceSession or engineering_only is not True:
                raise ExecutionAuthorityError("SERVICE_SESSION_MODE")
            _service_session.verify_live()
            if Path(root).resolve() != _service_session.chain_root.resolve():
                raise ExecutionAuthorityError("SERVICE_CHAIN_ROOT_MISMATCH")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.decoder_paths = decoder_paths
        self.allow_legacy_new_registrations = bool(allow_legacy_new_registrations)
        self._executors: dict[str, ExecutorFn] = {}
        self._stage_handlers: dict[str, ExecutorFn] = {}
        self._admitted_handlers: dict[str, tuple] = {}
        self._pending_execution: tuple | None = None
        self._authority = (_service_session.authority if _service_session is not None
                           else EngineeringAuthority(enabled=engineering_only))
        self._package_root = Path(__file__).resolve().parent
        self._engineering_public_keys: dict[str, bytes] = {}

    def register_executor(self, key: str, fn: ExecutorFn) -> None:
        raise ExecutionAuthorityError("LEGACY_EXECUTOR_DISABLED", "historical V1 evidence is inspect-only")


    def register_stage_handler(self, key: str, fn) -> None:
        """Register a V2 controller-only scientific handler.

        Handler source is statically audited before registration. It receives a
        StageScienceRuntime capability object rather than a filesystem root.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("stage handler key must be nonempty")
        from .v05_stage_architecture import require_controller_only_callable
        require_controller_only_callable(fn, role="scientific stage handler")
        old = self._stage_handlers.get(key)
        if old is not None and old is not fn:
            raise ScientificChainError(f"conflicting chain stage handler {key}")
        from .v05_stage_registry import CONTROLLER_ONLY_STAGE_HANDLERS, ENGINEERING_ONLY_STAGE_HANDLERS
        inventory = ENGINEERING_ONLY_STAGE_HANDLERS if self._authority.enabled else CONTROLLER_ONLY_STAGE_HANDLERS
        ref = f"{fn.__module__}:{fn.__name__}"
        if self._service_session is not None:
            self._service_session.require_handler(key, ref)
        if inventory.get(key) != ref:
            raise ExecutionAuthorityError("STAGE_HANDLER_NOT_IN_ACCEPTED_REGISTRY", key)
        self._stage_handlers[key] = fn
        self._admitted_handlers[key] = self._callable_identity(fn)

    @staticmethod
    def _callable_identity(fn) -> tuple:
        path = inspect.getsourcefile(fn)
        if not path or not hasattr(fn, "__code__"):
            raise ExecutionAuthorityError("HANDLER_SOURCE_UNAVAILABLE")
        return (f"{fn.__module__}:{fn.__name__}", hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest())

    def _require_handler(self, stage):
        key = stage["execution"].get("handler_key")
        fn = self._stage_handlers.get(key)
        if fn is None:
            raise ExecutionAuthorityError("STAGE_HANDLER_UNREGISTERED", str(key))
        from .v05_stage_architecture import require_controller_only_callable
        require_controller_only_callable(fn, role="scientific stage handler preflight")
        if self._callable_identity(fn) != self._admitted_handlers.get(key):
            raise ExecutionAuthorityError("ADMITTED_HANDLER_CHANGED", key)
        return fn

    def _engineering_trust(self, cdir: Path, *, add_session_key: bool = False) -> dict[str, bytes]:
        # Local keys authorize ENGINEERING only. This file is NOT a production
        # trust anchor. An official verifier never learns keys from this file.
        if self._service_session is not None:
            self._service_session.verify_live()
            return dict(self._service_session.public_keys)
        path = cdir / "ENGINEERING_PUBLIC_KEYS.json"
        keys = {}
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("role") != ENGINEERING_ROLE:
                raise ExecutionAuthorityError("ENGINEERING_TRUST_ROLE")
            keys = {k: bytes.fromhex(v) for k, v in obj["keys"].items()}
        if add_session_key:
            keys.update(self._authority.public_keys)
            write_json_atomic(path, {"role": ENGINEERING_ROLE, "keys": {k: v.hex() for k, v in keys.items()}})
        self._engineering_public_keys = keys
        return keys

    def _execution_bindings(self, cdir, reg, stage, *, run_id):
        fn = self._require_handler(stage)
        ref, module_sha, _ = self._callable_identity(fn)
        deps = []
        for sid in stage["depends_on"]:
            obj = self._load_commit(cdir, sid)
            if obj is None:
                raise ExecutionAuthorityError("DEPENDENCY_RECEIPT_REQUIRED", sid)
            deps.append([sid, obj["commit_sha256"]])
        return {
            "chain_id": reg["chain_id"], "registration_sha256": reg["registration_sha256"],
            "stage_id": stage["stage_id"], "handler_key": stage["execution"]["handler_key"],
            "handler_ref": ref, "question_sha256": stage["question_sha256"],
            "source_sha256": source_tree_digest(self._package_root), "handler_source_sha256": module_sha,
            "parameters_sha256": digest(dict(stage["execution"].get("parameters") or {})),
            "authority_sha256": digest(reg["parent_authority"]), "dependencies_sha256": digest(deps),
            "evidence_store_id": digest({"chain_dir": str(cdir.resolve(strict=True))}), "run_id": run_id,
        }

    def _validate_state_cursor(self, reg, cdir, state):
        if state.get("schema_id") != CHAIN_STATE_SCHEMA or state.get("authoritative") is not False:
            raise ExecutionAuthorityError("UNAUTHENTICATED_CHAIN_STATE")
        stages = {x["stage_id"]: x for x in reg["stages"]}
        expected = reg["stages"][0]["stage_id"]
        completed = state.get("completed_stages")
        if not isinstance(completed, list) or len(completed) != len(set(completed)):
            raise ExecutionAuthorityError("CHAIN_STATE_PATH_MISMATCH")
        last = None; rule = None
        for sid in completed:
            if sid != expected or sid not in stages:
                raise ExecutionAuthorityError("CHAIN_STATE_PATH_MISMATCH")
            commit = self._load_commit(cdir, sid)
            if commit is None:
                raise ExecutionAuthorityError("COMPLETED_STAGE_RECEIPT_MISSING", sid)
            last = sid
            rule = stages[sid]["transitions"].get(commit["outcome"])
            expected = rule["next_stage"] if rule and rule["action"] == "NEXT" else None
        if state.get("current_stage") not in ({expected, last} if last else {expected}):
            raise ExecutionAuthorityError("CHAIN_CURSOR_SKIPPED_STAGE")
        if state.get("status") == "COMPLETE":
            if not completed or not rule or rule["action"] != "END" or state.get("current_stage") is not None:
                raise ExecutionAuthorityError("COMPLETE_WITHOUT_AUTHENTICATED_END")
            if reg["durability"]["external_mirror_required"]:
                for sid in completed:
                    if not self._mirror_acknowledged(cdir, reg, sid, self._load_commit(cdir, sid)):
                        raise ExecutionAuthorityError("COMPLETE_WITHOUT_MIRROR")

    def chain_dir(self, chain_id: str) -> Path:
        validate_identifier(chain_id, field="chain_id")
        p = self.root / chain_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def freeze(self, registration: Mapping[str, Any]) -> dict[str, Any]:
        require_controller_execution_origin("ScientificChainController.freeze")
        reg = validate_chain_registration(registration)
        self._authority.require_run(reg)
        cdir = self.chain_dir(reg["chain_id"])
        rp = cdir / "chain_registration.json"
        if reg["schema_id"] == CHAIN_SCHEMA_V1 and not rp.exists() and not self.allow_legacy_new_registrations:
            raise ScientificChainValidationError("new V1 PREPARED_EXECUTOR chains are disabled; use V2 DECODER_STAGE")
        if rp.exists():
            old = json.loads(rp.read_text(encoding="utf-8"))
            if old != reg:
                raise ScientificChainError("conflicting frozen chain registration")
        else:
            write_json_atomic(rp, reg)
        (cdir / "stage_commits").mkdir(exist_ok=True)
        self._engineering_trust(cdir, add_session_key=True)
        _fsync_dir(cdir)
        return reg

    def _load_reg(self, chain_id: str) -> dict[str, Any]:
        p = self.chain_dir(chain_id) / "chain_registration.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        return validate_chain_registration(json.loads(p.read_text(encoding="utf-8")))

    def _state_path(self, cdir: Path) -> Path:
        return cdir / "CHAIN_STATE.json"

    def _events_path(self, cdir: Path) -> Path:
        return cdir / "events.jsonl"

    def _commit_path(self, cdir: Path, stage_id: str) -> Path:
        safe = stage_id.replace(":", "__")
        return cdir / "stage_commits" / f"{safe}.json"

    def _mirror_receipt_path(self, cdir: Path, stage_id: str) -> Path:
        safe = stage_id.replace(":", "__")
        return cdir / "mirror_receipts" / f"{safe}.json"

    def acknowledge_external_mirror(self, chain_id: str, stage_id: str, *, commit_sha256: str, mirror_uri: str, mirror_sha256: str) -> dict[str, Any]:
        require_controller_execution_origin("ScientificChainController.acknowledge_external_mirror")
        reg = self._load_reg(chain_id); cdir = self.chain_dir(chain_id)
        commit = self._load_commit(cdir, stage_id)
        if commit is None or commit.get("commit_sha256") != commit_sha256:
            raise ScientificChainError("external mirror receipt commit binding mismatch")
        _validate_sha(mirror_sha256, "mirror_sha256")
        if not isinstance(mirror_uri, str) or not mirror_uri:
            raise ScientificChainValidationError("mirror_uri required")
        base = {
            "schema_id": CHAIN_MIRROR_RECEIPT_SCHEMA,
            "chain_id": chain_id,
            "registration_sha256": reg["registration_sha256"],
            "stage_id": stage_id,
            "commit_sha256": commit_sha256,
            "mirror_uri": mirror_uri,
            "mirror_sha256": mirror_sha256,
            "status": "PASS",
            "acknowledged_utc": utc_now(),
        }
        obj = dict(base, receipt_sha256=canonical_sha256(base))
        p = self._mirror_receipt_path(cdir, stage_id); p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if old != obj:
                immutable = {k:v for k,v in obj.items() if k not in {"acknowledged_utc","receipt_sha256"}}
                old_immutable = {k:v for k,v in old.items() if k not in {"acknowledged_utc","receipt_sha256"}}
                if immutable != old_immutable:
                    raise ScientificChainError("conflicting external mirror receipt")
                return old
        write_json_atomic(p, obj); _fsync_dir(p.parent)
        st = self._load_state(reg, cdir)
        if st.get("status") == "PAUSED" and (st.get("stop") or {}).get("reason") == "EXTERNAL_MIRROR_REQUIRED" and (st.get("stop") or {}).get("stage") == stage_id:
            st["external_mirror_pending"] = False; st["status"] = "READY"; st["stop"] = None; self._write_state(cdir, st)
        self._event(cdir, chain_id=chain_id, kind="EXTERNAL_MIRROR_ACK", stage_id=stage_id, commit_sha256=commit_sha256, mirror_uri=mirror_uri, mirror_sha256=mirror_sha256)
        return obj

    def _mirror_acknowledged(self, cdir: Path, reg: dict[str, Any], stage_id: str, commit: Mapping[str, Any]) -> bool:
        p = self._mirror_receipt_path(cdir, stage_id)
        if not p.is_file(): return False
        obj = json.loads(p.read_text(encoding="utf-8"))
        expected = obj.get("receipt_sha256"); observed = canonical_sha256({k:v for k,v in obj.items() if k != "receipt_sha256"})
        return bool(expected == observed and obj.get("schema_id") == CHAIN_MIRROR_RECEIPT_SCHEMA and obj.get("registration_sha256") == reg["registration_sha256"] and obj.get("stage_id") == stage_id and obj.get("commit_sha256") == commit.get("commit_sha256") and obj.get("status") == "PASS")

    def _event(self, cdir: Path, *, chain_id: str, kind: str, **payload: Any) -> None:
        base = {
            "schema_id": CHAIN_EVENT_SCHEMA,
            "chain_id": chain_id,
            "kind": kind,
            "utc": utc_now(),
            "payload": payload,
        }
        base["event_sha256"] = canonical_sha256(base)
        _append_jsonl(self._events_path(cdir), base)

    def _initial_state(self, reg: dict[str, Any]) -> dict[str, Any]:
        first = reg["stages"][0]["stage_id"]
        st = {
            "schema_id": CHAIN_STATE_SCHEMA,
            "authoritative": False,
            "execution_role": ENGINEERING_ROLE,
            "chain_id": reg["chain_id"],
            "registration_sha256": reg["registration_sha256"],
            "status": "READY",
            "current_stage": first,
            "completed_stages": [],
            "stage_execution_count": 0,
            "started_utc": None,
            "updated_utc": utc_now(),
            "stop": None,
            "external_mirror_pending": bool(reg["durability"]["external_mirror_required"]),
        }
        st["state_sha256"] = canonical_sha256({k:v for k,v in st.items() if k != "state_sha256"})
        return st

    def _write_state(self, cdir: Path, state: dict[str, Any]) -> None:
        state = copy.deepcopy(state)
        state["updated_utc"] = utc_now()
        state["state_sha256"] = canonical_sha256({k:v for k,v in state.items() if k != "state_sha256"})
        write_json_atomic(self._state_path(cdir), state)
        _fsync_dir(cdir)

    def _load_state(self, reg: dict[str, Any], cdir: Path) -> dict[str, Any]:
        p = self._state_path(cdir)
        if not p.is_file():
            st = self._initial_state(reg)
            self._write_state(cdir, st)
            return st
        st = json.loads(p.read_text(encoding="utf-8"))
        expected = st.get("state_sha256")
        observed = canonical_sha256({k:v for k,v in st.items() if k != "state_sha256"})
        if expected != observed:
            raise ScientificChainError("chain state hash mismatch")
        if st.get("registration_sha256") != reg["registration_sha256"]:
            raise ScientificChainError("chain state registration mismatch")
        self._validate_state_cursor(reg, cdir, st)
        return st

    def _load_commit(self, cdir: Path, stage_id: str) -> dict[str, Any] | None:
        p = self._commit_path(cdir, stage_id)
        if not p.is_file():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        expected = obj.get("commit_sha256")
        observed = canonical_sha256({k:v for k,v in obj.items() if k != "commit_sha256"})
        if expected != observed or obj.get("schema_id") != CHAIN_STAGE_COMMIT_SCHEMA or obj.get("stage_id") != stage_id:
            raise ScientificChainError(f"stage commit integrity failure: {stage_id}")
        reg = self._load_reg(cdir.name)
        self._authority.require_run(reg)
        stage = next((x for x in reg["stages"] if x["stage_id"] == stage_id), None)
        if stage is None:
            raise ExecutionAuthorityError("COMMIT_STAGE_NOT_REGISTERED")
        receipt = obj.get("execution_receipt")
        if not isinstance(receipt, Mapping):
            raise ExecutionAuthorityError("MISSING_OR_MALFORMED_EXECUTION_RECEIPT")
        bindings = (receipt.get("payload") or {}).get("bindings")
        if not isinstance(bindings, Mapping):
            raise ExecutionAuthorityError("MISSING_OR_MALFORMED_EXECUTION_RECEIPT")
        expected_bindings = self._execution_bindings(cdir, reg, stage, run_id=bindings.get("run_id", ""))
        verified = verify_execution_receipt(
            receipt, expected_bindings=expected_bindings, result=obj["result"],
            trusted_public_keys=self._engineering_trust(cdir), required_role=ENGINEERING_ROLE,
        )
        if obj.get("authoritative") is not False or obj.get("science_sha256") != verified["science_sha256"]:
            raise ExecutionAuthorityError("COMMIT_AUTHORITY_OR_SCIENCE_MISMATCH")
        # Verify all duplicated fields; a valid signature must not bless edited
        # outer routing/outcome metadata with a freshly computed plain hash.
        expected_outcome = str(_json_pointer(obj["result"], stage["result_contract"]["outcome_pointer"]))
        if expected_outcome not in stage["result_contract"]["allowed_outcomes"]:
            expected_outcome = "__UNREGISTERED__"
        checks = {
            "chain_id": reg["chain_id"], "registration_sha256": reg["registration_sha256"],
            "question_sha256": stage["question_sha256"], "series": stage["series"],
            "result_sha256": canonical_sha256(obj["result"]), "outcome": expected_outcome,
        }
        if any(obj.get(k) != v for k, v in checks.items()):
            raise ExecutionAuthorityError("COMMIT_ROUTING_MISMATCH")
        return obj

    def _commit_stage(self, cdir: Path, reg: dict[str, Any], stage: dict[str, Any], execution: ChainExecutionResult) -> dict[str, Any]:
        require_controller_execution_origin("ScientificChainController._commit_stage")
        self._authority.require_run(reg)
        pending = self._pending_execution
        if pending is None or pending[0] is not execution:
            raise ExecutionAuthorityError("CONTROLLER_EXECUTION_TICKET_REQUIRED")
        approved_bindings, approved_result_sha = pending[1], pending[2]
        current_bindings = self._execution_bindings(cdir, reg, stage, run_id=approved_bindings["run_id"])
        result = copy.deepcopy(execution.result)
        if current_bindings != approved_bindings or digest(result) != approved_result_sha:
            raise ExecutionAuthorityError("EXECUTION_CHANGED_BEFORE_COMMIT")
        payload_sha = science_digest(result)
        receipt = self._authority.receipt(approved_bindings, result)
        verify_execution_receipt(receipt, expected_bindings=approved_bindings, result=result,
            trusted_public_keys=self._engineering_trust(cdir), required_role=ENGINEERING_ROLE)
        self._pending_execution = None  # single-use execution ticket
        outcome = _json_pointer(result, stage["result_contract"]["outcome_pointer"])
        if not isinstance(outcome, (str, int, float, bool)):
            raise ScientificChainError(f"stage outcome must be scalar: {stage['stage_id']}")
        outcome = str(outcome)
        if outcome not in stage["result_contract"]["allowed_outcomes"]:
            outcome = "__UNREGISTERED__"
        base = {
            "schema_id": CHAIN_STAGE_COMMIT_SCHEMA,
            "authoritative": False,
            "science_sha256": payload_sha,
            "execution_receipt": receipt,
            "chain_id": reg["chain_id"],
            "registration_sha256": reg["registration_sha256"],
            "stage_id": stage["stage_id"],
            "series": stage["series"],
            "question_sha256": stage["question_sha256"],
            "result_sha256": canonical_sha256(result),
            "result": result,
            "outcome": outcome,
            "run_record_sha256": canonical_sha256(execution.run_record) if execution.run_record is not None else None,
            "committed_utc": utc_now(),
        }
        obj = dict(base, commit_sha256=canonical_sha256(base))
        p = self._commit_path(cdir, stage["stage_id"])
        if p.exists():
            old = self._load_commit(cdir, stage["stage_id"])
            if old != obj:
                # timestamps differ on rerun; compare immutable scientific binding instead.
                immutable = {k:v for k,v in obj.items() if k not in {"committed_utc","commit_sha256"}}
                old_immutable = {k:v for k,v in old.items() if k not in {"committed_utc","commit_sha256"}}
                if immutable != old_immutable:
                    raise ScientificChainError(f"stage re-execution changed committed result: {stage['stage_id']}")
                return old
        write_json_atomic(p, obj)
        _fsync_dir(p.parent)
        return obj

    def _decoder_plan_executor(self, stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
        raise ExecutionAuthorityError("UNATTESTED_PLAN_PATH_DISABLED")

    def _execute_stage(self, stage: dict[str, Any], cdir: Path, reg: dict[str, Any]) -> ChainExecutionResult:
        require_controller_execution_origin("ScientificChainController._execute_stage")
        self._authority.require_run(reg)
        if stage["stage_kind"] == "HUMAN_REVIEW":
            raise ScientificChainReviewRequired(f"human review stage reached: {stage['stage_id']}")
        if stage["stage_kind"] != "DECODER_STAGE":
            raise ExecutionAuthorityError("UNATTESTED_EXECUTION_PATH_DISABLED")
        if self._pending_execution is not None:
            raise ExecutionAuthorityError("UNCOMMITTED_EXECUTION_EXISTS")
        fn = self._require_handler(stage)
        bindings = self._execution_bindings(cdir, reg, stage, run_id=new_run_id())
        permit = self._authority.begin(bindings)
        from .v05_stage_runtime import StageScienceRuntime
        params = dict(stage["execution"].get("parameters") or {})
        def load_dependency(sid):
            if sid not in stage["depends_on"]:
                raise ExecutionAuthorityError("DEPENDENCY_NOT_DECLARED", sid)
            return self._load_commit(cdir, sid)
        try:
            runtime = StageScienceRuntime(
                chain_dir=cdir, chain_id=reg["chain_id"], stage_id=stage["stage_id"],
                question_sha256=stage["question_sha256"],
                default_workers=int(params.get("workers", reg["budgets"].get("default_workers", 4))),
                memory_budget_bytes=params.get("reducer_memory_budget_bytes"),
                workspace_budget_bytes=params.get("workspace_budget_bytes"),
                _execution_permit=permit, _dependency_loader=load_dependency,
            )
            out = fn(copy.deepcopy(stage), runtime)
            if not isinstance(out, ChainExecutionResult):
                raise ScientificChainError("handler did not return ChainExecutionResult")
            if self._execution_bindings(cdir, reg, stage, run_id=bindings["run_id"]) != bindings:
                raise ExecutionAuthorityError("SOURCE_CHANGED_DURING_EXECUTION")
            # Do not let a caller-owned run_record/receipt authorize its result.
            science_digest(out.result)
            self._pending_execution = (out, bindings, digest(out.result))
            return out
        finally:
            permit.revoke()

    def run(self, registration_or_chain_id: Mapping[str, Any] | str) -> dict[str, Any]:
        require_controller_execution_origin("ScientificChainController.run")
        if not self._authority.enabled:
            self._authority.require_run({})  # stop before freeze/handler/science I/O
        if isinstance(registration_or_chain_id, Mapping):
            reg = self.freeze(registration_or_chain_id)
        else:
            reg = self._load_reg(str(registration_or_chain_id))
        self._authority.require_run(reg)
        cdir = self.chain_dir(reg["chain_id"])
        self._engineering_trust(cdir, add_session_key=True)
        state = self._load_state(reg, cdir)
        if state["status"] in {"COMPLETE", "FAILED"}:
            return state
        if state["started_utc"] is None:
            state["started_utc"] = utc_now()
        state["status"] = "RUNNING"
        self._write_state(cdir, state)
        self._event(cdir, chain_id=reg["chain_id"], kind="CHAIN_RUN_ENTER", current_stage=state["current_stage"])

        stages = {x["stage_id"]: x for x in reg["stages"]}
        t0 = time.monotonic()
        while True:
            no_deadline=os.environ.get('IG_DECODER_EXECUTION_POLICY')=='NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
            if not no_deadline and time.monotonic() - t0 > float(reg["budgets"]["max_chain_wall_seconds"]):
                state["status"] = "REVIEW_REQUIRED"; state["stop"] = {"reason":"CHAIN_WALL_BUDGET_EXHAUSTED","stage":state["current_stage"]}; self._write_state(cdir,state); return state
            sid = state["current_stage"]
            if sid is None:
                state["status"] = "COMPLETE"; state["stop"] = {"reason":"REGISTERED_END","stage":None}; self._write_state(cdir,state); return state
            stage = stages[sid]
            for dep in stage["depends_on"]:
                if dep not in state["completed_stages"]:
                    state["status"]="REVIEW_REQUIRED"; state["stop"]={"reason":"DEPENDENCY_NOT_COMMITTED","stage":sid,"dependency":dep}; self._write_state(cdir,state); return state
            if stage["stage_kind"] == "HUMAN_REVIEW":
                state["status"]="REVIEW_REQUIRED"; state["stop"]={"reason":"HUMAN_REVIEW_STAGE","stage":sid}; self._write_state(cdir,state); return state

            commit = self._load_commit(cdir, sid)
            if commit is None:
                # The stage-execution budget limits new executions, not post-commit
                # mirror acknowledgement or transition evaluation on resume.
                if state["stage_execution_count"] >= reg["budgets"]["max_stage_executions"]:
                    state["status"] = "REVIEW_REQUIRED"; state["stop"] = {"reason":"STAGE_BUDGET_EXHAUSTED","stage":sid}; self._write_state(cdir,state); return state
                self._event(cdir, chain_id=reg["chain_id"], kind="STAGE_START", stage_id=sid)
                try:
                    execution = self._execute_stage(stage, cdir, reg)
                    commit = self._commit_stage(cdir, reg, stage, execution)
                except ScientificChainReviewRequired:
                    state["status"]="REVIEW_REQUIRED"; state["stop"]={"reason":"HUMAN_REVIEW_STAGE","stage":sid}; self._write_state(cdir,state); return state
                except Exception as exc:
                    state["status"]="FAILED"; state["stop"]={"reason":"EXECUTION_FAILURE","stage":sid,"error":f"{type(exc).__name__}: {exc}"}; self._write_state(cdir,state); self._event(cdir,chain_id=reg["chain_id"],kind="STAGE_FAILED",stage_id=sid,error=state["stop"]["error"]); raise
                state["stage_execution_count"] += 1
                self._event(cdir, chain_id=reg["chain_id"], kind="STAGE_COMMIT", stage_id=sid, commit_sha256=commit["commit_sha256"], outcome=commit["outcome"])
            if sid not in state["completed_stages"]:
                state["completed_stages"].append(sid)

            if reg["durability"]["external_mirror_required"] and not self._mirror_acknowledged(cdir, reg, sid, commit):
                state["status"] = "PAUSED"
                state["external_mirror_pending"] = True
                state["stop"] = {"reason":"EXTERNAL_MIRROR_REQUIRED","stage":sid,"commit_sha256":commit["commit_sha256"]}
                self._write_state(cdir, state)
                return state
            state["external_mirror_pending"] = False

            outcome = commit["outcome"]
            if outcome == "__UNREGISTERED__":
                state["status"]="REVIEW_REQUIRED"; state["stop"]={"reason":"UNREGISTERED_OUTCOME","stage":sid}; self._write_state(cdir,state); return state
            rule = stage["transitions"][outcome]
            action = rule["action"]
            if action == "NEXT":
                nxt = rule["next_stage"]
                # auto_run=False is an intentional chain boundary after the current stage.
                state["current_stage"] = nxt
                self._write_state(cdir, state)
                self._event(cdir,chain_id=reg["chain_id"],kind="TRANSITION",stage_id=sid,outcome=outcome,next_stage=nxt)
                if not stages[nxt]["auto_run"]:
                    state["status"]="PAUSED"; state["stop"]={"reason":"REGISTERED_PAUSE_BEFORE_STAGE","stage":nxt}; self._write_state(cdir,state); return state
                continue
            if action == "END":
                state["current_stage"] = None; state["status"]="COMPLETE"; state["stop"]={"reason":rule["reason"],"stage":sid,"outcome":outcome}; self._write_state(cdir,state); return state
            if action == "REVIEW":
                state["status"]="REVIEW_REQUIRED"; state["stop"]={"reason":rule["reason"],"stage":sid,"outcome":outcome}; self._write_state(cdir,state); return state
            state["status"]="FAILED"; state["stop"]={"reason":rule["reason"],"stage":sid,"outcome":outcome}; self._write_state(cdir,state); return state

    def resume(self, chain_id: str) -> dict[str, Any]:
        require_controller_execution_origin("ScientificChainController.resume")
        reg = self._load_reg(chain_id)
        self._authority.require_run(reg)
        cdir = self.chain_dir(chain_id)
        st = self._load_state(reg, cdir)
        if st["status"] == "PAUSED":
            st["status"] = "READY"; st["stop"] = None; self._write_state(cdir, st)
        elif st["status"] == "REVIEW_REQUIRED":
            raise ScientificChainReviewRequired("review-required chain cannot resume without a new/continued frozen registration")
        return self.run(chain_id)

    def status(self, chain_id: str) -> dict[str, Any]:
        reg = self._load_reg(chain_id)
        self._authority.require_run(reg)
        cdir = self.chain_dir(chain_id); st = self._load_state(reg,cdir)
        commits = []
        for s in reg["stages"]:
            c = self._load_commit(cdir, s["stage_id"])
            if c is not None:
                commits.append({"stage_id":s["stage_id"],"outcome":c["outcome"],"commit_sha256":c["commit_sha256"],"result_sha256":c["result_sha256"]})
        return {
            "schema_id":"IG_DECODER_V05_SCIENTIFIC_CHAIN_STATUS_V2_AUTHENTICATED",
            "authoritative": False,
            "execution_role": ENGINEERING_ROLE,
            "chain_id":chain_id,
            "registration_sha256":reg["registration_sha256"],
            "status":st["status"],
            "current_stage":st["current_stage"],
            "completed_stages":list(st["completed_stages"]),
            "stage_execution_count":st["stage_execution_count"],
            "stop":copy.deepcopy(st["stop"]),
            "commits":commits,
            "external_mirror_pending":st["external_mirror_pending"],
        }
