from __future__ import annotations

"""Decoder v0.5 P6 generic structural/R workflow machinery.

This module is deliberately orchestration-only.  Mathematical semantics stay in
P5 science adapters and registered specialist capabilities.  The workflow layer
must not assume CAPS7, trees, H classes, or that a new law appears at every level.
"""

from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files
from typing import Any, Callable, Mapping, Sequence
import copy
import inspect
import json

from .canon import canonical_sha256, write_json_atomic
from .science_adapter_registry import get_science_adapter
from .uplift_architecture import structural_program


WORKFLOW_SCHEMA = "IG_DECODER_V05_WORKFLOW_REGISTRATION_V1"
WORKFLOW_RESULT_SCHEMA = "IG_DECODER_V05_WORKFLOW_RESULT_V1"
R_EVALUATION_SCHEMA = "IG_DECODER_V05_R_EVALUATION_V1"
STOP_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_STOP_V1"
PARENT_CAPSULE_SCHEMA = "IG_DECODER_V05_PARENT_CANDIDATE_CAPSULE_V1"
PARENT_COMMIT_SCHEMA = "IG_DECODER_V05_ELIGIBLE_PARENT_COMMIT_V1"


class WorkflowError(RuntimeError):
    pass


class UnknownWorkflowCapability(WorkflowError):
    pass


class WorkflowValidationError(WorkflowError):
    pass


class WorkflowCaseBudgetExceeded(WorkflowError):
    def __init__(self, *, requested: int, used: int, max_cases: int):
        super().__init__(f"workflow case budget exceeded requested={requested} used={used} max={max_cases}")
        self.requested = int(requested)
        self.used = int(used)
        self.max_cases = int(max_cases)


@dataclass
class WorkflowExecutionContext:
    """Per-capability logical case budget used during enumeration, not after it."""
    max_cases: int
    cases_used: int = 0

    @property
    def remaining_cases(self) -> int:
        return max(0, int(self.max_cases) - int(self.cases_used))

    def reserve_cases(self, count: int = 1) -> None:
        count = int(count)
        if count < 0:
            raise WorkflowValidationError("case reservation must be nonnegative")
        if self.cases_used + count > self.max_cases:
            raise WorkflowCaseBudgetExceeded(requested=count, used=self.cases_used, max_cases=self.max_cases)
        self.cases_used += count


def register_workflow_semantic_contract(capability_id: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Register a versioned machine-checkable semantic contract for a capability."""
    cid = str(capability_id)
    base = copy.deepcopy(dict(contract))
    required = {"contract_id", "version", "workflow_kind", "adapter_family", "public_descriptor"}
    if not required.issubset(base):
        raise ValueError(f"semantic contract missing fields for {cid}: {sorted(required-set(base))}")
    if ("parameter_keys" in base) == ("parameter_key_sets" in base):
        raise ValueError(f"semantic contract requires exactly one of parameter_keys/parameter_key_sets for {cid}")
    base["capability_id"] = cid
    sealed = dict(base)
    sealed["semantic_contract_sha256"] = canonical_sha256(base)
    old = _CAPABILITY_CONTRACTS.get(cid)
    if old is not None and old != sealed:
        raise RuntimeError(f"conflicting workflow semantic contract {cid}")
    _CAPABILITY_CONTRACTS[cid] = sealed
    return copy.deepcopy(sealed)


def get_workflow_semantic_contract(capability_id: str) -> dict[str, Any] | None:
    _ensure_builtin_capabilities()
    obj = _CAPABILITY_CONTRACTS.get(str(capability_id))
    return copy.deepcopy(obj) if obj is not None else None


def _invoke_capability(fn: CapabilityFn, *, execution_context: WorkflowExecutionContext, **kwargs) -> Mapping[str, Any]:
    params = inspect.signature(fn).parameters
    if "execution_context" in params:
        kwargs["execution_context"] = execution_context
    return fn(**kwargs)


def _validate_capability_semantics(*, reg: Mapping[str, Any], row: Mapping[str, Any] | None, capability_id: str, parameters: Mapping[str, Any], registered_alternatives: Sequence[str]) -> str | None:
    contract = get_workflow_semantic_contract(capability_id)
    if contract is None:
        return None
    if contract["workflow_kind"] != reg["workflow_kind"]:
        raise WorkflowValidationError(f"capability {capability_id} workflow_kind semantic mismatch")
    desc = get_science_adapter(str(reg["adapter_id"])).descriptor()
    if contract["adapter_family"] != "ANY" and desc.get("family") != contract["adapter_family"]:
        raise WorkflowValidationError(f"capability {capability_id} adapter family semantic mismatch")
    if contract["public_descriptor"] != "ANY" and desc.get("public_descriptor") != contract["public_descriptor"]:
        raise WorkflowValidationError(f"capability {capability_id} public descriptor semantic mismatch")
    if contract.get("parameter_key_sets") is not None:
        allowed_key_sets = [set(x) for x in contract.get("parameter_key_sets") or []]
        if set(parameters) not in allowed_key_sets:
            raise WorkflowValidationError(f"capability {capability_id} parameter contract mismatch")
    else:
        expected_keys = set(contract.get("parameter_keys") or [])
        if set(parameters) != expected_keys:
            raise WorkflowValidationError(f"capability {capability_id} parameter contract mismatch")
    if row is not None:
        allowed_stages = set(contract.get("allowed_stages") or [])
        if allowed_stages and row.get("stage") not in allowed_stages:
            raise WorkflowValidationError(f"capability {capability_id} stage semantic mismatch")
        exact_alts = contract.get("registered_alternatives")
        if exact_alts is not None and list(registered_alternatives) != list(exact_alts):
            raise WorkflowValidationError(f"capability {capability_id} registered alternatives semantic mismatch")
        if contract.get("stage_parameter_binding") and parameters.get("stage") != row.get("stage"):
            raise WorkflowValidationError(f"capability {capability_id} stage/parameter binding mismatch")
        if contract.get("review_alternative_parameter"):
            alt = parameters.get(contract["review_alternative_parameter"])
            if alt not in registered_alternatives:
                raise WorkflowValidationError(f"capability {capability_id} review alternative not registered")
    else:
        exact = contract.get("r_question_exact") or {}
        q = reg.get("question") or {}
        for k, v in exact.items():
            if q.get(k) != v:
                raise WorkflowValidationError(f"capability {capability_id} R semantic field mismatch: {k}")
        exact_alts = contract.get("registered_alternatives")
        if exact_alts is not None and list(registered_alternatives) != list(exact_alts):
            raise WorkflowValidationError(f"capability {capability_id} registered alternatives semantic mismatch")
    return str(contract["semantic_contract_sha256"])


CapabilityFn = Callable[..., Mapping[str, Any]]
_CAPABILITIES: dict[str, CapabilityFn] = {}
_CAPABILITY_CONTRACTS: dict[str, dict[str, Any]] = {}


def register_workflow_capability(capability_id: str):
    if not isinstance(capability_id, str) or not capability_id:
        raise ValueError("capability_id must be nonempty")
    def deco(fn: CapabilityFn):
        old = _CAPABILITIES.get(capability_id)
        if old is not None and old is not fn:
            raise RuntimeError(f"conflicting P6 workflow capability {capability_id}")
        _CAPABILITIES[capability_id] = fn
        return fn
    return deco


def _ensure_builtin_capabilities() -> None:
    # Import only registers pure science capabilities.  It does not create a
    # controller, worker launcher, checkpoint manager, publisher or certifier.
    from . import p6_capabilities as _p6_capabilities  # noqa: F401
    from . import g5_capabilities as _g5_capabilities  # noqa: F401


def list_workflow_capabilities() -> tuple[str, ...]:
    _ensure_builtin_capabilities()
    return tuple(sorted(_CAPABILITIES))


def get_workflow_capability(capability_id: str) -> CapabilityFn:
    _ensure_builtin_capabilities()
    fn = _CAPABILITIES.get(str(capability_id))
    if fn is None:
        raise UnknownWorkflowCapability(f"unregistered P6 science capability {capability_id}")
    return fn


def s_stage_templates() -> tuple[dict[str, Any], ...]:
    """Return the frozen S0-S6 meanings from the existing uplift architecture."""
    rows = []
    for row in structural_program():
        rows.append({
            "stage": row["stage"],
            "name": row["name"],
            "question": row["question"],
            "promotion": row["promotion"],
        })
    if [x["stage"] for x in rows] != [f"S{i}" for i in range(7)]:
        raise WorkflowValidationError("frozen structural stage sequence changed")
    return tuple(rows)


R_ENTRY_POINTS = (
    "HYPOTHESIS_FAMILY",
    "Q_A_O_FREEZE",
    "EXACT_BASELINE",
    "ACTION_TRANSITION_EVALUATION",
    "COUNTEREXAMPLE_EXTRACTION",
    "REFINEMENT",
    "ABLATION",
    "FRESH_CHALLENGE",
    "PROOF_OBLIGATIONS",
)

P6_WORKFLOW_CONTRACT_RESOURCE = "resources/v05/P6_WORKFLOW_CONTRACT_V1.json"

def load_p6_workflow_contract() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(P6_WORKFLOW_CONTRACT_RESOURCE).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_DECODER_V05_P6_WORKFLOW_CONTRACT_V1":
        raise WorkflowValidationError("bad P6 workflow contract schema")
    expected = obj.get("science_sha256")
    observed = canonical_sha256({k: v for k, v in obj.items() if k != "science_sha256"})
    if expected != observed:
        raise WorkflowValidationError("P6 workflow contract hash mismatch")
    if [x.get("stage") for x in obj.get("structural_stage_templates", [])] != [f"S{i}" for i in range(7)]:
        raise WorkflowValidationError("P6 workflow contract stage sequence mismatch")
    if tuple(obj.get("r_entry_points", [])) != R_ENTRY_POINTS:
        raise WorkflowValidationError("P6 workflow contract R entry points mismatch")
    return obj



_ALLOWED_WORKFLOW_KINDS = {"STRUCTURAL_S0_S6", "R_INVESTIGATION"}
_ALLOWED_WORKFLOW_MODES = {"DISCOVERY", "LAW_TRANSPORT", "RECURSIVE_DEPTH_APPLICATION"}
_ALLOWED_STAGE_OUTCOMES = {"PASS", "FALSIFIED", "REVIEW_REQUIRED", "BLOCKED"}
_ALLOWED_PREDICATES = {"PARENT_CONDITIONED", "STANDALONE_ACTION"}


def _validate_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise WorkflowValidationError(f"{name} must be lowercase sha256")
    return value


def _base_without_hash(reg: Mapping[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in reg.items() if k != "registration_sha256"}


def validate_workflow_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    load_p6_workflow_contract()
    if not isinstance(registration, Mapping):
        raise WorkflowValidationError("workflow registration must be object")
    reg = copy.deepcopy(dict(registration))
    required = {
        "schema_id", "registration_id", "status", "workflow_kind", "workflow_mode",
        "adapter_id", "subject", "question", "budgets", "stop_rules",
        "promotion_policy", "continuation", "registration_sha256",
    }
    if set(reg) != required:
        raise WorkflowValidationError(
            f"workflow registration fields mismatch missing={sorted(required-set(reg))} extra={sorted(set(reg)-required)}"
        )
    if reg["schema_id"] != WORKFLOW_SCHEMA:
        raise WorkflowValidationError("bad workflow registration schema")
    if reg["status"] != "ACTIVE_CANDIDATE":
        raise WorkflowValidationError("workflow registration is not ACTIVE_CANDIDATE")
    if reg["workflow_kind"] not in _ALLOWED_WORKFLOW_KINDS:
        raise WorkflowValidationError("unsupported workflow_kind")
    if reg["workflow_mode"] not in _ALLOWED_WORKFLOW_MODES:
        raise WorkflowValidationError("unsupported workflow_mode")
    if not isinstance(reg["registration_id"], str) or not reg["registration_id"]:
        raise WorkflowValidationError("registration_id must be nonempty")
    get_science_adapter(str(reg["adapter_id"]))  # fail early on unsupported family

    budgets = reg["budgets"]
    if not isinstance(budgets, Mapping) or set(budgets) != {"max_steps", "max_cases"}:
        raise WorkflowValidationError("budgets must contain max_steps/max_cases")
    if not isinstance(budgets["max_steps"], int) or budgets["max_steps"] < 0:
        raise WorkflowValidationError("max_steps must be nonnegative int")
    if not isinstance(budgets["max_cases"], int) or budgets["max_cases"] < 1:
        raise WorkflowValidationError("max_cases must be positive int")

    stop = reg["stop_rules"]
    expected_stop_keys = {
        "novelty_outside_registered_alternatives", "changed_assumptions",
        "budget_exhausted", "unmet_proof_obligations", "falsification",
    }
    if not isinstance(stop, Mapping) or set(stop) != expected_stop_keys or any(v != "STOP" for v in stop.values()):
        raise WorkflowValidationError("P6 stop_rules must be exact STOP policy")

    pp = reg["promotion_policy"]
    if not isinstance(pp, Mapping) or set(pp) != {"graduation_requested", "graduation_authorized", "authority_effect"}:
        raise WorkflowValidationError("promotion_policy must be exact")
    if pp["authority_effect"] != "NONE_P6":
        raise WorkflowValidationError("P6 workflow cannot carry promotion authority")
    if not isinstance(pp["graduation_requested"], bool) or not isinstance(pp["graduation_authorized"], bool):
        raise WorkflowValidationError("graduation flags must be bool")
    if pp["graduation_authorized"] is True:
        raise WorkflowValidationError("P6 generic workflow cannot authorize graduation")

    cont = reg["continuation"]
    if not isinstance(cont, Mapping) or set(cont) != {"of_registration_sha256", "version"}:
        raise WorkflowValidationError("continuation must be exact")
    if cont["of_registration_sha256"] is not None:
        _validate_sha(cont["of_registration_sha256"], "continuation.of_registration_sha256")
        if not isinstance(cont["version"], int) or cont["version"] < 1:
            raise WorkflowValidationError("continuation version must be positive")
    elif cont["version"] != 0:
        raise WorkflowValidationError("root workflow continuation version must be 0")

    q = reg["question"]
    if reg["workflow_kind"] == "STRUCTURAL_S0_S6":
        required_q = {"scope", "stage_rows"}
        if not isinstance(q, Mapping) or set(q) != required_q:
            raise WorkflowValidationError("structural question must contain scope/stage_rows")
        rows = q["stage_rows"]
        if not isinstance(rows, list) or len(rows) != 7:
            raise WorkflowValidationError("structural workflow requires exactly S0-S6")
        expected = [f"S{i}" for i in range(7)]
        if [x.get("stage") for x in rows] != expected:
            raise WorkflowValidationError("structural stage order must be S0-S6")
        for row in rows:
            if set(row) != {"stage", "capability_id", "parameters", "registered_alternatives"}:
                raise WorkflowValidationError("structural stage row fields mismatch")
            if not isinstance(row["parameters"], Mapping):
                raise WorkflowValidationError("stage parameters must be object")
            if not isinstance(row["registered_alternatives"], list):
                raise WorkflowValidationError("registered_alternatives must be list")
            get_workflow_capability(row["capability_id"])
            _validate_capability_semantics(
                reg=reg, row=row, capability_id=row["capability_id"],
                parameters=row["parameters"], registered_alternatives=row["registered_alternatives"],
            )
    else:
        required_q = {
            "legal_domain", "Q", "A", "O", "predicate", "exact_baseline",
            "candidate_family", "freshness_role", "proof_obligations",
            "capability_id", "parameters", "registered_alternatives",
        }
        if not isinstance(q, Mapping) or set(q) != required_q:
            raise WorkflowValidationError("R question fields mismatch")
        if q["predicate"] not in _ALLOWED_PREDICATES:
            raise WorkflowValidationError("unsupported R predicate")
        if not all(isinstance(q[k], str) and q[k] for k in ("legal_domain", "Q", "A", "O", "exact_baseline", "candidate_family", "freshness_role")):
            raise WorkflowValidationError("R semantic fields must be nonempty strings")
        if not isinstance(q["proof_obligations"], list):
            raise WorkflowValidationError("proof_obligations must be list")
        if not isinstance(q["parameters"], Mapping):
            raise WorkflowValidationError("R parameters must be object")
        if not isinstance(q["registered_alternatives"], list):
            raise WorkflowValidationError("R registered_alternatives must be list")
        get_workflow_capability(q["capability_id"])
        _validate_capability_semantics(
            reg=reg, row=None, capability_id=q["capability_id"],
            parameters=q["parameters"], registered_alternatives=q["registered_alternatives"],
        )

    if reg["workflow_kind"] == "STRUCTURAL_S0_S6" and any(str(r.get("capability_id", "")).startswith("g5.") for r in reg["question"]["stage_rows"]):
        active = [r for r in reg["question"]["stage_rows"] if r.get("capability_id") != "g5.stage.locked.v1"]
        if active:
            last_stage = str(active[-1]["stage"])
            subject_expect = {
                "S1": {"hierarchy":"G", "level":5, "parent":"G5:S0", "purpose":"PAIR_CONNECTION_CENSUS", "stage":"S1"},
                "S2": {"hierarchy":"G", "level":5, "parent":"G5:S1", "purpose":"PAIR_OBSERVER_QUOTIENT", "stage":"S2"},
                "S3": {"hierarchy":"G", "level":5, "parent":"G5:S2", "purpose":"HIGHER_ORDER_RESIDUAL", "stage":"S3"},
            }.get(last_stage)
            if subject_expect is not None and reg.get("subject") != subject_expect:
                raise WorkflowValidationError(f"G5 {last_stage} subject semantic contract mismatch")
            scope = reg["question"].get("scope")
            if not isinstance(scope, str) or f"G5:{last_stage}" not in scope:
                raise WorkflowValidationError(f"G5 {last_stage} scope semantic contract mismatch")

    declared = _validate_sha(reg["registration_sha256"], "registration_sha256")
    observed = canonical_sha256(_base_without_hash(reg))
    if declared != observed:
        raise WorkflowValidationError("workflow registration hash mismatch")
    return reg


def seal_workflow_registration(record_without_hash: Mapping[str, Any]) -> dict[str, Any]:
    base = copy.deepcopy(dict(record_without_hash))
    base.pop("registration_sha256", None)
    base["registration_sha256"] = canonical_sha256(base)
    return validate_workflow_registration(base)


def _stop_record(*, registration_sha256: str, reason: str, stage: str | None, details: Mapping[str, Any], continuation_required: bool) -> dict[str, Any]:
    out = {
        "schema_id": STOP_SCHEMA,
        "registration_sha256": registration_sha256,
        "reason": str(reason),
        "stage": stage,
        "details": copy.deepcopy(dict(details)),
        "continuation_required": bool(continuation_required),
        "authority_effect": "NONE_P6",
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _projection_hash(value: Any) -> str:
    return canonical_sha256(value)


def evaluate_r_rows(*, rows: Sequence[Mapping[str, Any]], predicate: str, max_cases: int) -> dict[str, Any]:
    if predicate not in _ALLOWED_PREDICATES:
        raise WorkflowValidationError("bad R predicate")
    if len(rows) > max_cases:
        raise WorkflowError("R_CASE_BUDGET_EXCEEDED")
    legal = [copy.deepcopy(dict(r)) for r in rows if bool(r.get("legal", False))]
    required = {"parent_id", "action_id", "legal", "Q", "A", "O", "exact_action"}
    for row in legal:
        if set(row) != required:
            raise WorkflowValidationError("R evaluation row fields mismatch")
    groups: dict[str, list[dict[str, Any]]] = {}
    q_fibres: dict[str, dict[str, Any]] = {}
    exact_actions: set[str] = set()
    child_outcomes: set[str] = set()
    for row in legal:
        qh = _projection_hash(row["Q"])
        ah = _projection_hash(row["A"])
        oh = _projection_hash(row["O"])
        eh = _projection_hash(row["exact_action"])
        exact_actions.add(eh); child_outcomes.add(oh)
        key = _projection_hash({"Q": row["Q"], "A": row["A"]}) if predicate == "PARENT_CONDITIONED" else ah
        enriched = dict(row, _Q_sha256=qh, _A_sha256=ah, _O_sha256=oh, _exact_sha256=eh)
        groups.setdefault(key, []).append(enriched)
        fib = q_fibres.setdefault(qh, {"group_keys": set(), "exact_actions": set(), "rows": 0})
        fib["group_keys"].add(key); fib["exact_actions"].add(eh); fib["rows"] += 1
    conflicts = []
    for key, members in sorted(groups.items()):
        outs: dict[str, list[dict[str, Any]]] = {}
        for m in members:
            outs.setdefault(m["_O_sha256"], []).append(m)
        if len(outs) > 1:
            witness_rows = []
            for _, ms in sorted(outs.items()):
                witness_rows.append({k: v for k, v in ms[0].items() if not k.startswith("_")})
                if len(witness_rows) == 2:
                    break
            conflicts.append({
                "group_key_sha256": key,
                "distinct_child_observations": len(outs),
                "witness_rows": witness_rows,
            })
    candidate_classes = len(groups)
    exact_count = len(exact_actions)
    fibre_rows = []
    for qh, fib in sorted(q_fibres.items()):
        cc = len(fib["group_keys"]); ec = len(fib["exact_actions"])
        fibre_rows.append({
            "Q_sha256": qh,
            "legal_rows": int(fib["rows"]),
            "candidate_action_classes": cc,
            "exact_action_cases": ec,
            "information_reduction_vs_exact_actions": ec - cc,
            "nontrivial_information_reduction": cc < ec,
        })
    out = {
        "schema_id": R_EVALUATION_SCHEMA,
        "predicate": predicate,
        "legal_rows": len(legal),
        "exact_action_cases": exact_count,
        "candidate_action_classes": candidate_classes,
        "exact_child_observations": len(child_outcomes),
        "Q_fibres": len(q_fibres),
        "per_Q_fibre": fibre_rows,
        "nonpredictive_classes": len(conflicts),
        "predictive": len(conflicts) == 0,
        "information_reduction_vs_exact_actions": exact_count - candidate_classes,
        "nontrivial_information_reduction": candidate_classes < exact_count,
        "conflicts": conflicts,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


@dataclass
class ControlledWorkflowEngine:
    """Pure workflow layer intended to run under the existing v0.5 Controller/runtime."""

    def run(self, registration: Mapping[str, Any], *, task_journal=None) -> dict[str, Any]:
        reg = validate_workflow_registration(registration)
        adapter = get_science_adapter(reg["adapter_id"])
        if reg["workflow_kind"] == "STRUCTURAL_S0_S6":
            return self._run_structural(reg, adapter, task_journal=task_journal)
        return self._run_r(reg, adapter, task_journal=task_journal)

    def _run_structural(self, reg: dict[str, Any], adapter: Any, *, task_journal=None) -> dict[str, Any]:
        stage_outputs = []
        stopped = None
        if reg["budgets"]["max_steps"] < 7:
            stopped = _stop_record(
                registration_sha256=reg["registration_sha256"], reason="BUDGET_EXHAUSTED",
                stage=None, details={"required_steps": 7, "max_steps": reg["budgets"]["max_steps"]}, continuation_required=True,
            )
        else:
            for row in reg["question"]["stage_rows"]:
                fn = get_workflow_capability(row["capability_id"])
                task_id = f"workflow-{row['stage'].lower()}"
                existing = task_journal.load(task_id) if task_journal is not None else None
                if existing is not None:
                    stage_rec = copy.deepcopy(existing["payload"])
                    if stage_rec.get("stage") != row["stage"] or stage_rec.get("capability_id") != row["capability_id"]:
                        raise WorkflowError(f"durable workflow stage binding mismatch: {row['stage']}")
                    raw = {"outcome": stage_rec["outcome"], "observed_alternative": stage_rec["observed_alternative"], **copy.deepcopy(stage_rec.get("result") or {})}
                    outcome = str(stage_rec["outcome"])
                    alternative = str(stage_rec["observed_alternative"])
                    stage_outputs.append(stage_rec)
                else:
                    exec_ctx = WorkflowExecutionContext(max_cases=int(reg["budgets"]["max_cases"]))
                    try:
                        raw = dict(_invoke_capability(
                            fn, execution_context=exec_ctx, adapter=adapter, stage=row["stage"],
                            parameters=dict(row["parameters"]), prior=tuple(stage_outputs), registration=reg,
                        ))
                    except WorkflowCaseBudgetExceeded as exc:
                        stopped = _stop_record(
                            registration_sha256=reg["registration_sha256"], reason="BUDGET_EXHAUSTED",
                            stage=row["stage"], details={"used_cases": exc.used, "requested_cases": exc.requested, "max_cases": exc.max_cases},
                            continuation_required=True,
                        )
                        break
                    outcome = str(raw.get("outcome", "BLOCKED"))
                    alternative = str(raw.get("observed_alternative", outcome))
                    if outcome not in _ALLOWED_STAGE_OUTCOMES:
                        raise WorkflowValidationError(f"capability returned unsupported outcome {outcome}")
                    stage_rec = {
                        "schema_id": "IG_DECODER_V05_S_STAGE_RESULT_V1",
                        "stage": row["stage"],
                        "template": next(x for x in s_stage_templates() if x["stage"] == row["stage"]),
                        "capability_id": row["capability_id"],
                        "outcome": outcome,
                        "observed_alternative": alternative,
                        "result": {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"outcome", "observed_alternative"}},
                        "promotion": False,
                    }
                    stage_rec["science_sha256"] = canonical_sha256(stage_rec)
                    if task_journal is not None:
                        task_journal.commit(task_id, stage_rec)
                    stage_outputs.append(stage_rec)
                changed = list(raw.get("changed_assumptions") or [])
                unmet = list(raw.get("unmet_proof_obligations") or [])
                if changed:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="CHANGED_ASSUMPTIONS",
                        stage=row["stage"], details={"changed_assumptions": changed}, continuation_required=True,
                    ); break
                if unmet:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="UNMET_PROOF_OBLIGATIONS",
                        stage=row["stage"], details={"unmet": unmet}, continuation_required=True,
                    ); break
                if alternative not in row["registered_alternatives"]:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="NOVELTY_OUTSIDE_REGISTERED_ALTERNATIVES",
                        stage=row["stage"], details={"observed_alternative": alternative, "registered_alternatives": row["registered_alternatives"]}, continuation_required=True,
                    ); break
                if outcome in {"FALSIFIED", "REVIEW_REQUIRED", "BLOCKED"}:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason={"FALSIFIED":"FALSIFICATION","REVIEW_REQUIRED":"REVIEW_REQUIRED","BLOCKED":"BLOCKED"}[outcome],
                        stage=row["stage"], details={"observed_alternative": alternative}, continuation_required=outcome != "FALSIFIED",
                    ); break
                if row["stage"] == "S6" and reg["promotion_policy"]["graduation_requested"]:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="GRADUATION_REVIEW_REQUIRED",
                        stage="S6", details={"mechanical_gate": "PASS", "graduation_authorized": False}, continuation_required=True,
                    ); break
        out = {
            "schema_id": WORKFLOW_RESULT_SCHEMA,
            "registration_sha256": reg["registration_sha256"],
            "workflow_kind": reg["workflow_kind"],
            "workflow_mode": reg["workflow_mode"],
            "adapter_id": reg["adapter_id"],
            "stage_results": stage_outputs,
            "stop_record": stopped,
            "status": "SCIENTIFIC_STOP" if stopped else "COMPLETE",
            "authority_effect": "NONE_P6",
            "graduated": False,
        }
        out["science_sha256"] = canonical_sha256(out)
        return out

    def _run_r(self, reg: dict[str, Any], adapter: Any, *, task_journal=None) -> dict[str, Any]:
        q = reg["question"]
        stopped = None
        eval_result = None
        raw: dict[str, Any] = {}
        proof_status = {
            "registered": list(q["proof_obligations"]),
            "satisfied": [],
            "unmet": list(q["proof_obligations"]),
        }
        if reg["budgets"]["max_steps"] < 1:
            stopped = _stop_record(
                registration_sha256=reg["registration_sha256"], reason="BUDGET_EXHAUSTED",
                stage="R", details={"required_steps": 1, "max_steps": reg["budgets"]["max_steps"]}, continuation_required=True,
            )
        else:
            fn = get_workflow_capability(q["capability_id"])
            existing = task_journal.load("workflow-r") if task_journal is not None else None
            if existing is not None:
                raw = copy.deepcopy(existing["payload"])
            else:
                exec_ctx = WorkflowExecutionContext(max_cases=int(reg["budgets"]["max_cases"]))
                try:
                    raw = dict(_invoke_capability(fn, execution_context=exec_ctx, adapter=adapter, parameters=dict(q["parameters"]), registration=reg))
                except WorkflowCaseBudgetExceeded as exc:
                    raw = {}
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="BUDGET_EXHAUSTED", stage="R",
                        details={"used_cases": exc.used, "requested_cases": exc.requested, "max_cases": exc.max_cases}, continuation_required=True,
                    )
                if task_journal is not None and stopped is None:
                    task_journal.commit("workflow-r", raw)
            registered = set(str(x) for x in q["proof_obligations"])
            if stopped is None:
                changed = list(raw.get("changed_assumptions") or [])
                observed_alternative = str(raw.get("observed_alternative", "ROWS_RETURNED"))
                satisfied = set(str(x) for x in (raw.get("satisfied_proof_obligations") or []))
                explicitly_unmet = set(str(x) for x in (raw.get("unmet_proof_obligations") or []))
                unmet = sorted((registered - satisfied) | explicitly_unmet)
                proof_status = {
                    "registered": sorted(registered),
                    "satisfied": sorted(registered & satisfied),
                    "unmet": unmet,
                }
                if changed:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="CHANGED_ASSUMPTIONS",
                        stage="R", details={"changed_assumptions": changed}, continuation_required=True,
                    )
                elif observed_alternative not in q["registered_alternatives"]:
                    stopped = _stop_record(
                        registration_sha256=reg["registration_sha256"], reason="NOVELTY_OUTSIDE_REGISTERED_ALTERNATIVES",
                        stage="R", details={"observed_alternative": observed_alternative, "registered_alternatives": q["registered_alternatives"]}, continuation_required=True,
                    )
                else:
                    rows = raw.get("rows", [])
                    if len(rows) > reg["budgets"]["max_cases"]:
                        stopped = _stop_record(
                            registration_sha256=reg["registration_sha256"], reason="BUDGET_EXHAUSTED",
                            stage="R", details={"observed_cases": len(rows), "max_cases": reg["budgets"]["max_cases"]}, continuation_required=True,
                        )
                    else:
                        eval_result = evaluate_r_rows(rows=rows, predicate=q["predicate"], max_cases=reg["budgets"]["max_cases"])
                        if eval_result["nonpredictive_classes"]:
                            stopped = _stop_record(
                                registration_sha256=reg["registration_sha256"], reason="FALSIFICATION",
                                stage="R", details={"counterexample": eval_result["conflicts"][0], "candidate_family": q["candidate_family"]}, continuation_required=False,
                            )
                        elif unmet:
                            stopped = _stop_record(
                                registration_sha256=reg["registration_sha256"], reason="UNMET_PROOF_OBLIGATIONS",
                                stage="R", details={"unmet": unmet}, continuation_required=True,
                            )
        out = {
            "schema_id": WORKFLOW_RESULT_SCHEMA,
            "registration_sha256": reg["registration_sha256"],
            "workflow_kind": reg["workflow_kind"],
            "workflow_mode": reg["workflow_mode"],
            "adapter_id": reg["adapter_id"],
            "r_entry_points": list(R_ENTRY_POINTS),
            "question_freeze": {
                "legal_domain": q["legal_domain"], "Q": q["Q"], "A": q["A"], "O": q["O"], "predicate": q["predicate"],
                "exact_baseline": q["exact_baseline"], "candidate_family": q["candidate_family"], "freshness_role": q["freshness_role"],
            },
            "proof_obligation_status": proof_status,
            "evaluation": eval_result,
            "stop_record": stopped,
            "status": "SCIENTIFIC_STOP" if stopped else "COMPLETE_BOUNDED_SURVIVAL",
            "authority_effect": "NONE_P6",
            "graduated": False,
        }
        out["science_sha256"] = canonical_sha256(out)
        return out



class ParentEligibilityLedger:
    """Two-phase candidate -> verified eligible-parent commit used by controlled workflows."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def candidate_capsule(self, *, workflow_registration_sha256: str, parent_ref: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        _validate_sha(workflow_registration_sha256, "workflow_registration_sha256")
        out = {
            "schema_id": PARENT_CAPSULE_SCHEMA,
            "workflow_registration_sha256": workflow_registration_sha256,
            "parent_ref": str(parent_ref),
            "payload": copy.deepcopy(dict(payload)),
            "status": "CANDIDATE_UNVERIFIED",
        }
        out["capsule_sha256"] = canonical_sha256(out)
        p = self.root / f"candidate-{out['capsule_sha256']}.json"
        write_json_atomic(p, out)
        return out

    def commit_eligible(self, capsule: Mapping[str, Any], *, verification_status: str, trigger_passed: bool) -> dict[str, Any]:
        cap = copy.deepcopy(dict(capsule))
        expected = cap.pop("capsule_sha256", None)
        if expected != canonical_sha256(cap):
            raise WorkflowError("candidate capsule hash mismatch")
        if cap.get("status") != "CANDIDATE_UNVERIFIED":
            raise WorkflowError("candidate capsule status mismatch")
        if verification_status != "PASS" or trigger_passed is not True:
            raise WorkflowError("parent cannot commit before verification and trigger PASS")
        out = {
            "schema_id": PARENT_COMMIT_SCHEMA,
            "capsule_sha256": expected,
            "workflow_registration_sha256": cap["workflow_registration_sha256"],
            "parent_ref": cap["parent_ref"],
            "status": "ELIGIBLE_COMMITTED",
            "verification_status": verification_status,
            "trigger_passed": True,
        }
        out["commit_sha256"] = canonical_sha256(out)
        write_json_atomic(self.root / f"eligible-{out['commit_sha256']}.json", out)
        return out

    def commit_eligible_verified(
        self, capsule: Mapping[str, Any], *, verification_artifact: Mapping[str, Any],
        verified_parent_science_sha256: str, trigger_passed: bool,
    ) -> dict[str, Any]:
        """Commit eligibility from an immutable verifier artifact, not caller labels.

        New scientific child workflows should use this method.  The legacy
        ``commit_eligible`` remains only for replay compatibility.
        """
        cap = copy.deepcopy(dict(capsule))
        expected = cap.pop("capsule_sha256", None)
        if expected != canonical_sha256(cap) or cap.get("status") != "CANDIDATE_UNVERIFIED":
            raise WorkflowError("candidate capsule hash/status mismatch")
        _validate_sha(str(verified_parent_science_sha256), "verified_parent_science_sha256")
        ver = copy.deepcopy(dict(verification_artifact))
        ver_declared = ver.get("verification_sha256") or ver.get("science_sha256")
        if not isinstance(ver_declared, str) or len(ver_declared) != 64:
            raise WorkflowError("verification artifact lacks immutable sha256 identity")
        if str(ver.get("status")) != "PASS":
            raise WorkflowError("parent verification artifact is not PASS")
        artifact_parent = ver.get("verified_parent_science_sha256") or ver.get("primary_science_sha256")
        if artifact_parent != verified_parent_science_sha256:
            raise WorkflowError("verification artifact parent science binding mismatch")
        if trigger_passed is not True:
            raise WorkflowError("parent trigger did not pass")
        out = {
            "schema_id": "IG_DECODER_V05_ELIGIBLE_PARENT_COMMIT_V1_1",
            "capsule_sha256": expected,
            "workflow_registration_sha256": cap["workflow_registration_sha256"],
            "parent_ref": cap["parent_ref"],
            "status": "ELIGIBLE_COMMITTED",
            "verification_artifact_sha256": ver_declared,
            "verified_parent_science_sha256": verified_parent_science_sha256,
            "trigger_passed": True,
            "admission_mode": "VERIFIER_ARTIFACT_BOUND",
        }
        out["commit_sha256"] = canonical_sha256(out)
        write_json_atomic(self.root / f"eligible-{out['commit_sha256']}.json", out)
        return out

    @staticmethod
    def derive_child_packet(commit: Mapping[str, Any], *, child_payload: Mapping[str, Any]) -> dict[str, Any]:
        c = copy.deepcopy(dict(commit)); expected = c.pop("commit_sha256", None)
        if expected != canonical_sha256(c) or c.get("status") != "ELIGIBLE_COMMITTED":
            raise WorkflowError("child derivation requires a valid eligible-parent commit")
        out = {
            "schema_id": "IG_DECODER_V05_CHILD_PACKET_V1",
            "eligible_parent_commit_sha256": expected,
            "payload": copy.deepcopy(dict(child_payload)),
            "status": "DERIVED_FROM_VERIFIED_PARENT",
        }
        out["packet_sha256"] = canonical_sha256(out)
        return out


def continuation_from_stop(*, prior_registration: Mapping[str, Any], prior_result: Mapping[str, Any], new_registration_id: str, mutator: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    old = validate_workflow_registration(prior_registration)
    stop = prior_result.get("stop_record")
    if not isinstance(stop, Mapping) or stop.get("continuation_required") is not True:
        raise WorkflowError("prior workflow does not require a continuation")
    if stop.get("registration_sha256") != old["registration_sha256"]:
        raise WorkflowError("stop/registration binding mismatch")
    nxt = _base_without_hash(old)
    nxt["registration_id"] = str(new_registration_id)
    nxt["continuation"] = {
        "of_registration_sha256": old["registration_sha256"],
        "version": int(old["continuation"]["version"]) + 1,
    }
    if mutator is not None:
        mutator(nxt)
    return seal_workflow_registration(nxt)
