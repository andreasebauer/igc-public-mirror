from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .safety import validate_identifier, validate_relative_path


CONTRACT_VERSION_V11 = "1.1.0"
CONTRACT_VERSION_V12 = "1.2.0"
CONTRACT_VERSION_V13 = "1.3.0"
CONTRACT_VERSION_V14 = "1.4.0"
CONTRACT_VERSION = "1.5.0"
REGISTRATION_SCHEMA_V11 = "IG_DECODER_V05_EXPERIMENT_REGISTRATION_V1_1"
PLAN_BINDING_SCHEMA_V11 = "IG_DECODER_V05_PLAN_BINDING_V1_1"
ADMISSION_SCHEMA_V11 = "IG_DECODER_V05_ADMISSION_V1_1"
REGISTRATION_SCHEMA_V12 = "IG_DECODER_V05_EXPERIMENT_REGISTRATION_V1_2"
REGISTRATION_SCHEMA_V13 = "IG_DECODER_V05_EXPERIMENT_REGISTRATION_V1_3"
REGISTRATION_SCHEMA_V14 = "IG_DECODER_V05_EXPERIMENT_REGISTRATION_V1_4"
REGISTRATION_SCHEMA = "IG_DECODER_V05_EXPERIMENT_REGISTRATION_V1_5"
PLAN_BINDING_SCHEMA_V12 = "IG_DECODER_V05_PLAN_BINDING_V1_2"
PLAN_BINDING_SCHEMA_V13 = "IG_DECODER_V05_PLAN_BINDING_V1_3"
PLAN_BINDING_SCHEMA_V14 = "IG_DECODER_V05_PLAN_BINDING_V1_4"
PLAN_BINDING_SCHEMA = "IG_DECODER_V05_PLAN_BINDING_V1_5"
ADMISSION_SCHEMA_V12 = "IG_DECODER_V05_ADMISSION_V1_2"
ADMISSION_SCHEMA_V13 = "IG_DECODER_V05_ADMISSION_V1_3"
ADMISSION_SCHEMA_V14 = "IG_DECODER_V05_ADMISSION_V1_4"
ADMISSION_SCHEMA = "IG_DECODER_V05_ADMISSION_V1_5"


class V05AdmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class V05RunnerPolicy:
    runner: str
    operation: str
    call_style: str = "context"
    dataset_param: str = "fixture_dataset_sha256"
    registration_required: bool = True


RUNNER_POLICIES: dict[str, V05RunnerPolicy] = {}


def register_runner_policy(*, runner: str, operation: str, call_style: str = "context", dataset_param: str = "fixture_dataset_sha256") -> V05RunnerPolicy:
    if runner in RUNNER_POLICIES:
        old = RUNNER_POLICIES[runner]
        new = V05RunnerPolicy(runner=runner, operation=operation, call_style=call_style, dataset_param=dataset_param)
        if old != new:
            raise RuntimeError(f"conflicting v0.5 runner policy for {runner}")
        return old
    p = V05RunnerPolicy(runner=runner, operation=operation, call_style=call_style, dataset_param=dataset_param)
    RUNNER_POLICIES[runner] = p
    return p


def runner_policy(runner: str) -> V05RunnerPolicy | None:
    return RUNNER_POLICIES.get(runner)


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or value == "0" * 64:
        raise V05AdmissionError(f"{field} must be a nonzero lowercase sha256")
    return value


def _without_self_hash(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "registration_sha256"}


def _validate_common_registration(record: dict, required: set[str]) -> None:
    extra = set(record) - required
    missing = required - set(record)
    if missing:
        raise V05AdmissionError(f"registration missing fields: {sorted(missing)}")
    if extra:
        raise V05AdmissionError(f"registration has unknown fields: {sorted(extra)}")
    validate_identifier(record["registration_id"], field="registration_id")
    validate_identifier(record["stage_id"], field="stage_id")
    if record["status"] != "ACTIVE_CANDIDATE":
        raise V05AdmissionError("registration is not ACTIVE_CANDIDATE")
    if record["experiment_kind"] not in {"REPLAY", "TEST_ONLY", "FALSIFICATION", "OBSERVATION"}:
        raise V05AdmissionError("unsupported experiment kind")
    p = record["protocol"]
    if not isinstance(p, dict) or set(p) != {"protocol_id", "version", "descriptor_sha256"}:
        raise V05AdmissionError("protocol binding must be exact")
    _require_sha(p["descriptor_sha256"], "protocol.descriptor_sha256")
    if not isinstance(record["subject"], dict) or not record["subject"]:
        raise V05AdmissionError("subject must be a nonempty object")
    if not isinstance(record["runner"], str) or not record["runner"]:
        raise V05AdmissionError("runner missing")
    if not isinstance(record["operation"], str) or not record["operation"]:
        raise V05AdmissionError("operation missing")
    ids = record["input_datasets"]
    if not isinstance(ids, list) or not ids:
        raise V05AdmissionError("input_datasets must be nonempty")
    roles = set()
    for d in ids:
        if not isinstance(d, dict) or set(d) != {"dataset_sha256", "logical_role"}:
            raise V05AdmissionError("input dataset binding must be exact")
        _require_sha(d["dataset_sha256"], "input dataset sha256")
        if not isinstance(d["logical_role"], str) or not d["logical_role"]:
            raise V05AdmissionError("input dataset logical_role missing")
        if d["logical_role"] in roles:
            raise V05AdmissionError("duplicate input dataset logical_role")
        roles.add(d["logical_role"])
    _require_sha(record["source_sha256"], "source_sha256")
    _require_sha(record["environment_sha256"], "environment_sha256")
    if record["authority_ceiling"] not in {"TEST_ONLY", "ENGINEERING_ONLY", "SCOPED_AUDIT_AUTHORITY"}:
        raise V05AdmissionError("unsupported authority ceiling")
    oc = record["output_contract"]
    if not isinstance(oc, dict) or set(oc) != {"logical_outputs", "science_oracle"}:
        raise V05AdmissionError("output_contract must contain logical_outputs and science_oracle")
    if not isinstance(oc["logical_outputs"], list) or not oc["logical_outputs"]:
        raise V05AdmissionError("logical_outputs must be nonempty")
    if len(set(oc["logical_outputs"])) != len(oc["logical_outputs"]):
        raise V05AdmissionError("logical_outputs must be unique")
    if not all(isinstance(x, str) and x for x in oc["logical_outputs"]):
        raise V05AdmissionError("logical_outputs entries must be nonempty strings")
    if not isinstance(oc["science_oracle"], dict):
        raise V05AdmissionError("science_oracle must be an object")


def _validate_v12_policies(record: dict) -> None:
    wp = record["worker_policy"]
    if not isinstance(wp, dict) or set(wp) != {"kind", "uid", "gid", "store_access", "publication_access"}:
        raise V05AdmissionError("worker_policy must be exact")
    if wp["kind"] != "POSIX_DROP_PRIVILEGE":
        raise V05AdmissionError("unsupported worker isolation kind")
    if isinstance(wp["uid"], bool) or not isinstance(wp["uid"], int) or wp["uid"] < 1:
        raise V05AdmissionError("worker uid must be a positive integer")
    if isinstance(wp["gid"], bool) or not isinstance(wp["gid"], int) or wp["gid"] < 1:
        raise V05AdmissionError("worker gid must be a positive integer")
    if wp["store_access"] != "SNAPSHOT_ONLY" or wp["publication_access"] != "DENY":
        raise V05AdmissionError("worker policy must deny store/publication writes and use snapshots")

    vp = record["verification_policy"]
    required_v = {"verifier_id", "mode", "required_checks", "independent_process", "cold_recompute"}
    if not isinstance(vp, dict) or set(vp) != required_v:
        raise V05AdmissionError("verification_policy must be exact")
    validate_identifier(vp["verifier_id"], field="verifier_id")
    if vp["mode"] != "COLD_RECOMPUTE_EXACT":
        raise V05AdmissionError("unsupported verification mode")
    if vp["independent_process"] is not True or vp["cold_recompute"] is not True:
        raise V05AdmissionError("v1.2 verifier must be independent and cold-recompute")
    if not isinstance(vp["required_checks"], list) or not vp["required_checks"]:
        raise V05AdmissionError("verification required_checks must be nonempty")
    if len(set(vp["required_checks"])) != len(vp["required_checks"]):
        raise V05AdmissionError("verification required_checks must be unique")

    pp = record["publication_policy"]
    if not isinstance(pp, dict) or set(pp) != {"publisher_id", "required_verification_status", "authority_effect", "protected_storage"}:
        raise V05AdmissionError("publication_policy must be exact")
    validate_identifier(pp["publisher_id"], field="publisher_id")
    if pp["required_verification_status"] != "PASS":
        raise V05AdmissionError("publisher requires PASS verification")
    ver = record.get("contract_version")
    allowed_effect = "NONE_P2_6" if ver == CONTRACT_VERSION else ("NONE_P2_5" if ver == CONTRACT_VERSION_V14 else ("NONE_P2_4" if ver == CONTRACT_VERSION_V13 else "NONE_P2_3"))
    if pp["authority_effect"] != allowed_effect:
        raise V05AdmissionError("publication authority effect mismatch")
    if pp["protected_storage"] != "OWNER_ONLY_POSIX":
        raise V05AdmissionError("P2.3 requires owner-only protected publication storage")


def _validate_v13_execution_policy(record: dict) -> None:
    ep = record["execution_policy"]
    required = {"logical_task_checkpointing", "resume_policy", "heartbeat_interval_seconds", "terminal_telemetry", "packaging_state_separate"}
    if not isinstance(ep, dict) or set(ep) != required:
        raise V05AdmissionError("execution_policy must be exact")
    if ep["logical_task_checkpointing"] != "DURABLE_APPEND_ONLY":
        raise V05AdmissionError("unsupported logical task checkpointing policy")
    if ep["resume_policy"] != "REUSE_VERIFIED_TASK_COMMITS":
        raise V05AdmissionError("unsupported resume policy")
    hb = ep["heartbeat_interval_seconds"]
    if isinstance(hb, bool) or not isinstance(hb, (int, float)) or hb <= 0 or hb > 60:
        raise V05AdmissionError("heartbeat interval must be in (0,60]")
    if ep["terminal_telemetry"] != "REQUIRED" or ep["packaging_state_separate"] is not True:
        raise V05AdmissionError("terminal telemetry and separate packaging state are mandatory")




def _validate_v14_budget_and_comparison(record: dict) -> None:
    rb = record["resource_budget"]
    required_rb = {
        "wall_seconds_max", "cpu_seconds_max", "peak_rss_bytes_max",
        "bytes_read_max", "bytes_written_max", "workspace_peak_bytes_max",
        "checkpoint_bytes_max", "logical_tasks_max",
    }
    if not isinstance(rb, dict) or set(rb) != required_rb:
        raise V05AdmissionError("resource_budget must be exact")
    for k in ("wall_seconds_max", "cpu_seconds_max"):
        v = rb[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise V05AdmissionError(f"{k} must be positive")
    for k in ("peak_rss_bytes_max", "bytes_read_max", "bytes_written_max", "workspace_peak_bytes_max", "checkpoint_bytes_max", "logical_tasks_max"):
        v = rb[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0 or (k == "logical_tasks_max" and v < 1):
            raise V05AdmissionError(f"{k} must be a nonnegative integer")
    sr = record["stop_rules"]
    expected_sr = {
        "auto_expand": False, "wall_budget": "TERMINATE_WORKER",
        "cpu_budget": "OS_LIMIT", "memory_budget": "TERMINATE_WORKER",
        "storage_budget": "TERMINATE_WORKER", "task_budget": "FAIL_CLOSED",
        "scientific_mismatch": "STOP",
    }
    if sr != expected_sr:
        raise V05AdmissionError("unsupported stop_rules")
    cc = record["comparison_contract"]
    required_cc = {
        "comparison_id", "role", "baseline_source_sha256", "projection_fields",
        "expected_disposition", "candidate_predicate",
    }
    if record.get("contract_version") == CONTRACT_VERSION:
        required_cc = set(required_cc) | {"review_reason"}
    if not isinstance(cc, dict) or set(cc) != required_cc:
        raise V05AdmissionError("comparison_contract must be exact")
    validate_identifier(cc["comparison_id"], field="comparison_id")
    if cc["role"] not in ({"POSITIVE", "EXPECTED_FALSIFICATION", "RECOVERY", "REVIEW_REQUIRED"} if record.get("contract_version") == CONTRACT_VERSION else {"POSITIVE", "EXPECTED_FALSIFICATION", "RECOVERY"}):
        raise V05AdmissionError("unsupported comparison role")
    _require_sha(cc["baseline_source_sha256"], "comparison baseline source")
    pf = cc["projection_fields"]
    if not isinstance(pf, list) or not pf or len(set(pf)) != len(pf) or not all(isinstance(x, str) and x for x in pf):
        raise V05AdmissionError("comparison projection_fields invalid")
    expected = {"POSITIVE": "MATCH", "EXPECTED_FALSIFICATION": "FALSIFIED_AS_EXPECTED", "RECOVERY": "MATCH", "REVIEW_REQUIRED": "REVIEW_REQUIRED"}[cc["role"]]
    if cc["expected_disposition"] != expected:
        raise V05AdmissionError("comparison expected_disposition mismatch")
    pred = cc["candidate_predicate"]
    if not isinstance(pred, dict):
        raise V05AdmissionError("candidate_predicate must be an object")
    if cc["role"] == "EXPECTED_FALSIFICATION":
        if set(pred) != {"field", "equals"} or not isinstance(pred["field"], str):
            raise V05AdmissionError("expected falsification requires exact candidate predicate")
    elif pred:
        raise V05AdmissionError("candidate_predicate must be empty outside expected falsification")
    if record.get("contract_version") == CONTRACT_VERSION:
        rr = cc.get("review_reason")
        if cc["role"] == "REVIEW_REQUIRED":
            if not isinstance(rr, str) or not rr.strip():
                raise V05AdmissionError("review-required comparison needs nonempty review_reason")
        elif rr != "":
            raise V05AdmissionError("review_reason must be empty outside REVIEW_REQUIRED")

def validate_registration(record: dict) -> dict:
    if not isinstance(record, dict):
        raise V05AdmissionError("registration must be an object")
    common = {
        "schema_id", "contract_version", "registration_id", "status", "experiment_kind",
        "protocol", "subject", "runner", "operation", "stage_id", "input_datasets",
        "source_sha256", "environment_sha256", "authority_ceiling", "output_contract",
        "registration_sha256",
    }
    schema_id = record.get("schema_id")
    if schema_id == REGISTRATION_SCHEMA_V11:
        if record.get("contract_version") != CONTRACT_VERSION_V11:
            raise V05AdmissionError("unsupported v1.1 contract version")
        _validate_common_registration(record, common)
    elif schema_id == REGISTRATION_SCHEMA_V12:
        if record.get("contract_version") != CONTRACT_VERSION_V12:
            raise V05AdmissionError("unsupported v1.2 contract version")
        required = common | {"worker_policy", "verification_policy", "publication_policy"}
        _validate_common_registration(record, required)
        _validate_v12_policies(record)
    elif schema_id == REGISTRATION_SCHEMA_V13:
        if record.get("contract_version") != CONTRACT_VERSION_V13:
            raise V05AdmissionError("unsupported v1.3 contract version")
        required = common | {"worker_policy", "verification_policy", "publication_policy", "execution_policy"}
        _validate_common_registration(record, required)
        _validate_v12_policies(record)
        _validate_v13_execution_policy(record)
    elif schema_id == REGISTRATION_SCHEMA_V14:
        if record.get("contract_version") != CONTRACT_VERSION_V14:
            raise V05AdmissionError("unsupported v1.4 contract version")
        required = common | {"worker_policy", "verification_policy", "publication_policy", "execution_policy", "resource_budget", "stop_rules", "comparison_contract"}
        _validate_common_registration(record, required)
        _validate_v12_policies(record)
        _validate_v13_execution_policy(record)
        _validate_v14_budget_and_comparison(record)
    elif schema_id == REGISTRATION_SCHEMA:
        if record.get("contract_version") != CONTRACT_VERSION:
            raise V05AdmissionError("unsupported v1.5 contract version")
        required = common | {"worker_policy", "verification_policy", "publication_policy", "execution_policy", "resource_budget", "stop_rules", "comparison_contract"}
        _validate_common_registration(record, required)
        _validate_v12_policies(record)
        _validate_v13_execution_policy(record)
        _validate_v14_budget_and_comparison(record)
    else:
        raise V05AdmissionError("unsupported registration schema")
    declared = _require_sha(record["registration_sha256"], "registration_sha256")
    observed = canonical_sha256(_without_self_hash(record))
    if declared != observed:
        raise V05AdmissionError("registration hash mismatch")
    return record


def seal_registration(record_without_hash: dict) -> dict:
    base = dict(record_without_hash)
    base.pop("registration_sha256", None)
    out = dict(base, registration_sha256=canonical_sha256(base))
    return validate_registration(out)


class V05RegistrationStore:
    def __init__(self, store_root: Path):
        self.root = Path(store_root) / "v05" / "registrations"
        self.root.mkdir(parents=True, exist_ok=True)

    def install(self, record: dict) -> dict:
        from .v05_origin_guard import require_controller_execution_origin
        require_controller_execution_origin("V05RegistrationStore.install")
        record = validate_registration(dict(record))
        sha = record["registration_sha256"]
        p = self.root / f"{sha}.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if old != record:
                raise V05AdmissionError("registration identity collision")
        else:
            write_json_atomic(p, record)
        return record

    def get(self, sha: str) -> dict:
        _require_sha(sha, "registration_sha256")
        p = self.root / f"{sha}.json"
        if not p.is_file():
            raise V05AdmissionError("registration not installed")
        return validate_registration(json.loads(p.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class V05Admission:
    schema_id: str
    contract_version: str
    registration_sha256: str
    run_id: str
    stage_id: str
    runner: str
    operation: str
    allowed_dataset_sha256: tuple[str, ...]
    authority_ceiling: str
    experiment_kind: str
    logical_outputs: tuple[str, ...]
    worker_policy: dict | None
    verification_policy: dict | None
    publication_policy: dict | None
    execution_policy: dict | None
    resource_budget: dict | None
    stop_rules: dict | None
    comparison_contract: dict | None
    _issuer_token: object


class V05AdmissionService:
    def __init__(self, store_root: Path):
        self.store_root = Path(store_root)
        self.registrations = V05RegistrationStore(store_root)
        self._issuer_token = object()
        self.protected_publication_root = self.store_root / "v05" / "protected_publications"
        self.protected_publication_root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.protected_publication_root, 0o700)
        except OSError:
            pass

    def requires_guard(self, plan: dict) -> bool:
        return any(runner_policy(st.get("runner", "")) is not None for st in plan.get("stages", []))

    def registration_for_plan(self, plan: dict) -> dict | None:
        binding = plan.get("v05")
        if not isinstance(binding, dict):
            return None
        sha = binding.get("registration_sha256")
        if not isinstance(sha, str):
            return None
        return self.registrations.get(sha)

    def _validate_binding(self, binding: dict, reg: dict) -> None:
        expected = {"schema_id", "contract_version", "registration_sha256"}
        if reg["schema_id"] == REGISTRATION_SCHEMA_V11:
            ok = binding.get("schema_id") == PLAN_BINDING_SCHEMA_V11 and binding.get("contract_version") == CONTRACT_VERSION_V11
            label = "v1.1"
        elif reg["schema_id"] == REGISTRATION_SCHEMA_V12:
            ok = binding.get("schema_id") == PLAN_BINDING_SCHEMA_V12 and binding.get("contract_version") == CONTRACT_VERSION_V12
            label = "v1.2"
        elif reg["schema_id"] == REGISTRATION_SCHEMA_V13:
            ok = binding.get("schema_id") == PLAN_BINDING_SCHEMA_V13 and binding.get("contract_version") == CONTRACT_VERSION_V13
            label = "v1.3"
        elif reg["schema_id"] == REGISTRATION_SCHEMA_V14:
            ok = binding.get("schema_id") == PLAN_BINDING_SCHEMA_V14 and binding.get("contract_version") == CONTRACT_VERSION_V14
            label = "v1.4"
        else:
            ok = binding.get("schema_id") == PLAN_BINDING_SCHEMA and binding.get("contract_version") == CONTRACT_VERSION
            label = "v1.5"
        if set(binding) != expected or not ok:
            raise V05AdmissionError(f"{label} plan binding mismatch")
        if binding.get("registration_sha256") != reg["registration_sha256"]:
            raise V05AdmissionError("plan binding registration identity mismatch")

    def _enforce_runtime_worker_boundary(self, reg: dict) -> None:
        if reg["schema_id"] not in {REGISTRATION_SCHEMA_V12, REGISTRATION_SCHEMA_V13, REGISTRATION_SCHEMA_V14, REGISTRATION_SCHEMA}:
            return
        wp = reg["worker_policy"]
        if os.name != "posix" or not hasattr(os, "geteuid"):
            raise V05AdmissionError("required POSIX worker isolation unavailable")
        euid = os.geteuid()
        if euid != 0:
            raise V05AdmissionError("P2.3 POSIX worker isolation requires privileged controller to drop worker uid")
        if wp["uid"] == euid:
            raise V05AdmissionError("worker uid must differ from controller uid")
        st = self.protected_publication_root.stat()
        if st.st_uid != euid:
            raise V05AdmissionError("protected publication root not owned by controller")
        if st.st_mode & 0o077:
            raise V05AdmissionError("protected publication root grants group/other permissions")

    def admit_plan(self, *, plan: dict, code_sha: str, env_sha: str) -> dict[str, V05Admission]:
        guarded = [st for st in plan.get("stages", []) if runner_policy(st.get("runner", "")) is not None]
        if not guarded:
            return {}
        binding = plan.get("v05")
        if not isinstance(binding, dict) or "registration_sha256" not in binding:
            raise V05AdmissionError("guarded route requires exact v05 plan binding")
        reg = self.registrations.get(binding["registration_sha256"])
        self._validate_binding(binding, reg)
        self._enforce_runtime_worker_boundary(reg)
        if reg["source_sha256"] != code_sha:
            raise V05AdmissionError("registration source identity mismatch")
        if reg["environment_sha256"] != env_sha:
            raise V05AdmissionError("registration environment identity mismatch")
        proto = reg["protocol"]
        if proto != {
            "protocol_id": plan.get("protocol_id"),
            "version": plan.get("protocol_version"),
            "descriptor_sha256": plan.get("descriptor_sha256"),
        }:
            raise V05AdmissionError("registration protocol binding mismatch")
        if reg["subject"] != plan.get("subject"):
            raise V05AdmissionError("registration subject mismatch")
        plan_inputs = plan.get("input_datasets", [])
        if reg["input_datasets"] != plan_inputs:
            raise V05AdmissionError("registration input dataset binding mismatch")
        if len(guarded) != 1:
            raise V05AdmissionError("bounded v0.5 contract permits exactly one guarded stage")
        st = guarded[0]
        pol = runner_policy(st["runner"])
        assert pol is not None
        if reg["runner"] != st["runner"] or reg["operation"] != pol.operation or reg["stage_id"] != st["stage_id"]:
            raise V05AdmissionError("registration stage/runner/operation mismatch")
        dataset_sha = st.get("params", {}).get(pol.dataset_param)
        allowed = tuple(d["dataset_sha256"] for d in reg["input_datasets"])
        if dataset_sha not in allowed:
            raise V05AdmissionError(f"stage dataset parameter {pol.dataset_param} is not registered")
        return {
            st["stage_id"]: V05Admission(
                schema_id=(ADMISSION_SCHEMA if reg["schema_id"] == REGISTRATION_SCHEMA else ADMISSION_SCHEMA_V14 if reg["schema_id"] == REGISTRATION_SCHEMA_V14 else ADMISSION_SCHEMA_V13 if reg["schema_id"] == REGISTRATION_SCHEMA_V13 else ADMISSION_SCHEMA_V12 if reg["schema_id"] == REGISTRATION_SCHEMA_V12 else ADMISSION_SCHEMA_V11),
                contract_version=reg["contract_version"],
                registration_sha256=reg["registration_sha256"],
                run_id=plan["run_id"],
                stage_id=st["stage_id"],
                runner=st["runner"],
                operation=reg["operation"],
                allowed_dataset_sha256=allowed,
                authority_ceiling=reg["authority_ceiling"],
                experiment_kind=reg["experiment_kind"],
                logical_outputs=tuple(reg["output_contract"]["logical_outputs"]),
                worker_policy=dict(reg.get("worker_policy")) if reg.get("worker_policy") else None,
                verification_policy=dict(reg.get("verification_policy")) if reg.get("verification_policy") else None,
                publication_policy=dict(reg.get("publication_policy")) if reg.get("publication_policy") else None,
                execution_policy=dict(reg.get("execution_policy")) if reg.get("execution_policy") else None,
                resource_budget=dict(reg.get("resource_budget")) if reg.get("resource_budget") else None,
                stop_rules=dict(reg.get("stop_rules")) if reg.get("stop_rules") else None,
                comparison_contract=dict(reg.get("comparison_contract")) if reg.get("comparison_contract") else None,
                _issuer_token=self._issuer_token,
            )
        }

    def validate_admission(self, admission: V05Admission) -> None:
        if not isinstance(admission, V05Admission) or admission._issuer_token is not self._issuer_token:
            raise V05AdmissionError("invalid or foreign stage capability")


class V05StageContext:
    """Narrow in-process staging-only view retained for the v1.1 predecessor.

    P2.3 v1.2 registrations require the POSIX isolated worker path instead.
    """

    __slots__ = ("_service", "_admission", "_datasets", "_work_dir")

    def __init__(self, *, service: V05AdmissionService, admission: V05Admission, datasets, work_dir: Path):
        service.validate_admission(admission)
        self._service = service
        self._admission = admission
        self._datasets = datasets
        self._work_dir = Path(work_dir).resolve(strict=False)

    @property
    def run_id(self) -> str:
        return self._admission.run_id

    @property
    def stage_id(self) -> str:
        return self._admission.stage_id

    @property
    def operation(self) -> str:
        return self._admission.operation

    @property
    def registration_sha256(self) -> str:
        return self._admission.registration_sha256

    def materialize_dataset(self, dataset_sha256: str, relative_destination: str = "fixture") -> Path:
        self._service.validate_admission(self._admission)
        if dataset_sha256 not in self._admission.allowed_dataset_sha256:
            raise V05AdmissionError("dataset capability denied")
        rel = validate_relative_path(relative_destination)
        dst = (self._work_dir / rel).resolve(strict=False)
        try:
            dst.relative_to(self._work_dir)
        except ValueError as exc:
            raise V05AdmissionError("staging path escape") from exc
        return self._datasets.materialize(dataset_sha256, dst)

    def staging_path(self, relative_path: str) -> Path:
        self._service.validate_admission(self._admission)
        rel = validate_relative_path(relative_path)
        p = (self._work_dir / rel).resolve(strict=False)
        try:
            p.relative_to(self._work_dir)
        except ValueError as exc:
            raise V05AdmissionError("staging path escape") from exc
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
