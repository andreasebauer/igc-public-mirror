from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic
from .checkpoints import CheckpointManager
from .core.l2 import composite_abc, relation_digests
from .datasets import DatasetStore
from .hashing import sha256_file
from .records import live_source_sha256, runtime_sha256
from .v05_runtime import V05TelemetryLedger
from .store import ArtifactStore
from .v05 import REGISTRATION_SCHEMA, REGISTRATION_SCHEMA_V14, REGISTRATION_SCHEMA_V13, REGISTRATION_SCHEMA_V12, PLAN_BINDING_SCHEMA, PLAN_BINDING_SCHEMA_V14, PLAN_BINDING_SCHEMA_V13, PLAN_BINDING_SCHEMA_V12, V05RegistrationStore
from .v05_evidence import ENVELOPE_SCHEMA, ENVELOPE_SCHEMA_V14, ENVELOPE_SCHEMA_V13, ENVELOPE_SCHEMA_V12
from .v05_scout_l15_verifier import replay_scout_l15, scout_science_projection
from .v05_oscout_phase0_verifier import replay_oscout_phase0, oscout_phase0_science_projection


VERIFICATION_SCHEMA_V12 = "IG_DECODER_V05_VERIFICATION_REPORT_V1_2"
VERIFICATION_SCHEMA_V13 = "IG_DECODER_V05_VERIFICATION_REPORT_V1_3"
VERIFICATION_SCHEMA_V14 = "IG_DECODER_V05_VERIFICATION_REPORT_V1_4"
VERIFICATION_SCHEMA = "IG_DECODER_V05_VERIFICATION_REPORT_V1_5"
VERIFIER_ID = "v05.base-l2.independent"


def _check(checks: list[dict], check_id: str, ok: bool, detail: Any = None) -> None:
    rec = {"check_id": check_id, "status": "PASS" if ok else "FAIL"}
    if detail is not None:
        rec["detail"] = detail
    checks.append(rec)


def _load_npy(path: Path):
    b = path.read_bytes()
    if b[:6] != b"\x93NUMPY":
        raise ValueError(f"not NPY: {path}")
    major = b[6]
    if major == 1:
        hlen = struct.unpack("<H", b[8:10])[0]
        off = 10
    elif major in (2, 3):
        hlen = struct.unpack("<I", b[8:12])[0]
        off = 12
    else:
        raise ValueError(f"unsupported NPY version {major}")
    enc = "utf-8" if major == 3 else "latin1"
    hdr = ast.literal_eval(b[off:off + hlen].decode(enc).strip())
    if hdr.get("fortran_order"):
        raise ValueError("Fortran-order NPY unsupported")
    descr = hdr["descr"]
    code = {"<i2": "h", "<i4": "i", "|i1": "b", "<i1": "b"}.get(descr)
    if code is None:
        raise ValueError(f"unsupported dtype {descr}")
    a = array.array(code)
    a.frombytes(b[off + hlen:])
    if sys.byteorder != "little" and descr.startswith("<"):
        a.byteswap()
    return list(a), tuple(hdr["shape"])


class _VerifierTables:
    def __init__(self, root: Path):
        d = root / "data"
        self.tri, self.trish = _load_npy(d / "ig_rm1_L1L1_triples.npy")
        self.cid, _ = _load_npy(d / "ig_rm1_L1L1_candidate_id.npy")
        self.oc, _ = _load_npy(d / "ig_rm1_opt_count.npy")
        self.ot, _ = _load_npy(d / "ig_rm1_opt_target.npy")
        self.os, _ = _load_npy(d / "ig_rm1_opt_supply.npy")
        self.om, _ = _load_npy(d / "ig_rm1_opt_missing.npy")
        self.on, _ = _load_npy(d / "ig_rm1_opt_need.npy")
        self.rk, _ = _load_npy(d / "ig_rm1_opt_rank.npy")
        self.lookup = {self.triple(i): i for i in range(self.trish[0])}
        self.rep = {}
        for i, c in enumerate(self.cid):
            self.rep.setdefault(c, i)

    def triple(self, tid):
        return tuple(self.tri[tid * 3:tid * 3 + 3])

    @staticmethod
    def at(a, s, i):
        return a[s * 10 + i]


def _cold_recompute(fixture: Path) -> dict:
    T = _VerifierTables(fixture)
    golden = json.loads((fixture / "metadata/golden_cases.json").read_text(encoding="utf-8"))["step10_small"]
    classes = tuple(golden["classes"])
    tids = tuple(T.rep[c] for c in classes)
    rel, diag = composite_abc(T, *tids)
    max_score = max(r[2] for r in rel)
    selected = [r for r in rel if r[2] == max_score]
    sr, ln = relation_digests(rel)
    return {
        "classes": list(classes),
        "representative_tids": list(tids),
        "boundary_records": len(rel),
        "compatible_paths": diag["internally_compatible_three_event_paths"],
        "selected_max_score": max_score,
        "selected_tied_max_records": len(selected),
        "sha256_sorted_repr": sr,
        "sha256_line_serialization": ln,
        "diagnostics": diag,
    }


def _verifier_impl_identity(reg: dict | None = None) -> dict:
    here = Path(__file__).resolve()
    files = {
        "v05_verifier.py": {"sha256": sha256_file(here), "size_bytes": here.stat().st_size},
    }
    runner = (reg or {}).get("runner")
    if runner == "adapter.scout_l15":
        scout = Path(__file__).resolve().parent / "v05_scout_l15_verifier.py"
        spec = Path(__file__).resolve().parent / "spectroscope.py"
        files["v05_scout_l15_verifier.py"] = {"sha256": sha256_file(scout), "size_bytes": scout.stat().st_size}
        files["spectroscope.py"] = {"sha256": sha256_file(spec), "size_bytes": spec.stat().st_size}
    elif runner == "adapter.oscout_phase0":
        osc = Path(__file__).resolve().parent / "v05_oscout_phase0_verifier.py"
        files["v05_oscout_phase0_verifier.py"] = {"sha256": sha256_file(osc), "size_bytes": osc.stat().st_size}
    else:
        core = Path(__file__).resolve().parent / "core" / "l2.py"
        files["core/l2.py"] = {"sha256": sha256_file(core), "size_bytes": core.stat().st_size}
    return {"files": files, "sha256": canonical_sha256(files)}


def verify_run(paths, run_id: str) -> dict:
    checks: list[dict] = []
    run_dir = Path(paths.runs) / run_id
    run_path = run_dir / "run.json"
    plan_path = run_dir / "plan.json"
    core_path = run_dir / "run_core.json"
    env_path = run_dir / "v05_envelope.json"

    required_paths = [run_path, plan_path, core_path, env_path]
    _check(checks, "records_present", all(p.is_file() for p in required_paths), [str(p) for p in required_paths if not p.is_file()])
    if not all(p.is_file() for p in required_paths):
        return _finish_report(run_id, None, checks, None)

    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        core = json.loads(core_path.read_text(encoding="utf-8"))
        envelope = json.loads(env_path.read_text(encoding="utf-8"))
        _check(checks, "records_json", True)
    except Exception as exc:
        _check(checks, "records_json", False, f"{type(exc).__name__}: {exc}")
        return _finish_report(run_id, None, checks, None)

    envelope_sha = envelope.get("envelope_sha256")
    envelope_base = {k: v for k, v in envelope.items() if k != "envelope_sha256"}
    _check(checks, "envelope_hash", envelope.get("schema_id") in {ENVELOPE_SCHEMA_V12, ENVELOPE_SCHEMA_V13, ENVELOPE_SCHEMA_V14, ENVELOPE_SCHEMA} and envelope_sha == canonical_sha256(envelope_base))
    _check(checks, "execution_terminal", run.get("lifecycle") == "COMPLETE_VALID" and envelope.get("execution_lifecycle") == "COMPLETE_VALID")

    binding = plan.get("v05", {})
    reg_sha = binding.get("registration_sha256")
    try:
        reg = V05RegistrationStore(paths.store).get(reg_sha)
        _check(checks, "registration_binding", reg.get("schema_id") in {REGISTRATION_SCHEMA_V12, REGISTRATION_SCHEMA_V13, REGISTRATION_SCHEMA_V14, REGISTRATION_SCHEMA} and envelope.get("registration_sha256") == reg_sha)
    except Exception as exc:
        reg = None
        _check(checks, "registration_binding", False, f"{type(exc).__name__}: {exc}")

    if reg is not None:
        ver = reg.get("contract_version")
        expected_binding = {
            "schema_id": PLAN_BINDING_SCHEMA if ver == "1.5.0" else PLAN_BINDING_SCHEMA_V14 if ver == "1.4.0" else PLAN_BINDING_SCHEMA_V13 if ver == "1.3.0" else PLAN_BINDING_SCHEMA_V12,
            "contract_version": ver,
            "registration_sha256": reg_sha,
        }
        _check(checks, "plan_binding", binding == expected_binding)
        _check(checks, "source_identity", run.get("code_identity", {}).get("source_sha256") == reg["source_sha256"] == envelope.get("source_sha256") == live_source_sha256())
        _check(checks, "environment_identity", run.get("environment_identity", {}).get("runtime_sha256") == reg["environment_sha256"] == envelope.get("environment_sha256") == runtime_sha256())
        _check(checks, "protocol_subject_binding", reg["protocol"] == run.get("protocol") and reg["subject"] == run.get("subject") == plan.get("subject"))
        _check(checks, "input_binding", reg["input_datasets"] == plan.get("input_datasets") == run.get("input_artifacts") == envelope.get("input_artifacts"))
    else:
        for cid in ("plan_binding", "source_identity", "environment_identity", "protocol_subject_binding", "input_binding"):
            _check(checks, cid, False, "registration unavailable")

    observed_plan = canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    observed_core = canonical_sha256({k: v for k, v in core.items() if k != "run_core_sha256"})
    _check(checks, "plan_hash", plan.get("plan_sha256") == observed_plan == envelope.get("plan_sha256"))
    _check(checks, "run_core_hash", core.get("run_core_sha256") == observed_core == run.get("run_core_sha256") == envelope.get("run_core_sha256"))

    store = ArtifactStore(paths.store)
    ds_store = DatasetStore(store)
    input_ok = True
    input_details = []
    for d in run.get("input_artifacts", []):
        v = ds_store.verify(d["dataset_sha256"])
        input_details.append(v)
        input_ok = input_ok and v.get("status") == "PASS"
    _check(checks, "input_dataset_bytes", input_ok, input_details)

    result_refs = run.get("result_artifacts", [])
    expected_outputs = set(reg["output_contract"]["logical_outputs"]) if reg else set()
    observed_outputs = {a.get("logical_name") for a in result_refs}
    _check(checks, "logical_output_set", bool(reg) and observed_outputs == expected_outputs and len(result_refs) == len(expected_outputs))
    result_ok = True
    for a in result_refs:
        result_ok = result_ok and store.verify(a["sha256"], a.get("size_bytes")).get("status") == "PASS"
    _check(checks, "result_artifact_bytes", result_ok)

    cm = CheckpointManager(run_dir, store)
    checkpoint_ok = True
    checkpoint_details = []
    for st in plan.get("stages", []):
        try:
            ptr = cm.current_pointer(st["stage_id"])
            cp = cm.current(st["stage_id"])
            ok = bool(ptr and cp and cp.get("status") == "COMPLETE_VALID")
            if ok:
                env_hits = [x for x in envelope.get("checkpoint_bindings", []) if x.get("stage_id") == st["stage_id"]]
                ok = len(env_hits) == 1 and env_hits[0]["checkpoint_sha256"] == ptr["checkpoint_sha256"] and env_hits[0]["checkpoint_content_sha256"] == cp["checkpoint_content_sha256"]
            checkpoint_ok = checkpoint_ok and ok
            checkpoint_details.append({"stage_id": st["stage_id"], "ok": ok, "stage_result": cp.get("stage_result") if cp else None})
        except Exception as exc:
            checkpoint_ok = False
            checkpoint_details.append({"stage_id": st["stage_id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    _check(checks, "checkpoint_binding", checkpoint_ok, checkpoint_details)

    worker_ok = False
    try:
        execute_cp = cm.current(reg["stage_id"]) if reg else None
        wb = execute_cp.get("stage_result", {}).get("worker_boundary", {}) if execute_cp else {}
        worker_ok = (
            reg is not None
            and wb.get("kind") == "POSIX_DROP_PRIVILEGE"
            and wb.get("uid") == reg["worker_policy"]["uid"]
            and wb.get("gid") == reg["worker_policy"]["gid"]
            and wb.get("isolated_process") is True
            and wb.get("store_write_access") == "DENIED_BY_POSIX"
            and wb.get("publication_write_access") == "DENIED_BY_POSIX"
        )
        _check(checks, "worker_boundary", worker_ok, wb)
    except Exception as exc:
        _check(checks, "worker_boundary", False, f"{type(exc).__name__}: {exc}")

    result_obj = None
    principal_name = None
    if reg is not None and len(reg.get("output_contract", {}).get("logical_outputs", [])) == 1:
        principal_name = reg["output_contract"]["logical_outputs"][0]
    principal_ref = next((a for a in result_refs if a.get("logical_name") == principal_name), None) if principal_name else None
    if principal_ref:
        try:
            result_obj = json.loads(store.blob_path(principal_ref["sha256"]).read_text(encoding="utf-8"))
            _check(checks, "result_json", isinstance(result_obj, dict))
        except Exception as exc:
            _check(checks, "result_json", False, f"{type(exc).__name__}: {exc}")
    else:
        _check(checks, "result_json", False, "principal output missing")

    recomputed = None
    cold_ok = False
    science_ok = False
    recomputed_projection = None
    if reg is not None and len(reg["input_datasets"]) == 1:
        try:
            with tempfile.TemporaryDirectory(prefix="ig-v05-verify-") as td:
                fixture = ds_store.materialize(reg["input_datasets"][0]["dataset_sha256"], Path(td) / "fixture")
                if reg.get("runner") == "adapter.base_l2_step10":
                    recomputed = _cold_recompute(fixture)
                    oracle = reg["output_contract"]["science_oracle"]
                    recomputed_projection = {k: recomputed[k] for k in oracle}
                    cold_ok = recomputed_projection == oracle
                    if result_obj is not None:
                        result_projection = {
                            "boundary_records": result_obj.get("boundary_records"),
                            "compatible_paths": result_obj.get("diagnostics", {}).get("internally_compatible_three_event_paths"),
                            "selected_max_score": result_obj.get("selected_max_score"),
                            "selected_tied_max_records": result_obj.get("selected_tied_max_records"),
                            "sha256_sorted_repr": result_obj.get("sha256_sorted_repr"),
                            "sha256_line_serialization": result_obj.get("sha256_line_serialization"),
                        }
                        science_ok = result_obj.get("status") == "PASS" and result_projection == oracle and result_obj.get("classes") == recomputed.get("classes") and result_obj.get("representative_tids") == recomputed.get("representative_tids")
                    else:
                        result_projection = None
                elif reg.get("runner") == "adapter.scout_l15":
                    recomputed = replay_scout_l15(fixture, Path(td) / "scout-replay")
                    oracle = reg["output_contract"]["science_oracle"]
                    recomputed_projection = scout_science_projection(recomputed)
                    cold_ok = recomputed.get("status") == "PASS" and recomputed_projection == oracle
                    result_projection = scout_science_projection(result_obj) if result_obj is not None else None
                    science_ok = bool(result_obj and result_obj.get("status") == "PASS" and result_projection == oracle and all(result_obj.get("comparisons", {}).values()))
                elif reg.get("runner") == "adapter.oscout_phase0":
                    recomputed = replay_oscout_phase0(fixture, Path(td) / "oscout-replay")
                    oracle = reg["output_contract"]["science_oracle"]
                    recomputed_projection = oscout_phase0_science_projection(recomputed)
                    cold_ok = recomputed.get("status") == "PASS" and recomputed_projection == oracle
                    result_projection = oscout_phase0_science_projection(result_obj) if result_obj is not None else None
                    science_ok = bool(result_obj and result_obj.get("status") == "PASS" and result_projection == oracle)
                else:
                    raise RuntimeError(f"unsupported v0.5 cold-recompute runner: {reg.get('runner')}")
            _check(checks, "cold_recompute_exact", cold_ok, {"observed": recomputed_projection, "expected": oracle})
            _check(checks, "science_projection_exact", science_ok, {"result": result_projection, "oracle": oracle})
        except Exception as exc:
            _check(checks, "cold_recompute_exact", False, f"{type(exc).__name__}: {exc}")
            _check(checks, "science_projection_exact", False, "cold recompute failed")
    else:
        _check(checks, "cold_recompute_exact", False, "registration/input scope unsupported")
        _check(checks, "science_projection_exact", False, "registration/input scope unsupported")

    # P2.4 durable logical-task journal integrity.  This verifier validates
    # the on-disk commit records independently rather than trusting the worker
    # journal reader.
    task_ok = True
    task_detail = []
    if reg is not None and reg.get("contract_version") in {"1.3.0", "1.4.0", "1.5.0"}:
        for st in plan.get("stages", []):
            task_root = Path(paths.workspace) / run_id / st["stage_id"] / "_task_state"
            try:
                scope = json.loads((task_root / "scope.json").read_text(encoding="utf-8"))
                scope_obs = canonical_sha256({k:v for k,v in scope.items() if k != "task_scope_sha256"})
                if scope.get("task_scope_sha256") != scope_obs:
                    raise RuntimeError("task scope hash mismatch")
                commits=[]
                for cp_path in sorted((task_root/"commits").glob("*.json")):
                    obj=json.loads(cp_path.read_text(encoding="utf-8"))
                    content={k:v for k,v in obj.items() if k != "commit_content_sha256"}
                    if canonical_sha256(content) != obj.get("commit_content_sha256"):
                        raise RuntimeError(f"task commit hash mismatch {cp_path.name}")
                    if obj.get("task_scope_sha256") != scope_obs or cp_path.stem != obj.get("task_id"):
                        raise RuntimeError(f"task commit binding mismatch {cp_path.name}")
                    if canonical_sha256(obj.get("payload")) != obj.get("task_payload_sha256"):
                        raise RuntimeError(f"task payload hash mismatch {cp_path.name}")
                    commits.append({"task_id":obj["task_id"],"task_payload_sha256":obj["task_payload_sha256"],"commit_content_sha256":obj["commit_content_sha256"]})
                stage_cp=cm.current(st["stage_id"]); declared=(stage_cp or {}).get("stage_result",{}).get("logical_tasks",{}).get("journal",{})
                ok = bool(declared and declared.get("task_scope_sha256") == scope_obs and declared.get("committed_tasks") == len(commits) and declared.get("task_bindings") == commits and declared.get("task_bindings_sha256") == canonical_sha256(commits))
                task_ok = task_ok and ok
                task_detail.append({"stage_id":st["stage_id"],"ok":ok,"commits":len(commits),"scope_sha256":scope_obs})
            except Exception as exc:
                task_ok=False; task_detail.append({"stage_id":st["stage_id"],"ok":False,"error":f"{type(exc).__name__}: {exc}"})
    elif reg is not None:
        task_detail="not required before v1.3"
    else:
        task_ok=False; task_detail="registration unavailable"
    _check(checks, "task_journal_integrity", task_ok, task_detail)

    telemetry_ok = True
    telemetry_detail=[]
    if reg is not None and reg.get("contract_version") in {"1.3.0", "1.4.0", "1.5.0"}:
        bindings=envelope.get("terminal_telemetry_bindings",[])
        if not bindings:
            telemetry_ok=False
        pass_seen=False
        for b in bindings:
            tp=run_dir/"v05_telemetry"/b.get("file","")
            try:
                obj=json.loads(tp.read_text(encoding="utf-8")); obs=canonical_sha256({k:v for k,v in obj.items() if k != "telemetry_sha256"})
                ok = obj.get("telemetry_sha256") == obs == b.get("telemetry_sha256") and obj.get("heartbeat_count",0) >= 1 and obj.get("terminal_status") == b.get("terminal_status")
                pass_seen = pass_seen or (ok and obj.get("terminal_status") == "PASS" and obj.get("return_code") == 0)
                telemetry_ok = telemetry_ok and ok
                telemetry_detail.append({"file":b.get("file"),"ok":ok,"terminal_status":obj.get("terminal_status"),"heartbeat_count":obj.get("heartbeat_count")})
            except Exception as exc:
                telemetry_ok=False; telemetry_detail.append({"file":b.get("file"),"ok":False,"error":f"{type(exc).__name__}: {exc}"})
        telemetry_ok = telemetry_ok and pass_seen
    elif reg is not None:
        telemetry_detail="not required before v1.3"
    else:
        telemetry_ok=False; telemetry_detail="registration unavailable"
    _check(checks, "heartbeat_terminal_telemetry", telemetry_ok, telemetry_detail)

    packaging_ok = True
    packaging_detail=None
    if reg is not None and reg.get("contract_version") in {"1.3.0", "1.4.0", "1.5.0"}:
        pp=run_dir/"v05_packaging.json"
        try:
            pobj=json.loads(pp.read_text(encoding="utf-8")); obs=canonical_sha256({k:v for k,v in pobj.items() if k != "state_sha256"})
            packaging_ok = pobj.get("schema_id") == "IG_DECODER_V05_PACKAGING_STATE_V1_3" and pobj.get("state_sha256") == obs and "packaging_state" not in run and "packaging" not in run and envelope.get("packaging_state_reference",{}).get("identity_scope") == "SEPARATE_MUTABLE_OPERATIONAL_STATE"
            packaging_detail={"state":pobj.get("state"),"attempt":pobj.get("attempt"),"run_lifecycle":run.get("lifecycle")}
        except Exception as exc:
            packaging_ok=False; packaging_detail=f"{type(exc).__name__}: {exc}"
    elif reg is not None:
        packaging_detail="not required before v1.3"
    else:
        packaging_ok=False; packaging_detail="registration unavailable"
    _check(checks, "packaging_state_separation", packaging_ok, packaging_detail)

    # P2.5 resource-budget and comparison-contract binding.  This is evaluated
    # from independently read run/checkpoint/telemetry bytes, not worker claims.
    if reg is not None and reg.get("contract_version") in {"1.4.0", "1.5.0"}:
        rb = reg.get("resource_budget", {})
        rs = run.get("resource_summary", {})
        tel_peak_rss = 0
        tel_peak_ws = 0
        tel_wall = 0.0
        tel_cpu = 0.0
        tel_budget_stops = []
        for b in envelope.get("terminal_telemetry_bindings", []):
            tp = run_dir / "v05_telemetry" / b.get("file", "")
            if tp.is_file():
                obj = json.loads(tp.read_text(encoding="utf-8"))
                tel_peak_rss = max(tel_peak_rss, int(obj.get("peak_job_rss_bytes_observed", 0) or obj.get("worker_self_maxrss_bytes", 0) or obj.get("peak_worker_rss_bytes_observed", 0) or 0))
                tel_peak_ws = max(tel_peak_ws, int(obj.get("peak_workspace_bytes_observed", 0) or 0))
                tel_wall += float(obj.get("wall_seconds", 0) or 0)
                if obj.get("worker_self_user_cpu_seconds") is not None and obj.get("worker_self_system_cpu_seconds") is not None:
                    tel_cpu += float(obj.get("worker_self_user_cpu_seconds") or 0) + float(obj.get("worker_self_system_cpu_seconds") or 0)
                else:
                    tel_cpu += float(obj.get("child_user_cpu_seconds_delta", 0) or 0) + float(obj.get("child_system_cpu_seconds_delta", 0) or 0)
                if obj.get("budget_stop_reason"):
                    tel_budget_stops.append(obj.get("budget_stop_reason"))
        execute_cp = cm.current(reg["stage_id"])
        stage_ru = (execute_cp or {}).get("resource_usage", {})
        logical = (execute_cp or {}).get("stage_result", {}).get("logical_tasks", {})
        budget_detail = {
            "wall_seconds": tel_wall,
            "cpu_seconds": tel_cpu,
            "peak_worker_rss_bytes": tel_peak_rss,
            "peak_workspace_bytes": tel_peak_ws,
            "bytes_read": int(stage_ru.get("bytes_read", 0) or 0),
            "bytes_written": int(stage_ru.get("bytes_written", 0) or 0),
            "checkpoint_bytes": int(stage_ru.get("checkpoint_bytes", 0) or 0),
            "logical_tasks": int(logical.get("tasks_total", 0) or 0),
            "budget_stop_reasons": tel_budget_stops,
            "cpu_accounting_semantics": "WORKER_SELF_IF_CLEAN_EXIT_ELSE_CONTROLLER_CHILD_DELTA",
            "rss_accounting_semantics": "WORKER_SELF_PROCESS_LIFETIME_MAXRSS_IF_AVAILABLE_ELSE_SAMPLED",
            "registered": rb,
        }
        no_deadline = (reg.get('execution_policy') == 'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'
                       or os.environ.get('IG_DECODER_EXECUTION_POLICY') == 'NO_AUTOMATIC_RUNTIME_DEADLINE_V1')
        budget_ok = (
            not tel_budget_stops
            and (no_deadline or tel_wall <= float(rb["wall_seconds_max"]))
            and (no_deadline or tel_cpu <= float(rb["cpu_seconds_max"]))
            and tel_peak_rss <= int(rb["peak_rss_bytes_max"])
            and int(stage_ru.get("bytes_read", 0) or 0) <= int(rb["bytes_read_max"])
            and int(stage_ru.get("bytes_written", 0) or 0) <= int(rb["bytes_written_max"])
            and tel_peak_ws <= int(rb["workspace_peak_bytes_max"])
            and int(stage_ru.get("checkpoint_bytes", 0) or 0) <= int(rb["checkpoint_bytes_max"])
            and int(logical.get("tasks_total", 0) or 0) <= int(rb["logical_tasks_max"])
        )
        _check(checks, "resource_budget_compliance", budget_ok, budget_detail)
        cc = reg.get("comparison_contract", {})
        cc_ok = envelope.get("resource_budget") == rb and envelope.get("stop_rules") == reg.get("stop_rules") and envelope.get("comparison_contract") == cc
        _check(checks, "comparison_contract_binding", cc_ok, cc)
    elif reg is not None:
        _check(checks, "resource_budget_compliance", True, "not required before v1.4")
        _check(checks, "comparison_contract_binding", True, "not required before v1.4")
    else:
        _check(checks, "resource_budget_compliance", False, "registration unavailable")
        _check(checks, "comparison_contract_binding", False, "registration unavailable")

    events_ok = False
    events_path = run_dir / "events.jsonl"
    if events_path.is_file():
        try:
            events = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            names = [x.get("event") for x in events]
            events_ok = "V05_STAGE_ADMITTED" in names and "V05_WORKER_COMPLETE" in names and "RUN_COMPLETE" in names and names[-1] == "V05_EXECUTION_ENVELOPE_SEALED"
            if reg is not None and reg.get("contract_version") in {"1.3.0", "1.4.0", "1.5.0"}:
                events_ok = events_ok and "V05_HEARTBEAT" in names
        except Exception:
            events_ok = False
    _check(checks, "events_terminal_consistency", events_ok)

    if reg is not None:
        declared_required = reg["verification_policy"]["required_checks"]
        observed = {c["check_id"]: c["status"] for c in checks}
        coverage = all(observed.get(cid) == "PASS" for cid in declared_required)
        _check(checks, "required_check_coverage", coverage, declared_required)
    else:
        _check(checks, "required_check_coverage", False, "registration unavailable")

    report = _finish_report(run_id, envelope_sha, checks, reg, recomputed=recomputed)
    try:
        V05TelemetryLedger(run_dir / "v05_telemetry" / "p3-controller-spans.jsonl").record(
            "VERIFICATION", component="v05_verifier.verify_run", status=report.get("status","FAIL"),
            details={"run_id":run_id,"verification_sha256":report.get("verification_sha256"),"required_checks":len(report.get("required_checks",[]))},
        )
    except Exception:
        pass
    return report


def _finish_report(run_id: str, envelope_sha: str | None, checks: list[dict], reg: dict | None, *, recomputed: dict | None = None) -> dict:
    required = reg.get("verification_policy", {}).get("required_checks", []) if reg else []
    check_map = {c["check_id"]: c["status"] for c in checks}
    status = "PASS" if required and all(c.get("status") == "PASS" for c in checks) and all(check_map.get(x) == "PASS" for x in required) and check_map.get("required_check_coverage") == "PASS" else "FAIL"
    impl = _verifier_impl_identity(reg)
    base = {
        "schema_id": VERIFICATION_SCHEMA if reg and reg.get("contract_version") == "1.5.0" else VERIFICATION_SCHEMA_V14 if reg and reg.get("contract_version") == "1.4.0" else VERIFICATION_SCHEMA_V13 if reg and reg.get("contract_version") == "1.3.0" else VERIFICATION_SCHEMA_V12,
        "contract_version": reg.get("contract_version") if reg else None,
        "verifier_id": reg.get("verification_policy", {}).get("verifier_id", VERIFIER_ID) if reg else VERIFIER_ID,
        "run_id": run_id,
        "registration_sha256": reg.get("registration_sha256") if reg else None,
        "envelope_sha256": envelope_sha,
        "status": status,
        "comparison_mode": "COLD_RECOMPUTE_EXACT",
        "checks": checks,
        "required_checks": required,
        "verifier_implementation": impl,
        "invocation": {
            "pid": os.getpid(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "gid": os.getgid() if hasattr(os, "getgid") else None,
            "live_source_sha256": live_source_sha256(),
            "runtime_sha256": runtime_sha256(),
        },
        "cold_recompute_projection": None if recomputed is None else (
            scout_science_projection(recomputed) if reg and reg.get("runner") == "adapter.scout_l15" else
            oscout_phase0_science_projection(recomputed) if reg and reg.get("runner") == "adapter.oscout_phase0" else {
                k: recomputed[k]
                for k in ("boundary_records", "compatible_paths", "selected_max_score", "selected_tied_max_records", "sha256_sorted_repr", "sha256_line_serialization")
            }
        ),
        "authority_effect": "NONE",
        "limitations": ([
            "BOUNDED_SCOUT_L15_COMPATIBILITY_SLICE_ONLY",
            "INDEPENDENT_PROCESS_AND_FROZEN_ORACLE_REPLAY_NOT_INDEPENDENT_SECOND_CODE_PROOF",
            "RECONNAISSANCE_AUTHORITY_ONLY",
            "NO_AUTHORITY_OR_GRADUATION_EFFECT",
        ] if reg and reg.get("runner") == "adapter.scout_l15" else [
            "BOUNDED_OSCOUT_PHASE0_COMPATIBILITY_SLICE_ONLY",
            "FROZEN_PHASE0_VERIFIER_REPLAY_NOT_INDEPENDENT_SECOND_CODE_PROOF",
            "NO_LIVE_O7_GENERATION",
            "NO_AUTHORITY_OR_GRADUATION_EFFECT",
        ] if reg and reg.get("runner") == "adapter.oscout_phase0" else [
            "BOUNDED_BASE_L2_STEP10_ONLY",
            "NO_MATHEMATICAL_GENERALITY_CLAIM",
            "NO_AUTHORITY_OR_GRADUATION_EFFECT",
        ]),
    }
    return dict(base, verification_sha256=canonical_sha256(base))


# Backwards-compatible name retained for existing P2.3-P2.6 callers.
verify_base_l2_run = verify_run

def _cli() -> int:
    from .paths import resolve_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--report", required=True)
    ns = ap.parse_args()
    report = verify_run(resolve_root(ns.root), ns.run_id)
    write_json_atomic(Path(ns.report), report)
    print(json.dumps({"status": report["status"], "verification_sha256": report["verification_sha256"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(_cli())
