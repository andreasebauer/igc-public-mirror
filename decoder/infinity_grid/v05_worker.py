from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
import traceback
from pathlib import Path

from .canon import canonical_sha256, write_json_atomic
from .records import utc_now
from .safety import validate_relative_path
from .v05 import V05Admission, V05AdmissionError
from .v05_tasks import V05TaskJournal, prepare_task_state


WORKER_JOB_SCHEMA_V12 = "IG_DECODER_V05_WORKER_JOB_V1_2"
WORKER_RESULT_SCHEMA_V12 = "IG_DECODER_V05_WORKER_RESULT_V1_2"
WORKER_JOB_SCHEMA_V13 = "IG_DECODER_V05_WORKER_JOB_V1_3"
WORKER_RESULT_SCHEMA_V13 = "IG_DECODER_V05_WORKER_RESULT_V1_3"
WORKER_TELEMETRY_SCHEMA_V13 = "IG_DECODER_V05_WORKER_TELEMETRY_V1_3"
WORKER_HEARTBEAT_SCHEMA_V13 = "IG_DECODER_V05_WORKER_HEARTBEAT_V1_3"
WORKER_JOB_SCHEMA_V14 = "IG_DECODER_V05_WORKER_JOB_V1_4"
WORKER_RESULT_SCHEMA_V14 = "IG_DECODER_V05_WORKER_RESULT_V1_4"
WORKER_TELEMETRY_SCHEMA_V14 = "IG_DECODER_V05_WORKER_TELEMETRY_V1_4"
WORKER_HEARTBEAT_SCHEMA_V14 = "IG_DECODER_V05_WORKER_HEARTBEAT_V1_4"
WORKER_JOB_SCHEMA = "IG_DECODER_V05_WORKER_JOB_V1_5"
WORKER_RESULT_SCHEMA = "IG_DECODER_V05_WORKER_RESULT_V1_5"
WORKER_TELEMETRY_SCHEMA = "IG_DECODER_V05_WORKER_TELEMETRY_V1_5"
WORKER_HEARTBEAT_SCHEMA = "IG_DECODER_V05_WORKER_HEARTBEAT_V1_5"


def _inside(root: Path, path: Path, *, label: str) -> Path:
    root = Path(root).resolve(strict=False)
    path = Path(path).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise V05AdmissionError(f"{label} escapes worker root") from exc
    return path


def _chmod_readonly_tree(root: Path) -> None:
    root = Path(root)
    for p in sorted(root.rglob("*"), reverse=True):
        try:
            os.chmod(p, 0o555 if p.is_dir() else 0o444)
        except OSError:
            pass
    try:
        os.chmod(root, 0o555)
    except OSError:
        pass


def _drop_privileges(uid: int, gid: int) -> None:
    os.setgroups([])
    os.setgid(int(gid))
    os.setuid(int(uid))
    os.umask(0o077)


class V05IsolatedWorkerContext:
    __slots__ = (
        "_run_id", "_stage_id", "_operation", "_registration_sha256",
        "_dataset_paths", "_output_root", "_task_state_root", "_task_scope_sha256", "_resource_budget", "_phase_ledger",
    )

    def __init__(self, job: dict):
        self._run_id = job["run_id"]
        self._stage_id = job["stage_id"]
        self._operation = job["operation"]
        self._registration_sha256 = job["registration_sha256"]
        self._dataset_paths = {k: Path(v).resolve(strict=True) for k, v in job["dataset_paths"].items()}
        self._output_root = Path(job["output_root"]).resolve(strict=True)
        self._task_state_root = Path(job["task_state_root"]).resolve(strict=True) if job.get("task_state_root") else None
        self._task_scope_sha256 = job.get("task_scope_sha256")
        self._resource_budget = dict(job.get("resource_budget") or {})
        from .v05_runtime import V05TelemetryLedger
        self._phase_ledger = V05TelemetryLedger(self._output_root / "v05_worker_spans.jsonl")

    @property
    def run_id(self) -> str: return self._run_id
    @property
    def stage_id(self) -> str: return self._stage_id
    @property
    def operation(self) -> str: return self._operation
    @property
    def registration_sha256(self) -> str: return self._registration_sha256

    def materialize_dataset(self, dataset_sha256: str, relative_destination: str = "fixture") -> Path:
        src = self._dataset_paths.get(dataset_sha256)
        if src is None:
            raise V05AdmissionError("dataset capability denied")
        rel = validate_relative_path(relative_destination)
        dst = _inside(self._output_root, self._output_root / rel, label="dataset destination")
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst, copy_function=shutil.copyfile)
        return dst

    def staging_path(self, relative_path: str) -> Path:
        rel = validate_relative_path(relative_path)
        p = _inside(self._output_root, self._output_root / rel, label="staging path")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def task_journal(self) -> V05TaskJournal:
        if self._task_state_root is None or not self._task_scope_sha256:
            raise V05AdmissionError("durable logical-task journal is not bound for this worker")
        return V05TaskJournal(self._task_state_root, expected_scope_sha256=self._task_scope_sha256, max_tasks=self._resource_budget.get("logical_tasks_max"))

    def phase(self, category: str, *, component: str, details: dict | None = None):
        return self._phase_ledger.span(category, component=component, details=details)

    def telemetry_coverage(self) -> dict:
        return self._phase_ledger.coverage()


def _load_job(path: Path) -> dict:
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(job, dict) or job.get("schema_id") not in {WORKER_JOB_SCHEMA_V12, WORKER_JOB_SCHEMA_V13, WORKER_JOB_SCHEMA_V14, WORKER_JOB_SCHEMA}:
        raise V05AdmissionError("invalid worker job schema")
    declared = job.get("job_sha256")
    observed = canonical_sha256({k: v for k, v in job.items() if k != "job_sha256"})
    if declared != observed:
        raise V05AdmissionError("worker job hash mismatch")
    common = {
        "schema_id", "contract_version", "job_sha256", "run_id", "stage_id",
        "runner", "operation", "registration_sha256", "dataset_paths",
        "output_root", "plan", "stage", "logical_outputs",
    }
    if job["schema_id"] == WORKER_JOB_SCHEMA:
        expected = common | {"task_state_root", "task_scope_sha256", "resource_budget", "stop_rules"}
        if job["contract_version"] != "1.5.0":
            raise V05AdmissionError("worker job contract version mismatch")
    elif job["schema_id"] == WORKER_JOB_SCHEMA_V14:
        expected = common | {"task_state_root", "task_scope_sha256", "resource_budget", "stop_rules"}
        if job["contract_version"] != "1.4.0":
            raise V05AdmissionError("worker job contract version mismatch")
    elif job["schema_id"] == WORKER_JOB_SCHEMA_V13:
        expected = common | {"task_state_root", "task_scope_sha256"}
        if job["contract_version"] != "1.3.0":
            raise V05AdmissionError("worker job contract version mismatch")
    else:
        expected = common
        if job["contract_version"] != "1.2.0":
            raise V05AdmissionError("worker job contract version mismatch")
    if set(job) != expected:
        raise V05AdmissionError("worker job fields mismatch")
    if not isinstance(job["dataset_paths"], dict) or not job["dataset_paths"]:
        raise V05AdmissionError("worker dataset_paths missing")
    if not isinstance(job["logical_outputs"], list) or not job["logical_outputs"]:
        raise V05AdmissionError("worker logical_outputs missing")
    return job


def worker_main(job_path: Path) -> dict:
    from .v05_origin_guard import require_worker_execution_origin
    require_worker_execution_origin("v05_worker.worker_main")
    job = _load_job(job_path)
    from . import adapters  # noqa: F401
    from .controller import RUNNERS

    runner = RUNNERS.get(job["runner"])
    if runner is None:
        raise V05AdmissionError("worker runner not registered")
    delay = os.environ.get("IG_V05_WORKER_TEST_INITIAL_DELAY_SECONDS")
    if delay:
        if os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") != "P3_LONG_SHARD_TEST":
            raise V05AdmissionError("P3 initial-delay fault injection requires explicit acknowledgement")
        time.sleep(max(0.0, float(delay)))
    ctx = V05IsolatedWorkerContext(job)
    res = runner(context=ctx, plan=job["plan"], stage=job["stage"])
    if not isinstance(res, dict): raise RuntimeError("worker result must be an object")
    outputs = res.get("outputs", {})
    if set(outputs) != set(job["logical_outputs"]): raise RuntimeError("worker logical output set differs from registration")
    out_root = Path(job["output_root"]).resolve(strict=True)
    encoded_outputs: dict[str, str] = {}
    for logical, raw in sorted(outputs.items()):
        p = _inside(out_root, Path(raw), label="worker output")
        if not p.is_file(): raise RuntimeError(f"worker output missing: {logical}")
        encoded_outputs[logical] = p.relative_to(out_root).as_posix()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema_id": WORKER_RESULT_SCHEMA if job["schema_id"] == WORKER_JOB_SCHEMA else WORKER_RESULT_SCHEMA_V14 if job["schema_id"] == WORKER_JOB_SCHEMA_V14 else WORKER_RESULT_SCHEMA_V13 if job["schema_id"] == WORKER_JOB_SCHEMA_V13 else WORKER_RESULT_SCHEMA_V12,
        "contract_version": job["contract_version"],
        "job_sha256": job["job_sha256"],
        "run_id": job["run_id"], "stage_id": job["stage_id"], "runner": job["runner"], "operation": job["operation"],
        "outputs": encoded_outputs,
        "stage_result": res.get("stage_result", {}),
        "worker_identity": {"uid": os.getuid() if hasattr(os, "getuid") else None, "gid": os.getgid() if hasattr(os, "getgid") else None, "pid": os.getpid()},
        "worker_resource_self": {
            "user_cpu_seconds": float(ru.ru_utime),
            "system_cpu_seconds": float(ru.ru_stime),
            "maxrss_bytes": int(ru.ru_maxrss) * 1024,
            "maxrss_semantics": "PROCESS_LIFETIME_HIGH_WATER_RSS",
        },
    }
    write_json_atomic(out_root / "worker_result.json", result)
    return result


def _write_telemetry(telemetry_dir: Path | None, base: dict) -> tuple[dict, Path | None]:
    rec = dict(base, telemetry_sha256=canonical_sha256(base))
    if telemetry_dir is None:
        return rec, None
    td = Path(telemetry_dir); td.mkdir(parents=True, exist_ok=True)
    p = td / f"{base['stage_id']}-{base['invocation_id']}.json"
    write_json_atomic(p, rec)
    return rec, p


def _tree_bytes(root: Path) -> int:
    """Best-effort workspace size under concurrent atomic writes.

    Files may legitimately disappear between directory enumeration and stat;
    telemetry must never turn that ordinary race into a worker failure.
    """
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    try:
        it = root.rglob("*")
        for p in it:
            try:
                if p.is_file():
                    total += p.stat().st_size
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                continue
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return total
    return total

def _proc_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0

def _proc_cpu_seconds(pid: int) -> float | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw.rsplit(")", 1)[1].strip().split()
        # fields 14/15 are indexes 11/12 after removing pid/comm.
        ticks = int(tail[11]) + int(tail[12])
        hz = int(os.sysconf("SC_CLK_TCK"))
        return ticks / hz
    except Exception:
        return None



def _proc_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None

def _proc_tree_pids(root_pid: int) -> tuple[int, ...]:
    """Snapshot Linux descendant tree, including root, race-tolerantly."""
    root_pid = int(root_pid)
    parent_to_children: dict[int, list[int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return (root_pid,)
    for ent in entries:
        if not ent.name.isdigit():
            continue
        pid = int(ent.name)
        ppid = _proc_ppid(pid)
        if ppid is not None:
            parent_to_children.setdefault(ppid, []).append(pid)
    seen = {root_pid}
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for child in parent_to_children.get(cur, []):
            if child not in seen:
                seen.add(child); stack.append(child)
    return tuple(sorted(seen))

def _proc_tree_rss_bytes(root_pid: int) -> int:
    return sum(_proc_rss_bytes(pid) for pid in _proc_tree_pids(root_pid))

def _proc_tree_cpu_seconds(root_pid: int) -> float | None:
    vals = [_proc_cpu_seconds(pid) for pid in _proc_tree_pids(root_pid)]
    known = [x for x in vals if x is not None]
    return sum(known) if known else None

def _terminate_process_tree(proc: subprocess.Popen, *, grace_seconds: float = 0.75) -> None:
    """Terminate one controller-owned worker session and all descendants."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try: proc.terminate()
            except Exception: pass
    else:
        try: proc.terminate()
        except Exception: pass
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if proc.poll() is None:
        if os.name == "posix":
            try: os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try: proc.kill()
                except Exception: pass
        else:
            try: proc.kill()
            except Exception: pass
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass

def _task_progress_snapshot(task_state_root: Path | None, *, last_count: int, last_change_monotonic: float, now: float) -> tuple[dict, int, float]:
    if task_state_root is None:
        return ({"queued": None, "running": None, "completed": None, "retried": None, "science_progress_kind": "UNKNOWN"}, last_count, last_change_monotonic)
    commits = len(list((task_state_root / "commits").glob("*.json"))) if (task_state_root / "commits").exists() else 0
    claims = len(list((task_state_root / "claims").glob("*.json"))) if (task_state_root / "claims").exists() else 0
    if commits != last_count:
        last_count = commits; last_change_monotonic = now
    return ({
        "queued": None, "running": claims, "completed": commits, "retried": None,
        "science_progress_kind": "DURABLE_LOGICAL_COMMITS",
        "seconds_since_science_progress": max(0.0, now - last_change_monotonic),
    }, last_count, last_change_monotonic)

def _drop_with_limits(uid: int, gid: int, budget: dict | None) -> None:
    budget = budget or {}
    no_deadline = os.environ.get("IG_DECODER_EXECUTION_POLICY") == "NO_AUTOMATIC_RUNTIME_DEADLINE_V1"
    cpu = None if no_deadline else budget.get("cpu_seconds_max")
    if cpu:
        import math
        lim = max(1, int(math.ceil(float(cpu))))
        try: resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
        except (ValueError, OSError): pass
    _drop_privileges(uid, gid)

def run_isolated_stage(*, admission: V05Admission, datasets, plan: dict, stage: dict, work_dir: Path,
                       dependency_bindings: list[dict] | None = None,
                       heartbeat_callback=None, heartbeat_path: Path | None = None, telemetry_dir: Path | None = None) -> tuple[dict, dict]:
    wp = admission.worker_policy
    if not wp or wp.get("kind") != "POSIX_DROP_PRIVILEGE": raise V05AdmissionError("isolated worker requires POSIX_DROP_PRIVILEGE policy")
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0: raise V05AdmissionError("isolated worker boundary unavailable")

    work = Path(work_dir).resolve(strict=False)
    try: os.chmod(Path(datasets.store.root), 0o750)
    except OSError: pass
    control, inputs, output = work / "_control", work / "_inputs", work / "_worker"
    for p in (control, inputs, output):
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    dataset_paths: dict[str, str] = {}
    for ds in admission.allowed_dataset_sha256:
        snap = inputs / ds
        datasets.materialize(ds, snap)
        _chmod_readonly_tree(snap)
        dataset_paths[ds] = str(snap.resolve(strict=True))
    os.chmod(inputs, 0o555)

    uid, gid = int(wp["uid"]), int(wp["gid"])
    protected = Path(datasets.store.root) / "v05" / "protected_publications"
    protected.mkdir(parents=True, exist_ok=True)
    try: os.chmod(protected, 0o700)
    except OSError: pass

    def denied_write_probe(target: Path) -> bool:
        from .v05_origin_guard import require_controller_execution_origin, _forked_worker_scope
        require_controller_execution_origin("v05_worker.denied_write_probe")
        probe = target / f".worker-write-probe-{os.getpid()}"
        pid = os.fork()
        if pid == 0:
            rc = 9
            try:
                with _forked_worker_scope(f"probe-{admission.stage_id}"):
                    _drop_privileges(uid, gid)
                    try:
                        probe.write_text("forbidden", encoding="utf-8")
                    except (PermissionError, OSError):
                        rc = 0
                    else:
                        probe.unlink(missing_ok=True); rc = 9
            except BaseException:
                rc = 8
            os._exit(rc)
        _, st = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(st) == 0

    store_denied, publication_denied = denied_write_probe(Path(datasets.store.root)), denied_write_probe(protected)
    if not store_denied or not publication_denied: raise V05AdmissionError("worker POSIX write isolation probe failed")
    os.chown(output, uid, gid); os.chmod(output, 0o700); os.chmod(control, 0o755)

    task_state_root = None; task_scope_sha = None
    if admission.execution_policy is not None:
        task_state_root = work / "_task_state"
        reg_obj = json.loads((Path(datasets.store.root)/"v05"/"registrations"/f"{admission.registration_sha256}.json").read_text())
        scope = {
            "contract_version": admission.contract_version,
            "run_id": admission.run_id,
            "stage_id": admission.stage_id,
            "registration_sha256": admission.registration_sha256,
            "plan_sha256": plan["plan_sha256"],
            "stage_spec_sha256": canonical_sha256(stage),
            "source_sha256": reg_obj["source_sha256"],
            "environment_sha256": reg_obj["environment_sha256"],
            "dataset_sha256": list(admission.allowed_dataset_sha256),
            "generator_identity_sha256": canonical_sha256({"runner": admission.runner, "operation": admission.operation}),
            "dependency_bindings_sha256": canonical_sha256(dependency_bindings or []),
            "dependency_bindings": dependency_bindings or [],
            "resource_budget": admission.resource_budget,
            "stop_rules": admission.stop_rules,
        }
        prepared = prepare_task_state(task_state_root, scope=scope, uid=uid, gid=gid)
        task_scope_sha = prepared["task_scope_sha256"]

    base_job = {
        "schema_id": WORKER_JOB_SCHEMA if admission.contract_version == "1.5.0" else WORKER_JOB_SCHEMA_V14 if admission.contract_version == "1.4.0" else WORKER_JOB_SCHEMA_V13 if admission.contract_version == "1.3.0" else WORKER_JOB_SCHEMA_V12,
        "contract_version": admission.contract_version,
        "run_id": admission.run_id, "stage_id": admission.stage_id, "runner": admission.runner, "operation": admission.operation,
        "registration_sha256": admission.registration_sha256, "dataset_paths": dataset_paths,
        "output_root": str(output.resolve(strict=True)), "plan": plan, "stage": stage,
        "logical_outputs": list(admission.logical_outputs),
    }
    if admission.contract_version in {"1.3.0", "1.4.0", "1.5.0"}:
        base_job.update(task_state_root=str(task_state_root.resolve(strict=True)), task_scope_sha256=task_scope_sha)
    if admission.contract_version in {"1.4.0", "1.5.0"}:
        base_job.update(resource_budget=dict(admission.resource_budget or {}), stop_rules=dict(admission.stop_rules or {}))
    job = dict(base_job, job_sha256=canonical_sha256(base_job))
    job_path = control / "job.json"; write_json_atomic(job_path, job); os.chmod(job_path, 0o444); os.chmod(control, 0o555)

    src_parent = str(Path(__file__).resolve().parents[1])
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": src_parent, "PYTHONHASHSEED": "0", "LANG": os.environ.get("LANG", "C.UTF-8"), "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8")}
    if os.environ.get("IG_DECODER_EXECUTION_POLICY"):
        env["IG_DECODER_EXECUTION_POLICY"] = os.environ["IG_DECODER_EXECUTION_POLICY"]
    ack = os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK")
    if ack in {"P2_4_RESUME_TEST", "P2_5_RECOVERY_TEST"} and os.environ.get("IG_V05_WORKER_TEST_TASK_DELAY_SECONDS"):
        env["IG_V05_WORKER_TEST_TASK_DELAY_SECONDS"] = os.environ["IG_V05_WORKER_TEST_TASK_DELAY_SECONDS"]; env["IG_V05_ENGINEERING_FAULT_INJECTION_ACK"] = ack
    if ack == "P3_LONG_SHARD_TEST" and os.environ.get("IG_V05_WORKER_TEST_INITIAL_DELAY_SECONDS"):
        env["IG_V05_WORKER_TEST_INITIAL_DELAY_SECONDS"] = os.environ["IG_V05_WORKER_TEST_INITIAL_DELAY_SECONDS"]; env["IG_V05_ENGINEERING_FAULT_INJECTION_ACK"] = ack
    budget = dict(admission.resource_budget or {})

    if budget:
        dataset_bytes = sum(int(sh["size_bytes"]) for ds in admission.allowed_dataset_sha256 for sh in datasets.load(ds)["shards"])
        if dataset_bytes > int(budget["bytes_read_max"]):
            raise V05AdmissionError("registered bytes_read budget is below immutable input size")
    interval = float((admission.execution_policy or {}).get("heartbeat_interval_seconds", 1.0))
    invocation_id = uuid.uuid4().hex
    started_utc = utc_now(); t0 = time.monotonic(); heartbeat_count = 0
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    commits_before = len(list((task_state_root / "commits").glob("*.json"))) if task_state_root else 0
    class _ForkedProcess:
        def __init__(self, pid: int, stdout_path: Path, stderr_path: Path):
            self.pid=pid; self.returncode=None; self._stdout=stdout_path; self._stderr=stderr_path
        def poll(self):
            if self.returncode is not None: return self.returncode
            got, st=os.waitpid(self.pid, os.WNOHANG)
            if got==0: return None
            self.returncode=os.waitstatus_to_exitcode(st); return self.returncode
        def wait(self, timeout=None):
            deadline=None if timeout is None else time.monotonic()+float(timeout)
            while self.poll() is None:
                if deadline is not None and time.monotonic()>=deadline: raise TimeoutError("forked worker wait timeout")
                time.sleep(0.02)
            return self.returncode
        def terminate(self):
            try: os.killpg(self.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try: os.kill(self.pid, signal.SIGTERM)
                except Exception: pass
        def kill(self):
            try: os.killpg(self.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try: os.kill(self.pid, signal.SIGKILL)
                except Exception: pass
        def communicate(self):
            self.wait()
            out=self._stdout.read_text(encoding="utf-8",errors="replace") if self._stdout.exists() else ""
            err=self._stderr.read_text(encoding="utf-8",errors="replace") if self._stderr.exists() else ""
            return out,err

    from .v05_origin_guard import require_controller_execution_origin, _forked_worker_scope
    require_controller_execution_origin("v05_worker.launch_controller_descended_worker")
    stdout_path=control/"worker.stdout.log"; stderr_path=control/"worker.stderr.log"
    pid=os.fork()
    if pid==0:
        rc=2
        try:
            os.setsid(); os.chdir(output)
            with stdout_path.open("w",encoding="utf-8") as so, stderr_path.open("w",encoding="utf-8") as se:
                os.dup2(so.fileno(),1); os.dup2(se.fileno(),2)
                with _forked_worker_scope(f"stage-{admission.stage_id}"):
                    os.environ.clear(); os.environ.update(env)
                    _drop_with_limits(uid,gid,budget)
                    result=worker_main(job_path)
                    print(json.dumps({"status":"PASS","job_sha256":result["job_sha256"]},sort_keys=True),flush=True)
                    rc=0
        except BaseException:
            traceback.print_exc(); rc=2
        os._exit(rc)
    proc=_ForkedProcess(pid,stdout_path,stderr_path)

    test_kill_raw = os.environ.get("IG_V05_TEST_KILL_AFTER_COMMITS")
    test_kill_after = int(test_kill_raw) if test_kill_raw else None
    if test_kill_after is not None and os.environ.get("IG_V05_ENGINEERING_FAULT_INJECTION_ACK") not in {"P2_4_RESUME_TEST", "P2_5_RECOVERY_TEST"}:
        _terminate_process_tree(proc)
        raise V05AdmissionError("engineering fault injection requires explicit recovery-test acknowledgement")
    next_hb = time.monotonic()
    fault_injected = False
    budget_stop_reason = None
    peak_worker_rss = 0
    peak_job_rss = 0
    peak_workspace_bytes = _tree_bytes(work)
    last_proc_cpu = None
    last_job_cpu = None
    storage_scan_interval = max(0.5, interval)
    next_storage_scan = t0
    last_commit_count = commits_before
    last_science_change = t0
    no_progress_threshold_seconds = max(5.0, interval * 5.0)
    cancel_request_path = work / "_cancel.request"
    stdout = ""; stderr = ""
    try:
        while proc.poll() is None:
            now = time.monotonic()
            worker_rss_now = _proc_rss_bytes(proc.pid)
            job_rss_now = _proc_tree_rss_bytes(proc.pid)
            peak_worker_rss = max(peak_worker_rss, worker_rss_now)
            peak_job_rss = max(peak_job_rss, job_rss_now)
            cpu_now = _proc_cpu_seconds(proc.pid)
            job_cpu_now = _proc_tree_cpu_seconds(proc.pid)
            if cpu_now is not None: last_proc_cpu = cpu_now
            if job_cpu_now is not None: last_job_cpu = job_cpu_now
            if now >= next_storage_scan:
                peak_workspace_bytes = max(peak_workspace_bytes, _tree_bytes(work))
                next_storage_scan = now + storage_scan_interval
            if cancel_request_path.exists() and budget_stop_reason is None:
                budget_stop_reason = "OPERATOR_CANCEL"; _terminate_process_tree(proc); break
            if budget and budget_stop_reason is None:
                no_deadline = os.environ.get("IG_DECODER_EXECUTION_POLICY") == "NO_AUTOMATIC_RUNTIME_DEADLINE_V1"
                if not no_deadline and now - t0 > float(budget["wall_seconds_max"]):
                    budget_stop_reason = "WALL_SECONDS"; _terminate_process_tree(proc); break
                if peak_job_rss > int(budget["peak_rss_bytes_max"]):
                    budget_stop_reason = "PEAK_RSS_BYTES"; _terminate_process_tree(proc); break
                if not no_deadline and budget.get("cpu_seconds_max") and last_job_cpu is not None and last_job_cpu > float(budget["cpu_seconds_max"]):
                    budget_stop_reason = "CPU_SECONDS"; _terminate_process_tree(proc); break
                if peak_workspace_bytes > int(budget["workspace_peak_bytes_max"]):
                    budget_stop_reason = "WORKSPACE_BYTES"; _terminate_process_tree(proc); break
                if task_state_root is not None and len(list((task_state_root / "commits").glob("*.json"))) > int(budget["logical_tasks_max"]):
                    budget_stop_reason = "LOGICAL_TASKS"; _terminate_process_tree(proc); break
            if now >= next_hb:
                heartbeat_count += 1
                progress, last_commit_count, last_science_change = _task_progress_snapshot(task_state_root, last_count=last_commit_count, last_change_monotonic=last_science_change, now=now)
                hb_base = {
                    "schema_id": WORKER_HEARTBEAT_SCHEMA if admission.contract_version == "1.5.0" else WORKER_HEARTBEAT_SCHEMA_V14 if admission.contract_version == "1.4.0" else WORKER_HEARTBEAT_SCHEMA_V13,
                    "run_id": admission.run_id, "stage_id": admission.stage_id, "invocation_id": invocation_id, "worker_pid": proc.pid, "heartbeat_index": heartbeat_count, "utc": utc_now(), "elapsed_seconds": now - t0,
                    "liveness": "ALIVE", "science_progress": progress,
                    "no_progress_threshold_seconds": no_progress_threshold_seconds,
                    "no_progress_action": "REPORT_ONLY",
                    "no_progress": bool(progress.get("seconds_since_science_progress", 0.0) > no_progress_threshold_seconds),
                    "worker_rss_bytes_sampled": worker_rss_now,
                    "worker_cpu_seconds_procfs": cpu_now,
                    "job_process_tree_rss_bytes_sampled": job_rss_now,
                    "job_process_tree_cpu_seconds_procfs": job_cpu_now,
                }
                hb = dict(hb_base, heartbeat_sha256=canonical_sha256(hb_base))
                if heartbeat_path is not None: write_json_atomic(Path(heartbeat_path), hb)
                if heartbeat_callback is not None: heartbeat_callback(hb)
                next_hb = now + interval
            if test_kill_after is not None and task_state_root is not None:
                n = len(list((task_state_root / "commits").glob("*.json")))
                if n >= test_kill_after:
                    _terminate_process_tree(proc); fault_injected = True; break
            time.sleep(min(0.05, interval / 4.0))
        stdout, stderr = proc.communicate()
    finally:
        if proc.poll() is None:
            _terminate_process_tree(proc)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    finished_utc = utc_now(); wall = time.monotonic() - t0
    commits_after = len(list((task_state_root / "commits").glob("*.json"))) if task_state_root else 0
    if budget_stop_reason == "OPERATOR_CANCEL":
        terminal_status = "CANCELLED_OPERATOR"
    elif budget_stop_reason is not None:
        terminal_status = f"BUDGET_EXCEEDED_{budget_stop_reason}"
    else:
        terminal_status = "INTERRUPTED_TEST_FAULT" if fault_injected else ("PASS" if proc.returncode == 0 else "FAILED")
    tel_base = {
        "schema_id": WORKER_TELEMETRY_SCHEMA if admission.contract_version == "1.5.0" else WORKER_TELEMETRY_SCHEMA_V14 if admission.contract_version == "1.4.0" else WORKER_TELEMETRY_SCHEMA_V13, "contract_version": admission.contract_version,
        "run_id": admission.run_id, "stage_id": admission.stage_id, "invocation_id": invocation_id,
        "worker_pid": proc.pid, "worker_uid": uid, "worker_gid": gid,
        "started_utc": started_utc, "finished_utc": finished_utc, "wall_seconds": wall,
        "child_user_cpu_seconds_delta": child_after.ru_utime - child_before.ru_utime,
        "child_system_cpu_seconds_delta": child_after.ru_stime - child_before.ru_stime,
        "heartbeat_count": heartbeat_count, "return_code": proc.returncode, "terminal_status": terminal_status,
        "task_commits_before": commits_before, "task_commits_after": commits_after,
        "stdout_bytes": len(stdout.encode("utf-8")), "stderr_bytes": len(stderr.encode("utf-8")),
        "test_fault_injected": fault_injected,
        "budget_stop_reason": budget_stop_reason,
        "peak_worker_rss_bytes_observed": peak_worker_rss,
        "peak_job_rss_bytes_observed": peak_job_rss,
        "peak_workspace_bytes_observed": peak_workspace_bytes,
        "worker_cpu_seconds_procfs_last_observed": last_proc_cpu,
        "job_cpu_seconds_procfs_last_observed": last_job_cpu,
        "job_resource_accounting_scope": "WORKER_PROCESS_TREE",
        "workspace_scan_interval_seconds": storage_scan_interval,
        "resource_budget": budget or None,
        "process_transport_policy": "CONTROLLER_DESCENDED_FORK_WORKER_CONTEXT",
        "heartbeat_semantics": "CONTROLLER_LIVENESS_SEPARATE_FROM_SCIENCE_PROGRESS",
        "no_progress_threshold_seconds": no_progress_threshold_seconds,
        "no_progress_action": "REPORT_ONLY",
    }
    telemetry, telemetry_file = _write_telemetry(telemetry_dir, tel_base)

    result_path = output / "worker_result.json"
    if proc.returncode != 0:
        raise RuntimeError(f"isolated worker failed rc={proc.returncode} terminal={terminal_status}: {stderr[-4000:]}")
    if not result_path.is_file(): raise RuntimeError("isolated worker did not emit worker_result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_result_schema = WORKER_RESULT_SCHEMA if admission.contract_version == "1.5.0" else WORKER_RESULT_SCHEMA_V14 if admission.contract_version == "1.4.0" else WORKER_RESULT_SCHEMA_V13 if admission.contract_version == "1.3.0" else WORKER_RESULT_SCHEMA_V12
    if result.get("schema_id") != expected_result_schema or result.get("job_sha256") != job["job_sha256"]: raise RuntimeError("isolated worker result binding mismatch")
    observed_identity = result.get("worker_identity", {})
    self_ru = result.get("worker_resource_self", {})
    worker_span_file = None
    span_src = output / "v05_worker_spans.jsonl"
    if span_src.is_file() and telemetry_dir is not None:
        span_dst = Path(telemetry_dir) / f"worker-spans-{stage['stage_id']}-{invocation_id}.jsonl"
        shutil.copyfile(span_src, span_dst); worker_span_file = str(span_dst)
    if observed_identity.get("uid") != uid or observed_identity.get("gid") != gid: raise RuntimeError("isolated worker did not run under registered uid/gid")
    if set(result.get("outputs", {})) != set(admission.logical_outputs): raise RuntimeError("isolated worker output set mismatch")
    outputs = {}
    for logical, rel in sorted(result["outputs"].items()):
        rel = validate_relative_path(rel); p = _inside(output, output / rel, label="returned worker output")
        if not p.is_file(): raise RuntimeError(f"isolated worker output missing: {logical}")
        outputs[logical] = p
    if budget:
        output_bytes = _tree_bytes(output)
        if output_bytes > int(budget["bytes_written_max"]):
            raise RuntimeError("registered bytes_written budget exceeded")
    _chmod_readonly_tree(output)
    # Clean-exit self resource accounting is more specific than controller-wide
    # RUSAGE_CHILDREN. Keep both notions separate and label sampled data honestly.
    telemetry["worker_self_user_cpu_seconds"] = self_ru.get("user_cpu_seconds")
    telemetry["worker_self_system_cpu_seconds"] = self_ru.get("system_cpu_seconds")
    telemetry["worker_self_maxrss_bytes"] = self_ru.get("maxrss_bytes")
    telemetry["worker_rss_semantics"] = self_ru.get("maxrss_semantics")
    base_no_hash = {k:v for k,v in telemetry.items() if k != "telemetry_sha256"}
    telemetry["telemetry_sha256"] = canonical_sha256(base_no_hash)
    if telemetry_file is not None: write_json_atomic(Path(telemetry_file), telemetry)
    meta = {
        "worker_uid": uid, "worker_gid": gid, "worker_pid": observed_identity.get("pid"), "job_sha256": job["job_sha256"],
        "store_write_denied": store_denied, "publication_write_denied": publication_denied,
        "stdout": stdout, "stderr": stderr,
        "telemetry": telemetry, "telemetry_file": str(telemetry_file) if telemetry_file else None,
        "task_scope_sha256": task_scope_sha, "worker_resource_self": self_ru, "worker_span_file": worker_span_file,
        "process_transport_policy": "CONTROLLER_DESCENDED_FORK_WORKER_CONTEXT",
    }
    return {"outputs": outputs, "stage_result": result.get("stage_result", {})}, meta


def _cli() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True); ns = ap.parse_args()
    try:
        result = worker_main(Path(ns.job)); print(json.dumps({"status": "PASS", "job_sha256": result["job_sha256"]}, sort_keys=True)); return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
