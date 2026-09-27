from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid import submission
from infinity_grid import portable_registry
from infinity_grid.project_stage import (
    ProjectStageError, bind_callables, callable_path, resolve_callable,
)
from infinity_grid import v05_controller_event_loop as loop


PROJECT_MODULE = '''
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult

def evaluate(payload):
    value = int(payload["value"])
    return {"signature": {"parity": value % 2}, "outcome_count": 1, "metrics": {}}

def handler(stage, runtime):
    tasks = []
    for value in range(8):
        payload = {"value": value}
        tasks.append(TaskSpec(
            task_id=f"v{value:02d}", task_kind="PROJECT_STAGE_FIXTURE",
            binding_sha256=canonical_sha256({"question": stage["question_sha256"], "payload": payload}),
            payload=payload, cost_weight=1.0))
    part = runtime.run_structural_partition(
        phase_id="PROJECT_PARTITION", tasks=tasks,
        evaluator_ref="project.fixture:evaluate", requested_workers=2, max_tasks=8)
    return ChainExecutionResult(result={"outcome": "PASS", "partition": part.summary})
'''


def _workspace(tmp_path: Path) -> tuple[Path, dict]:
    source = Path(loop.__file__).resolve().parents[1]
    project = tmp_path / "submitted-project"
    project.mkdir()
    (project / "fixture.py").write_text(PROJECT_MODULE, encoding="utf-8")
    spec = {
        "schema_id": submission.SPEC_SCHEMA, "job_id": "TEST.PROJECT.STAGE",
        "engine_source": str(source), "project_source": str(project),
        "question": {"stage_id": "ENG:PROJECT-STAGE", "description": "Source-bound project stage fixture",
                     "outcomes": ["PASS"], "stopping_rule": "Complete eight registered values."},
        "execution": {"kind": "STAGE", "handler_ref": "project.fixture:handler",
                      "evaluator_refs": ["project.fixture:evaluate"], "parameters": {}},
        "resources": {"workers": 2, "start_method": "fork", "memory_budget_bytes": 2 * 1024**3,
                      "workspace_budget_bytes": 2 * 1024**3, "wall_seconds_max": 60},
        "inputs": [],
        "environment": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                        "requirements": [], "artifacts": []},
        "output_contract": {"outcome": "PASS", "scientific_acceptance": "NONE; route fixture only"},
    }
    store = tmp_path / "store"
    portable_registry.initialize(store, "Project STAGE route fixture", source)
    state = submission.capture(store, spec)
    root = Path(state["workspace"])
    for item in list(state["pending_objects"]):
        readback = store / "objects" / item["object_name"]
        state = submission.confirm_save(
            root, item["sha256"], readback, "fixture_drive_id_0001",
            role=item["role"], logical_name=item["logical_name"])
    assert state["status"] == "SAVED"
    job = json.loads((root / "registry/TEST.PROJECT.STAGE.json").read_text())
    return root, job


def test_project_stage_executes_from_exact_captured_source(tmp_path):
    root, job = _workspace(tmp_path)
    code = (
        "import json,sys; "
        "from infinity_grid.v05_controller_event_loop import run_workspace_job; "
        "print(json.dumps(run_workspace_job(sys.argv[1],sys.argv[2]),sort_keys=True))"
    )
    env = dict(os.environ, PYTHONPATH=str(root / "source"), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-c", code, str(root), job["job_id"]],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["status"] == "COMPLETED"
    assert result["result"]["outcome"] == "PASS"
    assert result["result"]["partition"]["class_count"] == 2


def test_project_callable_requires_exact_source_and_module_binding(tmp_path):
    root, _job = _workspace(tmp_path)
    source = root / "source"
    sid, _pid = loop._source_ids(source)
    bindings = bind_callables(source, ["project.fixture:evaluate"], source_sha256=sid)
    binding = bindings["project.fixture:evaluate"]
    assert callable(resolve_callable("project.fixture:evaluate", binding))
    (source / "project/fixture.py").write_text(PROJECT_MODULE + "\nCHANGED = True\n", encoding="utf-8")
    with pytest.raises(ProjectStageError, match="PROJECT_STAGE_MODULE_HASH"):
        resolve_callable("project.fixture:evaluate", binding)


def test_unbound_or_outside_project_callable_refuses(tmp_path):
    root, _job = _workspace(tmp_path)
    source = root / "source"
    sid, _pid = loop._source_ids(source)
    with pytest.raises(ProjectStageError, match="PROJECT_STAGE_BINDING_REQUIRED"):
        resolve_callable("project.fixture:evaluate", None)
    with pytest.raises(ProjectStageError, match="PROJECT_STAGE_REFERENCE"):
        callable_path(source, "infinity_grid.controller_only_fixture:partition_evaluator")
