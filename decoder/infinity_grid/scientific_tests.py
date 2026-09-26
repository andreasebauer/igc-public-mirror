from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canon import canonical_sha256
from .jumpstart import JumpstartRuntime


class ScientificTestError(RuntimeError):
    pass


REGISTRY_SCHEMA = "IG_SCIENTIFIC_TEST_REGISTRY_V1"


def _resource_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "decoder" / "SCIENTIFIC_TEST_REGISTRY_v1.json"


def _json_pointer(obj: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return obj
    if not pointer.startswith("/"):
        raise ScientificTestError(f"JSON pointer must start with /: {pointer}")
    cur = obj
    for raw in pointer.split("/")[1:]:
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            cur = cur[tok]
        else:
            raise ScientificTestError(f"pointer traverses scalar at {tok}")
    return cur


class ScientificTestStore:
    def __init__(self, paths):
        self.paths = paths
        self.registry = json.loads(_resource_path().read_text(encoding="utf-8"))
        self._validate()

    def _validate(self) -> None:
        if self.registry.get("schema_id") != REGISTRY_SCHEMA:
            raise ScientificTestError("bad scientific test registry schema")
        tests = self.registry.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ScientificTestError("scientific test registry is empty")
        ids = [x.get("test_id") for x in tests]
        if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
            raise ScientificTestError("scientific test IDs must be unique nonempty strings")
        for t in tests:
            if t.get("execution") not in {"LAUNCH", "SMOKE"}:
                raise ScientificTestError(f"bad execution for {t['test_id']}")
            if not t.get("parent_target_alias"):
                raise ScientificTestError(f"missing parent target for {t['test_id']}")
            if t.get("replay_mode") not in {"DIRECT_TARGET", "EMBEDDED_GATE", "EMBEDDED_EVIDENCE_CHECK"}:
                raise ScientificTestError(f"bad replay_mode for {t['test_id']}")
            if t.get("evidence_relative_path") and "json_pointer" not in t:
                raise ScientificTestError(f"evidence pointer missing for {t['test_id']}")
        payload = {k: v for k, v in self.registry.items() if k != "registry_sha256"}
        observed = canonical_sha256(payload)
        if self.registry.get("registry_sha256") != observed:
            raise ScientificTestError("scientific test registry hash mismatch")

    def list(self, *, level: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.registry["tests"])
        if level:
            rows = [x for x in rows if x.get("level") == level]
        return sorted(rows, key=lambda x: x["test_id"])

    def show(self, test_id: str) -> dict[str, Any]:
        matches = [x for x in self.registry["tests"] if x["test_id"] == test_id]
        if len(matches) != 1:
            raise ScientificTestError(f"unknown scientific test {test_id}")
        return matches[0]

    def replay(self, test_id: str, *, workspace: Path | None = None, clean: bool = True) -> dict[str, Any]:
        spec = self.show(test_id)
        rt = JumpstartRuntime(self.paths)
        if spec["execution"] == "SMOKE":
            run = rt.smoke(spec["parent_target_alias"], workspace=workspace, clean=clean, timeout_seconds=spec.get("timeout_seconds"))
        else:
            run = rt.launch(spec["parent_target_alias"], workspace=workspace, clean=clean, detach=False)
            if run.get("status") != "COMPLETE" or int(run.get("returncode", 1)) != 0:
                raise ScientificTestError(f"parent target failed for {test_id}: {run}")
        work_root = Path(run["work_root"])
        evidence = None
        observed = None
        if spec.get("evidence_relative_path"):
            ep = work_root / spec["evidence_relative_path"]
            if not ep.is_file():
                raise ScientificTestError(f"evidence missing for {test_id}: {spec['evidence_relative_path']}")
            evidence = json.loads(ep.read_text(encoding="utf-8"))
            observed = _json_pointer(evidence, spec.get("json_pointer", "/"))
            if "expected_value" in spec and observed != spec["expected_value"]:
                raise ScientificTestError(f"{test_id} expected {spec['expected_value']!r}, observed {observed!r}")
        result = {
            "schema_id": "IG_SCIENTIFIC_TEST_REPLAY_RESULT_V1",
            "status": "PASS",
            "test_id": test_id,
            "level": spec.get("level"),
            "replay_mode": spec["replay_mode"],
            "parent_target_alias": spec["parent_target_alias"],
            "execution": spec["execution"],
            "observed": observed,
            "expected": spec.get("expected_value"),
            "evidence_relative_path": spec.get("evidence_relative_path"),
            "json_pointer": spec.get("json_pointer"),
            "workspace": run.get("workspace"),
            "work_root": run.get("work_root"),
            "parent_result": run,
            "authority_note": spec.get("authority_note"),
            "replay_semantics": spec.get("replay_semantics"),
        }
        return result

    def coverage(self) -> dict[str, Any]:
        levels = {f"O{i}": 0 for i in range(1, 8)}
        modes: dict[str, int] = {}
        for t in self.registry["tests"]:
            if t.get("level") in levels:
                levels[t["level"]] += 1
            modes[t["replay_mode"]] = modes.get(t["replay_mode"], 0) + 1
        return {
            "schema_id": "IG_SCIENTIFIC_TEST_COVERAGE_V1",
            "status": "PASS" if all(v > 0 for v in levels.values()) else "FAIL",
            "registry_sha256": self.registry["registry_sha256"],
            "tests": len(self.registry["tests"]),
            "levels": levels,
            "replay_modes": modes,
        }
