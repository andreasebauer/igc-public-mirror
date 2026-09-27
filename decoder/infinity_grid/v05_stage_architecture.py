from __future__ import annotations

"""Decoder 0.6 targeted scientific source preflight (historical module name).

The rule enforced here is intentionally simple and hard:
scientific stage modules may describe science and call the Decoder-owned stage
runtime, but shared execution infrastructure remains Decoder-owned.
Temporary files, ordinary main functions and pure caching are allowed.

Historical modules remain readable for provenance.  New V2 stage handlers and
worker evaluators are admitted only after this static gate passes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import ast
import inspect


ARCHITECTURE_GATE_SCHEMA = "IG_DECODER_V05_CONTROLLER_ONLY_STAGE_ARCHITECTURE_GATE_V1"


class StageArchitectureViolation(RuntimeError):
    pass


# Imports, ordinary file utilities and function names do not confer authority.
# Reject specific execution calls; the runtime process guard handles Python
# dispatch that static analysis cannot resolve.
_FORBIDDEN_IMPORT_PREFIXES = ()
_FORBIDDEN_CALL_NAMES = {
    'ProcessPoolExecutor', 'ThreadPoolExecutor', 'Pool', 'Process', 'Popen',
    'fork', 'forkpty', 'system', 'popen', 'exec', 'eval', '__import__',
}
_FORBIDDEN_EXACT_CALLS = {
    'subprocess.run', 'subprocess.call', 'subprocess.check_call', 'subprocess.check_output',
    'threading.Thread', 'threading.Timer', '_thread.start_new_thread',
    'joblib.Parallel', 'importlib.import_module', 'runpy.run_path', 'runpy.run_module',
    'os.posix_spawn', 'os.posix_spawnp', 'os.spawnl', 'os.spawnv',
}


def _forbidden_import(name):
    return False


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        p = _dotted_name(node.value)
        return f"{p}.{node.attr}" if p else node.attr
    return None


def _write_mode_literal(call: ast.Call) -> bool:
    # Fail closed for dynamic modes; recognize both open(path, mode) and
    # Path(...).open(mode). This is lint, not a security sandbox.
    mode_node = None
    direct = isinstance(call.func, ast.Name)
    index = 1 if direct else 0
    if len(call.args) > index:
        mode_node = call.args[index]
    for kw in call.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
        elif kw.arg is None:
            return True
    if mode_node is None:
        return False  # default read-only mode
    if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
        return True
    return any(ch in mode_node.value for ch in "wax+")


def audit_module_source(module_path: str | Path) -> dict[str, Any]:
    path = Path(module_path).resolve(strict=True)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if _forbidden_import(name):
                    violations.append({"line": node.lineno, "kind": "FORBIDDEN_IMPORT", "detail": name})
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if _forbidden_import(name):
                violations.append({"line": node.lineno, "kind": "FORBIDDEN_IMPORT", "detail": name})
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func) or ""
            head, dot, tail = name.partition(".")
            name = aliases.get(head, head) + (dot + tail if dot else "")
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _FORBIDDEN_CALL_NAMES or name in _FORBIDDEN_EXACT_CALLS or name.startswith("os.exec"):
                violations.append({"line": node.lineno, "kind": "FORBIDDEN_EXECUTION_CALL", "detail": name})
    return {
        "schema_id": ARCHITECTURE_GATE_SCHEMA,
        "module_path": str(path),
        "status": "PASS" if not violations else "FAIL",
        "violation_count": len(violations),
        "violations": sorted(violations, key=lambda x: (int(x["line"]), x["kind"], x["detail"])),
    }


def audit_callable(fn: Callable[..., Any]) -> dict[str, Any]:
    path = inspect.getsourcefile(fn)
    if not path:
        raise StageArchitectureViolation(f"cannot resolve source file for {fn!r}")
    result = audit_module_source(path)
    result["callable"] = f"{fn.__module__}:{getattr(fn, '__name__', type(fn).__name__)}"
    return result


def require_controller_only_callable(fn: Callable[..., Any], *, role: str) -> dict[str, Any]:
    result = audit_callable(fn)
    if result["status"] != "PASS":
        first = result["violations"][:5]
        raise StageArchitectureViolation(
            f"{role} violates Decoder controller-only stage architecture: {first}"
        )
    return result


def audit_registered_modules(paths: list[str | Path]) -> dict[str, Any]:
    rows = [audit_module_source(p) for p in paths]
    return {
        "schema_id": "IG_DECODER_V05_CONTROLLER_ONLY_ARCHITECTURE_AUDIT_V1",
        "status": "PASS" if all(x["status"] == "PASS" for x in rows) else "FAIL",
        "module_count": len(rows),
        "violation_count": sum(int(x["violation_count"]) for x in rows),
        "modules": rows,
    }
