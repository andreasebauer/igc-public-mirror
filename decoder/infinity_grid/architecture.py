from __future__ import annotations

import ast
import json
from importlib.resources import files
from typing import Any


def architecture_contract() -> dict[str, Any]:
    return json.loads(files("infinity_grid").joinpath("resources/architecture/ARC001_LAYER_CONTRACT_V1.json").read_text(encoding="utf-8"))


def _root(name: str) -> str:
    return name.split(".", 1)[0]


def verify_core_source_tree(package_root) -> dict[str, Any]:
    """Static fail-closed verifier for the pure-core source boundary."""
    from pathlib import Path
    root = Path(package_root)
    contract = architecture_contract()
    forbidden_imports = set(contract["forbidden_core_import_roots"])
    forbidden_calls = set(contract["forbidden_core_calls"])
    mutable_ctor = set(contract["forbidden_core_mutable_globals"])
    failures = []
    checked = 0
    for rel in contract["pure_core_modules"]:
        p = root / rel
        if not p.is_file():
            failures.append({"path": rel, "reason": "MISSING_CORE_MODULE"}); continue
        checked += 1
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if _root(a.name) in forbidden_imports:
                        failures.append({"path":rel,"line":node.lineno,"reason":"FORBIDDEN_IMPORT","name":a.name})
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 0 and _root(mod) in forbidden_imports:
                    failures.append({"path":rel,"line":node.lineno,"reason":"FORBIDDEN_IMPORT","name":mod})
                if node.level and mod and not mod.startswith(("canonical","boundary","l2","invariants")):
                    failures.append({"path":rel,"line":node.lineno,"reason":"CORE_IMPORTS_NONCORE_RELATIVE","name":mod})
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                failures.append({"path":rel,"line":node.lineno,"reason":"FORBIDDEN_CALL","name":node.func.id})
        # Mutable module globals are forbidden; function-local containers are fine.
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = getattr(node, "value", None)
                if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
                    failures.append({"path":rel,"line":node.lineno,"reason":"MUTABLE_MODULE_GLOBAL"})
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in mutable_ctor:
                    failures.append({"path":rel,"line":node.lineno,"reason":"MUTABLE_MODULE_GLOBAL","name":value.func.id})
    return {"schema_id":"IG_DECODER_ARC001_PURITY_RESULT_V1","status":"PASS" if not failures else "FAIL","core_modules_checked":checked,"failure_count":len(failures),"failures":failures,"contract_sha256":contract["contract_sha256"]}
