from __future__ import annotations

"""Source-bound project callables for registered STAGE jobs.

This module is deliberately narrow.  A ``project.*`` handler or evaluator is
admitted only when all of the following are true:

* the job registration names the callable;
* the complete captured workspace source matches its recorded source SHA-256;
* the callable resolves below that workspace's ``source/project`` directory;
* the exact module bytes match the binding made at admission; and
* the handler plus its local import closure pass the normal scientific preflight.

Project evaluators receive no Decoder kernel-service view.  Built-in
``infinity_grid.*`` callables continue to use the fixed registry and are not
affected by this route.
"""

import hashlib
import importlib
from pathlib import Path
import re
import sys
from typing import Any, Iterable


PROJECT_REF = re.compile(
    r"^project(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*$"
)
BINDING_SCHEMA = "IG_DECODER_PROJECT_STAGE_CALLABLE_BINDING_V1"


class ProjectStageError(RuntimeError):
    pass


def is_project_ref(ref: object) -> bool:
    return isinstance(ref, str) and PROJECT_REF.fullmatch(ref) is not None


def callable_path(source_root: str | Path, ref: str) -> Path:
    """Return the exact captured module path for one syntactically valid ref."""
    if not is_project_ref(ref):
        raise ProjectStageError("PROJECT_STAGE_REFERENCE")
    source = Path(source_root).resolve(strict=True)
    module = ref.split(":", 1)[0]
    path = (source / (module.replace(".", "/") + ".py")).resolve(strict=True)
    project = (source / "project").resolve(strict=True)
    if not path.is_relative_to(project) or not path.is_file():
        raise ProjectStageError("PROJECT_STAGE_MODULE_PATH")
    return path


def bind_callables(
    source_root: str | Path,
    refs: Iterable[str],
    *,
    source_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Freeze runtime-only bindings for the registered project callables.

    The binding is derived from the already captured source.  It is not a new
    user-supplied trust claim: the job registration binds the refs and the
    workspace manifest binds the whole source tree.
    """
    source = Path(source_root).resolve(strict=True)
    paths = {ref: callable_path(source, ref) for ref in refs if is_project_ref(ref)}
    if paths:
        from .workflow_guard import preflight
        preflight(source, paths.values())
    return {
        ref: {
            "schema_id": BINDING_SCHEMA,
            "ref": ref,
            "source_root": str(source),
            "source_sha256": str(source_sha256),
            "module_path": path.relative_to(source).as_posix(),
            "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for ref, path in sorted(paths.items())
    }


def resolve_callable(ref: str, binding: dict[str, Any] | None):
    """Resolve one project callable from its exact captured source or refuse."""
    if not is_project_ref(ref) or type(binding) is not dict:
        raise ProjectStageError("PROJECT_STAGE_BINDING_REQUIRED")
    required = {
        "schema_id", "ref", "source_root", "source_sha256",
        "module_path", "module_sha256",
    }
    if set(binding) != required or binding.get("schema_id") != BINDING_SCHEMA or binding.get("ref") != ref:
        raise ProjectStageError("PROJECT_STAGE_BINDING_FIELDS")
    source = Path(binding["source_root"]).resolve(strict=True)
    path = callable_path(source, ref)
    if path.relative_to(source).as_posix() != binding["module_path"]:
        raise ProjectStageError("PROJECT_STAGE_BINDING_PATH")
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding["module_sha256"]:
        raise ProjectStageError("PROJECT_STAGE_MODULE_HASH")
    from .v05_engineering_worker import engineering_source_tree_digest
    if engineering_source_tree_digest(source) != binding["source_sha256"]:
        raise ProjectStageError("PROJECT_STAGE_SOURCE_HASH")
    from .workflow_guard import preflight, scientific_call
    preflight(source, [path])
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module, name = ref.split(":", 1)
    prior = sys.modules.get(module)
    if prior is not None:
        origin = Path(getattr(prior, "__file__", "")).resolve()
        if origin != path:
            raise ProjectStageError("PROJECT_STAGE_MODULE_ORIGIN")
    importlib.invalidate_caches()
    with scientific_call(source):
        fn = getattr(importlib.import_module(module), name, None)
    if not callable(fn):
        raise ProjectStageError("PROJECT_STAGE_CALLABLE_MISSING")
    return fn
