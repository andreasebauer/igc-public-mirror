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
import importlib.abc
import importlib.machinery
from contextlib import contextmanager
from pathlib import Path
import re
import sys
from typing import Any, Iterable


PROJECT_REF = re.compile(
    r"^project(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*$"
)
BINDING_SCHEMA = "IG_DECODER_PROJECT_STAGE_CALLABLE_BINDING_V1"
_LOADED_PROJECT_MODULES = {}


@contextmanager
def _source_only_project_imports(source):
    """Compile admitted project bytes; stale timestamp-based pyc is not authority."""
    hashes={p.resolve():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source/'project').rglob('*.py')}
    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self,fullname):
            path=Path(self.path).resolve()
            raw=path.read_bytes()
            if hashes.get(path)!=hashlib.sha256(raw).hexdigest():
                raise ProjectStageError('PROJECT_STAGE_MODULE_HASH')
            return compile(raw,str(path),'exec',dont_inherit=True)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname!='project' and not fullname.startswith('project.'):
                return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec and spec.origin:
                origin=Path(spec.origin).resolve()
                if origin not in hashes:
                    raise ProjectStageError('PROJECT_STAGE_MODULE_ORIGIN')
                spec.loader=Loader(fullname,str(origin))
            return spec
    finder=Finder();sys.meta_path.insert(0,finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


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
    # Existing path equality does not prove which bytes supplied loaded code.
    # Only reuse modules imported by this resolver under the same verified pin.
    for loaded_name, loaded in list(sys.modules.items()):
        if loaded_name != 'project' and not loaded_name.startswith('project.'):
            continue
        origin_text = getattr(loaded,'__file__',None)
        if origin_text is None:
            paths=list(getattr(loaded,'__path__',()))
            spec=getattr(loaded,'__spec__',None)
            if (loaded_name == 'project' and paths and spec is not None
                    and spec.origin is None and all(Path(p).resolve()==source/'project' for p in paths)):
                continue
            raise ProjectStageError('PROJECT_STAGE_CACHED_MODULE_UNBOUND')
        origin = Path(origin_text).resolve()
        if not origin.is_relative_to(source/'project'):
            raise ProjectStageError('PROJECT_STAGE_MODULE_ORIGIN')
        expected=(loaded,binding['source_sha256'],str(origin),hashlib.sha256(origin.read_bytes()).hexdigest())
        if _LOADED_PROJECT_MODULES.get(loaded_name)!=expected:
            raise ProjectStageError('PROJECT_STAGE_CACHED_MODULE_UNBOUND')
    importlib.invalidate_caches()
    with _source_only_project_imports(source), scientific_call(source):
        fn = getattr(importlib.import_module(module), name, None)
    if engineering_source_tree_digest(source)!=binding['source_sha256']:
        raise ProjectStageError('PROJECT_STAGE_SOURCE_HASH')
    for loaded_name,loaded in list(sys.modules.items()):
        if loaded_name == 'project' or loaded_name.startswith('project.'):
            origin_text=getattr(loaded,'__file__',None)
            if origin_text:
                origin=Path(origin_text).resolve()
                if not origin.is_relative_to(source/'project'):
                    raise ProjectStageError('PROJECT_STAGE_MODULE_ORIGIN')
                _LOADED_PROJECT_MODULES[loaded_name]=(loaded,binding['source_sha256'],str(origin),hashlib.sha256(origin.read_bytes()).hexdigest())
    if not callable(fn):
        raise ProjectStageError("PROJECT_STAGE_CALLABLE_MISSING")
    return fn
