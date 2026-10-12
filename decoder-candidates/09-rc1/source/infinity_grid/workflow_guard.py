"""Focused pre-import checks and process-use checks for scientific code.

The worker process and these Python checks do not confine arbitrary native code
or a terminal that can edit the installed runtime. They enforce the supported
workflow and retain useful refusal evidence for accidental unsupported calls.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
from functools import wraps
from contextvars import ContextVar
import importlib.abc
import importlib.machinery
import hashlib
import os
from pathlib import Path
import sys
from .invocation import InvocationRefused

# Explicit runtime boundaries, never an exemption for a whole version prefix.
RUNTIME_MODULES = frozenset({
    'infinity_grid', 'infinity_grid.canon', 'infinity_grid.execution',
    'infinity_grid.v05_chain', 'infinity_grid.v05_stage_runtime',
    'infinity_grid.v05_origin_guard', 'infinity_grid.v05_execution_authority',
    'infinity_grid.controller', 'infinity_grid.v05_controller_event_loop',
    'infinity_grid.submission', 'infinity_grid.change_sessions',
    'infinity_grid.change_validation', 'infinity_grid.invocation',
    'infinity_grid.replay_dag_runner', 'infinity_grid.replay_root_job',
    'infinity_grid.workflow_guard', 'infinity_grid.v05_stage_registry',
    'infinity_grid.v05_kernel_services', 'infinity_grid.v05_stage_architecture',
})
_SCIENCE = ContextVar('decoder_scientific_call', default=None)
_HOOK_INSTALLED = False
_FINDERS = {}
_PROCESS_EVENTS = frozenset({'os.fork', 'os.forkpty', 'os.posix_spawn',
                            'os.exec', 'os.system', 'subprocess.Popen'})


def _module_file(source, module):
    for root in (Path(source)/'project', Path(source)):
        path = root.joinpath(*module.split('.'))
        for candidate in (path.with_suffix('.py'), path/'__init__.py'):
            if candidate.is_file(): return candidate.resolve()
    return None


def _imports(path, source):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    root = Path(source)/'project' if path.is_relative_to(Path(source)/'project') else Path(source)
    parts = path.relative_to(root).with_suffix('').parts
    package = list(parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (x.name for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (package[:len(package)-node.level+1] if node.level else [])
            if node.module: base += node.module.split('.')
            prefix = '.'.join(base)
            if prefix: yield prefix
            for alias in node.names:
                if alias.name != '*': yield '.'.join([*base, alias.name])


def preflight(source, paths):
    """Inspect explicit handlers AND their local helper import closure before import."""
    from .v05_stage_architecture import audit_module_source
    source = Path(source).resolve()
    pending = [Path(p).resolve() for p in paths]
    seen = set(); rows = []
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path)
        if not path.is_relative_to(source):
            raise InvocationRefused('HELPER_OUTSIDE_CAPTURE', 'preflight', checks=[str(path)])
        try:
            row = audit_module_source(path)
        except SyntaxError as exc:
            raise InvocationRefused('SOURCE_SYNTAX_ERROR', 'preflight', checks=[str(exc)]) from exc
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
        if row['status'] != 'PASS':
            raise InvocationRefused('PRIVATE_EXECUTION_PREFLIGHT', 'preflight',
                checks=[dict(path=str(path.relative_to(source)), **v) for v in row['violations']],
                next_operation='capture', required_arguments={'store': None, 'specification': None,
                    'correction': 'Keep pure helper logic. Submit parallel/resumable work as a STAGE using runtime.run_structural_partition or runtime.run_content_indexed_generation.'})
        for name in _imports(path, source):
            if name in RUNTIME_MODULES: continue
            target = _module_file(source, name)
            if target: pending.append(target)
    return {'status': 'PASS', 'scope': 'Targeted Python source checks; not arbitrary-code confinement',
            'modules': sorted(rows, key=lambda r:r['module_path'])}


def preflight_job(admission):
    source = admission['source']; ex = admission['job']['execution']
    if ex['kind'] == 'VALIDATION': return {'status':'ENGINEERING_VALIDATION'}
    # Exact native engineering adapters inspect test evidence under the saved
    # attempt's real authority; scientific handlers still receive full preflight.
    if ex['kind'] == 'STAGE' and ex['handler_ref'] in {
            'infinity_grid.change_validation:validate_revision',
            'infinity_grid.representative_qualification:handler'}:
        return {'status': 'NATIVE_ENGINEERING_ADAPTER'}
    paths = ([source/ex['entrypoint']] if ex['kind'] == 'SCRIPT' else
             [_module_file(source, r.split(':', 1)[0]) for r in [ex['handler_ref'], *ex['evaluator_refs']]])
    return preflight(source, paths)


class FrozenScienceImports(importlib.abc.MetaPathFinder):
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.files = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in self.source.rglob('*.py')}

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin not in {'built-in','frozen'}:
            file = Path(spec.origin).resolve()
            if file.is_relative_to(self.source):
                if self.files.get(str(file)) != hashlib.sha256(file.read_bytes()).hexdigest():
                    raise InvocationRefused('CHANGED_SOURCE_HELPER', 'import', checks=[fullname])
                if _SCIENCE.get() is None: return None
                frame = sys._getframe(1)
                importer = None
                while frame:
                    name = frame.f_globals.get('__name__', '')
                    if name.startswith('infinity_grid.') and name != __name__:
                        importer = name; break
                    if frame.f_code.co_filename.startswith(str(self.source/'project')):
                        importer = 'project'; break
                    frame = frame.f_back
                if fullname not in RUNTIME_MODULES and importer not in RUNTIME_MODULES:
                    preflight(self.source, [file])
        return None


def _audit(event, args):
    scope = _SCIENCE.get()
    if scope is None or event not in _PROCESS_EVENTS: return
    source, allow_scheduler = scope
    # Restored admission source and the loaded Decoder runtime can differ.
    # Inspect both roots; only the authentic shared scheduler retains permission.
    runtime_source = Path(__file__).resolve().parents[1]
    # The closest project/runtime frame distinguishes Decoder's shared scheduler from
    # a private pool hidden behind a standard-library helper.
    frame = sys._getframe(1)
    while frame:
        filename = Path(frame.f_code.co_filename)
        if filename.is_absolute() and (filename.is_relative_to(source) or filename.is_relative_to(runtime_source)):
            module = frame.f_globals.get('__name__')
            if module == __name__:
                frame = frame.f_back; continue
            if allow_scheduler and module == 'infinity_grid.execution' and frame.f_globals is vars(sys.modules[module]):
                from .v05_origin_guard import require_controller_execution_origin
                require_controller_execution_origin(event)
                return
            break
        frame = frame.f_back
    raise InvocationRefused('PRIVATE_EXECUTION_RUNTIME', event,
        checks=['Use Decoder shared task service for parallel work; this scientific process cannot create another process.'],
        next_operation='capture', required_arguments={'store': None, 'specification': None,
            'correction': 'Register a STAGE with the same pure evaluator and runtime.run_structural_partition.'})


@contextmanager
def scientific_call(source, *, allow_scheduler=False):
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit); _HOOK_INSTALLED = True
    source = Path(source).resolve()
    finder = _FINDERS.get(source)
    if finder is None:
        finder = _FINDERS[source] = FrozenScienceImports(source)
    added = finder not in sys.meta_path
    if added: sys.meta_path.insert(0, finder)
    token = _SCIENCE.set((source, allow_scheduler))
    try: yield
    finally:
        _SCIENCE.reset(token)
        if added: sys.meta_path.remove(finder)


def runtime_initialization(fn):
    """Native kernel initialization keeps byte checking without science-body lint.

    Registered evaluator/helper preflight still runs explicitly in _resolve_ref.
    This scope cannot be opened through the public source-checking API.
    """
    @wraps(fn)
    def initialize(*args, **kwargs):
        from .v05_origin_guard import require_native_caller
        require_native_caller('infinity_grid.execution', {'_pool_initializer', '_serial_initializer'}, 'native-worker-initialization')
        token = _SCIENCE.set(None)
        try: return fn(*args, **kwargs)
        finally: _SCIENCE.reset(token)
    return initialize
