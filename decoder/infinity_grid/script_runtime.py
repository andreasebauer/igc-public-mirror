"""Controller-owned execution of one captured serial project entrypoint."""
from __future__ import annotations
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
import traceback

from .v05_origin_guard import require_controller_execution_origin
from .canon import write_json_atomic


class _FrozenProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, project):
        self.project = project.resolve()
        self.files = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in self.project.rglob('*.py')}

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.origin and spec.origin not in {'built-in', 'frozen'}:
            origin = Path(spec.origin).resolve()
            if origin.is_relative_to(self.project):
                expected = self.files.get(str(origin))
                if expected is None or hashlib.sha256(origin.read_bytes()).hexdigest() != expected:
                    raise ImportError('UNREGISTERED_PROJECT_HELPER:' + str(origin))
        return None


def run_captured_script(admission, output_dir):
    from .v05_origin_guard import require_registered_dispatch, require_native_caller
    require_native_caller('infinity_grid.v05_controller_event_loop', {'_dispatch_workspace_job'}, 'run_captured_script')
    require_registered_dispatch(admission, output_dir, 'run_captured_script')
    job = admission['job']; ex = job['execution']; source = admission['source']
    from .execution_policy import automatic_deadlines
    timed = automatic_deadlines(job['resources'])
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'; work.mkdir(exist_ok=True)
    write_json_atomic(out / 'PARAMETERS.json', ex['parameters'])
    write_json_atomic(out / 'INPUTS.json', {k: str(v) for k, v in admission['artifacts'].items()})
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
                os.dup2(stdout.fileno(), 1); os.dup2(stderr.fileno(), 2)
                sys.stdout = stdout; sys.stderr = stderr
                os.chdir(work)
                project = source / 'project'
                sys.path.insert(0, str(project)); sys.path.insert(1, str(source))
                roots = {p.stem for p in project.glob('*.py')} | {p.name for p in project.iterdir() if p.is_dir()}
                for name in list(sys.modules):
                    if name.split('.')[0] in roots:
                        sys.modules.pop(name, None)
                sys.meta_path.insert(0, _FrozenProjectFinder(project))
                os.environ['DECODER_PARAMETERS_PATH'] = str(out / 'PARAMETERS.json')
                os.environ['DECODER_INPUTS_PATH'] = str(out / 'INPUTS.json')
                sys.argv = [str(source / ex['entrypoint']), *ex['argv']]
                try:
                    from .workflow_guard import scientific_call
                    with scientific_call(source):
                        runpy.run_path(str(source / ex['entrypoint']), run_name='__main__')
                    code = 0
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
                except BaseException as exc:
                    from .invocation import refusal_details
                    if hasattr(exc, 'code'):
                        write_json_atomic(out/'REFUSAL.json', refusal_details(exc, 'SCRIPT', admission['workspace'], job['job_id']))
                    traceback.print_exc(); code = 1
                stdout.flush(); stderr.flush()
                os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
        except BaseException:
            # Keep startup failures in controller-owned evidence too.
            with (out / 'startup_error.log').open('w') as f:
                traceback.print_exc(file=f)
        os._exit(code if 0 <= code <= 255 else 1)
    timeout = False
    try:
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if timed and time.monotonic() - started >= job['resources']['wall_seconds_max']:
                os.kill(pid, signal.SIGKILL); _, status = os.waitpid(pid, 0); timeout = True
                break
            from .preservation import poll
            poll()
            time.sleep(0.02)
    except BaseException:
        try:os.kill(pid,signal.SIGKILL)
        except OSError:pass
        try:os.waitpid(pid,0)
        except OSError:pass
        raise
    code = 124 if timeout else os.waitstatus_to_exitcode(status)
    result = {'status': 'PASS' if code == 0 else 'FAIL', 'exit_code': code,
              'outcome': 'PROCESS_COMPLETED' if code == 0 else 'PROCESS_FAILED',
              'scope': 'Captured script process outcome; scientific output acceptance is separate.',
              'execution_metadata': {'worker_pid': pid, 'workers_used': 1,
                                     'wall_seconds': time.monotonic() - started, 'start_method': 'fork'}}
    if (out/'REFUSAL.json').is_file():
        result['refusal'] = json.loads((out/'REFUSAL.json').read_text())
    write_json_atomic(out / 'SCRIPT_RESULT.json', result)
    return result
