"""Typed native command preparation and bounded read-only observations.

Mutating commands are returned as plans for the later durable worker. They are
never executed in an HTTP request. Only native status commands run here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from .models import Job, NativeObservation, Settings, Task

PIN_COMMIT = "792fc17f557c1414c882fa424b77f9a7f3bee458"
PIN_SOURCE = "03f734300ad3812237c8168321f5af99fe35cdbcd1d1b19840d71c68398bd07d"
PIN_VERSION = "0.8.0.dev84+lib"
MAX_BYTES = 2 * 1024 * 1024


class AdapterError(Exception):
    def __init__(self, code: str, status: int = 503):
        self.code = code
        self.status = status
        super().__init__(code)


def read_json(path: Path, limit: int = MAX_BYTES):
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise AdapterError("RECORD_TOO_LARGE")
    try:
        return json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise AdapterError("INVALID_JSON_RECORD") from exc


def contained(root: Path, candidate: str, *, directory: bool) -> Path:
    # Disallow symlinks even when they point back into the approved tree.
    raw = Path(candidate)
    if not raw.is_absolute():
        raw = root / raw
    if ".." in raw.parts:
        raise AdapterError("PATH_NOT_ALLOWED", 400)
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise AdapterError("PATH_NOT_ALLOWED", 400)
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise AdapterError("PATH_NOT_AVAILABLE", 503) from exc
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise AdapterError("PATH_WRONG_TYPE", 400)
    return resolved


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Command:
    operation: str
    target: str
    argv: tuple[str, ...]
    cwd: str
    source_sha256: str
    specification_sha256: str | None = None
    # Worker must re-check before dispatch; a prepared plan is not authorization.


class NativeAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.repository = Path(settings.engine_repository)
        self.source = self.repository / "decoder"

    def verify_source(self):
        source = self.source
        if source.is_symlink() or not source.is_dir():
            raise AdapterError("ENGINE_SOURCE_UNAVAILABLE")
        rows = []
        skip = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".engineering_tmp"}
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if any(x in skip for x in rel.parts) or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise AdapterError("ENGINE_SOURCE_SYMLINK")
            if path.is_file():
                rows.append([rel.as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()])
        actual = canonical_hash({"schema_id": "IG_DECODER_ENGINEERING_SOURCE_TREE_V1", "files": rows})
        if actual != PIN_SOURCE:
            raise AdapterError("ENGINE_SOURCE_MISMATCH")
        return {"status": "BYTE_EXACT_SOURCE_PASS", "files": len(rows), "source_sha256": actual}

    def command(self, operation: str, job: Job | None = None, *, task: Task | None = None,
                reason: str | None = None) -> Command:
        self.verify_source()
        interpreter = Path(self.settings.engine_python)
        # venv Python is often a symlink; resolve it only to check existence,
        # retaining the venv path for invocation and dependency isolation.
        if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise AdapterError("ENGINE_INTERPRETER_UNAVAILABLE")
        args: list[str]
        if operation == "capture" and task is not None:
            spec = contained(Path(self.settings.specification_root), task.specification, directory=False)
            if hashlib.sha256(spec.read_bytes()).hexdigest() != task.specification_sha256:
                raise AdapterError("SPECIFICATION_CHANGED", 409)
            read_json(spec)
            store = contained(Path(self.settings.capture_store), self.settings.capture_store, directory=True)
            args = ["capture", str(store), str(spec)]
            target = task.id
        elif job is not None and operation in {"status", "pending-saves", "preservation", "run", "pause"}:
            workspace = contained(Path(self.settings.workspace_root), job.workspace, directory=True)
            target = job.id
            if operation == "run":
                args = ["run", str(workspace), job.native_job_id]
            elif operation == "pause":
                if not reason or len(reason) > 500 or "\x00" in reason:
                    raise AdapterError("INVALID_PAUSE_REASON", 400)
                args = ["preserve", "pause", str(workspace), reason]
            elif operation == "preservation":
                args = ["preserve", "status", str(workspace)]
            else:
                args = [operation, str(workspace)]
        else:
            raise AdapterError("UNSUPPORTED_OPERATION", 400)
        return Command(operation, target, (str(interpreter), "-B", "-m", "infinity_grid.controller", *args),
                       str(self.source), PIN_SOURCE, task.specification_sha256 if operation == "capture" else None)

    def environment(self):
        # Do not pass the web bearer token or ambient PYTHONPATH to Decoder.
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR") if k in os.environ}
        env.update(PYTHONPATH=str(self.source), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        return env

    def observe(self, operation: str, job: Job) -> NativeObservation:
        if operation not in {"status", "pending-saves", "preservation"}:
            raise AdapterError("READ_OPERATION_REQUIRED", 400)
        cmd = self.command(operation, job)
        stdout, stderr, rc = self._read_process(cmd)
        native = None
        # Refusals are usually on stderr; retain both streams in every case.
        for stream in ((stderr, stdout) if rc else (stdout, stderr)):
            try:
                parsed = json.loads(stream)
                if isinstance(parsed, dict):
                    native = parsed
                    break
            except ValueError:
                pass
        return NativeObservation(observed_at=datetime.now(timezone.utc).isoformat(), exit_code=rc,
                                 native=native, stdout=stdout, stderr=stderr,
                                 classification="UNPARSEABLE" if native is None else "REFUSED" if rc else "RESPONSE")

    def _read_process(self, cmd: Command):
        # POSIX reader bounds memory and time for status only. Never use this
        # timeout policy for science jobs; those belong to the durable worker.
        try:
            process = subprocess.Popen(cmd.argv, cwd=cmd.cwd, env=self.environment(),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=False)
        except OSError as exc:
            raise AdapterError("NATIVE_STATUS_UNAVAILABLE") from exc
        chunks = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + 20
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AdapterError("NATIVE_STATUS_TIMEOUT")
                    for key, _ in selector.select(min(remaining, 0.25)):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        chunks[key.data].extend(block)
                        if sum(map(len, chunks.values())) > MAX_BYTES:
                            raise AdapterError("NATIVE_STATUS_OUTPUT_LIMIT")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdapterError("NATIVE_STATUS_TIMEOUT")
                try:
                    rc = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise AdapterError("NATIVE_STATUS_TIMEOUT") from exc
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()
        return chunks["stdout"].decode("utf-8", "replace"), chunks["stderr"].decode("utf-8", "replace"), rc
