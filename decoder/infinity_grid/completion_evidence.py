"""Publish a separate, closed evidence tree; never seal a mutable working tree.

Legacy completions keep their original working-tree verification semantics.
New snapshots retain every ordinary artifact and use SQLite's backup API only
for engine-owned task databases, after the existing quiescence gate.
"""
from contextlib import closing
from pathlib import Path
import os
import shutil
import sqlite3
import tempfile

PROTOCOL = 'SEPARATE_COMPLETION_EVIDENCE_V1'


def evidence_root(workspace, working, done):
    from .v05_controller_event_loop import ControllerLoopError
    if 'evidence_protocol' not in done and 'evidence_root' not in done:
        return working
    expected = 'runtime/sealed/' + working.name
    if done.get('evidence_protocol') != PROTOCOL or done.get('evidence_root') != expected:
        raise ControllerLoopError('COMPLETION_EVIDENCE_LOCATION')
    root = Path(workspace) / expected
    if root.is_symlink() or root.parent.is_symlink():
        raise ControllerLoopError('EVIDENCE_SYMLINK')
    return root


def stage_snapshot(working, destination):
    """Return a private tree ready for contract verification, without publishing."""
    from .preservation import _stable_file, require_quiescent_task_databases
    from .v05_controller_event_loop import _evidence_rows, ControllerLoopError
    working, destination = Path(working), Path(destination)
    require_quiescent_task_databases(working)
    before = _evidence_rows(working)
    destination.mkdir(parents=True, exist_ok=False)
    for row in before:
        rel = Path(row['path']); source = working / rel; target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        engine_db = (rel.parts[:2] == ('chain', 'decoder_stage_runtime')
                     and source.name in {'partition.sqlite3', 'state_store.sqlite3'})
        if engine_db:
            with closing(sqlite3.connect(source.as_uri()+'?mode=ro', uri=True, timeout=5)) as src, closing(sqlite3.connect(target)) as dst:
                src.backup(dst)
                if dst.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ControllerLoopError('COMPLETION_DATABASE_INVALID')
                if dst.execute('PRAGMA journal_mode=DELETE').fetchone()[0].lower() != 'delete':
                    raise ControllerLoopError('COMPLETION_DATABASE_NOT_CLOSED')
        else:
            target.write_bytes(_stable_file(source, quiescent=True))
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
    require_quiescent_task_databases(working)
    if _evidence_rows(working) != before:
        raise ControllerLoopError('COMPLETION_WORKING_EVIDENCE_CHANGED')
    require_quiescent_task_databases(destination)


def prepare_evidence(admission, working, result):
    """Check the exact snapshot that will be sealed, then publish without overwrite."""
    from .canon import write_json_atomic
    from .preservation import check_budget
    from .result_contracts import contract_for, verify
    from .v05_controller_event_loop import _evidence_rows, ControllerLoopError
    working = Path(working); parent = Path(admission['workspace'])/'runtime/sealed'
    check_budget(admission, extra_bytes=sum(r['size_bytes'] for r in _evidence_rows(working)))
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='completion-snapshot-', dir=parent.parent))
    try:
        staged = temp/'evidence'
        stage_snapshot(working, staged)
        verification = verify(contract_for(admission['workspace']), result, staged)
        write_json_atomic(staged/'RESULT_VERIFICATION.json', verification)
        target = parent/working.name
        if target.exists():
            if target.is_symlink() or _evidence_rows(target) != _evidence_rows(staged):
                raise ControllerLoopError('COMPLETION_SNAPSHOT_COLLISION')
        else:
            os.rename(staged, target)
        return target, verification
    finally:
        shutil.rmtree(temp)
