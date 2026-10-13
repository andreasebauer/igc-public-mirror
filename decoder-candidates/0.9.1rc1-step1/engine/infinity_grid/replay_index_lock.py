"""POSIX advisory serialization for the replay stores' mutable indexes.

The lock is deliberately nonblocking: competing callers must reopen/retry.
It protects cooperating local writers, not an external filesystem rollback.
Public operations acquire it once; private helpers run inside that operation.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def replay_index_lock(root: Path, error_type):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".replay-index.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise error_type("REPLAY_INDEX_BUSY: reopen and retry") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
