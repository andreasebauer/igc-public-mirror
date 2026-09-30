"""POSIX immutable byte publication with directory durability and retry checks."""
import os
from pathlib import Path
import tempfile


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def publish_bytes(path,raw,on_mismatch):
    """Existing objects are byte-checked and directory-synced on every retry.

    A failed directory fsync may leave the correct object visible. Failure still
    propagates; retry rechecks it and repeats directory barriers before success.
    Cooperating local processes and functioning POSIX fsync are prerequisites.
    """
    path=Path(os.path.abspath(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    # Repeat every ancestor barrier on retry: a previous failed attempt may
    # have created several directories before its first successful fsync.
    for directory in reversed(path.parent.parents):sync_directory(directory)
    if path.exists():
        if path.read_bytes()!=raw:on_mismatch()
        sync_directory(path.parent)
        return
    fd,temp=tempfile.mkstemp(prefix='.object-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        try:os.link(temp,path)
        except FileExistsError:
            if path.read_bytes()!=raw:on_mismatch()
    finally:
        Path(temp).unlink(missing_ok=True)
        sync_directory(path.parent)
