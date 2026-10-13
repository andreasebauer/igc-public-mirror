from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

from .core.canonical import (
    CANONICALIZER_ID,
    CANONICALIZER_VERSION,
    CanonicalEncodingError,
    canonical_bytes,
    canonical_text,
    canonical_sha256,
    normalize_json as _normalize_json,
)


def _fsync_dir(path: Path) -> None:
    """Durably fsync a directory or fail closed.

    Scientific checkpoint publication must never claim durability when the
    underlying filesystem rejects fsync.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, obj: Any, *, pretty: bool = True) -> None:
    """Atomically replace *path* using a unique same-directory temporary file.

    The previous fixed ``<target>.tmp`` name was unsafe under duplicate-controller
    races. A unique file in the target directory preserves same-filesystem rename
    atomicity while preventing writers from clobbering each other's scratch files.
    """
    import os
    import uuid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = canonical_text(obj, pretty=pretty).encode("utf-8")
    try:
        with tmp.open("xb") as h:
            h.write(data)
            h.flush()
            os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)
