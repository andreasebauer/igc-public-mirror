import hashlib
import json
from pathlib import Path

def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

from datetime import date, datetime
from decimal import Decimal

def _json_default(o):
    """Fallback serializer for hash_json — handles datetime, Path, Decimal, etc."""
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    # final fallback: string form
    return str(o)

def hash_json(obj, *, sort_keys: bool = True, separators=(",", ":"), encoding: str = "utf-8") -> str:
    """
    Stable SHA-256 over a canonical JSON representation of `obj`.
    - sort_keys=True for deterministic key order
    - separators without spaces to avoid whitespace variance
    - ensure_ascii=False so unicode stays stable; encoded to `encoding`
    """
    s = json.dumps(
        obj,
        sort_keys=sort_keys,
        separators=separators,
        ensure_ascii=False,
        default=_json_default,     # ← key addition
    )
    return hashlib.sha256(s.encode(encoding)).hexdigest()

def sha256_bytes(data: bytes) -> str:
    """Convenience helper: SHA-256 for raw bytes (used occasionally in tools)."""
    return hashlib.sha256(data).hexdigest()