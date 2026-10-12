from __future__ import annotations

"""Versioned exact canonical-byte encoder for Decoder structural signatures.

The existing Decoder canonical JSON byte language remains authoritative.  This
module validates the same canonical value domain without allocating a normalized
copy, then lets CPython's C JSON encoder serialize tuples/lists directly.  The
result is byte-for-byte IG_CANONICAL_JSON_V1.
"""

import hashlib
import json
import math
from typing import Any

from .core.canonical import CanonicalEncodingError

STRUCTURAL_ENCODER_ID = "IG_STRUCTURAL_CANONICAL_BYTES_FAST_V1"
STRUCTURAL_ENCODER_VERSION = "1.1.0"
STRUCTURAL_BYTE_LANGUAGE = "IG_CANONICAL_JSON_V1"


def _validate_canonical_domain(value: Any) -> None:
    """Validate canonical JSON domain without copying the nested structure."""
    stack = [value]
    while stack:
        x = stack.pop()
        if x is None or type(x) in (str, bool, int):
            continue
        if type(x) is float:
            if not math.isfinite(x):
                raise CanonicalEncodingError("$: non-finite float is not canonical JSON")
            continue
        if isinstance(x, (tuple, list)):
            stack.extend(x)
            continue
        if type(x) is dict:
            for key, item in x.items():
                if type(key) is not str:
                    raise CanonicalEncodingError(f"$: object key must be string, got {type(key).__name__}")
                stack.append(item)
            continue
        raise CanonicalEncodingError(f"$: unsupported canonical JSON type {type(x).__name__}")


def structural_canonical_bytes(value: Any) -> bytes:
    _validate_canonical_domain(value)
    # json.dumps serializes tuple and list identically.  With these exact options
    # the bytes are identical to Decoder canonical_bytes after normalize_json.
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def structural_canonical_sha256(value: Any) -> str:
    return hashlib.sha256(structural_canonical_bytes(value)).hexdigest()
