from __future__ import annotations

import hashlib
import json
import math
from typing import Any

CANONICALIZER_ID = "IG_CANONICAL_JSON_V1"
CANONICALIZER_VERSION = "1.0.0"


class CanonicalEncodingError(ValueError):
    pass


def normalize_json(value: Any, path: str = "$") -> Any:
    """Pure normalization for the Decoder canonical JSON value domain.

    No filesystem, clock, process, random source, environment or mutable global
    state is consulted.  The bytes intentionally remain identical to v0.26.x.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalEncodingError(f"{path}: non-finite float is not canonical JSON")
        return value
    if isinstance(value, tuple):
        return [normalize_json(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, list):
        return [normalize_json(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalEncodingError(f"{path}: object key must be string, got {type(key).__name__}")
            out[key] = normalize_json(item, f"{path}.{key}")
        return out
    raise CanonicalEncodingError(f"{path}: unsupported canonical JSON type {type(value).__name__}")


def canonical_bytes(obj: Any) -> bytes:
    obj = normalize_json(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_text(obj: Any, *, pretty: bool = False) -> str:
    obj = normalize_json(obj)
    if pretty:
        return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return canonical_bytes(obj).decode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()
