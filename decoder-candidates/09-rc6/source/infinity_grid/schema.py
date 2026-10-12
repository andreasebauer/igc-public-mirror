from __future__ import annotations

import re
import json
import math
from typing import Any

from .errors import SchemaValidationError


class ValidationError(SchemaValidationError):
    pass


def _typename(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, str):
        return "string"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, (list, tuple)):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def validate(schema: dict, obj: Any, path: str = "$", *, raise_on_error: bool = True):
    """Validate the strict JSON-Schema subset used by Decoder contracts.

    Supported keywords are deliberately explicit: type, const, enum, required,
    properties, additionalProperties, items, min/maxItems, min/maxProperties,
    min/maxLength, pattern, minimum/maximum, allOf/anyOf/oneOf/not. Unknown
    schema annotations are ignored; data is never silently repaired.
    """
    if schema.get("$id") == "urn:ig:storage:contract:1.0.0":
        from .storage_schema import _compiled, canonical_bytes, validate_record_bytes
        try:
            if schema != _compiled().schema:
                raise ValueError("PINNED_STORAGE_SCHEMA_MISMATCH")
            validate_record_bytes(canonical_bytes(obj))
            return []
        except Exception as exc:
            if raise_on_error:
                raise ValidationError(str(exc)) from exc
            return [str(exc)]
    errors: list[str] = []

    def branch_errors(s: dict, v: Any, p: str) -> list[str]:
        before = len(errors)
        walk(s, v, p)
        out = errors[before:]
        del errors[before:]
        return out

    def walk(s: dict, v: Any, p: str):
        if not isinstance(s, dict):
            errors.append(f"{p}: invalid schema node")
            return

        if "allOf" in s:
            for ss in s["allOf"]:
                walk(ss, v, p)
        if "anyOf" in s:
            branches = [branch_errors(ss, v, p) for ss in s["anyOf"]]
            if all(b for b in branches):
                errors.append(f"{p}: no anyOf branch matched")
                return
        if "oneOf" in s:
            branches = [branch_errors(ss, v, p) for ss in s["oneOf"]]
            matched = sum(1 for b in branches if not b)
            if matched != 1:
                errors.append(f"{p}: expected exactly one oneOf branch, matched {matched}")
                return
        if "not" in s and not branch_errors(s["not"], v, p):
            errors.append(f"{p}: matched forbidden schema")
            return

        typ = s.get("type")
        if typ:
            allowed = typ if isinstance(typ, list) else [typ]
            actual = _typename(v)
            if actual not in allowed and not (actual == "integer" and "number" in allowed):
                errors.append(f"{p}: type {actual} not in {allowed}")
                return

        def _json_equal(a, b):
            # Python bool is an int subclass; JSON Schema keeps boolean and
            # numeric instances distinct for const/enum matching.
            if isinstance(a, bool) or isinstance(b, bool):
                return isinstance(a, bool) and isinstance(b, bool) and a == b
            return a == b
        if "const" in s and not _json_equal(v, s["const"]):
            errors.append(f"{p}: expected const {s['const']!r}")
        if "enum" in s and not any(_json_equal(v, x) for x in s["enum"]):
            errors.append(f"{p}: value not in enum")

        if isinstance(v, dict):
            if "minProperties" in s and len(v) < s["minProperties"]:
                errors.append(f"{p}: too few properties")
            if "maxProperties" in s and len(v) > s["maxProperties"]:
                errors.append(f"{p}: too many properties")
            for key in s.get("required", []):
                if key not in v:
                    errors.append(f"{p}: missing required {key}")
            props = s.get("properties", {})
            for k, vv in v.items():
                if k in props:
                    walk(props[k], vv, p + "." + k)
                else:
                    additional = s.get("additionalProperties", True)
                    if additional is False:
                        errors.append(f"{p}: extra property {k!r}")
                    elif isinstance(additional, dict):
                        walk(additional, vv, p + "." + k)

        if isinstance(v, (list, tuple)):
            if "minItems" in s and len(v) < s["minItems"]:
                errors.append(f"{p}: too few items")
            if "maxItems" in s and len(v) > s["maxItems"]:
                errors.append(f"{p}: too many items")
            if s.get("uniqueItems") is True:
                seen=[]
                for i, vv in enumerate(v):
                    if any(vv == prior and not (isinstance(vv,bool) ^ isinstance(prior,bool)) for prior in seen):
                        errors.append(f"{p}: duplicate item at index {i}")
                        break
                    seen.append(vv)
            prefix = s.get("prefixItems", [])
            if isinstance(prefix, list):
                for i, ss in enumerate(prefix[:len(v)]):
                    walk(ss, v[i], f"{p}[{i}]")
            items = s.get("items")
            start = len(prefix) if isinstance(prefix, list) else 0
            if items is False and len(v) > start:
                errors.append(f"{p}: additional array items forbidden")
            elif isinstance(items, dict):
                for i, vv in enumerate(v[start:], start=start):
                    walk(items, vv, f"{p}[{i}]")

        if isinstance(v, str):
            if "minLength" in s and len(v) < s["minLength"]:
                errors.append(f"{p}: too short")
            if "maxLength" in s and len(v) > s["maxLength"]:
                errors.append(f"{p}: too long")
            if "pattern" in s and not re.search(s["pattern"], v):
                errors.append(f"{p}: pattern mismatch")

        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, float) and not math.isfinite(v):
                errors.append(f"{p}: non-finite number is not valid JSON")
                return
            if "minimum" in s and v < s["minimum"]:
                errors.append(f"{p}: below minimum")
            if "maximum" in s and v > s["maximum"]:
                errors.append(f"{p}: above maximum")
            if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]:
                errors.append(f"{p}: not above exclusiveMinimum")
            if "exclusiveMaximum" in s and v >= s["exclusiveMaximum"]:
                errors.append(f"{p}: not below exclusiveMaximum")

    walk(schema, obj, path)
    if errors and raise_on_error:
        raise ValidationError("; ".join(errors))
    return errors
