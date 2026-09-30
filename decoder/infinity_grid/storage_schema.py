"""Explicit, pinned Draft 2020-12 admission for storage sidecars.

Legacy payloads/validators are not rewritten. This checks envelope structure and
local scalar invariants, NOT dependency closure or scientific acceptance.
"""
from __future__ import annotations
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_ID = 'urn:ig:storage:contract:1.0.0'
SCHEMA_SHA256 = '04a4584a18eec033126405136177c81b45bb02afa9621f6e1060d69f3a30794c'
DEFAULT_MAX_BYTES = 2 * 1024 * 1024

class StorageSchemaError(ValueError):
    pass

def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StorageSchemaError('DUPLICATE_JSON_KEY')
        result[key] = value
    return result

def strict_loads(raw: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES,
                 max_depth: int = 64, max_nodes: int = 100000) -> Any:
    if type(raw) is not bytes or type(max_bytes) is not int or max_bytes < 1:
        raise StorageSchemaError('INVALID_JSON_INPUT_OR_BUDGET')
    if len(raw) > max_bytes:
        raise StorageSchemaError('RECORD_BUDGET_EXCEEDED')
    if raw.startswith(b'\xef\xbb\xbf'):
        raise StorageSchemaError('JSON_BOM_FORBIDDEN')
    try:
        text = raw.decode('utf-8', errors='strict')
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(
                               StorageSchemaError('NONFINITE_JSON')))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise StorageSchemaError('INVALID_JSON:' + str(exc)[:120]) from exc
    stack = [(value, 0)]; seen = 0
    while stack:
        node, depth = stack.pop(); seen += 1
        if depth > max_depth or seen > max_nodes:
            raise StorageSchemaError('JSON_STRUCTURE_BUDGET_EXCEEDED')
        if isinstance(node, str):
            try: node.encode('utf-8', errors='strict')
            except UnicodeError as exc: raise StorageSchemaError('LONE_SURROGATE') from exc
        elif type(node) is float and not math.isfinite(node):
            raise StorageSchemaError('NONFINITE_JSON')
        elif isinstance(node, dict):
            stack.extend((v, depth + 1) for pair in node.items() for v in pair)
        elif isinstance(node, list):
            stack.extend((v, depth + 1) for v in node)
    return value

def canonical_bytes(value: Any) -> bytes:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise StorageSchemaError('CANONICAL_JSON_INVALID') from exc
    return raw

def _deny_resource(uri):
    from referencing.exceptions import NoSuchResource
    raise NoSuchResource(ref=uri)

@lru_cache(maxsize=1)
def _compiled():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    path = Path(__file__).parent / 'resources/storage/IG_STORAGE_CONTRACT_V1.schema.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCHEMA_SHA256:
        raise StorageSchemaError('PINNED_SCHEMA_MISMATCH')
    schema = strict_loads(raw)
    if schema.get('$id') != SCHEMA_ID:
        raise StorageSchemaError('UNREGISTERED_SCHEMA')
    # This version admits no external/dynamic schema references at all.
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ('$ref', '$dynamicRef') and not (isinstance(v, str) and v.startswith('#/')):
                    raise StorageSchemaError('UNPINNED_SCHEMA_RESOURCE')
                stack.append(v)
        elif isinstance(node, list): stack.extend(node)
    Draft202012Validator.check_schema(schema)
    registry = Registry(retrieve=_deny_resource).with_resource(SCHEMA_ID, Resource.from_contents(schema))
    return Draft202012Validator(schema, registry=registry)

def validate_record_bytes(raw: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES,
                          require_canonical: bool = True) -> dict:
    record = strict_loads(raw, max_bytes=max_bytes)
    if not isinstance(record, dict): raise StorageSchemaError('STORAGE_RECORD_NOT_OBJECT')
    try:
        error = next(_compiled().iter_errors(record), None)
        if error is not None:
            raise StorageSchemaError('STORAGE_SCHEMA_REJECTED:' + error.message[:300])
    except StorageSchemaError: raise
    except Exception as exc:
        raise StorageSchemaError('STORAGE_SCHEMA_RESOLUTION_FAILED') from exc
    if require_canonical and canonical_bytes(record) != raw:
        raise StorageSchemaError('NONCANONICAL_STORAGE_RECORD')
    # Scalar invariants are not automatically supplied by JSON Schema.
    if record['schema_id'] == 'IG_STORAGE_COVERAGE_V1':
        counts = record.get('counts', record)
        # Field names are deliberately discovered from the actual pinned schema.
        names = ('distinct_tested_count', 'passed_count', 'failed_count', 'raw_attempt_count')
        if all(k in counts for k in names):
            tested, passed, failed, attempts = (int(counts[k]) for k in names)
            if tested != passed + failed or attempts < tested:
                raise StorageSchemaError('COVERAGE_ACCOUNTING_MISMATCH')
            expected = counts.get('expected_count'); skipped = counts.get('skipped_count'); pending = counts.get('not_run_count')
            if expected is not None and pending is not None and skipped is not None:
                if int(expected) != tested + int(skipped) + int(pending):
                    raise StorageSchemaError('COVERAGE_EXPECTED_COUNT_MISMATCH')
    for extension in record.get('extensions', []):
        if extension.get('required_to_interpret', False):
            raise StorageSchemaError('UNSUPPORTED_REQUIRED_EXTENSION')
    return record
