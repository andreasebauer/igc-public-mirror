from __future__ import annotations

"""Passive data-only request intake for origin-exclusivity migration C3.

External submission writes validated declarative data only.  It does not create
an execution context and does not call controller/service/runtime/worker paths.
Only a live C2 controller-root context may ingest a pending request into an
internal execution-intent record.  Ingestion itself still does not execute it.
"""

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Mapping

from .v05_origin_guard import (
    CONTROLLER_ROOT_ROLE,
    current_execution_context_snapshot,
    require_controller_execution_origin,
)

REQUEST_SCHEMA = 'IG_DECODER_PASSIVE_REQUEST_V1'
RECEIPT_SCHEMA = 'IG_DECODER_PASSIVE_REQUEST_RECEIPT_V1'
INGESTED_SCHEMA = 'IG_DECODER_INTERNAL_INGESTED_REQUEST_V1'

ALLOWED_REQUEST_FIELDS = frozenset({
    'schema_id', 'request_id', 'registered_job_id', 'requested_operation_id',
    'parent_source_sha256', 'input_artifacts', 'human_note',
})
ALLOWED_ARTIFACT_FIELDS = frozenset({'logical_name', 'sha256'})

_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
_LOGICAL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,191}$')
_SHA_RE = re.compile(r'^[0-9a-f]{64}$')


class PassiveRequestError(ValueError):
    pass


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def _fail(code: str, detail: str | None = None) -> PassiveRequestError:
    return PassiveRequestError(code if detail is None else f'{code}:{detail}')


def _validate_id(value: Any, field: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise _fail('PASSIVE_REQUEST_IDENTIFIER', field)
    return value


def _validate_sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise _fail('PASSIVE_REQUEST_SHA256', field)
    return value


def validate_passive_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if type(request) is not dict:
        raise _fail('PASSIVE_REQUEST_TYPE')
    unknown = set(request) - ALLOWED_REQUEST_FIELDS
    if unknown:
        raise _fail('PASSIVE_REQUEST_FIELD_FORBIDDEN', sorted(unknown)[0])
    required = {
        'schema_id', 'request_id', 'registered_job_id', 'requested_operation_id',
        'parent_source_sha256', 'input_artifacts',
    }
    missing = required - set(request)
    if missing:
        raise _fail('PASSIVE_REQUEST_FIELD_MISSING', sorted(missing)[0])
    if request['schema_id'] != REQUEST_SCHEMA:
        raise _fail('PASSIVE_REQUEST_SCHEMA')
    request_id = _validate_id(request['request_id'], 'request_id')
    registered_job_id = _validate_id(request['registered_job_id'], 'registered_job_id')
    requested_operation_id = _validate_id(request['requested_operation_id'], 'requested_operation_id')
    parent_source_sha256 = _validate_sha(request['parent_source_sha256'], 'parent_source_sha256')
    artifacts = request['input_artifacts']
    if type(artifacts) is not list or len(artifacts) > 128:
        raise _fail('PASSIVE_REQUEST_ARTIFACTS')
    normalized_artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, artifact in enumerate(artifacts):
        if type(artifact) is not dict:
            raise _fail('PASSIVE_REQUEST_ARTIFACT_TYPE', str(idx))
        unknown_art = set(artifact) - ALLOWED_ARTIFACT_FIELDS
        if unknown_art:
            raise _fail('PASSIVE_REQUEST_ARTIFACT_FIELD_FORBIDDEN', sorted(unknown_art)[0])
        if set(artifact) != ALLOWED_ARTIFACT_FIELDS:
            raise _fail('PASSIVE_REQUEST_ARTIFACT_FIELDS', str(idx))
        logical = artifact['logical_name']
        if type(logical) is not str or _LOGICAL_RE.fullmatch(logical) is None or '..' in logical.split('/'):
            raise _fail('PASSIVE_REQUEST_ARTIFACT_LOGICAL_NAME', str(idx))
        if logical in seen:
            raise _fail('PASSIVE_REQUEST_ARTIFACT_DUPLICATE', logical)
        seen.add(logical)
        normalized_artifacts.append({'logical_name': logical, 'sha256': _validate_sha(artifact['sha256'], f'artifact[{idx}]')})
    note = request.get('human_note')
    if note is not None and (type(note) is not str or len(note) > 2048):
        raise _fail('PASSIVE_REQUEST_HUMAN_NOTE')
    normalized = {
        'schema_id': REQUEST_SCHEMA,
        'request_id': request_id,
        'registered_job_id': registered_job_id,
        'requested_operation_id': requested_operation_id,
        'parent_source_sha256': parent_source_sha256,
        'input_artifacts': normalized_artifacts,
    }
    if note is not None:
        normalized['human_note'] = note
    return normalized


def _atomic_create_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        raise


def submit_passive_request(intake_root: str | Path, request: Mapping[str, Any]) -> dict[str, str]:
    """Validate and persist passive request data; never starts Decoder execution."""
    normalized = validate_passive_request(request)
    digest = _sha(normalized)
    root = Path(intake_root).resolve()
    pending = root / 'pending'
    path = pending / f"{normalized['request_id']}.json"
    try:
        _atomic_create_json(path, normalized)
    except FileExistsError as exc:
        raise _fail('PASSIVE_REQUEST_DUPLICATE', normalized['request_id']) from exc
    return {
        'schema_id': RECEIPT_SCHEMA,
        'request_id': normalized['request_id'],
        'request_sha256': digest,
        'state': 'PENDING_PASSIVE',
    }


def read_passive_request(intake_root: str | Path, request_id: str) -> dict[str, Any]:
    rid = _validate_id(request_id, 'request_id')
    path = Path(intake_root).resolve() / 'pending' / f'{rid}.json'
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise _fail('PASSIVE_REQUEST_NOT_FOUND', rid) from exc
    return validate_passive_request(obj)


def ingest_passive_request(
    intake_root: str | Path,
    request_id: str,
    *,
    accepted_registry: Mapping[str, Mapping[str, Any]],
    expected_parent_source_sha256: str,
    internal_root: str | Path,
) -> dict[str, Any]:
    """Controller-only conversion of passive data to an internal intent record.

    This function does not invoke the selected registered operation.
    """
    require_controller_execution_origin('passive-intake-ingest')
    ctx = current_execution_context_snapshot()
    if ctx is None or ctx.get('role') != CONTROLLER_ROOT_ROLE:
        raise _fail('PASSIVE_INGEST_CONTROLLER_CONTEXT')
    request = read_passive_request(intake_root, request_id)
    expected_parent = _validate_sha(expected_parent_source_sha256, 'expected_parent_source_sha256')
    if request['parent_source_sha256'] != expected_parent:
        raise _fail('PASSIVE_INGEST_PARENT_MISMATCH')
    job = accepted_registry.get(request['registered_job_id'])
    if type(job) is not dict:
        raise _fail('PASSIVE_INGEST_JOB_NOT_REGISTERED', request['registered_job_id'])
    allowed_ops = job.get('allowed_operations')
    if type(allowed_ops) not in (list, tuple) or request['requested_operation_id'] not in allowed_ops:
        raise _fail('PASSIVE_INGEST_OPERATION_NOT_REGISTERED', request['requested_operation_id'])
    registration_sha = _validate_sha(job.get('registration_sha256'), 'registry.registration_sha256')
    implementation_sha = _validate_sha(job.get('implementation_sha256'), 'registry.implementation_sha256')
    request_sha = _sha(request)
    claim_binding = request_claim_binding(request, registration_sha, implementation_sha, expected_parent)
    internal_execution_id = 'intent-' + _sha(claim_binding)[:32]
    claim = dict(claim_binding, internal_execution_id=internal_execution_id)
    iroot = Path(internal_root).resolve()
    claim_path = iroot / 'claims' / f"{request['request_id']}.json"
    try:
        _atomic_create_json(claim_path, claim)
    except FileExistsError:
        try:
            existing_claim = json.loads(claim_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise _fail('PASSIVE_INGEST_CLAIM_UNREADABLE', request['request_id']) from exc
        if existing_claim != claim:
            raise _fail('PASSIVE_INGEST_CLAIM_MISMATCH', request['request_id'])
    target = iroot / 'ingested' / f'{internal_execution_id}.json'
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding='utf-8'))
        except Exception as exc:
            raise _fail('PASSIVE_INGEST_RECORD_UNREADABLE', internal_execution_id) from exc
        required_existing = {
            'schema_id': INGESTED_SCHEMA,
            'internal_execution_id': internal_execution_id,
            'external_request_id': request['request_id'],
            'request_sha256': request_sha,
            'registered_job_id': request['registered_job_id'],
            'requested_operation_id': request['requested_operation_id'],
            'accepted_parent_source_sha256': expected_parent,
            'registration_sha256': registration_sha,
            'implementation_sha256': implementation_sha,
            'input_artifacts': request['input_artifacts'],
            'state': 'INGESTED_NOT_EXECUTED',
        }
        for key, value in required_existing.items():
            if existing.get(key) != value:
                raise _fail('PASSIVE_INGEST_RECORD_MISMATCH', key)
        return existing
    record = {
        'schema_id': INGESTED_SCHEMA,
        'internal_execution_id': internal_execution_id,
        'external_request_id': request['request_id'],
        'request_sha256': request_sha,
        'registered_job_id': request['registered_job_id'],
        'requested_operation_id': request['requested_operation_id'],
        'accepted_parent_source_sha256': expected_parent,
        'registration_sha256': registration_sha,
        'implementation_sha256': implementation_sha,
        'input_artifacts': request['input_artifacts'],
        'controller_session_id': ctx['controller_session_id'],
        'controller_root_execution_id': ctx['root_execution_id'],
        'controller_context_id': ctx['context_id'],
        'state': 'INGESTED_NOT_EXECUTED',
    }
    _atomic_create_json(target, record)
    return record


def request_claim_binding(request, registration_sha, implementation_sha, expected_parent):
    """Read-only identity calculation; no context or execution authority."""
    return {
        'schema_id': 'IG_DECODER_PASSIVE_REQUEST_CLAIM_V1',
        'external_request_id': request['request_id'],
        'request_sha256': _sha(request),
        'accepted_parent_source_sha256': expected_parent,
        'registration_sha256': registration_sha,
        'implementation_sha256': implementation_sha,
    }
