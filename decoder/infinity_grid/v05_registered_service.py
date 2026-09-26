from __future__ import annotations

"""Registered service connection for the L2 development increment.

The service selects an installed registration and invokes the existing chain
controller. Requests cannot provide calculations, executable references, keys or
completion records. This increment is engineering-only; official deployment and
worker permission separation remain explicit acceptance prerequisites.
"""

import argparse
import base64
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .v05_execution_authority import (
    ENGINEERING_ROLE, OFFICIAL_ROLE, EngineeringAuthority, ExecutionAuthorityError,
    RuntimePermit, _PERMIT_CONSTRUCTOR, canonical_bytes, digest, science_digest,
    source_tree_digest, validate_bindings,
)

POLICY_SCHEMA = "IG_DECODER_REGISTERED_SERVICE_POLICY_V1"
REQUEST_SCHEMA = "IG_DECODER_REGISTERED_SERVICE_REQUEST_V1"
RESPONSE_SCHEMA = "IG_DECODER_REGISTERED_SERVICE_RESPONSE_V1"
PUBLICATION_SCHEMA = "IG_DECODER_REGISTERED_SERVICE_PUBLICATION_V1"
PUBLICATION_RECORD_SCHEMA = "IG_DECODER_REGISTERED_CHAIN_SNAPSHOT_V1"
PUBLICATION_DOMAIN = b"IG-DECODER-REGISTERED-PUBLICATION-V1\x00"
MAX_REQUEST_BYTES = 4096
MAX_POLICY_BYTES = 1048576
POLICY_FIELDS = {
    "schema_id", "role", "source_sha256", "service_root", "signing_key_path",
    "public_key_hex", "registrations",
}
RECORD_FIELDS = {
    "schema_id", "role", "authoritative", "science_authority_effect",
    "policy_sha256", "source_sha256", "registration_sha256", "chain_id",
    "chain_status", "registration", "commits",
}


def _issue(code: str) -> ExecutionAuthorityError:
    return ExecutionAuthorityError(code)


def strict_json(raw: bytes | str) -> dict[str, Any]:
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise _issue("SERVICE_DUPLICATE_JSON_FIELD")
            obj[key] = value
        return obj
    def constant(_value):
        raise _issue("SERVICE_FINITE_JSON_REQUIRED")
    try:
        obj = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
        if type(obj) is not dict:
            raise _issue("SERVICE_JSON_OBJECT_REQUIRED")
        canonical_bytes(obj)
        return obj
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise _issue("SERVICE_JSON_FORMAT") from exc


def _absolute_path(value: Any) -> Path:
    if type(value) is not str or not value or len(value) > 4096:
        raise _issue("SERVICE_ABSOLUTE_PATH_REQUIRED")
    p = Path(value)
    if not p.is_absolute() or '..' in p.parts:
        raise _issue("SERVICE_ABSOLUTE_PATH_REQUIRED")
    for part in (p, *p.parents):
        if part.is_symlink():
            raise _issue("SERVICE_STANDARD_PATH_REQUIRED")
    return p


def _read_owned_file(path: Path, max_bytes: int) -> bytes:
    p = _absolute_path(str(path))
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(p, flags)
    except OSError as exc:
        raise _issue("SERVICE_FILE_UNAVAILABLE") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            raise _issue("SERVICE_FILE_PERMISSIONS")
        if st.st_size > max_bytes:
            raise _issue("SERVICE_FILE_SIZE")
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise _issue("SERVICE_FILE_SIZE")
        return raw
    finally:
        os.close(fd)


def _owned_directory(path: Path, *, create: bool) -> Path:
    p = _absolute_path(str(path))
    if create and not p.exists():
        p.mkdir(mode=0o700, parents=True)
    try:
        st = p.stat()
    except OSError as exc:
        raise _issue("SERVICE_DIRECTORY_UNAVAILABLE") from exc
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise _issue("SERVICE_DIRECTORY_PERMISSIONS")
    return p


def validate_request(request: Mapping[str, Any]) -> dict[str, str]:
    if type(request) is not dict or set(request) != {'schema_id', 'operation', 'registration_sha256'}:
        raise _issue("SERVICE_REQUEST_FIELDS")
    if request['schema_id'] != REQUEST_SCHEMA:
        raise _issue("SERVICE_REQUEST_SCHEMA")
    if request['operation'] not in ('run', 'resume', 'status', 'publish'):
        raise _issue("SERVICE_OPERATION_NOT_REGISTERED")
    ref = request['registration_sha256']
    if type(ref) is not str or len(ref) != 64 or any(c not in '0123456789abcdef' for c in ref):
        raise _issue("SERVICE_REGISTRATION_IDENTIFIER")
    return dict(request)


class _ServiceEngineeringAuthority(EngineeringAuthority):
    """Stable service-held engineering receipt key; no official signing role."""
    def __init__(self, session: 'RegisteredServiceSession', private_key: bytes) -> None:
        super().__init__(enabled=False)
        self.enabled = True
        self._session = session
        self._key = Ed25519PrivateKey.from_private_bytes(private_key)
        self._public = self._key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        if self._public.hex() != session._policy['public_key_hex']:
            raise _issue("SERVICE_KEY_POLICY_MISMATCH")

    def require_run(self, registration: Mapping[str, Any]) -> None:
        super().require_run(registration)
        self._session.require_registration(registration)

    def _entry_for_bindings(self, bindings: Mapping[str, Any]):
        b = validate_bindings(bindings)
        self._session.verify_live()
        entry = self._session.entry(b['registration_sha256'])
        reg = entry['registration']
        stage = next((s for s in reg['stages'] if s['stage_id'] == b['stage_id']), None)
        if stage is None or stage['stage_kind'] != 'DECODER_STAGE':
            raise _issue("SERVICE_STAGE_NOT_REGISTERED")
        from .v05_stage_registry import ENGINEERING_ONLY_STAGE_HANDLERS
        ref = ENGINEERING_ONLY_STAGE_HANDLERS[stage['execution']['handler_key']]
        mod, _name = ref.split(':')
        module = self._session.package_root.joinpath(*mod.split('.')[1:]).with_suffix('.py')
        expected = {
            'chain_id': reg['chain_id'], 'question_sha256': stage['question_sha256'],
            'handler_key': stage['execution']['handler_key'], 'handler_ref': ref,
            'source_sha256': self._session._policy['source_sha256'],
            'handler_source_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
            'parameters_sha256': digest(dict(stage['execution'].get('parameters') or {})),
            'authority_sha256': digest(reg['parent_authority']),
            'evidence_store_id': digest({'chain_dir': str((self._session.chain_root / reg['chain_id']).resolve(strict=True))}),
        }
        if any(b[k] != v for k, v in expected.items()):
            raise _issue("SERVICE_EXECUTION_BINDING_MISMATCH")
        return b, entry

    def begin(self, bindings: Mapping[str, Any]) -> RuntimePermit:
        b, entry = self._entry_for_bindings(bindings)
        return RuntimePermit(b, _constructor=_PERMIT_CONSTRUCTOR,
                             _evaluator_refs=tuple(entry['evaluator_refs']))

    def receipt(self, bindings: Mapping[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        b, _entry = self._entry_for_bindings(bindings)
        science_digest(result)
        return super().receipt(b, result)


class RegisteredServiceSession:
    """Installed engineering policy, held by the service rather than a request."""
    def __init__(self, policy_path: str | Path) -> None:
        if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'):
            raise _issue("SERVICE_PLATFORM_PREREQUISITE")
        self.policy_path = _absolute_path(str(policy_path))
        raw = _read_owned_file(self.policy_path, MAX_POLICY_BYTES)
        obj = strict_json(raw)
        if set(obj) != POLICY_FIELDS or obj.get('schema_id') != POLICY_SCHEMA:
            raise _issue("SERVICE_POLICY_FIELDS")
        # Official setup is kept separate until the remaining L2 conditions pass.
        # This check precedes private-key reads, store creation and chain work.
        if obj['role'] == OFFICIAL_ROLE:
            raise _issue("OFFICIAL_SERVICE_ACCEPTANCE_PENDING")
        if obj['role'] != ENGINEERING_ROLE:
            raise _issue("SERVICE_ROLE_NOT_REGISTERED")
        self._policy = copy.deepcopy(obj)
        self._policy_bytes_sha256 = hashlib.sha256(raw).hexdigest()
        self.policy_sha256 = digest(obj)
        self.package_root = Path(__file__).resolve().parent
        if obj['source_sha256'] != source_tree_digest(self.package_root):
            raise _issue("SERVICE_SOURCE_POLICY_MISMATCH")
        try:
            key = bytes.fromhex(obj['public_key_hex'])
        except (ValueError, TypeError) as exc:
            raise _issue("SERVICE_PUBLIC_KEY_FORMAT") from exc
        if len(key) != 32 or key.hex() != obj['public_key_hex']:
            raise _issue("SERVICE_PUBLIC_KEY_FORMAT")
        self._public_keys = {hashlib.sha256(key).hexdigest(): key}
        root = _absolute_path(obj['service_root'])
        key_path = _absolute_path(obj['signing_key_path'])
        if key_path == self.policy_path or key_path.is_relative_to(root) or key_path.is_relative_to(self.package_root):
            raise _issue("SERVICE_KEY_LOCATION")
        if root.is_relative_to(self.package_root) or self.package_root.is_relative_to(root):
            raise _issue("SERVICE_SOURCE_STORE_SEPARATION")
        from .v05_chain import validate_chain_registration
        from .v05_stage_registry import ENGINEERING_ONLY_STAGE_HANDLERS, ENGINEERING_ONLY_WORKER_EVALUATORS
        entries = obj['registrations']
        if type(entries) is not list or not entries or len(entries) > 128:
            raise _issue("SERVICE_CATALOG_REQUIRED")
        self._entries: dict[str, dict[str, Any]] = {}
        chain_ids = set()
        for item in entries:
            if type(item) is not dict or set(item) != {'registration', 'evaluator_refs'}:
                raise _issue("SERVICE_CATALOG_FIELDS")
            reg = validate_chain_registration(item['registration'])
            EngineeringAuthority(enabled=True).require_run(reg)
            sha = reg['registration_sha256']
            if sha in self._entries or reg['chain_id'] in chain_ids:
                raise _issue("SERVICE_CATALOG_DUPLICATE")
            refs = item['evaluator_refs']
            if (type(refs) is not list or not refs or any(type(x) is not str for x in refs)
                    or len(refs) != len(set(refs)) or any(x not in ENGINEERING_ONLY_WORKER_EVALUATORS for x in refs)):
                raise _issue("SERVICE_EVALUATOR_INVENTORY")
            for s in reg['stages']:
                if s['stage_kind'] == 'DECODER_STAGE' and s['execution']['handler_key'] not in ENGINEERING_ONLY_STAGE_HANDLERS:
                    raise _issue("SERVICE_HANDLER_INVENTORY")
            self._entries[sha] = copy.deepcopy(item)
            chain_ids.add(reg['chain_id'])
        raw_key = _read_owned_file(key_path, 32)
        if len(raw_key) != 32:
            raise _issue("SERVICE_PRIVATE_KEY_FORMAT")
        self.authority = _ServiceEngineeringAuthority(self, raw_key)
        self.service_root = _owned_directory(root, create=True)
        self.chain_root = _owned_directory(root / 'chains', create=True)

    @property
    def public_keys(self) -> dict[str, bytes]:
        return dict(self._public_keys)

    def verify_live(self) -> None:
        raw = _read_owned_file(self.policy_path, MAX_POLICY_BYTES)
        if hashlib.sha256(raw).hexdigest() != self._policy_bytes_sha256:
            raise _issue("SERVICE_POLICY_CHANGED")
        if source_tree_digest(self.package_root) != self._policy['source_sha256']:
            raise _issue("SERVICE_SOURCE_POLICY_MISMATCH")
        if hasattr(self, 'service_root'):
            _owned_directory(self.service_root, create=False)
            _owned_directory(self.chain_root, create=False)

    def entry(self, registration_sha256: str) -> dict[str, Any]:
        item = self._entries.get(registration_sha256)
        if item is None:
            raise _issue("SERVICE_REGISTRATION_NOT_ADMITTED")
        return copy.deepcopy(item)

    def require_registration(self, reg: Mapping[str, Any]) -> None:
        self.verify_live()
        item = self.entry(str(reg.get('registration_sha256', '')))
        if canonical_bytes(dict(reg)) != canonical_bytes(item['registration']):
            raise _issue("SERVICE_REGISTRATION_CONTENT_MISMATCH")

    def require_handler(self, key: str, ref: str) -> None:
        from .v05_stage_registry import ENGINEERING_ONLY_STAGE_HANDLERS
        admitted = {s['execution']['handler_key'] for e in self._entries.values()
                    for s in e['registration']['stages'] if s['stage_kind'] == 'DECODER_STAGE'}
        if key not in admitted or ENGINEERING_ONLY_STAGE_HANDLERS.get(key) != ref:
            raise _issue("SERVICE_HANDLER_INVENTORY")

    def publication_receipt(self, record: dict[str, Any]) -> dict[str, Any]:
        self.verify_live()
        if (set(record) != RECORD_FIELDS or record['policy_sha256'] != self.policy_sha256
                or record['role'] != ENGINEERING_ROLE or record['authoritative'] is not False
                or record['science_authority_effect'] != 'NONE'):
            raise _issue("SERVICE_PUBLICATION_RECORD")
        self.require_registration(record['registration'])
        body = {'schema_id': PUBLICATION_SCHEMA, 'algorithm': 'Ed25519',
                'role': ENGINEERING_ROLE, 'key_id': next(iter(self.public_keys)),
                'record': copy.deepcopy(record), 'record_sha256': digest(record)}
        return dict(body, signature=base64.b64encode(
            self.authority._key.sign(PUBLICATION_DOMAIN + canonical_bytes(body))).decode('ascii'))

    @contextmanager
    def request_slot(self):
        self.verify_live()
        path = self.service_root / 'request.lock'
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
                raise _issue("SERVICE_LOCK_PERMISSIONS")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise _issue("SERVICE_BUSY") from exc
            yield
        finally:
            os.close(fd)


def verify_service_publication(publication: Mapping[str, Any], *,
                               expected_policy_sha256: str, expected_source_sha256: str,
                               expected_registration_sha256: str,
                               trusted_public_keys: Mapping[str, bytes]) -> dict[str, Any]:
    """Verify publication origin against an independently supplied policy/key."""
    fields = {'schema_id', 'algorithm', 'role', 'key_id', 'record', 'record_sha256', 'signature'}
    if type(publication) is not dict or set(publication) != fields:
        raise _issue("SERVICE_PUBLICATION_FIELDS")
    p = dict(publication)
    if p['schema_id'] != PUBLICATION_SCHEMA or p['algorithm'] != 'Ed25519' or p['role'] != ENGINEERING_ROLE:
        raise _issue("SERVICE_PUBLICATION_ROLE")
    key_id = p['key_id']
    if type(key_id) is not str or len(key_id) != 64 or any(c not in '0123456789abcdef' for c in key_id):
        raise _issue("SERVICE_PUBLICATION_KEY")
    key = trusted_public_keys.get(key_id)
    if not isinstance(key, bytes) or len(key) != 32 or hashlib.sha256(key).hexdigest() != p['key_id']:
        raise _issue("SERVICE_PUBLICATION_KEY")
    record = p['record']
    if (type(record) is not dict or set(record) != RECORD_FIELDS
            or record['schema_id'] != PUBLICATION_RECORD_SCHEMA
            or record['policy_sha256'] != expected_policy_sha256
            or record['source_sha256'] != expected_source_sha256
            or record['registration_sha256'] != expected_registration_sha256
            or record['role'] != ENGINEERING_ROLE or record['authoritative'] is not False
            or record['science_authority_effect'] != 'NONE'
            or type(record['chain_status']) is not dict
            or record['chain_status'].get('status') != 'COMPLETE'
            or record['chain_status'].get('external_mirror_pending') is not False
            or digest(record) != p['record_sha256']):
        raise _issue("SERVICE_PUBLICATION_BINDING")
    try:
        sig = base64.b64decode(p['signature'], validate=True)
        if len(sig) != 64:
            raise ValueError('signature size')
        Ed25519PublicKey.from_public_bytes(key).verify(sig, PUBLICATION_DOMAIN + canonical_bytes(
            {k:v for k,v in p.items() if k != 'signature'}))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise _issue("SERVICE_PUBLICATION_SIGNATURE") from exc
    return {'status': 'VERIFIED', 'role': ENGINEERING_ROLE, 'authoritative': False,
            'record_sha256': p['record_sha256']}


class RegisteredExecutionService:
    """Configuration reader only. External/direct execution dispatch is closed."""
    def __init__(self, policy_path: str | Path) -> None:
        self.session = RegisteredServiceSession(policy_path)
    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        from .v05_route_closure import REJECT_DIRECT_EXECUTION_ROUTE
        raise ExecutionAuthorityError(REJECT_DIRECT_EXECUTION_ROUTE,'registered-service-dispatch')

def dispatch_from_controller_event(session: RegisteredServiceSession, request: dict[str,Any]) -> dict[str,Any]:
    """Internal controller-root service dispatch; no public ingress."""
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin('registered-service-internal-dispatch')
    req=validate_request(request)
    with session.request_slot():
        entry=session.entry(req['registration_sha256']); reg=entry['registration']
        from .v05_chain import ScientificChainController
        from .v05_stage_registry import ENGINEERING_ONLY_STAGE_HANDLERS, _resolve
        controller=ScientificChainController(session.chain_root,engineering_only=True,_service_session=session)
        for st in reg['stages']:
            if st['stage_kind']=='DECODER_STAGE':
                key=st['execution']['handler_key']; controller.register_stage_handler(key,_resolve(ENGINEERING_ONLY_STAGE_HANDLERS[key]))
        op=req['operation']
        if op!='run' and not (session.chain_root/reg['chain_id']/'chain_registration.json').is_file(): raise _issue('SERVICE_CHAIN_NOT_STARTED')
        if op=='run': controller.run(reg); result=controller.status(reg['chain_id'])
        elif op=='resume': controller.resume(reg['chain_id']); result=controller.status(reg['chain_id'])
        elif op=='status': result=controller.status(reg['chain_id'])
        else:
            from .v05_publication import publish_registered_chain_execution
            result=publish_registered_chain_execution(controller,reg['chain_id'])
        session.verify_live(); return {'schema_id':RESPONSE_SCHEMA,'status':'OK','role':ENGINEERING_ROLE,'authoritative':False,'operation':op,'policy_sha256':session.policy_sha256,'registration_sha256':reg['registration_sha256'],'result':result}

def main() -> int:
    from .v05_route_closure import REJECT_DIRECT_EXECUTION_ROUTE
    print(json.dumps({'schema_id':RESPONSE_SCHEMA,'status':'PAUSED','reason':REJECT_DIRECT_EXECUTION_ROUTE,'authoritative':False},sort_keys=True))
    return 4


if __name__ == '__main__':
    # Use the canonical module for class identity during module-style invocation.
    from infinity_grid.v05_registered_service import main as service_main
    raise SystemExit(service_main())
