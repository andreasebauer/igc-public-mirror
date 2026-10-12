from __future__ import annotations
import json
from importlib.resources import files
from .canon import canonical_sha256
from .errors import ArtifactCorruptError, ContractError
from .schema import validate

REGISTRY_SCHEMA_ID = "IG_V027_NATIVE_CONTRACT_REGISTRY_V1"

def _r(rel: str) -> dict:
    return json.loads(files("infinity_grid").joinpath("resources/v027").joinpath(rel).read_text(encoding="utf-8"))

def v027_contract_registry() -> dict:
    reg = _r("V027_NATIVE_CONTRACT_REGISTRY_V1.json")
    if reg.get("schema_id") != REGISTRY_SCHEMA_ID:
        raise ContractError("unexpected v0.27 contract registry schema")
    body = {k: v for k, v in reg.items() if k != "registry_sha256"}
    # Registry was hashed including everything except its self hash.
    expected = canonical_sha256(body)
    if expected != reg.get("registry_sha256"):
        raise ArtifactCorruptError("v0.27 contract registry hash mismatch")
    return reg

def v027_payload_schema(schema_id: str) -> dict:
    ent = v027_contract_registry().get("payload_schemas", {}).get(schema_id)
    if not ent:
        raise KeyError(schema_id)
    sch = _r(ent["schema_path"])
    if canonical_sha256(sch) != ent["schema_sha256"]:
        raise ArtifactCorruptError(f"v0.27 schema hash mismatch: {schema_id}")
    return sch

def validate_v027_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise ContractError("v0.27 payload must be object")
    sid = payload.get("schema_id") or payload.get("schema")
    if not isinstance(sid, str):
        raise ContractError("v0.27 payload lacks schema discriminator")
    try:
        sch = v027_payload_schema(sid)
    except KeyError as exc:
        raise ContractError(f"unregistered v0.27 payload schema: {sid}") from exc
    errs = validate(sch, payload, raise_on_error=False)
    if errs:
        raise ContractError(f"v0.27 payload {sid} failed strict contract: {'; '.join(errs[:20])}")
    return sid
