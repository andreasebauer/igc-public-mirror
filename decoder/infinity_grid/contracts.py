from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

from . import __version__, build_meta
from .canon import (
    CANONICALIZER_ID,
    CANONICALIZER_VERSION,
    canonical_sha256,
    write_json_atomic,
)
from .errors import (
    ArtifactCorruptError,
    ContractError,
    MigrationError,
    UnknownSchemaError,
    UnsupportedArtifactVersion,
)
from .schema import validate

ENVELOPE_SCHEMA_ID = "IG_ARTIFACT_ENVELOPE_V1"
ENVELOPE_VERSION = "1.0.0"
FAILURE_SCHEMA_ID = "IG_DECODER_FAILURE_RECORD_V1"
REGISTRY_SCHEMA_ID = "IG_GATE1_PAYLOAD_CONTRACT_REGISTRY_V1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _resource_json(rel: str) -> dict:
    return json.loads(files("infinity_grid").joinpath("resources").joinpath(rel).read_text(encoding="utf-8"))


def envelope_schema() -> dict:
    return _resource_json("contracts/ARTIFACT_ENVELOPE_V1.schema.json")


def legacy_unschematized_migration_registry() -> dict:
    reg = _resource_json("contracts/LEGACY_UNSCHEMATIZED_MIGRATIONS_V1.json")
    if reg.get("schema_id") != "IG_LEGACY_UNSCHEMATIZED_MIGRATION_REGISTRY_V1":
        raise ContractError("legacy unschematized migration registry has unexpected schema")
    entries = reg.get("entries")
    if not isinstance(entries, dict):
        raise ArtifactCorruptError("legacy unschematized migration registry entries must be an object")
    for digest, entry in entries.items():
        if not isinstance(digest, str) or not _SHA_RE.match(digest):
            raise ArtifactCorruptError("legacy unschematized migration registry has invalid digest key")
        if not isinstance(entry, dict) or set(entry) != {"target_schema_id", "historical_path"}:
            raise ArtifactCorruptError(f"invalid legacy migration entry for {digest}")
        if not isinstance(entry["target_schema_id"], str) or not entry["target_schema_id"]:
            raise ArtifactCorruptError(f"legacy migration target missing for {digest}")
        # The target must itself be a registered strict contract. This verifies the
        # migration registry cannot point to an ungoverned payload shape.
        payload_contract(entry["target_schema_id"])
    return reg


def payload_registry() -> dict:
    reg = _resource_json("contracts/PAYLOAD_CONTRACT_REGISTRY_V1.json")
    if reg.get("schema_id") != REGISTRY_SCHEMA_ID:
        raise ContractError("payload contract registry has unexpected schema")
    body = {k: v for k, v in reg.items() if k != "registry_sha256"}
    if canonical_sha256(body) != reg.get("registry_sha256"):
        raise ArtifactCorruptError("payload contract registry hash mismatch")
    return reg


def payload_contract(schema_id: str) -> dict:
    reg = payload_registry()
    entry = reg.get("payload_schemas", {}).get(schema_id)
    schema_loader = _resource_json
    if entry is None:
        # Gate 2 adds native oracle contracts in a separate extension registry so the
        # Gate 1 frozen-baseline registry remains byte-stable and historically meaningful.
        try:
            from .oracle import gate2_contract_registry
            ext = gate2_contract_registry()
            entry = ext.get("payload_schemas", {}).get(schema_id)
            if entry is not None:
                schema_loader = lambda rel: _resource_json("oracle/" + rel)
        except Exception:
            entry = None
    if entry is None:
        # DEL-001 promotes only explicitly contracted v0.27 runtime/science artifacts.
        try:
            from .v027_contracts import v027_contract_registry
            ext = v027_contract_registry()
            entry = ext.get("payload_schemas", {}).get(schema_id)
            if entry is not None:
                schema_loader = lambda rel: _resource_json("v027/" + rel)
        except Exception:
            entry = None
    if entry is None:
        raise UnknownSchemaError(f"unregistered payload schema: {schema_id}")
    if entry.get("decision") not in {"STRICT_VALIDATE_AND_WRAP", "NATIVE_STRICT"}:
        raise MigrationError(f"payload schema {schema_id} decision is {entry.get('decision')}")
    schema = schema_loader(entry["schema_path"])
    if canonical_sha256(schema) != entry.get("schema_sha256"):
        raise ArtifactCorruptError(f"payload schema registry hash mismatch: {schema_id}")
    return schema


def declared_schema_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise UnknownSchemaError("payload must be an object carrying schema_id or schema")
    sid = payload.get("schema_id") or payload.get("schema")
    if not isinstance(sid, str) or not sid:
        raise UnknownSchemaError("payload does not declare schema_id/schema")
    return sid


def validate_payload(payload: dict, *, schema_id: str | None = None) -> str:
    sid = schema_id or declared_schema_id(payload)
    schema = payload_contract(sid)
    errors = validate(schema, payload, raise_on_error=False)
    if errors:
        raise ContractError(f"payload {sid} failed strict contract: {'; '.join(errors[:20])}")
    return sid


def release_identity() -> dict:
    meta = build_meta()
    source = meta.get("source_sha256")
    if not isinstance(source, str) or not _SHA_RE.match(source):
        raise ContractError("build metadata lacks valid source_sha256")
    return {
        "package_version": __version__,
        "release_id": meta.get("release_id") or f"decoder-{__version__}-{source[:16]}",
        "source_sha256": source,
        "baseline_release": meta.get("baseline_release"),
        "semantic_delta": meta.get("semantic_delta", "NONE"),
    }


def _normalize_adapters(adapters: Iterable[dict] | None) -> list[dict]:
    out = []
    for a in adapters or []:
        if not isinstance(a, dict) or set(a) != {"adapter_id", "version"}:
            raise ContractError("adapter binding must contain exactly adapter_id and version")
        out.append({"adapter_id": str(a["adapter_id"]), "version": str(a["version"])})
    return sorted(out, key=lambda x: (x["adapter_id"], x["version"]))


def _normalize_dependencies(dependencies: Iterable[dict] | None) -> list[dict]:
    out = []
    for d in dependencies or []:
        if not isinstance(d, dict):
            raise ContractError("dependency must be object")
        allowed = {"role", "sha256", "size_bytes"}
        if set(d) - allowed or not {"role", "sha256"}.issubset(d):
            raise ContractError("dependency requires role+sha256 and optional size_bytes only")
        sha = str(d["sha256"])
        if not _SHA_RE.match(sha):
            raise ContractError("dependency sha256 is invalid")
        row = {"role": str(d["role"]), "sha256": sha}
        if "size_bytes" in d:
            size = int(d["size_bytes"])
            if size < 0:
                raise ContractError("dependency size_bytes must be nonnegative")
            row["size_bytes"] = size
        out.append(row)
    return sorted(out, key=lambda x: (x["role"], x["sha256"], x.get("size_bytes", -1)))


def _content_key(envelope: dict) -> dict:
    # Operational metadata (time, host, process, path, wall time) is deliberately
    # absent. Scientific/content identity can therefore survive resume/migration.
    return {
        "schema_id": envelope["schema_id"],
        "envelope_version": envelope["envelope_version"],
        "payload_schema_id": envelope["payload_schema_id"],
        "payload_sha256": envelope["payload_sha256"],
        "decoder_release": envelope["decoder_release"],
        "semantic_profile": envelope["semantic_profile"],
        "canonicalizer": envelope["canonicalizer"],
        "adapters": envelope["adapters"],
        "dependencies": envelope["dependencies"],
    }


def create_envelope(
    payload: dict,
    *,
    semantic_profile_id: str,
    semantic_profile_version: str = "1.0.0",
    adapters: Iterable[dict] | None = None,
    dependencies: Iterable[dict] | None = None,
    operational_metadata: dict | None = None,
    strict_payload: bool = True,
) -> dict:
    sid = declared_schema_id(payload)
    if strict_payload:
        validate_payload(payload, schema_id=sid)
    env = {
        "schema_id": ENVELOPE_SCHEMA_ID,
        "envelope_version": ENVELOPE_VERSION,
        "payload_schema_id": sid,
        "payload": payload,
        "payload_sha256": canonical_sha256(payload),
        "decoder_release": release_identity(),
        "semantic_profile": {"profile_id": semantic_profile_id, "version": semantic_profile_version},
        "canonicalizer": {"id": CANONICALIZER_ID, "version": CANONICALIZER_VERSION},
        "adapters": _normalize_adapters(adapters),
        "dependencies": _normalize_dependencies(dependencies),
        "operational_metadata": operational_metadata or {},
    }
    env["content_sha256"] = canonical_sha256(_content_key(env))
    env["envelope_sha256"] = canonical_sha256({k: v for k, v in env.items() if k != "envelope_sha256"})
    verify_envelope(env, strict_payload=strict_payload)
    return env


def _check_major(version: str, supported: int = 1) -> None:
    try:
        major = int(str(version).split(".", 1)[0])
    except Exception as exc:
        raise UnsupportedArtifactVersion(f"invalid artifact version: {version!r}") from exc
    if major != supported:
        raise UnsupportedArtifactVersion(f"unsupported artifact major {major}; supported major is {supported}")


def verify_envelope(envelope: dict, *, strict_payload: bool = True) -> dict:
    if not isinstance(envelope, dict):
        raise ContractError("artifact envelope must be object")
    if envelope.get("schema_id") != ENVELOPE_SCHEMA_ID:
        sid = envelope.get("schema_id")
        if isinstance(sid, str) and sid.startswith("IG_ARTIFACT_ENVELOPE_V"):
            raise UnsupportedArtifactVersion(f"unsupported envelope schema {sid}")
        raise ContractError(f"not a Decoder artifact envelope: {sid}")
    _check_major(envelope.get("envelope_version", ""), 1)
    errors = validate(envelope_schema(), envelope, raise_on_error=False)
    if errors:
        raise ContractError("artifact envelope schema failure: " + "; ".join(errors[:20]))
    payload = envelope["payload"]
    sid = declared_schema_id(payload)
    if sid != envelope["payload_schema_id"]:
        raise ArtifactCorruptError("payload schema binding mismatch")
    if strict_payload:
        validate_payload(payload, schema_id=sid)
    if canonical_sha256(payload) != envelope["payload_sha256"]:
        raise ArtifactCorruptError("payload content hash mismatch")
    if canonical_sha256(_content_key(envelope)) != envelope["content_sha256"]:
        raise ArtifactCorruptError("content identity hash mismatch")
    expected_env = canonical_sha256({k: v for k, v in envelope.items() if k != "envelope_sha256"})
    if expected_env != envelope["envelope_sha256"]:
        raise ArtifactCorruptError("envelope hash mismatch")
    return {
        "status": "PASS",
        "schema_id": ENVELOPE_SCHEMA_ID,
        "payload_schema_id": sid,
        "payload_sha256": envelope["payload_sha256"],
        "content_sha256": envelope["content_sha256"],
        "envelope_sha256": envelope["envelope_sha256"],
    }


def load_envelope(path: str | Path, *, strict_payload: bool = True) -> dict:
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactCorruptError(f"cannot parse artifact envelope {p}: {type(exc).__name__}: {exc}") from exc
    verify_envelope(obj, strict_payload=strict_payload)
    return obj


def write_envelope(path: str | Path, envelope: dict, *, strict_payload: bool = True) -> None:
    verify_envelope(envelope, strict_payload=strict_payload)
    write_json_atomic(Path(path), envelope)


def wrap_legacy_payload(
    payload: dict,
    *,
    semantic_profile_id: str = "LEGACY_V0_26_PRESERVED",
    operational_metadata: dict | None = None,
) -> dict:
    """Deterministically wrap a schema-tagged v0.26 payload without rewriting it."""
    return create_envelope(
        payload,
        semantic_profile_id=semantic_profile_id,
        operational_metadata=operational_metadata or {"migration": "WRAP_WITHOUT_PAYLOAD_REWRITE"},
        strict_payload=True,
    )


def classify_json_file(path: str | Path) -> dict:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "REJECT", "decision": "REJECT_MALFORMED_JSON", "path": str(p), "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"status": "MIGRATION_REQUIRED", "decision": "MIGRATION_REQUIRED_UNSCHEMATIZED", "path": str(p), "reason": "top-level JSON is not object"}
    sid = payload.get("schema_id") or payload.get("schema")
    if not sid:
        if "$schema" in payload:
            return {"status": "NON_RUNTIME_SCHEMA_DOCUMENT", "decision": "SCHEMA_DOCUMENT_NOT_RUNTIME_PAYLOAD", "path": str(p)}
        digest = canonical_sha256(payload)
        migration = legacy_unschematized_migration_registry().get("entries", {}).get(digest)
        if migration is not None:
            return {
                "status": "MIGRATABLE",
                "decision": "EXACT_DIGEST_SCHEMA_INJECTION_AND_WRAP",
                "path": str(p),
                "canonical_sha256": digest,
                "target_schema_id": migration["target_schema_id"],
                "historical_path": migration["historical_path"],
            }
        return {
            "status": "MIGRATION_REQUIRED",
            "decision": "MIGRATION_REQUIRED_UNSCHEMATIZED",
            "path": str(p),
            "canonical_sha256": digest,
            "reason": "no exact-digest migration rule",
        }
    try:
        validate_payload(payload, schema_id=sid)
        return {"status": "PASS", "decision": "STRICT_VALIDATE_AND_WRAP", "path": str(p), "schema_id": sid, "payload_sha256": canonical_sha256(payload)}
    except UnknownSchemaError as exc:
        return {"status": "UNSUPPORTED", "decision": "UNSUPPORTED_SCHEMA", "path": str(p), "schema_id": sid, "error": str(exc)}
    except ContractError as exc:
        return {"status": "REJECT", "decision": "REJECT_SCHEMA_INVALID", "path": str(p), "schema_id": sid, "error": str(exc)}


def failure_record(*, error: Exception, operation: str, artifact: str | None = None, context: dict | None = None) -> dict:
    return {
        "schema_id": FAILURE_SCHEMA_ID,
        "status": "FAIL",
        "operation": operation,
        "error_type": type(error).__name__,
        "message": str(error),
        "artifact": artifact,
        "context": context or {},
    }
