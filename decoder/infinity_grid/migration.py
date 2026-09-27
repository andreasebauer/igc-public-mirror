from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .canon import canonical_sha256, write_json_atomic
from .contracts import (
    classify_json_file,
    legacy_unschematized_migration_registry,
    validate_payload,
    wrap_legacy_payload,
    write_envelope,
)
from .errors import MigrationError
from .hashing import sha256_file
from .store import ArtifactStore


def import_source(paths, path: Path, *, source_id: str, expected_sha256: str, logical_role="SOURCE_ARCHIVE"):
    p = Path(path)
    observed = sha256_file(p)
    if observed != expected_sha256:
        raise RuntimeError(f"source hash mismatch for {source_id}: {observed} != {expected_sha256}")
    store = ArtifactStore(paths.store)
    rec = store.put_file(p, logical_role=logical_role, source_name=p.name)
    out = {
        "schema_id": "IG_SOURCE_IMPORT_RECORD_V0_17",
        "source_id": source_id,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "match": True,
        "artifact": rec,
    }
    d = paths.store / "source_records"
    d.mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / f"{source_id}.json", out)
    return out


def migrate_json_file_to_envelope(
    source: str | Path,
    destination: str | Path,
    *,
    semantic_profile_id: str = "LEGACY_V0_26_PRESERVED",
) -> dict:
    """Wrap an already schema-tagged legacy payload without rewriting the payload."""
    src = Path(source)
    payload = json.loads(src.read_text(encoding="utf-8"))
    env = wrap_legacy_payload(
        payload,
        semantic_profile_id=semantic_profile_id,
        operational_metadata={
            "migration": "WRAP_WITHOUT_PAYLOAD_REWRITE",
            "source_name": src.name,
            "source_payload_sha256": canonical_sha256(payload),
        },
    )
    write_envelope(Path(destination), env)
    return {
        "status": "PASS",
        "source": str(src),
        "destination": str(destination),
        "migration": "WRAP_WITHOUT_PAYLOAD_REWRITE",
        "payload_sha256": env["payload_sha256"],
        "content_sha256": env["content_sha256"],
        "envelope_sha256": env["envelope_sha256"],
    }


def migrate_unschematized_v026_payload(payload: dict) -> tuple[dict, dict]:
    """Apply an exact-digest migration to one known v0.26 unversioned payload.

    Gate 1 deliberately does not infer a schema from arbitrary unversioned JSON at
    load time.  Only payloads whose *pre-migration canonical digest* is frozen in
    LEGACY_UNSCHEMATIZED_MIGRATIONS_V1 may be migrated.  The only payload rewrite
    is insertion of the declared ``schema_id``; all historical fields and values
    remain byte-semantically identical under canonical JSON.
    """
    if not isinstance(payload, dict):
        raise MigrationError("unversioned v0.26 migration requires a JSON object")
    if payload.get("schema_id") or payload.get("schema"):
        raise MigrationError("payload already declares a schema; use ordinary envelope migration")
    before_sha = canonical_sha256(payload)
    registry = legacy_unschematized_migration_registry()
    rule = registry.get("entries", {}).get(before_sha)
    if rule is None:
        raise MigrationError(f"no exact-digest Gate-1 migration rule for unversioned payload {before_sha}")
    migrated = dict(payload)
    migrated["schema_id"] = rule["target_schema_id"]
    validate_payload(migrated, schema_id=rule["target_schema_id"])
    after_without_schema = dict(migrated)
    after_without_schema.pop("schema_id", None)
    if canonical_sha256(after_without_schema) != before_sha:
        raise MigrationError("legacy migration changed historical fields/values")
    evidence = {
        "status": "PASS",
        "migration": "EXACT_DIGEST_SCHEMA_INJECTION",
        "pre_migration_payload_sha256": before_sha,
        "target_schema_id": rule["target_schema_id"],
        "historical_path": rule["historical_path"],
        "post_migration_payload_sha256": canonical_sha256(migrated),
        "historical_fields_preserved": True,
    }
    return migrated, evidence


def migrate_unschematized_v026_file_to_envelope(
    source: str | Path,
    destination: str | Path,
    *,
    semantic_profile_id: str = "LEGACY_V0_26_PRESERVED",
) -> dict:
    src = Path(source)
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MigrationError(f"cannot parse legacy JSON {src}: {type(exc).__name__}: {exc}") from exc
    migrated, evidence = migrate_unschematized_v026_payload(payload)
    env = wrap_legacy_payload(
        migrated,
        semantic_profile_id=semantic_profile_id,
        operational_metadata={
            "migration": "EXACT_DIGEST_SCHEMA_INJECTION_AND_WRAP",
            "source_name": src.name,
            "pre_migration_payload_sha256": evidence["pre_migration_payload_sha256"],
            "target_schema_id": evidence["target_schema_id"],
        },
    )
    write_envelope(Path(destination), env)
    return {
        **evidence,
        "source": str(src),
        "destination": str(destination),
        "payload_sha256": env["payload_sha256"],
        "content_sha256": env["content_sha256"],
        "envelope_sha256": env["envelope_sha256"],
    }


def scan_legacy_json_tree(root: str | Path, *, baseline: dict | None = None) -> dict:
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*.json")):
        row = classify_json_file(p)
        row["relative_path"] = p.relative_to(root).as_posix()
        row.pop("path", None)
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    malformed = sum(1 for r in rows if r.get("decision") == "REJECT_MALFORMED_JSON")
    allowed = {"PASS", "MIGRATABLE", "MIGRATION_REQUIRED", "UNSUPPORTED", "REJECT", "NON_RUNTIME_SCHEMA_DOCUMENT"}
    report = {
        "schema_id": "IG_DECODER_MIGRATION_REPORT_V1",
        "status": "PASS_WITH_EXPLICIT_DECISIONS" if all(r["status"] in allowed for r in rows) else "FAIL",
        "baseline": baseline or {},
        "counts": {
            "total": len(rows),
            "pass": counts.get("PASS", 0),
            # Compatibility field retained from the first Gate-1 report. Known
            # exact-digest migrations still require a migration action, while the
            # per-file rows distinguish deterministic MIGRATABLE from unresolved.
            "migration_required": counts.get("MIGRATABLE", 0) + counts.get("MIGRATION_REQUIRED", 0),
            "unsupported": counts.get("UNSUPPORTED", 0),
            "malformed": malformed,
            "non_runtime_schema_documents": counts.get("NON_RUNTIME_SCHEMA_DOCUMENT", 0),
        },
        "rows": rows,
    }
    return report
