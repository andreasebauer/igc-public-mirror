"""Content-certified reusable data for Decoder 0.7.

Trust follows the exact bytes and their declared semantics.  The Decoder,
test, code, and environment versions that produced those bytes are retained
as diagnostic provenance only; they are never admission gates.

The store is deliberately small and inspectable::

    objects/<content sha256>.bin              physical bytes (deduplicated)
    blocks/<block id>.json                    meaning/schema/coverage
    certifications/<certification id>.json   immutable certification
    bases/<base id>.json                      logical base catalog
    deltas/<delta id>.json                    base-to-base difference
    revocations/<block id>/<record id>.json   exact-hash revocation
    ACTIVE_BASE.json                          replaceable local selection

This module does not promote a Decoder release, migrate a live project, or
rerun scientific work.  A test may reuse certified bytes across versions and
must refuse changed bytes, incompatible semantics, missing coverage, and
revoked hashes.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import time

from .canon import canonical_bytes, canonical_sha256, write_json_atomic

DATA_BLOCK = "IG_DECODER_DATA_BLOCK_V1"
DATA_CERTIFICATION = "IG_DECODER_DATA_CERTIFICATION_V1"
CERTIFIED_BASE = "IG_DECODER_CERTIFIED_BASE_V1"
DATA_DELTA = "IG_DECODER_CERTIFIED_BASE_DELTA_V1"
REVOCATION = "IG_DECODER_DATA_REVOCATION_V1"
READ_EVENT = "IG_DECODER_DATA_READ_V1"

CATEGORIES = frozenset({
    "core_objects", "core_relations", "public_states",
    "quotients_and_mappings", "reference_corpora",
    "positive_controls", "negative_controls", "schemas_and_dictionaries",
})
CANONICALIZATIONS = frozenset({"RAW_BYTES_V1", "CANONICAL_JSON_V1"})
HASH = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


class CertifiedBaseError(RuntimeError):
    """Stable refusal carrying a machine-readable code."""

    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}:{detail}")


def _root(store):
    return Path(store).resolve()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _token(value, field):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise CertifiedBaseError("DATA_TOKEN_INVALID", field)
    return value


def _texts(values, field, *, allow_empty=False):
    if not isinstance(values, list) or (not values and not allow_empty):
        raise CertifiedBaseError("DATA_LIST_REQUIRED", field)
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise CertifiedBaseError("DATA_LIST_INVALID", field)
    cleaned = sorted(set(v.strip() for v in values))
    if len(cleaned) != len(values):
        raise CertifiedBaseError("DATA_LIST_DUPLICATE", field)
    return cleaned


def _sealed(row):
    result = dict(row)
    result["record_sha256"] = canonical_sha256(result)
    return result


def _verify_record(row, schema):
    if not isinstance(row, dict) or row.get("schema_id") != schema:
        raise CertifiedBaseError("DATA_RECORD_SCHEMA_INVALID", schema)
    digest = row.get("record_sha256")
    body = {k: v for k, v in row.items() if k != "record_sha256"}
    if not isinstance(digest, str) or digest != canonical_sha256(body):
        raise CertifiedBaseError("DATA_RECORD_HASH_INVALID", schema)
    return row


def _write_immutable(path, row):
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise CertifiedBaseError("DATA_IMMUTABLE_CONFLICT", str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _read(path, schema):
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise CertifiedBaseError("DATA_RECORD_UNREADABLE", str(path)) from exc
    return _verify_record(row, schema)


def canonicalize(content, method):
    """Return the bytes whose hash is certified."""
    if method not in CANONICALIZATIONS:
        raise CertifiedBaseError("DATA_CANONICALIZATION_UNSUPPORTED", method)
    if method == "RAW_BYTES_V1":
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise CertifiedBaseError("DATA_BYTES_REQUIRED")
        return bytes(content)
    if isinstance(content, (bytes, bytearray, memoryview, str)):
        try:
            content = json.loads(bytes(content).decode() if not isinstance(content, str) else content)
        except (ValueError, UnicodeError) as exc:
            raise CertifiedBaseError("DATA_JSON_INVALID") from exc
    try:
        return canonical_bytes(content)
    except Exception as exc:
        raise CertifiedBaseError("DATA_JSON_NOT_CANONICALIZABLE") from exc


def add_block(store, *, category, content, canonicalization, schema_id, meaning,
              domain, coverage, invariants, producer=None):
    """Add one immutable logical block and deduplicate its physical bytes.

    ``producer`` is saved separately, so changing producer/test versions cannot
    change the block identity or its later admissibility.
    """
    if category not in CATEGORIES:
        raise CertifiedBaseError("DATA_CATEGORY_INVALID", category)
    raw = canonicalize(content, canonicalization)
    content_sha = _hash(raw)
    block = _sealed({
        "schema_id": DATA_BLOCK,
        "category": category,
        "content_sha256": content_sha,
        "size_bytes": len(raw),
        "canonicalization": canonicalization,
        "payload_schema_id": _token(schema_id, "schema_id"),
        "meaning": _token(meaning, "meaning"),
        "domain": _token(domain, "domain"),
        "coverage": _texts(coverage, "coverage"),
        "invariants": _texts(invariants, "invariants"),
    })
    root = _root(store)
    obj = root / "objects" / (content_sha + ".bin")
    obj.parent.mkdir(parents=True, exist_ok=True)
    if obj.exists():
        if obj.is_symlink() or obj.read_bytes() != raw:
            raise CertifiedBaseError("DATA_OBJECT_HASH_COLLISION", content_sha)
        physical_created = False
    else:
        with obj.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        physical_created = True
    block_id = block["record_sha256"]
    _write_immutable(root / "blocks" / (block_id + ".json"), block)
    producer_id = None
    if producer is not None:
        if not isinstance(producer, dict):
            raise CertifiedBaseError("DATA_PRODUCER_INVALID")
        diagnostic = _sealed({
            "schema_id": "IG_DECODER_DATA_PRODUCER_DIAGNOSTIC_V1",
            "block_id": block_id,
            "content_sha256": content_sha,
            "diagnostic_only_not_admission_authority": True,
            "producer": producer,
        })
        producer_id = diagnostic["record_sha256"]
        _write_immutable(root / "producers" / block_id / (producer_id + ".json"), diagnostic)
    return {"status": "DATA_BLOCK_ADDED", "block_id": block_id,
            "content_sha256": content_sha, "physical_object_created": physical_created,
            "producer_record_id": producer_id}


def load_block(store, block_id, *, verify_bytes=True):
    if not isinstance(block_id, str) or not HASH.fullmatch(block_id):
        raise CertifiedBaseError("DATA_BLOCK_ID_INVALID")
    root = _root(store)
    row = _read(root / "blocks" / (block_id + ".json"), DATA_BLOCK)
    if row["record_sha256"] != block_id:
        raise CertifiedBaseError("DATA_BLOCK_ID_MISMATCH")
    if verify_bytes:
        obj = root / "objects" / (row["content_sha256"] + ".bin")
        try:
            raw = obj.read_bytes()
        except OSError as exc:
            raise CertifiedBaseError("DATA_OBJECT_MISSING", row["content_sha256"]) from exc
        if len(raw) != row["size_bytes"] or _hash(raw) != row["content_sha256"]:
            raise CertifiedBaseError("DATA_OBJECT_CORRUPT", row["content_sha256"])
    return row


def certify_block(store, block_id, *, authority, basis):
    block = load_block(store, block_id)
    cert = _sealed({
        "schema_id": DATA_CERTIFICATION,
        "block_id": block_id,
        "content_sha256": block["content_sha256"],
        "certified_semantics_sha256": canonical_sha256({
            k: block[k] for k in ("category", "canonicalization", "payload_schema_id",
                                  "meaning", "domain", "coverage", "invariants")
        }),
        "authority": _token(authority, "authority"),
        "basis": _texts(basis, "basis"),
        "decision": "CERTIFIED_FOR_DECLARED_SEMANTICS",
        "producer_versions_are_diagnostic_only": True,
    })
    root = _root(store)
    cert_id = cert["record_sha256"]
    _write_immutable(root / "certifications" / (cert_id + ".json"), cert)
    pointer = root / "block_certifications" / (block_id + ".json")
    _write_immutable(pointer, cert)
    return {"status": "DATA_BLOCK_CERTIFIED", "block_id": block_id,
            "certification_id": cert_id}


def certification(store, block_id):
    row = _read(_root(store) / "block_certifications" / (block_id + ".json"), DATA_CERTIFICATION)
    if row.get("block_id") != block_id:
        raise CertifiedBaseError("DATA_CERTIFICATION_BINDING_INVALID")
    return row


def revoke(store, block_id, *, reason, evidence):
    """Revoke exactly one logical block hash; unrelated bytes stay trusted."""
    block = load_block(store, block_id)
    record = _sealed({
        "schema_id": REVOCATION,
        "block_id": block_id,
        "content_sha256": block["content_sha256"],
        "reason": _token(reason, "reason"),
        "evidence": _texts(evidence, "evidence"),
        "scope": "EXACT_BLOCK_HASH_AND_BASES_THAT_REFERENCE_IT",
    })
    rid = record["record_sha256"]
    _write_immutable(_root(store) / "revocations" / block_id / (rid + ".json"), record)
    return {"status": "DATA_BLOCK_REVOKED", "block_id": block_id, "revocation_id": rid}


def revocations(store, block_id):
    folder = _root(store) / "revocations" / block_id
    return [] if not folder.exists() else [_read(p, REVOCATION) for p in sorted(folder.glob("*.json"))]


def admit_block(store, block_id, *, schema_id, meaning, domain,
                required_coverage, required_invariants):
    block = load_block(store, block_id)
    cert = certification(store, block_id)
    if revocations(store, block_id):
        raise CertifiedBaseError("DATA_BLOCK_REVOKED", block_id)
    expected = {"payload_schema_id": schema_id, "meaning": meaning, "domain": domain}
    for field, value in expected.items():
        if block[field] != value:
            raise CertifiedBaseError("DATA_SEMANTICS_INCOMPATIBLE", field)
    missing_coverage = sorted(set(required_coverage) - set(block["coverage"]))
    if missing_coverage:
        raise CertifiedBaseError("DATA_COVERAGE_MISSING", ",".join(missing_coverage))
    missing_invariants = sorted(set(required_invariants) - set(block["invariants"]))
    if missing_invariants:
        raise CertifiedBaseError("DATA_INVARIANT_MISSING", ",".join(missing_invariants))
    return {"status": "DATA_BLOCK_ADMITTED", "block_id": block_id,
            "content_sha256": block["content_sha256"],
            "certification_id": cert["record_sha256"]}


def _base_materialization(rows):
    files = []
    for block in rows:
        files.append({"path": f"blocks/{block['category']}/{block['record_sha256']}.bin",
                      "sha256": block["content_sha256"], "size_bytes": block["size_bytes"]})
    return files, canonical_sha256(files)


def create_base(store, *, name, block_ids, activate=False):
    ids = sorted(set(block_ids))
    if not ids or len(ids) != len(block_ids):
        raise CertifiedBaseError("CERTIFIED_BASE_BLOCKS_INVALID")
    rows = []
    entries = []
    for block_id in ids:
        block = load_block(store, block_id)
        cert = certification(store, block_id)
        if revocations(store, block_id):
            raise CertifiedBaseError("CERTIFIED_BASE_CONTAINS_REVOKED", block_id)
        rows.append(block)
        entries.append({"block_id": block_id, "certification_id": cert["record_sha256"],
                        "category": block["category"], "content_sha256": block["content_sha256"]})
    files, materialization_sha = _base_materialization(rows)
    base = _sealed({
        "schema_id": CERTIFIED_BASE,
        "name": _token(name, "name"),
        "blocks": entries,
        "materialization_files": files,
        "materialization_sha256": materialization_sha,
        "admission_rule": "CONTENT_AND_DECLARED_SEMANTICS_NOT_PRODUCER_VERSION",
    })
    root = _root(store)
    base_id = base["record_sha256"]
    _write_immutable(root / "bases" / (base_id + ".json"), base)
    if activate:
        write_json_atomic(root / "ACTIVE_BASE.json", {
            "schema_id": "IG_DECODER_ACTIVE_CERTIFIED_BASE_V1", "base_id": base_id,
            "name": base["name"], "selection_only_not_release_promotion": True,
        })
    return {"status": "CERTIFIED_BASE_CREATED", "base_id": base_id,
            "materialization_sha256": materialization_sha, "block_count": len(entries)}


def load_base(store, base_id, *, admit=True):
    if not isinstance(base_id, str) or not HASH.fullmatch(base_id):
        raise CertifiedBaseError("CERTIFIED_BASE_ID_INVALID")
    row = _read(_root(store) / "bases" / (base_id + ".json"), CERTIFIED_BASE)
    if row["record_sha256"] != base_id:
        raise CertifiedBaseError("CERTIFIED_BASE_ID_MISMATCH")
    if admit:
        blocks = [load_block(store, item["block_id"]) for item in row["blocks"]]
        for item, block in zip(row["blocks"], blocks):
            cert = certification(store, item["block_id"])
            if (item["certification_id"] != cert["record_sha256"] or
                    item["content_sha256"] != block["content_sha256"]):
                raise CertifiedBaseError("CERTIFIED_BASE_BINDING_INVALID")
            if revocations(store, item["block_id"]):
                raise CertifiedBaseError("CERTIFIED_BASE_CONTAINS_REVOKED", item["block_id"])
        files, digest = _base_materialization(blocks)
        if files != row["materialization_files"] or digest != row["materialization_sha256"]:
            raise CertifiedBaseError("CERTIFIED_BASE_MATERIALIZATION_INVALID")
    return row


def materialize_base(store, base_id, destination):
    """Reconstruct a base into a new clean directory and verify every byte."""
    base = load_base(store, base_id)
    destination = Path(destination)
    if destination.exists():
        raise CertifiedBaseError("CERTIFIED_BASE_DESTINATION_EXISTS")
    destination.mkdir(parents=True)
    try:
        for item in base["materialization_files"]:
            source = _root(store) / "objects" / (item["sha256"] + ".bin")
            target = destination / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        manifest = {"base_id": base_id, "materialization_files": base["materialization_files"],
                    "materialization_sha256": base["materialization_sha256"]}
        write_json_atomic(destination / "CERTIFIED_BASE.json", manifest)
        verify_materialization(destination, base)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"status": "CERTIFIED_BASE_MATERIALIZED", "base_id": base_id,
            "materialization_sha256": base["materialization_sha256"], "path": str(destination.resolve())}


def verify_materialization(directory, base):
    for item in base["materialization_files"]:
        path = Path(directory) / item["path"]
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CertifiedBaseError("CERTIFIED_BASE_FILE_MISSING", item["path"]) from exc
        if len(raw) != item["size_bytes"] or _hash(raw) != item["sha256"]:
            raise CertifiedBaseError("CERTIFIED_BASE_FILE_CORRUPT", item["path"])
    return True


def create_delta(store, parent_base_id, target_base_id):
    parent = load_base(store, parent_base_id)
    target = load_base(store, target_base_id)
    before = {x["block_id"] for x in parent["blocks"]}
    after = {x["block_id"] for x in target["blocks"]}
    delta = _sealed({
        "schema_id": DATA_DELTA,
        "parent_base_id": parent_base_id,
        "target_base_id": target_base_id,
        "added_block_ids": sorted(after - before),
        "removed_block_ids": sorted(before - after),
        "target_materialization_sha256": target["materialization_sha256"],
    })
    delta_id = delta["record_sha256"]
    _write_immutable(_root(store) / "deltas" / (delta_id + ".json"), delta)
    return {"status": "CERTIFIED_BASE_DELTA_CREATED", "delta_id": delta_id,
            "added": len(after - before), "removed": len(before - after)}


def materialize_delta(store, delta_id, destination):
    delta = _read(_root(store) / "deltas" / (delta_id + ".json"), DATA_DELTA)
    if delta["record_sha256"] != delta_id:
        raise CertifiedBaseError("DATA_DELTA_ID_MISMATCH")
    parent = load_base(store, delta["parent_base_id"])
    target = load_base(store, delta["target_base_id"])
    calculated = (({x["block_id"] for x in parent["blocks"]} - set(delta["removed_block_ids"])) |
                  set(delta["added_block_ids"]))
    if calculated != {x["block_id"] for x in target["blocks"]}:
        raise CertifiedBaseError("DATA_DELTA_RECONSTRUCTION_INVALID")
    if target["materialization_sha256"] != delta["target_materialization_sha256"]:
        raise CertifiedBaseError("DATA_DELTA_TARGET_HASH_INVALID")
    result = materialize_base(store, delta["target_base_id"], destination)
    result["delta_id"] = delta_id
    result["status"] = "CERTIFIED_BASE_DELTA_MATERIALIZED"
    return result


def reference_counts(store):
    """Return physical-byte references from all retained logical bases."""
    root = _root(store)
    counts = {}
    for path in sorted((root / "bases").glob("*.json")) if (root / "bases").exists() else []:
        base = _read(path, CERTIFIED_BASE)
        for item in base["blocks"]:
            counts[item["content_sha256"]] = counts.get(item["content_sha256"], 0) + 1
    return counts


def compaction_plan(store, retained_base_ids):
    """Plan only. Deletion requires a separate explicit implementation/action."""
    root = _root(store)
    retained = set()
    for base_id in sorted(set(retained_base_ids)):
        base = load_base(store, base_id, admit=False)
        retained.update(item["content_sha256"] for item in base["blocks"])
    physical = {p.stem for p in (root / "objects").glob("*.bin")} if (root / "objects").exists() else set()
    return _sealed({
        "schema_id": "IG_DECODER_CERTIFIED_BASE_COMPACTION_PLAN_V1",
        "retained_base_ids": sorted(set(retained_base_ids)),
        "retained_content_sha256": sorted(retained),
        "unreferenced_content_sha256": sorted(physical - retained),
        "action": "DRY_RUN_NO_DELETION",
    })


def read_block(store, block_id, *, consumer, audit=True):
    """Read verified bytes and optionally record what a representative test used."""
    block = load_block(store, block_id)
    certification(store, block_id)
    if revocations(store, block_id):
        raise CertifiedBaseError("DATA_BLOCK_REVOKED", block_id)
    raw = (_root(store) / "objects" / (block["content_sha256"] + ".bin")).read_bytes()
    if audit:
        event = _sealed({
            "schema_id": READ_EVENT, "block_id": block_id,
            "content_sha256": block["content_sha256"],
            "consumer": _token(consumer, "consumer"), "recorded_ns": time.time_ns(),
        })
        _write_immutable(_root(store) / "read_audit" / (event["record_sha256"] + ".json"), event)
    return raw


def read_audit_report(store):
    folder = _root(store) / "read_audit"
    events = [] if not folder.exists() else [_read(p, READ_EVENT) for p in sorted(folder.glob("*.json"))]
    return {"schema_id": "IG_DECODER_DATA_READ_AUDIT_REPORT_V1",
            "event_count": len(events), "block_ids": sorted({e["block_id"] for e in events}),
            "consumers": sorted({e["consumer"] for e in events})}
