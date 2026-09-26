from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .canon import canonical_sha256
from .contracts import payload_registry, validate_payload, verify_envelope
from .current_state import attach_state_hash, verify_current_state
from .errors import ArtifactCorruptError, ContractError, MigrationError, UnknownSchemaError
from .migration import migrate_unschematized_v026_payload
from .oracle import gate2_contract_registry

SPEC_SCHEMA_ID = "IG_V026_TO_V027_COMPATIBILITY_SPEC_V1"
FIXTURE_INDEX_SCHEMA_ID = "IG_V026_COMPATIBILITY_FIXTURE_INDEX_V1"
CANDIDATE_REGISTRY_SCHEMA_ID = "IG_V027_CANDIDATE_COMPONENT_REGISTRY_V1"
CURRENT_STATE_V3_SCHEMA_ID = "IG_DECODER_CURRENT_STATE_V3"


def _compat_resource(*parts: str):
    node = files("infinity_grid").joinpath("resources").joinpath("compatibility")
    for part in parts:
        node = node.joinpath(part)
    return node


def _read_json(*parts: str) -> dict:
    try:
        obj = json.loads(_compat_resource(*parts).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactCorruptError(f"cannot read compatibility resource {'/'.join(parts)}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ArtifactCorruptError(f"compatibility resource {'/'.join(parts)} is not a JSON object")
    return obj


def _verify_self_hash(obj: dict, field: str, label: str) -> None:
    got = obj.get(field)
    if not isinstance(got, str) or len(got) != 64:
        raise ArtifactCorruptError(f"{label} lacks valid {field}")
    body = {k: v for k, v in obj.items() if k != field}
    expected = canonical_sha256(body)
    if got != expected:
        raise ArtifactCorruptError(f"{label} self hash mismatch: {got} != {expected}")


def compatibility_spec() -> dict:
    spec = _read_json("V026_TO_V027_COMPATIBILITY_SPEC_V1.json")
    if spec.get("schema_id") != SPEC_SCHEMA_ID:
        raise ContractError("unexpected compatibility specification schema")
    _verify_self_hash(spec, "spec_sha256", "compatibility specification")
    if spec.get("semantic_delta_from_v026") != "NONE":
        raise ContractError("MIG-001 must not contain a v0.27 semantic delta")
    if spec.get("target_semantic_release") != "0.27.0":
        raise ContractError("compatibility target must remain 0.27.0")
    return spec


def compatibility_fixture_index() -> dict:
    idx = _read_json("V026_COMPATIBILITY_FIXTURE_INDEX_V1.json")
    if idx.get("schema_id") != FIXTURE_INDEX_SCHEMA_ID:
        raise ContractError("unexpected compatibility fixture-index schema")
    _verify_self_hash(idx, "index_sha256", "compatibility fixture index")
    return idx


def candidate_component_registry() -> dict:
    reg = _read_json("V027_CANDIDATE_COMPONENT_REGISTRY_V1.json")
    if reg.get("schema_id") != CANDIDATE_REGISTRY_SCHEMA_ID:
        raise ContractError("unexpected v0.27 candidate registry schema")
    _verify_self_hash(reg, "registry_sha256", "v0.27 candidate registry")
    return reg


def all_v026_schema_ids() -> set[str]:
    ids = set(payload_registry().get("payload_schemas", {}))
    ids.update(gate2_contract_registry().get("payload_schemas", {}))
    ids.add("IG_DECODER_CURRENT_STATE_V2")
    ids.add("IG_ARTIFACT_ENVELOPE_V1")
    return ids


def migration_decision(schema_id: str) -> dict:
    spec = compatibility_spec()
    entry = spec.get("schema_decisions", {}).get(schema_id)
    if entry is None:
        raise UnknownSchemaError(f"MIG-001 has no v0.26 compatibility decision for {schema_id}")
    return copy.deepcopy(entry)


def _resource_bytes(rel: str) -> bytes:
    parts = rel.split("/")
    return _compat_resource(*parts).read_bytes()


def _verify_fixture_digest(schema_id: str, fixture_meta: dict) -> dict:
    rel = fixture_meta["fixture_path"]
    data = _resource_bytes(rel)
    file_sha = hashlib.sha256(data).hexdigest()
    if file_sha != fixture_meta["file_sha256"]:
        raise ArtifactCorruptError(f"fixture file hash mismatch for {schema_id}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ArtifactCorruptError(f"fixture parse failed for {schema_id}: {exc}") from exc
    if canonical_sha256(payload) != fixture_meta["canonical_sha256"]:
        raise ArtifactCorruptError(f"fixture canonical hash mismatch for {schema_id}")
    return payload


def verify_compatibility_fixtures() -> dict:
    spec = compatibility_spec()
    idx = compatibility_fixture_index()
    expected = all_v026_schema_ids()
    decisions = set(spec.get("schema_decisions", {}))
    fixtures = set(idx.get("fixtures", {}))
    failures: list[dict[str, Any]] = []

    if decisions != expected:
        failures.append({
            "reason": "schema_decision_coverage",
            "missing": sorted(expected - decisions),
            "extra": sorted(decisions - expected),
        })
    if fixtures != expected:
        failures.append({
            "reason": "fixture_coverage",
            "missing": sorted(expected - fixtures),
            "extra": sorted(fixtures - expected),
        })

    checked = 0
    for sid in sorted(expected):
        try:
            payload = _verify_fixture_digest(sid, idx["fixtures"][sid])
            decision = spec["schema_decisions"][sid]["decision"]
            if sid == "IG_ARTIFACT_ENVELOPE_V1":
                verify_envelope(payload)
            elif sid in {"IG_DECODER_CURRENT_STATE_V1", "IG_DECODER_CURRENT_STATE_V2"}:
                verify_current_state(payload)
                if decision != "READ_VERIFY_AND_REISSUE_CURRENT_STATE_V3":
                    raise ContractError(f"unexpected current-state decision {decision}")
            else:
                validate_payload(payload, schema_id=sid)
                before = canonical_sha256(payload)
                # Read compatibility is a no-op on the historical scientific/configuration payload.
                after = canonical_sha256(json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False)))
                if before != after:
                    raise ArtifactCorruptError(f"compatibility read changed payload identity for {sid}")
            checked += 1
        except Exception as exc:
            failures.append({"reason": "schema_fixture_failure", "schema_id": sid, "error": f"{type(exc).__name__}: {exc}"})

    unschematized_checked = 0
    for rec in idx.get("unschematized", []):
        try:
            payload = json.loads(_resource_bytes(rec["fixture_path"]).decode("utf-8"))
            migrated, evidence = migrate_unschematized_v026_payload(payload)
            if evidence["pre_migration_payload_sha256"] != rec["pre_migration_payload_sha256"]:
                raise ArtifactCorruptError("unversioned fixture pre-migration hash mismatch")
            if migrated.get("schema_id") != rec["target_schema_id"]:
                raise MigrationError("unversioned fixture migrated to unexpected schema")
            unschematized_checked += 1
        except Exception as exc:
            failures.append({"reason": "unschematized_fixture_failure", "historical_path": rec.get("historical_path"), "error": f"{type(exc).__name__}: {exc}"})

    malformed_checked = 0
    for rec in idx.get("malformed", []):
        data = _resource_bytes(rec["fixture_path"])
        if hashlib.sha256(data).hexdigest() != rec["file_sha256"]:
            failures.append({"reason": "malformed_fixture_hash", "historical_path": rec.get("historical_path")})
            continue
        try:
            json.loads(data.decode("utf-8"))
        except Exception:
            malformed_checked += 1
        else:
            failures.append({"reason": "malformed_fixture_unexpectedly_parsed", "historical_path": rec.get("historical_path")})

    return {
        "schema_id": "IG_MIG001_COMPATIBILITY_VERIFICATION_RESULT_V1",
        "status": "PASS" if not failures else "FAIL",
        "spec_sha256": spec["spec_sha256"],
        "expected_schema_types": len(expected),
        "schema_fixtures_checked": checked,
        "unschematized_migrations_checked": unschematized_checked,
        "malformed_rejections_checked": malformed_checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def verify_candidate_handoff(handoff_root: str | Path) -> dict:
    root = Path(handoff_root)
    reg = candidate_component_registry()
    failures = []
    checked = 0
    for rec in reg.get("files", []):
        p = root / rec["path"]
        if not p.is_file():
            failures.append({"reason": "missing_candidate_file", "path": rec["path"]})
            continue
        data = p.read_bytes()
        checked += 1
        if len(data) != rec["size_bytes"]:
            failures.append({"reason": "candidate_size_mismatch", "path": rec["path"]})
        got = hashlib.sha256(data).hexdigest()
        if got != rec["sha256"]:
            failures.append({"reason": "candidate_sha256_mismatch", "path": rec["path"], "expected": rec["sha256"], "observed": got})
    return {
        "schema_id": "IG_MIG001_CANDIDATE_HANDOFF_VERIFICATION_V1",
        "status": "PASS" if not failures else "FAIL",
        "files_checked": checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def reissue_current_state_v3(old_state: dict, *, mig001_release: dict, as_of_utc: str) -> dict:
    """Reissue a verified V1/V2 context as MIG-001 V3 without inventing live state.

    This is deliberately not a scientific migration. The scientific status and hard rules are copied.
    The operational snapshot is explicitly marked historical/stale until refreshed from the real runtime.
    """
    verify_current_state(old_state)
    if old_state.get("schema_id") not in {"IG_DECODER_CURRENT_STATE_V1", "IG_DECODER_CURRENT_STATE_V2"}:
        raise ContractError("MIG-001 reissue accepts only current-state V1/V2")
    canonical_code = copy.deepcopy(old_state["canonical_code"])
    canonical_code.setdefault("gate2_oracle_release", {})
    canonical_code["mig001_compatibility_release"] = copy.deepcopy(mig001_release)
    canonical_code["semantic_delta_from_v0_26"] = "NONE"
    canonical_code["next_semantic_target"] = "0.27.0"

    op = copy.deepcopy(old_state["operational_snapshot"])
    op["freshness"] = "HISTORICAL_SNAPSHOT_REQUIRES_LIVE_REFRESH"
    op["rule"] = "MIG-001 never promotes historical process status to current liveness. Refresh from the declared runtime source before operational action."

    state = {
        "schema_id": CURRENT_STATE_V3_SCHEMA_ID,
        "state_version": "3.0.0",
        "authority_class": "CURRENT_CONTEXT",
        "as_of_utc": as_of_utc,
        "purpose": "Single authoritative Decoder state after MIG-001. Compatibility is frozen; algebra semantics remain v0.26 and Gate 3 refactor is next.",
        "canonical_code": canonical_code,
        "gate_status": {
            "gate_0_baseline_freeze": "COMPLETE",
            "gate_1_machine_contracts": "COMPLETE",
            "gate_2_v0_26_oracle": "COMPLETE",
            "mig_001_compatibility_spec": "COMPLETE",
            "gate_3_refactor": "NEXT",
            "v0_27_semantic_integration": "BLOCKED_UNTIL_GATE3_AND_DETERMINISM",
        },
        "science_status": copy.deepcopy(old_state["science_status"]),
        "operational_snapshot": op,
        "hard_rules": copy.deepcopy(old_state["hard_rules"]),
        "legacy_state_reconciliation": copy.deepcopy(old_state["legacy_state_reconciliation"]),
    }
    state = attach_state_hash(state)
    verify_current_state(state)
    return state
