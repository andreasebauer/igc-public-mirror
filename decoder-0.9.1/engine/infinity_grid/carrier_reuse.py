from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic


class CertifiedCarrierReuseError(RuntimeError):
    pass


_BUNDLE_NAMES = {'IG_MATURATION_STATE_DAG_V1': 'EXACT_STATE_DAG_V1.json',
                 'PYTHON_PICKLE_EXACT_STATE_V1': 'EXACT_STATE_PICKLE_V1.pkl'}
_IDENTITY_FIELDS = ('source_sha256', 'registry_sha256', 'producer_experiment_id',
                    'authority_stage', 'authority_science_sha256')
_ROW_FIELDS = {'carrier_id', 'public_state', 'public_state_sha256', 'exact_carrier_payload',
               'exact_carrier_root_digest', 'construction_digest', *_IDENTITY_FIELDS,
               'certification_status'}


def _text(value, field):
    if type(value) is not str or not value:
        raise CertifiedCarrierReuseError('CARRIER_EXACT_TEXT_REQUIRED:' + field)
    return value


def _digest(value, field):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise CertifiedCarrierReuseError('CARRIER_EXACT_SHA256_REQUIRED:' + field)
    return value


def _identity(values):
    for field in _IDENTITY_FIELDS:
        (_digest if field.endswith('sha256') else _text)(values[field], field)
    return values


def _validated_rows(manifest, expected):
    if type(manifest) is not dict or set(manifest) != {'schema_id', 'exact_payload_schema',
            'status', 'carrier_count', 'carriers', 'artifact_sha256'}:
        raise CertifiedCarrierReuseError('CARRIER_MANIFEST_FIELDS')
    if manifest['schema_id'] != 'CERTIFIED_CARRIER_SET_V1' or manifest['status'] != 'CERTIFIED_PASS':
        raise CertifiedCarrierReuseError('certified carrier manifest schema/status mismatch')
    count = manifest['carrier_count']
    if type(count) is not int or count <= 0:
        raise CertifiedCarrierReuseError('CARRIER_EXACT_POSITIVE_COUNT_REQUIRED')
    rows = manifest['carriers']
    if type(rows) is not list or len(rows) != count:
        raise CertifiedCarrierReuseError('certified carrier count mismatch')
    schema = _text(manifest['exact_payload_schema'], 'exact_payload_schema')
    if schema not in _BUNDLE_NAMES:
        raise CertifiedCarrierReuseError('unsupported exact carrier payload schema: ' + schema)
    claimed = _digest(manifest['artifact_sha256'], 'artifact_sha256')
    if claimed != canonical_sha256({k: v for k, v in manifest.items() if k != 'artifact_sha256'}):
        raise CertifiedCarrierReuseError('certified carrier manifest hash mismatch')
    ids = set()
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_FIELDS:
            raise CertifiedCarrierReuseError('CARRIER_ROW_FIELDS')
        cid = _text(row['carrier_id'], 'carrier_id')
        if cid in ids:
            raise CertifiedCarrierReuseError('CARRIER_DUPLICATE_ID')
        ids.add(cid)
        _identity(row)
        for field in _IDENTITY_FIELDS:
            if row[field] != expected[field]:
                raise CertifiedCarrierReuseError('carrier ' + cid + ': identity mismatch ' + field)
        if row['certification_status'] != 'CERTIFIED_PASS':
            raise CertifiedCarrierReuseError('CARRIER_ROW_CERTIFICATION_STATUS')
        digest = _digest(row['construction_digest'], 'construction_digest')
        if _digest(row['exact_carrier_root_digest'], 'exact_carrier_root_digest') != digest:
            raise CertifiedCarrierReuseError('CARRIER_ROOT_CONSTRUCTION_MISMATCH')
        if row['exact_carrier_payload'] != _BUNDLE_NAMES[schema]:
            raise CertifiedCarrierReuseError('CARRIER_PAYLOAD_NAME_MISMATCH')
        if _digest(row['public_state_sha256'], 'public_state_sha256') != canonical_sha256(row['public_state']):
            raise CertifiedCarrierReuseError('carrier ' + cid + ': public state digest mismatch')
    return rows


def _save_exact_state_bundle(root: Path, carriers: Mapping[str, tuple[Any, Any]]) -> tuple[str, str, dict[str, str]]:
    """Persist exact carriers using Decoder-native DAG when available.

    Regime-scanner O7/Lift states use the existing content-addressed maturation DAG.  A
    pickle fallback exists only for simple implementation carriers that are themselves safely
    pickleable; live IG regime states never take that fallback because their engine contains
    modules and therefore intentionally crosses the DAG boundary instead.
    """
    from .maturation_parallel import state_dag_wire

    ids = sorted(carriers)
    roots = [carriers[cid][0] for cid in ids]
    try:
        dag = state_dag_wire(roots)
    except TypeError:
        payload_name = "EXACT_STATE_PICKLE_V1.pkl"
        payload = {cid: carriers[cid][0] for cid in ids}
        try:
            (root / payload_name).write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:
            raise CertifiedCarrierReuseError(
                "exact carrier is neither Decoder DAG-serializable nor safely pickleable"
            ) from exc
        return "PYTHON_PICKLE_EXACT_STATE_V1", payload_name, {
            cid: str(getattr(carriers[cid][0], "construction_digest", "")) for cid in ids
        }
    payload_name = "EXACT_STATE_DAG_V1.json"
    write_json_atomic(root / payload_name, dag)
    return "IG_MATURATION_STATE_DAG_V1", payload_name, {
        cid: str(dag["roots"][i]) for i, cid in enumerate(ids)
    }


def _load_exact_state_bundle(root: Path, *, schema: str, payload_name: str,
                             expected_root_digests=None, expected_carrier_ids=None) -> dict[str, Any]:
    if type(schema) is not str or schema not in _BUNDLE_NAMES or payload_name != _BUNDLE_NAMES[schema]:
        raise CertifiedCarrierReuseError('CARRIER_PAYLOAD_NAME_OR_SCHEMA_MISMATCH')
    path = root / payload_name
    if not path.is_file():
        raise CertifiedCarrierReuseError(f"certified carrier payload missing: {path}")
    if schema == "IG_MATURATION_STATE_DAG_V1":
        from .maturation_parallel import _states_from_dag
        obj = json.loads(path.read_text(encoding="utf-8"))
        if expected_root_digests is not None and (type(obj) is not dict or obj.get('roots') != list(expected_root_digests)):
            raise CertifiedCarrierReuseError('CARRIER_ORDERED_ROOT_OCCURRENCE_CLOSURE')
        try:
            _engine, roots = _states_from_dag(obj)
        except Exception as exc:
            raise CertifiedCarrierReuseError(f"certified carrier DAG reconstruction failed: {exc}") from exc
        return {str(st.construction_digest): st for st in roots}
    if schema == "PYTHON_PICKLE_EXACT_STATE_V1":
        try:
            payload = pickle.loads(path.read_bytes())
        except Exception as exc:
            raise CertifiedCarrierReuseError(f"certified carrier pickle reconstruction failed: {exc}") from exc
        if type(payload) is not dict or (expected_carrier_ids is not None and
                set(payload) != set(expected_carrier_ids)):
            raise CertifiedCarrierReuseError('CARRIER_PICKLE_ID_CLOSURE')
        if expected_carrier_ids is not None and [str(getattr(payload[cid], 'construction_digest', ''))
                for cid in expected_carrier_ids] != list(expected_root_digests):
            raise CertifiedCarrierReuseError('CARRIER_PICKLE_ROOT_CLOSURE')
        by_digest = {}
        for state in payload.values():
            digest = _digest(getattr(state, 'construction_digest', None), 'construction_digest')
            prior = by_digest.get(digest)
            if prior is not None and prior is not state and pickle.dumps(prior, protocol=pickle.HIGHEST_PROTOCOL) != pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL):
                raise CertifiedCarrierReuseError('CARRIER_PICKLE_ALIAS_COLLISION_REVIEW_REQUIRED')
            by_digest[digest] = state
        return by_digest
    raise CertifiedCarrierReuseError(f"unsupported exact carrier payload schema: {schema}")


def save_certified_carrier_set(
    root: str | Path,
    *,
    carriers: Mapping[str, tuple[Any, Any]],
    source_sha256: str,
    registry_sha256: str,
    producer_experiment_id: str,
    authority_stage: str,
    authority_science_sha256: str,
    certification_status: str = "CERTIFIED_PASS",
) -> dict[str, Any]:
    """Persist exact certified carriers as execution/replay artifacts.

    ``carriers`` maps carrier_id to ``(exact_state, public_state_payload)``.  Exact regime-scanner
    states are written through the already-established content-addressed state DAG representation,
    never through ad-hoc object pickle.  Admission is fail-closed on every frozen identity field.
    """
    _identity(dict(source_sha256=source_sha256, registry_sha256=registry_sha256,
                   producer_experiment_id=producer_experiment_id, authority_stage=authority_stage,
                   authority_science_sha256=authority_science_sha256))
    _text(certification_status, 'certification_status')
    if not isinstance(carriers, Mapping) or not carriers:
        raise CertifiedCarrierReuseError('CARRIER_NONEMPTY_MAPPING_REQUIRED')
    for cid, pair in carriers.items():
        _text(cid, 'carrier_id')
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise CertifiedCarrierReuseError('CARRIER_STATE_PUBLIC_PAIR_REQUIRED')
        _digest(getattr(pair[0], 'construction_digest', None), 'construction_digest')
        canonical_sha256(pair[1])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload_schema, state_bundle_name, root_digest_by_id = _save_exact_state_bundle(root, carriers)

    rows: list[dict[str, Any]] = []
    for cid in sorted(carriers):
        state, public_state = carriers[cid]
        digest = str(getattr(state, "construction_digest", ""))
        if len(digest) != 64:
            raise CertifiedCarrierReuseError(f"carrier {cid}: missing exact construction digest")
        if root_digest_by_id.get(cid) != digest:
            raise CertifiedCarrierReuseError(f"carrier {cid}: exact DAG root digest mismatch during save")
        public_sha = canonical_sha256(public_state)
        rows.append({
            "carrier_id": str(cid),
            "public_state": public_state,
            "public_state_sha256": public_sha,
            "exact_carrier_payload": state_bundle_name,
            "exact_carrier_root_digest": digest,
            "construction_digest": digest,
            "source_sha256": str(source_sha256),
            "registry_sha256": str(registry_sha256),
            "authority_stage": str(authority_stage),
            "authority_science_sha256": str(authority_science_sha256),
            "producer_experiment_id": str(producer_experiment_id),
            "certification_status": str(certification_status),
        })
    manifest = {
        "schema_id": "CERTIFIED_CARRIER_SET_V1",
        "exact_payload_schema": payload_schema,
        "status": str(certification_status),
        "carrier_count": len(rows),
        "carriers": rows,
    }
    manifest["artifact_sha256"] = canonical_sha256(manifest)
    write_json_atomic(root / "CERTIFIED_CARRIER_SET_V1.json", manifest)
    return manifest


def load_certified_carrier_set(
    root: str | Path,
    *,
    expected_source_sha256: str,
    expected_registry_sha256: str,
    expected_producer_experiment_id: str,
    expected_authority_stage: str,
    expected_authority_science_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    path = root / "CERTIFIED_CARRIER_SET_V1.json"
    if not path.is_file():
        raise CertifiedCarrierReuseError(f"certified carrier manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = _identity(dict(source_sha256=expected_source_sha256,
        registry_sha256=expected_registry_sha256, producer_experiment_id=expected_producer_experiment_id,
        authority_stage=expected_authority_stage, authority_science_sha256=expected_authority_science_sha256))
    rows = _validated_rows(manifest, expected)
    bundle_names = {str(row.get("exact_carrier_payload")) for row in rows}
    if len(bundle_names) != 1:
        raise CertifiedCarrierReuseError("certified carrier manifest must reference one exact DAG bundle")
    states_by_digest = _load_exact_state_bundle(
        root, schema=manifest['exact_payload_schema'], payload_name=next(iter(bundle_names)),
        expected_root_digests=[r['construction_digest'] for r in rows],
        expected_carrier_ids=[r['carrier_id'] for r in rows],
    )

    loaded: dict[str, Any] = {}
    for row in rows:
        checks = {
            "source_sha256": expected_source_sha256,
            "registry_sha256": expected_registry_sha256,
            "producer_experiment_id": expected_producer_experiment_id,
            "authority_stage": expected_authority_stage,
            "authority_science_sha256": expected_authority_science_sha256,
            "certification_status": "CERTIFIED_PASS",
        }
        for key, expected in checks.items():
            if str(row.get(key)) != str(expected):
                raise CertifiedCarrierReuseError(f"carrier {row.get('carrier_id')}: identity mismatch {key}")
        digest = str(row.get("construction_digest", ""))
        if digest != str(row.get("exact_carrier_root_digest", "")):
            raise CertifiedCarrierReuseError(f"carrier {row.get('carrier_id')}: exact root/construction digest mismatch")
        state = states_by_digest.get(digest)
        if state is None:
            raise CertifiedCarrierReuseError(f"carrier {row.get('carrier_id')}: exact state root missing from DAG")
        if str(getattr(state, "construction_digest", "")) != digest:
            raise CertifiedCarrierReuseError(f"carrier {row.get('carrier_id')}: construction digest mismatch")
        if canonical_sha256(row.get("public_state")) != str(row.get("public_state_sha256")):
            raise CertifiedCarrierReuseError(f"carrier {row.get('carrier_id')}: public state digest mismatch")
        loaded[str(row["carrier_id"])] = state
    if len(loaded) != manifest['carrier_count']:
        raise CertifiedCarrierReuseError("certified carrier count mismatch")
    return loaded, manifest


__all__ = [
    "CertifiedCarrierReuseError",
    "save_certified_carrier_set",
    "load_certified_carrier_set",
]
