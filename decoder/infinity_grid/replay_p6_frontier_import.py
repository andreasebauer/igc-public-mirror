from __future__ import annotations

"""Fail-closed import of the preserved P6 L2J3 replay frontier."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

from .canon import canonical_sha256, write_json_atomic
from .replay_l2j3_executor import L2J3_NODE_IDS
from .replay_dag_runner import CHECKPOINT_SCHEMA, ReplayDagRunner
from .replay_p5_frontier_import import P5_COMPLETED_NODE_IDS
from .replay_reference_data import ReplayReferenceDataStore, verify_reference_record


P6_CAPTURE_ID = "03ba469820dcb80d0e917205812979e05408dbcbeff3e10657e8be35f1f9d39c"
P6_STATE_OBJECT_SHA256 = "d420f10ce602e6d831603bc6218fe1bf7c39df7c5e9ddf9344f12fffd13a9e5b"
P6_FRONTIER_RECORD_SHA256 = "1aacb998edd5ddd38085dfca6b51c78adb6fcabcbf0055e3c1aa8a474a64a9cb"
P6_COMPLETED_NODE_IDS = (*P5_COMPLETED_NODE_IDS, *L2J3_NODE_IDS, "IG/L2J3/GATE/CERTIFY")
STAGE_SUFFIX = "/chain/decoder_stage_runtime/REPLAY__L0_TO_G8__ROOT/"


class P6FrontierImportError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P6FrontierImportError(message)


def _json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P6FrontierImportError(f"invalid P6 JSON: {name}") from exc
    _require(isinstance(value, dict), f"P6 JSON is not an object: {name}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != field})


def import_p6_frontier(*, state_zip: Path, expected_file_sha256: str,
                       manifest: Mapping[str, Any], state_root: Path,
                       reference_root: Path, root_run_id: str,
                       dataset_root_sha256: str) -> tuple[ReplayDagRunner, ReplayReferenceDataStore]:
    state_zip = Path(state_zip)
    _require(state_zip.is_file(), "P6 state object unavailable")
    actual_sha = hashlib.sha256(state_zip.read_bytes()).hexdigest()
    _require(actual_sha == expected_file_sha256 == P6_STATE_OBJECT_SHA256,
             "P6 state object hash mismatch")
    _require(not (Path(state_root) / "runner_state.json").exists(),
             "P6 frontier import requires an empty runner destination")
    _require(not (Path(reference_root) / "MANIFEST.json").exists(),
             "P6 frontier import requires an empty reference destination")

    with ZipFile(state_zip) as archive:
        names = archive.namelist()
        _require(len(names) <= 512, "P6 state object member limit exceeded")
        _require(sum(info.file_size for info in archive.infolist()) <= 6_000_000,
                 "P6 state object uncompressed-size limit exceeded")
        capture = _json(archive.read("CAPTURE.json"), "CAPTURE.json")
        _require(capture.get("capture_id") == P6_CAPTURE_ID, "P6 capture identity mismatch")
        prefixes = sorted({name.split(STAGE_SUFFIX, 1)[0] + STAGE_SUFFIX
                           for name in names if STAGE_SUFFIX in name})
        _require(len(prefixes) == 1, "P6 state object must contain exactly one replay stage")
        prefix = prefixes[0]
        status = _json(archive.read(prefix + "artifacts/REPLAY_ROOT_STATUS.json"), "P6 status")
        state = _json(archive.read(prefix + "replay_runner/runner_state.json"), "P6 runner state")
        reference_manifest = _json(
            archive.read(prefix + "replay_reference_data/MANIFEST.json"), "P6 reference manifest")
        _require(status.get("outcome") == "L2J3_HISTORICAL_PILOT_COMPLETE",
                 "P6 frontier is not complete")
        _require(status.get("science_executed") is False, "P6 frontier incorrectly claims fresh science")
        _require(status.get("runner_state_sha256") == state.get("state_sha256"),
                 "P6 status/runner binding mismatch")
        _require(status.get("reference_manifest_sha256") == reference_manifest.get("manifest_sha256"),
                 "P6 status/reference binding mismatch")
        _require(status.get("frontier_record_sha256") == P6_FRONTIER_RECORD_SHA256,
                 "P6 frontier record identity mismatch")
        _require(state.get("state_sha256") == _self_hash(state, "state_sha256"),
                 "P6 runner state hash mismatch")
        _require(state.get("manifest_sha256") == manifest.get("dag_sha256"),
                 "P6 frontier belongs to another replay manifest")
        _require(state.get("root_run_id") == root_run_id, "P6 root run identity mismatch")
        _require(state.get("dataset_root") == {"state": "EMPTY", "sha256": dataset_root_sha256},
                 "P6 dataset-root identity mismatch")
        _require(tuple(state.get("completed_node_ids", ())) == P6_COMPLETED_NODE_IDS,
                 "P6 completed frontier is not exactly L0, C0_HISTORICAL, and L2J3")
        _require(state.get("active_audit_capsule_sha256") is None,
                 "P6 frontier has an unresolved audit capsule")

        checkpoints: list[tuple[str, dict[str, Any]]] = []
        for node_id in P6_COMPLETED_NODE_IDS:
            digest = state["accepted_checkpoint_sha256_by_node"][node_id]
            name = prefix + f"replay_runner/checkpoints/{digest}.json"
            checkpoint = _json(archive.read(name), name)
            _require(checkpoint.get("schema_id") == CHECKPOINT_SCHEMA, "P6 checkpoint schema mismatch")
            _require(checkpoint.get("checkpoint_sha256") == digest == _self_hash(checkpoint, "checkpoint_sha256"),
                     f"P6 checkpoint hash mismatch: {node_id}")
            _require(checkpoint.get("node_id") == node_id, f"P6 checkpoint node mismatch: {node_id}")
            checkpoints.append((digest, checkpoint))

        records: list[dict[str, Any]] = []
        for record_id in reference_manifest.get("record_ids", []):
            digest = reference_manifest["record_sha256_by_id"][record_id]
            name = prefix + f"replay_reference_data/records/{digest}.json"
            record = verify_reference_record(_json(archive.read(name), name))
            _require(record["record_id"] == record_id and record["record_sha256"] == digest,
                     f"P6 reference binding mismatch: {record_id}")
            records.append(record)
        _require(P6_FRONTIER_RECORD_SHA256 in reference_manifest.get("record_sha256_by_id", {}).values(),
                 "P6 resume-frontier record missing")

    state_root = Path(state_root)
    for digest, checkpoint in checkpoints:
        write_json_atomic(state_root / "checkpoints" / f"{digest}.json", checkpoint)
    write_json_atomic(state_root / "runner_state.json", state)
    reference_root = Path(reference_root)
    for record in records:
        write_json_atomic(reference_root / "records" / f"{record['record_sha256']}.json", record)
    write_json_atomic(reference_root / "MANIFEST.json", reference_manifest)
    runner = ReplayDagRunner.resume(manifest, state_root)
    store = ReplayReferenceDataStore(reference_root)
    action = runner.next_action()
    _require(action.get("node_id") == "IG/NODE_IN/S/ASSERTION_MAPPED_STRUCTURE",
             "imported P6 frontier does not resume at NODE_IN/S")
    return runner, store
