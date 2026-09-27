from __future__ import annotations

"""Unified content-addressed reference-data contract for replay.

P3 defines storage and validation only.  It does not execute science, compare a
historical result, or grant audit authority.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .replay_index_lock import replay_index_lock


RECORD_SCHEMA = "IG_REPLAY_REFERENCE_RECORD_V1"
MANIFEST_SCHEMA = "IG_REPLAY_REFERENCE_DATASET_MANIFEST_V1"
TRANSACTION_SCHEMA = "IG_REPLAY_REFERENCE_TRANSACTION_V1"
COMMIT_SCHEMA = "IG_REPLAY_REFERENCE_COMMIT_V1"
RECORD_TYPES = frozenset({
    "CANONICAL_OBJECT", "MECHANISM", "GRADUATION", "SRCF_EVIDENCE",
    "NEGATIVE_RESULT", "AUDIT_AUTHORIZATION", "EARNED_ALGEBRA",
    "DEPENDENCY_LINK", "GENERATION_RECIPE", "COMPARISON", "RESUME_FRONTIER",
})
PROVENANCE_CLASSES = frozenset({
    "DERIVED_THEOREM", "CERTIFIED_CONSTRUCTION",
    "FINITE_COMPUTATIONAL_OBSERVATION", "IMPORTED_ASSUMPTION",
    "EXTERNAL_AUDIT_DECISION", "UNEXPLAINED",
})
EPISTEMIC_STATUSES = frozenset({"HISTORICAL", "REPLAYED", "OPEN", "EXTERNAL_DECISION"})
SCIENCE_EXECUTION_STATUSES = frozenset({"NONE", "EXECUTED"})
EQUALITY_MODES = frozenset({
    "EXACT_BYTES", "CANONICAL_JSON", "SET", "MULTISET", "COUNT",
    "CERTIFIED_SEMANTIC_EQUIVALENCE", "NOT_APPLICABLE",
})
COMPRESSION_MODES = frozenset({"EXACT", "LOSSLESS_COMPRESSED", "LOSSY_OBSERVER", "NOT_APPLICABLE"})
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^IGRD/[A-Z0-9][A-Z0-9_.-]*(?:/[A-Z0-9][A-Z0-9_.-]*)+$")


class ReferenceDataError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceDataError(message)


def _hash(value: Any, field: str) -> str:
    _require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None, f"invalid {field}")
    return value


def _strings(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    _require(isinstance(value, list), f"{field} must be a list")
    _require(all(isinstance(item, str) and bool(item) for item in value),
             f"{field} must contain nonempty strings")
    _require(len(value) == len(set(value)), f"{field} contains duplicates")
    _require(not nonempty or bool(value), f"{field} must be nonempty")
    return list(value)


def _hash_rows(value: Any, field: str, *, nonempty: bool = False) -> list[dict[str, str]]:
    _require(isinstance(value, list), f"{field} must be a list")
    _require(not nonempty or bool(value), f"{field} must be nonempty")
    rows = []
    refs = set()
    for index, row in enumerate(value):
        _require(isinstance(row, Mapping) and set(row) == {"ref", "sha256"},
                 f"{field}[{index}] fields")
        ref = row["ref"]
        _require(isinstance(ref, str) and bool(ref) and ref not in refs, f"{field}[{index}] ref")
        refs.add(ref)
        rows.append({"ref": ref, "sha256": _hash(row["sha256"], f"{field}[{index}].sha256")})
    return rows


def _equality(value: Any, field: str) -> dict[str, Any]:
    required = {"mode", "cardinality_semantics", "compression", "observer", "certificate_sha256"}
    _require(isinstance(value, Mapping) and set(value) == required, f"{field} fields")
    mode = value["mode"]
    _require(mode in EQUALITY_MODES, f"{field}.mode")
    cardinality = value["cardinality_semantics"]
    _require(cardinality in {"SET", "MULTISET", "COUNT", "ORDERED", "NOT_APPLICABLE"},
             f"{field}.cardinality_semantics")
    if mode in {"SET", "MULTISET", "COUNT"}:
        _require(cardinality == mode, f"{field} cardinality disagrees with equality mode")
    compression = value["compression"]
    _require(compression in COMPRESSION_MODES, f"{field}.compression")
    _require(isinstance(value["observer"], str) and bool(value["observer"]), f"{field}.observer")
    certificate = value["certificate_sha256"]
    if mode == "CERTIFIED_SEMANTIC_EQUIVALENCE":
        _hash(certificate, f"{field}.certificate_sha256")
    else:
        _require(certificate is None, f"{field} certificate only allowed for certified equivalence")
    return deepcopy(dict(value))


TYPE_FIELDS = {
    "CANONICAL_OBJECT": {"object_identity", "carrier_schema", "canonical_bytes_sha256", "formation_provenance"},
    "MECHANISM": {"mechanism_id", "input_record_ids", "output_record_ids", "determinism", "implementation_hashes"},
    "GRADUATION": {"decision", "graduated_scope", "authorizes", "evidence_record_ids"},
    "SRCF_EVIDENCE": {"series", "obligation_id", "result_identity", "equality_contract", "evidence_mode", "outcome"},
    "NEGATIVE_RESULT": {"obligation_id", "tested_scope", "negative_statement", "witnesses", "does_not_establish"},
    "AUDIT_AUTHORIZATION": {"authorization_id", "obligation_ids", "decision", "authorized_scope", "limitations", "evidence_hashes"},
    "EARNED_ALGEBRA": {"statement_id", "statement", "strength", "statement_scope", "supporting_record_ids", "nonclaims"},
    "DEPENDENCY_LINK": {"from_record_id", "to_record_id", "relation", "required"},
    "GENERATION_RECIPE": {"recipe_id", "implementation_ref", "implementation_sha256", "input_record_ids", "parameters", "expected_record_types", "determinism_contract"},
    "COMPARISON": {"obligation_id", "historical_record_ids", "replay_record_ids", "equality_contract", "outcome", "qualification_ids"},
    "RESUME_FRONTIER": {"root_run_id", "manifest_dag_sha256", "runner_state_sha256", "completed_node_ids", "checkpoint_sha256_by_node", "next_node_id", "frontier_status"},
}


def _payload(record_type: str, value: Any) -> dict[str, Any]:
    fields = TYPE_FIELDS[record_type]
    _require(isinstance(value, Mapping) and set(value) == fields, f"{record_type} payload fields")
    p = deepcopy(dict(value))
    if record_type == "CANONICAL_OBJECT":
        _hash(p["canonical_bytes_sha256"], "canonical_bytes_sha256")
        for field in ("object_identity", "carrier_schema", "formation_provenance"):
            _require(isinstance(p[field], str) and bool(p[field]), field)
    elif record_type == "MECHANISM":
        _strings(p["input_record_ids"], "input_record_ids")
        _strings(p["output_record_ids"], "output_record_ids", nonempty=True)
        _require(all(ID_RE.fullmatch(item) is not None
                     for item in p["input_record_ids"] + p["output_record_ids"]),
                 "mechanism record ids")
        _require(p["determinism"] in {"DETERMINISTIC", "DECLARED_NONDETERMINISTIC"}, "determinism")
        _hash_rows(p["implementation_hashes"], "implementation_hashes", nonempty=True)
    elif record_type == "GRADUATION":
        _require(p["decision"] in {"GRADUATED", "NOT_GRADUATED", "CONDITIONAL"}, "graduation decision")
        _require(isinstance(p["graduated_scope"], Mapping) and bool(p["graduated_scope"]), "graduated_scope")
        _strings(p["authorizes"], "authorizes")
        _strings(p["evidence_record_ids"], "evidence_record_ids", nonempty=True)
    elif record_type == "SRCF_EVIDENCE":
        _require(p["series"] in {"S", "R", "C", "F"}, "series")
        _require(isinstance(p["obligation_id"], str) and bool(p["obligation_id"]), "obligation_id")
        _hash(p["result_identity"], "result_identity")
        p["equality_contract"] = _equality(p["equality_contract"], "equality_contract")
        _require(p["evidence_mode"] in {"FRESH_RECOMPUTE", "VERIFIED_RESTORED_BLOCK", "SOURCE_INTEGRITY_ONLY", "HISTORICAL_RESULT_ONLY"}, "evidence_mode")
        _require(p["outcome"] in {"REPRODUCED", "DISCREPANCY", "OPEN", "NEGATIVE_REPRODUCED"}, "evidence outcome")
    elif record_type == "NEGATIVE_RESULT":
        for field in ("obligation_id", "negative_statement"):
            _require(isinstance(p[field], str) and bool(p[field]), field)
        _require(isinstance(p["tested_scope"], Mapping) and bool(p["tested_scope"]), "tested_scope")
        _require(isinstance(p["witnesses"], list), "witnesses")
        _strings(p["does_not_establish"], "does_not_establish", nonempty=True)
    elif record_type == "AUDIT_AUTHORIZATION":
        _require(isinstance(p["authorization_id"], str) and bool(p["authorization_id"]), "authorization_id")
        _strings(p["obligation_ids"], "obligation_ids", nonempty=True)
        _require(p["decision"] in {"CONTINUE", "CERTIFY_AND_ADVANCE"}, "audit decision")
        _require(isinstance(p["authorized_scope"], Mapping) and bool(p["authorized_scope"]), "authorized_scope")
        _strings(p["limitations"], "limitations", nonempty=True)
        _hash_rows(p["evidence_hashes"], "evidence_hashes", nonempty=True)
    elif record_type == "EARNED_ALGEBRA":
        for field in ("statement_id", "statement"):
            _require(isinstance(p[field], str) and bool(p[field]), field)
        _require(p["strength"] in {"EXACT", "THEOREM_BACKED", "BOUNDED_EMPIRICAL", "CONDITIONAL"}, "strength")
        _require(isinstance(p["statement_scope"], Mapping) and bool(p["statement_scope"]), "statement_scope")
        _strings(p["supporting_record_ids"], "supporting_record_ids", nonempty=True)
        _strings(p["nonclaims"], "nonclaims", nonempty=True)
    elif record_type == "DEPENDENCY_LINK":
        for field in ("from_record_id", "to_record_id"):
            _require(isinstance(p[field], str) and ID_RE.fullmatch(p[field]) is not None, field)
        _require(p["from_record_id"] != p["to_record_id"], "dependency self-loop")
        _require(p["relation"] in {"REQUIRES", "PRODUCES", "AUTHORIZES", "COMPARES", "QUALIFIES"}, "dependency relation")
        _require(isinstance(p["required"], bool), "dependency required")
    elif record_type == "GENERATION_RECIPE":
        for field in ("recipe_id", "implementation_ref"):
            _require(isinstance(p[field], str) and bool(p[field]), field)
        _hash(p["implementation_sha256"], "implementation_sha256")
        _strings(p["input_record_ids"], "input_record_ids")
        _require(isinstance(p["parameters"], Mapping), "parameters")
        _require(isinstance(p["expected_record_types"], list) and bool(p["expected_record_types"])
                 and all(item in RECORD_TYPES for item in p["expected_record_types"]), "expected_record_types")
        _require(p["determinism_contract"] in {"BYTE_IDENTICAL", "CANONICAL_SEMANTIC_IDENTITY"}, "determinism_contract")
    elif record_type == "COMPARISON":
        _require(isinstance(p["obligation_id"], str) and bool(p["obligation_id"]), "obligation_id")
        _strings(p["historical_record_ids"], "historical_record_ids", nonempty=True)
        _strings(p["replay_record_ids"], "replay_record_ids", nonempty=True)
        p["equality_contract"] = _equality(p["equality_contract"], "equality_contract")
        _require(p["outcome"] in {"REPRODUCED", "QUALIFIED_REPRODUCTION", "MISMATCH", "EQUIVALENCE_NOT_CERTIFIED"}, "comparison outcome")
        _strings(p["qualification_ids"], "qualification_ids")
    else:
        for field in ("root_run_id", "next_node_id"):
            _require(isinstance(p[field], str) and bool(p[field]), field)
        _hash(p["manifest_dag_sha256"], "manifest_dag_sha256")
        _hash(p["runner_state_sha256"], "runner_state_sha256")
        _strings(p["completed_node_ids"], "completed_node_ids")
        checkpoints = p["checkpoint_sha256_by_node"]
        _require(isinstance(checkpoints, Mapping) and set(checkpoints) == set(p["completed_node_ids"]), "frontier checkpoint index")
        for node_id, digest in checkpoints.items():
            _require(isinstance(node_id, str) and bool(node_id), "frontier node id")
            _hash(digest, f"checkpoint {node_id}")
        _require(p["frontier_status"] in {"READY", "WAITING_FOR_EXTERNAL_AUDIT", "WAITING_FOR_NODE_EXECUTOR_BINDING", "COMPLETE"}, "frontier_status")
    return p


REFERENCE_FIELDS = {
    "CANONICAL_OBJECT": (),
    "MECHANISM": ("input_record_ids",),
    "GRADUATION": ("evidence_record_ids",),
    "SRCF_EVIDENCE": (),
    "NEGATIVE_RESULT": (),
    "AUDIT_AUTHORIZATION": (),
    "EARNED_ALGEBRA": ("supporting_record_ids",),
    "DEPENDENCY_LINK": ("from_record_id", "to_record_id"),
    "GENERATION_RECIPE": ("input_record_ids",),
    "COMPARISON": ("historical_record_ids", "replay_record_ids"),
    "RESUME_FRONTIER": (),
}


def seal_reference_record(record: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(record))
    value.pop("record_sha256", None)
    required = {
        "schema_id", "record_id", "record_type", "layer", "payload", "provenance",
        "scope", "nonclaims", "dependencies", "epistemic_status", "science_execution",
        "authority_effect",
    }
    _require(set(value) == required, "reference record fields")
    _require(value["schema_id"] == RECORD_SCHEMA, "reference record schema")
    _require(isinstance(value["record_id"], str) and ID_RE.fullmatch(value["record_id"]) is not None,
             "reference record id")
    record_type = value["record_type"]
    _require(record_type in RECORD_TYPES, "reference record type")
    _require(isinstance(value["layer"], str) and bool(value["layer"]), "reference layer")
    value["payload"] = _payload(record_type, value["payload"])
    provenance = value["provenance"]
    _require(isinstance(provenance, Mapping)
             and set(provenance) == {"classification", "status", "source_hashes", "explanation"},
             "provenance fields")
    _require(provenance["classification"] in PROVENANCE_CLASSES, "provenance classification")
    _require(provenance["status"] in {"PINNED", "OPEN"}, "provenance status")
    sources = _hash_rows(provenance["source_hashes"], "provenance.source_hashes")
    if provenance["classification"] == "UNEXPLAINED":
        _require(provenance["status"] == "OPEN", "unexplained provenance must remain open")
    else:
        _require(bool(sources), "explained provenance requires a pinned source")
    _require(isinstance(provenance["explanation"], str) and bool(provenance["explanation"]), "provenance explanation")
    value["provenance"] = dict(provenance, source_hashes=sources)
    _require(isinstance(value["scope"], Mapping) and bool(value["scope"]), "reference scope")
    value["nonclaims"] = _strings(value["nonclaims"], "nonclaims")
    value["dependencies"] = _strings(value["dependencies"], "dependencies")
    _require(all(ID_RE.fullmatch(item) is not None for item in value["dependencies"]), "dependency id")
    referenced = set()
    for field in REFERENCE_FIELDS[record_type]:
        raw = value["payload"][field]
        referenced.update(raw if isinstance(raw, list) else [raw])
    _require(referenced == set(value["dependencies"]), "declared dependencies disagree with payload references")
    _require(value["epistemic_status"] in EPISTEMIC_STATUSES, "epistemic_status")
    _require(value["science_execution"] in SCIENCE_EXECUTION_STATUSES, "science_execution")
    if provenance["classification"] == "UNEXPLAINED":
        _require(value["epistemic_status"] == "OPEN", "unexplained record cannot be certified")
    _require(value["authority_effect"] in {"NONE", "HISTORICAL_AUTHORITY_RECORDED", "EXTERNAL_DECISION_RECORDED"}, "authority_effect")
    if record_type == "AUDIT_AUTHORIZATION":
        _require(value["provenance"]["classification"] == "EXTERNAL_AUDIT_DECISION",
                 "audit authorization requires external audit provenance")
        _require(value["authority_effect"] == "EXTERNAL_DECISION_RECORDED",
                 "audit authorization authority effect")
    else:
        _require(value["authority_effect"] != "EXTERNAL_DECISION_RECORDED",
                 "only audit authorization may record an external decision")
    value["record_sha256"] = canonical_sha256(value)
    return value


def verify_reference_record(record: Mapping[str, Any]) -> dict[str, Any]:
    expected = record.get("record_sha256")
    _hash(expected, "record_sha256")
    sealed = seal_reference_record(record)
    _require(sealed["record_sha256"] == expected, "reference record hash mismatch")
    return sealed


def empty_manifest() -> dict[str, Any]:
    value = {
        "schema_id": MANIFEST_SCHEMA,
        "contract_version": "1.0.0",
        "record_ids": [],
        "record_sha256_by_id": {},
        "record_type_counts": {name: 0 for name in sorted(RECORD_TYPES)},
        "status": "EMPTY",
        "science_executed": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


class ReplayReferenceDataStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.recover()

    @classmethod
    def initialize(cls, root: Path) -> "ReplayReferenceDataStore":
        root = Path(root)
        path = root / "MANIFEST.json"
        with replay_index_lock(root, ReferenceDataError):
            if not path.exists():
                write_json_atomic(path, empty_manifest())
        return cls(root)

    def _load_manifest(self) -> dict[str, Any]:
        path = self.root / "MANIFEST.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceDataError(f"reference manifest unavailable: {exc}") from exc
        _require(isinstance(value, dict), "reference manifest object")
        return value

    def _verify_manifest(self, manifest: Mapping[str, Any]) -> None:
        fields = {"schema_id", "contract_version", "record_ids", "record_sha256_by_id", "record_type_counts", "status", "science_executed", "manifest_sha256"}
        _require(set(manifest) == fields and manifest["schema_id"] == MANIFEST_SCHEMA, "reference manifest fields")
        _require(manifest["contract_version"] == "1.0.0", "reference contract version")
        ids = _strings(manifest["record_ids"], "manifest record_ids")
        _require(isinstance(manifest["record_sha256_by_id"], Mapping)
                 and set(manifest["record_sha256_by_id"]) == set(ids), "manifest record index")
        for record_id, digest in manifest["record_sha256_by_id"].items():
            _require(ID_RE.fullmatch(record_id) is not None, "manifest record id")
            _hash(digest, f"manifest record {record_id}")
        counts = manifest["record_type_counts"]
        _require(isinstance(counts, Mapping) and set(counts) == set(RECORD_TYPES)
                 and all(isinstance(v, int) and v >= 0 for v in counts.values())
                 and sum(counts.values()) == len(ids), "manifest type counts")
        _require(manifest["status"] in {"EMPTY", "POPULATED"}, "manifest status")
        _require(manifest["status"] == ("EMPTY" if not ids else "POPULATED"), "manifest status disagrees with records")
        _require(isinstance(manifest["science_executed"], bool), "manifest science_executed")
        _require(manifest["manifest_sha256"] == canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}),
                 "reference manifest hash mismatch")

    def verify(self) -> None:
        with replay_index_lock(self.root, ReferenceDataError):
            self.manifest = self._load_manifest()
            self._verify()
            self._committed_transactions()
            self._verify_record_inventory()

    def _verify_record_inventory(self) -> None:
        retained = {path.stem for path in (self.root / "records").glob("*.json")}
        _require(retained == set(self.manifest["record_sha256_by_id"].values()),
                 "REFERENCE_INDEX_ROLLBACK_OR_INTERRUPTED_PUT: retained record missing from index")

    def _verify(self) -> None:
        self._verify_manifest(self.manifest)
        observed = {name: 0 for name in RECORD_TYPES}
        observed_science_execution = False
        available = set()
        for record_id in self.manifest["record_ids"]:
            digest = self.manifest["record_sha256_by_id"][record_id]
            path = self.root / "records" / f"{digest}.json"
            try:
                record = verify_reference_record(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ReferenceDataError(f"reference record unavailable: {record_id}: {exc}") from exc
            _require(record["record_id"] == record_id and record["record_sha256"] == digest,
                     f"reference record binding mismatch: {record_id}")
            _require(set(record["dependencies"]).issubset(available),
                     f"reference dependency is missing or forward: {record_id}")
            available.add(record_id)
            observed[record["record_type"]] += 1
            observed_science_execution |= record["science_execution"] == "EXECUTED"
        _require(observed == dict(self.manifest["record_type_counts"]), "manifest type counts mismatch")
        _require(observed_science_execution == self.manifest["science_executed"],
                 "manifest science execution mismatch")

    def _proposed_manifest(self, record: Mapping[str, Any]) -> dict[str, Any]:
        value = deepcopy(self.manifest)
        value["record_ids"].append(record["record_id"])
        value["record_sha256_by_id"][record["record_id"]] = record["record_sha256"]
        value["record_type_counts"][record["record_type"]] += 1
        value["status"] = "POPULATED"
        value["science_executed"] |= record["science_execution"] == "EXECUTED"
        value["manifest_sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "manifest_sha256"})
        return value

    def put(self, record: Mapping[str, Any], *, _interrupt_after_record: bool = False) -> dict[str, Any]:
        with replay_index_lock(self.root, ReferenceDataError):
            self.manifest = self._load_manifest()
            self._recover()
            return self._put(record, _interrupt_after_record=_interrupt_after_record)

    def _put(self, record: Mapping[str, Any], *, _interrupt_after_record: bool) -> dict[str, Any]:
        sealed = verify_reference_record(record) if "record_sha256" in record else seal_reference_record(record)
        record_id = sealed["record_id"]
        index = self.manifest["record_sha256_by_id"]
        if record_id in index:
            _require(index[record_id] == sealed["record_sha256"], f"reference record id collision: {record_id}")
            return sealed
        _require(set(sealed["dependencies"]).issubset(index), f"missing reference dependency: {record_id}")
        proposed = self._proposed_manifest(sealed)
        transaction = {
            "schema_id": TRANSACTION_SCHEMA,
            "parent_manifest_sha256": self.manifest["manifest_sha256"],
            "proposed_manifest_sha256": proposed["manifest_sha256"],
            "record_id": record_id,
            "record_sha256": sealed["record_sha256"],
        }
        transaction["transaction_sha256"] = canonical_sha256(transaction)
        tx = transaction["transaction_sha256"]
        pending = self.root / "transactions" / f"{tx}.json"
        record_path = self.root / "records" / f"{sealed['record_sha256']}.json"
        if pending.exists():
            _require(json.loads(pending.read_text()) == transaction, "transaction collision")
        else:
            write_json_atomic(pending, transaction)
        if record_path.exists():
            _require(json.loads(record_path.read_text()) == sealed, "immutable reference record collision")
        else:
            write_json_atomic(record_path, sealed)
        if _interrupt_after_record:
            raise ReferenceDataError("INJECTED_INTERRUPTION_AFTER_RECORD")
        write_json_atomic(self.root / "MANIFEST.json", proposed)
        self.manifest = proposed
        self._commit(transaction)
        self._verify()
        self._verify_record_inventory()
        return sealed

    def _commit(self, transaction: Mapping[str, Any]) -> None:
        commit = {
            "schema_id": COMMIT_SCHEMA,
            "transaction_sha256": transaction["transaction_sha256"],
            "manifest_sha256": transaction["proposed_manifest_sha256"],
            "record_id": transaction["record_id"],
            "record_sha256": transaction["record_sha256"],
        }
        commit["commit_sha256"] = canonical_sha256(commit)
        path = self.root / "commits" / f"{commit['commit_sha256']}.json"
        if path.exists():
            _require(json.loads(path.read_text()) == commit, "commit collision")
        else:
            write_json_atomic(path, commit)

    def _transaction(self, path: Path) -> dict[str, Any]:
        tx = json.loads(path.read_text())
        fields = {"schema_id", "parent_manifest_sha256", "proposed_manifest_sha256",
                  "record_id", "record_sha256", "transaction_sha256"}
        _require(set(tx) == fields and tx["schema_id"] == TRANSACTION_SCHEMA,
                 "transaction fields")
        expected = canonical_sha256({k: v for k, v in tx.items() if k != "transaction_sha256"})
        _require(tx["transaction_sha256"] == expected and path.stem == expected,
                 "transaction hash mismatch")
        return tx

    def _committed_transactions(self) -> set[str]:
        """A valid old manifest must not hide retained committed records."""
        committed_transactions = set()
        for path in sorted((self.root / "commits").glob("*.json")):
            commit = json.loads(path.read_text())
            fields = {"schema_id", "transaction_sha256", "manifest_sha256",
                      "record_id", "record_sha256", "commit_sha256"}
            _require(set(commit) == fields and commit["schema_id"] == COMMIT_SCHEMA,
                     "commit fields")
            digest = canonical_sha256({k: v for k, v in commit.items() if k != "commit_sha256"})
            _require(commit["commit_sha256"] == digest and path.stem == digest, "commit hash mismatch")
            tx_digest = _hash(commit["transaction_sha256"], "commit transaction")
            tx = self._transaction(self.root / "transactions" / f"{tx_digest}.json")
            _require(commit["record_id"] == tx["record_id"]
                     and commit["record_sha256"] == tx["record_sha256"]
                     and commit["manifest_sha256"] == tx["proposed_manifest_sha256"],
                     "commit transaction binding mismatch")
            _require(self.manifest["record_sha256_by_id"].get(commit["record_id"])
                     == commit["record_sha256"],
                     "REFERENCE_INDEX_ROLLBACK: committed record absent or changed")
            committed_transactions.add(commit["transaction_sha256"])
        return committed_transactions

    def recover(self) -> None:
        with replay_index_lock(self.root, ReferenceDataError):
            self.manifest = self._load_manifest()
            self._recover()

    def _recover(self) -> None:
        self._verify()
        committed_transactions = self._committed_transactions()
        transactions = self.root / "transactions"
        pending = []
        for path in sorted(transactions.glob("*.json")):
            tx = self._transaction(path)
            if tx["transaction_sha256"] not in committed_transactions:
                pending.append(tx)
        _require(len(pending) <= 1, "ambiguous pending reference transactions")
        for tx in pending:
            record_path = self.root / "records" / f"{tx['record_sha256']}.json"
            record = verify_reference_record(json.loads(record_path.read_text()))
            _require(record["record_id"] == tx["record_id"]
                     and record["record_sha256"] == tx["record_sha256"], "transaction record mismatch")
            _require(set(record["dependencies"]).issubset(self.manifest["record_sha256_by_id"]),
                     "transaction dependency missing")
            if self.manifest["manifest_sha256"] == tx["parent_manifest_sha256"]:
                proposed = self._proposed_manifest(record)
                _require(proposed["manifest_sha256"] == tx["proposed_manifest_sha256"], "transaction proposed manifest mismatch")
                write_json_atomic(self.root / "MANIFEST.json", proposed)
                self.manifest = proposed
            else:
                _require(self.manifest["manifest_sha256"] == tx["proposed_manifest_sha256"]
                         and self.manifest["record_sha256_by_id"].get(tx["record_id"]) == tx["record_sha256"],
                         "unrecoverable transaction parent mismatch")
            self._commit(tx)
        self._verify()
        self._verify_record_inventory()
