"""Semantic selection of certified test inputs for Decoder 0.7.

Tests state what data they need; they do not pin the Decoder or test version
that produced the data.  Admission follows certified bytes and declared
meaning, schema, domain, coverage, invariants, and control role.  Producer
versions remain diagnostic provenance only.

The resolver chooses the fewest certified blocks that satisfy all required
needs, then the fewest total bytes, then the lexicographically smallest block
ID list.  That makes selection deterministic without a growing version graph.

Capture inputs have two disjoint classes:

* project inputs are scientific/business data supplied by the job;
* administrative inputs are supplied by Decoder or its runtime environment.

``submission_contract`` and ``runtime_dependency_snapshot`` are reserved
administrative names.  They must never be reported as missing project data.
Existing captures need no rewrite: classification is derived when read.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import re

from .canon import canonical_sha256
from . import certified_base as cb

TEST_NEEDS = "IG_DECODER_TEST_NEEDS_V1"
INPUT_SELECTION = "IG_DECODER_TEST_INPUT_SELECTION_V1"
RUN_RECEIPT = "IG_DECODER_RUN_RECEIPT_V1"
ADMINISTRATIVE_INPUTS = frozenset({"submission_contract", "runtime_dependency_snapshot"})
CONTROL_CATEGORIES = {
    "DATA": frozenset(cb.CATEGORIES - {"positive_controls", "negative_controls"}),
    "POSITIVE_CONTROL": frozenset({"positive_controls"}),
    "NEGATIVE_CONTROL": frozenset({"negative_controls"}),
}
HASH = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


class SemanticInputError(RuntimeError):
    """Stable refusal carrying a machine-readable code."""

    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}:{detail}")


def _token(value, field):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise SemanticInputError("TEST_NEEDS_TOKEN_INVALID", field)
    return value


def _texts(values, field, *, allow_empty=True):
    if not isinstance(values, list) or (not allow_empty and not values):
        raise SemanticInputError("TEST_NEEDS_LIST_REQUIRED", field)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise SemanticInputError("TEST_NEEDS_LIST_INVALID", field)
    cleaned = sorted(set(value.strip() for value in values))
    if len(cleaned) != len(values):
        raise SemanticInputError("TEST_NEEDS_LIST_DUPLICATE", field)
    return cleaned


def _sealed(row):
    result = dict(row)
    result["record_sha256"] = canonical_sha256(result)
    return result


def _verify(row, schema):
    if not isinstance(row, dict) or row.get("schema_id") != schema:
        raise SemanticInputError("SEMANTIC_RECORD_SCHEMA_INVALID", schema)
    digest = row.get("record_sha256")
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    if not isinstance(digest, str) or digest != canonical_sha256(body):
        raise SemanticInputError("SEMANTIC_RECORD_HASH_INVALID", schema)
    return row


def _read(path, schema):
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise SemanticInputError("SEMANTIC_RECORD_UNREADABLE", str(path)) from exc
    return _verify(row, schema)


def _write_immutable(path, row):
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise SemanticInputError("SEMANTIC_IMMUTABLE_CONFLICT", str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def create_test_needs(*, test_id, requirements):
    """Normalize and seal one test's semantic input contract.

    Optional needs are informative only.  If a test consumes an input, that
    input must be declared ``required``.  This keeps selection minimal and
    prevents optional data from silently changing a test's result.
    """
    if not isinstance(requirements, list) or not requirements:
        raise SemanticInputError("TEST_NEEDS_REQUIREMENTS_REQUIRED")
    normalized = []
    seen = set()
    fields = {"need_id", "schema_id", "meaning", "domain", "required_capabilities",
              "required_coverage", "required_invariants", "control_role", "required"}
    for row in requirements:
        if not isinstance(row, dict) or set(row) != fields:
            raise SemanticInputError("TEST_NEEDS_FIELDS_INVALID")
        need_id = _token(row["need_id"], "need_id")
        if need_id in seen:
            raise SemanticInputError("TEST_NEEDS_DUPLICATE", need_id)
        seen.add(need_id)
        role = row["control_role"]
        if role not in CONTROL_CATEGORIES:
            raise SemanticInputError("TEST_NEEDS_CONTROL_ROLE_INVALID", str(role))
        if type(row["required"]) is not bool:
            raise SemanticInputError("TEST_NEEDS_REQUIRED_INVALID", need_id)
        normalized.append({
            "need_id": need_id,
            "schema_id": _token(row["schema_id"], "schema_id"),
            "meaning": _token(row["meaning"], "meaning"),
            "domain": _token(row["domain"], "domain"),
            "required_capabilities": _texts(row["required_capabilities"], "required_capabilities"),
            "required_coverage": _texts(row["required_coverage"], "required_coverage"),
            "required_invariants": _texts(row["required_invariants"], "required_invariants"),
            "control_role": role,
            "required": row["required"],
        })
    normalized.sort(key=lambda item: item["need_id"])
    return _sealed({
        "schema_id": TEST_NEEDS,
        "test_id": _token(test_id, "test_id"),
        "requirements": normalized,
        "selection_rule": "MIN_BLOCK_COUNT_THEN_BYTES_THEN_BLOCK_IDS",
        "producer_versions_are_diagnostic_only": True,
        "optional_inputs_must_not_affect_result": True,
    })


def _eligible(block, need):
    required_coverage = set(need["required_coverage"]) | set(need["required_capabilities"])
    return (
        block["category"] in CONTROL_CATEGORIES[need["control_role"]]
        and block["payload_schema_id"] == need["schema_id"]
        and block["meaning"] == need["meaning"]
        and block["domain"] == need["domain"]
        and required_coverage <= set(block["coverage"])
        and set(need["required_invariants"]) <= set(block["invariants"])
    )


def resolve_test_inputs(store, base_id, needs, *, save=True):
    """Select the deterministic smallest certified subset for required needs."""
    needs = _verify(dict(needs), TEST_NEEDS)
    try:
        base = cb.load_base(store, base_id)
    except cb.CertifiedBaseError as exc:
        raise SemanticInputError(exc.code, exc.detail) from exc
    blocks = []
    for entry in base["blocks"]:
        try:
            block = cb.load_block(store, entry["block_id"])
            cert = cb.certification(store, entry["block_id"])
        except cb.CertifiedBaseError as exc:
            raise SemanticInputError(exc.code, exc.detail) from exc
        blocks.append((entry, block, cert))

    required = [row for row in needs["requirements"] if row["required"]]
    optional = [row for row in needs["requirements"] if not row["required"]]
    if not required:
        raise SemanticInputError("TEST_NEEDS_NO_REQUIRED_INPUTS")
    need_index = {row["need_id"]: index for index, row in enumerate(required)}
    coverage = {}
    for entry, block, cert in blocks:
        matched = sorted(row["need_id"] for row in required if _eligible(block, row))
        if matched:
            coverage[entry["block_id"]] = {
                "entry": entry, "block": block, "cert": cert, "needs": matched,
            }
    for row in required:
        if not any(row["need_id"] in item["needs"] for item in coverage.values()):
            raise SemanticInputError("TEST_INPUT_REQUIRED_NEED_UNSATISFIED", row["need_id"])

    full = (1 << len(required)) - 1
    # mask -> (count, bytes, tuple(block ids))
    best = {0: (0, 0, ())}
    for block_id, item in sorted(coverage.items()):
        mask = 0
        for need_id in item["needs"]:
            mask |= 1 << need_index[need_id]
        current = list(best.items())
        for prior_mask, score in current:
            combined = prior_mask | mask
            ids = tuple(sorted(score[2] + (block_id,)))
            candidate = (len(ids), score[1] + item["block"]["size_bytes"], ids)
            if combined not in best or candidate < best[combined]:
                best[combined] = candidate
    if full not in best:
        raise SemanticInputError("TEST_INPUT_SELECTION_FAILED")

    selected_ids = list(best[full][2])
    selected = []
    satisfied_by = {}
    for block_id in selected_ids:
        item = coverage[block_id]
        selected.append({
            "block_id": block_id,
            "content_sha256": item["block"]["content_sha256"],
            "size_bytes": item["block"]["size_bytes"],
            "certification_id": item["cert"]["record_sha256"],
            "category": item["block"]["category"],
        })
        for need_id in item["needs"]:
            satisfied_by.setdefault(need_id, []).append(block_id)
    optional_available = {
        row["need_id"]: sorted(entry["block_id"] for entry, block, _ in blocks if _eligible(block, row))
        for row in optional
    }
    selection = _sealed({
        "schema_id": INPUT_SELECTION,
        "test_needs_sha256": needs["record_sha256"],
        "test_id": needs["test_id"],
        "base_id": base_id,
        "base_materialization_sha256": base["materialization_sha256"],
        "selected_blocks": selected,
        "satisfied_by": {key: sorted(value) for key, value in sorted(satisfied_by.items())},
        "optional_available_not_selected": optional_available,
        "selection_metrics": {"block_count": len(selected), "total_size_bytes": best[full][1]},
        "selection_rule": needs["selection_rule"],
        "producer_versions_used_for_admission": False,
    })
    if save:
        _write_immutable(Path(store) / "selections" / (selection["record_sha256"] + ".json"), selection)
    return selection


def load_selection(store, selection_id, *, verify_blocks=True):
    if not isinstance(selection_id, str) or not HASH.fullmatch(selection_id):
        raise SemanticInputError("TEST_INPUT_SELECTION_ID_INVALID")
    row = _read(Path(store) / "selections" / (selection_id + ".json"), INPUT_SELECTION)
    if row["record_sha256"] != selection_id:
        raise SemanticInputError("TEST_INPUT_SELECTION_ID_MISMATCH")
    if verify_blocks:
        try:
            base = cb.load_base(store, row["base_id"])
        except cb.CertifiedBaseError as exc:
            raise SemanticInputError(exc.code, exc.detail) from exc
        if base["materialization_sha256"] != row["base_materialization_sha256"]:
            raise SemanticInputError("TEST_INPUT_BASE_BINDING_INVALID")
        base_ids = {item["block_id"] for item in base["blocks"]}
        for item in row["selected_blocks"]:
            if item["block_id"] not in base_ids:
                raise SemanticInputError("TEST_INPUT_BLOCK_NOT_IN_BASE", item["block_id"])
            try:
                block = cb.load_block(store, item["block_id"])
                cert = cb.certification(store, item["block_id"])
                if cb.revocations(store, item["block_id"]):
                    raise SemanticInputError("DATA_BLOCK_REVOKED", item["block_id"])
            except cb.CertifiedBaseError as exc:
                raise SemanticInputError(exc.code, exc.detail) from exc
            if (block["content_sha256"] != item["content_sha256"] or
                    block["size_bytes"] != item["size_bytes"] or
                    cert["record_sha256"] != item["certification_id"] or
                    block["category"] != item["category"]):
                raise SemanticInputError("TEST_INPUT_BLOCK_BINDING_INVALID", item["block_id"])
    return row


def record_run(store, *, selection_id, test_code_sha256, decoder_code_sha256,
               parameters, result):
    """Record exactly which certified bytes and code produced one result."""
    for value, field in ((test_code_sha256, "test_code_sha256"),
                         (decoder_code_sha256, "decoder_code_sha256")):
        if not isinstance(value, str) or not HASH.fullmatch(value):
            raise SemanticInputError("RUN_RECEIPT_HASH_INVALID", field)
    selection = load_selection(store, selection_id)
    # Ensure both values are canonical JSON before sealing the receipt.
    try:
        canonical_sha256(parameters)
        canonical_sha256(result)
    except Exception as exc:
        raise SemanticInputError("RUN_RECEIPT_VALUE_INVALID") from exc
    receipt = _sealed({
        "schema_id": RUN_RECEIPT,
        "test_id": selection["test_id"],
        "selection_id": selection_id,
        "base_id": selection["base_id"],
        "selected_blocks": [
            {key: item[key] for key in ("block_id", "content_sha256", "size_bytes")}
            for item in selection["selected_blocks"]
        ],
        "test_code_sha256": test_code_sha256,
        "decoder_code_sha256": decoder_code_sha256,
        "parameters": parameters,
        "result": result,
    })
    _write_immutable(Path(store) / "run_receipts" / (receipt["record_sha256"] + ".json"), receipt)
    return receipt


def verify_run_receipt(store, receipt_id, *, expected_test_code_sha256=None,
                       expected_decoder_code_sha256=None):
    if not isinstance(receipt_id, str) or not HASH.fullmatch(receipt_id):
        raise SemanticInputError("RUN_RECEIPT_ID_INVALID")
    row = _read(Path(store) / "run_receipts" / (receipt_id + ".json"), RUN_RECEIPT)
    if row["record_sha256"] != receipt_id:
        raise SemanticInputError("RUN_RECEIPT_ID_MISMATCH")
    selection = load_selection(store, row["selection_id"])
    expected = [{key: item[key] for key in ("block_id", "content_sha256", "size_bytes")}
                for item in selection["selected_blocks"]]
    if (row["base_id"] != selection["base_id"] or row["test_id"] != selection["test_id"] or
            row["selected_blocks"] != expected):
        raise SemanticInputError("RUN_RECEIPT_SELECTION_BINDING_INVALID")
    if expected_test_code_sha256 is not None and row["test_code_sha256"] != expected_test_code_sha256:
        raise SemanticInputError("RUN_RECEIPT_TEST_CODE_MISMATCH")
    if expected_decoder_code_sha256 is not None and row["decoder_code_sha256"] != expected_decoder_code_sha256:
        raise SemanticInputError("RUN_RECEIPT_DECODER_CODE_MISMATCH")
    return row


def classify_capture_inputs(capture_record):
    """Return project and administrative inputs without rewriting a capture."""
    try:
        rows = capture_record["job"]["input_artifacts"]
        environment_names = {row["logical_name"] for row in capture_record["environment"]["artifacts"]}
    except (KeyError, TypeError) as exc:
        raise SemanticInputError("CAPTURE_INPUT_CLASSIFICATION_INVALID") from exc
    project = []
    administrative = []
    seen = set()
    for row in rows:
        name = row.get("logical_name") if isinstance(row, dict) else None
        if not isinstance(name, str) or name in seen:
            raise SemanticInputError("CAPTURE_INPUT_CLASSIFICATION_INVALID", str(name))
        seen.add(name)
        item = dict(row)
        item["input_class"] = "ADMINISTRATIVE" if name in ADMINISTRATIVE_INPUTS | environment_names else "PROJECT"
        (administrative if item["input_class"] == "ADMINISTRATIVE" else project).append(item)
    return {"project_inputs": project, "administrative_inputs": administrative}


def validate_project_inputs(capture_record, required_project_names, *, reject_unexpected=False):
    """Validate only project inputs; administrative inputs are out of scope."""
    required = _texts(required_project_names, "required_project_names")
    reserved = sorted(set(required) & ADMINISTRATIVE_INPUTS)
    if reserved:
        raise SemanticInputError("PROJECT_INPUT_RESERVED_ADMINISTRATIVE", ",".join(reserved))
    classified = classify_capture_inputs(capture_record)
    actual = {row["logical_name"] for row in classified["project_inputs"]}
    missing = sorted(set(required) - actual)
    if missing:
        raise SemanticInputError("PROJECT_INPUT_MISSING", ",".join(missing))
    unexpected = sorted(actual - set(required))
    if reject_unexpected and unexpected:
        raise SemanticInputError("PROJECT_INPUT_UNEXPECTED", ",".join(unexpected))
    return classified
