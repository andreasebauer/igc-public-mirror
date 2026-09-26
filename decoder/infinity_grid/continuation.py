"""Verified-prefix continuation for interrupted deterministic campaigns.

Completed work is reused only as an exact ordered prefix.  A continuation is
bound to new capture/attempt identities and names a disjoint output namespace.
Failed terminal attempts cannot be continued, gaps cannot be skipped, and
budgets are cumulative across the prefix and suffix.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import zipfile

from . import submission as sub
from .canon import canonical_sha256
from .change_sessions import _immutable, _sealed, _verified

PLAN_SCHEMA = "IG_DECODER_CONTINUATION_PLAN_V1"
PREFIX_SCHEMA = "IG_DECODER_VERIFIED_PREFIX_MANIFEST_V1"
RECORD_SCHEMA = "IG_DECODER_CONTINUATION_RECORD_V1"


def _error(code, detail=""):
    raise sub.SubmissionError(code, str(detail))


def _actions(plan):
    if plan.get("schema_id") != PLAN_SCHEMA or not isinstance(plan.get("actions"), list) or not plan["actions"]:
        _error("CONTINUATION_PLAN")
    ids = []
    required = {"action_id", "source_sha256", "inputs_sha256", "budget_bytes"}
    for index, action in enumerate(plan["actions"]):
        if (set(action) != required or not all(isinstance(action[k], str) and len(action[k]) == 64
                                               for k in ("source_sha256", "inputs_sha256"))
                or not isinstance(action["action_id"], str) or not action["action_id"]
                or type(action["budget_bytes"]) is not int or action["budget_bytes"] < 0):
            _error("CONTINUATION_ACTION", index)
        ids.append(action["action_id"])
    if len(ids) != len(set(ids)):
        _error("CONTINUATION_DUPLICATE_ACTION")
    return plan["actions"]


def seal_plan(plan):
    _actions(plan)
    return _sealed(plan, "plan_sha256")


def create_prefix(plan, completed_rows, *, cumulative_evidence_bytes=None, role_usage=None):
    actions = _actions(plan)
    rows = []
    required = {"action_id", "source_sha256", "inputs_sha256", "result_sha256",
                "evidence_role", "evidence_sha256", "evidence_size_bytes", "status"}
    if len(completed_rows) > len(actions):
        _error("CONTINUATION_PREFIX_TOO_LONG")
    for index, row in enumerate(completed_rows):
        if set(row) != required or row.get("status") != "COMPLETED":
            _error("CONTINUATION_PREFIX_ROW", index)
        action = actions[index]
        if any(row[k] != action[k] for k in ("action_id", "source_sha256", "inputs_sha256")):
            _error("CONTINUATION_PREFIX_NOT_ORDERED", index)
        if (not isinstance(row["evidence_role"], str) or not row["evidence_role"]
                or type(row["evidence_size_bytes"]) is not int or row["evidence_size_bytes"] < 0
                or any(not isinstance(row[k], str) or len(row[k]) != 64
                       for k in ("result_sha256", "evidence_sha256"))):
            _error("CONTINUATION_PREFIX_EVIDENCE", index)
        rows.append(_sealed(dict(row), "row_sha256"))
    observed = sum(row["evidence_size_bytes"] for row in rows)
    if cumulative_evidence_bytes is None:
        cumulative_evidence_bytes = observed
    if type(cumulative_evidence_bytes) is not int or cumulative_evidence_bytes < observed:
        _error("CONTINUATION_PRIOR_ACCOUNTING")
    role_usage = {} if role_usage is None else role_usage
    if (not isinstance(role_usage, dict) or any(not isinstance(k, str) or not k
            or type(v) is not int or v < 0 for k, v in role_usage.items())):
        _error("CONTINUATION_ROLE_ACCOUNTING")
    body = {"schema_id": PREFIX_SCHEMA, "plan_sha256": plan.get("plan_sha256", canonical_sha256(plan)),
            "completed_count": len(rows), "rows": rows,
            "observed_evidence_bytes": observed,
            "cumulative_evidence_bytes": cumulative_evidence_bytes,
            "role_usage": dict(sorted(role_usage.items()))}
    return _sealed(body, "prefix_sha256")


def verify_prefix(plan, prefix):
    actions = _actions(plan)
    if (prefix.get("schema_id") != PREFIX_SCHEMA
            or prefix.get("prefix_sha256") != canonical_sha256({k: v for k, v in prefix.items() if k != "prefix_sha256"})
            or prefix.get("completed_count") != len(prefix.get("rows", []))
            or prefix.get("completed_count") > len(actions)):
        _error("CONTINUATION_PREFIX_SEAL")
    clean = create_prefix(plan, [{k: v for k, v in row.items() if k != "row_sha256"}
                                 for row in prefix["rows"]],
                          cumulative_evidence_bytes=prefix.get("cumulative_evidence_bytes"),
                          role_usage=prefix.get("role_usage"))
    if clean != prefix:
        _error("CONTINUATION_PREFIX_MISMATCH")
    return clean


def prefix_from_archive(plan, archive_path, manifest, *, cumulative_evidence_bytes, role_usage=None):
    """Verify and bind an existing ZIP prefix without executing any action."""
    actions = _actions(plan)
    records = manifest.get("records")
    count = manifest.get("complete_count")
    if not isinstance(records, list) or count != len(records) or count > len(actions):
        _error("CONTINUATION_ARCHIVE_MANIFEST")
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            _error("CONTINUATION_ARCHIVE_DUPLICATE_MEMBER")
        for index, item in enumerate(records):
            if (set(item) != {"action_id", "bytes", "member", "sha256", "uncompressed_bytes"}
                    or item["action_id"] != actions[index]["action_id"]):
                _error("CONTINUATION_ARCHIVE_ORDER", index)
            member = sub._relative(item["member"]).as_posix()
            try: raw = archive.read(member)
            except KeyError: _error("CONTINUATION_ARCHIVE_MEMBER_MISSING", member)
            if len(raw) != item["bytes"] or sub._sha(raw) != item["sha256"]:
                _error("CONTINUATION_ARCHIVE_MEMBER_HASH", member)
            try: decoded = gzip.decompress(raw)
            except (OSError, EOFError): _error("CONTINUATION_ARCHIVE_MEMBER_ENCODING", member)
            if len(decoded) != item["uncompressed_bytes"] or not decoded:
                _error("CONTINUATION_ARCHIVE_MEMBER_CONTENT", member)
            rows.append({"action_id": item["action_id"],
                "source_sha256": actions[index]["source_sha256"],
                "inputs_sha256": actions[index]["inputs_sha256"],
                "result_sha256": sub._sha(decoded), "evidence_role": "completed_prefix_segment",
                "evidence_sha256": item["sha256"], "evidence_size_bytes": item["bytes"],
                "status": "COMPLETED"})
        incomplete = manifest.get("empty_incomplete_member")
        if incomplete:
            member = sub._relative(incomplete).as_posix()
            try: raw = archive.read(member)
            except KeyError: _error("CONTINUATION_INCOMPLETE_MARKER_MISSING")
            if raw:
                _error("CONTINUATION_INCOMPLETE_MARKER_NOT_EMPTY")
    return create_prefix(plan, rows, cumulative_evidence_bytes=cumulative_evidence_bytes,
                         role_usage=role_usage)


def output_namespace(capture_id, attempt_id):
    for value in (capture_id, attempt_id):
        if not isinstance(value, str) or not value or "/" in value or "\\" in value or value in {".", ".."}:
            _error("CONTINUATION_NAMESPACE")
    return f"captures/{capture_id}/attempts/{attempt_id}/segments"


def continue_plan(store_root, plan, prefix, request):
    store = Path(store_root).resolve(); store.mkdir(parents=True, exist_ok=True)
    verify_prefix(plan, prefix); actions = _actions(plan)
    fields = {"previous_capture_id", "previous_attempt_id", "previous_status",
              "new_capture_id", "new_attempt_id", "max_cumulative_bytes",
              "additional_role_usage", "max_cumulative_roles", "reason"}
    if set(request) != fields:
        _error("CONTINUATION_REQUEST_FIELDS")
    if request["previous_status"] not in {"PAUSED", "INTERRUPTED"}:
        _error("CONTINUATION_TERMINAL_STATUS")
    if ((request["previous_capture_id"], request["previous_attempt_id"])
            == (request["new_capture_id"], request["new_attempt_id"])):
        _error("CONTINUATION_IDENTITY_REUSE")
    namespace = output_namespace(request["new_capture_id"], request["new_attempt_id"])
    completed = prefix["completed_count"]
    missing = actions[completed:]
    if not missing:
        _error("CONTINUATION_NOTHING_MISSING")
    cumulative = prefix["cumulative_evidence_bytes"] + sum(a["budget_bytes"] for a in missing)
    if type(request["max_cumulative_bytes"]) is not int or cumulative > request["max_cumulative_bytes"]:
        _error("CONTINUATION_CUMULATIVE_BUDGET")
    additional_roles = request["additional_role_usage"]
    role_caps = request["max_cumulative_roles"]
    if (not isinstance(additional_roles, dict) or not isinstance(role_caps, dict)
            or any(not isinstance(k, str) or not k or type(v) is not int or v < 0
                   for values in (additional_roles, role_caps) for k, v in values.items())):
        _error("CONTINUATION_ROLE_ACCOUNTING")
    roles = dict(prefix["role_usage"])
    for role, value in additional_roles.items(): roles[role] = roles.get(role, 0) + value
    if any(role not in role_caps or used > role_caps[role] for role, used in roles.items()):
        _error("CONTINUATION_CUMULATIVE_ROLE_BUDGET")
    if not isinstance(request["reason"], str) or not request["reason"].strip():
        _error("CONTINUATION_REASON_REQUIRED")
    body = {"schema_id": RECORD_SCHEMA,
            "plan_sha256": plan.get("plan_sha256", canonical_sha256(plan)),
            "prefix_sha256": prefix["prefix_sha256"],
            "previous": {"capture_id": request["previous_capture_id"],
                         "attempt_id": request["previous_attempt_id"],
                         "status": request["previous_status"]},
            "continuation": {"capture_id": request["new_capture_id"],
                             "attempt_id": request["new_attempt_id"],
                             "output_namespace": namespace},
            "completed_action_ids": [a["action_id"] for a in actions[:completed]],
            "selected_action_ids": [a["action_id"] for a in missing],
            "cumulative_budget_bytes": cumulative,
            "max_cumulative_bytes": request["max_cumulative_bytes"],
            "cumulative_role_usage": dict(sorted(roles.items())),
            "max_cumulative_roles": dict(sorted(role_caps.items())),
            "reason": request["reason"]}
    record = _sealed(body)
    path = store / "continuations" / (record["record_sha256"] + ".json")
    _immutable(path, record)
    latest = store / "continuations" / (request["new_capture_id"] + "--" + request["new_attempt_id"] + ".json")
    if latest.exists() and _verified(latest) != record:
        _error("CONTINUATION_DUPLICATE_ATTEMPT")
    _immutable(latest, record)
    return {"status": "CONTINUATION_READY", "record": record,
            "selected_actions": missing, "record_path": str(path)}
