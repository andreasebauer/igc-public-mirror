from __future__ import annotations

import json

import infinity_grid.v05_passive_intake as intake


def _ctx():
    return {
        "role": intake.CONTROLLER_ROOT_ROLE,
        "controller_session_id": "session-a",
        "root_execution_id": "root-a",
        "context_id": "ctx-a",
    }


def _registry():
    return {
        "DECODER.G6.SCIENCE": {
            "allowed_operations": ["G6_S7_DEPTH2_COMPLETION"],
            "registration_sha256": "1" * 64,
            "implementation_sha256": "2" * 64,
        }
    }


def _request():
    return {
        "schema_id": intake.REQUEST_SCHEMA,
        "request_id": "s7d2-resume-idempotency-test",
        "registered_job_id": "DECODER.G6.SCIENCE",
        "requested_operation_id": "G6_S7_DEPTH2_COMPLETION",
        "parent_source_sha256": "3" * 64,
        "input_artifacts": [{"logical_name": "science_plan", "sha256": "4" * 64}],
    }


def test_ingest_same_request_reuses_same_internal_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(intake, "require_controller_execution_origin", lambda _label: None)
    monkeypatch.setattr(intake, "current_execution_context_snapshot", _ctx)
    intake.submit_passive_request(tmp_path / "intake", _request())
    a = intake.ingest_passive_request(
        tmp_path / "intake", _request()["request_id"], accepted_registry=_registry(),
        expected_parent_source_sha256="3" * 64, internal_root=tmp_path / "internal",
    )
    b = intake.ingest_passive_request(
        tmp_path / "intake", _request()["request_id"], accepted_registry=_registry(),
        expected_parent_source_sha256="3" * 64, internal_root=tmp_path / "internal",
    )
    assert a["internal_execution_id"] == b["internal_execution_id"]
    assert len(list((tmp_path / "internal" / "ingested").glob("*.json"))) == 1
    assert len(list((tmp_path / "internal" / "claims").glob("*.json"))) == 1


def test_ingest_claim_rejects_mutated_same_request_id(monkeypatch, tmp_path):
    monkeypatch.setattr(intake, "require_controller_execution_origin", lambda _label: None)
    monkeypatch.setattr(intake, "current_execution_context_snapshot", _ctx)
    req = _request()
    intake.submit_passive_request(tmp_path / "intake", req)
    intake.ingest_passive_request(
        tmp_path / "intake", req["request_id"], accepted_registry=_registry(),
        expected_parent_source_sha256="3" * 64, internal_root=tmp_path / "internal",
    )
    pending = tmp_path / "intake" / "pending" / f"{req['request_id']}.json"
    obj = json.loads(pending.read_text())
    obj["human_note"] = "mutated after claim"
    pending.write_text(json.dumps(obj))
    try:
        intake.ingest_passive_request(
            tmp_path / "intake", req["request_id"], accepted_registry=_registry(),
            expected_parent_source_sha256="3" * 64, internal_root=tmp_path / "internal",
        )
    except intake.PassiveRequestError as exc:
        assert str(exc).startswith("PASSIVE_INGEST_CLAIM_MISMATCH")
    else:
        raise AssertionError("mutated same request_id must fail closed")
