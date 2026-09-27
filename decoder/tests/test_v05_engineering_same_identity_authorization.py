from __future__ import annotations

import os

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.v05_engineering_jobs import (
    JOB_SPEC_SCHEMA,
    SAME_IDENTITY_AUTHORIZATION_SCHEMA,
    SAME_IDENTITY_AUTHORIZATION_SCOPE,
    SAME_IDENTITY_AUTHORIZATION_TEXT,
    EngineeringJobError,
    _validate_same_identity_authorization,
    _worker_identity_mode,
    validate_job_parameters,
)


def _parameters(uid: int | None = None, gid: int | None = None) -> dict:
    return {
        "schema_id": JOB_SPEC_SCHEMA,
        "job_id": "AUTH.FIXTURE.001",
        "operation": "VALIDATE_SOURCE",
        "parent_source_sha256": "a" * 64,
        "parent_package_sha256": "b" * 64,
        "expected_candidate_source_sha256": "a" * 64,
        "expected_candidate_package_sha256": "b" * 64,
        "overlay_path": None,
        "overlay_sha256": None,
        "validation_groups": ["engineering_layer"],
        "worker_uid": os.geteuid() if uid is None else uid,
        "worker_gid": os.getegid() if gid is None else gid,
        "wall_seconds_max": 60,
        "same_identity_authorization": None,
    }


def _authorization(p: dict, **changes) -> dict:
    auth = {
        "schema_id": SAME_IDENTITY_AUTHORIZATION_SCHEMA,
        "authorization_id": "operator-approval-fixture-001",
        "authorized_by": "Test Operator",
        "scope": SAME_IDENTITY_AUTHORIZATION_SCOPE,
        "authorization_text": SAME_IDENTITY_AUTHORIZATION_TEXT,
        "job_id": p["job_id"],
        "parent_source_sha256": p["parent_source_sha256"],
        "worker_uid": p["worker_uid"],
        "worker_gid": p["worker_gid"],
        "reason": "Exercise the explicit one-job authorization contract.",
    }
    auth.update(changes)
    auth["authorization_sha256"] = canonical_sha256(auth)
    return auth


def test_same_identity_requires_explicit_authorization():
    p = validate_job_parameters(_parameters())
    with pytest.raises(EngineeringJobError, match="ENGINEERING_SAME_IDENTITY_AUTHORIZATION_REQUIRED"):
        _worker_identity_mode(p, os.geteuid(), os.getegid())


def test_authorization_is_bound_to_job_source_and_identity():
    p = validate_job_parameters(_parameters())
    for changes in ({"job_id": "OTHER"}, {"parent_source_sha256": "c" * 64}, {"worker_uid": p["worker_uid"] + 1}):
        with pytest.raises(EngineeringJobError, match="ENGINEERING_SAME_IDENTITY_AUTHORIZATION_BINDING"):
            _validate_same_identity_authorization(_authorization(p, **changes), p)


def test_authorization_hash_is_required_and_verified():
    p = validate_job_parameters(_parameters())
    auth = _authorization(p)
    auth["reason"] = "Changed after operator authorization."
    with pytest.raises(EngineeringJobError, match="ENGINEERING_SAME_IDENTITY_AUTHORIZATION_HASH_MISMATCH"):
        _validate_same_identity_authorization(auth, p)


def test_valid_one_job_authorization_is_accepted():
    p = validate_job_parameters(_parameters())
    auth = _authorization(p)
    p["same_identity_authorization"] = auth
    assert _worker_identity_mode(p, os.geteuid(), os.getegid()) == (True, auth)


def test_distinct_identity_needs_no_authorization():
    p = validate_job_parameters(_parameters(os.geteuid() + 1, os.getegid() + 1))
    assert _worker_identity_mode(p, os.geteuid(), os.getegid()) == (False, None)


def test_partially_shared_identity_is_never_authorized():
    p = validate_job_parameters(_parameters(os.geteuid(), os.getegid() + 1))
    p["same_identity_authorization"] = _authorization(p)
    with pytest.raises(EngineeringJobError, match="ENGINEERING_WORKER_IDENTITY_PARTIALLY_SHARED"):
        _worker_identity_mode(p, os.geteuid(), os.getegid())


def test_authorization_cannot_be_attached_to_distinct_identity():
    p = validate_job_parameters(_parameters(os.geteuid() + 1, os.getegid() + 1))
    p["same_identity_authorization"] = _authorization(p)
    with pytest.raises(EngineeringJobError, match="ENGINEERING_SAME_IDENTITY_AUTHORIZATION_NOT_APPLICABLE"):
        _worker_identity_mode(p, os.geteuid(), os.getegid())
