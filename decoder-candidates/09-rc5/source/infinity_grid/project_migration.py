"""Project-local Decoder migration without asking the historical engine to run.

The historical producer is evidence, not the authority for the new release.
This command verifies the current project head exactly, verifies the proposed
target and its qualification independently, and appends a local release event.
It never changes a shared Decoder pointer and never executes scientific work.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tempfile
import zipfile

from . import portable_registry as project, submission as sub
from .canon import canonical_sha256
from .change_sessions import source_version

SCHEMA = "IG_DECODER_PROJECT_ENGINE_MIGRATION_V1"
REQUEST_SCHEMA = "IG_DECODER_PROJECT_ENGINE_MIGRATION_REQUEST_V1"


def _error(code, detail=""):
    raise sub.SubmissionError(code, str(detail))


def _verified_record(path, expected_sha=None):
    raw = Path(path).read_bytes()
    if expected_sha is not None and sub._sha(raw) != expected_sha:
        _error("MIGRATION_QUALIFICATION_FILE_HASH")
    row = sub._read(path)
    seal = row.get("record_sha256")
    if seal != canonical_sha256({k: v for k, v in row.items() if k != "record_sha256"}):
        _error("MIGRATION_QUALIFICATION_SEAL")
    return row, sub._sha(raw)


def _extract_source(raw, destination):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if not names or len(names) != len(set(names)):
            _error("MIGRATION_TARGET_ARCHIVE_MEMBERS")
        for name in names:
            rel = sub._relative(name)
            info = archive.getinfo(name)
            if info.is_dir():
                continue
            target = Path(destination) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def _target(request):
    raw = Path(request["target_archive"]).read_bytes()
    if sub._sha(raw) != request["target_archive_sha256"]:
        _error("MIGRATION_TARGET_ARCHIVE_HASH")
    with tempfile.TemporaryDirectory(prefix="ig-project-migration-") as folder:
        source = Path(folder)
        _extract_source(raw, source)
        from .v05_controller_event_loop import _source_ids
        source_sha, package_sha = _source_ids(source)
        if source_sha != request["target_source_sha256"]:
            _error("MIGRATION_TARGET_SOURCE_HASH")
        if package_sha != request["target_package_sha256"]:
            _error("MIGRATION_TARGET_PACKAGE_HASH")
        version = source_version(source)
        normalized = project.engine_object(source)
    return raw, normalized, source_sha, package_sha, version


def migrate(coordination_root, request):
    """Append one verified project-local release, or return its exact retry."""
    root = Path(coordination_root).resolve(strict=True)
    request = sub._read(request) if isinstance(request, (str, Path)) else request
    fields = {
        "schema_id", "project_id", "expected_release_head", "expected_old_engine",
        "target_archive", "target_archive_sha256", "target_source_sha256",
        "target_package_sha256", "qualification", "qualification_sha256",
        "authorization", "reason",
    }
    if set(request) != fields or request.get("schema_id") != REQUEST_SCHEMA:
        _error("MIGRATION_REQUEST_FIELDS")
    authorization = request["authorization"]
    if (not isinstance(authorization, dict)
            or set(authorization) != {"operator", "decision", "scope"}
            or not all(isinstance(v, str) and v.strip() for v in authorization.values())
            or authorization["decision"] != "AUTHORIZE_PROJECT_LOCAL_MIGRATION"
            or authorization["scope"] != "ONE_PROJECT_NO_SHARED_POINTER"):
        _error("MIGRATION_AUTHORIZATION_REQUIRED")
    if not isinstance(request["reason"], str) or not request["reason"].strip():
        _error("MIGRATION_REASON_REQUIRED")

    meta = project.read(root / "PROJECT.json")
    if request["project_id"] != meta["project_id"]:
        _error("MIGRATION_PROJECT_ID")
    target_raw, normalized, source_sha, package_sha, version = _target(request)
    qualification, qualification_file_sha = _verified_record(
        request["qualification"], request["qualification_sha256"])
    if (qualification.get("schema_id") != "IG_DECODER_CODE_CANDIDATE_V1"
            or qualification.get("candidate_source_sha256") != source_sha
            or qualification.get("version") != version
            or not qualification.get("test_results")
            or any(row.get("outcome") != "PASS" for row in qualification["test_results"])):
        _error("MIGRATION_TARGET_NOT_QUALIFIED")

    with project.lock(root):
        head, release = project.current(root, "release")
        if release and release.get("engine") == sub._sha(normalized):
            migration_sha = release.get("migration")
            if not migration_sha:
                _error("MIGRATION_TARGET_ALREADY_CURRENT_WITHOUT_RECORD")
            record = json.loads(project.blob(root, migration_sha))
            if (record.get("target_source_sha256") != source_sha
                    or record.get("target_archive_sha256") != sub._sha(target_raw)):
                _error("MIGRATION_RETRY_CONFLICT")
            return {"status": "PROJECT_ALREADY_MIGRATED", "release_head": head,
                    "migration_sha256": migration_sha, "record": record}
        if head != request["expected_release_head"]:
            _error("MIGRATION_STALE_RELEASE_HEAD")
        if not release or release.get("engine") != request["expected_old_engine"]:
            _error("MIGRATION_OLD_ENGINE_MISMATCH")
        # This validates the exact historical producer bytes independently. It
        # deliberately does not import or execute that old engine.
        project.blob(root, request["expected_old_engine"])
        target_engine = project.put(root, normalized)
        record = {
            "schema_id": SCHEMA,
            "project_id": meta["project_id"],
            "previous_release_head": head,
            "historical_engine_sha256": request["expected_old_engine"],
            "target_engine_sha256": target_engine,
            "target_archive_sha256": sub._sha(target_raw),
            "target_source_sha256": source_sha,
            "target_package_sha256": package_sha,
            "target_version": version,
            "qualification_record_sha256": qualification["record_sha256"],
            "qualification_file_sha256": qualification_file_sha,
            "authorization": authorization,
            "reason": request["reason"],
            "scope": "PROJECT_LOCAL_ONLY_NO_SHARED_POINTER_NO_SCIENCE_EXECUTION",
        }
        record["record_sha256"] = canonical_sha256(record)
        migration_sha = project.put(root, sub._json_bytes(record))
        new_head = project.append(root, "release", [head], {
            "engine": target_engine, "version": version,
            "basis": "PROJECT_LOCAL_MIGRATION", "validation": None,
            "previous": head, "migration": migration_sha,
        }, "Project-local migration: " + request["reason"])
    return {"status": "PROJECT_MIGRATED", "release_head": new_head,
            "migration_sha256": migration_sha, "record": record}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Append-only project-local Decoder migration; never changes a shared pointer.")
    parser.add_argument("coordination_root")
    parser.add_argument("request_json")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(migrate(args.coordination_root, args.request_json), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        from .invocation import refusal_details
        print(json.dumps(refusal_details(exc, "project migrate"), sort_keys=True), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
