from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from .canon import canonical_sha256, write_json_atomic
from .hashing import sha256_file
from .v05 import V05RegistrationStore
from .v05_verifier import VERIFICATION_SCHEMA, VERIFICATION_SCHEMA_V14, VERIFICATION_SCHEMA_V13, VERIFICATION_SCHEMA_V12, VERIFIER_ID
from .v05_runtime import V05TelemetryLedger


PUBLICATION_SCHEMA_V12 = "IG_DECODER_V05_PUBLICATION_RECORD_V1_2"
PUBLICATION_SCHEMA_V13 = "IG_DECODER_V05_PUBLICATION_RECORD_V1_3"
PUBLICATION_SCHEMA_V14 = "IG_DECODER_V05_PUBLICATION_RECORD_V1_4"
PUBLICATION_SCHEMA = "IG_DECODER_V05_PUBLICATION_RECORD_V1_5"
PUBLISHER_ID = "v05.protected.publisher"


class V05PublicationError(RuntimeError):
    pass


def _protected_root(paths) -> Path:
    root = Path(paths.store) / "v05" / "protected_publications"
    from .v05_origin_guard import require_registered_output
    require_registered_output(root, "publish_verified_execution")
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    if os.name == "posix" and hasattr(os, "geteuid"):
        st = root.stat()
        if st.st_uid != os.geteuid():
            raise V05PublicationError("protected publication root owner mismatch")
        if st.st_mode & 0o077:
            raise V05PublicationError("protected publication root not owner-only")
    return root


def _load_bound_records(paths, run_id: str, report_path: Path):
    run_dir = Path(paths.runs) / run_id
    env_path = run_dir / "v05_envelope.json"
    if not env_path.is_file():
        raise V05PublicationError("execution envelope missing")
    envelope = json.loads(env_path.read_text(encoding="utf-8"))
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("schema_id") not in {VERIFICATION_SCHEMA_V12, VERIFICATION_SCHEMA_V13, VERIFICATION_SCHEMA_V14, VERIFICATION_SCHEMA}:
        raise V05PublicationError("unsupported verifier report")
    report_sha = report.get("verification_sha256")
    if report_sha != canonical_sha256({k: v for k, v in report.items() if k != "verification_sha256"}):
        raise V05PublicationError("verification report hash mismatch")
    env_sha = envelope.get("envelope_sha256")
    if env_sha != canonical_sha256({k: v for k, v in envelope.items() if k != "envelope_sha256"}):
        raise V05PublicationError("execution envelope hash mismatch")
    if report.get("run_id") != run_id or report.get("envelope_sha256") != env_sha:
        raise V05PublicationError("verification report not bound to execution envelope")
    if report.get("status") != "PASS":
        raise V05PublicationError("verification status is not PASS")
    reg_sha = envelope.get("registration_sha256")
    reg = V05RegistrationStore(paths.store).get(reg_sha)
    if report.get("registration_sha256") != reg_sha:
        raise V05PublicationError("verification registration mismatch")
    if report.get("verifier_id") != reg.get("verification_policy", {}).get("verifier_id"):
        raise V05PublicationError("verification report verifier identity mismatch")
    pp = reg.get("publication_policy", {})
    if pp.get("publisher_id") != PUBLISHER_ID or pp.get("required_verification_status") != "PASS":
        raise V05PublicationError("registration publication policy mismatch")
    expected_effect = "NONE_P2_6" if reg.get("contract_version") == "1.5.0" else ("NONE_P2_5" if reg.get("contract_version") == "1.4.0" else ("NONE_P2_4" if reg.get("contract_version") == "1.3.0" else "NONE_P2_3"))
    if pp.get("authority_effect") != expected_effect:
        raise V05PublicationError("publisher refuses unexpected authority effect")
    return envelope, report, reg


def publish_verified_execution(paths, run_id: str, report_path: Path) -> dict:
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin("publish_verified_execution")
    root = _protected_root(paths)
    envelope, report, reg = _load_bound_records(paths, run_id, report_path)
    env_path = Path(paths.runs) / run_id / "v05_envelope.json"
    report_path = Path(report_path)
    env_file_sha = sha256_file(env_path)
    report_file_sha = sha256_file(report_path)
    base = {
        "schema_id": PUBLICATION_SCHEMA if reg.get("contract_version") == "1.5.0" else PUBLICATION_SCHEMA_V14 if reg.get("contract_version") == "1.4.0" else PUBLICATION_SCHEMA_V13 if reg.get("contract_version") == "1.3.0" else PUBLICATION_SCHEMA_V12,
        "contract_version": reg.get("contract_version"),
        "publisher_id": PUBLISHER_ID,
        "run_id": run_id,
        "registration_sha256": reg["registration_sha256"],
        "envelope_sha256": envelope["envelope_sha256"],
        "envelope_file_sha256": env_file_sha,
        "verification_sha256": report["verification_sha256"],
        "verification_file_sha256": report_file_sha,
        "verification_status": report["status"],
        "publication_status": "PUBLISHED_VERIFIED",
        "authority_effect": reg["publication_policy"]["authority_effect"],
        "protected_storage": "OWNER_ONLY_POSIX",
        "publisher_identity": {
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "gid": os.getgid() if hasattr(os, "getgid") else None,
            "pid": os.getpid(),
        },
    }
    record = dict(base, publication_sha256=canonical_sha256(base))
    pub_dir = root / record["publication_sha256"]
    if pub_dir.exists():
        existing = json.loads((pub_dir / "publication.json").read_text(encoding="utf-8"))
        if existing != record:
            raise V05PublicationError("publication identity collision")
        return existing

    tmp = Path(tempfile.mkdtemp(prefix=".publish-", dir=root))
    try:
        shutil.copyfile(env_path, tmp / "envelope.json")
        shutil.copyfile(report_path, tmp / "verification.json")
        write_json_atomic(tmp / "publication.json", record)
        for p in tmp.iterdir():
            os.chmod(p, 0o444)
        os.chmod(tmp, 0o500)
        os.rename(tmp, pub_dir)
        try:
            dfd = os.open(str(root), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except Exception:
        if tmp.exists():
            try:
                os.chmod(tmp, 0o700)
            except OSError:
                pass
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        V05TelemetryLedger(Path(paths.runs) / run_id / "v05_telemetry" / "p3-controller-spans.jsonl").record(
            "PUBLICATION", component="v05_publication.publish_verified_execution", status="PASS",
            details={"run_id":run_id,"publication_sha256":record.get("publication_sha256"),"authority_effect":record.get("authority_effect")},
        )
    except Exception:
        pass
    return record


def _cli() -> int:
    from .paths import resolve_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--verification-report", required=True)
    ap.add_argument("--record", required=False)
    ns = ap.parse_args()
    try:
        record = publish_verified_execution(resolve_root(ns.root), ns.run_id, Path(ns.verification_report))
        if ns.record:
            write_json_atomic(Path(ns.record), record)
        print(json.dumps({"status": record["publication_status"], "publication_sha256": record["publication_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 4


def publish_registered_chain_execution(controller, chain_id: str) -> dict:
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin("publish_registered_chain_execution")
    """Publish a verified engineering chain through the registered service.

    This is an additional adapter in the shared publication module, not a
    scientific executor. Existing historical publication formats are unchanged.
    The record reports execution provenance and has no scientific authority.
    """
    from .v05_registered_service import (
        RegisteredServiceSession, PUBLICATION_RECORD_SCHEMA,
        _owned_directory, strict_json, verify_service_publication,
    )
    from .v05_execution_authority import ENGINEERING_ROLE, ExecutionAuthorityError, digest, canonical_bytes

    session = getattr(controller, '_service_session', None)
    if type(session) is not RegisteredServiceSession:
        raise ExecutionAuthorityError('REGISTERED_SERVICE_PUBLICATION_REQUIRED')
    session.verify_live()
    reg = controller._load_reg(chain_id)
    session.require_registration(reg)
    status = controller.status(chain_id)
    if status['status'] != 'COMPLETE' or status['external_mirror_pending'] is not False:
        raise ExecutionAuthorityError('SERVICE_PUBLICATION_AWAITS_COMPLETE_CHAIN')
    cdir = controller.chain_dir(chain_id)
    commits = [controller._load_commit(cdir, sid) for sid in status['completed_stages']]
    if not commits or any(c is None for c in commits):
        raise ExecutionAuthorityError('SERVICE_PUBLICATION_COMMIT_REQUIRED')
    # Status verifies the registered path; each commit is re-read with the
    # service's installed public key and exact question/source/dependency binds.
    if [c['stage_id'] for c in commits] != [c['stage_id'] for c in status['commits']]:
        raise ExecutionAuthorityError('SERVICE_PUBLICATION_PATH_MISMATCH')
    record = {
        'schema_id': PUBLICATION_RECORD_SCHEMA, 'role': ENGINEERING_ROLE,
        'authoritative': False, 'science_authority_effect': 'NONE',
        'policy_sha256': session.policy_sha256, 'source_sha256': session._policy['source_sha256'],
        'registration_sha256': reg['registration_sha256'], 'chain_id': chain_id,
        'chain_status': status, 'registration': reg, 'commits': commits,
    }
    publication = session.publication_receipt(record)
    expected = {
        'expected_policy_sha256': session.policy_sha256,
        'expected_source_sha256': session._policy['source_sha256'],
        'expected_registration_sha256': reg['registration_sha256'],
        'trusted_public_keys': session.public_keys,
    }
    verify_service_publication(publication, **expected)
    from .v05_origin_guard import require_registered_output
    require_registered_output(session.service_root / 'registered_publications', 'publish_registered_chain_execution')
    root = _owned_directory(session.service_root / 'registered_publications', create=True)
    destination = root / (digest(record) + '.json')
    if destination.is_symlink():
        raise ExecutionAuthorityError('SERVICE_STANDARD_PATH_REQUIRED')
    if destination.exists():
        old = strict_json(destination.read_bytes())
        verify_service_publication(old, **expected)
        if old != publication:
            raise ExecutionAuthorityError('SERVICE_PUBLICATION_CONTENT_MISMATCH')
        return old
    fd, temp_name = tempfile.mkstemp(prefix='.registered-publication-', dir=root)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(canonical_bytes(publication) + b'\n')
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, 0o400)
        # Exclusive creation keeps an existing immutable record intact.
        os.link(temporary, destination, follow_symlinks=False)
        dfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        temporary.unlink(missing_ok=True)
    from .v05_engineering_jobs import copy_release_for_publication
    copy_release_for_publication(controller, commits, publication, root)
    return publication


if __name__ == "__main__":
    raise SystemExit(_cli())
