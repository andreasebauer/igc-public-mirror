"""Bounded observation of native records; never grants native reuse authority.

Record hashes detect corruption and bind the selected registration. They do
not reverify artifacts, terminal checkpoints, producer authority or remote saves.
"""
import re
from pathlib import Path

from .native import AdapterError, canonical_hash, contained, read_json


def results_observation(settings, job):
    root = contained(Path(settings.workspace_root), job.workspace, directory=True)

    def record(relative):
        return read_json(contained(root, relative, directory=False))

    def bound(value, field):
        if not isinstance(value, dict) or not isinstance(value.get(field), str):
            raise AdapterError("RESULT_RECORD_INVALID", 409)
        digest = value[field]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise AdapterError("RESULT_RECORD_INVALID", 409)
        try:
            actual = canonical_hash({k: v for k, v in value.items() if k != field})
        except (TypeError, ValueError) as exc:
            raise AdapterError("RESULT_RECORD_INVALID", 409) from exc
        if actual != digest:
            raise AdapterError("RESULT_RECORD_HASH_MISMATCH", 409)
        return digest

    registration = record("registry/" + job.native_job_id + ".json")
    digest = bound(registration, "registration_sha256")
    if registration.get("job_id") != job.native_job_id:
        raise AdapterError("RESULT_JOB_MISMATCH", 409)
    request_id = "job-" + digest[:32]
    try:
        request = {"schema_id": "IG_DECODER_PASSIVE_REQUEST_V1", "request_id": request_id,
                   "registered_job_id": job.native_job_id,
                   "requested_operation_id": registration["execution"]["kind"],
                   "parent_source_sha256": registration["source_sha256"],
                   "input_artifacts": registration["input_artifacts"]}
    except (KeyError, TypeError) as exc:
        raise AdapterError("RESULT_RECORD_INVALID", 409) from exc
    response = {"job_id": job.id, "native_job_id": job.native_job_id,
                "record_status": "ABSENT", "reported": None,
                "evidence_verification": "NOT_RUN", "preservation": "UNKNOWN",
                "reusable": False}
    for folder, state in (("completed", "PUBLISHED_RECORD"),
                          ("prepared_completions", "PENDING_CHECKPOINT_RECORD")):
        relative = "runtime/intake/" + folder + "/" + request_id + ".json"
        # Missing is legitimate; symlinks, including dangling ones, are not.
        path = root / relative
        if not path.exists() and not any(p.is_symlink() for p in (path, *path.parents)):
            continue
        completion = record(relative)
        bound(completion, "completion_sha256")
        if (completion.get("schema_id") != "IG_DECODER_WORKSPACE_COMPLETION_V1"
                or completion.get("registration_sha256") != digest
                or completion.get("source_sha256") != registration.get("source_sha256")
                or completion.get("request_id") != request_id
                or completion.get("request_sha256") != canonical_hash(request)
                or completion.get("result_sha256") != canonical_hash(completion.get("result"))):
            raise AdapterError("RESULT_BINDING_MISMATCH", 409)
        response.update(record_status=state,
                        record_integrity="HASH_AND_REGISTRATION_MATCH",
                        reported={k: completion.get(k) for k in (
                            "status", "execution_status", "evidence_status", "scientific_outcome")},
                        completion_sha256=completion["completion_sha256"],
                        result=completion.get("result"))
        return response
    return response
