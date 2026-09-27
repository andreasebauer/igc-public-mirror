"""HTTP transport. Background submission is an injected durable gateway."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .models import AcceptedRequest, CaptureBody, Catalog, EmptyBody, Identifier, PauseBody, Settings
from .native import AdapterError, Command, NativeAdapter, PIN_COMMIT, PIN_SOURCE, PIN_VERSION, canonical_hash, read_json

IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128,
                                     pattern=r"^[A-Za-z0-9_.:-]+$")]

class SubmissionGateway(Protocol):
    """Must implement durable, atomic idempotency before any native dispatch.

    Same key and digest returns the same request. Different digest conflicts.
    Also prevent simultaneous runs for one native job with different keys.
    Revalidate command/source/workspace at dispatch; never trust stale plans.
    """
    def submit(self, command: Command, key: str, digest: str) -> AcceptedRequest: ...
    def get(self, request_id: str) -> dict: ...


class UnavailableGateway:
    def submit(self, command, key, digest):
        raise AdapterError("BACKGROUND_WORKER_NOT_CONFIGURED")

    def get(self, request_id):
        raise AdapterError("BACKGROUND_WORKER_NOT_CONFIGURED")


class BodyLimit:
    """Bound request bytes including chunked bodies before JSON parsing."""
    def __init__(self, app, limit=65536):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                response = JSONResponse({"error": {"code": "BODY_TOO_LARGE"}}, status_code=413)
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        index = 0

        async def replay():
            nonlocal index
            if index < len(messages):
                item = messages[index]
                index += 1
                return item
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings, token: str, gateway: SubmissionGateway | None = None,
               adapter: NativeAdapter | None = None) -> FastAPI:
    if len(token) < 32 or not token.isascii():
        raise ValueError("Supply an ASCII API token of at least 32 characters")
    native = adapter or NativeAdapter(settings)
    submissions = gateway or UnavailableGateway()
    bearer = HTTPBearer(auto_error=False)

    def authenticated(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        supplied = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="AUTHENTICATION_REQUIRED", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Decoder Web API", version="0.1.0", docs_url=None, redoc_url=None,
                  openapi_url=None, dependencies=[Depends(authenticated)])
    app.add_middleware(BodyLimit)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.middleware("http")
    async def response_policy(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(AdapterError)
    async def adapter_error(request, exc):
        return JSONResponse({"error": {"code": exc.code}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo submitted paths, secrets or arbitrary input into responses.
        return JSONResponse({"error": {"code": "INVALID_REQUEST", "fields": [
            {"location": list(e["loc"]), "type": e["type"]} for e in exc.errors()]}}, status_code=400)

    def catalogue():
        try:
            catalog = Catalog.model_validate(read_json(Path(settings.catalog)))
            for rows in (catalog.jobs, catalog.tasks, catalog.inputs):
                if len({x.id for x in rows}) != len(rows):
                    raise AdapterError("CATALOG_DUPLICATE_ID")
            return catalog
        except (OSError, ValidationError) as exc:
            raise AdapterError("CATALOG_UNAVAILABLE") from exc

    def job_by_id(job_id):
        for job in catalogue().jobs:
            if job.id == job_id:
                return job
        raise AdapterError("JOB_NOT_FOUND", 404)

    def accept(command: Command, key: str):
        digest = canonical_hash({"operation": command.operation, "target": command.target,
                                 "argv": list(command.argv), "cwd": command.cwd,
                                 "source_sha256": command.source_sha256,
                                 "specification_sha256": command.specification_sha256})
        return submissions.submit(command, key, digest)

    @app.get("/api/v1/openapi.json")
    def schema():
        return app.openapi()

    @app.get("/api/v1/engine")
    def engine():
        try:
            source = native.verify_source()
        except (AdapterError, OSError) as exc:
            source = {"status": "UNAVAILABLE", "reason": exc.code if isinstance(exc, AdapterError) else "SOURCE_READ_ERROR"}
        return {"version": PIN_VERSION, "commit": PIN_COMMIT, "source_sha256": PIN_SOURCE,
                "source": source, "runtime_verification": "NOT_RUN", "execution_ready": False,
                "background_worker": "UNCONFIGURED" if isinstance(submissions, UnavailableGateway) else "INJECTED_NOT_QUALIFIED"}

    @app.get("/api/v1/tasks")
    def tasks():
        return {"items": [{"id": t.id, "name": t.name, "specification_sha256": t.specification_sha256}
                          for t in catalogue().tasks]}

    @app.get("/api/v1/inputs")
    def inputs():
        return {"items": [i.model_dump() for i in catalogue().inputs]}

    @app.get("/api/v1/jobs")
    def jobs(offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=100)] = 50):
        items = catalogue().jobs
        return {"items": [{"id": j.id, "name": j.name, "native_job_id": j.native_job_id} for j in items[offset:offset+limit]],
                "total": len(items), "offset": offset}

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: Identifier):
        j = job_by_id(job_id)
        return {"id": j.id, "name": j.name, "native_job_id": j.native_job_id}

    @app.get("/api/v1/jobs/{job_id}/status")
    def status(job_id: Identifier):
        return native.observe("status", job_by_id(job_id))

    @app.get("/api/v1/jobs/{job_id}/preservation")
    def preservation(job_id: Identifier):
        j = job_by_id(job_id)
        return {"capture": native.observe("pending-saves", j), "outbox": native.observe("preservation", j)}

    # Lower-level typed capture route for frozen server-owned specifications.
    # Draft editing is a later UI service; it must freeze a spec before using this.
    @app.post("/api/v1/captures", status_code=202, response_model=AcceptedRequest)
    def capture(body: CaptureBody, idempotency_key: IdempotencyKey):
        task = next((t for t in catalogue().tasks if t.id == body.task_id), None)
        if task is None:
            raise AdapterError("TASK_NOT_FOUND", 404)
        return accept(native.command("capture", task=task), idempotency_key)

    @app.post("/api/v1/jobs/{job_id}/run", status_code=202, response_model=AcceptedRequest)
    def run(job_id: Identifier, body: EmptyBody, idempotency_key: IdempotencyKey):
        return accept(native.command("run", job_by_id(job_id)), idempotency_key)

    @app.post("/api/v1/jobs/{job_id}/pause", status_code=202, response_model=AcceptedRequest)
    def pause(job_id: Identifier, body: PauseBody, idempotency_key: IdempotencyKey):
        return accept(native.command("pause", job_by_id(job_id), reason=body.reason), idempotency_key)

    @app.get("/api/v1/requests/{request_id}")
    def request_status(request_id: Identifier):
        return submissions.get(request_id)

    @app.get("/api/v1/jobs/{job_id}/results")
    def results(job_id: Identifier):
        job_by_id(job_id)
        # Fail closed until result record binding is verified with native fixtures.
        raise AdapterError("RESULT_READER_AWAITING_NATIVE_FIXTURE_VERIFICATION")

    return app


def app_from_environment():
    """uvicorn ig_web.api:app_from_environment --factory --host 127.0.0.1"""
    settings = Settings.model_validate(read_json(Path(os.environ["IG_WEB_CONFIG"])))
    return create_app(settings, os.environ["IG_WEB_TOKEN"])
