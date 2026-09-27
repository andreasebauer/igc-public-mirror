from typing import Annotated, Literal

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

Identifier = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Job(StrictModel):
    id: Identifier
    name: str = Field(min_length=1, max_length=200)
    native_job_id: Identifier
    workspace: str


class Task(StrictModel):
    id: Identifier
    name: str = Field(min_length=1, max_length=200)
    specification: str
    specification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Input(StrictModel):
    id: Identifier
    name: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    available: bool = False


class Catalog(StrictModel):
    jobs: list[Job] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    inputs: list[Input] = Field(default_factory=list)


class Settings(StrictModel):
    engine_repository: str
    engine_python: str
    workspace_root: str
    specification_root: str
    capture_store: str
    catalog: str
    worker_state: str | None = None
    # No browser-selected paths. Token is supplied separately via environment.
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost", "testserver"])

    @field_validator("engine_repository", "engine_python", "workspace_root", "specification_root", "capture_store", "catalog", "worker_state")
    @classmethod
    def absolute_path(cls, value):
        if value is None:
            return value
        if not Path(value).is_absolute() or ".." in Path(value).parts or "\x00" in value:
            raise ValueError("Configuration paths must be absolute without traversal")
        return value


class EmptyBody(StrictModel):
    pass


class PauseBody(StrictModel):
    reason: str = Field(min_length=1, max_length=500, pattern=r"^[^\x00]+$")


class CaptureBody(StrictModel):
    task_id: Identifier


class NativeObservation(StrictModel):
    observed_at: str
    exit_code: int
    native: dict | None
    stdout: str
    stderr: str
    classification: Literal["RESPONSE", "REFUSED", "UNPARSEABLE"]


class AcceptedRequest(StrictModel):
    request_id: Identifier
    status: Literal["queued", "dispatching", "running", "finished", "refused", "interrupted", "needs_reconciliation"]
