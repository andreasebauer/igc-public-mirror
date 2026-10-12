from __future__ import annotations

"""Controller-only handlers for registered Decoder engineering jobs.

Handlers contain no execution or durable-write infrastructure. They only describe
which registered engineering capability the Decoder-owned stage runtime should use.
"""

from typing import Mapping, Any
from .v05_chain import ChainExecutionResult


def engineering_job_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    result = runtime.run_registered_engineering_job(dict(stage["execution"].get("parameters") or {}))
    return ChainExecutionResult(result={"outcome": "PASS", "engineering_job": result})


def engineering_acceptance_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    result = runtime.accept_registered_engineering_candidate(dict(stage["execution"].get("parameters") or {}))
    return ChainExecutionResult(result={"outcome": "PASS", "engineering_acceptance": result})
