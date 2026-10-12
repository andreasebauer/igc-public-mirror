from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canon import write_json_atomic
from .records import utc_now


_STANDARD_PHASES = (
    "PREFLIGHT",
    "AUTHORITY_LOAD",
    "INPUT_MATERIALIZATION",
    "CARRIER_REPRODUCTION",
    "TASK_GENERATION",
    "SCIENCE_CENSUS",
    "MERGE",
    "INTERPRETATION",
    "RESULT_SEAL",
    "COLD_MATERIALIZATION",
    "COLD_REPLAY",
    "CERTIFICATION_COMPARE",
    "CLOSEOUT",
    "COMPLETE",
)


@dataclass
class RuntimeTelemetry:
    """Single-writer operational telemetry for one Decoder experiment.

    All wall-clock/progress fields are execution-only and must never enter the
    stable science payload.  Workers never write these files; the parent process
    owns all mutation.
    """

    root: Path
    experiment_id: str
    workers_requested: Any = "AUTO"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    wall_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "RUNTIME_STATUS.json"
        self.timings_path = self.root / "RUNTIME_TIMINGS.json"
        self._t0 = time.monotonic()
        self._phase_t0 = self._t0
        self._started_at = utc_now()
        self._phase_started_at = self._started_at
        self._last_write = 0.0
        self._phase = "PREFLIGHT"
        self._phase_seconds: dict[str, float] = {}
        self._status = "RUNNING"
        self._last_error: str | None = None
        self._last_checkpoint_at: str | None = None
        self._workers_leased = 0
        self._workers_active = 0
        self._tasks_total = 0
        self._tasks_started = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._kernels_total = 0
        self._kernels_started = 0
        self._kernels_completed = 0
        self._completion_samples: list[tuple[float, int, int]] = []
        self._detail: dict[str, Any] = {}
        self._write(force=True)

    @property
    def phase_name(self) -> str:
        return self._phase

    def phase(self, name: str, *, tasks_total: int | None = None, kernels_total: int | None = None) -> None:
        name = str(name)
        if name not in _STANDARD_PHASES:
            raise ValueError(f"unknown Decoder runtime phase {name!r}")
        now = time.monotonic()
        if self._phase:
            self._phase_seconds[self._phase] = self._phase_seconds.get(self._phase, 0.0) + max(0.0, now - self._phase_t0)
        self._phase = name
        self._phase_t0 = now
        self._phase_started_at = utc_now()
        if tasks_total is not None:
            self._tasks_total = max(0, int(tasks_total))
            self._tasks_started = 0
            self._tasks_completed = 0
            self._tasks_failed = 0
        if kernels_total is not None:
            self._kernels_total = max(0, int(kernels_total))
            self._kernels_started = 0
            self._kernels_completed = 0
        self._completion_samples.clear()
        self._write(force=True)

    def set_workers(self, *, leased: int, active: int | None = None) -> None:
        self._workers_leased = max(0, int(leased))
        self._workers_active = self._workers_leased if active is None else max(0, int(active))
        self._write()

    def mark_started(self, *, task_units: int = 1, kernel_units: int = 1) -> None:
        self._tasks_started += max(0, int(task_units))
        self._kernels_started += max(0, int(kernel_units))
        self._write()

    def mark_completed(self, *, task_units: int = 1, kernel_units: int = 1) -> None:
        self._tasks_completed += max(0, int(task_units))
        self._kernels_completed += max(0, int(kernel_units))
        now = time.monotonic()
        self._completion_samples.append((now, self._tasks_completed, self._kernels_completed))
        cutoff = now - 300.0
        self._completion_samples = [x for x in self._completion_samples if x[0] >= cutoff]
        self._write(force=False)

    def mark_failed(self, error: BaseException | str, *, task_units: int = 1) -> None:
        self._tasks_failed += max(0, int(task_units))
        self._last_error = str(error)
        self._write(force=True)

    def set_detail(self, **fields: Any) -> None:
        """Attach mutable execution-only detail to the live status record."""
        for key, value in fields.items():
            self._detail[str(key)] = value
        self._write(force=True)

    def checkpoint(self) -> None:
        self._last_checkpoint_at = utc_now()
        self._write(force=True)

    def fail(self, error: BaseException | str) -> None:
        self._status = "FAIL"
        self._last_error = str(error)
        self._write(force=True)
        self._write_timings()

    def complete(self) -> None:
        if self._phase != "COMPLETE":
            self.phase("COMPLETE")
        self._status = "COMPLETE"
        self._workers_active = 0
        self._write(force=True)
        self._write_timings()

    def _rates(self) -> tuple[float | None, float | None]:
        # Use phase-cumulative throughput for the public live ETA.  Completion can
        # arrive in shard-sized bursts after long expensive tasks; a short rolling
        # delta between two adjacent shard completions can therefore report a
        # physically meaningless hundreds-of-tasks/second spike.  Cumulative phase
        # rate is conservative, monotone in evidence horizon, and remains valid for
        # heterogeneous cost-weighted shards.
        elapsed = max(0.001, time.monotonic() - self._phase_t0)
        tr = self._tasks_completed / elapsed if self._tasks_completed else None
        kr = self._kernels_completed / elapsed if self._kernels_completed else None
        return tr, kr

    def _eta(self, kernel_rate: float | None) -> tuple[float | None, str]:
        min_done = max(4, math.ceil(0.05 * self._kernels_total)) if self._kernels_total else 4
        if self._kernels_completed < min_done or not kernel_rate or kernel_rate <= 0:
            return None, "LOW"
        remaining = max(0, self._kernels_total - self._kernels_completed)
        return remaining / kernel_rate, "MEDIUM"

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        task_rate, kernel_rate = self._rates()
        eta, eta_conf = self._eta(kernel_rate)
        return {
            "schema": "DECODER_RUNTIME_STATUS_V1",
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self._status,
            "phase": self._phase,
            "started_at": self._started_at,
            "phase_started_at": self._phase_started_at,
            "updated_at": utc_now(),
            "elapsed_seconds": round(max(0.0, now - self._t0), 6),
            "phase_elapsed_seconds": round(max(0.0, now - self._phase_t0), 6),
            "workers_requested": self.workers_requested,
            "workers_leased": self._workers_leased,
            "workers_active": self._workers_active,
            "tasks_total": self._tasks_total,
            "tasks_started": self._tasks_started,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "kernels_total": self._kernels_total,
            "kernels_started": self._kernels_started,
            "kernels_completed": self._kernels_completed,
            "throughput_tasks_per_second": None if task_rate is None else round(task_rate, 6),
            "throughput_kernels_per_second": None if kernel_rate is None else round(kernel_rate, 6),
            "eta_seconds": None if eta is None else round(eta, 3),
            "eta_confidence": eta_conf,
            "last_checkpoint_at": self._last_checkpoint_at,
            "last_error": self._last_error,
            "detail": dict(self._detail),
        }

    def _write(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_write) < max(0.1, float(self.wall_interval_seconds)):
            return
        write_json_atomic(self.status_path, self.snapshot())
        self._last_write = now

    def _write_timings(self) -> None:
        now = time.monotonic()
        phase_seconds = dict(self._phase_seconds)
        if self._phase:
            phase_seconds[self._phase] = phase_seconds.get(self._phase, 0.0) + max(0.0, now - self._phase_t0)
        obj = {
            "schema": "DECODER_RUNTIME_TIMINGS_V1",
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "phase_seconds": {k: round(v, 6) for k, v in sorted(phase_seconds.items())},
            "primary_total_seconds": round(max(0.0, now - self._t0), 6),
            "cold_total_seconds_if_present": None,
            "wall_total_seconds_if_present": round(max(0.0, now - self._t0), 6),
            "science_hash_excluded": True,
        }
        write_json_atomic(self.timings_path, obj)


__all__ = ["RuntimeTelemetry"]
