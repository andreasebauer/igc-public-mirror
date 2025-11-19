# igc/metrics/runner.py
"""
Metrics Runner — DB-driven metric pipeline executor.

This module is the "sim suite" equivalent for metrics:
it knows how to take a job from the unified ledger, look up
its metric wiring (metrics_steps / metricinputmatcher), load
the right frame fields, run all steps (compute → flatten → final),
and write final artifacts to the output_path that OE/ledger
already decided.

OE/core.py stays responsible for:
  - seeding simmetjobs
  - finalizing output_path via PathRegistry
  - marking jobs running/written/failed
  - jobexecutionlog + errorlog + viewer events

This module is pure execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from igc.db.pg import cx, fetchall_dict
from igc.ledger.sim import get_simulation_full


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricStep:
    step: int              # 1,2,3 (compute, flatten, final)
    role: str              # 'compute' | 'flatten' | 'final'
    lib_name: str          # e.g. 'pyfftw', 'numpy', 'scipy.ndimage', 'itk', 'gudhi', 'writer'
    kf_name: str           # kernel family, e.g. 'fft', 'stats', 'morph', 'topology', 'scan', 'serialize'
    op_name: str           # concrete op, e.g. 'power_spectrum', 'minmaxmeanstd', 'betti', 'csv', ...
    logical_name: str      # internal logical key, e.g. 's1_compute', 'mask', 'voxel_stats'
    inputs_from: List[str] # list of logical/input tokens, e.g. ['psi'], ['phi','psi','eta']
    artifact_file: str     # suggested file name for this step (often 'intermediate.npy', or 'sigma8.csv' for final)
    artifact_ext: str      # 'npy', 'csv', 'json', ...
    fanout_index: int | None  # for multi-branch flatten steps (echo profile), else None


@dataclass
class MetricPipeline:
    metric_id: int
    name: str
    steps: List[MetricStep]


# ---------------------------------------------------------------------------
# Pipeline loading from DB
# ---------------------------------------------------------------------------

def load_metric_pipelines(metric_ids: Sequence[int]) -> Dict[int, MetricPipeline]:
    """
    Load metric pipelines from public.metrics_steps for the given metric_ids.

    This is called once per OE run(sim_id), not per job. It builds a
    MetricPipeline for each metric_id, driven entirely by the DB wiring.
    """
    metric_ids = sorted({int(mid) for mid in metric_ids if mid is not None})
    if not metric_ids:
        return {}

    # 1) Load basic metric names
    metrics_sql = """
        SELECT id, name
        FROM public.metrics
        WHERE id = ANY(%s)
        ORDER BY id
    """
    # 2) Load steps from metrics_steps
    steps_sql = """
        SELECT
            met_id,
            met_name,
            mim_step,
            mim_role,
            mim_lib_name,
            mim_kf_name,
            mim_op_name,
            mim_logical_name,
            mim_inputs_from,
            mim_artifact_file,
            mim_artifact_ext,
            mim_fanout_index
        FROM public.metrics_steps
        WHERE met_id = ANY(%s)
        ORDER BY met_id, mim_step, mim_role, mim_fanout_index NULLS FIRST, mim_id
    """

    with cx() as conn:
        rows_metrics = fetchall_dict(conn, metrics_sql, (metric_ids,))
        rows_steps = fetchall_dict(conn, steps_sql, (metric_ids,))

    name_by_id: Dict[int, str] = {int(r["id"]): (r.get("name") or f"metric_{r['id']}") for r in rows_metrics}
    pipelines: Dict[int, MetricPipeline] = {}

    for mid in metric_ids:
        pipelines[mid] = MetricPipeline(metric_id=mid, name=name_by_id.get(mid, f"metric_{mid}"), steps=[])

    for r in rows_steps:
        mid = int(r["met_id"])
        if mid not in pipelines:
            continue
        inputs_raw = r.get("mim_inputs_from") or ""
        inputs_from = [tok for tok in (inputs_raw.split("|") if inputs_raw else []) if tok]

        step = MetricStep(
            step=int(r["mim_step"]),
            role=str(r["mim_role"] or "").lower(),
            lib_name=str(r["mim_lib_name"] or ""),
            kf_name=str(r["mim_kf_name"] or ""),
            op_name=str(r["mim_op_name"] or ""),
            logical_name=str(r["mim_logical_name"] or ""),
            inputs_from=inputs_from,
            artifact_file=str(r["mim_artifact_file"] or ""),
            artifact_ext=str(r["mim_artifact_ext"] or "").lstrip("."),
            fanout_index=int(r["mim_fanout_index"]) if r.get("mim_fanout_index") is not None else None,
        )
        pipelines[mid].steps.append(step)

    return pipelines


# ---------------------------------------------------------------------------
# Entry point called from OE/core.run()
# ---------------------------------------------------------------------------

def execute_metric_job(job: Dict[str, Any], pipelines: Dict[int, MetricPipeline]) -> None:
    """
    Execute a *single* metric job, including all compute/flatten/final steps.

    - job is the dict returned by ledger.fetch_job_ledger_record()
    - pipelines is the preloaded {metric_id: MetricPipeline}

    This function:
      - loads the frame's base fields (psi, phi, eta, pi) as needed
      - runs all MetricSteps for this metric_id sequentially
      - writes all final artifacts to disk (job["output_path"] and siblings)
    It does NOT touch DB status or logs; OE.run() handles that.
    """
    mid = int(job.get("metric_id") or 0)
    if not mid or mid not in pipelines:
        raise ValueError(f"execute_metric_job: unknown metric_id {mid}")

    pipeline = pipelines[mid]
    sim_id = int(job["sim_id"])
    frame = int(job.get("job_frame") or 0)

    output_path_str = job.get("output_path") or ""
    if not output_path_str:
        # OE/ledger should have finalized this before running
        raise ValueError(f"execute_metric_job: job {job['job_id']} has no output_path")

    output_path = Path(output_path_str)
    metrics_dir = output_path.parent
    frame_dir = metrics_dir.parent

    # Load sim metadata once
    sim = get_simulation_full(sim_id) or {}
    sim_shape = (
        int(sim.get("sim_gridx") or 0),
        int(sim.get("sim_gridy") or 0),
        int(sim.get("sim_gridz") or 0),
    )
    sim_dx = float(sim.get("sim_dx") or 1.0)
    sim_dt = float(sim.get("sim_dt_per_at") or 1.0)
    sim_phi_threshold = float(sim.get("sim_phi_threshold") or 0.5)

    # Artifact map for this job: keys are tokens from logical_name / inputs_from
    artifacts: Dict[str, Any] = {}

    # Preload base fields (psi, phi, eta, pi) only if needed anywhere in the pipeline
    needed_tokens = {tok for step in pipeline.steps for tok in step.inputs_from}
    base_fields = _detect_needed_frame_fields(needed_tokens)

    for field_name in base_fields:
        arr = _load_frame_field(frame_dir, field_name)
        artifacts[field_name] = arr

    # --- Execute all steps in the pipeline ---
    for step in pipeline.steps:
        # Resolve inputs
        inputs = [_resolve_artifact(artifacts, token, frame_dir, metric_id=mid, step=step) for token in step.inputs_from]

        if step.role in ("compute", "flatten"):
            # Generic numeric kernel
            out = run_kernel(
                lib_name=step.lib_name,
                kf_name=step.kf_name,
                op_name=step.op_name,
                inputs=inputs,
                sim_meta={
                    "sim_id": sim_id,
                    "shape": sim_shape,
                    "dx": sim_dx,
                    "dt": sim_dt,
                    "phi_threshold": sim_phi_threshold,
                    "frame": frame,
                    "metric_id": mid,
                    "logical_name": step.logical_name,
                },
            )
            if step.logical_name:
                artifacts[step.logical_name] = out
        elif step.role == "final":
            # Writer/serializer step — may have multiple finals per metric
            # Decide output path:
            primary_final_name = output_path.name  # basename that OE/ledger chose
            artifact_file = (step.artifact_file or "").lstrip("/")
            ext = step.artifact_ext or output_path.suffix.lstrip(".") or "dat"

            if artifact_file and artifact_file == primary_final_name:
                # Primary final: keep the exact path OE/ledger chose
                out_path = output_path
            elif artifact_file:
                # DB-specified artifact file: write under the metrics_dir
                out_path = metrics_dir / artifact_file
            else:
                # If DB didn't specify, fall back to primary final + role/op suffix in metrics_dir
                stem = output_path.stem
                out_path = metrics_dir / f"{stem}_{step.logical_name or step.op_name}.{ext}"

            run_writer(
                op_name=step.op_name,
                inputs=inputs,
                path=out_path,
                fmt=ext,
                sim_meta={
                    "sim_id": sim_id,
                    "shape": sim_shape,
                    "dx": sim_dx,
                    "dt": sim_dt,
                    "phi_threshold": sim_phi_threshold,
                    "frame": frame,
                    "metric_id": mid,
                    "logical_name": step.logical_name,
                },
            )
        else:
            raise ValueError(f"execute_metric_job: unknown role {step.role!r} for metric {mid}")


# ---------------------------------------------------------------------------
# Helpers for frame field loading and artifact resolution
# ---------------------------------------------------------------------------

_BASE_FIELD_TOKENS = {"psi", "phi", "eta", "pi"}


def _detect_needed_frame_fields(tokens: set[str]) -> List[str]:
    """From all inputs_from tokens, determine which base frame fields to load."""
    return sorted(tok for tok in tokens if tok in _BASE_FIELD_TOKENS)


def _load_frame_field(frame_dir: Path, token: str) -> np.ndarray:
    """
    Load a base field from the frame directory by token.

    This assumes the same naming as OE's sim writer:
      psi -> psi_field.npy
      phi -> phi_field.npy
      eta -> eta.npy
      pi  -> pi.npy

    If your actual filenames differ, adjust this mapping.
    """
    mapping = {
        "psi": "psi.npy",
        "phi": "phi_field.npy",
        "eta": "eta.npy",
        "pi": "pi.npy",
    }
    fname = mapping.get(token)
    if not fname:
        raise ValueError(f"Unknown base field token {token!r}")
    path = frame_dir / fname
    if not path.is_file():
        raise FileNotFoundError(f"Frame field {token!r} not found at {path}")
    return np.load(path)


def _resolve_artifact(
    artifacts: Dict[str, Any],
    token: str,
    frame_dir: Path,
    metric_id: int,
    step: MetricStep,
) -> Any:
    """
    Resolve an input token to an artifact array/object.

    Priority:
      1. artifacts[token] if present (in-memory result of a previous step)
      2. For base fields, this should have been preloaded
      3. For some tokens (e.g. 'collapse_mask', 'labels') you may later add
         disk fallbacks (loading artifacts written by other metrics).

    For now we require that any non-base token be produced within the pipeline
    of this job; otherwise this is a wiring/ordering error.
    """
    if token in artifacts:
        return artifacts[token]

    if token in _BASE_FIELD_TOKENS:
        # Should have been preloaded
        raise RuntimeError(f"Base field {token!r} not preloaded in artifacts")

    # TODO: if you later want cross-metric re-use, this is where you'd
    #       look for artifacts from other metrics on disk.
    raise KeyError(f"Artifact {token!r} not found for metric {metric_id}, step {step.step} ({step.logical_name})")


# ---------------------------------------------------------------------------
# Kernel dispatch stubs
# ---------------------------------------------------------------------------

def run_kernel(
    lib_name: str,
    kf_name: str,
    op_name: str,
    inputs: List[Any],
    sim_meta: Dict[str, Any],
) -> Any:
    """
    Generic numeric kernel dispatcher.

    This is where you bridge from DB wiring (lib_name/kf_name/op_name)
    to your actual math implementations (pyfftw, numpy, scipy.ndimage,
    itk, gudhi, etc.).

    For now, this is a skeleton: you must fill in the actual call-sites
    to your libraries.

    Example sketch:

      if lib_name == "pyfftw" and kf_name == "fft" and op_name == "power_spectrum":
          psi = inputs[0]
          return compute_power_spectrum_pyfftw(psi, sim_meta)

      elif lib_name == "scipy.ndimage" and kf_name == "morph" and op_name == "threshold":
          phi = inputs[0]
          return compute_collapse_mask(phi, sim_meta["phi_threshold"])

      elif lib_name == "gudhi" and kf_name == "topology" and op_name == "betti":
          labels = inputs[0]
          return compute_betti_numbers(labels, sim_meta)

      ...

    Anything not wired yet should raise NotImplementedError to fail fast.
    """
    key = (lib_name, kf_name, op_name)

    # numpy-based reductions used by scalar metrics (e.g. eta_min/max/mean/std, psi_* stats)
    if key == ("numpy", "reduce", "minmaxmeanstd"):
        import numpy as np
        arr = np.asarray(inputs[0])
        # [min, max, mean, std] over the entire field
        return np.array(
            [arr.min(), arr.max(), arr.mean(), arr.std()],
            dtype=float,
        )

    elif key == ("numpy", "reduce", "hist"):
        # For simple stats metrics, we currently treat "hist" as a pass-through
        # for the already-reduced statistics. If you later want a true histogram,
        # this branch can be extended.
        return inputs[0]

    # Fallback for anything not wired yet
    raise NotImplementedError(f"run_kernel not wired for {key}")


def run_writer(
    op_name: str,
    inputs: List[Any],
    path: Path,
    fmt: str,
    sim_meta: Dict[str, Any],
) -> None:
    """
    Writer/serializer dispatcher for 'final' steps.

    This handles writer/serialize operations like:
      - csv: write scalar/row dicts to CSV
      - betti: write Betti tables
      - mask_final: write NPY mask
      - threshold: write per-voxel stats
      - shellscan: write JSON echo list

    For now, this is a skeleton. You will plug in your real IO helpers.

    Example sketch:

      if op_name == "csv":
          write_scalar_csv(inputs[0], path)
      elif op_name == "mask_final":
          np.save(path, inputs[0])
      elif op_name == "betti":
          write_betti_csv(inputs[0], path)
      elif op_name == "shellscan":
          write_echo_profile_json(inputs, path)
      else:
          raise NotImplementedError(...)

    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Simple CSV writer for scalar / small-vector metrics
    if op_name == "csv":
        import csv
        import numpy as np

        data = inputs[0]
        arr = np.atleast_1d(np.asarray(data))

        # Build header: sim/frame/metric/logical_name + one column per value
        n = int(arr.size)
        header = ["sim_id", "frame", "metric_id", "logical_name"] + [f"value_{i}" for i in range(n)]

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            row = [
                sim_meta.get("sim_id"),
                sim_meta.get("frame"),
                sim_meta.get("metric_id"),
                sim_meta.get("logical_name"),
            ] + [float(x) for x in arr.ravel()]
            writer.writerow(row)
        return

    # Fallback for anything not wired yet
    raise NotImplementedError(f"run_writer not wired for op_name={op_name!r}, fmt={fmt!r}, path={path}")