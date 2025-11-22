from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _iter_frame_dirs(sim_root: Path) -> List[Path]:
    """Return sorted Frame_XXXX directories under sim_root."""
    if not sim_root.exists():
        return []
    frames: List[Path] = []
    for p in sim_root.iterdir():
        if p.is_dir() and p.name.startswith("Frame_"):
            frames.append(p)
    frames.sort(key=lambda p: int(p.name.split("_")[-1]))
    return frames


def _bundle_csv_per_metric(
    sim_root: Path,
    frame_dirs: List[Path],
    run_ts: str,
) -> None:
    """
    Aggregate per-frame CSVs into combined CSVs under:

        sim_root / "bundle" / run_ts / "<metric>_combined.csv"

    Strategy:
      - Look at CSV files under each Frame_XXXX/run_ts subdirectory.
      - Group by basename (e.g. 'coherence_length.csv', 'betti_numbers.csv', ...).
      - For each basename, read one row per frame and append into a combined CSV.
    """
    bundle_dir = sim_root / "bundle" / run_ts
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Collect: basename -> list of (frame_index, csv_path)
    by_name: Dict[str, List[tuple[int, Path]]] = {}

    for frame_dir in frame_dirs:
        try:
            frame_idx = int(frame_dir.name.split("_")[-1])
        except Exception:
            continue
        subdir = frame_dir / run_ts
        if not subdir.exists() or not subdir.is_dir():
            continue
        for csv_path in subdir.glob("*.csv"):
            if not csv_path.is_file():
                continue
            name = csv_path.name
            by_name.setdefault(name, []).append((frame_idx, csv_path))

    for name, entries in by_name.items():
        if not entries:
            continue
        # Sort by frame index
        entries.sort(key=lambda t: t[0])

        # Combined file name: <basename_without_ext>_combined.csv
        stem = name.rsplit(".", 1)[0]
        out_name = f"{stem}_combined.csv"
        out_path = bundle_dir / out_name

        header: List[str] | None = None
        rows: List[List[Any]] = []

        for frame_idx, csv_path in entries:
            try:
                with csv_path.open("r", newline="") as f:
                    reader = csv.reader(f)
                    this_header = next(reader, None)
                    if this_header is None:
                        continue
                    if header is None:
                        header = this_header
                    # Simple header consistency: same length
                    elif len(this_header) != len(header):
                        continue
                    row = next(reader, None)
                    if row is None:
                        continue
                    rows.append(row)
            except Exception:
                # Ignore individual file errors; bundling must not break OE
                continue

        if header is None or not rows:
            continue

        try:
            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
        except Exception:
            # Bundling for this metric failed; continue with others
            continue


def _build_npy_index(
    sim_root: Path,
    frame_dirs: List[Path],
    run_ts: str,
) -> None:
    """
    Build simple JSON index files for NPY metrics for a given metrics run.

    For each *.npy under Frame_XXXX/run_ts:
      - group by filename (e.g. 'collapse_mask.npy')
      - write bundle/run_ts/<name>_index.json with frame + relative path entries.

    Base PDE fields (psi.npy, pi.npy, eta.npy, phi_field.npy, phi_cone.npy)
    are skipped; only metric NPYs are indexed.
    """
    bundle_dir = sim_root / "bundle" / run_ts
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # name -> list of {frame:int, path:str}
    index: Dict[str, List[Dict[str, Any]]] = {}

    # Only NPY filenames we consider "metric" NPYs.
    METRIC_NPY_NAMES = {
        "collapse_mask.npy",
        # later: add other metric NPY names here if needed
    }

    for frame_dir in frame_dirs:
        try:
            frame_idx = int(frame_dir.name.split("_")[-1])
        except Exception:
            continue
        subdir = frame_dir / run_ts
        if not subdir.exists() or not subdir.is_dir():
            continue
        for npy_path in subdir.glob("*.npy"):
            if not npy_path.is_file():
                continue
            name = npy_path.name
            if name not in METRIC_NPY_NAMES:
                # Skip base fields and non-metric NPYs
                continue
            rel = npy_path.relative_to(sim_root)
            index.setdefault(name, []).append(
                {"frame": frame_idx, "path": str(rel)}
            )

    for name, entries in index.items():
        if not entries:
            continue
        out_path = bundle_dir / f"{name}_index.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metric": name,
                        "sim_root": str(sim_root),
                        "run_ts": run_ts,
                        "frames": sorted(
                            entries, key=lambda e: int(e.get("frame", 0))
                        ),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            # Index generation failure must not break OE
            continue


def bundle_metrics_for_sim(jobs: List[Dict[str, Any]]) -> None:
    """
    Aggregate per-frame metric outputs for a completed metrics run.

    - Derives sim_root and run timestamp (e.g. '20251122_0450') from the first
      metric job's output_path:
        /data/simulations/{label}/{tt}/Frame_NNNN/{run_ts}/{metric_id}.ext
      sim_root = parents[2]
      run_ts   = parents[0].name
    - Scans Frame_XXXX/run_ts/ under sim_root.
    - For each CSV metric, builds a combined CSV in:
        sim_root/bundle/run_ts/<metric>_combined.csv
    - For NPY metrics (e.g. collapse_mask.npy), builds an index JSON in
        sim_root/bundle/run_ts/<name>_index.json

    This function is idempotent: re-running it will overwrite combined CSV/JSON
    deterministically. It should never raise; errors are only printed.
    """
    # Filter to metric jobs that actually wrote something
    metric_jobs = [
        j
        for j in jobs
        if j.get("metric_id") is not None and j.get("output_path")
    ]
    if not metric_jobs:
        return

    from pathlib import Path as _Path

    try:
        first_path = _Path(metric_jobs[0]["output_path"])
    except Exception:
        return

    # /data/simulations/{label}/{tt}/Frame_NNNN/{run_ts}/{metric_id}.ext
    # sim_root = parents[2] → /data/simulations/{label}/{tt}
    try:
        sim_root = first_path.parents[2]
        run_ts = first_path.parents[0].name
    except Exception:
        return

    if not sim_root.exists():
        return

    frame_dirs = _iter_frame_dirs(sim_root)
    if not frame_dirs:
        return

    try:
        _bundle_csv_per_metric(sim_root, frame_dirs, run_ts)
        _build_npy_index(sim_root, frame_dirs, run_ts)
    except Exception as e:
        # Bundling must never break OE; log and continue
        print(f"[MetricsBundler] ⚠ bundling failed for {sim_root} (run_ts={run_ts}): {e}")