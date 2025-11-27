from __future__ import annotations

import csv
import json
import zipfile
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

def bundle_metrics_for_sweep(sweep_root: Path, member_roots: List[Path]) -> None:
    """
    Aggregate per-sim combined CSV metrics for a sweep into sweep-level master CSVs.

    For each metric combined CSV:
        member_root/bundle/<run_ts>/<metric>_combined.csv

    This function creates:
        sweep_root/bundle_sweep/<metric>_combined_master.csv

    where each master CSV:
      - adds a "sweep_param_value" column right after "sim_id" (or prepends
        both sim_id and sweep_param_value if sim_id is not present),
      - contains rows from all member files.
    It also creates:
        sweep_root/bundle_sweep/sim_meta.json
        sweep_root/bundle_sweep.zip

    bundling must never break the caller; all errors are swallowed.
    """
    sweep_root = Path(sweep_root)
    sweep_bundle_root = sweep_root / "bundle_sweep"
    sweep_bundle_root.mkdir(parents=True, exist_ok=True)

    # metric_name (e.g. "betti_numbers_combined.csv") ->
    #   {"header": List[str], "rows": List[Tuple[int, str, List[str]]]}
    metrics: Dict[str, Dict[str, Any]] = {}

    # Collect sweep meta information for sweep-level sim_meta.json
    sweep_param: str | None = None
    sweep_members: List[Dict[str, Any]] = []
    base_sim_id: int | None = None
    base_label: str | None = None

    for member in member_roots:
        member = Path(member)
        if not member.is_dir():
            continue

        # 1) Load member sim_meta.json to get sim_id, sweep_param, sweep_value, label
        meta_path = member / "sim_meta.json"
        sim_id: int | None = None
        param_name: str | None = None
        param_value: Any | None = None
        label: str | None = None

        try:
            if meta_path.is_file():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                sim_id = int(meta.get("sim_id") or 0) or None
                param_name = meta.get("sweep_param") or None
                param_value = meta.get("sweep_value")
                label = (meta.get("label") or meta.get("sim_label") or "").strip() or None
        except Exception:
            # If we cannot read meta, skip this member entirely.
            continue

        if sim_id is None:
            # Without sim_id we cannot meaningfully tag rows
            continue

        # Initialize sweep-level meta once
        if base_sim_id is None:
            base_sim_id = sim_id
        if base_label is None and label:
            base_label = label
        if sweep_param is None and param_name:
            sweep_param = param_name

        # Human-readable tag, e.g. "k_psi_restore_0.1"
        if param_name is not None and param_value is not None:
            tag = f"{param_name}_{param_value}"
        else:
            tag = ""

        sweep_members.append(
            {
                "member_root": str(member),
                "sim_id": sim_id,
                "sweep_param": param_name,
                "sweep_value": param_value,
                "tag": tag,
            }
        )

        # 2) Scan this member's bundle/<run_ts>/*_combined.csv
        bundle_dir = member / "bundle"
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            continue

        for run_dir in bundle_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for csv_path in run_dir.glob("*_combined.csv"):
                if not csv_path.is_file():
                    continue
                name = csv_path.name  # e.g. "betti_numbers_combined.csv"
                try:
                    with csv_path.open("r", newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header is None:
                            continue
                        entry = metrics.setdefault(
                            name, {"header": header, "rows": []}
                        )
                        # Simple header consistency: same length
                        if len(header) != len(entry["header"]):  # type: ignore[arg-type]
                            continue
                        for row in reader:
                            if not row:
                                continue
                            # Store (sim_id, tag, row_values)
                            entry["rows"].append((sim_id, tag, row))  # type: ignore[index]
                except Exception:
                    # Ignore individual file errors; sweep bundling must not break metrics
                    continue

    # 3) Write master CSVs with extra sweep_param_value column
    for name, data in metrics.items():
        orig_header: List[str] = data.get("header") or []  # type: ignore[assignment]
        rows: List[Any] = data.get("rows") or []           # type: ignore[assignment]
        if not orig_header or not rows:
            continue

        # Determine position of sim_id in original header (if present)
        try:
            sim_idx = orig_header.index("sim_id")
        except ValueError:
            sim_idx = -1

        # Build new header: sim_id, sweep_param_value, then rest.
        if sim_idx == 0:
            # Header starts with sim_id, e.g. ["sim_id", "frame", ...]
            new_header = ["sim_id", "sweep_param_value"] + orig_header[1:]
        else:
            # No sim_id column in source; prepend both
            new_header = ["sim_id", "sweep_param_value"] + orig_header

        stem = name.rsplit(".", 1)[0]
        out_name = f"{stem}_master.csv"
        out_path = sweep_bundle_root / out_name

        try:
            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(new_header)

                for sim_id, tag, row in rows:
                    if sim_idx == 0 and len(row) >= 1:
                        # sim_id already present in row[0]
                        # replace it with our sim_id (from meta) to be safe
                        base_row = row[1:]
                        new_row = [sim_id, tag] + base_row
                    else:
                        # sim_id not present in original row; prepend both
                        new_row = [sim_id, tag] + row
                    writer.writerow(new_row)
        except Exception:
            # Sweep-level bundling must never break the caller
            continue

    # 4) Write a sweep-level sim_meta.json with summary information
    try:
        meta_out = {
            "sim_id": base_sim_id,
            "label": base_label,
            "sweep_param": sweep_param,
            "members": sweep_members,
        }
        with (sweep_bundle_root / "sim_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, default=str)
    except Exception:
        # Not fatal if we cannot write sweep-level meta
        pass

    # 5) Package bundle_sweep directory into a zip for convenience
    try:
        # Derive a descriptive ZIP name from sweep_param and sweep values if possible.
        # We have sweep_param and sweep_members from above.
        fname = "bundle_sweep.zip"
        try:
            vals = []
            for m in sweep_members:
                v = m.get("sweep_value")
                if v is not None:
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
            if sweep_param and vals:
                vals_sorted = sorted(vals)
                vmin = vals_sorted[0]
                vmax = vals_sorted[-1]

                # Turn float values into safe tokens, e.g. 0.001 -> "0_001"
                def token(x: float) -> str:
                    s = f"{x}"
                    # Replace characters that are problematic in filenames
                    return (
                        s.replace("-", "m")
                         .replace("+", "")
                         .replace(".", "_")
                    )

                fname = f"bundle_sweep_{sweep_param}_{token(vmin)}_to_{token(vmax)}.zip"
        except Exception:
            # Fall back to default name if anything goes wrong
            fname = "bundle_sweep.zip"

        zip_path = sweep_root / fname
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sweep_bundle_root.rglob("*"):
                if path.is_file():
                    # Store paths inside the zip relative to the sweep_root
                    arcname = path.relative_to(sweep_root)
                    zf.write(path, arcname)
    except Exception:
        # Zip creation is optional; ignore errors
        pass     