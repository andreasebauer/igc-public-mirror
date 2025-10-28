#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv

try:
    import imageio.v3 as iio
    HAS_IMG = True
except Exception:
    HAS_IMG = False

def main():
    ap = argparse.ArgumentParser(description="Filesystem-only collapse_mask (pp0 smoke test)")
    ap.add_argument("--store", default="/data/igc", help="Root store")
    ap.add_argument("--sim", required=True, help="Sim label, e.g. DEV_pp0")
    ap.add_argument("--frame", type=int, required=True, help="Frame number, e.g. 0")
    ap.add_argument("--tau", type=float, default=0.5, help="phi threshold")
    args = ap.parse_args()

    fdir = Path(args.store) / f"Sim_{args.sim}" / f"Frame_{args.frame:04d}"
    phi_path = fdir / "phi_field.npy"
    if not phi_path.exists():
        raise SystemExit(f"Missing {phi_path}")

    phi = np.load(phi_path)  # float64
    mask = (phi >= args.tau)

    # Write outputs under Metric_collapse_mask/Step_3/
    out_dir = fdir / "Metric_collapse_mask" / "Step_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "collapse_mask.npy"
    np.save(mask_path, mask)

    # CSV summary
    voxels_total = mask.size
    voxels_solid = int(mask.sum())
    fraction = voxels_solid / voxels_total if voxels_total > 0 else 0.0

    csv_path = out_dir / "collapse_mask.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_label","frame","phi_threshold","voxels_solid","voxels_total","fraction_solid"])
        w.writerow([args.sim, args.frame, args.tau, voxels_solid, voxels_total, f"{fraction:.9f}"])

    # Optional center-slice PNG (helps eyeballing)
    if HAS_IMG and phi.ndim == 3:
        z0 = phi.shape[2] // 2
        # normalize phi to 0..255 for a quick grayscale
        ph2d = phi[:,:,z0]
        pmin, pmax = float(ph2d.min()), float(ph2d.max())
        if pmax > pmin:
            norm = (ph2d - pmin) / (pmax - pmin)
        else:
            norm = np.zeros_like(ph2d)
        rgba = (norm * 255).astype(np.uint8)
        png_path = out_dir / "phi_center_slice.png"
        iio.imwrite(png_path, rgba)

        # mask slice overlay (white where sealed)
        m2d = mask[:,:,z0].astype(np.uint8) * 255
        iio.imwrite(out_dir / "collapse_mask_center_slice.png", m2d)

    print(f"Wrote: {mask_path}")
    print(f"Wrote: {csv_path}")
    if HAS_IMG:
        print(f"(Optional PNGs) in {out_dir}")

if __name__ == "__main__":
    main()
