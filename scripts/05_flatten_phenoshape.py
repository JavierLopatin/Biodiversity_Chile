#!/usr/bin/env python3
"""Flatten the stored PhenoShape curves from 1,082 NetCDF cubes into modelling tables.

The cubes hold `phenoshape` as (index, doy, y, x) per plot, which is the right shape for
storage but the wrong shape for fitting anything. This script emits the same curves as
flat tables keyed on (plot_id, index), so they join directly against
`lsp_all_auto.parquet` on those two columns.

Three products, because two different models want two different things:

  phenoshape_by_index      one row per (plot_id, index, px)   -- px in {center, mean5x5}
                           10,820 rows. The compact table: Random Forest on the curve,
                           or any model that treats the plot as one observation.

  phenoshape_pixels        one row per (plot_id, index, y, x) -- all 25 pixels
                           135,250 rows. The augmentation substrate for the CNN.
                           NOT independent: 25 rows per plot share a footprint, so they
                           must never be split across CV folds.

  phenoshape_doy_grid      one row per plot_id
                           The actual day-of-year of each of the 52 steps.

Why the grid needs its own table. The 52 steps are *positional*, not a shared calendar:
each plot gets its own 52-step grid over its own observation window, all rotated to the
same global trough anchor (DOY 108). Measured across plots the grids differ by at most
2 days at any step, so treating step i as comparable between plots is safe -- but that is
an empirical fact about this run, not a guarantee, so the real DOY values ship alongside.

Usage:
    python scripts/05_flatten_phenoshape.py
    python scripts/05_flatten_phenoshape.py --csv --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

NGS = 52  # curve steps, must match --ngs of scripts/02
STEP_COLS = [f"s{i:02d}" for i in range(NGS)]


def flatten(pheno_dir: Path, limit: int | None = None):
    files = sorted(pheno_dir.glob("*.nc"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no .nc files in {pheno_dir}")

    compact, pixels, grids = [], [], []
    anchor_seen: set[int] = set()

    for k, f in enumerate(files, 1):
        with xr.open_dataset(f) as ds:
            pid = str(ds.attrs["plot_id"])
            idx = [str(v) for v in np.asarray(ds["index"])]
            doy = np.asarray(ds.doy).astype(int)
            ph = np.asarray(ds.phenoshape, dtype=np.float32)  # (index, doy, y, x)
            anchor_seen.add(int(ds.attrs["doy_anchor"]))

            if ph.shape[1] != NGS:
                raise SystemExit(f"{f.name}: {ph.shape[1]} steps, expected {NGS}")

            ny, nx = ph.shape[2], ph.shape[3]
            cy, cx = ny // 2, nx // 2

            grids.append({"plot_id": pid, **dict(zip(STEP_COLS, doy))})

            for i, name in enumerate(idx):
                compact.append({"plot_id": pid, "index": name, "px": "center",
                                **dict(zip(STEP_COLS, ph[i, :, cy, cx]))})
                compact.append({"plot_id": pid, "index": name, "px": "mean5x5",
                                **dict(zip(STEP_COLS, np.nanmean(ph[i], axis=(1, 2))))})
                for yy in range(ny):
                    for xx in range(nx):
                        pixels.append({"plot_id": pid, "index": name, "y": yy, "x": xx,
                                       **dict(zip(STEP_COLS, ph[i, :, yy, xx]))})

        if k % 200 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] {pid}")

    if len(anchor_seen) > 1:
        print(f"  WARNING: mixed doy_anchor across plots: {sorted(anchor_seen)}")

    return (pd.DataFrame(compact), pd.DataFrame(pixels),
            pd.DataFrame(grids), sorted(anchor_seen))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pheno-dir", default="data/derived/phenology")
    p.add_argument("--out-dir", default="data/derived")
    p.add_argument("--csv", action="store_true",
                   help="also write CSV (compact table and grid only; the per-pixel "
                        "table is ~70 MB as CSV and stays parquet-only)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"flattening {args.pheno_dir} ...")
    compact, pixels, grid, anchors = flatten(Path(args.pheno_dir), args.limit)

    for df, name in ((compact, "phenoshape_by_index"),
                     (pixels, "phenoshape_pixels"),
                     (grid, "phenoshape_doy_grid")):
        fp = out / f"{name}.parquet"
        df.to_parquet(fp, index=False)
        print(f"  {name:22s} {df.shape[0]:>7,} x {df.shape[1]:<4} -> {fp} "
              f"({fp.stat().st_size / 1e6:.1f} MB)")
        if args.csv and name != "phenoshape_pixels":
            fc = out / f"{name}.csv"
            df.to_csv(fc, index=False)
            print(f"  {'':22s} {'':>7} {'':4}    {fc} ({fc.stat().st_size / 1e6:.1f} MB)")

    n_plots = compact.plot_id.nunique()
    print(f"\nplots: {n_plots} | indices: {sorted(compact['index'].unique())}")
    print(f"doy_anchor: {anchors} (global rotation, trough_anchored)")
    print(f"steps: {NGS}  columns {STEP_COLS[0]}..{STEP_COLS[-1]} (positional; "
          f"real DOY in phenoshape_doy_grid)")
    print("\njoin key against lsp_all_auto.parquet: (plot_id, index)")


if __name__ == "__main__":
    main()
