#!/usr/bin/env python3
"""Route 1 of docs/09_predictors.md section 5: composites straight from the stored cubes.

``data/derived/phenology/{plot_id}.nc`` kept the full observation series next to the fitted
curve — ``obs_{ndvi,evi,kndvi,nbr,savi}`` and ``obs_band_{blue,...,swir2}``, all
``(time, y, x)`` and already cloud-masked. Everything below is therefore a local
computation: no datacube, no requester-pays, no S3.

Three families of predictor come out, all per pixel and all free of temporal shape:

  ``gm_*``      the six-band geometric median plus the five indices derived from it, and the
                three MADs (``emad``, ``smad``, ``bcmad``) that measure scatter about it.

  ``obs_*``     distribution statistics of each index over the *real observations* rather
                than over the 52-step interpolation. This is a different question from the
                composite block built on the curve: that one already carries the smoother's
                fingerprint, this one does not.

  ``seas_*``    median per austral season and the summer-minus-winter contrast — phenology
                reduced to two numbers, the crudest control against a fitted curve.

Written per pixel, never pre-aggregated. How to summarise 25 pixels into a plot is a
modelling decision (median? trimmed mean? which pixels are even valid?) and the current
``*_mean5x5`` columns made it silently, with ``np.nanmean`` over between 1 and 25 pixels and
nothing recording which. ``scripts/18b_aggregate_cube_predictors.py`` makes that choice
explicit and auditable.

Usage:
    python scripts/18_geomedian_from_cubes.py
    python scripts/18_geomedian_from_cubes.py --limit 50      # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import geomedian as gmod                                     # noqa: E402

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]


def process_plot(path: Path) -> list[dict]:
    """One row per pixel of one plot."""
    with xr.open_dataset(path) as ds:
        plot_id = ds.attrs["plot_id"]
        ny, nx = ds.sizes["y"], ds.sizes["x"]
        month = pd.to_datetime(ds.time.values).month.to_numpy()
        bands = np.stack([ds[f"obs_band_{b}"].values for b in gmod.BANDS], axis=-1)
        idx = {n: ds[f"obs_{n}"].values for n in INDICES}

    rows = []
    for iy in range(ny):
        for ix in range(nx):
            row: dict = {"plot_id": plot_id, "y": iy, "x": ix}

            X = bands[:, iy, ix, :]
            ok = np.isfinite(X).all(axis=1)
            row["gm_count"] = float(ok.sum())
            if ok.sum() >= gmod.MIN_OBS:
                Xo = X[ok]
                gm = gmod.geometric_median(Xo)
                for b, v in zip(gmod.BANDS, gm):
                    row[f"gm_band_{b}"] = float(v)
                for name, v in gmod.indices_from_bands(dict(zip(gmod.BANDS, gm))).items():
                    row[f"gm_{name}"] = v
                row["gm_emad"], row["gm_smad"], row["gm_bcmad"] = gmod.mads(Xo, gm)

            for name in INDICES:
                v = idx[name][:, iy, ix]
                for k, val in gmod.composite_stats(v).items():
                    row[f"obs_{name}_{k}"] = val
                for k, val in gmod.seasonal_medians(v, month).items():
                    row[f"seas_{name}_{k}"] = val
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cubes", default="data/derived/phenology")
    p.add_argument("--out", default="data/derived/cube_predictors_pixels.parquet")
    p.add_argument("--limit", type=int, default=0, help="process only the first N cubes")
    args = p.parse_args()

    files = sorted(Path(args.cubes).glob("*.nc"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no .nc cubes under {args.cubes}")
    print(f"{len(files)} cubos", flush=True)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0, rows = time.time(), []
    for i, f in enumerate(files, 1):
        rows.extend(process_plot(f))
        if i % 100 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i}/{len(files)}  {el:6.1f}s  eta {el / i * (len(files) - i):6.1f}s",
                  flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    n_px = len(df)
    gm_ok = int(df["gm_ndvi"].notna().sum()) if "gm_ndvi" in df else 0
    print(f"\n-> {args.out}  ({n_px:,} filas x {df.shape[1]} columnas)")
    print(f"   pixeles con geomediana: {gm_ok:,} / {n_px:,} ({100 * gm_ok / n_px:.1f}%)")
    print(f"   observaciones limpias por pixel: mediana {df['gm_count'].median():.0f}, "
          f"min {df['gm_count'].min():.0f}, max {df['gm_count'].max():.0f}")
    print(f"   parcelas: {df['plot_id'].nunique()}")


if __name__ == "__main__":
    main()
