#!/usr/bin/env python3
"""Raw-series curves for Living Trees Chile, adapted to the same flat format as
Parcelas-CL's `phenoshape_by_index_raw{100,36}.parquet` (`scripts/29_refit_curves_from_cubes.py
--raw-series`), then concatenated into unified files for the whole 3,102-plot pool.

Living Trees' satellite series (`data/derived/living_trees/series_<index>_<px>.parquet`,
one row per real observation, `site_id`/`time`/`sensor`/value) never went through the
NetCDF-cube pipeline Parcelas-CL's curves are built from -- there is no per-plot cube to
call `biodiv.curves.raw_series()` on. This calls the SAME underlying function,
`biodiv.curves.interp_grid(t, v, ngs, roll, t_min, t_max)`, directly per site (`raw_series`
just loops it over a (y,x) pixel grid Living Trees never had) -- confirmed by reading
`curves.py`'s source: `interp_grid` only needs 1-D `t`/`v`, it never touches the grid.

Crosswalk: `site_id` (`sites.parquet`, satellite side) <-> `PlotObservationID`
(`living_trees_plots.parquet`, species side) via exact lat/lon match -- verified this
session, 2,020/2,020 matched, 0 ambiguous (the one satellite-only site, `LT1717`, is the
already-diagnosed all-unidentified-species plot, correctly absent from the species side).

`px="mean5x5"` caveat, stated once here rather than silently: Parcelas-CL's mean5x5 curve
averages 25 per-pixel FITTED curves; Living Trees' `series_<var>_mean5x5.parquet` only ever
stored the already-spatially-averaged RAW OBSERVATIONS (no per-pixel grid was retained at
extraction time) -- so Living Trees' mean5x5 curve is average-then-interpolate, Parcelas-CL's
is interpolate-then-average. Real, small, documented -- not pretended away. `px="center"`
has no such issue (a single pixel in both sources).

Output is additive: `phenoshape_by_index_raw{100,36}_unified.parquet` are NEW files,
concatenating Parcelas-CL's existing curves (re-prefixed `PCL_`, matching
`scripts/51_build_unified_dataset.py`'s convention) with Living Trees' new ones
(`PlotObservationID` as-is -- it already carries the `LT_` prefix, minted by
`scripts/50_build_living_trees.py`) -- the originals are never touched.

Usage:
    python scripts/67_build_living_trees_curves.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv.curves import interp_grid  # noqa: E402

DERIVED = Path("data/derived")
LT_DIR = DERIVED / "living_trees"
INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
PX_LEVELS = ["center", "mean5x5"]
NGS_VARIANTS = [100, 36]
ROLL = 5


def build_crosswalk() -> pd.DataFrame:
    lt_plots = pd.read_parquet(DERIVED / "living_trees_plots.parquet")
    sites = pd.read_parquet(LT_DIR / "sites.parquet")
    cw = lt_plots.merge(sites[["site_id", "lat", "lon"]], on=["lat", "lon"], how="left")
    n_missing = cw["site_id"].isna().sum()
    if n_missing:
        raise SystemExit(f"{n_missing} Living Trees plots sin site_id -- revisar antes de seguir")
    return cw[["PlotObservationID", "site_id"]]


def build_index_px(index: str, px: str, cw: pd.DataFrame, ngs: int) -> pd.DataFrame:
    series = pd.read_parquet(LT_DIR / f"series_{index}_{px}.parquet")
    series = series.merge(cw, on="site_id", how="inner")  # drops the 1 species-less site
    val_col = index

    rows = []
    n_short = 0
    for pid, g in series.groupby("PlotObservationID"):
        t = g["time"].values.astype("datetime64[D]").astype(float)
        v = g[val_col].values.astype(float)
        curve = interp_grid(t, v, ngs, roll=ROLL, t_min=t.min(), t_max=t.max())
        if np.isnan(curve).all():
            n_short += 1
        # `PlotObservationID` already carries the "LT_" prefix (scripts/50) -- do not add
        # a second one, unlike Parcelas-CL's plot_id which needs "PCL_" added here.
        row = {"plot_id": str(pid), "index": index, "px": px}
        row.update({f"s{k:02d}": float(c) for k, c in enumerate(curve)})
        rows.append(row)
    if n_short:
        print(f"    {index}/{px} ngs={ngs}: {n_short} parcelas bajo el piso de 5 obs (NaN)")
    return pd.DataFrame(rows)


def main() -> None:
    cw = build_crosswalk()
    print(f"crosswalk: {len(cw)} parcelas Living Trees con site_id")

    for ngs in NGS_VARIANTS:
        sfx = f"_raw{ngs}"
        print(f"\n=== ngs={ngs} ===")
        lt_frames = []
        for index in INDICES:
            for px in PX_LEVELS:
                lt_frames.append(build_index_px(index, px, cw, ngs))
        lt_curves = pd.concat(lt_frames, ignore_index=True)

        pcl_path = DERIVED / f"phenoshape_by_index{sfx}.parquet"
        pcl_curves = pd.read_parquet(pcl_path)
        pcl_curves = pcl_curves.copy()
        pcl_curves["plot_id"] = "PCL_" + pcl_curves["plot_id"].astype(str)

        unified = pd.concat([pcl_curves, lt_curves], ignore_index=True)
        out_path = DERIVED / f"phenoshape_by_index{sfx}_unified.parquet"
        unified.to_parquet(out_path, index=False)

        print(f"  Parcelas-CL: {len(pcl_curves)} filas, {pcl_curves['plot_id'].nunique()} parcelas")
        print(f"  Living Trees: {len(lt_curves)} filas, {lt_curves['plot_id'].nunique()} parcelas")
        print(f"  -> {out_path} ({len(unified)} filas, {unified['plot_id'].nunique()} parcelas)")


if __name__ == "__main__":
    main()
