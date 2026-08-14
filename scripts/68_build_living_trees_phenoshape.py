#!/usr/bin/env python3
"""PhenoShape composite curves for Living Trees Chile, anchored to the same DOY-108,
7-day-step grid Parcelas-CL's stored composite already uses, then concatenated into a
unified `phenoshape_by_index_unified.parquet` + `phenoshape_doy_grid_unified.parquet`.

Fits `phenosensing.pheno.PhenoShape` (harmonic, `n_harmonics=3`, `rollWindow=5`,
`nGS=52` -- same defaults as `scripts/29_refit_curves_from_cubes.py --recon harmonic`)
on a synthetic `(time, y=1, x=1)` DataArray built from each plot's flat satellite series
(`data/derived/living_trees/series_<index>_<px>.parquet`) -- confirmed feasible this
session on a real plot before committing to the full run: `PhenoShape` only needs a
`doy` coordinate along `time`, it never touches `y`/`x` beyond their size.

**The anchor problem, and why it needs fixing, not just fitting.** `PhenoShape`'s own
output grid is `linspace(min(observed doy), max(observed doy), nGS)` -- whatever that
plot's own data happens to span. Parcelas-CL's stored `phenoshape_by_index.parquet` (no
suffix) instead lands on a strikingly consistent DOY~108, 7-day-step grid for every one
of its 1,082 plots (`s00` measured 106-111, `(s01-s00)` exactly 7 for all of them) --
`src/biodiv/substrates.py`'s `_rotate` calls this "the stored global anchor (DOY 108)"
explicitly. Living Trees spans the whole country (Coquimbo to Magallanes, 2011-2020,
not Parcelas-CL's 30-38 S band) and will not land on that grid by coincidence -- fitting
it unaligned would make "step 0" mean a different calendar date per source, silently
breaking the seasonal-shape comparison the 2D substrates are built to make.

Fix: fit PhenoShape naturally per plot (its own min/max-doy grid), then resample the
resulting 52-point curve circularly onto the FIXED target grid `108 + 7*k (mod 365)`
via `np.interp(..., period=365)` -- the same circular-DOY-alignment operation already
used in this codebase for a related purpose (`curve_level_at()`,
`scripts/29_refit_curves_from_cubes.py`), not a new technique.

Usage:
    python scripts/68_build_living_trees_phenoshape.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/mnt/rapidita_4T/GitHub/PhenoSensing")
import phenosensing  # noqa: E402,F401  (registers the .pheno accessor)

DERIVED = Path("data/derived")
LT_DIR = DERIVED / "living_trees"
INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
PX_LEVELS = ["center", "mean5x5"]
NGS = 52
ROLL = 5
N_HARMONICS = 3

#: Target grid Parcelas-CL's stored composite already lands on -- measured this session
#: (phenoshape_doy_grid.parquet: s00 = 106-111 across all 1,082 plots, mode 108; step
#: exactly 7 for every plot, every step). Fixed here so both sources share one "step 0
#: = calendar date" convention, per `_rotate`'s "stored global anchor (DOY 108)".
ANCHOR_DOY = 108
TARGET_DOY = (ANCHOR_DOY + 7 * np.arange(NGS)) % 365
TARGET_DOY = np.where(TARGET_DOY == 0, 365, TARGET_DOY).astype(float)


def build_crosswalk() -> pd.DataFrame:
    lt_plots = pd.read_parquet(DERIVED / "living_trees_plots.parquet")
    sites = pd.read_parquet(LT_DIR / "sites.parquet")
    cw = lt_plots.merge(sites[["site_id", "lat", "lon"]], on=["lat", "lon"], how="left")
    if cw["site_id"].isna().any():
        raise SystemExit(f"{cw['site_id'].isna().sum()} Living Trees plots sin site_id")
    return cw[["PlotObservationID", "site_id"]]


def fit_one(t: pd.Series, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Natural PhenoShape fit -> (curve, own_doy), both length NGS."""
    arr = v.reshape(-1, 1, 1)
    da = xr.DataArray(arr, dims=("time", "y", "x"),
                      coords={"time": t.values, "y": [0], "x": [0]})
    doy = da["time"].dt.dayofyear.values
    da = da.assign_coords(doy=("time", doy))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape = da.pheno.PhenoShape(interpolType="harmonic", rollWindow=ROLL, nGS=NGS,
                                    recon_params={"n_harmonics": N_HARMONICS},
                                    rollMode="shrink")
    return shape.values.reshape(-1), shape["doy"].values.astype(float)


def anchor_to_target(curve: np.ndarray, own_doy: np.ndarray) -> np.ndarray:
    """Circularly resample onto the fixed DOY-108, 7-day grid (see ANCHOR_DOY above)."""
    order = np.argsort(own_doy)
    return np.interp(TARGET_DOY, own_doy[order], curve[order], period=365)


def build_index_px(index: str, px: str, cw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = pd.read_parquet(LT_DIR / f"series_{index}_{px}.parquet")
    series = series.merge(cw, on="site_id", how="inner")
    val_col = index

    curve_rows, doy_rows = [], []
    n_fail = 0
    for pid, g in series.groupby("PlotObservationID"):
        g = g.sort_values("time")
        if g[val_col].notna().sum() < 5:
            n_fail += 1
            continue
        t = pd.to_datetime(g["time"])
        v = g[val_col].values.astype(float)
        try:
            curve, own_doy = fit_one(t, v)
        except Exception:
            n_fail += 1
            continue
        anchored = anchor_to_target(curve, own_doy)
        row = {"plot_id": str(pid), "index": index, "px": px}
        row.update({f"s{k:02d}": float(c) for k, c in enumerate(anchored)})
        curve_rows.append(row)
        if px == PX_LEVELS[0] and index == INDICES[0]:
            doy_row = {"plot_id": str(pid)}
            doy_row.update({f"s{k:02d}": float(d) for k, d in enumerate(TARGET_DOY)})
            doy_rows.append(doy_row)
    if n_fail:
        print(f"    {index}/{px}: {n_fail} parcelas sin ajuste (< 5 obs o PhenoShape fallo)")
    return pd.DataFrame(curve_rows), pd.DataFrame(doy_rows)


def main() -> None:
    cw = build_crosswalk()
    print(f"crosswalk: {len(cw)} parcelas Living Trees con site_id")

    lt_frames, doy_frames = [], []
    for index in INDICES:
        for px in PX_LEVELS:
            print(f"  ajustando {index}/{px} ...")
            c, d = build_index_px(index, px, cw)
            lt_frames.append(c)
            if len(d):
                doy_frames.append(d)
    lt_curves = pd.concat(lt_frames, ignore_index=True)

    pcl_curves = pd.read_parquet(DERIVED / "phenoshape_by_index.parquet").copy()
    pcl_curves["plot_id"] = "PCL_" + pcl_curves["plot_id"].astype(str)
    unified = pd.concat([pcl_curves, lt_curves], ignore_index=True)
    out_path = DERIVED / "phenoshape_by_index_unified.parquet"
    unified.to_parquet(out_path, index=False)
    print(f"\nParcelas-CL: {len(pcl_curves)} filas, {pcl_curves['plot_id'].nunique()} parcelas")
    print(f"Living Trees: {len(lt_curves)} filas, {lt_curves['plot_id'].nunique()} parcelas")
    print(f"-> {out_path} ({len(unified)} filas, {unified['plot_id'].nunique()} parcelas)")

    pcl_doy = pd.read_parquet(DERIVED / "phenoshape_doy_grid.parquet").copy()
    pcl_doy["plot_id"] = "PCL_" + pcl_doy["plot_id"].astype(str)
    lt_doy = pd.concat(doy_frames, ignore_index=True)
    doy_unified = pd.concat([pcl_doy, lt_doy], ignore_index=True)
    doy_out = DERIVED / "phenoshape_doy_grid_unified.parquet"
    doy_unified.to_parquet(doy_out, index=False)
    print(f"-> {doy_out} ({len(doy_unified)} parcelas)")


if __name__ == "__main__":
    main()
