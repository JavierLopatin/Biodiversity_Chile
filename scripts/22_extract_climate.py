#!/usr/bin/env python3
"""Bioclimatic normals per plot from CR2MET, via Data Cube Chile.

Why this block exists. The GDM comparison showed that geographic distance alone reproduces
observed compositional dissimilarity at Spearman +0.435, and that the whole phenological and
spectral predictor set adds only +0.057 on top of it. Distance is a stand-in for something,
and over 7.6 degrees of latitude in Mediterranean Chile that something is overwhelmingly the
aridity gradient. Elevation and a vegetation curve are weak proxies for climate; climate is
not in the stored cubes, and it is the one thing the datacube is still needed for.

**The normal period is 1971-2000, and that is a causal choice, not a convention.** The
earliest census in the subset is 2003, so a normal ending in 2000 precedes every response
measurement. A 1991-2020 normal — the current WMO standard — would carry information from
after most censuses were taken, which is exactly what docs/05_data_acquisition.md rules out
for the Landsat windows; the same rule has to apply here or it is not a rule.

**Long-term normals rather than the three years before each census.** Community composition
is a slow variable and responds to the climate a site has had, not to one particular triennium.
It also happens to be the only option that covers the whole subset: CR2MET ends in 2021 and
38% of these plots were censused in 2022 or later, so a causal three-year window does not
exist for them. Using a normal makes the block complete rather than 38% missing.

CR2MET (Boisier et al.) is daily ``pr`` (mm/day), ``tmin`` and ``tmax`` (degrees C) on a
0.05 degree grid over continental Chile, 1960-2021. The 1,082 plots fall in 253 distinct
cells, so the block describes the plot's climatic setting, not its 150 m window — different
support from every other predictor here, and that is stated rather than hidden.

Usage:
    python scripts/22_extract_climate.py
    python scripts/22_extract_climate.py --start 1971 --end 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ID_COL = "PlotObservationID"
PAD = 0.1                 # degrees of margin around the plot bounding box


def monthly_fields(dc, bbox: tuple[float, float, float, float], year: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Monthly precipitation total and mean tmin/tmax for one year, shaped (12, ny, nx)."""
    lon0, lon1, lat0, lat1 = bbox
    ds = dc.load(product="cr2met", x=(lon0, lon1), y=(lat0, lat1),
                 time=(f"{year}-01-01", f"{year}-12-31"),
                 measurements=["pr", "tmin", "tmax"])
    if ds.sizes.get("time", 0) == 0:
        raise SystemExit(f"CR2MET devolvio 0 fechas para {year}")
    g = ds.groupby("time.month")
    pr = g.sum("time")["pr"].values           # mm per month
    tmin = g.mean("time")["tmin"].values
    tmax = g.mean("time")["tmax"].values
    return pr, tmin, tmax, ds


def quarter_totals(x: np.ndarray) -> np.ndarray:
    """Rolling 3-month sums over a circular calendar, shaped (12, ...)."""
    return np.stack([x[[m % 12, (m + 1) % 12, (m + 2) % 12]].sum(axis=0) for m in range(12)])


def quarter_means(x: np.ndarray) -> np.ndarray:
    return np.stack([x[[m % 12, (m + 1) % 12, (m + 2) % 12]].mean(axis=0) for m in range(12)])


def derive(pr_m: np.ndarray, tmin_m: np.ndarray, tmax_m: np.ndarray,
           pr_annual: np.ndarray) -> dict[str, np.ndarray]:
    """Bioclimatic variables from the monthly normals, shaped (ny, nx) each.

    A deliberately small set. Nineteen WorldClim-style variables would be mostly
    redundant at this sample size; these are the axes that separate Mediterranean Chile —
    how much water, how hot, how strongly seasonal, how concentrated in winter, and how
    variable between years.
    """
    tmean_m = 0.5 * (tmin_m + tmax_m)
    mat = tmean_m.mean(axis=0)
    map_ = pr_m.sum(axis=0)

    pq, tq = quarter_totals(pr_m), quarter_means(tmean_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_cv = np.where(map_ > 0, pr_m.std(axis=0, ddof=1) / (map_ / 12.0), np.nan)
        # De Martonne aridity index: the classic P/(T+10) water-availability summary, and
        # the single variable that orders this study area from Coquimbo to Biobio.
        demartonne = np.where(mat > -10, map_ / (mat + 10.0), np.nan)
        summer_frac = np.where(map_ > 0, pr_m[[11, 0, 1]].sum(axis=0) / map_, np.nan)
        inter_cv = np.where(pr_annual.mean(axis=0) > 0,
                            pr_annual.std(axis=0, ddof=1) / pr_annual.mean(axis=0), np.nan)

    return {
        "clim_map": map_,
        "clim_mat": mat,
        "clim_tmax_warmest": tmax_m.max(axis=0),
        "clim_tmin_coldest": tmin_m.min(axis=0),
        "clim_dtr": (tmax_m - tmin_m).mean(axis=0),
        "clim_p_seasonality_cv": p_cv,
        "clim_t_seasonality_sd": tmean_m.std(axis=0, ddof=1),
        "clim_p_driest_quarter": pq.min(axis=0),
        "clim_p_wettest_quarter": pq.max(axis=0),
        "clim_t_warmest_quarter": tq.max(axis=0),
        "clim_t_coldest_quarter": tq.min(axis=0),
        "clim_p_summer_frac": summer_frac,
        "clim_demartonne": demartonne,
        "clim_p_interannual_cv": inter_cv,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plots", default="data/derived/plots_subset.parquet")
    p.add_argument("--out", default="data/derived/climate.parquet")
    p.add_argument("--start", type=int, default=1971)
    p.add_argument("--end", type=int, default=2000)
    args = p.parse_args()

    import datacube

    plots = pd.read_parquet(args.plots)
    bbox = (plots.lon.min() - PAD, plots.lon.max() + PAD,
            plots.lat.min() - PAD, plots.lat.max() + PAD)
    print(f"bbox lon {bbox[0]:.2f}..{bbox[1]:.2f}  lat {bbox[2]:.2f}..{bbox[3]:.2f}")
    print(f"normal {args.start}-{args.end} "
          f"(termina antes del primer censo, {int(plots.Year.min())})")

    dc = datacube.Datacube()
    years = list(range(args.start, args.end + 1))
    pr_stack, tmin_stack, tmax_stack, ref = [], [], [], None
    t0 = time.time()
    for i, y in enumerate(years, 1):
        pr, tmin, tmax, ds = monthly_fields(dc, bbox, y)
        pr_stack.append(pr)
        tmin_stack.append(tmin)
        tmax_stack.append(tmax)
        ref = ref if ref is not None else ds
        if i % 5 == 0 or i == len(years):
            el = time.time() - t0
            print(f"  {i}/{len(years)} anos  {el:6.1f}s  "
                  f"eta {el / i * (len(years) - i):6.1f}s", flush=True)

    PR = np.stack(pr_stack)                       # (n_years, 12, ny, nx)
    pr_m = PR.mean(axis=0)
    tmin_m = np.stack(tmin_stack).mean(axis=0)
    tmax_m = np.stack(tmax_stack).mean(axis=0)
    fields = derive(pr_m, tmin_m, tmax_m, PR.sum(axis=1))

    lat = ref.latitude.values
    lon = ref.longitude.values
    iy = np.abs(plots.lat.to_numpy()[:, None] - lat[None, :]).argmin(axis=1)
    ix = np.abs(plots.lon.to_numpy()[:, None] - lon[None, :]).argmin(axis=1)
    print(f"\ngrilla {len(lat)} x {len(lon)}; celdas CR2MET distintas ocupadas: "
          f"{len(set(zip(iy.tolist(), ix.tolist())))}")

    out = pd.DataFrame({ID_COL: plots[ID_COL].to_numpy()})
    for name, arr in fields.items():
        out[name] = arr[iy, ix]

    n_nan = int(out.drop(columns=[ID_COL]).isna().to_numpy().sum())
    print(f"NaN en el bloque: {n_nan}")
    if n_nan:
        bad = out.columns[out.isna().any()].tolist()
        print(f"  columnas afectadas: {bad}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"-> {args.out}  ({len(out)} filas x {out.shape[1]} columnas)\n")
    print(out.drop(columns=[ID_COL]).describe().T
          .to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
