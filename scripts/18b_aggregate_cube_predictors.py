#!/usr/bin/env python3
"""Aggregate the per-pixel cube predictors to plot level, with the rule written down.

The existing ``*_mean5x5`` columns were produced by ``np.nanmean`` over the 25 pixels
(``scripts/04_recompute_lsp.py:234``). That hides two things at once: 5.1% of pixels fail
to yield an LSP metric at all, and nothing records how many pixels went into any given
average — a value could be a mean of 25 or a mean of 1 and it would look identical.

This script makes the choice explicit and keeps every alternative side by side, so "which
aggregation?" becomes a screened factor (row X10 of docs/09_predictors.md) rather than a
default nobody chose:

  ``_median``   robust to a single failing pixel; the recommended default
  ``_mean``     what the existing columns use, kept for the comparison
  ``_trimmed``  20% trimmed mean, between the two
  ``_sd``       dispersion *between* the 25 pixels. Two roles at once: a spectral
                heterogeneity predictor (SVH, axis A of docs/01_state_of_the_art.md) and a
                diagnostic of how uniform the window is.
  ``_cv``       the same dispersion made scale-free, so a dry site and a wet site are
                comparable
  ``_center``   the centre pixel alone, for the pixel-footprint ablation
  ``n_px_valid`` per variable: the number that was missing

Usage:
    python scripts/18b_aggregate_cube_predictors.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

ID_COL = "PlotObservationID"

#: The centre of a 5x5 window. Plots are anchored on it in scripts/02_extract_phenology.py.
CENTRE = (2, 2)

TRIM = 0.2


def aggregate(px: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in px.columns if c not in ("plot_id", "y", "x")]
    g = px.groupby("plot_id", sort=True)

    out = pd.DataFrame(index=g.size().index)
    out["n_px"] = g.size()

    med = g[value_cols].median()
    mean = g[value_cols].mean()
    sd = g[value_cols].std(ddof=1)
    cnt = g[value_cols].count()
    trimmed = g[value_cols].agg(
        lambda s: trim_mean(s.dropna(), TRIM) if s.notna().sum() >= 3 else np.nan)

    centre = (px[(px["y"] == CENTRE[0]) & (px["x"] == CENTRE[1])]
              .set_index("plot_id")[value_cols].reindex(out.index))

    with np.errstate(invalid="ignore", divide="ignore"):
        cv = sd / mean.abs().where(mean.abs() > 1e-9)

    parts = [
        med.add_suffix("_median"), mean.add_suffix("_mean"),
        trimmed.add_suffix("_trimmed"), sd.add_suffix("_sd"),
        cv.add_suffix("_cv"), centre.add_suffix("_center"),
        cnt.add_suffix("_npx"),
    ]
    return pd.concat([out] + parts, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pixels", default="data/derived/cube_predictors_pixels.parquet")
    p.add_argument("--plots", default="data/derived/plots_subset.parquet")
    p.add_argument("--out", default="data/derived/cube_predictors.parquet")
    args = p.parse_args()

    px = pd.read_parquet(args.pixels)
    print(f"{len(px):,} filas de pixel, {px['plot_id'].nunique()} parcelas")

    agg = aggregate(px)
    plots = pd.read_parquet(args.plots)
    ids = plots[ID_COL].astype(str)
    agg.index = agg.index.astype(str)

    missing = set(ids) - set(agg.index)
    if missing:
        raise SystemExit(f"{len(missing)} parcelas sin predictores de cubo, "
                         f"ej. {sorted(missing)[:5]}")

    agg = agg.reindex(ids)
    agg.index.name = "plot_id"
    agg = agg.reset_index()
    agg[ID_COL] = plots[ID_COL].to_numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False)
    print(f"-> {args.out}  ({len(agg)} filas x {agg.shape[1]} columnas)")

    npx = agg[[c for c in agg.columns if c.endswith("_npx")]]
    print(f"   pixeles validos por variable: min {int(npx.min().min())}, "
          f"mediana {int(npx.median().median())}, max {int(npx.max().max())}")
    frac_full = float((npx == 25).all(axis=1).mean())
    print(f"   parcelas con los 25 pixeles validos en todas las variables: "
          f"{100 * frac_full:.1f}%")
    numeric = agg.select_dtypes("number")
    print(f"   NaN global en el bloque agregado: "
          f"{100 * numeric.isna().to_numpy().mean():.3f}%")


if __name__ == "__main__":
    main()
