#!/usr/bin/env python3
"""Build the modellable subset of Parcelas-CL and the cross-validation schemes.

Filter cascade (expected counts come from direct inspection of the released data):

    0. Parcelas-CL, all plots                         1485
    1. + year known                                   1443
    2. + latitude within range (30-38S)               1254
    3. + year >= 1999 (Landsat era)                   1253
    4. + unique coordinate                            1082
    5. + richness >= 2 (flagged, not filtered)         969

Step 4 removes md001 (Becerra 1999): 8 localities x 20 plots sharing an identical
coordinate. Those are site-level coordinates, not plot-level; for a pixel-based model they
would be 20 identical predictor vectors carrying 20 different labels.

Usage:
    python scripts/01_build_subset.py --zip data/20602096.zip --out-dir data/derived
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import cv_groups, io_parcelas  # noqa: E402


def build(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plots = io_parcelas.load_plots(args.zip)
    cascade = [("0. Parcelas-CL, all plots", len(plots))]

    df = plots[plots["Year"].notna()].copy()
    cascade.append(("1. + year known", len(df)))

    df = df[(df["lat"].abs() >= args.lat_min) & (df["lat"].abs() < args.lat_max)]
    cascade.append((f"2. + latitude {args.lat_min}-{args.lat_max}S", len(df)))

    df = df[df["Year"] >= args.min_year]
    cascade.append((f"3. + year >= {args.min_year}", len(df)))

    df = df[~df["coord_shared"]]
    cascade.append(("4. + unique coordinate", len(df)))

    cascade.append(("5. (flag) richness >= 2", int((df["richness"] >= 2).sum())))

    df = df.copy()
    df["Year"] = df["Year"].astype(int)
    # Causal window y-2..y: uses no information posterior to the census, and stays defined
    # for the 2026 plots (a centred y-1..y+1 window does not exist for them).
    df["win_start"] = df["Year"] - (args.window - 1)
    df["win_end"] = df["Year"]
    df["win_years"] = args.window
    # Grouping cell for the datacube loads
    df["cell"] = (
        (df["X"] / (args.cell_km * 1000)).round().astype(int).astype(str)
        + "_"
        + (df["Y"] / (args.cell_km * 1000)).round().astype(int).astype(str)
    )

    keep = [
        "PlotObservationID", "Owner", "metadata_id", "Location", "Year",
        "PlotSize_m2", "X", "Y", "lon", "lat", "richness", "n_records",
        "stratum", "Abundance_parameter", "pixel_id", "cell",
        "win_start", "win_end", "win_years",
    ]
    subset = df[keep].sort_values("PlotObservationID").reset_index(drop=True)
    subset.to_parquet(out / "plots_subset.parquet", index=False)

    # --- cross-validation ---------------------------------------------------
    # Three schemes with different jobs:
    #   lodo_dataset      -- unstratified on purpose. Most projects are single-year, so this
    #                        is leave-one-year-and-protocol-out too. The distribution shift
    #                        between folds IS the quantity being measured, so do not balance
    #                        it away; report per-fold composition alongside the score.
    #   kfold5_dataset    -- stratified on richness, for a comparable headline estimate.
    #   kfold5_location   -- finer spatial split, also stratified.
    schemes = {
        "lodo_dataset": cv_groups.leave_one_group_out(subset, "metadata_id", args.min_test),
        "kfold5_dataset": cv_groups.grouped_kfold(subset, "metadata_id", args.k,
                                                  stratify_on=args.stratify),
        "kfold5_location": cv_groups.grouped_kfold(subset, "Location", args.k,
                                                   stratify_on=args.stratify),
    }
    cv = pd.concat(
        [d.assign(scheme=name) for name, d in schemes.items()], ignore_index=True
    )
    cv.to_parquet(out / "cv_folds.parquet", index=False)

    report = cv_groups.fold_report(subset, cv, response=args.stratify or "richness")
    report.to_csv(out / "cv_fold_report.csv", index=False)

    cv_groups.summarise(subset, "metadata_id").to_csv(out / "groups_dataset.csv", index=False)
    cv_groups.summarise(subset, "Location").to_csv(out / "groups_location.csv", index=False)

    # --- reports ------------------------------------------------------------
    pd.DataFrame(cascade, columns=["step", "n_plots"]).to_csv(
        out / "filter_cascade.csv", index=False
    )

    summary = {
        "cascade": {k: v for k, v in cascade},
        "n_subset": len(subset),
        "year_range": [int(subset["Year"].min()), int(subset["Year"].max())],
        "window": f"y-{args.window - 1}..y (causal)",
        "n_cells": int(subset["cell"].nunique()),
        "n_cell_year": int(subset.groupby(["cell", "Year"]).ngroups),
        "n_datasets": int(subset["metadata_id"].nunique()),
        "n_locations": int(subset["Location"].nunique()),
        "strata": subset["stratum"].value_counts().to_dict(),
        "plots_per_year": subset["Year"].value_counts().sort_index().to_dict(),
        "richness": {
            "median": float(subset["richness"].median()),
            "n_richness_1": int((subset["richness"] == 1).sum()),
            "n_richness_ge2": int((subset["richness"] >= 2).sum()),
        },
        "plotsize_na": int(subset["PlotSize_m2"].isna().sum()),
        "cv": {
            name: {
                "scheme": d.attrs.get("scheme"),
                "n_folds": d.attrs.get("n_folds"),
                "fold_sizes": d.attrs.get("fold_sizes"),
                "excluded_from_test": d.attrs.get("excluded_from_test"),
            }
            for name, d in schemes.items()
        },
        "cv_fold_report": str(out / "cv_fold_report.csv"),
        "reconcile_with_paper": io_parcelas.reconcile_with_paper(plots),
        "quality_flags": io_parcelas.quality_flags(args.zip),
    }
    (out / "subset_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("FILTER CASCADE")
    for k, v in cascade:
        print(f"  {k:<40s} {v:>6d}")
    print(f"\nsubset: {len(subset)} plots, {summary['year_range'][0]}-{summary['year_range'][1]}")
    print(f"expected datacube loads (cell {args.cell_km}km x year): {summary['n_cell_year']}")
    print(f"strata: {summary['strata']}")
    print("\nCross-validation:")
    for name, d in schemes.items():
        print(f"  {name:18s} {d.attrs.get('scheme')}  folds={d.attrs.get('n_folds')}"
              f"  sizes={d.attrs.get('fold_sizes')}")
        if d.attrs.get("excluded_from_test"):
            print(f"     never test (groups < {args.min_test} plots): "
                  f"{d.attrs['excluded_from_test']}")
    print(f"\nwritten to {out}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="data/20602096.zip")
    p.add_argument("--out-dir", default="data/derived")
    p.add_argument("--lat-min", type=float, default=30.0)
    p.add_argument("--lat-max", type=float, default=38.0)
    p.add_argument("--min-year", type=int, default=1999)
    p.add_argument("--window", type=int, default=3,
                   help="length of the causal window in years (y-w+1..y)")
    p.add_argument("--cell-km", type=float, default=5.0,
                   help="grouping cell size for datacube loads")
    p.add_argument("--k", type=int, default=5, help="folds for the grouped CV")
    p.add_argument("--min-test", type=int, default=20,
                   help="minimum group size to serve as a test fold in LODO")
    p.add_argument("--stratify", default="richness",
                   help="response variable balanced across grouped-kfold folds "
                        "(empty string disables stratification)")
    build(p.parse_args())


if __name__ == "__main__":
    main()
