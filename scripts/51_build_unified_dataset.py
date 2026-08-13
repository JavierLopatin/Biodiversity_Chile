#!/usr/bin/env python3
"""Build the unified Parcelas-CL + Living Trees Chile occurrence and plot tables.

Parcelas-CL side reuses the existing 1,082-plot modelling subset
(`data/derived/plots_subset.parquet`) unchanged -- no relaxing of its lat 30-38S filter,
so nothing already reported this session shifts underneath it. Only Living Trees Chile
(`scripts/50_build_living_trees.py`) brings the geography south of that band.

IDs are prefixed (`PCL_<id>` / `LT_<id>`, already applied by script 50) so the two ID
schemes cannot collide even though nothing in the original pipeline enforces uniqueness.

A single equal-area CRS is used for both datasets' metric coordinates (`X_m`, `Y_m`):
Parcelas-CL's own X/Y are raw UTM 19S, low-distortion only near its central meridian --
fine for Chile central, not for a footprint reaching Magallanes (~55S). The unified
`X_m`/`Y_m` are a Lambert Azimuthal Equal-Area centred on Chile, built from lon/lat
(WGS84) for both datasets, kept separate from Parcelas-CL's existing UTM columns.

Usage:
    python scripts/51_build_unified_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import io_parcelas  # noqa: E402

LAEA_CHILE = "+proj=laea +lat_0=-35 +lon_0=-71 +datum=WGS84 +units=m +no_defs"


def build_parcelas(zip_path: str, subset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = pd.read_parquet(subset_path)
    keep_ids = set(subset["PlotObservationID"])

    long = io_parcelas.load_long(zip_path)
    long = long[long["PlotObservationID"].isin(keep_ids)].copy()
    # Same aggregation as scripts/07_compute_taxonomic_beta_responses.R: a handful of
    # (plot, species) pairs have 2 records (different strata/growth forms of one taxon).
    agg = (
        long.groupby(["PlotObservationID", "Accepted_species"], as_index=False)
        .agg(Value=("Value", "sum"), Abundance_parameter=("Abundance_parameter", "first"))
        .rename(columns={"Accepted_species": "species"})
    )
    agg["PlotObservationID"] = "PCL_" + agg["PlotObservationID"].astype(str)

    plots = subset.copy()
    plots["PlotObservationID"] = "PCL_" + plots["PlotObservationID"].astype(str)
    plots["source"] = "parcelas_cl"
    return agg, plots


def build(args: argparse.Namespace) -> None:
    derived = Path(args.derived)

    pcl_long, pcl_plots = build_parcelas(args.zip, derived / "plots_subset.parquet")
    lt_long = pd.read_parquet(derived / "living_trees_long.parquet")
    lt_plots = pd.read_parquet(derived / "living_trees_plots.parquet")

    occ = pd.concat([pcl_long, lt_long], ignore_index=True)
    plots = pd.concat([pcl_plots, lt_plots], ignore_index=True)

    dup = plots["PlotObservationID"].duplicated()
    assert not dup.any(), f"{dup.sum()} PlotObservationID duplicados tras unificar"
    assert plots["lon"].notna().all() and plots["lat"].notna().all(), \
        "lon/lat faltante en alguna parcela unificada"

    tf = Transformer.from_crs("EPSG:4326", LAEA_CHILE, always_xy=True)
    x_m, y_m = tf.transform(plots["lon"].to_numpy(), plots["lat"].to_numpy())
    plots["X_m"], plots["Y_m"] = x_m, y_m

    pcl_species = set(pcl_long["species"].unique())
    lt_species = set(lt_long["species"].unique())
    overlap = pcl_species & lt_species
    print(f"especies Parcelas-CL: {len(pcl_species)}, Living Trees: {len(lt_species)}, "
          f"solapadas (mismo binomio exacto): {len(overlap)}")
    if overlap:
        print(f"  {sorted(overlap)}")

    occ.to_parquet(derived / "occurrences_unified.parquet", index=False)
    plots.to_parquet(derived / "plots_unified.parquet", index=False)

    print(f"\n-> {derived / 'occurrences_unified.parquet'} "
          f"({len(occ)} filas, {occ['PlotObservationID'].nunique()} parcelas)")
    print(f"-> {derived / 'plots_unified.parquet'} "
          f"({len(plots)} parcelas, por fuente: {plots['source'].value_counts().to_dict()})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="data/20602096.zip")
    p.add_argument("--derived", default="data/derived")
    build(p.parse_args())


if __name__ == "__main__":
    main()
