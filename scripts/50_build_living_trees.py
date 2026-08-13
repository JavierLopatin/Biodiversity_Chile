#!/usr/bin/env python3
"""Build the plot and occurrence tables for Living Trees Chile, from the Excel source.

Living Trees Chile is a forest-inventory dataset (tree-level DBH/height records, fixed
500 m² plots, 2011-2020), a second data source alongside Parcelas-CL. Three things this
script gets right that a naive read would not:

1. **`pd.read_excel` needs `keep_default_na=False`.** Without it, the species code
   `"NA"` (Nothofagus alpina) is read as a missing value, silently corrupting ~2.5% of
   the records. `io_parcelas.py` already applies the equivalent for the Parcelas-CL CSV.

2. **The real plot identity is the coordinate, not `um` (or `region+um`).** `um` alone
   repeats across regions (78 cases, different coordinates each time — hence
   `region+um` looking unique at first). But 2 coordinate pairs (7-decimal, sub-metre
   precision) each carry TWO distinct `region+um` labels — the same physical site split
   into two site codes. Plots are grouped by exact coordinate; the 2 merged cases are
   logged, not silently absorbed.

3. **`SE` ("Sin especie") + `GD`** (1 row, no match in the `codes` sheet) are the only
   genuinely unidentified records (0.25% of rows) — excluded from the occurrence table.
   `ND` (Nothofagus dombeyi) and `NA` (Nothofagus alpina) are real, common species and
   are kept.

Abundance currency: basal area per tree (`BA = pi/4 * (D/100)^2` m², D in cm), scaled by
the expansion factor `exp` (trees/ha represented by the sampled tree) and summed per
(plot, species) — the standard forestry proxy for dominance, comparable in spirit to
`Abundance_parameter="Basal_area"` already in Parcelas-CL's vocabulary
(`src/biodiv/io_parcelas.py:25-30`), just unused there until now.

Usage:
    python scripts/50_build_living_trees.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

#: Species codes with no usable taxon identity: "Sin especie" (no species recorded) and
#: one code ("GD") absent from the `codes` sheet, unresolved.
UNRESOLVED_CODES = {"SE", "GD"}


def load_species_map(xlsx_path: Path) -> dict[str, str]:
    codes = pd.read_excel(xlsx_path, sheet_name="codes", keep_default_na=False, na_values=[])
    codes = codes[["Species_code", "Unnamed: 1"]].rename(columns={"Unnamed: 1": "species"})
    codes = codes[codes["Species_code"] != ""]
    return dict(zip(codes["Species_code"], codes["species"]))


def build(args: argparse.Namespace) -> None:
    xlsx = Path(args.xlsx)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tree = pd.read_excel(xlsx, sheet_name="tree-level", keep_default_na=False, na_values=[])
    n_raw = len(tree)
    # `keep_default_na=False` reads blanks as "" rather than NaN -- coerce the numeric
    # columns explicitly, same pattern as `io_parcelas.load_long`.
    for col in ["D", "DAT", "H", "exp", "Elevation", "Slope", "Date", "Latitude", "Longitude"]:
        tree[col] = pd.to_numeric(tree[col], errors="coerce")

    species_map = load_species_map(xlsx)
    codes_present = set(tree["Species_code"].unique())
    unmatched = sorted(codes_present - set(species_map) - UNRESOLVED_CODES)
    if unmatched:
        raise SystemExit(
            f"código(s) de especie sin resolver, no incluidos en UNRESOLVED_CODES: {unmatched}"
        )

    excluded = tree["Species_code"].isin(UNRESOLVED_CODES)
    print(f"excluidas por especie no identificada (SE/GD): {excluded.sum()} de {n_raw} filas "
          f"({100 * excluded.mean():.2f}%)")
    coords_before = tree[["Latitude", "Longitude"]].drop_duplicates()
    tree = tree[~excluded].copy()
    tree["species"] = tree["Species_code"].map(species_map)
    coords_after = tree[["Latitude", "Longitude"]].drop_duplicates()
    n_sites_lost = len(coords_before) - len(coords_after)
    if n_sites_lost:
        print(f"sitios perdidos por completo (100% de sus arboles SE/GD): {n_sites_lost}")

    # --- plot_key por coordenada unica, no region+um ---------------------------------------
    coords = tree[["Latitude", "Longitude"]].drop_duplicates().reset_index(drop=True)
    coords["PlotObservationID"] = "LT_" + coords.index.astype(str)
    tree = tree.merge(coords, on=["Latitude", "Longitude"], how="left")

    tree["site_um"] = (
        tree["Chilean administrative region"].astype(str) + "_" + tree["um"].astype(str)
    )
    n_um_per_key = tree.groupby("PlotObservationID")["site_um"].nunique()
    merged_keys = n_um_per_key[n_um_per_key > 1].index
    print(f"sitios fusionados por coordenada compartida entre 2 um distintos: {len(merged_keys)}")
    for k in merged_keys:
        ums = sorted(tree.loc[tree["PlotObservationID"] == k, "site_um"].unique())
        print(f"  {k}: {ums}")

    n_region_conflict = (
        tree.groupby("PlotObservationID")["Chilean administrative region"].nunique() > 1
    ).sum()
    if n_region_conflict:
        raise SystemExit(
            f"{n_region_conflict} PlotObservationID mezclan 2 regiones administrativas "
            "distintas bajo la misma coordenada -- revisar antes de seguir"
        )

    # --- metadata por parcela (antes del filtro de D, no depende de el) --------------------
    first = tree.groupby("PlotObservationID", as_index=False).first()
    um_merged = (
        tree.groupby("PlotObservationID")["site_um"]
        .apply(lambda s: "|".join(sorted(s.unique())))
        .rename("um_merged")
    )
    n_records = tree.groupby("PlotObservationID").size().rename("n_records")

    plots = first[[
        "PlotObservationID", "Chilean administrative region", "Latitude", "Longitude",
        "Elevation", "Slope", "Chilean forest type", "Date",
    ]].rename(columns={
        "Chilean administrative region": "region",
        "Latitude": "lat",
        "Longitude": "lon",
        "Elevation": "elevation",
        "Slope": "slope",
        "Chilean forest type": "forest_type",
        "Date": "Year",
    })
    plots = plots.merge(um_merged, on="PlotObservationID").merge(n_records, on="PlotObservationID")
    plots["PlotSize_m2"] = 500.0
    plots["source"] = "living_trees"

    # --- area basal por arbol, escalada por factor de expansion -----------------------------
    n_no_d = tree["D"].isna().sum()
    print(f"arboles sin D (excluidos del area basal): {n_no_d} de {len(tree)} "
          f"({100 * n_no_d / len(tree):.2f}%)")
    tree_ba = tree[tree["D"].notna()].copy()
    # BA = pi/4 * (D_cm/100)^2 m^2 por arbol, x exp (arboles/ha que representa) -> m^2/ha
    tree_ba["basal_area"] = (math.pi / 4) * (tree_ba["D"] / 100.0) ** 2 * tree_ba["exp"]

    long = (
        tree_ba.groupby(["PlotObservationID", "species"], as_index=False)["basal_area"]
        .sum()
        .rename(columns={"basal_area": "Value"})
    )
    long["Abundance_parameter"] = "Basal_area"

    richness = long.groupby("PlotObservationID")["species"].nunique().rename("richness")
    plots = plots.merge(richness, on="PlotObservationID", how="left")
    plots["richness"] = plots["richness"].fillna(0).astype(int)

    long.to_parquet(out / "living_trees_long.parquet", index=False)
    plots.to_parquet(out / "living_trees_plots.parquet", index=False)

    print(f"\n-> {out / 'living_trees_long.parquet'} "
          f"({len(long)} filas, {long['PlotObservationID'].nunique()} parcelas, "
          f"{long['species'].nunique()} especies)")
    print(f"-> {out / 'living_trees_plots.parquet'} ({len(plots)} parcelas)")
    print(f"riqueza mediana: {plots['richness'].median()}, "
          f"parcelas con riqueza 0 (todos sus arboles sin D): {(plots['richness'] == 0).sum()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", default="data/Living_Trees_Chile.xlsx")
    p.add_argument("--out-dir", default="data/derived")
    build(p.parse_args())


if __name__ == "__main__":
    main()
