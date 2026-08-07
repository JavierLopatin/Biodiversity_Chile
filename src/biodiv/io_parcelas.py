"""Reading and normalisation of Parcelas-CL from the Zenodo archive (10.5281/zenodo.20602096).

The CSV is read directly from the zip: nothing is extracted to disk, so the downloaded
archive stays the single source of truth.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

CSV_NAME = "Parcelas_CL.csv"
CRS_SOURCE = "EPSG:32719"  # WGS84 / UTM 19S, as declared by the data authors
CRS_WGS84 = "EPSG:4326"

# The binary `Abundance` flag in the CSV contradicts the aggregation reported in the data
# paper: it marks Basal_area plots as having abundance (=1), while the paper counts them as
# presence/absence (224 NA + 147 Basal_area = 371 = the reported 25%). We therefore
# stratify on Abundance_parameter, which is consistent within each plot, never on the flag.
STRATUM = {
    "Cover": "cover",
    "Cover_st": "cover",
    "Abundance": "counts",
    "Basal_area": "basal",
    "NA": "presence",
}


def load_long(zip_path: str | Path) -> pd.DataFrame:
    """Return the CSV in long format (one row per plot x taxon)."""
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(CSV_NAME).decode("utf-8", "replace")
    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"].replace("NA", ""), errors="coerce")
    df["PlotSize_m2"] = pd.to_numeric(df["PlotSize_m2"].replace("NA", ""), errors="coerce")
    return df


def load_plots(zip_path: str | Path) -> pd.DataFrame:
    """Aggregate to plot level and add lat/lon, richness and abundance stratum."""
    long = load_long(zip_path)

    first = long.groupby("PlotObservationID", as_index=False).first()
    richness = (
        long.groupby("PlotObservationID")["Accepted_species"]
        .nunique()
        .rename("richness")
        .reset_index()
    )
    n_records = (
        long.groupby("PlotObservationID").size().rename("n_records").reset_index()
    )
    plots = first.merge(richness, on="PlotObservationID").merge(n_records, on="PlotObservationID")

    tf = Transformer.from_crs(CRS_SOURCE, CRS_WGS84, always_xy=True)
    lon, lat = tf.transform(plots["X"].to_numpy(), plots["Y"].to_numpy())
    plots["lon"], plots["lat"] = lon, lat

    plots["stratum"] = plots["Abundance_parameter"].map(STRATUM).fillna("unknown")

    # 30 m Landsat pixel identifier: groups plots that fall in the same pixel
    plots["pixel_id"] = (
        (plots["X"] / 30).round().astype("Int64").astype(str)
        + "_"
        + (plots["Y"] / 30).round().astype("Int64").astype(str)
    )
    # Exactly repeated coordinate (not the same thing as sharing a pixel)
    coord = plots["X"].astype(str) + "_" + plots["Y"].astype(str)
    plots["coord_shared"] = coord.map(coord.value_counts()) > 1

    return plots


def reconcile_with_paper(plots: pd.DataFrame) -> dict:
    """Regression checks against Cerda-Paredes et al. (2026).

    Detects the case where the published dataset changes underneath the pipeline.
    """
    strat = plots["Abundance_parameter"].value_counts().to_dict()
    cover = strat.get("Cover", 0) + strat.get("Cover_st", 0)
    counts = strat.get("Abundance", 0)
    pa = strat.get("NA", 0) + strat.get("Basal_area", 0)
    return {
        "n_plots": int(len(plots)),
        "n_plots_paper": 1485,
        "richness_median": float(plots["richness"].median()),
        "richness_median_paper": 5.0,
        "cover_plots": int(cover),
        "cover_plots_paper": 625,
        "counts_plots": int(counts),
        "counts_plots_paper": 489,
        "presence_absence_plots": int(pa),
        "presence_absence_plots_paper": 371,
    }


def quality_flags(zip_path: str | Path) -> dict:
    """Data-quality issues found in the released CSV, reported explicitly."""
    long = load_long(zip_path)
    flag_incons = (
        long.groupby("PlotObservationID")["Abundance"].nunique().gt(1).sum()
    )
    param_incons = (
        long.groupby("PlotObservationID")["Abundance_parameter"].nunique().gt(1).sum()
    )
    cover = long.loc[long["Abundance_parameter"] == "Cover", "Value"]
    return {
        "plots_inconsistent_abundance_flag": int(flag_incons),
        "plots_inconsistent_abundance_parameter": int(param_incons),
        "cover_max": float(cover.max()) if len(cover) else np.nan,
        "cover_over_100": int((cover > 100).sum()),
        "n_accepted_taxa": int(long["Accepted_species"].nunique()),
        "taxa_by_rank": long.groupby("Accepted_name_rank")["Accepted_species"]
        .nunique()
        .to_dict(),
    }
