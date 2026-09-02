"""Reading and normalisation of Living_Trees_Chile.xlsx into a plot-level site table.

The workbook is **tree-level**: 59,408 rows, one per measured stem, with the plot's
coordinate repeated on every tree of the plot. The extraction unit is therefore not a row
but a site, and *which* key defines a site is the one decision that matters here.

**The site key is the coordinate, not `um`.** `um` is the sampling unit assigned in the
field, and 78 of the 1,943 `um` carry more than one coordinate — grouping by `um` would
average plots that sit in different Landsat pixels. Going the other way, 2 coordinates are
shared by more than one `um`, so `um` is neither finer nor coarser than the coordinate: it
is a different partition. `um` is kept as a site attribute (`um_ids`) for traceability and
never used to group.

Measured on the delivered file:

    tree rows                                   59,408
    sites, (Latitude, Longitude, Date)           2,021
    um                                           1,943
    um with more than one coordinate                78
    coordinates with more than one um                 2
    sites sharing a 30 m Landsat pixel                0
    rows without Date                                46  (one site, um 17048, Aysen)
    Date                                     2011..2020
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

SHEET = "tree-level"
CRS_WGS84 = "EPSG:4326"     # Latitude/Longitude come as decimal degrees
CRS_TARGET = "EPSG:32719"   # WGS84 / UTM 19S, the CRS of the whole project

#: Causal window, `y-2..y`, as in `scripts/01_build_subset.py`. Uses no information
#: posterior to the census and stays defined for the most recent plots.
WINDOW_YEARS = 3

#: Landsat pixel side, for the `pixel_id` that groups sites falling in the same pixel.
PIXEL_M = 30

# Excel headers -> the snake_case names used downstream. `Date` is the year the plot was
# established (per the workbook's own `metadata` sheet), not a full date.
SITE_ATTRS = {
    "Chilean administrative region": "region",
    "Relief strip": "relief_strip",
    "Elevation": "elevation",
    "Slope": "slope",
    "Chilean forest type": "forest_type",
    "Forest stand development": "stand_development",
    "Stand development code": "stand_development_code",
    "Plant functional type": "pft",
    "exp": "exp",
}

SITE_KEY = ["Latitude", "Longitude", "Date"]


def load_trees(xlsx: str | Path) -> pd.DataFrame:
    """The `tree-level` sheet, with the numeric columns coerced."""
    df = pd.read_excel(xlsx, sheet_name=SHEET)
    for c in ("Date", "Latitude", "Longitude", "D", "DAT", "H"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("um", "Elevation", "Slope", "exp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def load_sites(xlsx: str | Path, window: int = WINDOW_YEARS,
               cell_km: float = 5.0) -> pd.DataFrame:
    """One row per site: the extraction unit, with its causal window and projected position.

    Rows without `Date` are dropped: there is no census year, so there is no window to
    define. See `dropped_report` for the count.
    """
    trees = load_trees(xlsx)
    kept = trees.dropna(subset=SITE_KEY)

    agg = {v: (k, "first") for k, v in SITE_ATTRS.items()}
    sites = kept.groupby(SITE_KEY, as_index=False, sort=True).agg(
        n_trees=("um", "size"),
        n_um=("um", "nunique"),
        um_ids=("um", lambda s: ",".join(str(u) for u in sorted(set(s.dropna())))),
        **agg,
    )
    sites = sites.rename(columns={"Latitude": "lat", "Longitude": "lon", "Date": "year"})
    sites["year"] = sites["year"].astype(int)

    # Sorted by (lat, lon, year) through the groupby above, so the id is stable across runs
    # as long as the workbook does not change.
    sites.insert(0, "site_id", [f"LT{i:04d}" for i in range(len(sites))])

    tf = Transformer.from_crs(CRS_WGS84, CRS_TARGET, always_xy=True)
    x, y = tf.transform(sites["lon"].to_numpy(), sites["lat"].to_numpy())
    sites["X"], sites["Y"] = x, y

    sites["win_start"] = sites["year"] - (window - 1)
    sites["win_end"] = sites["year"]
    sites["win_years"] = window

    sites["pixel_id"] = (
        (sites["X"] / PIXEL_M).round().astype(int).astype(str) + "_"
        + (sites["Y"] / PIXEL_M).round().astype(int).astype(str)
    )
    sites["cell"] = cell_id(sites["X"].to_numpy(), sites["Y"].to_numpy(), cell_km)

    # `exp` is the expansion factor, trees/ha represented by each stem, so the plot area is
    # its inverse. 250 or 500 m2 here: less than one 900 m2 Landsat pixel, which is why the
    # 5x5 mean is extracted alongside the centre pixel rather than instead of it.
    sites["plot_size_m2"] = 10000.0 / sites["exp"].astype(float)

    cols = ["site_id", "lat", "lon", "X", "Y", "year", "win_start", "win_end", "win_years",
            "pixel_id", "cell", "n_trees", "n_um", "um_ids", "plot_size_m2",
            *SITE_ATTRS.values()]
    return sites[cols]


def cell_id(x, y, km: float):
    """Load-grouping cell, same convention as `scripts/01_build_subset.py`."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return pd.Series(
        np.round(xa / (km * 1000)).astype(int).astype(str) + "_"
        + np.round(ya / (km * 1000)).astype(int).astype(str)
    )


def dropped_report(xlsx: str | Path) -> dict:
    """What was discarded on the way from tree rows to sites, and why."""
    trees = load_trees(xlsx)
    no_date = trees[trees["Date"].isna()]
    sites = load_sites(xlsx)
    return {
        "n_tree_rows": int(len(trees)),
        "n_rows_dropped_no_date": int(len(no_date)),
        "um_dropped_no_date": sorted(set(no_date["um"].dropna().astype(int).tolist())),
        "n_sites": int(len(sites)),
        "n_um": int(trees["um"].nunique()),
        "n_um_multi_coord": int(
            trees.groupby("um")[["Latitude", "Longitude"]].nunique().max(axis=1).gt(1).sum()
        ),
        "n_coord_multi_um": int(
            trees.groupby(["Latitude", "Longitude"])["um"].nunique().gt(1).sum()
        ),
        "n_sites_sharing_pixel": int(len(sites) - sites["pixel_id"].nunique()),
        "year_range": [int(sites["year"].min()), int(sites["year"].max())],
    }
