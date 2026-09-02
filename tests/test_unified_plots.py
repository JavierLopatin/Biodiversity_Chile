"""Invariants of `data/derived/plots_unified.parquet`, the 3,102-plot pool.

The one that matters is that **`X`/`Y` are populated for every plot**. Those columns are
Parcelas-CL's UTM 19S coordinates; Living Trees' plot table carries only `lon`/`lat`, so
before `scripts/51` learned to derive them the 2,020 `LT_` rows held NaN. Nothing raised:
code reaching for them got NaN, and a `dc.load` bounding box built from a NaN corner failed
downstream as `IllegalArgumentException: Points of LinearRing do not form a closed
linestring` -- a geometry error several frames away from the missing coordinate that caused
it. These tests pin the column so that cannot come back silently.

`X_m`/`Y_m` are a different pair (Lambert Azimuthal Equal-Area, for distances across the full
footprint) and are deliberately not asserted against UTM here.

Read-only against the committed table; skipped when it has not been built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "data/derived/plots_unified.parquet"

UTM19S = "EPSG:32719"
#: pyproj round-trips to well under a micrometre; 1 cm is loose enough to survive a PROJ
#: version bump and still tight enough that a wrong CRS or a swapped axis order fails.
TOL_M = 0.01

pytestmark = pytest.mark.skipif(not PLOTS.exists(),
                                reason="plots_unified.parquet not built")


@pytest.fixture(scope="module")
def plots() -> pd.DataFrame:
    return pd.read_parquet(PLOTS).set_index("PlotObservationID")


def test_lon_lat_complete(plots):
    assert plots["lon"].notna().all()
    assert plots["lat"].notna().all()


def test_no_nan_projected_coordinates(plots):
    """The regression this file exists for."""
    n_x, n_y = int(plots["X"].isna().sum()), int(plots["Y"].isna().sum())
    assert n_x == 0 and n_y == 0, (
        f"{n_x} NaN X and {n_y} NaN Y; by source: "
        f"{plots[plots['X'].isna()]['source'].value_counts().to_dict()}")


def test_xy_matches_utm19s_reprojection_of_lonlat(plots):
    tr = Transformer.from_crs("EPSG:4326", UTM19S, always_xy=True)
    ex, ey = tr.transform(plots["lon"].to_numpy(), plots["lat"].to_numpy())
    dx = np.abs(plots["X"].to_numpy() - ex)
    dy = np.abs(plots["Y"].to_numpy() - ey)
    assert np.nanmax(dx) < TOL_M, f"max |X - reproject(lon)| = {np.nanmax(dx)} m"
    assert np.nanmax(dy) < TOL_M, f"max |Y - reproject(lat)| = {np.nanmax(dy)} m"


@pytest.mark.parametrize("prefix", ["PCL_", "LT_"])
def test_both_sources_present_and_projected(plots, prefix):
    """Checked per half: a fill that silently covered only one source would still pass the
    table-wide assertion above if the other half happened to be empty."""
    sub = plots[plots.index.str.startswith(prefix)]
    assert len(sub) > 0, f"no {prefix} rows"
    assert sub["X"].notna().all() and sub["Y"].notna().all()


def test_ids_unique_and_prefixed(plots):
    assert plots.index.is_unique
    assert plots.index.str.startswith(("PCL_", "LT_")).all()
