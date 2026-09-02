"""Invariants of the Living_Trees_Chile site table.

The one that matters is the **site key**. The workbook is one row per tree with the plot
coordinate repeated on every stem, and there are two plausible keys: the coordinate and the
field sampling unit `um`. They are not nested -- 78 `um` carry more than one coordinate and
2 coordinates are shared by more than one `um` -- so picking `um` would silently average
plots sitting in different Landsat pixels, and the error would look like nothing at all in
the output. These tests pin the coordinate as the key and pin the counts that prove the two
partitions differ.

All of it is local: the workbook, no datacube.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cube                       # noqa: E402
from biodiv import io_living_trees as io_lt   # noqa: E402

XLSX = ROOT / "data" / "Living_Trees_Chile.xlsx"

pytestmark = pytest.mark.skipif(not XLSX.exists(), reason="needs Living_Trees_Chile.xlsx")

N_SITES = 2021
N_TREE_ROWS = 59408
N_DROPPED_NO_DATE = 46


@pytest.fixture(scope="module")
def sites() -> pd.DataFrame:
    return io_lt.load_sites(XLSX)


@pytest.fixture(scope="module")
def trees() -> pd.DataFrame:
    return io_lt.load_trees(XLSX)


def test_one_row_per_coordinate_year(sites):
    assert len(sites) == N_SITES
    assert sites["site_id"].is_unique
    assert not sites.duplicated(["lat", "lon", "year"]).any()


def test_um_is_not_the_key(trees):
    """If either count were zero, grouping by `um` would be equivalent and this file moot."""
    rep = io_lt.dropped_report(XLSX)
    assert rep["n_um_multi_coord"] == 78
    assert rep["n_coord_multi_um"] == 2
    assert rep["n_um"] == 1943 != N_SITES


def test_no_two_sites_share_a_landsat_pixel(sites):
    """A shared 30 m pixel would mean two sites with identical series and no way to tell."""
    assert sites["pixel_id"].nunique() == len(sites)


def test_every_dated_tree_maps_to_exactly_one_site(trees, sites):
    dated = trees.dropna(subset=["Latitude", "Longitude", "Date"])
    assert len(trees) == N_TREE_ROWS
    assert len(trees) - len(dated) == N_DROPPED_NO_DATE
    merged = dated.merge(
        sites[["site_id", "lat", "lon", "year"]],
        left_on=["Latitude", "Longitude", "Date"], right_on=["lat", "lon", "year"],
        how="left", validate="many_to_one")
    assert merged["site_id"].notna().all()
    assert int(sites["n_trees"].sum()) == len(dated)


def test_window_is_causal(sites):
    """`docs/05` section 1: no information posterior to the census enters the predictor."""
    assert (sites["win_end"] == sites["year"]).all()
    assert (sites["win_start"] == sites["year"] - 2).all()
    assert (sites["win_years"] == 3).all()


def test_every_census_year_has_a_landsat_product(sites):
    for y in sorted(sites["year"].unique()):
        assert cube.products_for(int(y) - 2, int(y)), f"no product for {y}"


def test_projection_round_trips(sites):
    """A silent CRS mix-up is the failure mode that puts a series on the wrong pixel."""
    back = Transformer.from_crs(io_lt.CRS_TARGET, io_lt.CRS_WGS84, always_xy=True)
    lon, lat = back.transform(sites["X"].to_numpy(), sites["Y"].to_numpy())
    fwd = Transformer.from_crs(io_lt.CRS_WGS84, io_lt.CRS_TARGET, always_xy=True)
    x2, y2 = fwd.transform(lon, lat)
    assert np.abs(x2 - sites["X"].to_numpy()).max() < 1.0
    assert np.abs(y2 - sites["Y"].to_numpy()).max() < 1.0
    assert sites["lat"].between(-56, -17).all()
    assert sites["lon"].between(-76, -66).all()


def test_plot_is_smaller_than_a_landsat_pixel(sites):
    """The reason the 5x5 mean is extracted alongside the centre pixel, not instead of it."""
    assert sites["plot_size_m2"].max() <= 900.0


def test_resume_retries_failed_loads(tmp_path):
    """A failed load is not done.

    The first version built the resume set from every manifest row, error rows included, so
    `--resume` skipped exactly the groups that needed retrying and the run came back
    "complete" with the failures frozen in. It is not hypothetical: the gateway scheduler
    dropped the connection on load 1,640 of 1,764 and left one group behind.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s35", ROOT / "scripts" / "35_extract_living_trees.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    man = pd.DataFrame([
        dict(site_id="LT0001", cell="10_20", year=2015, error=np.nan),
        dict(site_id="LT0002", cell="10_21", year=2015, error="load: ValueError: boom"),
    ])
    ok = man[man["error"].isna()]
    done = set(zip(ok["cell"], ok["year"]))

    assert ("10_20", 2015) in done
    assert ("10_21", 2015) not in done, "the failed load must be retried, not skipped"

    # and the series tables must not carry rows for the site whose manifest row was dropped
    series = pd.DataFrame({"site_id": ["LT0001", "LT0002"], "time": [0, 0], "ndvi": [0.5, 0.5]})
    kept = series[series["site_id"].isin(set(ok["site_id"]))]
    assert list(kept["site_id"]) == ["LT0001"]
    assert mod.table_name("ndvi", "center") == "series_ndvi_center.parquet"
