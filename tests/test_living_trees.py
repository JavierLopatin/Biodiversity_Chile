"""Regression guards for Living Trees Chile's plot metadata (`scripts/50_build_living_trees.py`)
and the site_id <-> PlotObservationID crosswalk to its satellite series
(`data/derived/living_trees/`).

Both guard a real bug found and fixed this session: `PlotSize_m2` was hardcoded to 500.0 for
every Living Trees plot, when `sites.parquet` (the satellite-extraction output) shows 564 of
2,020 plots are actually 250 m2 (a nested-subplot design, `exp=40` vs `exp=20`). A silent
regression back to the hardcoded value would flatten exactly the area-richness gradient
`docs/08_modelling.md`'s Spearman(area, hill_q0)=+0.33 check exists to quantify.

    python -m pytest tests/test_living_trees.py -q
"""

from __future__ import annotations

import pandas as pd

DERIVED = "data/derived"


def test_plot_size_is_not_uniformly_500():
    plots = pd.read_parquet(f"{DERIVED}/plots_unified.parquet")
    lt = plots[plots["source"] == "living_trees"]

    sizes = lt["PlotSize_m2"].value_counts()
    assert set(sizes.index) == {500.0, 250.0}, (
        f"expected exactly {{500.0, 250.0}}, got {dict(sizes)} -- if this is {{500.0}} alone, "
        "the scripts/50 hardcode regressed"
    )
    # measured this session: 1,456 at 500 m2 (exp=20), 564 at 250 m2 (exp=40); allow the split
    # to drift a little (re-extractions/coordinate merges can shift it by a handful of plots)
    # but not collapse to one bucket
    assert sizes[500.0] > 1000
    assert sizes[250.0] > 400
    assert lt["PlotSize_m2"].isna().sum() == 0


def test_stratum_is_basal_not_nan():
    plots = pd.read_parquet(f"{DERIVED}/plots_unified.parquet")
    lt = plots[plots["source"] == "living_trees"]
    assert (lt["stratum"] == "basal").all()


def test_crosswalk_covers_every_plot_by_exact_coordinate():
    """`site_id` (satellite series) <-> `PlotObservationID` (species/facets), by exact
    (lat, lon) -- the join `scripts/67_build_living_trees_curves.py` depends on."""
    lt_plots = pd.read_parquet(f"{DERIVED}/living_trees_plots.parquet")
    sites = pd.read_parquet(f"{DERIVED}/living_trees/sites.parquet")

    m = lt_plots.merge(sites[["site_id", "lat", "lon"]], on=["lat", "lon"], how="left")
    assert m["site_id"].notna().all(), "every species-side plot must resolve to a site_id"
    assert (m.groupby("PlotObservationID")["site_id"].nunique() == 1).all()

    # the one satellite-only site (all-unidentified-species plot, correctly absent from the
    # species side) is the sole expected orphan -- more than that means something upstream
    # changed and needs a look, not a silent widening of this bound
    unmatched = sites.merge(lt_plots[["PlotObservationID", "lat", "lon"]],
                            on=["lat", "lon"], how="left")
    assert unmatched["PlotObservationID"].isna().sum() <= 1
