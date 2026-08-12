"""Invariants of the unlabelled sampling pool.

Two of these were written after the code failed them, and both failed silently -- the pool
came out looking fine and only measuring it revealed the problem:

* **minimum spacing** was enforced with one-point-per-grid-bucket, which is a weaker
  constraint than a minimum distance: adjacent buckets put two points one pixel apart, and the
  closest pair measured 50 m against a 1,000 m target. Measuring the spacing in raster pixels
  rather than projected metres then still let pairs through at 972 m.
* **class mix** was steered by ranking every candidate on class weight, which took shrubland
  first in every cell and produced 97.7 % shrubland with no forest at all, against a 48 %
  target -- the pool would have been a shrubland autoencoder.

Both are the same kind of defect the whole design exists to avoid: a pool that looks larger
than it is.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mb              # noqa: E402

PLOTS = ROOT / "data" / "derived" / "plots_subset.parquet"

pytestmark = pytest.mark.skipif(
    not mb.available_years() or not PLOTS.exists(),
    reason="needs the MapBiomas rasters and plots_subset.parquet")

MIN_SEP_KM = 1.0
EXCLUDED = {9, 18, 24, 15, 23, 25, 33, 34, 29, 0, 67, 79, 80}


def _sampler():
    spec = importlib.util.spec_from_file_location(
        "s32", ROOT / "scripts" / "32_sample_unlabelled.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pool():
    """A real pool over a handful of cells. Slow enough to build once, real enough to trust."""
    mod = _sampler()
    args = argparse.Namespace(
        n=20000, years_per_cell=2, min_sep_km=MIN_SEP_KM, cell_km=10.0, window=3,
        resolution=30, index="kndvi", screen_year=2014, cells=30, seed=0,
        derived="data/derived", out="x", dry_run=True)
    df, _ = mod.build_pool(args)
    assert len(df) > 100, "sampler produced too little to test"
    return df


def test_every_sample_is_native(pool):
    assert mb.is_native(pool.mb_code.to_numpy()).all()
    assert not (set(pool.mb_code.unique()) & EXCLUDED)


def test_minimum_spacing_is_metres_not_pixels(pool):
    """The constraint is a distance in the projected CRS, and it must actually hold."""
    worst = np.inf
    for _, g in pool.groupby(["cell", "year"]):
        if len(g) < 2:
            continue
        p = g[["X", "Y"]].to_numpy()
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        np.fill_diagonal(d, np.inf)
        worst = min(worst, d.min())
    assert worst >= MIN_SEP_KM * 1000 - 1.0, f"closest pair {worst:.0f} m"


def test_no_sample_sits_on_a_labelled_plot(pool):
    """The input-side leakage claim, checked rather than asserted.

    Spacing samples from each other does not space them from the plots. Before the exclusion
    existed, 29 of 16,937 samples fell inside a plot's own 5x5 window -- 10 inside its 75 m
    footprint, the closest at 14 m -- while `docs/15` claimed the pool was free of exactly
    that. Small, real, and the kind of thing that is never noticed once it is written down as
    a property instead of measured.
    """
    from scipy.spatial import cKDTree
    plots = pd.read_parquet(PLOTS)
    d, _ = cKDTree(plots[["X", "Y"]].to_numpy()).query(pool[["X", "Y"]].to_numpy())
    assert d.min() >= MIN_SEP_KM * 1000 - 1.0, (
        f"{(d < MIN_SEP_KM * 1000).sum()} samples within {MIN_SEP_KM} km of a plot, "
        f"closest {d.min():.0f} m")


def test_class_mix_is_not_a_single_class(pool):
    """Ranking by class weight collapsed the pool to 97.7 % shrubland. Quotas must not."""
    share = pool.mb_class.value_counts(normalize=True)
    assert share.iloc[0] < 0.80, f"dominated by {share.index[0]} at {share.iloc[0]:.1%}"
    assert (pool.mb_class.str.contains("Forest")).any(), "no forest in the pool"
    assert pool.mb_class.nunique() >= 3


def test_causal_window_is_always_three_years(pool):
    """Same window as the labelled extraction, or transfer is confounded with a domain shift."""
    assert ((pool.win_end - pool.win_start) == 2).all()
    assert (pool.win_end == pool.year).all()


def test_displaced_maps_only_outside_the_collection(pool):
    """1999-2024 must match exactly; only 2025-2026 may stretch, and only to 2024."""
    have = set(mb.available_years())
    inside = pool[pool.year.isin(have)]
    assert (inside.mb_delta == 0).all()
    outside = pool[~pool.year.isin(have)]
    if len(outside):
        assert (outside.mb_year_used == max(have)).all()


def test_sample_ids_are_unique(pool):
    assert pool.sample_id.is_unique
