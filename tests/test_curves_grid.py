"""The step grid built from observation-level series.

`scripts/32` stores observations with their dates instead of a fixed grid, so the grid is
built downstream -- by `scripts/29` for the labelled plots and by `scripts/31 --unlabelled`
for the MapBiomas pool. Both go through `biodiv.curves`, and these tests pin the behaviour the
two share, in particular the one that fails silently: a grid pinned to each series' own first
and last observation instead of to the causal window, which makes two samples from the same
window carry step axes covering different amounts of real time. That is the same class of bug
as `tests/test_step_cols.py`, one level further down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import curves                        # noqa: E402


def _obs(n=70, lo=0.0, hi=1095.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.choice(np.arange(lo, hi), n, replace=False)).astype(float)
    return t, np.tanh(np.sin(t / 58.0) ** 2)


def test_grid_length_is_what_was_asked():
    t, v = _obs()
    for ngs in (24, 52, 156, 196):
        assert curves.interp_grid(t, v, ngs, roll=1).shape == (ngs,)


def test_window_pinning_makes_series_comparable():
    """Two series over the same window must share a time axis, whatever they observed."""
    lo, hi = 0.0, 1095.0
    ta, va = _obs(seed=1, lo=lo, hi=hi)
    tb, vb = _obs(seed=2, lo=200.0, hi=900.0)          # a shorter clear-observation span
    a = curves.interp_grid(ta, va, 52, roll=1, t_min=lo, t_max=hi)
    b = curves.interp_grid(tb, vb, 52, roll=1, t_min=lo, t_max=hi)
    assert np.isfinite(a).all() and np.isfinite(b).all()

    # unpinned, b's grid stretches over its own span and step k means a different date
    b_free = curves.interp_grid(tb, vb, 52, roll=1)
    assert not np.allclose(b, b_free), "pinning must actually change the grid"


def test_too_few_observations_returns_nan_not_garbage():
    """The floor is 5, matching the plot pipeline. Below it, say so instead of inventing."""
    t = np.array([10.0, 20.0, 30.0, 40.0])
    v = np.array([0.1, 0.2, 0.3, 0.4])
    out = curves.interp_grid(t, v, 52)
    assert np.isnan(out).all()


def test_non_finite_observations_are_dropped_not_propagated():
    t, v = _obs()
    v = v.copy()
    v[::3] = np.nan
    out = curves.interp_grid(t, v, 52, roll=1)
    assert np.isfinite(out).all()


def test_degenerate_window_is_nan():
    t, v = _obs()
    assert np.isnan(curves.interp_grid(t, v, 52, t_min=100.0, t_max=100.0)).all()


def test_smoother_shrinks_variance_without_shifting_the_mean():
    t, v = _obs()
    raw = curves.interp_grid(t, v, 156, roll=1)
    sm = curves.interp_grid(t, v, 156, roll=5)
    assert sm.std() < raw.std()
    assert sm.mean() == pytest.approx(raw.mean(), abs=0.02)


def test_unsorted_observations_are_handled():
    """`np.interp` silently returns nonsense on an unsorted x -- the sort is load-bearing."""
    t, v = _obs()
    p = np.random.default_rng(3).permutation(len(t))
    assert np.allclose(curves.interp_grid(t, v, 52, roll=1),
                       curves.interp_grid(t[p], v[p], 52, roll=1))


def test_raw_series_matches_interp_grid_per_pixel():
    """`raw_series` is the 2-D caller; it must agree with the 1-D one it delegates to."""
    xr = pytest.importorskip("xarray")
    t, v = _obs()
    days = t.astype("datetime64[D]")
    cube = np.stack([np.stack([v, v * 0.5], -1)] * 1, -1)      # (time, 2, 1)
    da = xr.DataArray(cube, dims=("time", "y", "x"), coords={"time": days})
    out, doy = curves.raw_series(da, 52, roll=3)
    assert out.shape == (52, 2, 1) and doy.shape == (52,)
    tt = days.astype(float)
    np.testing.assert_allclose(
        out[:, 0, 0],
        curves.interp_grid(tt, v, 52, roll=3, t_min=tt.min(), t_max=tt.max()))
