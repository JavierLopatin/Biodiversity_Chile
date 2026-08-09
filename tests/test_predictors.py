"""Tests for the shape-free predictor blocks, the window CV scheme and the GDM family.

Section 10 of docs/09_predictors.md asks for these specifically. The ones that matter are
the invariance tests: a composite that changed when the time axis was permuted would not be
a control for phenology, and a GDM coefficient that could go negative would not be a
turnover model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import composites, cv as cvmod, gdm as gmod, geomedian as gm   # noqa: E402


# --------------------------------------------------------------------------------------
# composites: the whole point is that they carry no temporal information
# --------------------------------------------------------------------------------------

def test_curve_stats_are_invariant_to_permuting_time():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(20, 52))
    a = composites.curve_stats(V)
    b = composites.curve_stats(V[:, rng.permutation(52)])
    for k in composites.STAT_NAMES:
        assert np.allclose(a[k], b[k]), f"{k} changed when the time axis was shuffled"


def test_curve_stats_range_is_max_minus_min():
    V = np.array([[1.0, 5.0, 3.0, 2.0]])
    s = composites.curve_stats(V)
    assert s["range"][0] == pytest.approx(4.0)
    assert s["iqr"][0] == pytest.approx(np.percentile(V[0], 75) - np.percentile(V[0], 25))


# --------------------------------------------------------------------------------------
# geomedian
# --------------------------------------------------------------------------------------

def test_geometric_median_of_symmetric_cloud_is_the_centre():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert np.allclose(gm.geometric_median(X), np.zeros(2), atol=1e-5)


def test_geometric_median_resists_an_outlier_that_moves_the_mean():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(size=(200, 6)) * 0.01, np.full((1, 6), 50.0)])
    g = gm.geometric_median(X)
    assert np.linalg.norm(g) < np.linalg.norm(X.mean(axis=0))
    assert np.linalg.norm(g) < 1.0


def test_geometric_median_survives_landing_on_a_data_point():
    """The unmodified Weiszfeld iteration divides by zero here; Vardi-Zhang does not."""
    X = np.vstack([np.zeros((5, 3)), np.ones((1, 3))])
    g = gm.geometric_median(X)
    assert np.all(np.isfinite(g))
    assert np.allclose(g, np.zeros(3), atol=1e-6)


def test_mads_are_zero_when_every_observation_is_the_geomedian():
    X = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), (10, 1))
    e, s, b = gm.mads(X, X[0])
    assert (e, s, b) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_indices_from_bands_match_the_cube_definitions():
    b = dict(blue=0.05, green=0.08, red=0.10, nir=0.30, swir1=0.25, swir2=0.20)
    out = gm.indices_from_bands(b)
    assert out["ndvi"] == pytest.approx((0.30 - 0.10) / (0.30 + 0.10))
    assert out["kndvi"] == pytest.approx(np.tanh(out["ndvi"] ** 2))
    assert out["nbr"] == pytest.approx((0.30 - 0.20) / (0.30 + 0.20))
    assert out["savi"] == pytest.approx(((0.30 - 0.10) / (0.30 + 0.10 + 0.5)) * 1.5)


def test_seasonal_medians_use_the_austral_calendar():
    month = np.array([1, 1, 1, 7, 7, 7])
    v = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    out = gm.seasonal_medians(v, month)
    assert out["djf"] == pytest.approx(1.0)
    assert out["jja"] == pytest.approx(0.0)
    assert out["djf_jja"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# window components — docs/09_predictors.md section 10, check 4
# --------------------------------------------------------------------------------------

def test_window_components_merge_overlapping_plots_and_separate_distant_ones():
    plots = pd.DataFrame({
        "X": [0.0, 100.0, 260.0, 10_000.0],      # 0-100 overlap (100<150); 100-260 do not
        "Y": [0.0, 0.0, 0.0, 0.0],
    })
    comp = cvmod.window_components(plots, half_m=150.0)
    assert comp.iloc[0] == comp.iloc[1]
    assert comp.iloc[1] != comp.iloc[2]
    assert comp.iloc[3] not in set(comp.iloc[:3])


def test_window_components_are_transitive_through_a_chain():
    """A and C do not overlap each other but both overlap B, so all three are one group."""
    plots = pd.DataFrame({"X": [0.0, 100.0, 200.0], "Y": [0.0, 0.0, 0.0]})
    comp = cvmod.window_components(plots, half_m=150.0)
    assert comp.nunique() == 1


# --------------------------------------------------------------------------------------
# GDM
# --------------------------------------------------------------------------------------

def test_isplines_are_monotone_and_bounded():
    x = np.linspace(0, 10, 200)
    B = gmod.ispline_basis(x, np.array([0.0, 5.0, 10.0]))
    assert B.shape == (200, 3)
    assert np.all(np.diff(B, axis=0) >= -1e-12), "an I-spline must be non-decreasing"
    assert B.min() >= -1e-12 and B.max() <= 1 + 1e-12


def test_isplines_are_degenerate_but_finite_on_a_constant_predictor():
    B = gmod.ispline_basis(np.full(10, 3.0), np.array([3.0, 3.0, 3.0]))
    assert np.all(np.isfinite(B))


def test_gdm_deviance_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    A = np.abs(rng.normal(size=(200, 5)))
    y = rng.uniform(0.1, 0.99, size=200)
    beta = np.abs(rng.normal(size=5)) + 0.1
    _, g = gmod._deviance(beta, A, y)
    num = np.zeros_like(g)
    for k in range(len(beta)):
        step = np.zeros_like(beta)
        step[k] = 1e-6
        num[k] = (gmod._deviance(beta + step, A, y)[0]
                  - gmod._deviance(beta - step, A, y)[0]) / 2e-6
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4)


def test_gdm_coefficients_never_go_negative():
    """More environmental difference must never predict *less* compositional difference."""
    rng = np.random.default_rng(2)
    A = np.hstack([np.ones((400, 1)), np.abs(rng.normal(size=(400, 4)))])
    y = rng.uniform(0, 1, size=400)               # pure noise: nothing to gain from any term
    beta = gmod.fit_gdm(A, y)
    assert np.all(beta >= -1e-12)


def test_gdm_recovers_a_planted_signal_better_than_the_null():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=120)
    i, j = np.triu_indices(120, 1)
    y = -np.expm1(-(0.1 + 3.0 * np.abs(x[i] - x[j])))
    S, knots = gmod.spline_transform(x[:, None])
    A = gmod.pair_design(S, i, j)
    pred = gmod.predict_gdm(A, gmod.fit_gdm(A, y))
    assert gmod.deviance_explained(y, pred) > 0.9


def test_deviance_explained_is_zero_at_the_mean_and_one_at_a_perfect_fit():
    rng = np.random.default_rng(6)
    y = rng.uniform(0.05, 0.95, size=500)
    assert gmod.deviance_explained(y, y) == pytest.approx(1.0, abs=1e-9)
    assert gmod.deviance_explained(y, np.full_like(y, y.mean())) == pytest.approx(0.0,
                                                                                  abs=1e-9)


def test_gdm_predictions_stay_inside_the_unit_interval():
    rng = np.random.default_rng(4)
    A = np.abs(rng.normal(size=(100, 3))) * 20
    pred = gmod.predict_gdm(A, np.array([1.0, 5.0, 5.0]))
    assert pred.min() >= 0.0 and pred.max() <= 1.0


def test_sparse_cca_returns_sparse_unit_loadings():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 30))
    Z = np.zeros((60, 10))
    Z[:, 0] = X[:, 3] + 0.05 * rng.normal(size=60)      # only column 3 carries signal
    W = gmod.sparse_cca(X, Z, n_components=2, cx=0.2, cz=0.3)
    assert W.shape == (30, 2)
    assert np.allclose(np.linalg.norm(W, axis=0), 1.0, atol=1e-6)
    assert (np.abs(W[:, 0]) < 1e-8).sum() > 15, "the L1 budget should zero most loadings"
    assert np.argmax(np.abs(W[:, 0])) == 3
