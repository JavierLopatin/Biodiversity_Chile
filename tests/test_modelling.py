"""Gate checks for the modelling stage.

These are the assertions the plan requires to hold *before* any result is believed. They are
not unit tests of convenience: each one guards a failure mode that produces a plausible
number rather than an error.

    python -m pytest tests/test_modelling.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cv as cvmod            # noqa: E402
from biodiv import features as feat       # noqa: E402
from biodiv import substrates as sub      # noqa: E402
from biodiv import targets as tg          # noqa: E402
from biodiv.losses import MaskedHuberLoss, MaskedMSELoss   # noqa: E402
from biodiv.models_conv import PhenoNetS, Pheno1D, count_params  # noqa: E402
from biodiv.models_tabular import MLPMulti                 # noqa: E402

DERIVED = "data/derived"
FOLDS = ROOT / DERIVED / "cv_folds_modelling.parquet"


@pytest.fixture(scope="module")
def ids():
    return feat.plot_ids(DERIVED)


@pytest.fixture(scope="module")
def cv():
    if not FOLDS.exists():
        pytest.skip("run scripts/08_build_modelling_folds.py first")
    return cvmod.load_schemes(FOLDS)


# --------------------------------------------------------------------------------------
# alignment — the easiest place to produce a meaningless R-squared silently
# --------------------------------------------------------------------------------------

def test_design_and_targets_share_plot_order(ids):
    X, xids = feat.build_design("lsp+topo+area", index="ndvi", derived=DERIVED, ids=ids)
    yids, Y, names = tg.load_targets(DERIVED, "all", plot_ids=ids)
    assert list(xids) == list(ids)
    assert list(yids) == list(ids)
    assert len(X) == len(Y) == len(ids)


@pytest.mark.parametrize("name", ["curve1d", "curve5", "stack5", "reshape", "gaf", "pxcube"])
def test_substrate_shares_plot_order(ids, name):
    s = sub.make_substrate(name, index="ndvi", derived=DERIVED, ids=ids)
    assert list(s.plot_ids) == list(ids)
    assert s.X.shape[0] == len(ids)
    assert np.isfinite(s.X).all(), f"{name} produced non-finite values"


# --------------------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------------------

def test_no_train_test_overlap(cv):
    for scheme in sorted(set(cv["scheme"])):
        for fold, _, tr, te in cvmod.iter_folds(cv, scheme):
            assert len(tr.intersection(te)) == 0, f"{scheme} fold {fold}"


def test_partitioning_schemes_cover_every_plot_once(cv, ids):
    for scheme in sorted(set(cv["scheme"])):
        if scheme in cvmod.NON_PARTITIONING:
            continue
        sfolds = cvmod.scheme_folds(cv, scheme)
        test_ids = sfolds.loc[sfolds["split"] == "test", "PlotObservationID"]
        # los esquemas de subconjunto cubren menos, pero siguen sin repetir una parcela: es
        # justo la distinción que `SUBSET_SCHEMES` documenta
        if scheme not in cvmod.SUBSET_SCHEMES:
            assert len(test_ids) == len(ids), scheme
        else:
            assert 0 < len(test_ids) < len(ids), scheme
        assert test_ids.duplicated().sum() == 0, scheme


def test_inner_split_is_grouped_not_random(cv, ids):
    """The stopping epoch must not be chosen against a same-group plot."""
    plots = feat.load_tables(DERIVED).plots
    # kfold5_window is the primary scheme and its grouping column is *derived*, not stored:
    # if `ensure_group_col` stops materialising it the inner split silently falls back to a
    # KeyError at best and an ungrouped split at worst.
    for scheme in ("kfold5_window", "kfold5_owner", "kfold5_dataset"):
        group_col = cvmod.SCHEME_GROUP[scheme]
        plots_s = cvmod.ensure_group_col(plots, group_col)
        gmap = plots_s.set_index("PlotObservationID")[group_col]
        for fold, _, tr, te in cvmod.iter_folds(cv, scheme):
            fit, val = cvmod.inner_split(tr, plots_s, group_col, seed=0)
            assert len(fit.intersection(val)) == 0
            assert set(gmap[fit]).isdisjoint(set(gmap[val])), (
                f"{scheme} fold {fold}: inner split shares a {group_col} group")


def test_pixel_rows_follow_their_plot(cv, ids):
    """25 rows per plot share a 150 m footprint; splitting them across folds is direct leakage."""
    px = pd.read_parquet(ROOT / DERIVED / "phenoshape_pixels.parquet",
                         columns=["plot_id", "index", "y", "x"])
    px = px[px["index"] == "ndvi"]
    for fold, _, tr, te in cvmod.iter_folds(cv, "kfold5_owner"):
        rows_tr = cvmod.pixel_rows(px, tr)
        rows_te = cvmod.pixel_rows(px, te)
        assert set(rows_tr["plot_id"]).isdisjoint(set(rows_te["plot_id"]))
        assert len(rows_tr) + len(rows_te) == len(px)


# --------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------

def test_p_lcbd_can_never_be_a_target():
    for bad in tg.DROPPED:
        with pytest.raises(ValueError):
            tg.resolve_targets([bad])


def test_target_scaler_roundtrips_and_preserves_nan(ids):
    _, Y, _ = tg.load_targets(DERIVED, "all", plot_ids=ids)
    scaler = tg.fit_target_scaler(Y)
    Z = tg.apply_target_scaler(Y, scaler)
    assert np.array_equal(np.isnan(Y), np.isnan(Z))
    back = tg.inverse_target_scaler(Z, scaler)
    ok = ~np.isnan(Y)
    assert np.allclose(back[ok], Y[ok], atol=1e-6)


def test_smearing_stays_inside_the_training_range(ids):
    """The guard that stopped a 1-D CNN predicting 50 species for plots observed at 3."""
    _, Y, _ = tg.load_targets(DERIVED, "all", plot_ids=ids)
    rng = np.random.default_rng(0)
    scaler = tg.fit_target_scaler(Y)
    Z = tg.apply_target_scaler(Y, scaler)
    # deliberately hostile: predictions pushed far out, residuals wide
    pred = np.nan_to_num(Z, nan=0.0) + rng.normal(0, 3.0, Z.shape)
    resid = rng.normal(0, 2.0, Z.shape)
    out = tg.inverse_with_smearing(pred, scaler, resid, y_train=Y, seed=0)
    lo, hi = np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
    assert np.isfinite(out).all()
    assert (out >= lo - 1e-9).all() and (out <= hi + 1e-9).all()


#: Column sums of `inverse_with_smearing` on the real target table, frozen from the
#: implementation that produced every published R2. Regenerate ONLY when the estimator is
#: deliberately changed and the metrics in docs/20 are regenerated with it.
_SMEARING_COLSUM = [
    8866.16997486212, 4986.319965826697, 4119.632031874911, 0.9984697404633459,
    -11.409154447689009, 16.96664112830853, 1.9473174938930822, 82.13340493221826,
    -45.15936563098572, 276939.6136739014, 223648.11555233222, -180.54948205683408,
    -139.87303862614237, -133.3321375279994, 76199.56552668483,
]

#: First row of the same call, so a change that cancels out in the sums still trips.
_SMEARING_ROW0 = [
    6.91434347545928, 1.4080187672180304, 1.4013394831175805, 0.001057344227354028,
    0.04776225792764514, -0.10280481184819219, 0.0019409892853785894, 0.2535598284187128,
    -0.18070101052676596, 223.18807545799842, 219.63150447869404, -0.030493392989832106,
    -0.7831005381207994, 0.36094782509861345, 43.9834147709471,
]


def test_smearing_values_are_frozen(ids):
    """Every published R2 is computed after this call, so its values are an interface.

    `test_smearing_stays_inside_the_training_range` only checks finiteness and bounds: a
    change that altered the size of the bias correction would pass it silently and move
    every number in docs/20 without anything failing. This pins the values themselves.
    """
    _, Y, _ = tg.load_targets(DERIVED, "all", plot_ids=ids)
    rng = np.random.default_rng(0)
    scaler = tg.fit_target_scaler(Y)
    Z = tg.apply_target_scaler(Y, scaler)
    pred = np.nan_to_num(Z, nan=0.0) + rng.normal(0, 0.5, Z.shape)
    resid = rng.normal(0, 0.4, Z.shape)
    out = tg.inverse_with_smearing(pred, scaler, resid, y_train=Y, seed=0)
    np.testing.assert_allclose(np.nansum(out, axis=0), _SMEARING_COLSUM, rtol=1e-12)
    np.testing.assert_allclose(out[0], _SMEARING_ROW0, rtol=1e-12)


def test_smearing_table_matches_the_exact_path(ids):
    """The map path reads the estimator off a grid; this bounds what that costs.

    1e-4 is deliberately loose against the ~7e-6 measured on a real tile-year, so the test
    fails on a broken table rather than on the last digits of an interpolation.
    """
    _, Y, _ = tg.load_targets(DERIVED, "all", plot_ids=ids)
    rng = np.random.default_rng(0)
    scaler = tg.fit_target_scaler(Y)
    Z = tg.apply_target_scaler(Y, scaler)
    pred = np.nan_to_num(Z, nan=0.0) + rng.normal(0, 0.5, Z.shape)
    resid = rng.normal(0, 0.4, Z.shape)

    exact = tg.inverse_with_smearing(pred, scaler, resid, y_train=Y, seed=0)
    table = tg.build_smearing_table(scaler, resid, Y, seed=0)
    got = tg.inverse_with_smearing_table(pred, table)

    assert np.array_equal(np.isnan(got), np.isnan(exact))
    rel = np.abs(got - exact) / np.maximum(np.abs(exact), 1e-300)
    assert np.nanmax(rel) < 1e-4, f"worst relative error {np.nanmax(rel):.2e}"
    # the guards the exact path applies must survive tabulation
    lo, hi = np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
    assert np.isfinite(got).all()
    assert (got >= lo - 1e-9).all() and (got <= hi + 1e-9).all()


def test_smearing_table_preserves_the_nan_pattern(ids):
    """NaN in, NaN out -- the map writes those pixels as no-data."""
    _, Y, _ = tg.load_targets(DERIVED, "all", plot_ids=ids)
    rng = np.random.default_rng(1)
    scaler = tg.fit_target_scaler(Y)
    Z = tg.apply_target_scaler(Y, scaler)
    resid = rng.normal(0, 0.4, Z.shape)
    table = tg.build_smearing_table(scaler, resid, Y, seed=0)

    pred = np.nan_to_num(Z, nan=0.0)
    pred[3, 2] = np.nan
    pred[7, :] = np.nan
    got = tg.inverse_with_smearing_table(pred, table)
    assert np.isnan(got[3, 2]) and np.isnan(got[7]).all()
    assert np.isfinite(got[0]).all()


# --------------------------------------------------------------------------------------
# losses and models
# --------------------------------------------------------------------------------------

def test_masked_loss_matches_the_unmasked_one_when_nothing_is_masked():
    import torch
    torch.manual_seed(0)
    p, y = torch.randn(16, 9), torch.randn(16, 9)
    m = torch.ones_like(y)
    assert torch.allclose(MaskedMSELoss()(p, y, m), torch.nn.functional.mse_loss(p, y))
    assert torch.allclose(MaskedHuberLoss(1.0)(p, y, m),
                          torch.nn.functional.huber_loss(p, y, delta=1.0), atol=1e-6)


def test_fully_masked_batch_returns_zero_with_a_gradient():
    import torch
    p = torch.randn(8, 9, requires_grad=True)
    loss = MaskedHuberLoss()(p, torch.zeros(8, 9), torch.zeros(8, 9))
    assert float(loss) == 0.0
    loss.backward()                                  # must not raise


@pytest.mark.parametrize("width,expected", [("A", 5097), ("B", 16073), ("C", 55689)])
def test_phenonets_parameter_budget(width, expected):
    """Counts quoted in docs/08_modelling.md are measured, and stay measured."""
    m = PhenoNetS(c_in=1, width=PhenoNetS.WIDTHS[width], n_out=9, n_ctx=28)
    assert count_params(m) == expected
    assert count_params(m) < 60_000


def test_every_substrate_runs_through_one_model(ids):
    """AdaptiveAvgPool2d(1) is what makes the substrate benchmark a fair comparison."""
    import torch
    for name in ["reshape", "gaf", "cwt", "stack5", "pxcube"]:
        s = sub.make_substrate(name, index="ndvi", derived=DERIVED, ids=ids[:8])
        m = PhenoNetS(c_in=s.X.shape[1], n_out=9, n_ctx=28, pad_mode=s.pad_mode)
        out = m(torch.from_numpy(s.X[:4]), torch.zeros(4, 28))
        assert out.shape == (4, 9), name


def test_mlp_and_1d_accept_their_inputs():
    import torch
    assert MLPMulti(46, n_out=9)(torch.randn(4, 46)).shape == (4, 9)
    assert Pheno1D(c_in=5, n_out=9, n_ctx=28)(torch.randn(4, 5, 52),
                                              torch.zeros(4, 28)).shape == (4, 9)


# --------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------

def test_reshape_unfolds_exactly():
    from biodiv.transforms1d import ReshapeTransform
    x = np.linspace(0.1, 0.9, 52)
    img = ReshapeTransform().transform(x)
    back = sub.unfold_to_doy(img, "reshape")
    assert np.allclose(back, x, atol=1e-6)


def test_serpentine_unfolds_exactly():
    from biodiv.transforms1d import SerpentineTransform
    x = np.linspace(0.1, 0.9, 52)
    back = sub.unfold_to_doy(SerpentineTransform().transform(x), "serpentine")
    assert np.allclose(back, x, atol=1e-6)


def test_reshape_pads_by_wrapping_not_with_zeros():
    """Zero padding invents a trough at the boundary that a kernel reads as a real event."""
    from biodiv.transforms1d import ReshapeTransform
    x = np.linspace(0.3, 0.8, 52)
    flat = ReshapeTransform().transform(x).reshape(-1)
    assert np.allclose(flat[52:64], x[:12], atol=1e-6)
    assert flat.min() > 0.0


def test_normalisation_is_off_by_default():
    """Per-sample min-max would delete mean greenness and amplitude, which are the signal."""
    from biodiv.transforms1d import ReshapeTransform
    x = np.linspace(0.3, 0.8, 52)
    assert ReshapeTransform().transform(x).max() == pytest.approx(0.8, abs=1e-6)
    assert ReshapeTransform(normalize="perSample").transform(x).max() == pytest.approx(1.0)


def test_fractional_roll_is_circular_and_sub_step():
    """One step is 7 days; an integer roll would be a 7-35 day jitter, not the 3-5 intended."""
    from biodiv.augment import frac_roll
    x = np.sin(np.linspace(0, 2 * np.pi, 52, endpoint=False))
    assert np.allclose(frac_roll(x, 0.0), x, atol=1e-9)
    assert np.allclose(frac_roll(x, 52.0), x, atol=1e-9)          # wraps
    half = frac_roll(x, 0.5)
    assert not np.allclose(half, x)
    assert np.abs(half - x).max() < np.abs(np.roll(x, 1) - x).max()


# --------------------------------------------------------------------------------------
# kfold5_window — the primary scheme
# --------------------------------------------------------------------------------------

def test_window_scheme_never_splits_a_shared_extraction_window(cv, ids):
    """The reason this scheme exists.

    53% of plots share their 150 m extraction window with another plot: 575 plots in 135
    connected components, the largest with 17. `kfold5_owner` splits 7 of those components
    across folds (47 plots, recensuses 2-20 m apart filed under different contributors) and
    `kfold5_random` splits 97%. Here it must be exactly zero, by construction.
    """
    plots = feat.load_tables(DERIVED).plots
    comp = cvmod.ensure_group_col(plots, "window_component")
    gmap = comp.set_index("PlotObservationID")["window_component"]
    for fold, _, tr, te in cvmod.iter_folds(cv, "kfold5_window"):
        shared = set(gmap[tr]).intersection(set(gmap[te]))
        assert not shared, f"fold {fold} splits window components {sorted(shared)[:5]}"


def test_window_components_are_symmetric_and_transitive(ids):
    """A component is a connected component: if A overlaps B and B overlaps C, all three
    share a label even when A and C do not overlap each other."""
    plots = feat.load_tables(DERIVED).plots
    comp = cvmod.ensure_group_col(plots, "window_component")
    xy = comp[["X", "Y"]].to_numpy()
    lab = comp["window_component"].to_numpy()
    dx = np.abs(xy[:, 0][:, None] - xy[:, 0][None, :])
    dy = np.abs(xy[:, 1][:, None] - xy[:, 1][None, :])
    overlap = (dx < 150) & (dy < 150)
    same = lab[:, None] == lab[None, :]
    assert not (overlap & ~same).any(), "two overlapping plots landed in different components"


# --------------------------------------------------------------------------------------
# the five facets
# --------------------------------------------------------------------------------------

def test_every_target_has_a_facet_and_every_facet_a_source(ids):
    assert set(tg.FACET_OF) == set(tg.TARGETS_ALL)
    # TARGET_SOURCE is a superset, not an exact match: it also carries diagnostic-only
    # targets (e.g. TARGETS_MAIN_SAR) that are deliberately excluded from TARGETS_ALL so
    # the 79-run matrix and the row count below stay untouched.
    assert set(tg.TARGETS_ALL) <= set(tg.TARGET_SOURCE)
    assert sum(len(v) for v in tg.FACETS.values()) == len(tg.TARGETS_ALL)


def test_the_three_parquets_join_without_losing_a_plot(ids):
    """load_targets concatenates three files written by three scripts; a mismatched index
    would silently produce NaN columns rather than an error."""
    yids, Y, names = tg.load_targets(DERIVED, "all", plot_ids=ids)
    assert list(yids) == list(ids)
    assert Y.shape == (len(ids), 15)
    n_nan = dict(zip(names, np.isnan(Y).sum(axis=0)))
    # the two by-design gaps, and nothing else
    assert all(n_nan[t] == 0 for t in tg.TARGETS_MAIN + tg.TARGETS_DARK)
    assert all(n_nan[t] == 536 for t in tg.TARGETS_COVER)      # the cover tier
    assert all(n_nan[t] == 113 for t in tg.TARGETS_PHYLO)      # no species in the tree


@pytest.mark.parametrize("bad", ["pd_faith", "completeness", "dark_pd", "dark_mpd"])
def test_the_redundant_facets_can_never_become_targets(bad):
    """Each of these was measured and rejected: pd_faith +0.948 with richness, completeness
    +0.988, dark_pd +0.96 with dark_n, dark_mpd +0.504 with plot area. See
    docs/12_phylo_and_rarefaction.md."""
    assert bad in tg.DROPPED
    with pytest.raises(ValueError):
        tg.resolve_targets([bad])
