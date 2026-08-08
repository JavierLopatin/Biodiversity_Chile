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
        assert len(test_ids) == len(ids), scheme
        assert test_ids.duplicated().sum() == 0, scheme


def test_inner_split_is_grouped_not_random(cv, ids):
    """The stopping epoch must not be chosen against a same-group plot."""
    plots = feat.load_tables(DERIVED).plots
    for scheme in ("kfold5_owner", "kfold5_dataset"):
        group_col = cvmod.SCHEME_GROUP[scheme]
        gmap = plots.set_index("PlotObservationID")[group_col]
        for fold, _, tr, te in cvmod.iter_folds(cv, scheme):
            fit, val = cvmod.inner_split(tr, plots, group_col, seed=0)
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
