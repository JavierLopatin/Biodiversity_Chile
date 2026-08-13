"""Response variables for modelling: which columns become heads, and how they are scaled.

`data/derived/biodiversity_responses.parquet` ships 11 numeric columns. Only 9 are targets.

**`p_lcbd_pa` and `p_lcbd_cover` are dropped.** Their Spearman correlation with the matching
`lcbd_*` column is **-1.00 exactly**: the permutation p-value is a rank transform of LCBD
itself within each tier. Fitting both would manufacture two duplicate "results" and would
double-count LCBD in any multi-task loss. If a significance panel is wanted, recompute it
post hoc from predicted LCBD.

**Two tiers, one masked head set.** The 6 `TARGETS_MAIN` columns exist for all 1,082 plots;
the 3 `TARGETS_COVER` columns exist only for the 546 plots in the ``cover`` abundance
stratum and are NaN elsewhere. That NaN is by design (script 07), not missing data: no plot
is ever dropped and nothing is ever imputed. The masked loss consumes it directly.

**Scaling is not cosmetic.** `hill_*` has skew 2.2-2.8 and `lcbd_*` lives on a 1e-3 scale.
Without a per-target power transform the LCBD heads contribute essentially zero gradient to
a joint loss and the model silently becomes a richness-only model. The transform is fitted
on the training fold only, and every reported metric is computed after inverting it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

ID_COL = "PlotObservationID"

#: Available for all 1,082 plots.
TARGETS_MAIN = ["hill_q0", "hill_q1", "hill_q2", "lcbd_pa", "pcoa1_pa", "pcoa2_pa"]

#: Available only for the 546 plots of the ``cover`` stratum; NaN elsewhere, by design.
TARGETS_COVER = ["lcbd_cover", "pcoa1_cover", "pcoa2_cover"]

#: Phylogenetic facet, from ``phylo_responses.parquet`` (script 25). 113 plots have no
#: species in the tree and stay NaN -- same masking contract as the cover tier.
#: `pd_faith` is deliberately absent: r = +0.948 with richness, so it is `hill_q0`
#: relabelled. See docs/12_phylo_and_rarefaction.md section 4.
TARGETS_PHYLO = ["mpd", "mntd", "ses_pd", "ses_mpd", "ses_mntd"]

#: Dark diversity, from ``dark_diversity.parquet`` (script 27). Complete for all plots.
#: Only the thresholded count survives the selection criterion: `completeness` is +0.988
#: with richness, `dark_pd` is +0.96 with `dark_n`, and `dark_mpd` is +0.50 with plot area.
#: See docs/12_phylo_and_rarefaction.md section 8.
TARGETS_DARK = ["dark_n"]

TARGETS_ALL = TARGETS_MAIN + TARGETS_COVER + TARGETS_PHYLO + TARGETS_DARK

#: Species-area-corrected richness, from `scripts/41_sar_correct_richness.R`. Diagnostic
#: only -- not part of TARGETS_ALL/TARGETS_MAIN, so the 79-run matrix and
#: `tests/test_modelling.py`'s `Y.shape == (n, 15)` are untouched. See that script's
#: docstring for why iNEXT/Chao-Jost rarefaction is not usable on this dataset instead.
TARGETS_MAIN_SAR = ["hill_q0_sar", "hill_q1_sar", "hill_q2_sar"]

#: Which parquet each target lives in. `load_targets` joins them on PlotObservationID.
TARGET_SOURCE = (
    {t: "biodiversity_responses.parquet" for t in TARGETS_MAIN + TARGETS_COVER}
    | {t: "phylo_responses.parquet" for t in TARGETS_PHYLO}
    | {t: "dark_diversity.parquet" for t in TARGETS_DARK}
    | {t: "biodiversity_responses_sar.parquet" for t in TARGETS_MAIN_SAR}
)

#: Reporting groups. Averaging R2 across facets that behave differently hides both: alpha
#: is not predictable across contributors while composition is, so one grand mean reports
#: neither. Every results table is broken down by these.
FACETS = {
    "alpha":      ["hill_q0", "hill_q1", "hill_q2"],
    "beta_pa":    ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"],
    "beta_cover": ["lcbd_cover", "pcoa1_cover", "pcoa2_cover"],
    "phylo":      TARGETS_PHYLO,
    "dark":       TARGETS_DARK,
}

#: target -> facet
FACET_OF = {t: f for f, ts in FACETS.items() for t in ts}

#: Never fit, never score.
#:
#: `p_lcbd_*`  -- Spearman == -1.00 with the matching lcbd_* column: the permutation
#:               p-value is a rank transform of LCBD itself within each tier.
#: `pd_faith`  -- +0.948 with richness (docs/12 section 4). Kept as a descriptor.
#: `completeness` -- +0.988 with richness. The measure was *proposed* as independent of
#:               pool size; on these data it is not (docs/12 section 8).
#: `dark_pd`   -- +0.96 with `dark_n`: the Faith trap again, one level up.
#: `dark_mpd`  -- +0.504 with log plot area. Measures protocol, not ecology.
#: `n_sp_tree`, `n_obs` -- richness by another name.
DROPPED = ["p_lcbd_pa", "p_lcbd_cover", "p_ses_pd", "p_ses_mpd", "p_ses_mntd",
           "pd_faith", "completeness", "dark_pd", "dark_lin", "dark_mpd",
           "dark_prob", "pool_n", "n_sp_tree", "n_obs", "near_frac", "jaccard_fav"]

#: Reported but not treated as an independent result: r(hill_q1, hill_q2) = 0.95. Kept in
#: the multi-output head as a cheap multi-task regulariser.
SECONDARY = ["hill_q2"]

TARGET_SETS = {
    "all": TARGETS_ALL,
    "main": TARGETS_MAIN,
    "cover": TARGETS_COVER,
    "phylo": TARGETS_PHYLO,
    "dark": TARGETS_DARK,
    #: the 9 taxonomic targets, i.e. what the 79 runs of docs/08_modelling.md scored
    "taxonomic": TARGETS_MAIN + TARGETS_COVER,
    #: raw vs species-area-corrected richness, side by side, for the kfold5_owner/
    #: kfold_time diagnostic -- does removing the plot-size confound recover any R2?
    "richness_sar_test": ["hill_q0", "hill_q1", "hill_q2"] + TARGETS_MAIN_SAR,
}


def resolve_targets(target_set: str | list[str]) -> list[str]:
    """Accept a named set or an explicit list; refuse anything on the DROPPED list."""
    names = TARGET_SETS[target_set] if isinstance(target_set, str) else list(target_set)
    bad = sorted(set(names) & set(DROPPED))
    if bad:
        raise ValueError(
            f"{bad} are a monotone transform of their lcbd_* column (Spearman -1.00) "
            "and must never be fitted as targets. See src/biodiv/targets.py."
        )
    return names


def load_targets(derived: Path | str = "data/derived",
                 target_set: str | list[str] = "all",
                 plot_ids: pd.Index | None = None) -> tuple[pd.Index, np.ndarray, list[str]]:
    """Return ``(plot_ids, Y, names)`` with ``Y`` shaped (n_plots, n_targets), NaN preserved.

    ``plot_ids`` forces a canonical row order. Always pass it: the design matrix and the
    target matrix are built from different parquet files whose natural row order differs,
    and a silent misalignment between X and Y is the single easiest way to produce a
    plausible-looking but meaningless R-squared.
    """
    names = resolve_targets(target_set)
    derived = Path(derived)

    # The facets live in three parquets written by three different scripts (07, 25, 27).
    # They are joined here rather than merged on disk so that each script stays the single
    # owner of its own output and a rerun of one never has to touch the others.
    frames = []
    for src in dict.fromkeys(TARGET_SOURCE[t] for t in names):
        cols = [t for t in names if TARGET_SOURCE[t] == src]
        f = derived / src
        if not f.exists():
            raise FileNotFoundError(
                f"{f} is missing but {cols} were requested. Run the script that writes it: "
                f"07 for biodiversity_responses, 25 for phylo_responses, 27 for dark_diversity.")
        frames.append(pd.read_parquet(f).set_index(ID_COL)[cols])
    df = pd.concat(frames, axis=1)
    if plot_ids is not None:
        missing = pd.Index(plot_ids).difference(df.index)
        if len(missing):
            raise KeyError(f"{len(missing)} plot ids absent from responses, e.g. {list(missing[:5])}")
        df = df.loc[plot_ids]
    return df.index, df[names].to_numpy(dtype=np.float64), names


def fit_target_scaler(y_train: np.ndarray) -> PowerTransformer:
    """Yeo-Johnson + standardise, per target, fitted on the training fold only.

    Vendored from ``Trait_2DCNN/training/data_loader.py:305-327``. `PowerTransformer` cannot
    see NaN, so missing entries are median-filled *for the fit only* — the fitted lambda is
    then applied to the observed entries and the NaN pattern is restored by
    :func:`apply_target_scaler`. Median-filling a column that is 50% NaN (the cover tier)
    biases lambda slightly toward the centre of the observed distribution; that is
    acceptable because the alternative, fitting on a different plot subset per target, makes
    the heads incomparable.
    """
    filled = np.asarray(y_train, dtype=np.float64).copy()
    for j in range(filled.shape[1]):
        col = filled[:, j]
        m = np.isnan(col)
        if m.all():
            raise ValueError(f"target column {j} is entirely NaN in this training fold")
        col[m] = np.nanmedian(col)
    scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    scaler.fit(filled)
    return scaler


def apply_target_scaler(y: np.ndarray, scaler: PowerTransformer) -> np.ndarray:
    """Transform while preserving the NaN pattern that drives the loss mask."""
    y = np.asarray(y, dtype=np.float64)
    nan = np.isnan(y)
    out = scaler.transform(np.nan_to_num(y, nan=0.0))
    out[nan] = np.nan
    return out


def inverse_target_scaler(y_scaled: np.ndarray, scaler: PowerTransformer) -> np.ndarray:
    """Back to original units. Every reported metric is computed on this side."""
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    nan = np.isnan(y_scaled)
    out = scaler.inverse_transform(np.nan_to_num(y_scaled, nan=0.0))
    out[nan] = np.nan
    return out


def target_mask(y: np.ndarray) -> np.ndarray:
    """1.0 where the target is observed, 0.0 where it is NaN — the loss weight."""
    return (~np.isnan(np.asarray(y, dtype=np.float64))).astype(np.float32)


# --------------------------------------------------------------------------------------
# retransformation bias
# --------------------------------------------------------------------------------------
#
# Models are fitted on the Yeo-Johnson scale so that LCBD (1e-3) and richness (1-50) produce
# comparable gradient. But the inverse transform of a conditional *mean* is not the
# conditional mean: for a right-skewed target it lands near the conditional median, which is
# systematically below it. Measured here, that alone drove the training-mean baseline to
# R2 = -0.11 on `hill_q0` under random CV, where it must be ~0 by construction.
#
# Duan's (1983) smearing estimator removes it non-parametrically: average the inverse
# transform over the empirical distribution of the model's own residuals, rather than
# inverting the point prediction. It costs `n_draws` extra inverse transforms and needs no
# distributional assumption, which matters because the nine targets are not lognormal, not
# normal, and not alike.

SMEARING_DRAWS = 128

#: Residual draws are winsorised before smearing. The inverse Yeo-Johnson is convex and
#: unbounded, so a single draw from the tail can dominate the average: measured without this
#: guard, RF on `hill_q1` returned R2 = -41.8 purely from one exploded prediction. Trimming
#: the extreme 5% of draws keeps the bias correction and removes the blow-up.
SMEARING_TRIM = (2.5, 97.5)


def inverse_with_smearing(pred_scaled: np.ndarray, scaler: PowerTransformer,
                          resid_scaled: np.ndarray,
                          y_train: np.ndarray | None = None,
                          n_draws: int = SMEARING_DRAWS,
                          seed: int = 0) -> np.ndarray:
    """Back-transform with a guarded Duan (1983) smearing estimator.

    ``resid_scaled`` are the model's residuals on the *training* fold, in transformed units,
    shaped (n_train, n_targets) with NaN where the target was unobserved. They must come from
    predictions the model did not fit directly — out-of-bag for a forest, inner-validation for
    a network — otherwise the residual spread is too small and the correction under-shoots.

    ``y_train`` (original units) clips the result to the observed training range. A model that
    extrapolates a Hill number to 10^4 species is not making a prediction, it is reporting a
    numerical artefact of the inverse transform, and letting that through would silently
    dominate every squared-error metric.

    Falls back to the plain inverse for any target whose residuals are unusable.
    """
    pred_scaled = np.asarray(pred_scaled, dtype=np.float64)
    resid_scaled = np.asarray(resid_scaled, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_t = pred_scaled.shape[1]

    draws = np.zeros((n_draws, n_t))
    usable = np.zeros(n_t, dtype=bool)
    for j in range(n_t):
        r = resid_scaled[:, j]
        r = r[np.isfinite(r)]
        if r.size >= 20:
            lo, hi = np.percentile(r, SMEARING_TRIM)
            draws[:, j] = rng.choice(np.clip(r, lo, hi), size=n_draws, replace=True)
            usable[j] = True

    nan = np.isnan(pred_scaled)
    base = np.nan_to_num(pred_scaled, nan=0.0)

    # Clip in the *fitted* space, before inverting, not only after. Trimming the residual
    # draws is not sufficient on its own: a prediction that already sits high on the
    # Yeo-Johnson scale plus an ordinary residual still lands where the convex inverse is
    # steep, and the average over draws is then dominated by that one arm. Measured without
    # this, the 1-D CNN predicted 50 species for plots observed at 3, and three such plots
    # drove a whole fold to R2 = -5.5. Bounding the fitted-space value to the range the
    # training targets actually occupy is the same statement as "do not extrapolate the
    # target", made where the arithmetic is still linear.
    lo_s = hi_s = None
    if y_train is not None:
        ys = apply_target_scaler(np.asarray(y_train, dtype=np.float64), scaler)
        lo_s = np.nanmin(ys, axis=0)
        hi_s = np.nanmax(ys, axis=0)
        base = np.clip(base, lo_s, hi_s)

    plain = scaler.inverse_transform(base)

    if usable.any():
        acc = np.zeros_like(base)
        for k in range(n_draws):
            z = base + draws[k]
            if lo_s is not None:
                z = np.clip(z, lo_s, hi_s)
            acc += scaler.inverse_transform(z)
        out = np.where(usable[None, :], acc / n_draws, plain)
    else:
        out = plain

    if y_train is not None:
        y_train = np.asarray(y_train, dtype=np.float64)
        out = np.clip(out, np.nanmin(y_train, axis=0), np.nanmax(y_train, axis=0))

    out[nan] = np.nan
    return out
