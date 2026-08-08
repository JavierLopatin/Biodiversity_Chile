"""Scoring. Pooled out-of-fold, original units, per target.

Two decisions that are easy to get wrong and hard to notice afterwards:

**Pool, do not average per-fold R-squared.** ``cv_groups.grouped_kfold`` documents why:
richness variance in Parcelas-CL is ~80% *between* groups, so a test fold that happens to
hold the low-richness projects has almost no response variance and returns R-squared near
zero however good the model is. Averaging those five numbers is meaningless. One score over
all 1,082 out-of-fold predictions uses the full response variance and sidesteps the problem;
per-fold values are kept as diagnostics only.

**Score in original units.** Targets are fitted on a Yeo-Johnson scale so that LCBD (1e-3)
and richness (1-50) contribute comparable gradient. An R-squared computed on that scale is
not the R-squared a reader will assume. Predictions are inverted before scoring; the
transformed-scale value is retained as a separate diagnostic column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

METRIC_NAMES = ["n", "R2", "RMSE", "nRMSE", "MAE", "Bias", "spearman"]


def _one(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(pred) & np.isfinite(obs)
    n = int(ok.sum())
    if n < 3:
        return {k: np.nan for k in METRIC_NAMES} | {"n": n}
    p, o = pred[ok], obs[ok]
    resid = p - o
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    rmse = float(np.sqrt(ss_res / n))
    # Normalised by the 1-99 percentile range rather than min-max: a single outlier plot
    # (and the PCoA axes have them) otherwise deflates nRMSE for every model equally and
    # makes the column useless for comparison.
    lo, hi = np.percentile(o, [1, 99])
    rng = float(hi - lo)
    return {
        "n": n,
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "RMSE": rmse,
        "nRMSE": rmse / rng if rng > 0 else np.nan,
        "MAE": float(np.mean(np.abs(resid))),
        "Bias": float(np.mean(resid)),
        "spearman": float(stats.spearmanr(p, o).statistic),
    }


def compute_metrics(pred: np.ndarray, obs: np.ndarray, names: list[str]) -> pd.DataFrame:
    """One row per target. ``pred``/``obs`` are (n_plots, n_targets); NaN means unobserved."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {obs.shape}")
    rows = [{"target": t, **_one(pred[:, j], obs[:, j])} for j, t in enumerate(names)]
    return pd.DataFrame(rows)


def pooled_metrics(oof: pd.DataFrame, names: list[str],
                   by: list[str] | None = None) -> pd.DataFrame:
    """Metrics from a long out-of-fold table with ``<target>_obs`` / ``<target>_pred`` columns.

    ``by`` groups first (e.g. ``["seed"]`` to get the per-seed pooled scores that are the
    paired unit for the significance tests, or ``["fold"]`` for the diagnostic view).
    """
    def _block(g: pd.DataFrame) -> pd.DataFrame:
        pred = np.column_stack([g[f"{t}_pred"].to_numpy(float) for t in names])
        obs = np.column_stack([g[f"{t}_obs"].to_numpy(float) for t in names])
        return compute_metrics(pred, obs, names)

    if not by:
        return _block(oof)
    out = []
    for key, g in oof.groupby(by[0] if len(by) == 1 else by, sort=True):
        keys = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        out.append(_block(g).assign(**keys))
    return pd.concat(out, ignore_index=True)


def ensemble_oof(oof: pd.DataFrame, names: list[str], id_col: str = "PlotObservationID") -> pd.DataFrame:
    """Average predictions across seeds → the ensemble result, one row per plot.

    Reported *alongside* the mean of the per-seed scores, never instead of it: the ensemble
    is essentially always the better number, and quoting only that hides the seed variance
    that tells the reader how stable a 15k-parameter model on n=1082 really is.
    """
    agg = {f"{t}_pred": "mean" for t in names} | {f"{t}_obs": "first" for t in names}
    return oof.groupby(id_col, sort=True).agg(agg).reset_index()


def paired_tests(per_seed: pd.DataFrame, value: str = "R2",
                 model_col: str = "run_id", alpha: float = 0.05,
                 targets: list[str] | None = None) -> pd.DataFrame:
    """All-pairs paired t-test and Wilcoxon over (target x seed) scores, Bonferroni-corrected.

    Adapted from ``Trait_2DCNN/evaluation/statistical_comparison.py:96-171``, with Wilcoxon
    added: with 9 targets x 5 seeds = 45 paired values the R-squared differences are not
    normal, and the rank test is the honest companion to the t-test.

    ``targets`` restricts the paired units. Pass it whenever the claim is about a subset:
    testing "model A predicts composition better" over all nine targets pools in the three
    alpha-diversity heads, where every model is at or below the training mean, and the
    resulting spread swamps the effect being tested.
    """
    import itertools

    if targets:
        per_seed = per_seed[per_seed["target"].isin(targets)]
    wide = per_seed.pivot_table(index=["target", "seed"], columns=model_col, values=value)
    models = list(wide.columns)
    n_pairs = max(1, len(models) * (len(models) - 1) // 2)
    rows = []
    for a, b in itertools.combinations(models, 2):
        pair = wide[[a, b]].dropna()
        if len(pair) < 3:
            continue
        d = pair[a].to_numpy() - pair[b].to_numpy()
        t_p = float(stats.ttest_rel(pair[a], pair[b]).pvalue)
        try:
            w_p = float(stats.wilcoxon(pair[a], pair[b]).pvalue)
        except ValueError:          # all differences zero
            w_p = 1.0
        rows.append(dict(
            model_a=a, model_b=b, n_pairs=len(pair),
            mean_a=float(pair[a].mean()), mean_b=float(pair[b].mean()),
            delta=float(d.mean()), sd_delta=float(d.std(ddof=1)),
            p_ttest=t_p, p_wilcoxon=w_p,
            p_ttest_bonf=min(1.0, t_p * n_pairs),
            p_wilcoxon_bonf=min(1.0, w_p * n_pairs),
            significant=min(1.0, max(t_p, w_p) * n_pairs) < alpha,
        ))
    return pd.DataFrame(rows).sort_values("delta", ascending=False, ignore_index=True)


def paired_plot_test(oof_a: pd.DataFrame, oof_b: pd.DataFrame, target: str,
                     id_col: str = "PlotObservationID") -> dict:
    """Plot-level paired test on squared error — n=1082 pairs instead of 45.

    Far more powerful than the aggregate test, and the right tool for the two headline
    finalists. Not used for the broad transform screen, where aggregate scores are the
    quantity of interest.
    """
    a = oof_a.set_index(id_col)
    b = oof_b.set_index(id_col)
    ids = a.index.intersection(b.index)
    ea = (a.loc[ids, f"{target}_pred"] - a.loc[ids, f"{target}_obs"]) ** 2
    eb = (b.loc[ids, f"{target}_pred"] - b.loc[ids, f"{target}_obs"]) ** 2
    ok = ea.notna() & eb.notna()
    ea, eb = ea[ok], eb[ok]
    return dict(
        target=target, n=int(ok.sum()),
        mse_a=float(ea.mean()), mse_b=float(eb.mean()),
        p_ttest=float(stats.ttest_rel(ea, eb).pvalue),
        p_wilcoxon=float(stats.wilcoxon(ea, eb).pvalue),
    )
