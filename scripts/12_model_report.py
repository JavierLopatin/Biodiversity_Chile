#!/usr/bin/env python3
"""Aggregate every run into the comparison tables and figures.

Three products, in order of what a reader needs:

1. **The model comparison table.** Pooled out-of-fold R2 per (run, scheme, target), seed mean
   +/- sd next to the seed-ensemble value. Sorted so the ordering of the families is visible
   at a glance.

2. **The optimism gap.** `R2(kfold5_random) - R2(kfold5_owner)` and the same against
   `kfold5_block20`, per model. This is a result, not a diagnostic: risk R8 of
   `docs/02_innovation_and_impact.md` asks for both numbers precisely so the gap can be
   quoted, and a model whose gap is large is a model that has learned where the plots are
   rather than what grows there.

3. **Paired significance tests.** Paired t-test and Wilcoxon over the (target x seed) scores,
   Bonferroni-corrected within each comparison family. A ranking without these is a ranking
   of noise: with 9 targets and 3-5 seeds, differences of 0.02 in R2 are routine.

Everything is a glob over `results/models/*/*/`, so this script never needs to know which
driver produced a run.

Usage:
    python scripts/12_model_report.py
    python scripts/12_model_report.py --scheme kfold5_window --family C2D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import metrics as mx                # noqa: E402
from biodiv import targets as tg                # noqa: E402

PRIMARY = "kfold5_window"
SPATIAL = "kfold5_block20"
OPTIMISTIC = "kfold5_random"


def load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-seed long table, pooled table)."""
    summ = root / "summary.csv"
    if not summ.exists():
        raise SystemExit(f"{summ} not found — no runs have completed yet.")
    per_seed = pd.read_csv(summ)

    pooled = []
    for f in sorted(root.glob("*/*/pooled_metrics.csv")):
        run_id, scheme = f.parts[-3], f.parts[-2]
        pooled.append(pd.read_csv(f).assign(run_id=run_id, scheme=scheme))
    pooled = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    meta = per_seed[["run_id", "scheme", "family", "features", "substrate", "index",
                     "fusion", "model"]].drop_duplicates()
    if not pooled.empty:
        pooled = pooled.merge(meta, on=["run_id", "scheme"], how="left")
    return per_seed, pooled


#: Alpha and beta are summarised separately, and that is not presentation. Measured on the
#: first complete tier, phenology predicts *composition* under leave-contributor-out CV
#: (`pcoa1_pa` up to R2 = +0.41, `lcbd_pa` +0.17) and does not predict *alpha diversity* at
#: all (`hill_q0`, `hill_q1` negative for every model, including the ones that do well on
#: composition). A single mean over the six complete targets averages a real result against a
#: null one and reports neither.
ALPHA = ["hill_q0", "hill_q1", "hill_q2"]
BETA = ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"]
BETA_COVER = ["lcbd_cover", "pcoa1_cover", "pcoa2_cover"]


def comparison_table(pooled: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, scheme): alpha, beta and overall summaries plus every target."""
    keys = ["run_id", "scheme", "family", "substrate", "index", "fusion"]

    def block(names: list[str], label: str) -> pd.DataFrame:
        d = pooled[pooled["target"].isin(names)]
        return (d.groupby(keys, dropna=False)
                .agg(**{label: ("R2_mean", "mean"), f"{label}_sd": ("R2_sd", "mean"),
                        f"{label}_ens": ("R2_ensemble", "mean")})
                .reset_index())

    out = block(tg.TARGETS_MAIN, "R2_main")
    for names, label in ((ALPHA, "R2_alpha"), (BETA, "R2_beta"),
                         (BETA_COVER, "R2_beta_cover")):
        out = out.merge(block(names, label)[keys + [label]], on=keys, how="left")
    wide = pooled.pivot_table(index=["run_id", "scheme"], columns="target",
                              values="R2_mean").reset_index()
    return (out.merge(wide, on=["run_id", "scheme"], how="left")
            .sort_values("R2_beta", ascending=False, ignore_index=True))


def family_tests(per_seed: pd.DataFrame, scheme: str,
                 only: str | None = None) -> pd.DataFrame:
    """Paired tests run **within** each comparison family, not across all runs at once.

    Bonferroni must be scoped to the set of comparisons a claim is drawn from. Correcting one
    pooled table of every pair against every other — 3,321 pairs on this matrix — multiplies
    every p-value by 3,321 and guarantees nothing survives, including differences the design
    was built to test. Each family below corresponds to one question, and the correction runs
    over that family's pairs only.
    """
    s = per_seed[per_seed["scheme"] == scheme]
    families = {
        # does the curve beat its scalar summaries, and do the indices combine? (gap G1)
        "rf_feature_blocks": s[s["family"] == "RF"],
        # the substrate benchmark: manufactured second axis against real ones
        "substrates_2d": s[s["family"].isin(["C2D", "C1D"])],
        # tabular DL against the forests on comparable features
        "tabular": s[s["family"].isin(["MLP", "RF"])],
        # the headline: best of each family plus the two controls that bound it
        "families_vs_controls": s[s["run_id"].isin(_headline_runs(s))],
    }
    out = []
    for name, sub in families.items():
        if only and only not in name:
            continue
        if sub["run_id"].nunique() < 2:
            continue
        # Every claim in this project is about composition, so that is what gets tested.
        # Pooling in the three alpha heads — where no model beats the training mean — adds
        # spread that has nothing to do with the difference under test.
        out.append(mx.paired_tests(sub, targets=BETA).assign(comparison=name, targets="beta"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def headline_plot_tests(root: Path, per_seed: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Per-plot paired tests on squared error for the headline contrasts.

    The aggregate test cannot settle these. With three composition targets and three seeds
    there are nine paired values, so the Wilcoxon signed-rank statistic has a minimum
    attainable p of 2/2^9 = 0.0039; Bonferroni over the fifteen headline pairs pushes its
    floor to 0.059, above alpha, and the test can never reject however large the effect. The
    plot-level test pairs the squared errors of the two models on the same 1,082 plots, which
    is both the correct unit for "is this model better on this data" and two orders of
    magnitude more powerful.
    """
    runs = _headline_runs(per_seed[per_seed["scheme"] == scheme])
    oof = {}
    for r in runs:
        f = root / r / scheme / "oof_predictions.csv"
        if f.exists():
            d = pd.read_csv(f)
            oof[r] = mx.ensemble_oof(d, [c[:-4] for c in d.columns if c.endswith("_obs")])
    import itertools
    rows = []
    pairs = list(itertools.combinations(sorted(oof), 2))
    n_tests = max(1, len(pairs) * len(BETA))
    for a, b in pairs:
        for t in BETA:
            if f"{t}_pred" not in oof[a] or f"{t}_pred" not in oof[b]:
                continue
            r = mx.paired_plot_test(oof[a], oof[b], t)
            rows.append(dict(model_a=a, model_b=b, **r,
                             delta_mse=r["mse_a"] - r["mse_b"],
                             p_ttest_bonf=min(1.0, r["p_ttest"] * n_tests),
                             p_wilcoxon_bonf=min(1.0, r["p_wilcoxon"] * n_tests)))
    out = pd.DataFrame(rows)
    if len(out):
        out["significant"] = out[["p_ttest_bonf", "p_wilcoxon_bonf"]].max(axis=1) < 0.05
        out["better"] = np.where(out["delta_mse"] < 0, out["model_a"], out["model_b"])
    return out


def _headline_runs(s: pd.DataFrame) -> list[str]:
    """Best run of each family on the composition targets, plus B00 and B03."""
    beta = s[s["target"].isin(BETA)]
    picks = []
    for fam in ("RF", "MLP", "C1D", "C2D"):
        g = beta[beta["family"] == fam]
        if not g.empty:
            picks.append(g.groupby("run_id")["R2"].mean().idxmax())
    picks += [r for r in s["run_id"].unique() if r.startswith(("B00", "B03"))]
    return picks


def optimism_gap(pooled: pd.DataFrame) -> pd.DataFrame:
    """R2(random) - R2(grouped), per run and target. The size of this is itself the finding."""
    p = pooled[pooled["scheme"].isin([PRIMARY, SPATIAL, OPTIMISTIC])]
    w = p.pivot_table(index=["run_id", "target"], columns="scheme", values="R2_mean")
    for s in (PRIMARY, SPATIAL, OPTIMISTIC):
        if s not in w.columns:
            w[s] = np.nan
    w["gap_vs_owner"] = w[OPTIMISTIC] - w[PRIMARY]
    w["gap_vs_block20"] = w[OPTIMISTIC] - w[SPATIAL]
    return w.reset_index().sort_values("gap_vs_owner", ascending=False, ignore_index=True)


def plot_comparison(tbl: pd.DataFrame, out: Path, scheme: str, top: int = 25) -> None:
    """Two panels, beta and alpha, because one average over both would hide the finding."""
    d = tbl[tbl["scheme"] == scheme].head(top).iloc[::-1]
    if d.empty:
        return
    colors = {"BASE": "#999999", "RF": "#4c72b0", "MLP": "#dd8452",
              "C1D": "#55a868", "C2D": "#c44e52"}
    cols = [colors.get(f, "#8172b3") for f in d["family"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3, 0.30 * len(d))), sharey=True)
    for ax, key, title in zip(axes, ("R2_beta", "R2_alpha"),
                              ("composición y unicidad\n(lcbd_pa, pcoa1_pa, pcoa2_pa)",
                               "diversidad alfa\n(hill_q0, hill_q1, hill_q2)")):
        ax.barh(np.arange(len(d)), d[key], height=0.72, color=cols)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel("pooled out-of-fold $R^2$", fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(np.arange(len(d)))
    axes[0].set_yticklabels(d["run_id"], fontsize=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[1].legend(handles, colors.keys(), fontsize=6, loc="lower right", frameon=False)
    fig.suptitle(f"Model comparison — {scheme}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_gap(gap: pd.DataFrame, out: Path) -> None:
    d = gap.dropna(subset=["gap_vs_owner"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, g in d.groupby("target"):
        ax.scatter(g[PRIMARY], g[OPTIMISTIC], s=14, label=t, alpha=0.75)
    lo = float(np.nanmin([d[PRIMARY].min(), d[OPTIMISTIC].min(), -0.5]))
    ax.plot([lo, 1], [lo, 1], "k--", lw=0.8)
    ax.set_xlabel("$R^2$ — grouped CV by window component (kfold5_window)")
    ax.set_ylabel("$R^2$ — random CV (optimistic)")
    ax.set_title("How much of the score is spatial and protocol autocorrelation", fontsize=9)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_per_target(pooled: pd.DataFrame, out: Path, scheme: str) -> None:
    d = pooled[pooled["scheme"] == scheme]
    if d.empty:
        return
    best = (d[d["target"].isin(BETA)].groupby("run_id")["R2_mean"].mean()
            .sort_values(ascending=False).head(8).index)
    d = d[d["run_id"].isin(best)]
    order = tg.TARGETS_ALL
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / max(len(best), 1)
    for i, r in enumerate(best):
        g = d[d["run_id"] == r].set_index("target").reindex(order)
        ax.bar(np.arange(len(order)) + i * w, g["R2_mean"].to_numpy(), width=w, label=r)
    ax.set_xticks(np.arange(len(order)) + 0.4)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("pooled out-of-fold $R^2$")
    ax.set_title(f"Per-target performance of the eight best models — {scheme}", fontsize=9)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/models")
    p.add_argument("--tables", default="results/tables")
    p.add_argument("--figures", default="results/figures")
    p.add_argument("--scheme", default=PRIMARY)
    p.add_argument("--family", default=None, help="restrict the significance tests")
    args = p.parse_args()

    root, tdir, fdir = Path(args.root), Path(args.tables), Path(args.figures)
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    per_seed, pooled = load_all(root)
    print(f"{per_seed['run_id'].nunique()} runs, {len(pooled)} pooled rows")

    tbl = comparison_table(pooled)
    tbl.to_csv(tdir / "model_comparison.csv", index=False)
    print(f"-> {tdir / 'model_comparison.csv'}")
    top = tbl[tbl["scheme"] == args.scheme].head(15)
    print(f"\ntop 15 by mean R2 over the beta targets — lcbd_pa, pcoa1_pa, pcoa2_pa "
          f"({args.scheme}).\nAlpha is shown beside it because the two behave differently "
          "and a single average reports neither:")
    print(top[["run_id", "family", "substrate", "index", "R2_beta", "R2_alpha",
               "R2_main"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    gap = optimism_gap(pooled)
    gap.to_csv(tdir / "optimism_gap.csv", index=False)
    print(f"\n-> {tdir / 'optimism_gap.csv'}")

    tests = family_tests(per_seed, args.scheme, only=args.family)
    if len(tests):
        tests.to_csv(tdir / "paired_tests.csv", index=False)
        n_sig = int(tests["significant"].sum())
        print(f"\n-> {tdir / 'paired_tests.csv'}  ({len(tests)} pairs in "
              f"{tests['comparison'].nunique()} comparison families, {n_sig} significant)")
        if n_sig:
            print(tests[tests["significant"]]
                  .sort_values("delta", ascending=False)
                  .head(12)[["comparison", "model_a", "model_b", "delta",
                             "p_ttest_bonf", "p_wilcoxon_bonf"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plots = headline_plot_tests(root, per_seed, args.scheme)
    if len(plots):
        plots.to_csv(tdir / "headline_plot_tests.csv", index=False)
        print(f"\n-> {tdir / 'headline_plot_tests.csv'}  "
              f"({int(plots['significant'].sum())}/{len(plots)} significant, "
              "per-plot paired squared error, n=1082)")
        print(plots[plots["significant"]]
              .sort_values("delta_mse")
              .head(12)[["target", "model_a", "model_b", "better", "delta_mse",
                         "p_ttest_bonf"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plot_comparison(tbl, fdir / "fig08_model_comparison.pdf", args.scheme)
    plot_gap(gap, fdir / "fig09_optimism_gap.pdf")
    plot_per_target(pooled, fdir / "fig10_per_target.pdf", args.scheme)
    print(f"-> figures in {fdir}")


if __name__ == "__main__":
    main()
