#!/usr/bin/env python3
# ---
# Source for `notebooks/archive/03_explore_modelling_results.ipynb`.
#
# Cells are separated by `# %%` (jupytext "percent" format). This .py file is the version
# that gets reviewed and diffed — a .ipynb does not review. Regenerate the notebook with:
#
#     python notebooks/build_notebook.py notebooks/archive/03_explore_modelling_results.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # Modelling results
#
# Reads `results/models/summary.csv` and the tables in `results/tables/`. Nothing is recomputed
# here: this notebook **reads** what scripts 09–13 produced and arranges it for inspection. If
# a number looks wrong, the place to fix it is the script, not the notebook.
#
# Reading order, which is the order in which the conclusions depend on one another:
#
# 1. **The controls.** Without `B02` (area only) and `B03` (coordinates only) alongside, no
#    richness R2 means anything.
# 2. **The optimism gap.** `R2(random CV) - R2(grouped CV)`. How much of the score was knowing
#    where the plot is rather than what grows in it.
# 3. **The three contrasts:** curve vs LSP, convolution vs MLP, 2-D vs 1-D.
# 4. **By vegetation index** and **by substrate**.
# 5. **Attribution over the phenological year** — the only output that reads as ecology.
#
# > For the five-facet results (phylogeny and dark diversity included) under the single
# > validation scheme, see `05_results_all_facets.ipynb`. Sections 1–9 here predate that and
# > several of their numbers come from the earlier `kfold5_owner` run; they are kept because
# > the optimism gap and the pixel ablation live in them.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from biodiv import targets as tg          # noqa: E402

MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
INTERP = ROOT / "results" / "interpretation"

# One validation scheme, by project decision (docs/10_findings.md 1b). The other two appear
# only in section 2, where the optimism gap IS the result; their rows come from the earlier
# run and are not extended.
PRIMARY, SPATIAL, OPTIMISTIC = "kfold5_window", "kfold5_block20", "kfold5_random"
pd.set_option("display.width", 200, "display.max_columns", 60)

summary = pd.read_csv(MODELS / "summary.csv")
print(f"{summary['run_id'].nunique()} runs, {summary['scheme'].nunique()} schemes, "
      f"{len(summary):,} rows (run x scheme x target x seed)")
summary.head()

# %% [markdown]
# ## 1. The controls first
#
# `B00` is the trained-mean predictor. Under random CV it **must** give R2 ~ 0; if it does not,
# the partition or the target scaling is broken and no other number is trustworthy.
#
# `B02` uses only plot area and abundance stratum. Its R2 is the share of richness attributable
# to sampling effort, with zero remote sensing.

# %%
def pooled(scheme=PRIMARY):
    """Seed-mean of the pooled out-of-fold R2, one row per run and target."""
    s = summary[summary["scheme"] == scheme]
    return (s.groupby(["run_id", "family", "target"])["R2"].mean()
            .unstack("target")
            .reindex(columns=tg.TARGETS_ALL))


ctrl = pooled(PRIMARY).loc[lambda d: d.index.get_level_values("family") == "BASE"]
print("Controls under kfold5_window (CV grouped by window component):")
display(ctrl.round(3))

print("\nB00 under kfold5_random — must be ~0 in every column:")
b00 = pooled(OPTIMISTIC)
display(b00[b00.index.get_level_values("run_id").str.startswith("B00")].round(3))

# %% [markdown]
# ## 2. The optimism gap
#
# A result, not a diagnostic: risk R8 of `docs/02_innovation_and_impact.md` explicitly asks for
# both numbers so the difference can be quoted. A large gap means the model learned *where* the
# plots are.

# %%
gap_path = TABLES / "optimism_gap.csv"
if gap_path.exists():
    gap = pd.read_csv(gap_path)
    g = gap.dropna(subset=["gap_vs_owner"]).sort_values("gap_vs_owner", ascending=False)
    display(g.head(15).round(3))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for t, sub in g.groupby("target"):
        ax.scatter(sub[PRIMARY], sub[OPTIMISTIC], s=22, alpha=0.75, label=t)
    lim = [min(-1.0, g[[PRIMARY, OPTIMISTIC]].min().min()), 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$R^2$ — CV grouped by contributor")
    ax.set_ylabel("$R^2$ — random CV (optimistic)")
    ax.set_title("How much of the score is spatial and protocol autocorrelation")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    plt.show()
else:
    print("optimism_gap.csv does not exist yet — run scripts/12_model_report.py")

# %% [markdown]
# ## 3. The three contrasts
#
# | Contrast | Runs | Question |
# |---|---|---|
# | curve vs LSP | `RF03` vs `RF01` | does the curve add over its 18 scalar summaries? (gap G1) |
# | convolution vs MLP | `C1D01` vs `MLP02` | does reading the curve as a sequence add anything? |
# | 2-D vs 1-D | best `C2D*` vs `C1D01` | does a second dimension add anything? |

# %%
# Alpha and beta are summarised separately, and that is not cosmetic: under contributor CV,
# phenology predicts **composition** and does not predict **alpha diversity**. A single average
# over the six complete targets mixes a real result with a null one and reports neither.
ALPHA = tg.FACETS["alpha"]
BETA = tg.FACETS["beta_pa"]
PHYLO = tg.FACETS["phylo"]
DARK = tg.FACETS["dark"]


def block_mean(df, cols):
    return df[[c for c in cols if c in df.columns]].mean(axis=1)


P = pooled(PRIMARY)
P = P.assign(R2_beta=block_mean(P, BETA), R2_alpha=block_mean(P, ALPHA),
             R2_phylo=block_mean(P, PHYLO), R2_dark=block_mean(P, DARK),
             R2_main=block_mean(P, tg.TARGETS_MAIN)).sort_values("R2_beta", ascending=False)
print("Top 20 runs by mean R2 over the composition targets (kfold5_window):")
display(P.head(20).round(3))

# %%
d = P.head(28).iloc[::-1]
colors = {"BASE": "#999999", "RF": "#4c72b0", "MLP": "#dd8452",
          "C1D": "#55a868", "C2D": "#c44e52"}
cols = [colors.get(f, "#8172b3") for f in d.index.get_level_values("family")]
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
for ax, key, title in zip(axes, ("R2_beta", "R2_alpha"),
                          ("composition and uniqueness\n(lcbd_pa, pcoa1_pa, pcoa2_pa)",
                           "alpha diversity\n(hill_q0, hill_q1, hill_q2)")):
    ax.barh(np.arange(len(d)), d[key], color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("pooled out-of-fold $R^2$")
    ax.set_title(title, fontsize=9)
axes[0].set_yticks(np.arange(len(d)))
axes[0].set_yticklabels(d.index.get_level_values("run_id"), fontsize=7)
axes[1].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
               colors.keys(), fontsize=7, loc="lower right", frameon=False)
fig.suptitle("Model comparison — kfold5_window", fontsize=11)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Vegetation index and substrate
#
# Two different questions: which index carries more information, and whether the indices
# combine or are redundant. `stack5` and `curve5` use all five at once — the first as rows of
# an image, the second as channels of a 1-D signal — so the pair isolates what the index axis
# gains by being a spatial dimension.
#
# > The averages below are **unbalanced by design**: kNDVI ran with far more CNN
# > configurations than the other indices. `05_results_all_facets.ipynb` §4 does the paired
# > comparison and reaches the opposite ranking. Trust that one.

# %%
s = summary[(summary["scheme"] == PRIMARY) & summary["target"].isin(tg.TARGETS_MAIN)]

by_index = (s[s["index"].notna() & (s["index"] != "")]
            .groupby(["family", "index"], as_index=False)["R2"].mean()
            .pivot(index="index", columns="family", values="R2"))
print("Mean R2 by vegetation index and family:")
display(by_index.round(3))

by_sub = (s[s["substrate"].notna() & (s["substrate"] != "")]
          .groupby("substrate")["R2"].agg(R2="mean", sd="std", n="size")
          .reset_index()
          .sort_values("R2", ascending=False))
print("\nMean R2 by substrate:")
display(by_sub.round(3))

# %%
if len(by_sub) > 1:
    fig, ax = plt.subplots(figsize=(7, 4))
    manufactured = {"reshape", "serpentine", "hilbert", "gaf", "mtf", "ndi", "cwt",
                    "cos2d", "spectrogram"}
    d = by_sub.iloc[::-1]
    cols = ["#c44e52" if x in manufactured else "#4c72b0" for x in d["substrate"]]
    ax.barh(np.arange(len(d)), d["R2"], xerr=d["sd"].fillna(0), capsize=2, color=cols)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["substrate"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean $R^2$ over the six complete targets")
    ax.set_title("Substrates: red = manufactured second dimension, blue = real", fontsize=9)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Per target
#
# The targets are not equally predictable and need not be. `hill_q0` is confounded with plot
# area (rho = 0.33); `lcbd_cover` and the `*_cover` axes only exist for 546 plots; the PCoA axes
# explain ~13 % of compositional variance, so there is a low ceiling by construction.

# %%
best = P.head(8).index.get_level_values("run_id")   # ordered by R2_beta
d = summary[(summary["scheme"] == PRIMARY) & summary["run_id"].isin(best)]
piv = d.groupby(["run_id", "target"])["R2"].mean().unstack().reindex(columns=tg.TARGETS_ALL)

fig, ax = plt.subplots(figsize=(9, 4))
w = 0.8 / len(piv)
for i, (r, row) in enumerate(piv.iterrows()):
    ax.bar(np.arange(len(row)) + i * w, row.to_numpy(), width=w, label=r)
ax.set_xticks(np.arange(len(tg.TARGETS_ALL)) + 0.4)
ax.set_xticklabels(tg.TARGETS_ALL, rotation=30, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("pooled out-of-fold $R^2$")
ax.legend(fontsize=6, ncol=2, frameon=False)
ax.set_title("Per-target performance of the eight best runs", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Observed against predicted
#
# R2 summarises; the scatter shows *how* it fails. Look for: compression towards the mean (the
# model predicts the mean and little else), per-fold bias (a level shift between contributors),
# and high-richness plots systematically underestimated.

# %%
top_run = P.index.get_level_values("run_id")[0]
oof = pd.read_csv(MODELS / top_run / PRIMARY / "oof_predictions.csv")
oof = oof[oof["seed"] == oof["seed"].min()]

show = ["hill_q0", "hill_q1", "lcbd_pa", "pcoa1_pa", "pcoa2_pa", "lcbd_cover"]
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, t in zip(axes.ravel(), show):
    if f"{t}_pred" not in oof:
        ax.axis("off"); continue
    x, y = oof[f"{t}_obs"], oof[f"{t}_pred"]
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=8, alpha=0.35, c=oof.loc[ok, "fold"], cmap="viridis")
    lim = [min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    r2 = 1 - ((y[ok] - x[ok]) ** 2).sum() / ((x[ok] - x[ok].mean()) ** 2).sum()
    ax.set_title(f"{t}   $R^2$={r2:.3f}", fontsize=9)
    ax.set_xlabel("observed", fontsize=7); ax.set_ylabel("predicted", fontsize=7)
    ax.tick_params(labelsize=6)
fig.suptitle(f"{top_run} — {PRIMARY}, color = fold", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Attribution over the phenological year
#
# The output that reads as ecology rather than as a metric: which weeks of the year the model
# looks at for each diversity facet. Produced by `scripts/13_interpretability.py` with
# Integrated Gradients, folded back onto the 52-week axis by each substrate's own inverse map,
# and labelled with the real DOY from `phenoshape_doy_grid.parquet`.

# %%
attr_files = sorted(INTERP.glob("doy_attribution_*.csv")) if INTERP.exists() else []
if attr_files:
    attr = pd.read_csv(attr_files[0])
    ts = [t for t in tg.TARGETS_ALL if t in set(attr["target"])]
    fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True)
    for ax, t in zip(axes.ravel(), ts):
        g = attr[attr["target"] == t].sort_values("step")
        ax.fill_between(g["step"], 0, g["attribution"], alpha=0.3, color="#c44e52")
        ax.plot(g["step"], g["attribution"], lw=1.2, color="#c44e52")
        ax.set_title(t, fontsize=8); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(ts):]:
        ax.axis("off")
    doy = attr.groupby("step")["doy"].first()
    for ax in axes[-1]:
        ax.set_xticks(np.arange(0, 52, 8))
        ax.set_xticklabels([int(doy.get(s, 0)) for s in np.arange(0, 52, 8)], fontsize=6)
        ax.set_xlabel("day of year", fontsize=7)
    fig.suptitle(f"Integrated gradients — {attr_files[0].stem.replace('doy_attribution_', '')}",
                 fontsize=10)
    plt.tight_layout(); plt.show()

    print("\nWeek of maximum attribution, per target:")
    display(attr.loc[attr.groupby("target")["attribution"].idxmax()]
            [["target", "step", "doy", "attribution"]].round(4))
else:
    print("no attributions yet — run scripts/13_interpretability.py --auto")

# %%
imp = sorted(INTERP.glob("rf_block_importance_*.csv")) if INTERP.exists() else []
if imp:
    b = pd.read_csv(imp[0]).set_index("target")
    print("Feature-block importance (Random Forest, permutation over OOF):")
    display(b.round(3))
    b.plot(kind="barh", stacked=True, figsize=(7, 4),
           title="Where the signal comes from, by predictor block")
    plt.xlabel("share of importance"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Significance
#
# With 9 targets and 3–5 seeds, R2 differences of 0.02 are noise. Any claim that one model wins
# needs this table: paired t-test and Wilcoxon over the (target x seed) scores, Bonferroni
# corrected within each comparison family.

# %%
tests_path = TABLES / "paired_tests.csv"
if tests_path.exists():
    tests = pd.read_csv(tests_path)
    sig = tests[tests["significant"]].sort_values("delta", ascending=False)
    print(f"{len(sig)} of {len(tests)} pairs significant after Bonferroni")
    display(sig.head(20).round(4))
else:
    print("paired_tests.csv does not exist yet — run scripts/12_model_report.py")

# %% [markdown]
# ## 9. Centre pixel against the 5x5 mean
#
# `results/px_ablation/` is a separate run repeating RF01/RF03/RF04 and the 1-D CNN with the
# whole design moved to the centre pixel, against the same design over the 5x5 window. The
# pairing is exact: same model, index, folds and seed, differing only in footprint. Three
# things pull in opposite directions — match with plot size (78.5–10,000 m2 against 900
# m2/pixel), per-pixel reconstruction noise, and missing data (200 NaN cells against 2) — which
# is why it is measured rather than decided.

# %%
px_path = TABLES / "pixel_ablation.csv"
if px_path.exists():
    px = pd.read_csv(px_path)
    for block in ("beta", "alpha"):
        d = px[px.block == block].sort_values("delta_center_minus_5x5")
        print(f"\n=== {block} ===  (delta > 0 favours the centre pixel)")
        display(d[["stem", "mean5x5", "center", "delta_center_minus_5x5"]].round(3))
        print(f"  mean delta = {d.delta_center_minus_5x5.mean():+.4f}   "
              f"centre wins in {int((d.delta_center_minus_5x5 > 0).sum())}/{len(d)}")

    d = px[px.block == "beta"].sort_values("mean5x5")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(d))))
    ax.barh(y - 0.19, d["mean5x5"], height=0.36, label="5x5 mean", color="#4c72b0")
    ax.barh(y + 0.19, d["center"], height=0.36, label="centre pixel", color="#dd8452")
    ax.set_yticks(y); ax.set_yticklabels(d["stem"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("pooled out-of-fold $R^2$ — composition targets")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.show()
else:
    print("pixel_ablation.csv does not exist yet — run scripts/15_pixel_ablation.py --run")

# %% [markdown]
# ## 10. Phylogenetic diversity and accumulation curves
#
# The facet `scripts/07_compute_taxonomic_beta_responses.R` left open because there was "no
# phylogeny referenced in this repo". `scripts/25_compute_phylo_responses.R` closes it with the
# `GBOTB.extended.TPL` megatree from V.PhyloMaker2 (74,529 tips; GBOTB backbone of Smith &
# Brown 2018 extended with Zanne et al. 2014).
#
# The curves replicate **Fig. 4b,c of the Parcelas-CL paper** with the method that figure uses
# — iNEXT rarefaction/extrapolation (Chao et al.) — and add the comparison that is missing
# there: this project's 1,082-plot subset against the full 1,485. The question is whether
# filtering by central Chile, year >= 1999 and unique coordinate cost representativeness, and
# whether it cost it taxonomically, phylogenetically, or both.
#
# **The paper's n.** Parcelas-CL reports 675 species, but that counts 54 records determined
# only to genus and 7 only to family. At species rank there are 597 taxa, and collapsing
# subspecies and varieties to the binomial leaves **601**, which is what a tree can represent:
# giving a genus-level record a tip would invent a lineage nobody observed.
#
# **The scale of the phylogenetic panel.** The paper's axis reaches ~60, not the tens of
# thousands of a PD summed in millions of years: it is `meanPD`, PD divided by tree depth
# (390.7 Ma here), which reads as an effective number of lineages.
#
# **The band.** It is not iNEXT's. It is the dispersion of subsampling plots without
# replacement, centred on the curve by construction, and it covers only the interpolated
# stretch. Both alternatives fail measurably and are documented in `scripts/lib/pd_inext.R`.
# For the extrapolated stretch there is no subsampling analogue, so it goes without a band:
# its uncertainty lives precisely in the species that have not been seen.
#
# **What the paper does not document.** Cerda-Paredes et al. report phylogenetic diversity in
# Fig. 4c but never state the tree, the package or the backbone. The tree used here is our own
# and is documented; if the authors share theirs, theirs is preferable.

# %%
CURVES = ROOT / "data" / "derived" / "rarefaction_inext.csv"
ASYMPT = ROOT / "data" / "derived" / "rarefaction_asymptote.csv"
PHYLO = ROOT / "data" / "derived" / "phylo_responses.parquet"

if CURVES.exists():
    cur = pd.read_csv(CURVES)
    colors = {"Parcelas-CL completo": "#2f6f7f", "subset del proyecto": "#c1553b"}
    labels = {"Parcelas-CL completo": "full Parcelas-CL",
              "subset del proyecto": "project subset"}
    panels = [("taxonomica", "Taxonomic richness"),
              ("filogenetica_meanPD", "Phylogenetic richness (meanPD)")]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)
    for ax, (metric, ylab) in zip(axes, panels):
        for d, g in cur[cur["metric"] == metric].groupby("dataset"):
            g = g.sort_values("n")
            obs = g[g["method"] == "Observed"]
            interp = g[g["method"] != "Extrapolation"]
            # the observed point belongs to both stretches, otherwise the line breaks
            extrap = pd.concat([obs, g[g["method"] == "Extrapolation"]])
            ax.plot(interp["n"], interp["value"], lw=1.8, color=colors[d], label=labels[d])
            ax.plot(extrap["n"], extrap["value"], lw=1.8, ls="--", color=colors[d])
            ax.plot(obs["n"], obs["value"], "o", ms=6, color=colors[d])
            b = g[g["lo"].notna()]
            ax.fill_between(b["n"], b["lo"], b["hi"], alpha=0.18, color=colors[d], lw=0)
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[1].set_xlabel("Sampling units (plots)")
    fig.suptitle("Rarefaction and extrapolation — replica of Parcelas-CL Fig. 4b,c\n"
                 "solid = observed, dashed = extrapolated to 2n", fontsize=10)
    plt.tight_layout(); plt.show()

    print("How much is left to discover, according to the sampling itself:\n")
    for metric, lab in panels:
        for d, g in cur[cur["metric"] == metric].groupby("dataset"):
            g = g.sort_values("n")
            o = g[g["method"] == "Observed"].iloc[0]; e = g.iloc[-1]
            print(f"  {lab:32s} {labels[d]:20s} n={o['n']:>4.0f}: {o['value']:8.1f}"
                  f"  ->  2n={e['n']:>4.0f}: {e['value']:8.1f}  (+{100*(e['value']/o['value']-1):.1f} %)")

    if ASYMPT.exists():
        asy = pd.read_csv(ASYMPT)
        print("\nTaxonomic richness asymptote (Chao2) — the total the sampling implies:\n")
        for _, a in asy.iterrows():
            print(f"  {a['Assemblage']:22s} observed {a['TD_obs']:5.1f}  ->  "
                  f"asymptote {a['TD_asy']:6.1f} +- {a['s.e.']:.1f}   "
                  f"({100*(1-a['TD_obs']/a['TD_asy']):.0f} % of the flora unseen)")

    # What decides whether filtering cost representativeness: at equal plot counts, do the
    # two curves coincide? If the subset sits below, the filter was not neutral. The two sets
    # are evaluated on different n grids (each on its own knots), so matching by exact value
    # finds nothing: they have to be interpolated onto a common grid.
    print("\nAt equal effort (interpolated stretch, common grid):\n")
    for metric, lab in panels:
        g = cur[(cur["metric"] == metric) & (cur["method"] != "Extrapolation")]
        a = g[g["dataset"] == "Parcelas-CL completo"].sort_values("n")
        b = g[g["dataset"] == "subset del proyecto"].sort_values("n")
        grid = np.linspace(20, min(a["n"].max(), b["n"].max()), 200)
        ratio = (np.interp(grid, b["n"], b["value"])
                 / np.interp(grid, a["n"], a["value"]))
        print(f"  {lab:32s} the subset retains {100*ratio.mean():.1f} % "
              f"(from {100*ratio[0]:.1f} % at 20 plots to {100*ratio[-1]:.1f} % "
              f"at {grid[-1]:.0f})")
else:
    print("curves do not exist yet — run scripts/26_rarefaction_inext.R")

# %% [markdown]
# ### 10.1 Which phylogenetic facets work as targets
#
# Faith's `PD` correlates +0.95 with richness: it is richness relabelled, and richness is
# exactly what cannot be predicted under contributor CV (§7.1 of `docs/08_modelling.md`). The
# ones carrying new information are **MPD** (nearly orthogonal to richness) and **MNTD**, plus
# the **SES**, which separate "there are many species" from "there are many distinct lineages".

# %%
if PHYLO.exists():
    ph = pd.read_parquet(PHYLO)
    print(f"{len(ph)} plots, {ph.pd_faith.isna().sum()} without phylogenetic coverage\n")
    metrics = ["pd_faith", "mpd", "mntd", "ses_pd", "ses_mpd", "ses_mntd"]
    display(ph[metrics + ["n_sp_tree"]].describe().round(3))

    # the target-selection criterion, as a table
    rows = []
    resp_tax = pd.read_parquet(ROOT / "data/derived/biodiversity_responses.parquet")
    m = ph.merge(resp_tax, on="PlotObservationID", how="left")
    from scipy import stats as st
    for v in metrics:
        ok = m[v].notna()
        rows.append(dict(
            metric=v,
            r_richness=st.spearmanr(m.loc[ok, v], m.loc[ok, "n_sp_tree"]).statistic,
            r_hill_q0=st.spearmanr(m.loc[ok, v], m.loc[ok, "hill_q0"]).statistic,
            r_pcoa1=st.spearmanr(m.loc[ok, v], m.loc[ok, "pcoa1_pa"]).statistic,
            r_lcbd=st.spearmanr(m.loc[ok, v], m.loc[ok, "lcbd_pa"]).statistic))
    tab = pd.DataFrame(rows).round(3)
    print("Spearman against richness and against the existing taxonomic targets.")
    print("A metric with high |r| against hill_q0 does not add a new target:\n")
    display(tab)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, v in zip(axes.ravel(), metrics):
        ax.scatter(m["n_sp_tree"], m[v], s=7, alpha=0.3, c="#4c72b0")
        ok = m[v].notna() & m["n_sp_tree"].notna()
        r = st.spearmanr(m.loc[ok, "n_sp_tree"], m.loc[ok, v]).statistic
        ax.set_title(f"{v}   ρ={r:+.3f}", fontsize=9)
        ax.set_xlabel("species in the tree", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Each phylogenetic metric against richness — the flatter, the more new "
                 "information it carries", fontsize=10)
    plt.tight_layout(); plt.show()
else:
    print("responses do not exist yet — run scripts/25_compute_phylo_responses.R")

# %% [markdown]
# ## 11. Dark diversity
#
# The species that **could** be in a plot and are not (Partel, Szava-Kovats & Zobel 2011),
# estimated from co-occurrence with `DarkDiv` (Carmona & Partel 2021), hypergeometric method.
# `scripts/27_compute_dark_diversity.R`.
#
# Three decisions that change the result, each measured in `docs/12_phylo_and_rarefaction.md`:
#
# - **The pool is estimated from all 1,485 plots**, not from the 1,082 of the subset. With the
#   reduced pool the dependence on plot area rises from rho = +0.085 to **+0.315**.
# - **It is counted with a threshold (p > 0.9), not by summing probabilities.** Summing gives a
#   median of 253 dark species for plots with 5 observed — that is adding up 596 small numbers,
#   not an ecological result.
# - **Completeness `log(obs/dark)` does not work as a target**: rho = +0.988 with richness. It
#   is richness relabelled, like Faith's PD, and for the same structural reason.
#
# What comes out of this is **`dark_n`**, and it is the facet with the least contributor
# confounding in the project: R2 by `Owner` of 0.22, against 0.70 for log-richness. That
# matters because it is exactly the confounding that sinks alpha under `kfold5_window`.

# %%
DARK = ROOT / "data" / "derived" / "dark_diversity.parquet"
DARKSP = ROOT / "data" / "derived" / "dark_diversity_spatial.csv"

if DARK.exists():
    dk = pd.read_parquet(DARK)
    tax = pd.read_parquet(ROOT / "data/derived/biodiversity_responses.parquet")
    md = dk.merge(tax, on="PlotObservationID", how="left")
    print(f"{len(dk)} plots, {dk.dark_n.isna().sum()} without coverage")
    print(f"dark species: median {dk.dark_n.median():.0f}, "
          f"range {dk.dark_n.min():.0f}-{dk.dark_n.max():.0f}   "
          f"(observed: median {dk.n_obs.median():.0f})\n")

    from scipy import stats as st
    rows = []
    for v in ["dark_n", "pool_n", "dark_pd", "dark_mpd", "dark_prob", "completeness"]:
        ok = md[v].notna()
        rows.append(dict(
            metric=v,
            r_richness=st.spearmanr(md.loc[ok, v], md.loc[ok, "n_obs"]).statistic,
            r_hill_q0=st.spearmanr(md.loc[ok, v], md.loc[ok, "hill_q0"]).statistic,
            r_pcoa1=st.spearmanr(md.loc[ok, v], md.loc[ok, "pcoa1_pa"]).statistic,
            r_lcbd=st.spearmanr(md.loc[ok, v], md.loc[ok, "lcbd_pa"]).statistic))
    print("Selection criterion: low |r| against richness and against the targets that already"
          "\nexist. A metric with high r against hill_q0 does not add a new target:\n")
    display(pd.DataFrame(rows).round(3))

    if DARKSP.exists():
        sp = pd.read_csv(DARKSP)
        print("\nSpatial validation — is the pool local, or a national list?")
        print("The absolute number says nothing without the baseline (any absent species):\n")
        for _, r in sp.iterrows():
            print(f"  {r['km']:>3.0f} km: of the dark species {100*r['oscuras']:4.1f} % have "
                  f"been seen nearby, against {100*r['base']:4.1f} % for any absent "
                  f"species  ->  {r['razon']:.1f}x")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (v, lab) in zip(axes, [
            ("dark_n", "dark species (p > 0.9)"),
            ("completeness", "completeness  log(obs/dark)"),
            ("near_frac", "fraction of dark species seen < 50 km away")]):
        ok = md[v].notna()
        ax.scatter(md.loc[ok, "n_obs"], md.loc[ok, v], s=7, alpha=0.25, c="#2f6f7f")
        r = st.spearmanr(md.loc[ok, "n_obs"], md.loc[ok, v]).statistic
        ax.set_title(f"{lab}\nρ = {r:+.3f}", fontsize=9)
        ax.set_xlabel("observed species", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("The middle panel is the negative result: completeness is richness again",
                 fontsize=10)
    plt.tight_layout(); plt.show()
else:
    print("does not exist yet — run scripts/27_compute_dark_diversity.R")
