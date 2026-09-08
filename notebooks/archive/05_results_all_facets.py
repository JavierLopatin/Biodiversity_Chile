#!/usr/bin/env python3
# ---
# Source for `notebooks/archive/05_results_all_facets.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/archive/05_results_all_facets.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # Results: five facets, one scheme
#
# Reads `results/tables/results_*.csv`, produced by `scripts/28_results_tables.py`. Nothing is
# recomputed here.
#
# Two things changed relative to `03_explore_modelling_results.ipynb`, and they are why this
# notebook exists separately instead of being one more section:
#
# 1. **`kfold5_window` is the only validation scheme** (`docs/10_findings.md` §1b). The numbers
#    in `03` under `kfold5_owner` are **not comparable** with these. It is not that one set is
#    wrong: they measure different things, and mixing them in one figure would be the error.
# 2. **From 9 targets to 15.** The phylogenetic facet (MPD, MNTD and the three SES) and dark
#    diversity are added. Five facets: `alpha`, `beta_pa`, `beta_cover`, `phylo`, `dark`.
#
# And three metrics instead of one. A high R2 with large bias is a model that gets the shape of
# the cloud right and the level wrong — and with targets that go through Yeo-Johnson and come
# back via Duan's estimator, bias is exactly where that breaks. `%RMSE` and `bias` are
# normalised by the target's 1–99 range, so they are comparable across facets living on
# different scales (1–50 species against 1e-3).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from biodiv import targets as tg              # noqa: E402

TABLES = ROOT / "results" / "tables"
pd.set_option("display.width", 220, "display.max_columns", 40)

full = pd.read_csv(TABLES / "results_full.csv")
facet = pd.read_csv(TABLES / "results_by_facet.csv")
best = pd.read_csv(TABLES / "results_best.csv")

FACET_ORDER = ["alpha", "beta_pa", "beta_cover", "phylo", "dark"]
COLORS = {"BASE": "#999999", "RF": "#2f6f7f", "MLP": "#dd8452",
          "C1D": "#55a868", "C2D": "#c1553b"}

print(f"{full.run_id.nunique()} runs x {full.target.nunique()} targets "
      f"= {len(full)} rows, scheme kfold5_window")
print("\nseed noise (median R2 sd) per facet:")
print(full.groupby("facet")["R2_sd"].median().round(4).to_string())

# %% [markdown]
# ## 1. The control to look at before anything else
#
# `B03_coords` uses **only longitude, latitude and elevation**. Zero remote sensing, zero
# phenology. Under `kfold5_window` it is a brutal competitor, for a structural reason: the
# scheme groups by connected component of 150 m window overlap, not by contributor, so **every
# contributor is spread across folds**. Memorising the local level is free.
#
# The column that matters is not the best model's R2: it is the difference against this
# control, read against seed noise.

# %%
ctrl = (facet[facet.family == "BASE"]
        .pivot_table(index="run_id", columns="facet", values="R2")
        .reindex(columns=FACET_ORDER))
best_model = facet[facet.family != "BASE"].groupby("facet")["R2"].max().reindex(FACET_ORDER)
noise = full.groupby("facet")["R2_sd"].median().reindex(FACET_ORDER)

summary = pd.DataFrame({
    "B01_topo+area": ctrl.loc["B01_topo-area"],
    "B03_coords": ctrl.loc["B03_coords"],
    "best model": best_model,
    "gain over coords": best_model - ctrl.loc["B03_coords"],
    "seed noise (sd)": noise,
})
summary["beats noise?"] = np.where(
    summary["gain over coords"] > 2 * summary["seed noise (sd)"], "yes", "NO")
display(summary.round(3))

fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(FACET_ORDER))
ax.barh(y + 0.2, ctrl.loc["B03_coords"], height=0.38, color="#999999",
        label="coordinates only (B03)")
ax.barh(y - 0.2, best_model, height=0.38, color="#2f6f7f", label="best model")
for i, f in enumerate(FACET_ORDER):
    d = best_model[f] - ctrl.loc["B03_coords", f]
    ax.text(best_model[f] + 0.008, i - 0.2, f"+{d:.3f}", va="center", fontsize=8,
            color="#2f6f7f" if d > 2 * noise[f] else "#999999")
ax.set_yticks(y); ax.set_yticklabels(FACET_ORDER)
ax.set_xlabel("mean facet $R^2$")
ax.set_title("How much remote sensing adds over knowing where the plot is", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Read it this way.** On `alpha`, `beta_pa` and `beta_cover` the best model ties with the
# coordinates: the gains are of the order of seed noise. The two new facets are the only ones
# where remote sensing adds something geography does not give — and on `phylo` the model
# **doubles** the control.
#
# That inverts the expectation the project started from. The facets added last, almost as a
# complement, are the ones that justify the predictor.

# %% [markdown]
# ## 2. The three metrics, per facet
#
# `R2` says how much variance is explained; `%RMSE` how much error remains in units of the
# target's range; `|bias|` whether the error is centred. All three for the best model of each
# target.

# %%
b = best.copy()
b["%RMSE"] = 100 * b["nRMSE"]
b["bias %"] = 100 * b["nBias"]
b["facet"] = pd.Categorical(b["facet"], FACET_ORDER, ordered=True)
display(b.sort_values(["facet", "target"])[
    ["facet", "target", "run_id", "family", "index", "n", "R2", "R2_sd",
     "%RMSE", "bias %", "spearman"]].round(3).reset_index(drop=True))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, lab) in zip(axes, [("R2", "$R^2$"), ("%RMSE", "%RMSE"),
                                 ("bias %", "bias (% of range)")]):
    for i, f in enumerate(FACET_ORDER):
        d = b[b.facet == f]
        ax.scatter([i] * len(d), d[col], s=45, alpha=0.8,
                   color=[COLORS.get(x, "#8172b3") for x in d.family])
    ax.set_xticks(range(len(FACET_ORDER)))
    ax.set_xticklabels(FACET_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(lab)
    ax.grid(axis="y", alpha=0.25)
    if col == "bias %":
        ax.axhline(0, color="k", lw=0.8)
axes[0].legend([plt.Line2D([], [], marker="o", ls="", color=c) for c in COLORS.values()],
               COLORS.keys(), fontsize=7, frameon=False, ncol=2)
fig.suptitle("The best model for each target, on all three metrics", fontsize=10)
plt.tight_layout(); plt.show()

print("Bias stays under 2 % of range almost everywhere: the trimmed Duan estimator is doing\n"
      "its job. The exceptions are in beta_cover, the 546-plot stratum — half the data and\n"
      "twice the per-fold variance.")

# %% [markdown]
# ## 3. Model families
#
# The comparison that decides whether the deep-learning apparatus was worth it.

# %%
fam = (facet.pivot_table(index="family", columns="facet", values="R2", aggfunc="mean")
       .reindex(columns=FACET_ORDER)
       .reindex(["BASE", "RF", "MLP", "C1D", "C2D"]))
display(fam.round(3))

fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.16
for k, family in enumerate(fam.index):
    ax.bar(np.arange(len(FACET_ORDER)) + (k - 2) * w, fam.loc[family], width=w,
           color=COLORS[family], label=family)
ax.set_xticks(range(len(FACET_ORDER))); ax.set_xticklabels(FACET_ORDER)
ax.set_ylabel("mean facet $R^2$")
ax.set_title("Random Forest wins four of five facets; the CNNs win none", fontsize=10)
ax.legend(fontsize=8, frameon=False, ncol=5)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.show()

print("This matches the expectation in docs/11_next_steps.md §3: with 1,082 plots the\n"
      "architecture is not the lever. Screening result 4 already anticipated it — X14 with\n"
      "511 columns falls below X17 with 87.")

# %% [markdown]
# ## 4. Vegetation indices — and why the raw average lies
#
# The matrix design is **unbalanced**: kNDVI ran with 32 CNN configurations and the other
# indices with 3, because stage 4a screened the eleven substrates on kNDVI and stage 4b only
# expanded the best ones over the rest. Averaging carelessly charges kNDVI with the bad
# substrates the others never ran, and leaves it last.
#
# The correct comparison is **within each family**, and for C2D restricted to the substrates
# that did run on all five indices.

# %%
ix = facet[facet["index"].notna()].copy()
ABLATION = r"_(?:w[ACX]|calendar|nglobal|nperSample|film|patch|none|mixup|noaug|ctx)"
ix["is_ablation"] = ix.run_id.str.contains(ABLATION)

print("runs per index and family — the imbalance that has to be corrected:")
display(ix.pivot_table(index="index", columns="family", values="run_id", aggfunc="nunique"))

base = ix[~ix.is_ablation]
# within C2D, only the substrates present in all five indices
c2 = base[base.family == "C2D"]
shared = c2.groupby("substrate")["index"].nunique()
c2 = c2[c2.substrate.isin(shared[shared == 5].index)]
fair = pd.concat([base[base.family != "C2D"], c2])

tab = fair.pivot_table(index="index", columns="family", values="R2", aggfunc="mean")
tab["mean"] = tab.mean(axis=1)
display(tab.round(3).sort_values("mean", ascending=False))

fig, ax = plt.subplots(figsize=(8, 4))
t = tab.drop(columns="mean").loc[tab.sort_values("mean", ascending=False).index]
x = np.arange(len(t))
for k, family in enumerate(t.columns):
    ax.bar(x + (k - (len(t.columns) - 1) / 2) * 0.2, t[family], width=0.2,
           color=COLORS.get(family, "#8172b3"), label=family)
ax.set_xticks(x); ax.set_xticklabels(t.index)
ax.set_ylabel("mean $R^2$ over the five facets")
ax.set_title("Paired comparison of indices, within each family", fontsize=10)
ax.legend(fontsize=8, frameon=False, ncol=4)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.show()

spread = (tab.drop(columns="mean").max() - tab.drop(columns="mean").min()).round(3)
print("spread between the best and worst index, within each family:")
print(spread.to_string())
print(f"\nseed noise: {full.R2_sd.median():.3f}. For RF the spread across indices is the same\n"
      "order of magnitude: the index barely matters. kNDVI wins or ties in all four families.")

# %% [markdown]
# ## 5. Each facet in detail
#
# The ten best runs of each facet with the three metrics. `R2_min` and `R2_max` are across the
# targets **within** the facet: a facet with a wide spread is not well summarised by its mean.

# %%
for f in FACET_ORDER:
    d = facet[facet.facet == f].head(10).copy()
    d["%RMSE"] = 100 * d["nRMSE"]
    d["|bias|"] = 100 * d["nBias_abs"]
    print(f"\n===== {f}  ({', '.join(tg.FACETS[f])})")
    display(d[["run_id", "family", "index", "substrate", "R2", "R2_min", "R2_max",
               "%RMSE", "|bias|", "spearman"]].round(3).reset_index(drop=True))

# %% [markdown]
# ## 6. Target by target
#
# The facet mean hides that targets within a facet do not behave alike. The phylogenetic SES
# metrics, by construction, discount richness — and what is left is the hardest thing to see
# from phenology.

# %%
per_target = (full.groupby(["facet", "target"])
              .agg(best_R2=("R2", "max"), median_R2=("R2", "median"), n=("n", "first"))
              .reset_index())
per_target["facet"] = pd.Categorical(per_target["facet"], FACET_ORDER, ordered=True)
per_target = per_target.sort_values(["facet", "best_R2"], ascending=[True, False])

fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(per_target))[::-1]
cmap = dict(zip(FACET_ORDER, ["#4c72b0", "#2f6f7f", "#55a868", "#dd8452", "#c1553b"]))
ax.barh(y, per_target.best_R2, color=[cmap[f] for f in per_target.facet], alpha=0.85)
ax.barh(y, per_target.median_R2, color="k", alpha=0.25, height=0.35)
ax.set_yticks(y)
ax.set_yticklabels([f"{r.target}  (n={r.n})" for r in per_target.itertuples()], fontsize=8)
ax.set_xlabel("$R^2$ — light bar: best model; dark bar: median across runs")
ax.axvline(0, color="k", lw=0.8)
ax.set_title("The 15 targets, ordered within their facet", fontsize=10)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=cmap[f]) for f in FACET_ORDER],
          FACET_ORDER, fontsize=8, frameon=False, loc="lower right")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.show()

display(per_target.round(3).reset_index(drop=True))

# %% [markdown]
# ## 7. GDM — the alternative to the PCoA axes
#
# `scripts/21_run_gdm.py`. Instead of reducing composition to axes and predicting each plot's
# position, it models the **dissimilarity between every pair** as a function of environmental
# distance. The relevant comparison is the two *ceilings*: the rho a model would reach if it
# predicted perfectly the 2 axes the project models today, and the 8.

# %%
GDM = TABLES / "gdm_comparison.csv"
if GDM.exists():
    g = pd.read_csv(GDM)
    g = g[g.scheme == "kfold5_window"]
    if len(g):
        cols = {"geo_rho": "geography only", "gdm_rho": "GDM", "gdm_geo_rho": "GDM + geo",
                "sgdm_rho": "SGDM", "rf_axes_rho": "RF on 8 axes",
                "oracle2_rho": "ceiling: 2 true axes",
                "oracle_rho": "ceiling: 8 true axes"}
        tab = g.groupby("spec")[list(cols)].mean().rename(columns=cols)
        display(tab.round(3))

        fig, ax = plt.subplots(figsize=(9, 4.2))
        m = tab.mean().sort_values()
        bar_colors = ["#c1553b" if "ceiling" in k else "#2f6f7f" for k in m.index]
        ax.barh(range(len(m)), m.values, color=bar_colors)
        ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=8)
        ax.set_xlabel("Spearman rho against observed Jaccard, out-of-fold pairs")
        ax.set_title("GDM clears the ceiling of the PCoA-axis route", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout(); plt.show()

        print("The 2 axes the project models today carry 13.8 % of the compositional\n"
              "variance. Predicting them perfectly has a ceiling no architecture crosses;\n"
              "GDM clears it because it chases a better-defined objective. This is the open\n"
              "decision in docs/11_next_steps.md §4.")
    else:
        print("gdm_comparison.csv has no kfold5_window rows yet — the queue is still running")
else:
    print("does not exist yet — run scripts/run_window_queue.sh")

# %% [markdown]
# ## 8. What has to be said when reporting this
#
# - **The `alpha` R2 under `kfold5_window` is not the one under `kfold5_owner`.** It goes from
#   -0.02 to +0.49. The model did not improve: the scheme spreads each contributor across folds
#   and getting the local level right is free again. The number always travels with that
#   caveat attached.
# - **Dark diversity is the facet with the least contributor confounding** in the project
#   (R2 by `Owner` of 0.22 against 0.70 for log-richness, `docs/12` §8). Its R2 is more solid
#   than alpha's even though it is lower.
# - **`pd_faith` and `completeness` are absent on purpose.** They correlate +0.948 and +0.988
#   with richness; including them would have produced two "results" that are `hill_q0` under
#   another name. They sit in `tg.DROPPED` with the number that justifies it.
