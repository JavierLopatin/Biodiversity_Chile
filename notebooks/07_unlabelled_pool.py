# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # The unlabelled pool: where it is sampled, and why there
#
# The masked autoencoder was tried once and **gained nothing**: 0.359 against 0.362
# unpretrained (`docs/14` §4). The method worked — reconstruction MSE fell from 0.0095 to
# 0.0031 — so what failed was the data. This notebook shows what replaces it, and shows the
# measurements the replacement was designed against rather than asserting them.
#
# Read in this order, because each panel answers the objection raised by the one before:
#
# 1. **Where the plots are** — and why sampling "Chile" would not resemble them.
# 2. **Where the new pool is** — the map, both layers together.
# 3. **What it cost to get there** — the coverage the old design could not reach.
# 4. **Whether it is the same *kind* of vegetation** — class mix against the plots.
# 5. **Whether it is the same *era*** — year distribution against the plots.
# 6. **Whether it leaks** — distance from every sample to the nearest plot.
# 7. **What came back** — the kNDVI series themselves.
# 8. **Whether it was worth it** — the coverage gate, which is closer than it looks.
#
# Nothing is recomputed from the datacube here. The extraction ran against Data Cube Chile
# on 2026-08-11: 2,898 loads over 16 dask-gateway workers, 5.2 h, **16,950 of 16,950 series
# with zero failed loads**.

# %%
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mb

# Two series throughout — the labelled plots and the unlabelled pool — so the palette is two
# categorical hues, held fixed across every panel. Colour follows the entity, never the rank.
# Validated: CVD separation dE 20.0 (protan) / 31.5 (tritan), normal-vision 28.3, both >= 3:1
# against the surface.
C_PLOT, C_POOL = "#d95f02", "#1a8fb3"
INK, MUTED, GRID = "#1f2328", "#5b6570", "#dfe3e8"

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

UP = ROOT / "data/derived/unlabelled"
plots = pd.read_parquet(ROOT / "data/derived/plots_subset.parquet")

# the manifest is the extracted pool: one row per series that actually came back
pool = pd.read_csv(UP / "manifest.csv")
if "error" in pool:
    failed = pool[pool.error.notna()]
    pool = pool[pool.error.isna()].reset_index(drop=True)
    print(f"failed loads   : {len(failed)}")
series = pd.read_parquet(UP / "series.parquet")

print(f"labelled plots : {len(plots):,}")
print(f"unlabelled pool: {len(pool):,}   in {pool.cell.nunique():,} cells of 10 km")
print(f"datacube loads : {pool.groupby(['cell','year']).ngroups:,}  (cell x year)")
print(f"kNDVI series   : {len(series):,} observations, "
      f"{series.memory_usage(deep=True).sum()/1e6:.0f} MB in memory")

# %% [markdown]
# ## 1. The map
#
# Both layers, one panel. The pool is drawn first and small; the plots sit on top, because
# the question the map has to answer is *does the pool surround the plots* — not the reverse.
#
# The study region is 280 km wide and 850 km tall, so the honest aspect ratio is a tall
# strip. Squashing it to a square would make the northern sampling look denser than it is.

# %%
fig = plt.figure(figsize=(9.2, 7.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.5], height_ratios=[1, 1],
                      wspace=0.30, hspace=0.26)
ax = fig.add_subplot(gs[:, 2])

ax.scatter(pool.lon, pool.lat, s=1.1, c=C_POOL, alpha=0.30, lw=0,
           label=f"MAE pool  (n = {len(pool):,})", rasterized=True)
ax.scatter(plots.lon, plots.lat, s=6, c=C_PLOT, alpha=0.85,
           lw=0.3, edgecolor="white", label=f"Parcelas-CL  (n = {len(plots):,})")

ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Native-vegetation sampling, whole study region", loc="left")
ax.set_aspect(1 / np.cos(np.deg2rad(34)))
ax.grid(True, lw=0.4, alpha=0.5)
leg = ax.legend(loc="lower left", markerscale=3.2, fontsize=7.5,
                handletextpad=0.4, borderpad=0.4)
for h in leg.legend_handles:
    h.set_alpha(1.0)

# Two zooms, chosen to show the two things the full map cannot resolve: that pool points
# stand off from the plots, and that they keep >= 1 km from each other.
zooms = [("Coastal range, ~33.0 S", -71.60, -71.15, -33.20, -32.85),
         ("Andean foothills, ~35.5 S", -71.30, -70.85, -35.65, -35.30)]
for k, (name, w, e, s, n) in enumerate(zooms):
    az = fig.add_subplot(gs[k, 0:2])
    mp = pool[(pool.lon.between(w, e)) & (pool.lat.between(s, n))]
    mq = plots[(plots.lon.between(w, e)) & (plots.lat.between(s, n))]
    az.scatter(mp.lon, mp.lat, s=15, c=C_POOL, alpha=0.75, lw=0)
    az.scatter(mq.lon, mq.lat, s=34, c=C_PLOT, marker="^", lw=0.4, edgecolor="white")
    az.set_title(f"{name}   ({len(mp)} pool, {len(mq)} plots)", loc="left", fontsize=8)
    az.set_aspect(1 / np.cos(np.deg2rad(abs((s + n) / 2))))
    az.grid(True, lw=0.4, alpha=0.5)
    az.tick_params(labelsize=6.5)

fig.suptitle("Where the unlabelled pool is drawn", x=0.005, ha="left",
             fontsize=11, weight="bold")
plt.show()

# %% [markdown]
# ## 2. What the old design could not reach
#
# The previous sampler jittered centres ±5 km around each plot. That is not a small
# difference of taste — it caps how much of the country the pool can ever see, and no
# increase in `n` lifts the cap. It is the same failure that already cost one attempt: the
# 135,250 pixel curves came from the 25-pixel windows of the same plots and correlate at
# **+0.844** within a plot, against **+0.258** between plots.

# %%
reach_jitter = 19_583      # km2, union of the +-5 km boxes around the 1,082 plots
reach_native = 83_153      # km2, native under the causal-window rule (2016-2018)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.4, 2.9),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})

bars = [("jitter ±5 km\n(previous design)", reach_jitter, "#b9c2cc"),
        ("MapBiomas native mask\n(this design)", reach_native, C_POOL)]
for i, (lab, v, c) in enumerate(bars):
    a0.barh(i, v, height=0.5, color=c)
    a0.text(v + 1800, i, f"{v:,} km²", va="center", fontsize=8, color=INK)
a0.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=7.5)
a0.set_xlim(0, reach_native * 1.28)
a0.set_xlabel("reachable native area")
a0.set_title(f"Reach: {reach_jitter/reach_native:.0%} → 100 %", loc="left")
a0.invert_yaxis(); a0.grid(True, axis="x", lw=0.4, alpha=0.5)

# Samples stacked per seed plot under the old design -- the redundancy, made literal.
ns = np.array([4_000, 20_000, 50_000])
a1.plot(ns, ns / len(plots), "o-", color="#b9c2cc", lw=2, ms=7, label="jitter ±5 km")
a1.axhline(1.0, color=C_POOL, lw=2, label="native mask (independent sites)")
for n, v in zip(ns, ns / len(plots)):
    a1.annotate(f"{v:.0f}×", (n, v), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5, color=INK)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("samples drawn"); a1.set_ylabel("samples per seed plot")
a1.set_title("Raising n only stacks them", loc="left")
a1.legend(fontsize=7); a1.grid(True, lw=0.4, alpha=0.5)
plt.show()

print(f"reach   {reach_jitter:,} -> {reach_native:,} km2   ({reach_native/reach_jitter:.1f}x)")

# %% [markdown]
# ## 3. Is it the same kind of vegetation?
#
# A pool of native cover is not automatically a pool of *the plots'* native cover. The
# sampler draws a per-class quota inside every cell, taken from the plots' own class shares,
# so the pool is shrubland-dominated because the labelled set is.
#
# **This is bounded by what the landscape holds.** Wetland is over-represented and forest
# under-represented relative to the plots: a cell with little shrubland fills its shortfall
# with whatever is there. That is the honest outcome and it is left visible rather than
# forced — the alternative would be discarding cells until the histogram matched, which
# would undo the coverage this design exists to buy.

# %%
codes, _, _ = mb.sample_at(plots.lon.to_numpy(), plots.lat.to_numpy(), 2014)
pl_cls = pd.Series([mb.class_name(int(c)) for c in codes])
pl_nat = pl_cls[mb.is_native(codes)].value_counts(normalize=True)
po_nat = pool.mb_class.value_counts(normalize=True)

order = po_nat.index.union(pl_nat.index, sort=False)
order = po_nat.reindex(order).fillna(0).sort_values().index
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(7.4, 3.3))
ax.barh(y + 0.20, [100 * pl_nat.get(c, 0) for c in order], height=0.36,
        color=C_PLOT, label=f"Parcelas-CL (n = {mb.is_native(codes).sum():,})")
ax.barh(y - 0.20, [100 * po_nat.get(c, 0) for c in order], height=0.36,
        color=C_POOL, label=f"MAE pool (n = {len(pool):,})")
for i, c in enumerate(order):                       # direct labels, not a number per bar
    for off, s, col in ((0.20, pl_nat.get(c, 0), C_PLOT), (-0.20, po_nat.get(c, 0), C_POOL)):
        if s > 0.03:
            ax.text(100 * s + 0.8, i + off, f"{100*s:.0f}", va="center",
                    fontsize=7, color=MUTED)
ax.set_yticks(y, order, fontsize=7.5)
ax.set_xlabel("share of samples (%)")
ax.set_title("Native class mix: pool against plots", loc="left")
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(True, axis="x", lw=0.4, alpha=0.5)
plt.show()

pd.DataFrame({"plots %": (100 * pl_nat).round(1),
              "pool %": (100 * po_nat).round(1)}).fillna(0).sort_values(
                  "pool %", ascending=False)

# %% [markdown]
# ## 4. Is it the same era?
#
# Each sample carries a causal 3-year window `y−2..y`, and the years are drawn from the
# plots' own histogram. If they were drawn uniformly instead, the pretraining pool would sit
# in epochs the labelled set does not contain, and any transfer result would be confounded
# with a shift of era rather than isolating pretraining.

# %%
yr_pl = plots.Year.value_counts(normalize=True).sort_index()
yr_po = pool.year.value_counts(normalize=True).sort_index()
years = sorted(set(yr_pl.index) | set(yr_po.index))
x = np.arange(len(years))

fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.2, 2.9),
                             gridspec_kw={"width_ratios": [2, 1], "wspace": 0.28})
a0.bar(x + 0.20, [100 * yr_pl.get(y, 0) for y in years], width=0.36,
       color=C_PLOT, label="Parcelas-CL")
a0.bar(x - 0.20, [100 * yr_po.get(y, 0) for y in years], width=0.36,
       color=C_POOL, label="MAE pool")
a0.set_xticks(x, [str(y) for y in years], rotation=60, fontsize=6.5)
a0.set_ylabel("share (%)"); a0.set_title("Census year", loc="left")
a0.legend(fontsize=7.5); a0.grid(True, axis="y", lw=0.4, alpha=0.5)

# the residual, which is what actually matters -- a mirrored histogram should hug zero
d = np.array([100 * (yr_po.get(y, 0) - yr_pl.get(y, 0)) for y in years])
a1.axhline(0, color=MUTED, lw=0.9)
a1.bar(x, d, width=0.6, color=np.where(d >= 0, C_POOL, C_PLOT))
a1.set_xticks(x, [str(y) for y in years], rotation=60, fontsize=6.5)
a1.set_ylabel("pool − plots (pp)")
a1.set_title(f"Residual (max |Δ| = {np.abs(d).max():.1f} pp)", loc="left")
a1.grid(True, axis="y", lw=0.4, alpha=0.5)
plt.show()

n_disp = int((pool.mb_delta != 0).sum())
print(f"MapBiomas maps available : {min(mb.available_years())}-{max(mb.available_years())} "
      f"({len(mb.available_years())} years)")
print(f"samples on a displaced map: {n_disp:,} ({100*n_disp/len(pool):.1f} %), "
      f"all from {sorted(pool.loc[pool.mb_delta != 0, 'year'].unique())}, "
      f"mapped to {sorted(pool.loc[pool.mb_delta != 0, 'mb_year_used'].unique())}")

# %% [markdown]
# ## 5. Does it leak?
#
# Pretraining sees no targets, so no response can leak. The input side is the real question:
# the 135,250 pixel curves **were** the test plots' own pixels, which is standard in
# self-supervised learning but is not nothing.
#
# This pool is cleaner by construction. Every sample is a fresh site, and the minimum
# spacing is enforced in projected metres. The plot below is the check, not the claim.

# %%
from scipy.spatial import cKDTree

tree = cKDTree(plots[["X", "Y"]].to_numpy())
d_plot, _ = tree.query(pool[["X", "Y"]].to_numpy())

sep = []
for _, g in pool.groupby(["cell", "year"]):
    if len(g) < 2:
        continue
    p = g[["X", "Y"]].to_numpy()
    dd = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(dd, np.inf)
    sep.append(dd.min(axis=1))
sep = np.concatenate(sep)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.6, 2.8), gridspec_kw={"wspace": 0.30})
a0.hist(d_plot / 1000, bins=60, color=C_POOL, lw=0)
a0.axvline(0.075, color=C_PLOT, lw=1.6, ls="--")
a0.text(0.085, a0.get_ylim()[1] * 0.92, "plot footprint (75 m)", fontsize=7, color=C_PLOT)
a0.set_xlabel("distance to nearest plot (km)"); a0.set_ylabel("samples")
a0.set_title(f"Closest sample: {d_plot.min()/1000:.2f} km", loc="left")
a0.grid(True, lw=0.4, alpha=0.5)

a1.hist(sep / 1000, bins=60, color=C_POOL, lw=0)
a1.axvline(1.0, color=C_PLOT, lw=1.6, ls="--")
a1.text(1.05, a1.get_ylim()[1] * 0.92, "1 km minimum", fontsize=7, color=C_PLOT)
a1.set_xlabel("distance to nearest other sample (km)")
a1.set_title(f"Closest pair: {sep.min()/1000:.3f} km", loc="left")
a1.grid(True, lw=0.4, alpha=0.5)
plt.show()

print(f"samples inside a plot footprint (<75 m): {(d_plot < 75).sum()}")
print(f"pairs closer than 1 km                 : {(sep < 999).sum()}")

# %% [markdown]
# ## 6. What came back
#
# **Observations with their dates**, not a fitted curve and not a fixed grid. That is the rule
# of `docs/05` §1: acquire at observation level, decide the pooling downstream. It is why
# `docs/14` §2 could test eight step resolutions (24 to 196, spread 0.011 — below seed noise)
# without a second extraction, and why the grid below is built here rather than stored.
#
# ⚠️ **One asymmetry that was not designed and has to be declared.** The pool's series are
# *denser* than the plots': the plots average ~65 clear observations over their 3-year window,
# the pool ~96. Nothing about the extraction differs — same sensors, same QA mask, same
# window. The pool simply sits in latitudes and years with better Landsat availability than
# the plot sample happens to occupy. It does not invalidate the transfer, because both go
# through the same step grid, but pretraining and fine-tuning are seeing curves interpolated
# from different amounts of real data, and that belongs in the reporting.

# %%
from biodiv import curves

obs = pool.set_index("sample_id").n_obs
plot_obs_ref = 65                                        # measured on the labelled plots

fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.8, 2.9),
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
a0.hist(obs, bins=60, color=C_POOL, lw=0)
a0.axvline(plot_obs_ref, color=C_PLOT, lw=1.8, ls="--")
# above the bars, not across them: the mode sits right of the reference line
a0.annotate(f"plots ≈ {plot_obs_ref}", (plot_obs_ref, a0.get_ylim()[1] * 1.03),
            xytext=(-4, 0), textcoords="offset points", fontsize=7.5, color=C_PLOT,
            ha="right", annotation_clip=False)
a0.set_xlabel("clear observations per series"); a0.set_ylabel("samples")
a0.set_title(f"Pool median {obs.median():.0f}, denser than the plots", loc="left")
a0.grid(True, lw=0.4, alpha=0.5)

# 200 real curves on the 52-step grid the model will actually see
rng = np.random.default_rng(0)
sids = rng.choice(pool.sample_id.to_numpy(), 200, replace=False)
sub = series[series.sample_id.isin(set(sids))].copy()
sub["t"] = sub["time"].values.astype("datetime64[D]").astype(float)
win = pool.set_index("sample_id")[["win_start", "win_end"]]
grid = []
for sid, g in sub.groupby("sample_id", sort=False):
    w = win.loc[sid]
    lo = np.datetime64(f"{int(w.win_start)}-01-01", "D").astype(float)
    hi = np.datetime64(f"{int(w.win_end)}-12-31", "D").astype(float)
    c = curves.interp_grid(g["t"].to_numpy(), g["kndvi"].to_numpy(), 52, t_min=lo, t_max=hi)
    if np.isfinite(c).all():
        grid.append(c)
grid = np.asarray(grid)

for c in grid:
    a1.plot(c, color=C_POOL, lw=0.5, alpha=0.10)
a1.plot(grid.mean(0), color=C_POOL, lw=2.4, label=f"pool mean (n = {len(grid)})")
a1.fill_between(np.arange(52), np.percentile(grid, 10, axis=0),
                np.percentile(grid, 90, axis=0), color=C_POOL, alpha=0.18, lw=0)
a1.set_xlabel("step over the 3-year causal window"); a1.set_ylabel("kNDVI")
a1.set_title("200 real series on the 52-step grid", loc="left")
a1.legend(fontsize=7.5); a1.grid(True, lw=0.4, alpha=0.5)
plt.show()

per_load = pool.groupby(["cell", "year"]).size()
pd.DataFrame([
    ("series extracted", f"{len(pool):,} of 16,950  (0 failed loads)"),
    ("10 km cells visited", f"{pool.cell.nunique():,}"),
    ("datacube loads (cell x year)", f"{per_load.size:,}"),
    ("observations, total", f"{len(series):,}"),
    ("observations per series", f"median {obs.median():.0f}, p10 {obs.quantile(.1):.0f}, "
                                f"max {obs.max()}"),
    ("kNDVI range", f"{series.kndvi.min():.4f} – {series.kndvi.max():.4f}  "
                    f"({series.kndvi.isna().sum()} nulls)"),
    ("size on disk", "10 MB"),
    ("index stored", "kNDVI only"),
    ("causal window", "y-2..y  (3 years)"),
], columns=["", "value"]).set_index("")

# %% [markdown]
# ## 7. Was it worth it? The gate, and how close it is
#
# The previous attempt had no gate, and that is why it burned a full pretraining run to learn
# something a cheap measurement says in seconds. The **participation ratio** — how many
# directions the curves actually occupy — is that measurement
# (`scripts/34_participation_dimension.py`).

# %%
import subprocess

pr = {"MapBiomas pool\n(new)": 1.48, "pixel curves\n(the ones that failed)": 1.28,
      "plot curves": 1.24}
fig, ax = plt.subplots(figsize=(6.4, 2.5))
cols = [C_POOL, "#b9c2cc", C_PLOT]
for i, ((k, v), c) in enumerate(zip(pr.items(), cols)):
    ax.barh(i, v, height=0.5, color=c)
    ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
ax.axvline(1.28, color=MUTED, lw=1.2, ls="--")
ax.set_yticks(range(3), list(pr), fontsize=7.5)
ax.set_xlim(1.0, 1.62); ax.set_xlabel("participation dimension")
ax.set_title("Coverage gained: +16 %, not a transformation", loc="left")
ax.invert_yaxis(); ax.grid(True, axis="x", lw=0.4, alpha=0.5)
plt.show()

print(subprocess.run([sys.executable, "scripts/34_participation_dimension.py"],
                     cwd=ROOT, capture_output=True, text=True).stdout)

# %% [markdown]
# ### Read the margin, not the verdict
#
# The gate passes at **1.48 against 1.28 — a factor of 1.16**. Two things about that number
# deserve to survive into whatever gets written next:
#
# 1. **The 1.15× threshold is not the project's.** It was chosen when the script was written
#    and nothing validates it. The pass is real but it is a pass against an invented bar.
# 2. **The measurement is resolution-bound.** The reference curves are stored fitted at 52
#    steps and cannot be re-gridded from the parquet, so a comparison at any other step count
#    pits the pool against a frozen reference and the verdict flips — 1.24 at 24 steps, 1.71
#    at 156. The script now refuses to give a verdict outside 52 rather than produce a
#    convenient one.
#
# So: the pool went from reaching 29 % of the native territory to 100 %, and bought a 16 %
# gain in occupied directions. That is a real improvement and a modest one. **It does not
# guarantee the autoencoder will transfer** — it says the experiment is now worth running,
# which the previous one, at 1.28, was not.
#
# When the ablations run, the GPU noise floor in this project is 0.010 of standard deviation
# (`docs/14` §3b), so **nothing below 0.022 distinguishes two models**. Each ablation needs
# three seeds or it is not interpretable.
#
# ```bash
# python scripts/31_pretrain_mae.py --unlabelled data/derived/unlabelled
# python scripts/31_pretrain_mae.py --unlabelled data/derived/unlabelled \
#     --landcover 'Shrubland,Secondary Forest,Primary Forest'
# ```
