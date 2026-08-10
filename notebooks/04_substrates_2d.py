#!/usr/bin/env python3
# ---
# Source for `notebooks/04_substrates_2d.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/04_substrates_2d.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # From phenological curve to image
#
# The project's 2-D CNNs do not see a time series: they see an image. This notebook shows
# **which image, exactly**, for each of the eleven transforms, on real curves from real plots.
#
# The question is not cosmetic. A signal-to-image transform is a bet about which temporal
# relationships deserve to end up **spatially adjacent**, because a 3x3 kernel only sees
# neighbours. `reshape` bets that consecutive weeks and weeks 8 apart matter; `gaf` bets on
# every pair of weeks; `cwt` bets on scale. If the bet does not match the ecology, the
# convolution mixes things that do not belong together and the model loses to an MLP reading
# the flat curve.
#
# And that is exactly the project's result: under `kfold5_window` the Random Forest beats C2D
# on all five facets. Looking at the images is how you understand why, and how you decide
# whether the problem is the transform or the sample size.
#
# Every panel carries its `pad_mode`, which is the other half of the bet: an axis representing
# cyclic time has to be padded by wrapping, not with zeros, or the kernel reads a collapse at
# the border that phenology never had.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from biodiv import features as feat          # noqa: E402
from biodiv import substrates as sub         # noqa: E402
from biodiv import transforms1d as t1        # noqa: E402

DERIVED = str(ROOT / "data" / "derived")


def by_calendar(doy, y):
    """Reorder (doy, y) into calendar order for plotting.

    `step_to_doy` is NOT monotonic: curves are trough-anchored, so step 0 is DOY 108 and the
    array jumps from 364 to 1 at step 36. Plotting against that axis without reordering makes
    matplotlib join the last December point to the first January one and draw a spurious
    straight line across the whole figure -- it looks like a trend and it is nothing.
    """
    o = np.argsort(doy)
    return np.asarray(doy)[o], np.asarray(y)[o]


def as_panels(img):
    """Return [(2d array, suffix)] ready for imshow.

    `gaf` and `cos2d` return (H, W, 2): two fields, not one. GAF gives the summation and the
    difference field; both enter the model as separate channels. `make_substrate` transposes
    them to (C, H, W) before stacking, so the model sees them correctly -- but the raw output
    of `transform()` carries the channel last and has to be split by hand.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        return [(a, "")]
    return [(a[..., c], f" ch{c}") for c in range(a.shape[-1])]


ids = feat.plot_ids(DERIVED)
doy = sub.step_to_doy(DERIVED)

print(f"{len(ids)} plots, {len(doy)}-step curves "
      f"(DOY {doy[0]:.0f} to {doy[-1]:.0f}, mean step {np.diff(doy).mean():.1f} days)")
print(f"registered transforms: {', '.join(t1.TRANSFORMS)}")

# %% [markdown]
# ## 1. The source curves
#
# `PhenoShape` fits a mean phenological year over three years of Landsat observations and
# interpolates it to 52 weekly steps. The curves come trough-anchored (DOY 108): the array
# starts where vegetation is at its minimum, not on 1 January.
#
# That anchor is **global** — the same constant for every plot. Anchoring each plot to its own
# peak would erase the phase differences between matorral that greens in winter and
# sclerophyll that greens in summer, and those differences are signal, not noise.

# %%
curves, cids = sub.load_curves("kndvi", DERIVED, ids=ids)
curves = curves[:, 0, :]        # (N, 52)

# three plots with deliberately different phenology, picked by curve amplitude
ampl = curves.max(axis=1) - curves.min(axis=1)
order = np.argsort(ampl)
EXAMPLES = {"low amplitude": order[len(order) // 20],
            "median amplitude": order[len(order) // 2],
            "high amplitude": order[-len(order) // 20]}

# Two views of the same data, and the contrast is the point: the model sees the left-hand
# order (trough-anchored) and the ecologist reads the right-hand one (calendar).
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for lab, i in EXAMPLES.items():
    axes[0].plot(np.arange(len(doy)), curves[i], lw=1.8, label=f"{lab} — {cids[i]}")
    axes[1].plot(*by_calendar(doy, curves[i]), lw=1.8)
axes[0].set_xlabel("step (weeks since the trough, DOY 108)")
axes[0].set_title("as the model sees it: trough-anchored", fontsize=9)
axes[0].legend(fontsize=8, frameon=False)
axes[1].set_xlabel("day of year")
axes[1].set_title("calendar order", fontsize=9)
axes[1].axvline(108, color="k", lw=0.8, ls=":")
axes[1].text(112, axes[1].get_ylim()[1] * 0.95, "DOY 108 = step 0", fontsize=7, va="top")
for ax in axes:
    ax.set_ylabel("kNDVI"); ax.grid(alpha=0.25)
fig.suptitle("Three 52-step phenological curves", fontsize=10)
plt.tight_layout(); plt.show()

print("curve amplitude (max - min):")
for lab, i in EXAMPLES.items():
    print(f"  {lab:18s} {ampl[i]:.3f}   mean {curves[i].mean():.3f}")

# %% [markdown]
# ## 1b. Found while plotting: the curve does not close the year
#
# Ordering by calendar surfaced something that was not documented. **The fitted curve has a
# systematic step at the year boundary** — between DOY 364 and DOY 1, which are two days apart.
#
# `PhenoShape` fits over DOY 1–365 without imposing periodicity, so the spline endpoints are
# free and need not agree. They do not: the jump is ~4x the typical weekly change, in all five
# indices, and it is **negative in 67–71 % of plots** — not symmetric noise, an edge bias of
# the fit.
#
# It matters because **five substrates pad `circular`** (`curve1d`, `curve5`, `gaf`, `mtf`,
# `ndi`) and section 4 shows `reshape` padding by wrapping. Wrapping assumes December joins
# January; in these data it does not. A kernel crossing that boundary reads a drop in
# vegetation that never happened.
#
# **This is a plausible — not demonstrated — hypothesis for why the CNNs do not win**: the
# Random Forest ignores column order and cannot be affected, while the whole convolutional
# family is built on a circularity the data violate. Testing it means refitting the curves
# with periodicity imposed and re-running, which has not been done.

# %%
INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
wrap_step = int(np.argmin(np.diff(doy)))          # the step where DOY jumps from 364 to 1

rows = []
for ix in INDICES:
    ci, _ = sub.load_curves(ix, DERIVED, ids=ids)
    ci = ci[:, 0, :]
    jump = ci[:, wrap_step + 1] - ci[:, wrap_step]
    typical = np.median(np.abs(np.diff(ci, axis=1)))
    rows.append(dict(index=ix, mean_jump=jump.mean(), median_jump=np.median(jump),
                     typical_weekly_change=typical,
                     ratio=np.median(np.abs(jump)) / typical,
                     pct_negative=100 * (jump < 0).mean(),
                     pct_of_amplitude=100 * np.median(np.abs(jump))
                     / np.median(ci.max(1) - ci.min(1))))
disc = pd.DataFrame(rows)
print(f"The array jumps from DOY {doy[wrap_step]:.0f} to {doy[wrap_step+1]:.0f} at step "
      f"{wrap_step}.\n`ratio` is the median jump over the typical weekly change:\n")
display(disc.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
step_abs = np.abs(np.diff(curves, axis=1)).mean(axis=0)
axes[0].bar(np.arange(len(step_abs)), step_abs, color="#2f6f7f")
axes[0].bar([wrap_step], [step_abs[wrap_step]], color="#c1553b")
axes[0].axhline(np.median(step_abs), color="k", lw=0.8, ls=":", label="median")
axes[0].set_xlabel("step transition"); axes[0].set_ylabel("mean |delta kNDVI|")
axes[0].set_title(f"transition {wrap_step}->{wrap_step+1} stands out against the other 50",
                  fontsize=9)
axes[0].legend(fontsize=7, frameon=False)

jump = curves[:, wrap_step + 1] - curves[:, wrap_step]
axes[1].hist(jump, bins=60, color="#2f6f7f")
axes[1].axvline(0, color="k", lw=1)
axes[1].set_xlabel("year-boundary jump (kNDVI)")
axes[1].set_ylabel("plots")
axes[1].set_title(f"{100*(jump<0).mean():.0f} % are negative — bias, not noise", fontsize=9)
for ax in axes:
    ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. The eleven transforms, on the same curve
#
# All of them receive the same 52-step vector. What changes is what ends up next to what.
#
# | | what it does | what becomes adjacent |
# |---|---|---|
# | `reshape` | pads to 64 and folds into 8x8, row by row | consecutive weeks, and weeks 8 apart |
# | `serpentine` | same, alternating the direction of each row | idem, without the jump at row ends |
# | `hilbert` | walks the curve along a Hilbert curve | temporal neighbours, with fewer jumps than reshape |
# | `gaf` | Gramian angular field: cosine of the angle sum of each pair | every pair of weeks (i, j) |
# | `mtf` | Markov transition field between quantiles | the probability of moving between levels |
# | `ndi` | normalised difference between each pair of weeks | relative contrast (i - j)/(i + j) |
# | `cwt` | continuous wavelet transform | scale x time |
# | `cos2d` | outer product of the cosine of the signal | modulation between positions |
# | `spectrogram` | STFT over sliding windows | frequency x time |
#
# The first three are **foldings**: they reorder the 52 values without inventing anything,
# which is why `unfold_to_doy` inverts them exactly. The rest are **expansions**: they produce
# a 52x52 matrix (or similar) out of 52 numbers, so they add no information — they only
# present it so a local convolution can read relationships that in the flat vector would be 40
# positions apart.

# %%
IDX = EXAMPLES["median amplitude"]
x = curves[IDX]

panels = [("original curve", None, "")]
for name in t1.TRANSFORMS:
    try:
        panels += [(name, m, suf) for m, suf in as_panels(t1.make_transform(name).transform(x))]
    except Exception as e:                      # cwt needs PyWavelets
        panels.append((name, e, ""))

ncol = 4
nrow = -(-len(panels) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow))
axes = axes.ravel()
for ax, (name, arr, suf) in zip(axes, panels):
    if arr is None:
        ax.plot(np.arange(len(x)), x, lw=1.8, color="#2f6f7f")
        ax.set_title("original curve (52 steps)", fontsize=9)
        ax.set_xlabel("step", fontsize=7)   # step, not DOY: the DOY axis is not monotonic
        ax.grid(alpha=0.25); ax.tick_params(labelsize=6)
        continue
    if isinstance(arr, Exception):
        ax.text(0.5, 0.5, f"{name}\nunavailable:\n{type(arr).__name__}",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    im = ax.imshow(arr, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(f"{name}{suf}  {arr.shape}\npad={sub.PAD_MODE.get(name, '?')}", fontsize=8)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)
for ax in axes[len(panels):]:
    ax.axis("off")
fig.suptitle(f"The same 52 weeks, eleven ways of looking at them — plot {cids[IDX]}",
             fontsize=11)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. The foldings are invertible — and that is what makes attribution readable
#
# `reshape`, `serpentine` and `hilbert` lose nothing: `unfold_to_doy` recovers the exact curve.
# It matters because that is what turns "the model looks at this pixel" into "the model looks
# at this week", which is the only form in which an attribution map reads as ecology.
#
# Expansions are not invertible and `unfold_to_doy` uses the row marginal: approximate, but
# enough to locate the time of year.

# %%
rows = []
for name in t1.TRANSFORMS:
    try:
        img = np.asarray(t1.make_transform(name).transform(x))
        # unfold_to_doy expects (C, H, W), which is how make_substrate stacks; the raw output
        # carries the channel last
        arg = np.transpose(img, (2, 0, 1)) if img.ndim == 3 else img
        back = sub.unfold_to_doy(arg, name)
        err = float(np.max(np.abs(back - x))) if back.shape == x.shape else np.nan
        rows.append(dict(substrate=name, shape=str(img.shape),
                         channels=1 if img.ndim == 2 else img.shape[-1],
                         pixels=int(np.prod(img.shape)), max_inversion_error=err,
                         invertible="exact" if err < 1e-9 else "approximate"))
    except Exception as e:
        rows.append(dict(substrate=name, shape="—", channels=np.nan, pixels=np.nan,
                         max_inversion_error=np.nan,
                         invertible=f"unavailable ({type(e).__name__})"))
inv = pd.DataFrame(rows)
print("52 input values. `pixels` is how many numbers each transform expands them into:\n")
display(inv)

# %% [markdown]
# ## 4. Padding is not a detail
#
# `reshape` needs 64 values to fill an 8x8 and there are only 52. Padding with zeros invents a
# vegetation collapse at the end of the year that the kernel reads as a real event. The
# project pads by **wrapping** (`wrap_pad`): the 12 missing values are the first 12 weeks
# again, which is what phenology actually does — December joins January.
#
# Read together with section 1b this is a caveat, not a contradiction: wrapping is the right
# choice given the alternative, but the wrap itself is not seamless in these data.

# %%
flat_wrap = t1.ReshapeTransform().transform(x).reshape(-1)
flat_zero = np.concatenate([x, np.zeros(64 - len(x))])

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].plot(flat_wrap, lw=1.5, color="#2f6f7f", label="wrap (what is used)")
axes[0].plot(flat_zero, lw=1.5, color="#c1553b", ls="--", label="zeros")
axes[0].axvline(51.5, color="k", lw=0.8, ls=":")
axes[0].set_title("the 64 values that fill the 8x8", fontsize=9)
axes[0].set_xlabel("position")
axes[0].legend(fontsize=7, frameon=False)
axes[0].grid(alpha=0.25)
for ax, arr, lab in zip(axes[1:], [flat_wrap, flat_zero], ["wrap", "zeros"]):
    im = ax.imshow(arr.reshape(8, 8), cmap="viridis", interpolation="nearest")
    ax.axhline(6.5, color="w", lw=1.2, ls=":")
    ax.set_title(f"padded with {lab}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("The last row and a half is padding; with zeros it is a winter that never existed",
             fontsize=10)
plt.tight_layout(); plt.show()

print(f"minimum of the wrapped image: {flat_wrap.min():.3f}   (the curve's real minimum)")
print(f"minimum of the zero image   : {flat_zero.min():.3f}   (invented)")

# %% [markdown]
# ## 5. What the transform has to preserve: the differences between plots
#
# A transform that normalises each sample by its own range produces pretty images and deletes
# the signal: mean greenness and seasonal amplitude are **exactly** what separates matorral
# from sclerophyll. That is why `normalize="none"` is the project default, and why the three
# curves from section 1 have to still look different after transforming.

# %%
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for col, (lab, i) in enumerate(EXAMPLES.items()):
    for row, norm in enumerate(["none", "perSample"]):
        img = t1.make_transform("reshape", normalize=norm).transform(curves[i])
        im = axes[row, col].imshow(img, cmap="viridis", interpolation="nearest",
                                   vmin=0 if norm == "none" else None,
                                   vmax=curves.max() if norm == "none" else None)
        axes[row, col].set_title(f"{lab}\nnormalize={norm}", fontsize=8)
        axes[row, col].tick_params(labelsize=6)
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
fig.suptitle("Top: the three plots are distinguishable. Bottom, normalised: nearly the same "
             "image", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. The substrates that are not transforms of one curve
#
# Three registry entries do not come from folding a 1-D signal. They exploit the fact that
# there are **five indices** and **25 pixels** per plot, instead of inventing a second
# dimension.
#
# - **`stack5`** — 5 indices x 52 weeks. The vertical axis is real: a kernel crossing it reads
#   relationships between indices within the same week. It is the honest alternative to `gaf`
#   and friends.
# - **`curve5`** — the same 5 indices, but as *channels* of a 1-D signal. Paired with `stack5`
#   it isolates exactly what making the index axis a spatial dimension buys.
# - **`pxcube`** — 25 pixels x 52 weeks, the 150 m window without averaging.
#
# Their `pad_mode` is **mixed**: zeros on the vertical axis (indices and pixels are not
# cyclic) and circular on the horizontal (weeks are). That is why `SepConv2d` accepts a
# different padding per axis.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, name in zip(axes, ["stack5", "curve5", "pxcube"]):
    s = sub.make_substrate(name, index="kndvi" if name == "pxcube" else None,
                           derived=DERIVED, ids=ids[:64])
    j = min(IDX, s.X.shape[0] - 1)
    arr = s.X[j]
    arr = arr[0] if arr.ndim == 3 else arr
    im = ax.imshow(arr, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(f"{name}  {tuple(s.shape)}\npad={s.pad_mode}", fontsize=9)
    ax.set_xlabel("week", fontsize=7)
    ax.set_ylabel("index" if name != "pxcube" else "pixel", fontsize=7)
    if s.rows and len(s.rows) <= 6:
        ax.set_yticks(range(len(s.rows)))
        ax.set_yticklabels(s.rows, fontsize=6)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("The three substrates with a real second dimension", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. And with all that, which one wins?
#
# The empirical answer, under `kfold5_window`, averaged over the five facets. Read it with the
# usual caveat: seed noise is 0.008–0.017, so differences smaller than that are ties.

# %%
FULL = ROOT / "results" / "tables" / "results_by_facet.csv"
if FULL.exists():
    f = pd.read_csv(FULL)
    conv = f[f.family.isin(["C1D", "C2D"]) & f.substrate.notna()]
    # base configuration only: the 4c ablations (width, fusion, mixup...) are deliberately
    # degraded variants and contaminate any per-substrate average
    base = conv[~conv.run_id.str.contains(
        r"_(?:w[ACX]|calendar|nglobal|nperSample|film|patch|none|mixup|noaug|ctx)")]
    tab = (base.pivot_table(index="substrate", columns="facet", values="R2", aggfunc="mean")
           .assign(mean=lambda d: d.mean(axis=1))
           .sort_values("mean", ascending=False))
    print("Mean R2 per substrate, base configuration:\n")
    display(tab.round(3))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    d = tab["mean"].sort_values()
    ax.barh(range(len(d)), d.values, color="#2f6f7f")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel("mean $R^2$ over the five facets")
    ax.set_title("No 2-D folding clearly beats reading the curve as a 1-D signal", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout(); plt.show()
else:
    print("tables do not exist yet — run scripts/28_results_tables.py")
