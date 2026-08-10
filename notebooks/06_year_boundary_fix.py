#!/usr/bin/env python3
# ---
# Source for `notebooks/06_year_boundary_fix.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/06_year_boundary_fix.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # The phenological year did not close — diagnosis and fix
#
# A phenological curve describes one cycle, so its last step is adjacent to its first.
# Nothing in `PhenoShape` enforced that, and the two ends drifted apart by ~4x the typical
# week-to-week change, systematically in one direction.
#
# This notebook is the record of the whole thing: the measurement on 1,082 plots, the two
# causes located in the library code, the fix, and its verification on real Landsat cubes.
#
# | | |
# |---|---|
# | full write-up | `docs/13_phenology_year_boundary.md` |
# | upstream fix | `PhenoSensing` commit `eb2dff8`, `tests/test_periodicity.py` (19 tests) |
# | how it was found | `notebooks/04_substrates_2d.ipynb` §1b |
#
# **Why it is not cosmetic.** Five of the project's substrates pad `circular` on the time
# axis and `reshape`/`serpentine`/`hilbert` pad by wrapping. All of them assume December
# joins January. A kernel crossing that boundary reads a vegetation collapse that never
# happened — which is a plausible, and still undemonstrated, reason the CNNs never beat the
# Random Forest.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
# phenosensing is used from its working tree, not installed: the fix lives there and this
# notebook has to exercise the fixed code, not whatever is on the path
PHENO = Path("/mnt/rapidita_4T/GitHub/PhenoSensing")
if PHENO.exists():
    sys.path.insert(0, str(PHENO))

import phenosensing  # noqa: F401,E402  (registers the .pheno accessor)
from phenosensing import _numba                       # noqa: E402
from phenosensing.reconstruction import get_reconstructor, list_reconstructors  # noqa: E402
from phenosensing.utils import _moving_average        # noqa: E402

from biodiv import features as feat                   # noqa: E402
from biodiv import substrates as sub                  # noqa: E402

DERIVED = str(ROOT / "data" / "derived")
CUBES = ROOT / "data" / "derived" / "phenology"

print("phenosensing from:", Path(phenosensing.__file__).parent)
print("reconstructors   :", ", ".join(list_reconstructors()))
print("numba active     :", _numba.NUMBA_AVAILABLE)
print("cubes available  :", len(list(CUBES.glob("*.nc"))))

# %% [markdown]
# ## 1. The symptom, on all 1,082 plots
#
# `step_to_doy` is not monotonic: the stored curves are trough-anchored at DOY 108, so the
# array runs 108 → 364, then wraps to 1 → 100. The year boundary sits at **step 36**, in the
# middle of the array.
#
# Measuring the jump at that step against the other 50 transitions:

# %%
ids = feat.plot_ids(DERIVED)
doy_grid = sub.step_to_doy(DERIVED)
wrap_step = int(np.argmin(np.diff(doy_grid)))

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
rows = []
for ix in INDICES:
    c, _ = sub.load_curves(ix, DERIVED, ids=ids)
    c = c[:, 0, :]
    jump = c[:, wrap_step + 1] - c[:, wrap_step]
    typical = np.median(np.abs(np.diff(c, axis=1)))
    rows.append(dict(index=ix, median_jump=np.median(jump),
                     typical_step=typical,
                     ratio=np.median(np.abs(jump)) / typical,
                     pct_negative=100 * (jump < 0).mean()))
sym = pd.DataFrame(rows)
print(f"The array jumps DOY {doy_grid[wrap_step]:.0f} -> {doy_grid[wrap_step+1]:.0f} "
      f"at step {wrap_step}.\n")
display(sym.round(4))
print("Two things separate this from noise: the magnitude (~4x the typical step) and the\n"
      "sign — negative in 67-71 % of plots, where symmetric noise would give 50 %.")

# %%
curves, cids = sub.load_curves("kndvi", DERIVED, ids=ids)
curves = curves[:, 0, :]
order = np.argsort(doy_grid)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
# the curve in calendar order, one plot, with the boundary marked
j = len(curves) // 2
axes[0].plot(doy_grid[order], curves[j][order], lw=1.8, color="#2f6f7f")
axes[0].scatter([doy_grid[wrap_step], doy_grid[wrap_step + 1]],
                [curves[j][wrap_step], curves[j][wrap_step + 1]],
                s=60, color="#c1553b", zorder=5)
axes[0].set_xlabel("day of year"); axes[0].set_ylabel("kNDVI")
axes[0].set_title(f"one plot ({cids[j]}) — the two red points\nare two days apart", fontsize=9)

step_abs = np.abs(np.diff(curves, axis=1)).mean(axis=0)
axes[1].bar(np.arange(len(step_abs)), step_abs, color="#2f6f7f")
axes[1].bar([wrap_step], [step_abs[wrap_step]], color="#c1553b")
axes[1].axhline(np.median(step_abs), color="k", lw=0.8, ls=":", label="median")
axes[1].set_xlabel("step transition"); axes[1].set_ylabel("mean |delta kNDVI|")
axes[1].set_title("one transition out of 51 stands out", fontsize=9)
axes[1].legend(fontsize=7, frameon=False)

jump = curves[:, wrap_step + 1] - curves[:, wrap_step]
axes[2].hist(jump, bins=60, color="#2f6f7f")
axes[2].axvline(0, color="k", lw=1)
axes[2].set_xlabel("year-boundary jump (kNDVI)"); axes[2].set_ylabel("plots")
axes[2].set_title(f"{100*(jump<0).mean():.0f} % negative — a bias, not noise", fontsize=9)
for ax in axes:
    ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Cause one — the moving average left the ends unsmoothed
#
# `phenosensing/utils.py`, before the fix:
#
# ```python
# def _moving_average(a, n=3):
#     out = np.convolve(a, np.ones(n), "valid") / n
#     return np.concatenate([a[: n // 2], out, a[-(n // 2) :]])   # add values of tail
# ```
#
# The `"valid"` convolution shortens the array, and the gap is filled by **copying the raw
# input back**. With the default `rollWindow=5` that leaves the first 2 and the last 2 steps
# completely unsmoothed while the other 48 are averaged over 5 neighbours — and for a year,
# those four sit on the two sides of the *same* boundary.
#
# `mode="legacy"` keeps the old behaviour available, so the two can be compared directly:

# %%
rng = np.random.default_rng(0)
a = 0.4 + 0.15 * np.sin(2 * np.pi * np.arange(52) / 52) + 0.03 * rng.normal(size=52)

legacy = _moving_average(a, 5, mode="legacy")
fixed = _moving_average(a, 5, mode="wrap")

print("first two steps identical to the raw input?")
print(f"  legacy : {np.allclose(legacy[:2], a[:2])}      <- the bug")
print(f"  wrap   : {np.allclose(fixed[:2], a[:2])}")
print(f"\nmean preserved?   raw {a.mean():.6f}   legacy {legacy.mean():.6f}   "
      f"wrap {fixed.mean():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].plot(a, lw=1, color="#999999", label="raw")
axes[0].plot(legacy, lw=1.8, color="#c1553b", label="legacy")
axes[0].plot(fixed, lw=1.8, color="#2f6f7f", label="wrap (fixed)")
for k in (0, 1, 50, 51):
    axes[0].axvline(k, color="#c1553b", lw=0.6, alpha=0.35)
axes[0].set_title("the four shaded steps are where they differ", fontsize=9)
axes[0].set_xlabel("step"); axes[0].legend(fontsize=8, frameon=False)

axes[1].plot(np.abs(legacy - fixed), lw=1.5, color="#c1553b")
axes[1].set_title("|legacy - wrap|: zero everywhere except the ends", fontsize=9)
axes[1].set_xlabel("step")
for ax in axes:
    ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### The trap: there are two implementations
#
# `_numba._mov_avg` carries its own copy of the same logic, and **that is the one that runs**
# when numba is installed. Fixing only the numpy version would have changed nothing in
# production.
#
# The test caught this by failing: it patched `utils._moving_average`, compared the result
# against itself, and got a difference of exactly zero. Both are now fixed and pinned against
# each other.

# %%
for n in (3, 5, 7):
    x = rng.normal(size=52)
    same = np.allclose(_moving_average(x, n), _numba._mov_avg(x, n))
    print(f"window {n}: numpy and numba agree -> {same}")

# %% [markdown]
# ## 3. Cause two — no reconstructor was periodic
#
# All nine registry entries treated DOY as an *open* interval: nothing tied `f(1)` to
# `f(365)`. The fix adds **`harmonic`** — Fourier regression, periodic by construction, the
# standard device in the field (HANTS; Zhu & Woodcock 2014).

# %%
doy_obs = np.sort(rng.choice(np.arange(1, 366), 80, replace=False)).astype(float)
y_obs = (0.4 + 0.2 * np.sin(2 * np.pi * (doy_obs - 100) / 365.25)
         + 0.05 * rng.normal(size=80))
grid = np.linspace(1, 365, 52)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.scatter(doy_obs, y_obs, s=14, color="#999999", label="observations", zorder=3)
for name, colour in [("linear", "#c1553b"), ("harmonic", "#2f6f7f")]:
    f = get_reconstructor(name)(doy_obs, y_obs, grid, **({"n_harmonics": 3}
                                                         if name == "harmonic" else {}))
    gap = abs(f[-1] - f[0])
    step = np.median(np.abs(np.diff(f)))
    ax.plot(grid, f, lw=1.8, color=colour,
            label=f"{name} — gap at the boundary {gap:.4f} ({gap/step:.2f}x a step)")
    ax.plot([grid[-1], grid[0] + 365], [f[-1], f[0]], lw=1.2, ls=":", color=colour)
ax.set_xlabel("day of year"); ax.set_ylabel("index")
ax.set_title("the dotted segment is the wrap: how far the curve is from closing", fontsize=9)
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# %% [markdown]
# `n_harmonics` controls what the fit can represent: 3 resolves the annual, semi-annual and
# four-monthly components, which covers bimodal and shoulder-season phenologies. More
# harmonics track finer structure at the cost of fitting noise.

# %%
truth = 0.45 + 0.2 * np.cos(2 * np.pi * grid / 365.25) + 0.07 * np.sin(6 * np.pi * grid / 365.25)
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(grid, truth, lw=2.5, color="k", alpha=0.35, label="signal (harmonics 1 and 3)")
for k, colour in [(1, "#c1553b"), (2, "#dd8452"), (3, "#2f6f7f")]:
    f = get_reconstructor("harmonic")(grid, truth, grid, n_harmonics=k)
    ax.plot(grid, f, lw=1.4, color=colour,
            label=f"k={k}   MSE {np.mean((f-truth)**2):.2e}")
ax.set_xlabel("day of year"); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
ax.set_title("k=3 completes the span and the fit becomes exact", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Verification on real cubes
#
# The `.nc` cubes store `obs_{index}` — the raw Landsat observations — alongside the fitted
# curve, so the whole fit can be redone without touching the datacube.
#
# **Two traps, both worth knowing before running this at scale.**
#
# 1. **The cubes lost the per-observation DOY.** `doy` became the 52-step dimension of
#    `phenoshape` and overwrote the original `time`-indexed coordinate, so `PhenoShape`
#    aborts with `AttributeError`. It is recoverable from `time.dt.dayofyear`, and script 02
#    should store it under another name so the collision does not recur.
# 2. **The stored curve is trough-anchored** (`doy_anchor: 108`), the refit is not. So the
#    array wrap of the stored curve is a benign mid-year transition while the refit's array
#    wrap *is* the year boundary. Comparing "last vs first element" of both measures
#    different things and makes the harmonic look worse. The boundary has to be located by
#    DOY, whatever index it falls on.

# %%
def year_boundary_step(curves, doy):
    """Median |jump| across pixels at the transition where DOY crosses the year end."""
    f = np.asarray(curves).reshape(np.asarray(curves).shape[0], -1).T
    d = np.asarray(doy)
    diffs = np.diff(d)
    if diffs.min() > 0:                      # monotonic grid: the boundary is the array wrap
        jump = np.abs(f[:, 0] - f[:, -1])
    else:                                    # anchored grid: it is wherever DOY goes back
        j = int(np.argmin(diffs))
        jump = np.abs(f[:, j + 1] - f[:, j])
    return float(np.median(jump)), float(np.median(np.abs(np.diff(f, axis=1))))


def load_obs(path, index):
    ds = xr.open_dataset(path)
    da = ds[f"obs_{index}"].assign_coords(
        doy=("time", ds.time.dt.dayofyear.values), year=("time", ds.year.values))
    return da, ds


cube_files = sorted(CUBES.glob("*.nc"))
if cube_files:
    rows = []
    for path in cube_files:
        da, ds = load_obs(path, "kndvi")
        r = {"plot": path.stem, "n_obs": int(da.sizes["time"])}
        r["stored"], _ = year_boundary_step(ds["phenoshape"].sel(index="kndvi").values,
                                            ds.doy.values)
        for recon in ["linear", "harmonic"]:
            out = da.pheno.PhenoShape(interpolType=recon, rollWindow=5, nGS=52)
            r[recon], _ = year_boundary_step(out.values, out.doy.values)
        rows.append(r)
    ver = pd.DataFrame(rows).set_index("plot")
    ver["gain_movavg"] = ver["stored"] / ver["linear"]
    ver["gain_harmonic"] = ver["stored"] / ver["harmonic"]
    print("Median |kNDVI| jump AT THE YEAR BOUNDARY, over the 25 pixels of each plot:\n")
    display(ver.round(5))
    m = ver.median()
    print(f"\nmedian over plots:  stored {m['stored']:.5f}   "
          f"moving-average fix {m['linear']:.5f}   + harmonic {m['harmonic']:.5f}")
    print(f"fixing the moving average alone: {m['stored']/m['linear']:.1f}x smaller")
    print(f"adding the harmonic reconstructor: {m['stored']/m['harmonic']:.1f}x smaller")
else:
    print("no cubes yet — the download is still running")

# %%
if cube_files:
    path = cube_files[0]
    da, ds = load_obs(path, "kndvi")
    stored = ds["phenoshape"].sel(index="kndvi").values
    sdoy = ds.doy.values
    so = np.argsort(sdoy)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    px = stored.reshape(stored.shape[0], -1).T
    for row in px[:8]:
        axes[0].plot(sdoy[so], row[so], lw=1, alpha=0.6, color="#c1553b")
    axes[0].set_title(f"{path.stem} — stored curves (8 of 25 pixels)", fontsize=9)

    out = da.pheno.PhenoShape(interpolType="harmonic", rollWindow=5, nGS=52)
    ndoy = out.doy.values
    npx = out.values.reshape(out.values.shape[0], -1).T
    for row in npx[:8]:
        axes[1].plot(ndoy, row, lw=1, alpha=0.6, color="#2f6f7f")
    axes[1].set_title("refitted with the fix + harmonic", fontsize=9)

    for ax, d, arr in [(axes[0], sdoy, px), (axes[1], ndoy, npx)]:
        # mark the boundary and draw the wrap as a dotted segment
        diffs = np.diff(d)
        j = int(np.argmin(diffs)) if diffs.min() < 0 else len(d) - 1
        ax.axvline(365 if diffs.min() > 0 else d[j], color="k", lw=0.8, ls=":")
        ax.set_xlabel("day of year"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("kNDVI")
    obs_doy = da.doy.values
    axes[0].scatter(obs_doy, da.values.reshape(len(obs_doy), -1).mean(axis=1),
                    s=8, color="#333333", alpha=0.5, zorder=4, label="observations (mean)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("The dotted line is the year boundary", fontsize=10)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. What is done and what is left
#
# **Done, upstream in `PhenoSensing` (`eb2dff8`)**
#
# - `_moving_average` pads properly, `mode` forwarded to `np.pad`: `"wrap"` by default,
#   `"reflect"` for open series, `"legacy"` to reproduce pre-fix output.
# - `_numba._mov_avg` fixed to match, and the two are pinned against each other.
# - `harmonic` reconstructor registered.
# - `tests/test_periodicity.py`, 19 tests. The regression guard is
#   `test_phenoshape_curve_closes_the_year`: for every pixel, the wrap-around step may not
#   exceed 3x the typical step. It fails on the old code. Suite went 59 → 78 passing, with
#   the same 5 pre-existing failures.
#
# **Left**
#
# - **Re-extract this project's curves** from the cubes. The download is in progress; the
#   adapter is `scripts/29_refit_curves_from_cubes.py`, not yet written.
# - Then re-derive `phenoshape_*`, the LSP, and re-run the model matrix. Topography, climate,
#   the biodiversity/phylogeny/dark responses and the geomedian blocks are unaffected.
# - **Open upstream item, deliberately left out of the commit:** `xnew = linspace(min(x),
#   max(x))` spans the observed range rather than `[1, 365]`, so curves from different pixels
#   are not comparable step by step. A more invasive API change, worth reviewing separately.
# - **Separate upstream debt found while running the suite:** `tests/golden/` was already
#   failing before this fix — 78 % of elements mismatched, max difference 275 days. It was no
#   safety net and should be regenerated.
