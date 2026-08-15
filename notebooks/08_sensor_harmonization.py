#!/usr/bin/env python3
# ---
# Source for `notebooks/08_sensor_harmonization.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/08_sensor_harmonization.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # Cross-sensor (TM/ETM+ vs OLI) harmonization
#
# Started from a question about the `kfold_time`/`kfold_loc_time` collapse (R2 goes
# strongly negative under temporal holdout, having been +0.32-0.40 under `kfold5_window`):
# is part of it explained by unharmonized reflectance, since census years earlier than 2013
# only ever draw on TM/ETM+ and years after 2022 only ever draw on OLI (`landsat7_c2l2_sr`
# stops in the catalog in 2022 -- `src/biodiv/cube.py` `SENSOR_YEARS`), so year and sensor
# regime are confounded by construction.
#
# | | |
# |---|---|
# | bias test | `scripts/40_sensor_harmonization_test.py` |
# | correction | `src/biodiv/harmonize.py` |
# | wired into | `scripts/29_refit_curves_from_cubes.py --harmonize ...` |
# | tests | `tests/test_harmonize.py` |
#
# **The first version of this notebook got it wrong** in the same spirit as
# `06_year_boundary.ipynb`: it measured one flat offset (the whole-year mean residual) and
# applied it everywhere. §1b is the record of what that missed.

# %%
import json
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

from biodiv import harmonize                          # noqa: E402

CUBES = ROOT / "data" / "derived" / "phenology"
DERIVED = ROOT / "data" / "derived"
OFFSETS_PATH = ROOT / "results" / "tables" / "sensor_harmonization.json"

print("cubes available:", len(list(CUBES.glob("*.nc"))))

# %% [markdown]
# ## 1. The measured bias, by tercile of vegetation level
#
# Residual = observation - that plot's own PhenoShape curve, paired at the plot level within
# each tercile of the curve's own level (low/mid/high), so site and season are controlled for
# by construction and the level-dependence is not averaged away. See the script's docstring.

# %%
offsets_raw = json.loads(OFFSETS_PATH.read_text())
rows = []
for idx, r in offsets_raw["per_index"].items():
    if r.get("status") == "PENDIENTE":
        continue
    for b in r["bins"]:
        if b.get("status") == "PENDIENTE":
            continue
        rows.append(dict(index=idx, level=b["level_mediana"], n_parcelas=b["n_parcelas"],
                         offset_oli_menos_tm_etm=b["offset_oli_menos_tm_etm_mediana"],
                         wilcoxon_p=b["wilcoxon_p"]))
bias = pd.DataFrame(rows).set_index(["index", "level"]).sort_index()
display(bias.round(6))

# %% [markdown]
# NDVI, EVI, kNDVI and SAVI all show a large, highly significant offset that **grows with the
# vegetation level** (OLI reads higher, and increasingly so toward the growth peak). NBR does
# not (the offsets sit near zero, one bin even flips sign) -- consistent with the literature:
# NBR leans on SWIR, which is less sensitive to the NIR/Red relative-spectral-response
# mismatch between ETM+ and OLI that drives the others.

# %% [markdown]
# ## 1b. Why not one flat number
#
# The first version measured a single whole-year offset (+0.0401 for kNDVI). Applied, it
# overcorrected the low season -- where the real gap is close to zero -- into visibly negative
# values, and barely touched the real gap at the growth peak, which is several times larger.
# J. Lopatin caught this from the before/after plot in §2: the correction was making the two
# sensor clusters *more* separated in the low season, not less.
#
# The mechanism: regressing the residual directly on the plot's own curve level, separately
# per sensor family, gives opposite-signed slopes --

# %%
level_rows = []
for f in sorted(CUBES.glob("*.nc")):
    ds = xr.open_dataset(f)
    sensor = np.asarray(ds["sensor"].values)
    family = np.array([harmonize.SENSOR_FAMILY.get(s, "other") for s in sensor])
    if len(set(family) & {"tm_etm", "oli"}) < 2 or "obs_kndvi" not in ds:
        continue
    obs = ds["obs_kndvi"].mean(dim=["y", "x"], skipna=True).values
    curve = ds["phenoshape"].sel(index="kndvi").mean(dim=["y", "x"], skipna=True).values
    doy_obs = ds["time"].dt.dayofyear.values.astype(float)
    grid_doy = ds["doy"].values.astype(float)
    order = np.argsort(grid_doy)
    level = np.interp(doy_obs, grid_doy[order], curve[order], period=365)
    resid = obs - level
    for fam, lvl, r in zip(family, level, resid):
        if fam in ("tm_etm", "oli") and np.isfinite(lvl) and np.isfinite(r):
            level_rows.append((fam, lvl, r))

from scipy import stats as st                          # noqa: E402

lvl_df = pd.DataFrame(level_rows, columns=["family", "level", "residual"])
print(f"n = {len(lvl_df)} observations\n")
for fam in ("tm_etm", "oli"):
    g = lvl_df[lvl_df["family"] == fam]
    slope, intercept, r, p, se = st.linregress(g["level"], g["residual"])
    print(f"  {fam:8s}  slope={slope:+.4f}  intercept={intercept:+.4f}  "
         f"r={r:+.3f}  p={p:.2e}  n={len(g)}")
print("\nOpposite signs: TM/ETM+ undershoots more at high vegetation, OLI overshoots more.")
print("A gain-type mismatch, not a level shift -- hence the correction is measured per")
print("tercile of level (§1) and interpolated, not a single subtracted constant.")

# %% [markdown]
# ## 2. Two example plots, before and after
#
# Both have a large, roughly balanced mix of ETM+ and OLI observations inside their causal
# window (found by sorting `manifest.csv` on `n_obs` among plots with both families present).

# %%
EXAMPLES = ["PCL1205", "PCL1203"]
offsets = harmonize.load_offsets(OFFSETS_PATH)


def plot_pre_post(ax_pre, ax_post, plot_id, index="kndvi"):
    ds = xr.open_dataset(CUBES / f"{plot_id}.nc")
    obs = ds[f"obs_{index}"].mean(dim=["y", "x"], skipna=True)
    doy = ds["time"].dt.dayofyear.values.astype(float)
    sensor = np.asarray(ds["sensor"].values)
    family = np.array([harmonize.SENSOR_FAMILY.get(s, "other") for s in sensor])

    grid_doy = ds["doy"].values.astype(float)
    order = np.argsort(grid_doy)
    curve = ds["phenoshape"].sel(index=index).mean(dim=["y", "x"], skipna=True).values

    level = np.interp(doy, grid_doy[order], curve[order], period=365)
    corrected = harmonize.apply_offset(obs, index, offsets, level)

    for ax, values, title in [(ax_pre, obs.values, "antes"),
                              (ax_post, corrected.values, "despues (nivel-dependiente)")]:
        for fam, colour, label in [("tm_etm", "#2f6f7f", "TM/ETM+"),
                                   ("oli", "#c1553b", "OLI")]:
            m = family == fam
            ax.scatter(doy[m], values[m], s=14, alpha=0.6, color=colour, label=label)
        ax.plot(grid_doy[order], curve[order], lw=1.4, color="k", alpha=0.5,
               label="curva (sin corregir)", zorder=1)
        ax.set_title(f"{plot_id} — {title}", fontsize=9)
        ax.set_xlabel("day of year"); ax.grid(alpha=0.25)


fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(11, 3.6 * len(EXAMPLES)), sharey="row")
for i, pid in enumerate(EXAMPLES):
    plot_pre_post(axes[i, 0], axes[i, 1], pid)
axes[0, 0].set_ylabel("kNDVI"); axes[0, 0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.show()

# %% [markdown]
# The low season now sits close to overlapping (not overcorrected into negative values as the
# flat offset did), and the growth-peak gap is visibly reduced, though not fully closed --
# expected, since the correction is a *population* tercile median and any single plot has its
# own noise around it. The black curve shown is the *original*, uncorrected PhenoShape fit,
# kept fixed across both panels so the shift in the points is the only thing that changes; a
# refit against the corrected observations is what
# `scripts/29_refit_curves_from_cubes.py --harmonize ...` produces (§3), not recomputed here.

# %% [markdown]
# ## 3. Effect on the actual model-input curve
#
# `scripts/29_refit_curves_from_cubes.py --raw-series --ngs 100 --harmonize ...` is what
# RF06/MLP06 actually train on (`BIODIV_CURVES=_raw100_sensorharm`). Comparing that output
# against the unharmonized `_raw100` curve, same two plots:

# %%
h = pd.read_parquet(DERIVED / "phenoshape_by_index_raw100_sensorharm.parquet")
n = pd.read_parquet(DERIVED / "phenoshape_by_index_raw100.parquet")
h = h[(h["index"] == "kndvi") & (h["px"] == "mean5x5")].set_index("plot_id")
n = n[(n["index"] == "kndvi") & (n["px"] == "mean5x5")].set_index("plot_id")
scol = [c for c in h.columns if c.startswith("s")]

fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(6 * len(EXAMPLES), 3.4), sharey=True)
for ax, pid in zip(axes, EXAMPLES):
    ax.plot(n.loc[pid, scol].to_numpy(), lw=1.6, color="#2f6f7f", label="original (_raw100)")
    ax.plot(h.loc[pid, scol].to_numpy(), lw=1.6, color="#c1553b",
           label="armonizado (_raw100_sensorharm)")
    ax.set_title(pid, fontsize=9); ax.set_xlabel("paso (grilla raw100)"); ax.grid(alpha=0.25)
axes[0].set_ylabel("kNDVI"); axes[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.show()

maxdiff = (h[scol] - n[scol]).abs().max(axis=1)
print(f"diferencia maxima, estas 2 parcelas: {maxdiff.loc[EXAMPLES].round(4).to_dict()}")
print("ya no es un numero unico (antes 0.0401 siempre) -- varia con el nivel de cada tramo")
print("de la curva de cada parcela, como corresponde a una correccion tipo ganancia.")

# %% [markdown]
# ## 4. Did the fixed correction help the model?
#
# `RF06`/`MLP06`, `--force` (para no reusar los resultados corridos con el offset plano, que
# quedaban cacheados bajo el mismo `run_id`). R2 mean over 5 facets,
# `results/models/summary.csv`.

# %%
df = pd.read_csv(ROOT / "results" / "models" / "summary.csv")
pairs = [
    ("RF06_curve_all-topo-area_raw100", "RF06_curve_all-topo-area_raw100_sensorharm", "RF06"),
    ("MLP06_curve_kndvi_raw100", "MLP06_curve_kndvi_raw100_sensorharm", "MLP06"),
]
schemes = ["kfold_time", "kfold_loc_time", "time_within_owner"]
for base, harm, label in pairs:
    sub = df[df["run_id"].isin([base, harm]) & df["scheme"].isin(schemes)]
    piv = sub.groupby(["scheme", "run_id"])["R2"].mean().unstack().reindex(schemes)
    piv.columns = ["original", "armonizado"]
    piv["delta"] = piv["armonizado"] - piv["original"]
    print(f"=== {label} ===")
    display(piv.round(4))

# %% [markdown]
# Delta is negative in 5 of 6 cells. The one positive (RF06 / `time_within_owner`, +0.006) is
# noise -- the same order of magnitude as the negative fluctuations next to it, not a
# distinguishable improvement.
#
# **Already settled, negatively -- twice over.** The first pass (flat, whole-year offset) was
# wrong in shape, as the before/after in §2 first showed and §1b confirms quantitatively
# (opposite-signed level-vs-residual slopes per sensor family). Fixing the shape (§1, tercile-
# based, gain-type correction) changed the curves correctly -- verified visually (§2) and at
# the exact magnitude expected (§3) -- but the model result did not improve. That rules out
# "the correction was broken" as an explanation for the earlier null result: cross-sensor bias
# is real, large, and now properly corrected, and it still is not what drives the
# `kfold_time`/`kfold_loc_time` collapse.
#
# Browning (Miranda et al. 2023) was checked separately and ruled out the same way -- error is
# *lower*, not higher, in the browning-affected latitude band and years. What is left
# standing, with real but modest evidence: the Owner x Year confound (Cramer's V=0.73,
# `docs/16_stemp_protocol.md`) and a weak-but-significant correlation between per-plot
# temporal isolation and error (Spearman rho 0.10-0.21, p<1e-4 for most model/scheme pairs)
# -- neither alone explains the full size of the collapse.
