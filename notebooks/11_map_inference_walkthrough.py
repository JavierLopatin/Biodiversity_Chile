#!/usr/bin/env python3
# ---
# Source for `notebooks/11_map_inference_walkthrough.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/11_map_inference_walkthrough.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # How a facet map is actually produced
#
# The processing path of `scripts/73_map_inference.py`, shown as code and then **run** on one
# tile: load the tile's whole Landsat archive span once, cut a causal `y-2..y` window per
# target year, build a 100-step raw kNDVI curve for every pixel, fold it into the serpentine
# image, attach the pixel's topography, run the five all-data seed checkpoints, back-transform,
# and write a ten-band GeoTIFF.
#
# The companion notebook, `10_map_inference_run.ipynb`, is the record of the production run —
# what was measured, what broke, where it stands. This one is only the mechanism.
#
# **What runs here and what does not.** Section 2 executes the real tile path, on one tile and
# one year, so it finishes in a couple of minutes. It raises no cluster and writes no manifest.
# Section 3 shows how the full 5,769-tile run is launched, **commented out**: it takes days,
# and a run that hangs off a Jupyter kernel dies with the kernel — which is how the previous
# attempt was lost.

# %%
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
pd.set_option("display.width", 120)
print(ROOT)

# %% [markdown]
# ## 1. The code that actually runs
#
# Shown from the modules rather than copied into cells: a notebook that pastes the
# orchestration is a second copy that drifts, and the whole argument for
# `src/biodiv/maptask.py` is that there is exactly one tile loop. `inspect.getsource` keeps
# this section honest — if it disagrees with the code, it is because the code changed.

# %%
import inspect                                         # noqa: E402

from biodiv import maptask as mt                       # noqa: E402

import importlib.util                                # noqa: E402

_spec = importlib.util.spec_from_file_location("s73", ROOT / "scripts" / "73_map_inference.py")
s73 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s73)


def show(obj, drop_doc=False):
    src = inspect.getsource(obj)
    if drop_doc:                    # the docstrings are the reasoning; keep them by default
        src = re.sub(r'"""(?:.|\n)*?"""\n\s*', "", src, count=1)
    print(src.rstrip())
    print("-" * 100)


print("### the load: one tile, whole archive span, on a THREAD pool")
show(mt.load_tile)

# %% [markdown]
# `load_tile` is where the run died the first time at scale. `cfg.load_client` came from
# `--workers`, whose default is 4, so `da.compute()` ran with no explicit scheduler — and
# inside a dask worker that hands the graph back to the distributed scheduler already running
# the task. Every tile returned
# `FutureCancelledError ... scheduler-connection-lost`. The gateway path now forces
# `load_client=False` so no command line can reintroduce it.

# %%
print("### the tile loop: curves, mask, ensemble, GeoTIFF, one row per year")
show(mt.run_tile)

# %%
print("### the worker entry point, and the per-process state it caches")
show(mt.run_tile_remote)
show(mt.worker_state)

# %% [markdown]
# Two things in `worker_state` are the difference between a working worker and 5,769 failed
# tiles: the ensemble is built once per process and keyed on the checkpoint bytes (rebuilding
# five torch models per tile would be pure waste over hundreds of tiles), and
# `configure_s3_access` is called again on the worker — not redundantly, because a worker that
# joined after the client configured the cluster would otherwise read `usgs-landsat` unsigned
# and get `AccessDenied` on every scene.

# %%
print("### raising the cluster, shipping the code, submitting the tiles")
show(s73.fan_out_gateway)

# %%
print("### what gets shipped, and how the manifest stays resumable")
show(s73.package_biodiv)
show(s73.resume_done)

# %% [markdown]
# ## 2. One tile, end to end
#
# The same `mt.run_tile` the workers call, on the Cauquenes pilot tile, for a single year so
# the archive span is 2018–2020 and this finishes in a couple of minutes. No cluster is raised:
# a second gateway cluster is what cost the production run its scheduler
# (`10_map_inference_run.ipynb` §5), and the point
# here is the tile path, not the fan-out.

# %%
import os                                              # noqa: E402
import tempfile                                        # noqa: E402

# `training_targets` reads the unified target tables; both are needed for the range clipping
os.environ["BIODIV_UNIFIED"] = "1"
os.environ["BIODIV_CURVES"] = "_raw100"

import datacube                                        # noqa: E402
from datacube.utils.aws import configure_s3_access     # noqa: E402

from biodiv import mapinfer as mi                      # noqa: E402

# Not optional and not idempotent-by-luck: `usgs-landsat` is requester-pays, and without this
# every scene read comes back RasterioIOError('AccessDenied'). It has to happen before the
# Datacube is built -- the same boot order `scripts/73` and `scripts/35` follow.
configure_s3_access(aws_unsigned=False, requester_pays=True)

CKPT = ROOT / ("results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_"
               "maekndvi_m06_ctr_FINAL_alldata/final")
OOF = ROOT / ("results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_"
              "maekndvi_m06_ctr/kfold5_block20_unified/oof_predictions.csv")

ens = mi.FacetEnsemble(sorted(CKPT.glob("model_seed*.pt")))
resid = mi.oof_residuals_scaled(OOF, ens.members[0].scaler, ens.targets)
y_train = mi.training_targets(ROOT / "data" / "derived", ens.targets)
perm = mi.serpentine_perm(mi.NGS)
print(f"{len(ens.members)} seeds | targets {ens.targets}")
print(f"context ({len(ens.context_columns)}): {ens.context_columns}")
print(f"smearing residuals usable per target: {np.isfinite(resid).sum(axis=0).tolist()}")

# %%
dc = datacube.Datacube(app="nb10-one-tile")
out = Path(tempfile.mkdtemp(prefix="nb10_"))

TILE_M = 9990
tile = dict(tile_id="t18_600", xmin=18 * TILE_M, ymin=600 * TILE_M,
            xmax=19 * TILE_M, ymax=601 * TILE_M)
cfg = mt.TileConfig(
    years=(2020,), dest=str(out), resolution=30, area_m2=900.0, stratum="basal",
    mask="mapbiomas", batch=8192, load_threads=4, load_client=False, torch_threads=0,
    tags=dict(tag="notebook_10_demo", years="2020-2020", area_m2=900.0, stratum="basal",
              smearing="oof", clip=True,
              caveat=("predicted values are conditional means and under-disperse the upper "
                      "tail; use as a relative surface")))

rows = mt.run_tile(dc, tile, cfg, ens, resid, y_train, perm)
if rows and rows[0].get("status") == "ok":
    display(pd.DataFrame(rows)[["tile_id", "year", "status", "n_px", "n_native", "n_pred",
                                "n_dates_window", "seconds", "load_seconds"]])
else:
    # Say which tile and why, instead of failing three cells later on a missing variable
    raise RuntimeError(f"the demo tile did not produce a raster: {rows}")

# %% [markdown]
# `n_pred` below `n_native` would mean pixels that are native but have no complete 100-step
# curve; here they coincide. `load_seconds` is small only because the span is three years —
# the production tiles carry 1998–2026 and pay ~1,100 s.

# %%
import matplotlib.pyplot as plt                        # noqa: E402
import rasterio                                        # noqa: E402

tif = Path(rows[0]["file"])
with rasterio.open(tif) as src:
    bands = {n: src.read(i) for i, n in enumerate(src.descriptions, 1)}
    tags = src.tags()

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, name, cmap in zip(axes, ["td_inext_q0", "pd_inext_q0", "lcbd_count_sorensen", "n_obs"],
                          ["viridis", "viridis", "magma", "cividis"]):
    a = bands[name]
    finite = a[np.isfinite(a)]
    im = ax.imshow(a, cmap=cmap,
                   vmin=np.percentile(finite, 2) if finite.size else None,
                   vmax=np.percentile(finite, 98) if finite.size else None)
    ax.set_title(name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"t18_600, {tags['year']} -- three published facets and the observation count\n"
             f"white = not native vegetation (MapBiomas {tags['map_year']}) or no complete curve",
             fontsize=10)
fig.tight_layout()
plt.show()

# %%
# The three publishable facets against the plot range they were trained on: the maps
# under-disperse the upper tail by construction (docs/21 section 6), and this is where that
# shows rather than a caveat to take on trust.
tr = pd.DataFrame(y_train, columns=ens.targets)
summary = []
for t in ("td_inext_q0", "pd_inext_q0", "lcbd_count_sorensen"):
    a = bands[t][np.isfinite(bands[t])]
    summary.append(dict(facet=t, map_median=np.median(a), map_max=a.max(),
                        train_median=tr[t].median(), train_max=tr[t].max()))
display(pd.DataFrame(summary).set_index("facet").round(3))
print("The map maximum sitting well below the training maximum is the compression, not a bug.")

# %% [markdown]
# ## 3. Is dask actually parallelising the load? Measured on this tile
#
# `load_tile` passes `dask_chunks={"time": 1}` whenever `load_threads > 0`, so the load *is* a
# lazy dask graph, one chunk per acquisition date. What the gateway path changes is not the
# chunking but **which scheduler executes it**: an in-process thread pool rather than the
# distributed scheduler. There are two layers of parallelism and they are easy to conflate:
#
# | layer | granularity | who runs it |
# |---|---|---|
# | distributed | one whole tile = one task | `dask.distributed`, 20 tiles at a time in production |
# | threaded | the `{"time": 1}` chunks of that one tile | a local pool, 2–4 threads |
#
# Only `--load-threads 0` with `--workers 0` gives `dask_chunks=None` and a plain synchronous
# read. The cell below runs the same tile under each setting and times it, so the claim is a
# table rather than an assertion.
#
# Two caveats on reading these numbers. The demo span is three years (~200 dates), not the
# 29 the production tiles carry, so the absolute seconds are much smaller than the ~1,100 s a
# real tile pays — the *ratios* are the point. And the pod is running 14 tile-processes while
# this executes, so every row here is measured under load and is pessimistic.

# %%
import time                                            # noqa: E402
from dataclasses import replace                        # noqa: E402

timings = []
for lt in (0, 1, 2, 4, 8):
    probe = replace(cfg, load_threads=lt)
    t0 = time.time()
    da = mt.load_tile(dc, tile, probe)
    dt = time.time() - t0
    timings.append(dict(load_threads=lt,
                        dask_chunks="None" if lt == 0 else "{'time': 1}",
                        scheduler="synchronous read" if lt == 0 else f"threads, {lt} workers",
                        dates=int(da.sizes["time"]),
                        px=int(da.sizes["y"]) * int(da.sizes["x"]),
                        seconds=round(dt, 1)))
    print(timings[-1], flush=True)
    del da

t = pd.DataFrame(timings).set_index("load_threads")
t["vs sync"] = (t.seconds.iloc[0] / t.seconds).round(2).astype(str) + "x"
display(t)

# %% [markdown]
# The row at `load_threads=0` is the one that answers the question: it is the same load with
# dask switched off entirely, and the gap to the threaded rows is dask's contribution. The gap
# between 4 and 8 is the GIL — GDAL parses the COG headers holding it, which is why production
# uses 2 threads inside a gateway worker running four tiles and 4 inside a pod process running
# one, rather than the 8 or 16 that would look natural.

# %% [markdown]
# ## 4. How the full run is launched
#
# Not from here. The commands below are the ones that produced the run in
# `10_map_inference_run.ipynb`, kept as comments so this notebook cannot start days of compute
# by being run top to bottom.
#
# Three things about them are decisions, not defaults:
#
# * **`setsid`**, because the first attempt at this run was a child of the terminal session and
#   died with it, leaving nothing — not even a manifest to resume from.
# * **the split**, because the allocation grants ~5 worker pods per cluster and the pod has 36
#   cores of its own: neither side alone is the machine. Interleaved 3:2 rather than blocked,
#   since tiles differ tenfold in archive density and contiguous blocks would hand one side all
#   the cheap ones.
# * **`--workers 0`**, because its default of 4 makes each worker push its load back to the
#   scheduler that is running it, which killed the first full-scale attempt outright.

# %%
# ---- prepare the tile lists (results/figures/tiles_native_10km_run.csv is the 5,769 tiles
#      that survive the < 20 native-pixel filter; see notebook 10 section 3) --------------
#
# import pandas as pd
# t = pd.read_csv("results/figures/tiles_native_10km_run.csv")
# i = t.index % 5
# t[i < 3].to_csv("results/figures/tiles_run_gateway.csv", index=False)   # 3,462 tiles
# t[i >= 3].to_csv("results/figures/tiles_run_pod.csv", index=False)      # 2,307 tiles
#
# ---- the gateway half: 5 workers x 8 cores, 4 tiles each ------------------------------
#      2 load threads and 2 torch threads per tile, not 4 and 8: four tiles share one
#      process and therefore one GIL, and the load is what the GIL blocks.
#
# B=s3://easido-prod-user-scratch/$(python -c \
#     "import boto3;print(boto3.client('sts').get_caller_identity()['UserId'])")/biodiv
# CK=results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL_alldata/final
# OOF=results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr/kfold5_block20_unified/oof_predictions.csv
#
# setsid nohup env BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python -u scripts/73_map_inference.py \
#     --tiles-file results/figures/tiles_run_gateway.csv --years 2000-2026 \
#     --area-m2 900 --stratum basal --mask mapbiomas \
#     --ckpt-dir "$CK" --oof-csv "$OOF" \
#     --gw-workers 5 --worker-cores 8 --worker-memory 28 --worker-threads 4 \
#     --load-threads 2 --torch-threads 2 --workers 0 \
#     --dest "$B/maps/chile_30m_2000_2026" --mapbiomas-dir "$B/MapBiomas" \
#     --out results/maps --tag chile_gw --resume > logs/chile_gw.log 2>&1 &
#
# ---- the pod half: 14 tile-processes on 36 cores --------------------------------------
#      real processes, so no shared GIL: 4 load threads each and one torch thread, which is
#      ~5.6x more core-efficient than one many-threaded process. Memory is the binding
#      limit, not cores -- a tile load peaks at 2.4 GB.
#
# setsid nohup env BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python -u scripts/73_map_inference.py \
#     --tiles-file results/figures/tiles_run_pod.csv --years 2000-2026 \
#     --area-m2 900 --stratum basal --mask mapbiomas \
#     --ckpt-dir "$CK" --oof-csv "$OOF" \
#     --jobs 14 --workers 0 --load-threads 4 --torch-threads 1 \
#     --dest "$B/maps/chile_30m_2000_2026" --mapbiomas-dir "$B/MapBiomas" \
#     --out results/maps --tag chile_pod --resume > logs/chile_pod.log 2>&1 &
#
# ---- mopping up ----------------------------------------------------------------------
#      --resume skips only rows with status "ok", so failures and no_data are retried. When
#      one half finishes, relaunch it against the FULL list and it picks up whatever the
#      other has not reached.

# %% [markdown]
# Before any of this may run at all, the gate: `scripts/74_check_map_consistency.py` has to
# report ALL PASS. It checks, over the 3,102 training plots, that the map path reproduces the
# training path's context, image and model output exactly, and that the target scaler closes
# its round trip. No maps are produced if it fails — and that gate is the reason the tile loop
# lives in `src/biodiv/maptask.py` rather than in the script, so that the code it checked is
# the code the workers run.
#
# ```bash
# BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python scripts/74_check_map_consistency.py \
#     --oof-csv "$OOF"
# ```
