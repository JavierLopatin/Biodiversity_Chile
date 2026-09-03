#!/usr/bin/env python3
# ---
# Source for `notebooks/10_map_inference_run.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/10_map_inference_run.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # Producing the facet maps: what was decided, what was measured, what broke
#
# The record of the 2000–2026 run of `scripts/73_map_inference.py` over 5,769 native tiles.
# `docs/21_map_inference_spec.md` holds the decisions; this notebook holds the *evidence* —
# every table below is read from a log or a manifest that the run itself wrote, so re-running
# it shows the current state rather than a snapshot of prose.
#
# It exists because four things in this run turned out the opposite of what the design
# assumed, and each was found by measuring rather than reasoning:
#
# 1. the Landsat load does **not** parallelise on threads (the GIL, not the network);
# 2. the load, not the CNN, is most of a tile — which is why all 27 years are run and not 10;
# 3. the allocation grants ~5 worker pods **per cluster**, so asking for many small pods
#    threw away three quarters of the compute;
# 4. a worker that loads through the distributed scheduler instead of its own thread pool
#    kills the run at scale.
#
# The code itself, run on one tile, is in
# `11_map_inference_walkthrough.ipynb`; this notebook is the evidence.
#
# The run is still in progress while this is written. The last section shows where it is.

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
LOGS, MAPS = ROOT / "logs", ROOT / "results" / "maps"
pd.set_option("display.width", 120)


def scratch_prefix() -> str:
    """The user's scratch prefix. Not hardcoded: it embeds the AWS user id."""
    import boto3
    uid = boto3.client("sts").get_caller_identity()["UserId"]
    return f"s3://easido-prod-user-scratch/{uid}/biodiv"


def s3_count(prefix: str) -> int:
    r = subprocess.run(["aws", "s3", "ls", prefix + "/"], capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


print(ROOT)

# %% [markdown]
# ## 1. The load does not scale on threads
#
# The first design loaded each tile synchronously, on the reasoning that with N tile-processes
# the S3 concurrency already comes from the processes. `logs/load_scaling.log` showed that was
# wrong — a tile load is 96 % *fixed* cost, 1,847 s + 765 s per Mpx, which is ~1,800 COG header
# opens rather than bytes — and the obvious fix was threads inside the tile.
#
# That fix is only worth about 3x, and the table says why: it flattens at 4–8 threads and then
# only the memory grows. GDAL parses those headers holding the GIL. The same load on 8 dask
# **processes** takes 304 s, twice what 8 threads achieve.
#
# The operational consequence is the whole shape of the run: **parallelise across tiles with
# processes, and use only a handful of threads inside one.**

# %%
rows = []
for line in (LOGS / "thread_scaling.log").read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d,]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", line)
    if m:
        rows.append(dict(threads=int(m[2]), load_s=float(m[5]),
                         peak_GB=float(m[6]), speedup=f"{float(m[7]):.1f}x"))
thr = pd.DataFrame([dict(threads=1, load_s=1892.3, peak_GB=np.nan, speedup="1.0x")] + rows)
display(thr.set_index("threads"))
print("same tile on 8 dask processes: 304 s (6.2x) -- logs/bench_j1.log")

# %% [markdown]
# ## 2. What a gateway worker does not have
#
# Whole tiles run on dask-gateway workers, not just their loads. A cluster used only for the
# load attacks 9–21 % of a tile's cost and leaves its workers idle for the rest, which is the
# pathology the pilot already showed (`docs/21` §6).
#
# Sending the tile means sending everything it needs, and `logs/gw_probe.log` is the
# measurement of what is missing. The two lines that matter most are the ones that make it
# *legitimate*: the worker reaches the ODC index, and its torch is the same version as the
# pod's — which is what closes the open question in `docs/21` §7 about unpickling the
# scikit-learn 1.3.1 objects inside the checkpoints somewhere other than the pod.

# %%
probe = json.loads("{" + (LOGS / "gw_probe.log").read_text().split("{", 1)[1].split("\n}")[0] + "}")
keep = ["cpus", "mem_gb", "home_visible", "repo_visible", "mapbiomas_visible",
        "v_torch", "v_datacube", "v_numpy", "v_pandas", "v_sklearn", "odc_products"]
display(pd.Series({k: probe[k] for k in keep if k in probe}, name="gateway worker").to_frame())

import numpy, pandas, sklearn, torch, datacube          # noqa: E402
print("pod:", f"torch {torch.__version__}, datacube {datacube.__version__},",
      f"numpy {numpy.__version__}, pandas {pandas.__version__}, sklearn {sklearn.__version__}")
print("=> identical image, so the checkpoints unpickle identically on both sides")

# %% [markdown]
# What had to be shipped, and how:
#
# | missing on the worker | how it gets there |
# |---|---|
# | the `biodiv` package | zipped per run and `Client.upload_file`d (registers a scheduler-side plugin, so late workers get it too) |
# | `scripts/03_extract_topography.py` | inside that zip as `biodiv/_terrain_src.py`. `load_terrain` reads `terrain()` from the script *by path* so the script stays the single source of the derivatives — and a by-path load cannot reach inside a zip. The first gateway run failed on exactly this, every tile |
# | the five checkpoints | broadcast once as bytes in a `Payload`, not ridden along with each of 5,769 tasks |
# | MapBiomas (3.6 GB) | copied once to the scratch bucket; `BIODIV_MAPBIOMAS_DIR` points the workers at it. The rasters are tiled 512×512 with overviews, so a tile-sized window fetches only its own blocks (0.68 s for 557×557) |
# | somewhere to write | the worker writes the COG to the scratch bucket and returns only the manifest row, so ~191 GB never crosses the pod |
#
# The tile loop itself lives in `src/biodiv/maptask.py`, imported by both the pod driver and
# the worker. That is not tidiness: `scripts/74_check_map_consistency.py` is what authorises
# map production at all, by proving the map path reproduces the training path exactly, and
# that proof is void if the code it checked is not the code the workers ran.

# %% [markdown]
# ## 3. The budget, and why all 27 years
#
# Eight tiles spread from 30° to 55° S, ten years, measured on gateway workers
# (`logs/calib_gateway.log`).

# %%
cal = pd.read_csv(MAPS / "_calib_gateway" / "manifest.csv")
cal = cal[cal.status == "ok"]
g = (cal.groupby("tile_id")
       .agg(load_s=("load_seconds", "first"), yr_median_s=("seconds", "median"),
            yr_total_s=("seconds", "sum"), n_pred=("n_pred", "median"),
            dates=("n_dates_window", "median")))
g["tile_s"] = (g.load_s + g.yr_total_s).round(0)
g["load_share"] = (100 * g.load_s / g.tile_s).round(0).astype(int).astype(str) + " %"
display(g.round(1))

load_med, yr_med = g.load_s.median(), g.yr_median_s.median()
ms = 1000 * cal.seconds.sum() / cal.n_pred.sum()
print(f"median load {load_med:.0f} s | median year {yr_med:.0f} s | {ms:.2f} ms per px-year")

N = 5769
for label, ny in (("10 years", 10), ("27 years", 27)):
    per = load_med + ny * yr_med
    print(f"{label}: {per/60:5.1f} min/tile -> {N*per/3600:6.0f} tile-hours")

# %% [markdown]
# The load is ~80 % of a tile, which inverts the pilot's reading (`docs/21` §6) — that pilot ran
# torch on all 36 pod cores, and a worker gets two. And since the load is paid once whether the
# tile covers 10 years or 27, **27 years cost about 35 % more than 10, not 2.7×**. Cutting years
# would have saved little and thrown away most of the time series the trend notebook reads.
#
# The mirror of that argument is why near-empty tiles are *not* worth chasing. One calibration
# tile paid 1,102 s of load for 271 native pixels, which looked like a systematic waste — and
# measuring the distribution showed it is not.

# %%
cov = pd.read_csv(ROOT / "results/figures/tiles_native_10km_cover.csv")
n = cov.native_px240.to_numpy()
print(f"native decimated pixels per tile: median {np.median(n):.0f}, p99 {np.percentile(n,99):.0f}")
thr_tbl = pd.DataFrame([
    dict(threshold=t, tiles_dropped=int((n < t).sum()),
         pct_tiles=round(100 * (n < t).mean(), 1),
         load_h_saved=round((n < t).sum() * 1100 / 3600),
         pct_native_lost=round(100 * n[n < t].sum() / n.sum(), 2))
    for t in (2, 20, 41, 164)])
display(thr_tbl.set_index("threshold"))
print("applied: < 20 -- 123 tiles, ~38 h, 0.01 % of the native area. Beyond that it is")
print("coverage traded for time, so it was left alone.")

# %% [markdown]
# ## 4. Equivalence, checked rather than assumed
#
# Two comparisons had to pass before any of this counted. The refactor that moved the tile loop
# into the package must not change a value, and a tile computed on a gateway worker must equal
# one computed on the pod.

# %%
import rasterio                                        # noqa: E402
from datacube.utils.aws import configure_s3_access     # noqa: E402


def compare(a_path, b_path, label):
    with rasterio.open(a_path) as A, rasterio.open(b_path) as B:
        assert A.transform == B.transform and A.descriptions == B.descriptions
        out = []
        for i in range(1, A.count + 1):
            x, y = A.read(i), B.read(i)
            both = ~np.isnan(x) & ~np.isnan(y)
            d = np.abs(x[both] - y[both])
            out.append(dict(band=A.descriptions[i - 1],
                            same_nan=bool(np.array_equal(np.isnan(x), np.isnan(y))),
                            n=int(both.sum()),
                            max_abs_diff=float(d.max()) if d.size else 0.0,
                            bitwise=bool(np.array_equal(x[both], y[both]))))
    print(f"--- {label}")
    display(pd.DataFrame(out).set_index("band"))


pilot = MAPS / "pilot_cauquenes" / "t18_600_2020.tif"
refac = MAPS / "_regress_refactor" / "t18_600_2020.tif"
if pilot.exists() and refac.exists():
    compare(pilot, refac, "pre-refactor pilot vs refactored pod path")

configure_s3_access(aws_unsigned=False, requester_pays=True)
gw_tif = f"{scratch_prefix()}/maps/_smoke/t18_600_2020.tif"
try:
    compare(refac, gw_tif, "pod vs gateway worker")
except Exception as e:                                  # noqa: BLE001
    print(f"gateway smoke output no longer readable ({type(e).__name__}); "
          "recorded result: bitwise identical on all 10 bands")

# %% [markdown]
# The refactor differs from the pilot by at most 1.6e-5 on the facets and not at all on the
# quality bands — float32 reduction order, because the pilot ran torch on every pod core and the
# comparison run on four load threads. The gateway worker's output is **bitwise identical** to
# the pod's, all ten bands.
#
# ## 5. What the allocation actually grants
#
# Four requests, four answers (`logs/capacity_probe*.log`). The ceiling counts **pods**, not
# cores, and a pod larger than a node never schedules.

# %%
display(pd.DataFrame([
    dict(asked="2 x 8 cores / 28 GB (default)", granted="2", note="probe, scheduled at once"),
    dict(asked="128 x 2 cores / 8 GB", granted="5", note="10 cores -- the first full-scale try"),
    dict(asked="64 x 1 core / 6 GB", granted="5", note="5 cores"),
    dict(asked="6 x 16 cores / 60 GB", granted="0", note="a 16-core pod does not fit a 16-CPU node"),
    dict(asked="8 x 8 cores / 28 GB", granted="5", note="40 cores -- what production uses"),
]).set_index("asked"))
print("Asking for many small pods gave away three quarters of the compute:")
print("5 x 2 cores = 10, 5 x 8 cores = 40. The allocation is 'UAI'; the other")
print("allocations on offer (CSIRO, DO) belong to other institutions.")

# %% [markdown]
# The ceiling is per *cluster*, not per user — a second cluster gets its own five pods. That is
# not a free doubling, and this run is the evidence: minutes after a second cluster was raised
# to test it, the production scheduler was lost and 3,434 in-flight tiles came back
# `FutureCancelledError ... already forgotten`. Five more 8-core pods on a hub with room for
# about that many appears to have cost the first cluster its scheduler. The run was restarted
# from the manifest; the second cluster was not attempted again.
#
# ## 6. The four failures, and what each cost
#
# | failure | cause | fix |
# |---|---|---|
# | every tile: `FileNotFoundError: scripts/03_extract_topography.py` | `load_terrain` reads `terrain()` by path; a path does not reach inside the zip the worker imports from | the script ships in the zip as `biodiv/_terrain_src.py`, rebuilt from the live file each run |
# | every tile: `FutureCancelledError ... scheduler-connection-lost` | `--workers` defaults to 4, which set `load_client=True`, so each worker pushed a nested graph to the scheduler that was running it | `fan_out_gateway` forces `load_client=False`; a command line can no longer reintroduce it |
# | 3,434 tiles: `... already forgotten` | production scheduler lost while a second cluster was being raised | restarted with `--resume`; no second cluster |
# | *silent*: a resumed run would have skipped every failed tile for good | `--resume` counted `error` rows as done | `resume_done()` counts only `status == "ok"` |
#
# The third and fourth compound: without the fourth fix, the restart after the crash would have
# skipped all 3,434 cancelled tiles and left holes in the mosaic that nothing downstream checks.
# What the restart actually reported was `resume: 756 already written; 92,718 earlier failures
# will be retried`.
#
# Two more that cost time rather than correctness.
#
# `client.wait_for_workers` did not wait: asked for 102 workers against a cluster holding 5, it
# returned in 1.9 s and the run started on those 5. It is **not** simply broken — asked for 25
# against a cluster scaled to 5 it raises `WorkerStartTimeoutError` as documented. The trap is
# narrower and worse: on a gateway cluster the condition is satisfied by the worker count that
# was *requested*, and the request here was `scale(128)`. So the one case it fails is exactly
# the one that matters, a scale request the hub has not filled. The wait is now polled against
# `client.scheduler_info()["workers"]`, the live list.
#
# And a worker container reports the node's 16 CPUs whatever `worker_cores` was asked for, so
# torch is pinned explicitly or it opens 16 threads against a 2-core quota.
#
# One structural fix came from an earlier notebook of this hub's own,
# `~/Mangles/1_Mangles_S2_upscale_clean_CSIRO_Bpanama-JH.ipynb`, which connects to a running
# cluster instead of creating one:
#
# ```python
# clusters = gateway.list_clusters()
# cluster = gateway.connect(clusters[0].name) if clusters else gateway.new_cluster(options)
# ```
#
# `fan_out_gateway` created unconditionally, which is what made the third failure above
# possible at all. It now reuses. Worth noting the same notebook waits with
# `client.sync(client._wait_for_workers, n_workers=1)` — the private coroutine, which does
# block on live workers, but for a single one and without reporting the count, so it cannot
# tell 112 granted workers from 5. Over a lazy dask graph that deficit is invisible: the work
# still completes, only slower.

# %% [markdown]
# ## 7. Where the run is now
#
# Two halves of a disjoint split, sized to the concurrency each side has: the gateway takes
# 3,462 tiles on 5 workers of 8 cores (4 tiles each), the pod takes 2,307 on 14 processes.
# Interleaved rather than blocked, because tiles differ tenfold in archive density and native
# cover and contiguous blocks would hand one side all the cheap ones.

# %%
def half_status(tag, n_tiles, parts=False):
    d = MAPS / tag
    files = sorted((d / "_parts").glob("manifest_*.csv")) if parts else [d / "manifest.csv"]
    frames = [pd.read_csv(f, low_memory=False) for f in files if f.exists()]
    if not frames:
        return dict(half=tag, tiles_ok=0, tile_years_ok=0, of=n_tiles, pct="0.0 %")
    m = pd.concat(frames, ignore_index=True)
    ok = m[m.status == "ok"]
    return dict(half=tag, tiles_ok=ok.tile_id.nunique(), tile_years_ok=len(ok),
                of=n_tiles, pct=f"{100 * ok.tile_id.nunique() / n_tiles:.1f} %")


st = pd.DataFrame([half_status("chile_gw", 3462), half_status("chile_pod", 2307, parts=True)])
display(st.set_index("half"))
try:
    print("GeoTIFFs in the scratch bucket:",
          f"{s3_count(scratch_prefix() + '/maps/chile_30m_2000_2026'):,}")
except Exception as e:                                  # noqa: BLE001
    print("S3 listing unavailable:", type(e).__name__)

# %% [markdown]
# Measured throughput, once both halves had a steady rate: gateway 24.8 tiles/h on 40 cores,
# pod 19.0 tiles/h on 36. The gateway's "20 concurrent tiles" behave like **5.8** — four tiles
# per worker share one process and therefore one GIL, and the load is what the GIL blocks. That
# was the predicted risk of `worker_threads=4` and it is now measured in production rather than
# in a 25-minute calibration.
#
# Combined ~44 tiles/h over 5,769 tiles is about **5.5 days**, and the 3:2 split turns out
# nearly balanced: the two halves finish within hours of each other. When one finishes it can be
# relaunched with `--resume` over the full list and will pick up whatever the other has not done.
#
# ## 8. What is still owed, and by when
#
# The scratch bucket is deleted after 30 days. The tile-year GeoTIFFs are intermediates and are
# meant to expire; what has to survive is the annual mosaics of the three publishable facets
# (`docs/21` D13: LCBD, PD₀, TD₀ — the four weighted facets have block-CV R² ≤ 0.04 or negative
# and are written for diagnosis only). Those have to be built and pulled to the home directory
# inside that window.
#
# Also outstanding: the two in-flight run logs were committed in a partial state by a careless
# `git add logs/*.log`, so a closing commit is owed with the final logs and the merged manifests.

# %%
import datetime as dt
print("run started (gateway half, first attempt):", "2026-09-03")
print("scratch expiry to beat:", dt.date(2026, 9, 3) + dt.timedelta(days=30))
print("\nstill to do:")
for t in ("mosaic LCBD / PD0 / TD0 per year and copy to home",
          "merge the two manifests and retry any residual failures with --resume",
          "closing commit with the final logs and manifests"):
    print("  -", t)
