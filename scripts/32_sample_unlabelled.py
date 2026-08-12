#!/usr/bin/env python3
"""Sample unlabelled native-vegetation time series, for masked-autoencoder pretraining.

THE EXTRACTION RUNS ON THE MACHINE WITH DATA CUBE CHILE. Everything up to `--dry-run` is
local: the candidate pool is built from the MapBiomas rasters on disk and can be inspected,
tested and regenerated without any datacube access.

WHY THIS EXISTS. The 135,250 pixel curves already on disk come from the 25-pixel windows of
the 1,082 labelled plots, and are almost redundant with the plots themselves:

    variance between plots        91.5 %
    variance within a plot         8.5 %
    correlation, pixels of the same plot   +0.844
    correlation, pixels of different plots +0.258
    participation dimension, 135,250 pixel curves   1.3
    participation dimension,   1,082 plot curves    1.2

125x more images spanning 1.08x more of the space, and the masked autoencoder trained on them
gained nothing (0.359 against 0.362 unpretrained, `docs/14` section 4). Coverage, not count,
is what is missing.

WHAT CHANGED, AND WHY IT IS NOT A TWEAK. The previous design jittered sample centres +-5 km
around the plots. Measured, that reaches 19,583 km2 of the 67,004 km2 of native cover -- 29 %
-- and raising `n` only stacks more samples onto the same plots (46 per plot at n=50,000).
That is the failure that already cost one attempt, at 5 km instead of 150 m. With MapBiomas
the whole territory can be masked to native cover and sampled directly, and there turn out to
be only **1,484 populated 10 km cells**, so every one of them can be visited. Coverage comes
from *which cells* are visited, not from spending one datacube load per point.

WHAT IS STORED. **Observation-level kNDVI with its dates -- not a fitted curve, not a fixed
grid.** This is the rule of `docs/05` section 1: acquire at observation level, decide the
pooling downstream. `docs/14` section 2 tested eight step resolutions (24 to 196) and they
spread 0.011, below seed noise -- but that was only measurable because all eight derived from
the observations without returning to the cube. Storing a grid here would freeze that choice
and cost a second extraction to revisit it. `raw_series` in `scripts/29_refit_curves_from_cubes.py`
is what builds the grid, locally.

The 5x5 window is **not** stored. `scripts/31_pretrain_mae.py:61-64` reshapes the (N, 25, 52)
cube to (N*25, 1, 52): the 25 pixels are 25 independent curves and the 2-D image comes from
the substrate transform of the series, not from spatial neighbours. The spatial structure is
destroyed before the model sees it, so keeping it costs ~180x the disk for curves that
correlate at +0.844 with each other.

Usage:
    python scripts/32_sample_unlabelled.py --dry-run
    python scripts/32_sample_unlabelled.py --dry-run --cells 40      # quick check
    python scripts/32_sample_unlabelled.py --out data/derived/unlabelled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mb            # noqa: E402

#: kNDVI is the project's primary index (`docs/14`). The five indices all derive from the same
#: six bands of the same load, so widening this costs storage but no extra `dc.load`.
INDEX = "kndvi"

#: Latitude bands for matching the plots' geographic distribution. Sampling native cover
#: uniformly would drift towards whatever is geometrically large, which here is the dry north
#: and the high Andes -- both nearly absent from the labelled set.
LAT_BAND_DEG = 0.5


def plot_table(derived: str) -> pd.DataFrame:
    return pd.read_parquet(Path(derived) / "plots_subset.parquet")


def plot_envelope(p: pd.DataFrame) -> dict:
    return dict(
        x=(float(p["X"].min()), float(p["X"].max())),
        y=(float(p["Y"].min()), float(p["Y"].max())),
        lon=(float(p["lon"].min()), float(p["lon"].max())),
        lat=(float(p["lat"].min()), float(p["lat"].max())),
        years=sorted(p["Year"].astype(int).unique().tolist()),
        n_plots=len(p),
    )


def year_weights(p: pd.DataFrame) -> pd.Series:
    """The plots' own year histogram, so pretraining sees the same temporal mix as fine-tuning.

    Drawing years uniformly would put the pretraining pool in epochs the labelled set does not
    contain, and any transfer result would then be confounded with a shift of era.
    """
    return p["Year"].astype(int).value_counts(normalize=True).sort_index()


def lat_weights(p: pd.DataFrame) -> pd.Series:
    band = (p["lat"] / LAT_BAND_DEG).round().astype(int)
    return band.value_counts(normalize=True).sort_index()


def class_weights(p: pd.DataFrame) -> pd.Series:
    """Native-class shares at the plots, measured through MapBiomas.

    Used to steer the within-cell draw so the pool is made of the same *kinds* of vegetation
    as the labelled set -- shrubland-dominated -- rather than whatever each cell happens to
    hold most of.
    """
    codes, _, _ = mb.sample_at(p["lon"].to_numpy(), p["lat"].to_numpy(), 2014)
    s = pd.Series(codes)
    s = s[mb.is_native(s.to_numpy())]
    return s.value_counts(normalize=True).sort_index()


def cell_id(x: float, y: float, km: float) -> str:
    return f"{int(round(x / (km * 1000)))}_{int(round(y / (km * 1000)))}"


def candidate_cells(env: dict, cell_km: float, screen_year: int,
                    decim: int = 5) -> pd.DataFrame:
    """The 10 km cells holding native cover, found with one decimated pass.

    A screening pass only: it decides *where to look*, using a single mid-collection year. The
    binding native test is `window_native`, applied per cell at full resolution against that
    sample's own causal window.
    """
    bounds = (env["lon"][0], env["lat"][0], env["lon"][1], env["lat"][1])
    path, _, _ = mb.year_map(screen_year)
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, src.transform)
        a = src.read(1, window=win,
                     out_shape=(int(win.height // decim), int(win.width // decim)))
    lon, lat = mb.grid_lonlat(bounds, a.shape)
    rr, cc = np.nonzero(mb.is_native(a))
    if not len(rr):
        return pd.DataFrame(columns=["cell", "x", "y", "lat", "n_native"])
    X, Y = mb.ll_to_utm(lon[cc], lat[rr])
    km = cell_km * 1000
    cx = np.round(X / km).astype(int)
    cy = np.round(Y / km).astype(int)
    df = pd.DataFrame({"cx": cx, "cy": cy, "lat": lat[rr]})
    g = df.groupby(["cx", "cy"]).agg(n_native=("lat", "size"), lat=("lat", "mean"))
    g = g.reset_index()
    g["cell"] = g["cx"].astype(str) + "_" + g["cy"].astype(str)
    g["x"] = g["cx"] * km
    g["y"] = g["cy"] * km
    return g[["cell", "x", "y", "lat", "n_native"]].sort_values("cell", ignore_index=True)


def allocate(cells: pd.DataFrame, latw: pd.Series, total: int,
             floor: int = 2, cap: int = 40) -> np.ndarray:
    """Points per cell, weighted so the pool matches the plots' latitudinal distribution.

    Every populated cell keeps at least ``floor`` points -- the coverage argument is that all
    of them get visited -- while ``cap`` stops a densely-sampled latitude from collapsing back
    into the concentration this design exists to avoid.
    """
    band = (cells["lat"] / LAT_BAND_DEG).round().astype(int)
    per_band = band.value_counts()
    w = band.map(lambda b: latw.get(b, 0.0) / per_band[b]).to_numpy(float)
    if w.sum() <= 0:
        w = np.ones(len(cells))
    n = np.clip(np.round(w / w.sum() * total).astype(int), floor, cap)
    return n


def draw_points(bounds_ll, win_start: int, win_end: int, n: int, min_sep_m: float,
                classw: pd.Series, rng,
                plot_tree=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """``n`` native points inside one cell, class-steered and at least ``min_sep_m`` apart.

    The native test is the AND across the window's own years (`mapbiomas.window_native`), so a
    pixel that was plantation and got cleared inside the window never enters.

    ``plot_tree`` keeps samples away from the labelled plots themselves. Spacing the samples
    from *each other* does not do this, and the difference is not theoretical: without the
    exclusion, 29 of 16,937 samples landed inside a plot's own 5x5 window and 10 inside its
    footprint, the closest at 14 m. That is exactly the input-side leakage this pool is
    supposed to be free of, and it would have been claimed rather than measured.
    """
    mask, used = mb.window_native(bounds_ll, win_start, win_end)
    if not mask.any():
        return np.array([]), np.array([]), np.array([]), used
    path, _, _ = mb.year_map(win_end)
    codes = mb._read(path, bounds_ll)
    lon, lat = mb.grid_lonlat(bounds_ll, mask.shape)
    rr, cc = np.nonzero(mask)
    cls = codes[rr, cc]

    # Candidate coordinates in the projected CRS, so every distance below is in real metres.
    # Measuring in raster pixels instead leaves a residual: MapBiomas is in degrees, and
    # converting the cell corners gives a pixel size that is only locally right -- measured,
    # that let pairs through at 972 m against a 1,000 m target.
    ux, uy = mb.ll_to_utm(lon[cc], lat[rr])

    # Drop candidates sitting on a labelled plot, BEFORE the quotas are computed -- otherwise
    # a class whose only pixels are inside a plot still gets its quota and then cannot fill it.
    if plot_tree is not None:
        keep = plot_tree.query(np.column_stack([ux, uy]))[0] >= min_sep_m
        if not keep.any():
            return np.array([]), np.array([]), np.array([]), used
        rr, cc, cls, ux, uy = rr[keep], cc[keep], cls[keep], ux[keep], uy[keep]

    # Quota per class, then draw within each. Ranking all candidates by class weight instead
    # would take shrubland first in every cell and starve forest everywhere -- measured, that
    # gave 97.7 % shrubland against a 48 % target and no forest at all. Quotas are capped by
    # what the cell actually holds and the shortfall is redistributed, so a cell with no
    # shrubland still yields its points.
    present = [c for c in np.unique(cls)]
    w = np.array([classw.get(int(c), 0.0) for c in present], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(present))
    quota = {int(c): int(np.floor(n * wi / w.sum())) for c, wi in zip(present, w)}
    for c in present:                                   # spend the rounding remainder
        if sum(quota.values()) >= n:
            break
        quota[int(c)] += 1

    # A spatial hash with bucket side = min_sep, so anything closer is guaranteed to sit in
    # the 3x3 bucket neighbourhood -- then the actual distance decides. One point per bucket
    # is a different and weaker constraint: two points in adjacent buckets can be one pixel
    # apart, which measured 50 m against the same target.
    grid: dict[tuple[int, int], list[int]] = {}
    taken = []

    def far_enough(i) -> bool:
        gx, gy = int(ux[i] // min_sep_m), int(uy[i] // min_sep_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if (ux[i] - ux[j]) ** 2 + (uy[i] - uy[j]) ** 2 < min_sep_m ** 2:
                        return False
        grid.setdefault((gx, gy), []).append(i)
        return True

    def take_from(pool_idx, want):
        got = 0
        for i in pool_idx:
            if got >= want:
                break
            if far_enough(i):
                taken.append(i)
                got += 1
        return got

    order_all = rng.permutation(len(rr))
    for c in sorted(present, key=lambda c: -classw.get(int(c), 0.0)):
        sel = order_all[cls[order_all] == c]
        take_from(sel, quota[int(c)])
    if len(taken) < n:                                  # redistribute the shortfall
        take_from(order_all, n - len(taken))

    taken = np.array(taken, dtype=int)
    if not len(taken):
        return np.array([]), np.array([]), np.array([]), used
    return lon[cc[taken]], lat[rr[taken]], cls[taken], used


def build_pool(args) -> tuple[pd.DataFrame, dict]:
    p = plot_table(args.derived)
    env = plot_envelope(p)
    yw, lw, cw = year_weights(p), lat_weights(p), class_weights(p)
    rng = np.random.default_rng(args.seed)

    print(f"labelled envelope: X {env['x'][0]:.0f}..{env['x'][1]:.0f}  "
          f"Y {env['y'][0]:.0f}..{env['y'][1]:.0f}  years {min(env['years'])}"
          f"..{max(env['years'])}  ({env['n_plots']} plots)")
    print(f"MapBiomas: {len(mb.available_years())} annual maps, "
          f"{min(mb.available_years())}..{max(mb.available_years())}")

    cells = candidate_cells(env, args.cell_km, args.screen_year)
    print(f"{len(cells)} cells of {args.cell_km:g} km hold native cover")
    target = args.n
    if args.cells:
        keep = min(args.cells, len(cells))
        # scale the target with the subset, or every cell hits the cap and the quick check
        # reports a class and year mix that the full run will not reproduce
        target = max(1, int(round(args.n * keep / len(cells))))
        cells = cells.sample(keep, random_state=args.seed)
        cells = cells.sort_values("cell", ignore_index=True)
        print(f"  limited to {len(cells)} cells (--cells), target n scaled to {target:,}")

    # Every labelled plot, so no sample is drawn on top of one. Pretraining sees no targets,
    # but it does see inputs, and a sample on a test plot's own ground is input-side leakage.
    from scipy.spatial import cKDTree
    plot_tree = cKDTree(p[["X", "Y"]].to_numpy())

    per_cell = allocate(cells, lw, target)
    half = args.cell_km * 1000 / 2
    rows, t0 = [], time.time()
    for i, (_, c) in enumerate(cells.iterrows(), 1):
        years = rng.choice(yw.index.to_numpy(), size=args.years_per_cell,
                           replace=False if args.years_per_cell <= len(yw) else True,
                           p=yw.to_numpy())
        want = max(1, int(round(per_cell[i - 1] / args.years_per_cell)))
        for yr in years:
            yr = int(yr)
            ws, we = yr - (args.window - 1), yr
            west, south = mb.utm_to_ll(c.x - half, c.y - half)
            east, north = mb.utm_to_ll(c.x + half, c.y + half)
            lon, lat, cls, used = draw_points((west, south, east, north), ws, we, want,
                                              args.min_sep_km * 1000, cw, rng,
                                              plot_tree=plot_tree)
            for lo, la, cd in zip(lon, lat, cls):
                X, Y = mb.ll_to_utm(lo, la)
                rows.append(dict(X=float(X), Y=float(Y), lon=float(lo), lat=float(la),
                                 year=yr, win_start=ws, win_end=we, cell=c.cell,
                                 mb_code=int(cd), mb_class=mb.class_name(int(cd)),
                                 mb_year_used=used[-1]["year_used"],
                                 mb_delta=used[-1]["delta"]))
        if i % 200 == 0 or i == len(cells):
            print(f"  [{i}/{len(cells)}] {len(rows):,} points, "
                  f"{time.time()-t0:.0f} s", flush=True)

    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool, env
    pool["sample_id"] = [f"U{i:06d}" for i in range(len(pool))]
    return pool, env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="target number of samples")
    p.add_argument("--years-per-cell", type=int, default=2, dest="years_per_cell",
                   help="distinct causal windows per cell. This is the cost knob: it "
                        "multiplies the number of `dc.load` calls one-for-one, and more of "
                        "them decouples year from place")
    p.add_argument("--min-sep-km", type=float, default=1.0, dest="min_sep_km",
                   help="minimum spacing between sample centres. Without it, n=20,000 "
                        "re-concentrates and the coverage gain is lost")
    p.add_argument("--cell-km", type=float, default=10.0, dest="cell_km",
                   help="side of the grouping cell. Loads are grouped by (cell, year) as in "
                        "scripts/02; the cell size trades calls against peak memory, because "
                        "the whole cell is materialised at once")
    p.add_argument("--window", type=int, default=3, help="years, as in scripts/01")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--index", default=INDEX, help="vegetation index to store")
    p.add_argument("--screen-year", type=int, default=2014, dest="screen_year",
                   help="year used for the cheap cell-screening pass only")
    p.add_argument("--cells", type=int, default=0,
                   help="limit to this many cells, for a quick check")
    p.add_argument("--workers", type=int, default=4,
                   help="Dask workers (0 = no dask), as in scripts/02. This is not a tuning "
                        "knob: without dask the COG reads from S3 are serial and a single "
                        "cell measured 796 s, against ~17 s with 4 local workers")
    p.add_argument("--gateway", action="store_true",
                   help="use the EASI dask-gateway cluster instead of a LocalCluster. Each "
                        "gateway worker gets its own 8 cores and 28 GB, where local workers "
                        "share this machine's 8; the load is S3-bound, so more workers is "
                        "close to linear")
    p.add_argument("--resume", action="store_true",
                   help="skip (cell, year) groups already in the manifest. A run of this "
                        "length will be interrupted, and re-loading a done cell costs the "
                        "same as loading it the first time")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many loads, for a timed pilot")
    p.add_argument("--pool", default=None, metavar="PARQUET",
                   help="read a previously planned pool instead of rebuilding it. Building "
                        "it reads 3 rasters per cell and takes ~6 min; a resumed run should "
                        "not pay that again, and re-planning risks a pool that differs from "
                        "the one already half-extracted")
    p.add_argument("--save-pool", default=None, metavar="PARQUET",
                   help="write the planned pool and exit (implies --dry-run)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="data/derived/unlabelled")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.pool:
        pool = pd.read_parquet(args.pool)
        env = plot_envelope(plot_table(args.derived))
        print(f"pool read from {args.pool}: {len(pool):,} samples")
    else:
        pool, env = build_pool(args)
        if args.save_pool:
            pool.to_parquet(args.save_pool, index=False)
            print(f"  -> {args.save_pool}  ({len(pool):,} samples)")
            return
    if pool.empty:
        print("no candidate points -- check the MapBiomas rasters")
        return

    groups = list(pool.groupby(["cell", "year"], sort=True))
    print(f"\n{len(pool):,} candidate samples, {len(groups):,} loads (cell x year)")
    print(f"native classes: "
          f"{pool.mb_class.value_counts().to_dict()}")
    assert mb.is_native(pool.mb_code.to_numpy()).all(), "non-native class in the pool"

    yr = pool.year.value_counts(normalize=True).sort_index()
    ref = year_weights(plot_table(args.derived))
    print("\nyear distribution (pool vs plots):")
    for y in sorted(set(yr.index) | set(ref.index)):
        print(f"  {y}  {100*yr.get(y,0):5.1f} %   plots {100*ref.get(y,0):5.1f} %")
    if (pool.mb_delta != 0).any():
        n_str = int((pool.mb_delta != 0).sum())
        print(f"\n{n_str:,} samples matched to a displaced map "
              f"(max |delta| = {int(pool.mb_delta.abs().max())} y)")

    outdir = ROOT / args.out
    if args.dry_run:
        print("\n" + pool.head().to_string(index=False))
        print(f"\nsamples per load: median "
              f"{pool.groupby(['cell','year']).size().median():.0f}, "
              f"max {pool.groupby(['cell','year']).size().max()}")
        print("\nwould then, on the datacube machine:")
        print(f"  1. load the causal window per cell and compute {args.index}")
        print("  2. read the QA-masked observations at each centre, with their dates")
        print(f"  3. write series.parquet (sample_id, time, {args.index}) + manifest.csv")
        return

    import datacube
    from datacube.utils.aws import configure_s3_access
    from biodiv import cube as cubemod

    # n_workers=4 is the default scripts/02 used for the labelled extraction, and the reason
    # it finished: the COG reads from S3 are what dominates, and they only parallelise here.
    client = cluster = None
    t_boot = time.time()
    if args.workers > 0 and args.gateway:
        from dask_gateway import Gateway
        gw = Gateway()
        cluster = gw.new_cluster(gw.cluster_options())
        cluster.scale(args.workers)
        client = cluster.get_client()
        link = cluster.dashboard_link
        print("\n" + "=" * 78)
        print(f"  DASK DASHBOARD   {link}")
        print(f"  cluster {cluster.name}   scaling to {args.workers} workers")
        print("=" * 78 + "\n", flush=True)
        client.wait_for_workers(1, timeout=600)      # pods take a while to schedule
        print(f"first worker up after {time.time()-t_boot:.0f} s; "
              f"{len(client.scheduler_info()['workers'])} ready", flush=True)
    elif args.workers > 0:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=args.workers, processes=True,
                               dashboard_address=None)
        client = Client(cluster)
        print(f"dask: local cluster, {args.workers} workers")

    # mandatory on the Data Observatory: the buckets are requester-pays and without this
    # `dc.load` returns empty rasters rather than failing
    configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
    dc = datacube.Datacube(app="unlabelled_pretrain_pool")

    outdir.mkdir(parents=True, exist_ok=True)
    man_path, ser_path = outdir / "manifest.csv", outdir / "series.parquet"
    series, man, t0 = [], [], time.time()
    done: set = set()
    if args.resume and man_path.exists():
        prev = pd.read_csv(man_path)
        man = prev.to_dict("records")
        if ser_path.exists():
            series.append(pd.read_parquet(ser_path))
        done = set(zip(prev.get("cell", pd.Series(dtype=str)),
                       prev.get("year", pd.Series(dtype=int))))
        print(f"resume: {len(done):,} (cell, year) groups already done, "
              f"{len(prev):,} manifest rows")

    def flush():
        pd.DataFrame(man).to_csv(man_path, index=False)
        if series:
            pd.concat(series, ignore_index=True).to_parquet(ser_path, index=False)

    buf = 3 * args.resolution
    chunks = {"time": 1} if args.workers > 0 else None

    for gi, ((cell, year), g) in enumerate(groups, 1):
        if (cell, year) in done:
            continue
        if args.limit and gi > args.limit:
            print(f"--limit {args.limit} reached")
            break
        bbox = (g["X"].min() - buf, g["Y"].min() - buf,
                g["X"].max() + buf, g["Y"].max() + buf)
        try:
            ds = cubemod.load_window(dc, bbox, int(g["win_start"].iloc[0]),
                                     int(g["win_end"].iloc[0]), args.resolution,
                                     dask_chunks=chunks)
            if ds is None:
                raise ValueError("no datasets in window")
            idx, _ = cubemod.to_indices(ds)
            da = idx[args.index]
            # materialise the whole cell at once: this is where the S3 reads parallelise
            da = da.compute() if hasattr(da, "compute") else da
        except Exception as e:                  # a bad cell must not end the run
            for _, pt in g.iterrows():
                man.append(dict(sample_id=pt.sample_id, cell=cell, year=int(year),
                                error=f"load: {type(e).__name__}: {e}"))
            print(f"  [{gi}/{len(groups)}] {cell} {year}: LOAD FAILED -- {e}", flush=True)
            continue

        for _, pt in g.iterrows():
            try:
                v = da.sel(x=pt.X, y=pt.Y, method="nearest")
                vals = np.asarray(v.values, dtype="float32")
                times = np.asarray(v.time.values)
                ok = np.isfinite(vals)
                series.append(pd.DataFrame({"sample_id": pt.sample_id,
                                            "time": times[ok],
                                            args.index: vals[ok]}))
                man.append(dict(sample_id=pt.sample_id, X=pt.X, Y=pt.Y, lon=pt.lon,
                                lat=pt.lat, year=int(pt.year), win_start=int(pt.win_start),
                                win_end=int(pt.win_end), cell=pt.cell,
                                mb_code=int(pt.mb_code), mb_class=pt.mb_class,
                                mb_year_used=int(pt.mb_year_used),
                                mb_delta=int(pt.mb_delta), n_obs=int(ok.sum())))
            except Exception as e:
                man.append(dict(sample_id=pt.sample_id, cell=cell, year=int(year),
                                error=f"{type(e).__name__}: {e}"))
        ok_n = sum("error" not in r for r in man)
        print(f"  [{gi}/{len(groups)}] {cell} {year}: {len(g)} -- {ok_n:,} written, "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        if gi % 20 == 0:            # checkpoint, so an interrupted run keeps its work
            flush()

    flush()
    mandf = pd.DataFrame(man)
    ok = mandf[mandf.get("error").isna()] if "error" in mandf else mandf
    print(f"\n{len(ok):,} series written of {len(pool):,} attempted")
    if "n_obs" in ok and len(ok):
        print(f"observations per series: median {ok.n_obs.median():.0f}, "
              f"min {ok.n_obs.min()}, max {ok.n_obs.max()}")
        print("\nby land cover:")
        print(ok.groupby("mb_class").size().to_string())
    (outdir / "sampling.json").write_text(json.dumps(
        dict(vars(args)) | {"envelope": env, "native_codes": sorted(mb.NATIVE)},
        indent=2, default=str))
    # a gateway cluster keeps its pods until told otherwise, so it must not be left orphaned
    if cluster is not None and args.gateway:
        cluster.shutdown()
        print("gateway cluster shut down")

    print(f"\n  -> {outdir}/  (series.parquet, manifest.csv, sampling.json)")
    print("\nThen, locally: `scripts/31_pretrain_mae.py --unlabelled` builds the step grid "
          "with\n`raw_series` and pretrains. Measure the participation dimension first -- if "
          "it does not\nclear 1.3, the coverage problem is not solved and the GPU time is "
          "wasted.")


if __name__ == "__main__":
    main()
