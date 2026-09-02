#!/usr/bin/env python3
"""Extract the causal 3-year Landsat series for the Living_Trees_Chile plots.

THE EXTRACTION RUNS ON THE MACHINE WITH DATA CUBE CHILE. Everything up to `--dry-run` is
local: the site table is built from the workbook on disk and can be inspected, tested and
regenerated without any datacube access.

This is `scripts/32_sample_unlabelled.py` with the pool swapped for a workbook. The load
loop, the dask boot order, the resume/checkpoint pattern and the per-site error column are
the same code that ran 2,898 loads in 5.2 h without a single failure, so they are copied
rather than reinvented. Three things differ, and each is a consequence of the data:

**The extraction unit is the coordinate, not `um`.** The workbook is one row per tree with
the plot coordinate repeated; 78 of the 1,943 `um` carry more than one coordinate, so
grouping by `um` would average plots sitting in different Landsat pixels. See
`biodiv.io_living_trees`.

**Cells are 5 km, not 10, and the buffer is `patch * resolution`.** The sites are dispersed
-- median 1 site per 10 km cell -- so grouping buys almost nothing (1,764 loads at 5 km
against 2,021 unGrouped) while a large bbox costs memory, and here 11 variables are
materialised per load instead of one.

**All five indices and all six scaled bands are stored, at the centre pixel and as the 5x5
mean.** Same `dc.load`, so no extra S3 reads. Indices are lossy -- SAVI cannot be recovered
from a stored NDVI -- and the plots are 250-500 m2, smaller than one 900 m2 pixel, so the
5x5 mean carries the stand signal and absorbs GPS error while the centre pixel stays
available for the strict reading.

**One parquet per product.** `series_<variable>_<px>.parquet`, 22 tables with the minimal
`site_id, time, sensor, value` schema of `data/derived/unlabelled/series.parquet`. Fitted
curves are NOT produced: that is the rule of `docs/05` section 1 -- acquire at observation
level, decide the pooling downstream. `raw_series`/`interp_grid` in `src/biodiv/curves.py`
build any grid locally without returning to the cube.

Usage:
    python scripts/35_extract_living_trees.py --dry-run
    python scripts/35_extract_living_trees.py --save-sites data/derived/living_trees/sites.parquet
    python scripts/35_extract_living_trees.py --gateway --workers 16 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cube                       # noqa: E402
from biodiv import io_living_trees as io_lt   # noqa: E402

#: The five indices and the six scaled bands they were built from.
VARS = list(cube.INDEX_NAMES) + list(cube.BAND_VARS)

#: The two spatial readings kept per site.
PX = ["center", "mean5x5"]


def table_name(var: str, px: str) -> str:
    return f"series_{var}_{px}.parquet"


def site_table(args) -> pd.DataFrame:
    if args.sites:
        return pd.read_parquet(args.sites)
    return io_lt.load_sites(args.xlsx, window=args.window, cell_km=args.cell_km)


def crop(ds, site, patch: int):
    """Centre pixel and the patch x patch window around a site, by nearest index.

    The centre is taken by index rather than from the cropped window so that a site near
    the edge of the loaded bbox -- where the slice is clamped and no longer centred -- still
    gets its own pixel and not a neighbour's.
    """
    iy = int(np.abs(ds.y.values - site.Y).argmin())
    ix = int(np.abs(ds.x.values - site.X).argmin())
    half = patch // 2
    y0, y1 = max(iy - half, 0), min(iy + half + 1, ds.sizes["y"])
    x0, x1 = max(ix - half, 0), min(ix + half + 1, ds.sizes["x"])
    return ds.isel(y=iy, x=ix), ds.isel(y=slice(y0, y1), x=slice(x0, x1))


def series_rows(site_id: str, times, sensors, centre, window) -> dict:
    """Build the per-(variable, px) frames for one site, dropping masked observations."""
    out = {}
    for var in VARS:
        a = np.asarray(centre[var].values, dtype="float32")
        ok = np.isfinite(a)
        out[(var, "center")] = pd.DataFrame(
            {"site_id": site_id, "time": times[ok], "sensor": sensors[ok], var: a[ok]})

        w = np.asarray(window[var].values, dtype="float32")      # (time, y, x)
        n_px = np.isfinite(w).sum(axis=(1, 2)).astype("int16")
        with warnings.catch_warnings():          # all-masked dates are expected, not a bug
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(w, axis=(1, 2)).astype("float32")
        ok = n_px > 0
        out[(var, "mean5x5")] = pd.DataFrame(
            {"site_id": site_id, "time": times[ok], "sensor": sensors[ok],
             var: m[ok], "n_px": n_px[ok]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", default="data/Living_Trees_Chile.xlsx")
    p.add_argument("--sites", default=None, help="site parquet to reuse instead of the xlsx")
    p.add_argument("--save-sites", default=None, help="write the site table and stop")
    p.add_argument("--out", default="data/derived/living_trees")
    p.add_argument("--window", type=int, default=3, help="causal window length, y-(w-1)..y")
    p.add_argument("--cell-km", type=float, default=5.0, help="load-grouping cell side")
    p.add_argument("--patch", type=int, default=5, help="window side in pixels")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--workers", type=int, default=4, help="dask workers (0 = no dask)")
    p.add_argument("--gateway", action="store_true", help="dask-gateway instead of local")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="stop after N loads (0 = all)")
    p.add_argument("--flush-every", type=int, default=50, help="checkpoint interval, loads")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sites = site_table(args)
    outdir = ROOT / args.out
    if args.save_sites:
        Path(args.save_sites).parent.mkdir(parents=True, exist_ok=True)
        sites.to_parquet(args.save_sites, index=False)
        print(f"  -> {args.save_sites}  ({len(sites):,} sites)")
        return

    groups = list(sites.groupby(["cell", "year"], sort=True))
    print(f"{len(sites):,} sites | {len(groups):,} loads (cell x year) | "
          f"causal window y-{args.window - 1}..y")

    if args.dry_run:
        n = sites.groupby(["cell", "year"]).size()
        span = sites.groupby(["cell", "year"]).agg(
            dx=("X", lambda s: s.max() - s.min()), dy=("Y", lambda s: s.max() - s.min()))
        print(f"sites per load: median {n.median():.0f}, max {n.max()}")
        print(f"max bbox: {span.dx.max()/1000:.1f} x {span.dy.max()/1000:.1f} km")
        print("\nproducts by census year:")
        for y, k in sites.groupby("year").size().items():
            prods = cube.products_for(int(y) - (args.window - 1), int(y))
            print(f"  {y} ({k:4d} sites): {[q.replace('_c2l2_sr', '') for q in prods]}")
        missing = [int(y) for y in sites.year.unique()
                   if not cube.products_for(int(y) - (args.window - 1), int(y))]
        print(f"\nyears with no Landsat product: {missing or 'none'}")
        print(f"\nwould write {len(VARS) * len(PX)} tables to {outdir}/:")
        print(f"  {table_name(VARS[0], PX[0])} ... {table_name(VARS[-1], PX[-1])}")
        return

    import datacube
    from datacube.utils.aws import configure_s3_access

    # Boot order is not negotiable: cluster, then configure_s3_access(client=...), then the
    # Datacube. Reversed, every worker read of the requester-pays usgs-landsat bucket comes
    # back AccessDenied.
    client = cluster = None
    t_boot = time.time()
    if args.workers > 0 and args.gateway:
        from dask_gateway import Gateway
        gw = Gateway()
        cluster = gw.new_cluster(gw.cluster_options())
        cluster.scale(args.workers)
        client = cluster.get_client()
        print("\n" + "=" * 78)
        print(f"  DASK DASHBOARD   {cluster.dashboard_link}")
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

    configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
    dc = datacube.Datacube(app="living_trees_series")

    outdir.mkdir(parents=True, exist_ok=True)
    man_path = outdir / "manifest.csv"
    parts: dict[tuple[str, str], list[pd.DataFrame]] = {(v, k): [] for v in VARS for k in PX}
    man: list[dict] = []
    done: set = set()
    if args.resume and man_path.exists():
        prev = pd.read_csv(man_path)
        # A failed load is NOT done. Building `done` from every manifest row -- which is what
        # this did first -- makes --resume skip exactly the groups that need retrying, and the
        # run comes back "complete" with the failures frozen in. Real case: the gateway
        # scheduler dropped the connection on load 1,640 of 1,764 and the client reconnected
        # by itself, so one group was left behind out of an otherwise clean run.
        n_err = int(prev["error"].notna().sum()) if "error" in prev else 0
        if n_err:
            prev = prev[prev["error"].isna()]
            print(f"resume: dropping {n_err} failed manifest rows -- those loads are retried")
        man = prev.to_dict("records")
        keep = set(prev["site_id"])
        for key in parts:
            f = outdir / table_name(*key)
            if f.exists():
                # Keep only sites the manifest still vouches for, so a retried group cannot
                # append a second copy of rows a partial write already left behind.
                old = pd.read_parquet(f)
                parts[key].append(old[old["site_id"].isin(keep)])
        done = set(zip(prev.get("cell", pd.Series(dtype=str)),
                       prev.get("year", pd.Series(dtype=int))))
        print(f"resume: {len(done):,} (cell, year) loads already done, "
              f"{len(prev):,} manifest rows")

    def flush() -> None:
        pd.DataFrame(man).to_csv(man_path, index=False)
        for key, frames in parts.items():
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(
                    outdir / table_name(*key), index=False)

    buf = args.patch * args.resolution
    chunks = {"time": 1} if args.workers > 0 else None
    t0 = time.time()
    n_loaded = 0

    for gi, ((cell, year), g) in enumerate(groups, 1):
        if (cell, year) in done:
            continue
        if args.limit and n_loaded >= args.limit:
            print(f"--limit {args.limit} reached")
            break
        n_loaded += 1
        bbox = (g["X"].min() - buf, g["Y"].min() - buf,
                g["X"].max() + buf, g["Y"].max() + buf)
        try:
            ds = cube.load_window(dc, bbox, int(g["win_start"].iloc[0]),
                                  int(g["win_end"].iloc[0]), args.resolution,
                                  dask_chunks=chunks)
            if ds is None:
                raise ValueError("no datasets in window")
            idx, _ = cube.to_indices(ds)
            # materialise the whole cell at once: this is where the S3 reads parallelise
            idx = idx[VARS].compute() if hasattr(idx[VARS], "compute") else idx[VARS]
        except Exception as e:                  # a bad cell must not end the run
            for _, site in g.iterrows():
                man.append(dict(site_id=site.site_id, cell=cell, year=int(year),
                                error=f"load: {type(e).__name__}: {e}"))
            print(f"  [{gi}/{len(groups)}] {cell} {year}: LOAD FAILED -- {e}", flush=True)
            continue

        times = np.asarray(idx.time.values)
        sensors = np.asarray(idx.sensor.values)
        for _, site in g.iterrows():
            try:
                centre, window = crop(idx, site, args.patch)
                for key, df in series_rows(site.site_id, times, sensors,
                                           centre, window).items():
                    if len(df):
                        parts[key].append(df)
                stats = cube.observation_stats(window["ndvi"])
                man.append(dict(
                    site_id=site.site_id, X=site.X, Y=site.Y, lon=site.lon, lat=site.lat,
                    year=int(site.year), win_start=int(site.win_start),
                    win_end=int(site.win_end), cell=site.cell,
                    n_obs_center=int(np.isfinite(
                        np.asarray(centre["ndvi"].values, dtype="float32")).sum()),
                    n_obs_mean5x5=stats["n_obs"],
                    max_doy_gap=stats["max_doy_gap"], mean_doy_gap=stats["mean_doy_gap"],
                    frac_valid_px=stats["frac_valid_px"],
                    n_obs_per_year=json.dumps(stats["n_obs_per_year"]),
                    n_years_with_obs=len(stats["n_obs_per_year"]),
                    patch_y=window.sizes["y"], patch_x=window.sizes["x"],
                    sensors=",".join(sorted(set(sensors.tolist()))),
                ))
            except Exception as e:
                man.append(dict(site_id=site.site_id, cell=cell, year=int(year),
                                error=f"{type(e).__name__}: {e}"))
        ok_n = sum("error" not in r or pd.isna(r.get("error")) for r in man)
        print(f"  [{gi}/{len(groups)}] {cell} {year}: {len(g)} sites -- {ok_n:,} written, "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        if n_loaded % args.flush_every == 0:
            flush()

    flush()
    mandf = pd.DataFrame(man)
    ok = mandf[mandf["error"].isna()] if "error" in mandf else mandf
    print(f"\n{len(ok):,} sites written of {len(sites):,}")
    if "n_obs_mean5x5" in ok and len(ok):
        print(f"observations per site (5x5): median {ok.n_obs_mean5x5.median():.0f}, "
              f"min {ok.n_obs_mean5x5.min()}, max {ok.n_obs_mean5x5.max()}")
        print("\nby region, median observations:")
        by_lat = ok.assign(band=(ok.lat / 5).round() * 5).groupby("band").agg(
            n=("site_id", "size"), obs=("n_obs_mean5x5", "median"))
        print(by_lat.to_string())
    (outdir / "extraction.json").write_text(json.dumps(
        dict(vars(args)) | {
            "n_sites": int(len(sites)), "n_loads": len(groups),
            "n_written": int(len(ok)), "vars": VARS, "px": PX,
            "envelope": {"lon": [float(sites.lon.min()), float(sites.lon.max())],
                         "lat": [float(sites.lat.min()), float(sites.lat.max())],
                         "years": sorted(sites.year.unique().tolist())},
        }, indent=2, default=str))

    # a gateway cluster keeps its pods until told otherwise, so it must not be left orphaned
    if cluster is not None and args.gateway:
        cluster.shutdown()
        print("gateway cluster shut down")
    print(f"\n  -> {outdir}/  ({len(VARS) * len(PX)} series tables, manifest.csv, "
          f"extraction.json)")


if __name__ == "__main__":
    main()
