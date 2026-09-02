#!/usr/bin/env python3
"""Multitemporal facet maps from the deployed 2D-CNN+MAE ensemble, tile by tile.

For every tile and every target year ``y`` in ``--years``: load the clear-sky kNDVI
observations of the tile for ``y-2 .. y`` from Data Cube Chile, build the 100-step raw
series per pixel exactly as for the training plots (`biodiv.mapinfer`), fold it into the
serpentine image, attach the centre-pixel topography (Copernicus GLO-30, same derivatives
as `scripts/03`) and the two constant covariates (plot area, recording protocol), run the
five all-data seed checkpoints, back-transform and average, and write one GeoTIFF per
(tile, year) with the seven facets plus quality layers.

The Landsat archive is read ONCE per tile for the whole ``years[0]-2 .. years[-1]`` span
and every year's window is cut from it in memory: 27 target years share 29 archive years,
so per-year loading would read the same scenes ~3x over.

Decisions that are not defaults but recorded choices (see docs/21_map_inference_spec.md):
``--area-m2`` and ``--stratum`` (the non-mappable covariates), ``--mask`` (native
vegetation from MapBiomas, nearest annual map), the tile size, and the smearing residuals.

Usage:
    # pilot: one 10 km tile around a plot cluster, all years, local dask
    python scripts/73_map_inference.py --bbox -72.50 -36.10 -72.35 -35.95 --years 2000-2026 \\
        --workers 4 --out results/maps --tag pilot
    # full extent from the plot envelope, gateway cluster, resumable
    python scripts/73_map_inference.py --from-plots --margin-km 20 --tile-km 10 --dry-run
    python scripts/73_map_inference.py --from-plots --margin-km 20 --workers 16 --gateway --resume

Requires BIODIV_UNIFIED=1 and BIODIV_CURVES=_raw100 only for ``--oof-csv``/range clipping
(they load the unified target tables); the model itself needs neither.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mb          # noqa: E402
from biodiv import mapinfer as mi           # noqa: E402

UTM = "EPSG:32719"
DEFAULT_CKPT = ("results/models_unified/"
                "C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL_alldata/final")
DEFAULT_OOF = ("results/models_unified/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr/"
               "kfold5_block20_unified/oof_predictions.csv")


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def to_utm(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    return tr.transform(np.asarray(lon, float), np.asarray(lat, float))


def to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    tr = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    return tr.transform(np.asarray(x, float), np.asarray(y, float))


def extent_from_bbox(bbox_ll: list[float]) -> tuple[float, float, float, float]:
    w, s, e, n = bbox_ll
    xs, ys = to_utm(np.array([w, e, w, e]), np.array([s, s, n, n]))
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def extent_from_plots(derived: Path, margin_km: float) -> tuple[float, float, float, float]:
    p = pd.read_parquet(derived / "plots_unified.parquet")
    xs, ys = to_utm(p["lon"].to_numpy(), p["lat"].to_numpy())
    m = margin_km * 1000
    return float(xs.min() - m), float(ys.min() - m), float(xs.max() + m), float(ys.max() + m)


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------------------
# masks and output
# --------------------------------------------------------------------------------------

def native_mask(year: int, template, crs: str = UTM) -> tuple[np.ndarray, dict]:
    """Native-vegetation mask on the tile grid from the nearest MapBiomas annual map."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    path, used, delta = mb.year_map(year)
    x, y = template.x.values, template.y.values
    res = float(abs(x[1] - x[0]))
    # bounds in lon/lat with a margin, so the reprojection has full coverage
    xs = np.array([x.min() - res, x.max() + res, x.min() - res, x.max() + res])
    ys = np.array([y.min() - res, y.min() - res, y.max() + res, y.max() + res])
    lons, lats = to_lonlat(xs, ys)
    pad = 0.01
    with rasterio.open(path) as src:
        win = from_bounds(lons.min() - pad, lats.min() - pad, lons.max() + pad,
                          lats.max() + pad, src.transform)
        arr = src.read(1, window=win)
        src_tr = src.window_transform(win)
        src_crs = src.crs
    dst = np.zeros((len(y), len(x)), dtype=arr.dtype)
    dst_tr = grid_transform(template)
    reproject(arr, dst, src_transform=src_tr, src_crs=src_crs, dst_transform=dst_tr,
              dst_crs=crs, resampling=Resampling.nearest)
    native = np.isin(dst, list(mb.NATIVE))
    return native, dict(map_year=int(used), map_delta=int(delta), map_path=str(path))


def grid_transform(template):
    from rasterio.transform import Affine
    x, y = template.x.values, template.y.values
    res = float(abs(x[1] - x[0]))
    return Affine(res, 0.0, float(x.min()) - res / 2, 0.0, -res, float(y.max()) + res / 2)


def write_tile_year(path: Path, template, layers: dict[str, np.ndarray], tags: dict,
                    crs: str = UTM) -> None:
    import rasterio
    names = list(layers)
    h, w = layers[names[0]].shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=len(names),
                       dtype="float32", crs=crs, transform=grid_transform(template),
                       nodata=np.nan, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3, BIGTIFF="IF_SAFER") as dst:
        for i, n in enumerate(names, 1):
            dst.write(np.asarray(layers[n], np.float32), i)
            dst.set_band_description(i, n)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                   help="lon/lat extent (WGS84)")
    g.add_argument("--from-plots", action="store_true", dest="from_plots",
                   help="extent = envelope of plots_unified.parquet (+ --margin-km)")
    g.add_argument("--tiles-file", default=None, dest="tiles_file",
                   help="csv with tile_id,xmin,ymin,xmax,ymax (UTM 19S) to run")
    p.add_argument("--margin-km", type=float, default=20.0, dest="margin_km")
    p.add_argument("--tile-km", type=float, default=10.0, dest="tile_km")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--years", default="2000-2026")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT, dest="ckpt_dir")
    p.add_argument("--area-m2", type=float, default=500.0, dest="area_m2",
                   help="plot area held constant for every pixel (median of the count pool)")
    p.add_argument("--stratum", default="basal", choices=["cover", "counts", "presence", "basal"],
                   help="recording-protocol indicator held constant (Living Trees = basal)")
    p.add_argument("--mask", default="mapbiomas", choices=["mapbiomas", "none"])
    p.add_argument("--oof-csv", default=DEFAULT_OOF, dest="oof_csv",
                   help="oof_predictions.csv of the same config under kfold5_block20_unified, "
                        "for Duan smearing (default: the block20 run of the deployed config). "
                        "Pass an empty string to disable; the plain inverse under-predicts the "
                        "upper tail of TD0 by about 0.25 R2, so disabling is for diagnostics only")
    p.add_argument("--no-clip", action="store_true", dest="no_clip",
                   help="do not clip predictions to the observed training range")
    p.add_argument("--workers", type=int, default=4, help="dask workers (0 = no dask)")
    p.add_argument("--gateway", action="store_true", help="EASI dask-gateway cluster")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0, help="run only the first N tiles")
    p.add_argument("--jobs", type=int, default=1,
                   help="tile-level parallelism: N worker processes, each taking whole tiles "
                        "end to end. The CNN is 95%% of a tile-year and scales badly on "
                        "threads (16 threads buy only 2.9x), so N single-threaded processes "
                        "beat one many-threaded one by ~5.6x per core.")
    p.add_argument("--manifest", default="manifest.csv",
                   help="manifest filename inside --out/--tag; --jobs gives each worker its "
                        "own and merges them, which avoids locking a shared append")
    p.add_argument("--torch-threads", type=int, default=0, dest="torch_threads",
                   help="0 = leave torch alone; workers under --jobs are given 1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--out", default="results/maps")
    p.add_argument("--tag", default="run")
    args = p.parse_args()

    if args.torch_threads:
        import torch
        torch.set_num_threads(args.torch_threads)

    derived = Path(args.derived)
    years = parse_years(args.years)
    tile_m = args.tile_km * 1000
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # ---- tiles --------------------------------------------------------------------
    if args.tiles_file:
        tiles = pd.read_csv(args.tiles_file)
    else:
        ext = (extent_from_bbox(args.bbox) if args.bbox
               else extent_from_plots(derived, args.margin_km))
        tiles = mi.tile_grid(ext, tile_m, args.resolution)
    if args.limit:
        tiles = tiles.head(args.limit)
    tiles.to_csv(out / "tiles.csv", index=False)
    print(f"{len(tiles)} tiles of {args.tile_km:g} km, years {years[0]}-{years[-1]} "
          f"({len(years)}), archive span {years[0] - 2}-{years[-1]}")

    # ---- model ----------------------------------------------------------------------
    ckpts = sorted(Path(args.ckpt_dir).glob("model_seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir}")
    ens = mi.FacetEnsemble(ckpts, device=args.device)
    targets = ens.targets
    print(f"ensemble: {len(ckpts)} seeds, targets {targets}, "
          f"context {ens.context_columns}")

    resid = y_train = None
    if args.oof_csv:
        if not Path(args.oof_csv).exists():
            raise SystemExit(f"--oof-csv not found: {args.oof_csv}. The maps must be produced "
                             "with Duan smearing (the plain inverse loses ~0.25 R2 on TD0); "
                             "fetch the csv or pass --oof-csv '' deliberately.")
        resid = mi.oof_residuals_scaled(args.oof_csv, ens.members[0].scaler, targets)
        print(f"smearing: residuals from {args.oof_csv} "
              f"({np.isfinite(resid).sum(axis=0).tolist()} usable per target)")
    else:
        print("WARNING: smearing disabled -- diagnostic run only, not for publication")
    if not args.no_clip:
        try:
            y_train = mi.training_targets(derived, targets)
            print("range clipping: training targets loaded")
        except Exception as e:                      # noqa: BLE001
            print(f"range clipping unavailable ({type(e).__name__}: {e}); predictions unclipped")
    perm = mi.serpentine_perm(mi.NGS)

    meta = dict(tag=args.tag, years=f"{years[0]}-{years[-1]}", tile_km=args.tile_km,
                resolution=args.resolution, area_m2=args.area_m2, stratum=args.stratum,
                mask=args.mask, ckpt_dir=str(args.ckpt_dir), n_seeds=len(ckpts),
                smearing="oof" if resid is not None else "none",
                clip=y_train is not None, targets=targets,
                # Declared, not corrected (docs/21 section 6): the ensemble predicts a
                # conditional mean, so the map's upper tail is compressed relative to the
                # plot observations. Carried on every tile so the caveat cannot be separated
                # from the raster.
                caveat="predicted values are conditional means and under-disperse the upper "
                       "tail; use as a relative surface")
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    if args.dry_run:
        print(json.dumps(meta, indent=1))
        print(tiles.head(10).to_string(index=False))
        return

    if args.jobs > 1:
        fan_out(args, tiles, out)
        return

    # ---- datacube + dask ----------------------------------------------------------------
    import datacube
    from datacube.utils.aws import configure_s3_access

    client = cluster = None
    if args.workers > 0 and args.gateway:
        from dask_gateway import Gateway
        gw = Gateway()
        cluster = gw.new_cluster(gw.cluster_options())
        cluster.scale(args.workers)
        client = cluster.get_client()
        print(f"dask-gateway cluster {cluster.name}, scaling to {args.workers}; "
              f"dashboard {dashboard_url(cluster)}", flush=True)
        client.wait_for_workers(1, timeout=600)
    elif args.workers > 0:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=args.workers, processes=True, threads_per_worker=1)
        client = Client(cluster)
        # Same line the gateway branch prints: on JupyterHub `dashboard_link` resolves through
        # the service proxy, so it is the only form of the URL that is reachable from outside
        # the pod. Without it a local run gives no way to watch the workers.
        print(f"dask: local cluster, {args.workers} workers; "
              f"dashboard {dashboard_url(cluster)}", flush=True)
    # Boot order is not negotiable, same as `scripts/35`: cluster, then
    # configure_s3_access(client=...), then the Datacube. `usgs-landsat` is requester-pays, and
    # without this every worker read comes back RasterioIOError('AccessDenied: Access Denied').
    # Passing `client` is what propagates the setting to the workers; setting it only in the
    # driver process leaves the dask path broken. See `src/biodiv/cube.py`.
    configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)

    chunks = {"time": 1} if args.workers > 0 else None
    dc = datacube.Datacube(app="biodiv-map-inference")

    man_path = out / args.manifest
    done: set[tuple[str, int]] = set()
    if args.resume and man_path.exists():
        m = pd.read_csv(man_path)
        done = set(zip(m["tile_id"], m["year"].astype(int)))
        print(f"resume: {len(done)} (tile, year) already written")

    try:
        for ti, t in enumerate(tiles.itertuples(index=False), 1):
            todo = [y for y in years if (t.tile_id, y) not in done]
            if not todo:
                continue
            t0 = time.time()
            bbox = (t.xmin, t.ymin, t.xmax, t.ymax)
            da = mi.load_kndvi(dc, bbox, years[0] - 2, years[-1], resolution=args.resolution,
                               dask_chunks=chunks)
            if da is None:
                print(f"[{ti}/{len(tiles)}] {t.tile_id}: no Landsat observations, skipped")
                _append(man_path, [dict(tile_id=t.tile_id, year=y, status="no_data")
                                   for y in todo])
                continue
            da = da.compute() if hasattr(da.data, "compute") else da
            t_load = time.time() - t0
            times = da.time.values
            ny, nx = da.sizes["y"], da.sizes["x"]
            obs = da.values.reshape(len(times), ny * nx)
            template = da.isel(time=0)
            lon_c, lat_c = to_lonlat(np.array([(t.xmin + t.xmax) / 2]),
                                     np.array([(t.ymin + t.ymax) / 2]))
            terrain = mi.load_terrain(dc, template, float(lat_c[0]), resolution=args.resolution)
            topo = {k: v.reshape(-1) for k, v in terrain.items()}
            ctx = mi.context_frame(ens.context_columns, topo, args.area_m2, args.stratum)
            print(f"[{ti}/{len(tiles)}] {t.tile_id}: {len(times)} dates, {ny}x{nx} px, "
                  f"load {t_load:.0f}s", flush=True)

            rows = []
            for y in todo:
                ty = time.time()
                curves, n_obs, lo, hi = mi.year_curves(times, obs, y)
                if args.mask == "mapbiomas":
                    native, minfo = native_mask(y, template)
                else:
                    native, minfo = np.ones((ny, nx), bool), {}
                nat = native.reshape(-1)
                run = np.isfinite(curves).all(axis=1) & nat
                pred = np.full((ny * nx, len(targets)), np.nan, np.float32)
                if run.any():
                    images = mi.images_from_curves(curves[run], perm)
                    pred[run] = ens.predict(images, ctx.iloc[np.flatnonzero(run)],
                                            resid_scaled=resid, y_train=y_train,
                                            batch=args.batch)
                layers = {tg_: pred[:, j].reshape(ny, nx) for j, tg_ in enumerate(targets)}
                layers["n_obs"] = n_obs.reshape(ny, nx).astype(np.float32)
                layers["span_days"] = np.where(np.isfinite(curves).all(axis=1), hi - lo,
                                               np.nan).reshape(ny, nx).astype(np.float32)
                layers["native"] = native.astype(np.float32)
                tags = dict(meta, year=y, window=f"{y - 2}-01-01/{y}-12-31",
                            grid_first_day=lo, grid_last_day=hi, **minfo)
                fname = out / f"{t.tile_id}_{y}.tif"
                write_tile_year(fname, template, layers, tags)
                rows.append(dict(tile_id=t.tile_id, year=y, status="ok",
                                 n_px=ny * nx, n_native=int(nat.sum()),
                                 n_pred=int(run.sum()), n_dates_window=int(
                                     mi.window_mask(times, y).sum()),
                                 grid_first=lo, grid_last=hi, seconds=round(time.time() - ty, 1),
                                 load_seconds=round(t_load, 1), file=str(fname)))
            _append(man_path, rows)
            print(f"    {len(todo)} years in {time.time() - t0:.0f}s "
                  f"(pred px/yr median {int(np.median([r['n_pred'] for r in rows]))})",
                  flush=True)
    finally:
        if cluster is not None and args.gateway:
            cluster.shutdown()
            print("gateway cluster shut down")
        elif client is not None:
            client.close()
    print(f"\n-> {out}/ (manifest.csv, run.json, tiles.csv, <tile>_<year>.tif)")


def dashboard_url(cluster) -> str:
    """Absolute, clickable dashboard URL.

    ``cluster.dashboard_link`` honours ``distributed.dashboard.link``, which EASI sets to
    ``{JUPYTERHUB_SERVICE_PREFIX}proxy/{port}/status`` -- a *relative* path, because the pod
    has no idea what its external hostname is (``JUPYTERHUB_PUBLIC_URL`` and
    ``JUPYTERHUB_HOST`` are both empty here). A relative path is useless to anyone reading
    the log outside the browser session, so take the scheme+host from the one place that
    does know it, ``gateway.public-address`` in /etc/dask/dask.yaml, and prepend it.

    Falls back to whatever dask gave us if that key is absent on some other deployment.
    """
    link = str(getattr(cluster, "dashboard_link", "") or "")
    if not link or link.startswith("http"):
        return link
    try:
        import dask.config
        from urllib.parse import urlsplit
        pub = dask.config.get("gateway.public-address", "") or ""
        u = urlsplit(pub)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}{link}"
    except Exception:                                # noqa: BLE001
        pass
    return link


def fan_out(args, tiles: pd.DataFrame, out: Path) -> None:
    """Run whole tiles across ``args.jobs`` worker processes, then merge their manifests.

    Each worker is this same script re-invoked with ``--jobs 1``, its own slice of the tile
    grid and its own manifest file. Three reasons for separate processes rather than threads
    or a shared dask cluster:

    * the CNN is ~95 % of a tile-year and scales badly on threads (16 torch threads buy 2.9x
      on 16 cores, 18 % efficiency), so one single-threaded process per core is ~5.6x more
      core-efficient than one many-threaded process;
    * with N tile-processes the S3 concurrency already comes from the processes, so each
      worker uses plain synchronous reads (``--workers 0``) and no dask client is created --
      a client per worker would multiply connections for no gain;
    * a per-worker manifest merged at the end is safe by construction, where concurrent
      appends to one csv would need locking.

    ``--resume`` is honoured: each worker skips the (tile, year) pairs already in the merged
    manifest, which is read in before the split.
    """
    import subprocess

    parts_dir = out / "_parts"
    parts_dir.mkdir(exist_ok=True)
    merged = out / args.manifest

    # Round-robin rather than contiguous blocks: tiles differ in native cover and therefore
    # in cost, and interleaving keeps the workers from finishing at wildly different times.
    assignments = [tiles.iloc[i::args.jobs] for i in range(args.jobs)]
    assignments = [a for a in assignments if len(a)]
    print(f"--jobs {args.jobs}: {len(tiles)} tiles -> "
          f"{[len(a) for a in assignments]} per worker", flush=True)

    procs, part_files = [], []
    for i, part in enumerate(assignments):
        tf = parts_dir / f"tiles_{i}.csv"
        part.to_csv(tf, index=False)
        man_i = f"_parts/manifest_{i}.csv"
        part_files.append(out / man_i)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--tiles-file", str(tf), "--out", str(args.out), "--tag", args.tag,
               "--years", args.years, "--resolution", str(args.resolution),
               "--tile-km", str(args.tile_km), "--derived", str(args.derived),
               "--ckpt-dir", str(args.ckpt_dir), "--area-m2", str(args.area_m2),
               "--stratum", args.stratum, "--mask", args.mask, "--device", args.device,
               "--batch", str(args.batch), "--manifest", man_i,
               "--jobs", "1", "--workers", "0", "--torch-threads", "1"]
        if args.oof_csv:
            cmd += ["--oof-csv", args.oof_csv]
        if args.no_clip:
            cmd.append("--no-clip")
        if args.resume:
            cmd.append("--resume")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        log = (parts_dir / f"worker_{i}.log").open("w")
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
                      log))

    t0 = time.time()
    failed = []
    for i, p, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append((i, rc))
        print(f"  worker {i} exited {rc} after {time.time() - t0:.0f}s", flush=True)

    frames = [pd.read_csv(f) for f in part_files if f.exists()]
    if frames:
        man = pd.concat(frames, ignore_index=True).sort_values(["tile_id", "year"])
        man.to_csv(merged, index=False)
        print(f"merged {len(frames)} worker manifests -> {merged} ({len(man)} rows)")
    if failed:
        raise SystemExit(f"workers failed: {failed}; logs in {parts_dir}")


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
