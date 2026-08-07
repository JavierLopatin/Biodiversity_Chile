#!/usr/bin/env python3
"""Extract PhenoShape curves and LSP metrics per plot from Data Cube Chile.

Design (see `docs/05_data_acquisition.md`):

- **Causal 3-year window** (`y-2..y`), uniform across all plots. Uniform and not adaptive
  on purpose: if window length depended on cloudiness it would be confounded with latitude,
  elevation and year — that is, with the very gradient being modelled.
- **Observation-level data is stored alongside the curve.** Pooling is a downstream
  decision, not an acquisition one, so substrate A (the year x DOY phenocube) remains
  derivable without returning to the cube.
- **5x5 pixel window** per plot, stored pixel by pixel rather than averaged: this is both
  the augmentation of `docs/03` section 4.3 and the spectral-mixing diagnostic.
- Loads are grouped by (cell, census year) to avoid re-reading the same COG blocks.

Usage:
    python scripts/02_extract_phenology.py --dry-run
    python scripts/02_extract_phenology.py --limit 5
    python scripts/02_extract_phenology.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/jovyan/PhenoSensing")

import phenosensing  # noqa: E402,F401  -- registers the `.pheno` xarray accessor

from biodiv import cube  # noqa: E402

INDICES = cube.INDEX_NAMES
LSP_METRICS = [
    "sos", "pos", "eos", "vsos", "vpos", "veos", "los", "msp", "mau",
    "vmsp", "vmau", "ampl", "ios", "rog", "ros", "sw", "trough", "mos",
]


def clean_for_netcdf(obj):
    """xarray refuses to write when `units` sits in both attrs and encoding."""
    for c in list(obj.coords):
        obj[c].attrs.pop("units", None)
        obj[c].encoding.pop("units", None)
    return obj


def extract_plot(idx: xr.Dataset, plot: pd.Series, patch: int, nGS: int,
                 recon: str, hemisphere: str) -> tuple[xr.Dataset, dict]:
    """Crop the plot window and compute PhenoShape + LSP per pixel and index."""
    half = patch // 2
    iy = int(np.abs(idx.y.values - plot["Y"]).argmin())
    ix = int(np.abs(idx.x.values - plot["X"]).argmin())
    y0, y1 = max(0, iy - half), min(idx.sizes["y"], iy + half + 1)
    x0, x1 = max(0, ix - half), min(idx.sizes["x"], ix + half + 1)
    win = idx.isel(y=slice(y0, y1), x=slice(x0, x1))

    shapes, lsps, stats = {}, {}, {}
    for name in INDICES:
        da = win[name]
        stats[name] = cube.observation_stats(da)
        if stats[name]["n_obs"] < 10:
            continue
        shape = da.pheno.PhenoShape(interpolType=recon, rollWindow=5, nGS=nGS)
        shapes[name] = shape
        lsp = shape.pheno.PhenoLSP(nGS=nGS, hemisphere=hemisphere)
        lsps[name] = lsp

    if not shapes:
        raise ValueError("not enough observations in any index")

    curves = xr.concat([shapes[n] for n in shapes], dim="index").assign_coords(
        index=list(shapes)
    )
    lsp_vars = {}
    for m in LSP_METRICS:
        present = [n for n in lsps if m in lsps[n]]
        if present:
            lsp_vars[f"lsp_{m}"] = xr.concat(
                [lsps[n][m] for n in present], dim="index"
            ).assign_coords(index=present)

    out = xr.Dataset({"phenoshape": curves, **lsp_vars})

    # Raw observation series, for the indices AND the scaled bands they came from. This is
    # what allows substrate A, the sensitivity window, and any index not yet thought of to
    # be derived later without returning to the cube -- SAVI could not be recovered from
    # stored NDVI, which is the mistake this avoids repeating.
    out = out.assign({f"obs_{n}": win[n] for n in INDICES})
    out = out.assign({f"obs_{b}": win[b] for b in cube.BAND_VARS if b in win})

    ref = stats.get("ndvi", next(iter(stats.values())))
    meta = {
        "plot_id": plot["PlotObservationID"],
        "census_year": int(plot["Year"]),
        "win_start": int(plot["win_start"]),
        "win_end": int(plot["win_end"]),
        "n_obs": ref["n_obs"],
        "max_doy_gap": ref["max_doy_gap"],
        "mean_doy_gap": ref["mean_doy_gap"],
        "frac_valid_px": ref["frac_valid_px"],
        "n_obs_per_year": json.dumps(ref["n_obs_per_year"]),
        "n_years_with_obs": len(ref["n_obs_per_year"]),
        "patch_y": win.sizes["y"],
        "patch_x": win.sizes["x"],
        "sensors": ",".join(sorted(set(np.asarray(win.sensor.values).tolist()))),
        "indices_ok": ",".join(sorted(shapes)),
        "plot_size_m2": float(plot["PlotSize_m2"]) if pd.notna(plot["PlotSize_m2"]) else np.nan,
        "n_landsat_px_covered": (
            float(plot["PlotSize_m2"]) / 900.0 if pd.notna(plot["PlotSize_m2"]) else np.nan
        ),
        "metadata_id": plot["metadata_id"],
        "Location": plot["Location"],
        "lat": float(plot["lat"]),
        "lon": float(plot["lon"]),
    }
    for k, v in meta.items():
        out.attrs[k] = v
    return out, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="data/derived/plots_subset.parquet")
    p.add_argument("--out-dir", default="data/derived/phenology")
    p.add_argument("--patch", type=int, default=5, help="window side in pixels")
    p.add_argument("--ngs", type=int, default=52, help="curve steps (52 ~ weekly)")
    p.add_argument("--recon", default="linear", help="PhenoShape reconstructor")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--workers", type=int, default=4,
                   help="Dask LocalCluster workers (0 = no dask)")
    # Anchoring is set to "auto" (per-pixel harmonic phase) on ecological grounds, not on
    # an aggregate score. Central Chile holds vegetation with opposite seasonal phase in the
    # same scene: in an audit of 85 extracted plots, 38 peaked in Nov-Dec but 18 (21%) peaked
    # in May-Aug -- austral winter -- which is the matorral/shrubland pattern of greening on
    # the winter rains and senescing through the summer drought. Circular concentration of
    # the peak was R = 0.42, i.e. several phases coexist.
    #
    # A single global rotation cannot serve both. It mis-anchors the winter-peaking minority
    # systematically, and because that minority is an ecologically distinct community type,
    # the resulting LSP bias would be correlated with the very target being predicted. That
    # is worse than a higher aggregate degenerate rate, which is what an unstratified score
    # over the majority phase would reward:
    #
    #     anchoring   degenerate (sos==pos)   median LOS   IQR LOS
    #     north               13.9%               129         42
    #     south                0.3%               185         27
    #     auto                 9.9%               150         71
    #
    # The wider LOS spread under "auto" is partly real phase diversity, not only noise.
    #
    # Known risk, measured and therefore monitored rather than assumed away: per-pixel
    # anchoring on a weak seasonal signal can fit noise, giving different offsets to
    # neighbouring pixels of the same plot and breaking the 5x5 augmentation. The per-pixel
    # anchor, seasonality strength and aseasonal flag are stored so within-plot coherence
    # can be audited; see scripts/04_recompute_lsp.py.
    #
    # Only the `lsp_*` variables depend on this. `phenoshape` and the raw observations are
    # anchoring-independent, so switching never requires returning to the cube.
    p.add_argument("--hemisphere", default="auto",
                   help="phase anchoring: auto (per-pixel harmonic, default), "
                        "south (global austral rotation) or north (no rotation)")
    p.add_argument("--limit", type=int, default=None, help="process only N cells")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    plots = pd.read_parquet(args.subset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = list(plots.groupby(["cell", "Year"], sort=True))
    print(f"plots: {len(plots)} | loads (cell x year): {len(groups)}")

    if args.dry_run:
        sizes = [len(g) for _, g in groups]
        per_year = plots.groupby("Year").size()
        print(f"plots per load: median {int(np.median(sizes))}, max {max(sizes)}")
        print(f"causal window: y-{plots['win_years'].iloc[0] - 1}..y")
        print("products by census year:")
        for y in sorted(per_year.index):
            prods = cube.products_for(int(y) - 2, int(y))
            print(f"  {y} ({per_year[y]:4d} plots): "
                  f"{[q.replace('_c2l2_sr', '') for q in prods]}")
        return

    import datacube
    from datacube.utils.aws import configure_s3_access

    # Cluster first, then `client=` in configure_s3_access: credentials must propagate to
    # the workers, otherwise every read from a worker returns AccessDenied.
    # n_workers=4 is the recommended default for satellite loads (Dask defaults to 2).
    client = None
    if args.workers > 0:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=args.workers, processes=True,
                               dashboard_address=None)
        client = Client(cluster)
        print(f"dask: {args.workers} workers")

    configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
    dc = datacube.Datacube(app="biodiv_pheno")

    manifest_path = out_dir / "manifest.csv"
    done = set()
    if args.resume and manifest_path.exists():
        done = set(pd.read_csv(manifest_path)["plot_id"])
        print(f"resume: {len(done)} plots already done")

    rows, failures = [], []
    if args.limit:
        groups = groups[: args.limit]

    for gi, ((cell, year), g) in enumerate(groups, 1):
        todo = g[~g["PlotObservationID"].isin(done)]
        if todo.empty:
            continue
        t0 = time.time()
        buf = args.patch * args.resolution
        bbox = (todo["X"].min() - buf, todo["Y"].min() - buf,
                todo["X"].max() + buf, todo["Y"].max() + buf)
        y_end = int(todo["win_end"].iloc[0])
        y_start = int(todo["win_start"].iloc[0])
        try:
            chunks = {"time": 1} if args.workers > 0 else None
            ds = cube.load_window(dc, bbox, y_start, y_end, args.resolution,
                                  dask_chunks=chunks)
            if ds is None:
                raise ValueError("no datasets in window")
            idx, _ = cube.to_indices(ds)
            # Materialise the whole cell at once: this is where the S3 reads parallelise.
            # PhenoShape then works on in-memory numpy (its numba fast path requires that
            # the array carry no dask chunks).
            if chunks:
                idx = idx.compute()
        except Exception as e:
            print(f"[{gi}/{len(groups)}] {cell} {year}: LOAD FAILED -- {e}")
            failures.append(dict(cell=cell, year=int(year), stage="load", error=str(e)))
            continue

        ok = 0
        for _, plot in todo.iterrows():
            try:
                out, meta = extract_plot(idx, plot, args.patch, args.ngs,
                                         args.recon, args.hemisphere)
                clean_for_netcdf(out).to_netcdf(
                    out_dir / f"{plot['PlotObservationID']}.nc"
                )
                rows.append(meta)
                ok += 1
            except Exception as e:
                failures.append(dict(plot_id=plot["PlotObservationID"], stage="extract",
                                     error=f"{type(e).__name__}: {e}",
                                     tb=traceback.format_exc(limit=2)))
        print(f"[{gi}/{len(groups)}] {cell} {year}: {ok}/{len(todo)} plots, "
              f"{ds.sizes['time']} obs, {time.time() - t0:.0f}s")

        if rows:
            pd.DataFrame(rows).to_csv(
                manifest_path, mode="a" if manifest_path.exists() else "w",
                header=not manifest_path.exists(), index=False,
            )
            rows = []

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)
        print(f"\nfailures: {len(failures)} (see failures.csv)")
    print(f"\noutput in {out_dir}/")


if __name__ == "__main__":
    main()
