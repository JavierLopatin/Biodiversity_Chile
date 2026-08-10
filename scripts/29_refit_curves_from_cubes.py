#!/usr/bin/env python3
"""Re-fit the PhenoShape curves from the stored `.nc` cubes, with the periodicity fix.

Why this exists. `scripts/02_extract_phenology.py` builds the cubes *from Data Cube Chile*
and cannot read them back: its `main()` goes through `cube.load_window(dc, ...)`. But the
cubes carry `obs_{index}` -- the raw Landsat observations -- precisely so the fit can be
redone without the datacube. This script is that path.

What changes. The curves shipped in `phenoshape_by_index.parquet` were fitted before
`PhenoSensing` commit `eb2dff8`, so they carry the year-boundary artefact documented in
`docs/13_phenology_year_boundary.md`: a step between DOY 364 and DOY 1 of ~4x the typical
week-to-week change, negative in 67-71% of plots. Measured on six real cubes, fixing the
moving average alone shrinks it ~1.5x, which is the intended amount: the rest of the step
is a real interannual signal and must survive. `--recon harmonic` shrinks it 10x by forcing
`f(1) = f(365)`, and in doing so deletes that signal -- do not use it here.

TWO TRAPS, both hit while developing this and both handled below:

1. **The cubes lost the per-observation DOY.** `doy` became the 52-step dimension of
   `phenoshape` and overwrote the `time`-indexed coordinate the observations need, so
   `PhenoShape` aborts with `AttributeError: 'DataArray' object has no attribute 'doy'`.
   Recovered here from `time.dt.dayofyear`. Script 02 should store it under another name so
   the collision does not recur.

2. **The stored curve is trough-anchored** (`doy_anchor: 108`) and a fresh fit is not, so
   "last element vs first element" measures a benign mid-year transition on one and the year
   boundary on the other. `year_boundary_step` locates the boundary by DOY instead.

Outputs, all under `--out` (default `data/derived`), suffixed so nothing is overwritten
until the results have been looked at:

    phenoshape_by_index{suffix}.parquet     plot-level curves, per index, per px level
    phenoshape_pixels{suffix}.parquet       the 25 pixels, per index
    phenoshape_doy_grid{suffix}.parquet     step -> DOY
    refit_report{suffix}.csv                per plot: boundary step before and after

Usage:
    python scripts/29_refit_curves_from_cubes.py --dry-run        # what it would do
    python scripts/29_refit_curves_from_cubes.py --limit 6        # the sample cubes
    python scripts/29_refit_curves_from_cubes.py --workers 8
    python scripts/29_refit_curves_from_cubes.py --recon linear   # fix only, no harmonic
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: `phenosensing` is used from its working tree: the fix lives there and is not pip-installed.
#: Overridable with --phenosensing for a machine that has it on the path already.
DEFAULT_PHENO = Path("/mnt/rapidita_4T/GitHub/PhenoSensing")

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
NGS = 52
ROLL = 5

#: The commit that fixed `_moving_average` and added the `harmonic` reconstructor. Recorded
#: in the report so a parquet can always be traced to the code that produced it.
PHENO_FIX_COMMIT = "eb2dff8"


# --------------------------------------------------------------------------------------
# the two traps
# --------------------------------------------------------------------------------------

def observations(ds: xr.Dataset, index: str) -> xr.DataArray:
    """`obs_{index}` with the `doy` and `year` coordinates PhenoShape needs.

    The saved cube has `doy` as the 52-step output dimension, which shadowed the
    per-observation coordinate. Rebuilding it from `time` is exact -- `time` is the
    acquisition timestamp, and script 02 derived `doy` from it in the first place.
    """
    da = ds[f"obs_{index}"]
    doy = ds["time"].dt.dayofyear.values
    year = ds["year"].values if "year" in ds.coords else ds["time"].dt.year.values
    return da.assign_coords(doy=("time", doy), year=("time", year))


def year_boundary_step(curves: np.ndarray, doy: np.ndarray) -> tuple[float, float]:
    """Median |jump| across pixels at the transition where DOY crosses the year end.

    Returns ``(boundary_jump, typical_step)``. The boundary is found from the DOY grid, not
    assumed to be the array wrap: a trough-anchored curve has it in the middle.
    """
    f = np.asarray(curves, dtype=float)
    f = f.reshape(f.shape[0], -1).T
    d = np.asarray(doy)
    diffs = np.diff(d)
    if diffs.size == 0 or diffs.min() > 0:
        jump = np.abs(f[:, 0] - f[:, -1])
    else:
        j = int(np.argmin(diffs))
        jump = np.abs(f[:, j + 1] - f[:, j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmedian(jump)), float(np.nanmedian(np.abs(np.diff(f, axis=1))))


# --------------------------------------------------------------------------------------
# per-plot work
# --------------------------------------------------------------------------------------

def refit_plot(path: str, recon: str, n_harmonics: int, pheno_path: str,
               roll_mode: str = "shrink", ngs: int = NGS) -> dict:
    """Re-fit every index of one cube. Returns curves, pixels and a diagnostic row.

    Runs in a worker process, so it takes plain types and imports `phenosensing` itself.
    """
    sys.path.insert(0, pheno_path)
    import phenosensing  # noqa: F401  (registers the .pheno accessor)

    warnings.filterwarnings("ignore")
    path = Path(path)
    out = {"plot_id": path.stem, "curves": [], "pixels": [], "doy": None, "report": {}}
    kw = {"n_harmonics": n_harmonics} if recon == "harmonic" else {}

    with xr.open_dataset(path) as ds:
        stored = ds["phenoshape"] if "phenoshape" in ds else None
        stored_doy = ds["doy"].values if "doy" in ds.coords else None
        rep = {"plot_id": path.stem, "n_obs": int(ds.sizes.get("time", 0))}

        for index in INDICES:
            if f"obs_{index}" not in ds:
                continue
            da = observations(ds, index)
            shape = da.pheno.PhenoShape(interpolType=recon, rollWindow=ROLL, nGS=ngs,
                                        recon_params=kw or None, rollMode=roll_mode)
            arr = shape.values                       # (doy, y, x)
            out["doy"] = shape["doy"].values.astype(float)

            # plot level: the 5x5 mean and the centre pixel, matching what script 02 wrote
            cy, cx = arr.shape[1] // 2, arr.shape[2] // 2
            flat = arr.reshape(arr.shape[0], -1)
            for px, series in (("mean5x5", np.nanmean(flat, axis=1)),
                               ("center", arr[:, cy, cx])):
                out["curves"].append(
                    {"plot_id": path.stem, "index": index, "px": px,
                     **{f"s{k:02d}": float(v) for k, v in enumerate(series)}})

            # (y, x), not a flat index: `substrates.load_pixel_curves` does
            # set_index(["plot_id", "y", "x"]).sort_index(), so the schema has to match or
            # the reshape to (N, 25, 52) silently scrambles the pixel axis
            ny, nx = arr.shape[1], arr.shape[2]
            for yy in range(ny):
                for xx in range(nx):
                    out["pixels"].append(
                        {"plot_id": path.stem, "index": index, "y": yy, "x": xx,
                         **{f"s{k:02d}": float(v) for k, v in enumerate(arr[:, yy, xx])}})

            new_jump, new_step = year_boundary_step(arr, out["doy"])
            rep[f"{index}_after"] = new_jump
            rep[f"{index}_after_step"] = new_step
            if stored is not None and index in list(stored["index"].values):
                old_jump, old_step = year_boundary_step(
                    stored.sel(index=index).values, stored_doy)
                rep[f"{index}_before"] = old_jump
                rep[f"{index}_before_step"] = old_step

        out["report"] = rep
    return out


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cubes", default="data/derived/phenology")
    p.add_argument("--out", default="data/derived")
    p.add_argument("--suffix", default="_v2",
                   help="appended to every output name; '' overwrites the originals")
    p.add_argument("--recon", default="harmonic",
                   help="reconstructor; 'harmonic' also closes the year, 'linear' relies "
                        "only on the moving-average fix")
    p.add_argument("--n-harmonics", type=int, default=3, dest="n_harmonics")
    p.add_argument("--roll-mode", default="shrink", dest="roll_mode",
                   choices=["shrink", "reflect", "wrap", "legacy"],
                   help="edge handling of the moving average; 'wrap' deletes the "
                        "interannual trend and is here only for reproducing old output")
    p.add_argument("--ngs", type=int, default=NGS,
                   help="steps per cycle. 52 is weekly and was a choice, not a constraint: "
                        "the image a 2-D model sees is sqrt(ngs) on a side, and the paper "
                        "this design comes from flags image size as understudied")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N cubes, for a trial run")
    p.add_argument("--phenosensing", default=str(DEFAULT_PHENO))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cubes = sorted(Path(args.cubes).glob("*.nc"))
    if args.limit:
        cubes = cubes[: args.limit]
    if not cubes:
        raise SystemExit(f"no .nc cubes in {args.cubes}")

    sys.path.insert(0, args.phenosensing)
    import phenosensing
    from phenosensing.reconstruction import list_reconstructors

    print(f"{len(cubes)} cubes  |  reconstructor={args.recon}"
          f"{f' (k={args.n_harmonics})' if args.recon == 'harmonic' else ''}"
          f"  |  nGS={args.ngs} rollWindow={ROLL} rollMode={args.roll_mode}")
    print(f"phenosensing: {Path(phenosensing.__file__).parent}")
    if args.recon not in list_reconstructors():
        raise SystemExit(
            f"{args.recon!r} is not registered. Available: {list_reconstructors()}.\n"
            f"'harmonic' needs PhenoSensing >= {PHENO_FIX_COMMIT}.")
    if args.dry_run:
        print(f"\nwould write, under {args.out}:")
        for n in ("phenoshape_by_index", "phenoshape_pixels", "phenoshape_doy_grid",
                  "refit_report"):
            print(f"  {n}{args.suffix}.{'csv' if n == 'refit_report' else 'parquet'}")
        return

    t0 = time.time()
    curves, pixels, reports, doy_rows, failed = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(refit_plot, str(c), args.recon, args.n_harmonics,
                          args.phenosensing, args.roll_mode, args.ngs): c for c in cubes}
        for i, fut in enumerate(as_completed(futs), 1):
            cube = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                    # one bad cube must not kill the run
                failed.append({"plot_id": cube.stem, "error": f"{type(e).__name__}: {e}"})
                continue
            curves += r["curves"]
            pixels += r["pixels"]
            reports.append(r["report"])
            doy_rows.append({"plot_id": r["plot_id"],
                             **{f"s{k:02d}": int(round(v)) for k, v in enumerate(r["doy"])}})
            if i % 100 == 0 or i == len(cubes):
                print(f"  [{i}/{len(cubes)}] {(time.time()-t0)/60:.1f} min", flush=True)

    doy_rows = sorted(doy_rows, key=lambda r: r["plot_id"])
    if not reports:
        raise SystemExit("every cube failed; nothing written")

    out = Path(args.out)
    sfx = args.suffix
    pd.DataFrame(curves).to_parquet(out / f"phenoshape_by_index{sfx}.parquet", index=False)
    pd.DataFrame(pixels).to_parquet(out / f"phenoshape_pixels{sfx}.parquet", index=False)
    # one row PER PLOT, wide: `xnew = linspace(min(x), max(x))` differs between plots, so a
    # single shared grid would be wrong. Matches the original schema exactly.
    pd.DataFrame(doy_rows).to_parquet(out / f"phenoshape_doy_grid{sfx}.parquet", index=False)
    rep = pd.DataFrame(reports).sort_values("plot_id")
    rep.to_csv(out / f"refit_report{sfx}.csv", index=False)

    print(f"\n{len(rep)} plots refitted in {(time.time()-t0)/60:.1f} min"
          f"{f', {len(failed)} FAILED' if failed else ''}")
    if failed:
        pd.DataFrame(failed).to_csv(out / f"refit_failures{sfx}.csv", index=False)
        for f in failed[:5]:
            print(f"  {f['plot_id']}: {f['error']}")

    # --- did it work? the whole point, printed rather than assumed
    print("\nYear-boundary step, median over plots:\n")
    print(f"  {'index':7s} {'before':>10s} {'after':>10s} {'shrink':>8s} "
          f"{'after / typical step':>21s}")
    for ix in INDICES:
        b, a = f"{ix}_before", f"{ix}_after"
        if a not in rep:
            continue
        av = rep[a].median()
        st = rep[f"{ix}_after_step"].median()
        bv = rep[b].median() if b in rep else np.nan
        shrink = f"{bv/av:.1f}x" if np.isfinite(bv) and av > 0 else "--"
        print(f"  {ix:7s} {bv:10.5f} {av:10.5f} {shrink:>8s} {av/st:20.2f}x")
    print("\nThe ratio should come DOWN but NOT to 1. Part of the boundary step is real: a\n"
          "multi-year composite joins the end of one year to the start of the next, so if\n"
          "productivity changed the two ends genuinely differ. Regressed on the interannual\n"
          "trend the step has slope -1.31, against the -1 the compositing predicts. A ratio\n"
          "near 1 would mean that signal had been erased too -- which is what `--recon\n"
          "harmonic` does. See docs/13_phenology_year_boundary.md.")

    print(f"\n  -> {out}/phenoshape_{{by_index,pixels,doy_grid}}{sfx}.parquet")
    print(f"  -> {out}/refit_report{sfx}.csv")
    print("\nNext: scripts/04_recompute_lsp.py on the new curves, then the model matrix.")


if __name__ == "__main__":
    main()
