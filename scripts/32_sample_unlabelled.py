#!/usr/bin/env python3
"""Sample unlabelled 5x5 windows across the study region, for masked-autoencoder pretraining.

RUNS ON THE MACHINE WITH DATA CUBE CHILE. Everything else in this project is now local; this
is the one step that is not.

Why this exists. The 135,250 pixel curves already on disk come from the 25-pixel windows of
the 1,082 labelled plots, and they are almost redundant with the plots themselves:

    variance between plots        91.5 %
    variance within a plot         8.5 %
    correlation, pixels of the same plot   +0.844
    correlation, pixels of different plots +0.258
    participation dimension, 135,250 pixel curves   1.3
    participation dimension,   1,082 plot curves    1.2

125x more images spanning 1.08x more of the space. A masked autoencoder trained on them sees
about a thousand distinct objects repeated twenty-five times. Coverage, not count, is what is
missing -- and coverage is exactly what sampling away from the plots buys.

ON URBAN, CROPS AND PLANTATIONS. Two defensible positions, and the argument does not settle
it:

* **exclude them** -- fine-tuning is on native vegetation, and a representation that spends
  capacity on double-cropped irrigated fields is spending it on phenology the labelled set
  never contains;
* **include them** -- the labelled manifold is extraordinarily narrow (participation
  dimension 1.2 out of 52). Cropland has structure native vegetation does not: abrupt
  harvest drops, two peaks a year, bare-soil troughs. A representation that can encode those
  is strictly richer, and the regression head is what specialises.

So this script does not choose. It **stratifies and labels**: every sample carries its
`landcover_chile_2014` class, so one extraction supports pretraining on natural cover only,
on everything, or on any mixture -- and the comparison between them becomes an ablation
rather than an assumption. Deciding now, by argument, would bake in an answer the data can
give for free.

WHAT IS HELD FIXED. The pretraining images have to be the same *kind* of object as the
fine-tuning ones or the transfer is confounded with a domain shift: same 5x5 window at 30 m,
same causal 3-year window ending at a census-like year, same five indices, same cloud
masking, same `group_by="solar_day"`. The sampling of *where* is the only thing that changes.

Usage:
    python scripts/32_sample_unlabelled.py --n 4000 --dry-run
    python scripts/32_sample_unlabelled.py --n 4000 --out data/derived/unlabelled
    python scripts/32_sample_unlabelled.py --n 8000 --strata climate --natural-only
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

#: Classes of `landcover_chile_2014` treated as anthropic. The legend is not documented in
#: the catalogue, so the codes are resolved at runtime from the measurement metadata and
#: matched by name -- hardcoding integers against an undocumented legend is how a sample of
#: "native vegetation" quietly becomes a sample of orchards.
ANTHROPIC_PATTERNS = ("urban", "ciudad", "industrial", "crop", "cultivo", "agric",
                      "planta", "forestal exótic", "exotic", "mining", "miner")


def resolve_legend(dc) -> dict[int, str]:
    """Class code -> name, from the product's own flag definitions."""
    import datacube.utils.masking as masking          # noqa: F401
    m = dc.list_measurements().loc["landcover_chile_2014"]
    for _, row in m.iterrows():
        flags = row.get("flags_definition") or {}
        for spec in flags.values():
            vals = spec.get("values")
            if isinstance(vals, dict) and len(vals) > 3:
                return {int(k): str(v) for k, v in vals.items()}
    return {}


def is_anthropic(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in ANTHROPIC_PATTERNS)


def plot_envelope(derived: str) -> dict:
    """The region and the causal-window years the labelled plots actually occupy.

    Sampling has to cover the same environmental space, not the country: a pretraining pool
    drawn uniformly over Chile would be mostly Atacama and Patagonia, neither of which the
    labelled set contains.
    """
    p = pd.read_parquet(Path(derived) / "plots_subset.parquet")
    return dict(
        x=(float(p["X"].min()), float(p["X"].max())),
        y=(float(p["Y"].min()), float(p["Y"].max())),
        years=sorted(p["Year"].astype(int).unique().tolist()),
        n_plots=len(p),
    )


def sample_points(env: dict, n: int, strata: str, seed: int,
                  derived: str) -> pd.DataFrame:
    """Candidate centres, either uniform over the envelope or stratified.

    ``strata="climate"`` matches the joint distribution of elevation and latitude that the
    plots occupy, rather than the bounding box. Uniform sampling of a bounding box
    over-represents whatever is geometrically large, which here is the dry north and the
    high Andes -- both nearly absent from the labelled set.
    """
    rng = np.random.default_rng(seed)
    if strata == "uniform":
        return pd.DataFrame({
            "X": rng.uniform(*env["x"], n),
            "Y": rng.uniform(*env["y"], n),
            "Year": rng.choice(env["years"], n),
        })

    plots = pd.read_parquet(Path(derived) / "plots_subset.parquet")
    # resample plot locations and jitter well beyond the 150 m window, so a sample is never
    # the same ground as the plot it was seeded from but stays in its environmental setting
    idx = rng.integers(0, len(plots), n)
    jitter_km = 5.0
    return pd.DataFrame({
        "X": plots["X"].to_numpy()[idx] + rng.uniform(-jitter_km, jitter_km, n) * 1000,
        "Y": plots["Y"].to_numpy()[idx] + rng.uniform(-jitter_km, jitter_km, n) * 1000,
        "Year": plots["Year"].to_numpy()[idx].astype(int),
        "seed_plot": plots["PlotObservationID"].to_numpy()[idx],
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=4000, help="windows to sample")
    p.add_argument("--strata", default="climate", choices=["climate", "uniform"])
    p.add_argument("--natural-only", action="store_true", dest="natural_only",
                   help="drop anthropic classes at extraction. Prefer NOT to: the class is "
                        "recorded either way, so keeping them lets the choice be an "
                        "ablation instead of an assumption")
    p.add_argument("--patch", type=int, default=5)
    p.add_argument("--window", type=int, default=3, help="years, as in scripts/01")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--cell-km", type=float, default=10.0, dest="cell_km",
                   help="side of the grouping cell. Loads are grouped by (cell, year) as in "
                        "scripts/02; ungrouped, n=4000 would be ~8000 dc.load calls. The "
                        "cell size trades calls against peak memory, because the whole cell "
                        "is materialised at once (measured for n=4000: 20 km -> 261 loads "
                        "and 2.2 GB, 10 km -> 490 and 0.6 GB, 5 km -> 957 and 0.1 GB)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="data/derived/unlabelled")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    env = plot_envelope(args.derived)
    print(f"labelled envelope: X {env['x'][0]:.0f}..{env['x'][1]:.0f}  "
          f"Y {env['y'][0]:.0f}..{env['y'][1]:.0f}  years {min(env['years'])}"
          f"..{max(env['years'])}  ({env['n_plots']} plots)")

    pts = sample_points(env, args.n, args.strata, args.seed, args.derived)
    pts["win_start"] = pts["Year"] - (args.window - 1)
    pts["win_end"] = pts["Year"]
    pts["sample_id"] = [f"U{i:05d}" for i in range(len(pts))]
    # same grouping key as scripts/01, so the loads batch exactly as the labelled extraction
    # does. Ungrouped, n=4000 means 4000 `dc.load` calls for the imagery plus 4000 for the
    # land cover; grouped, it is one per (cell, year) pair.
    km = args.cell_km * 1000
    pts["cell"] = ((pts["X"] / km).round().astype(int).astype(str) + "_"
                   + (pts["Y"] / km).round().astype(int).astype(str))
    groups = list(pts.groupby(["cell", "Year"], sort=True))
    print(f"{len(pts)} candidate windows, strata={args.strata}, "
          f"{len(groups)} loads (cell x year)")

    if args.dry_run:
        print("\n" + pts.head().to_string(index=False))
        print(f"\nsamples per load: median {pts.groupby(['cell', 'Year']).size().median():.0f}, "
              f"max {pts.groupby(['cell', 'Year']).size().max()}")
        print("\nwould then, on the datacube machine:")
        print("  1. read landcover_chile_2014 over each cell and record the class per centre")
        print(f"  2. load the causal window per cell and compute the five indices")
        print(f"  3. crop {args.patch}x{args.patch} pixels around each centre -- the same "
              "geometry as the labelled plots")
        print("  4. write one .nc per sample, same layout as data/derived/phenology/")
        return

    import datacube
    from datacube.utils.aws import configure_s3_access
    from biodiv import cube as cubemod

    # mandatory on the Data Observatory: the buckets are requester-pays and without this
    # `dc.load` returns empty rasters rather than failing
    configure_s3_access(aws_unsigned=False, requester_pays=True)
    dc = datacube.Datacube(app="unlabelled_pretrain_pool")

    legend = resolve_legend(dc)
    print(f"landcover legend: {len(legend)} classes"
          + (f", e.g. {list(legend.items())[:3]}" if legend else " (not resolved)"))

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    half = args.patch // 2
    buf = args.patch * args.resolution

    def crop(da, x: float, y: float):
        """The `patch`x`patch` pixels around (x, y), by nearest centre.

        Identical to `extract_plot` in `scripts/02`: the pretraining images have to be the
        same *kind* of object as the fine-tuning ones. Writing the whole loaded window
        instead would hand the autoencoder 10x10 images to learn from and then fine-tune it
        on 5x5 ones, so any transfer result would be confounded with a change of geometry.
        """
        iy = int(np.abs(da.y.values - y).argmin())
        ix = int(np.abs(da.x.values - x).argmin())
        return da.isel(y=slice(max(0, iy - half), iy + half + 1),
                       x=slice(max(0, ix - half), ix + half + 1))

    for gi, ((cell, year), g) in enumerate(groups, 1):
        bbox = (g["X"].min() - buf, g["Y"].min() - buf,
                g["X"].max() + buf, g["Y"].max() + buf)
        try:
            lc = dc.load(product="landcover_chile_2014", x=(bbox[0], bbox[2]),
                         y=(bbox[1], bbox[3]), output_crs="EPSG:32719",
                         resolution=(-args.resolution, args.resolution))
            ds = cubemod.load_window(dc, bbox, int(g["win_start"].iloc[0]),
                                     int(g["win_end"].iloc[0]), args.resolution)
            if ds is None:
                raise ValueError("no datasets in window")
            idx, _ = cubemod.to_indices(ds)
            idx = idx.compute() if hasattr(idx, "compute") else idx
        except Exception as e:                  # a bad cell must not end the run
            for _, pt in g.iterrows():
                rows.append(dict(sample_id=pt.sample_id, X=pt.X, Y=pt.Y, year=int(pt.Year),
                                 error=f"load: {type(e).__name__}: {e}"))
            print(f"  [{gi}/{len(groups)}] {cell} {year}: LOAD FAILED -- {e}", flush=True)
            continue

        lcv = lc.to_array().squeeze("variable", drop=True) if lc else None
        ok_cell = 0
        for _, pt in g.iterrows():
            try:
                code = (int(crop(lcv, pt.X, pt.Y).values.ravel()[0])
                        if lcv is not None else -1)
                name = legend.get(code, str(code))
                anthropic = is_anthropic(name)
                if args.natural_only and anthropic:
                    continue
                out = crop(idx, pt.X, pt.Y).assign_attrs(
                    sample_id=pt.sample_id, landcover_code=code, landcover=name,
                    anthropic=int(anthropic), win_start=int(pt.win_start),
                    win_end=int(pt.win_end), X=float(pt.X), Y=float(pt.Y))
                out.to_netcdf(outdir / f"{pt.sample_id}.nc")
                rows.append(dict(sample_id=pt.sample_id, X=pt.X, Y=pt.Y, year=int(pt.Year),
                                 landcover_code=code, landcover=name,
                                 anthropic=int(anthropic),
                                 n_obs=int(out.sizes.get("time", 0)),
                                 patch_y=int(out.sizes.get("y", 0)),
                                 patch_x=int(out.sizes.get("x", 0))))
                ok_cell += 1
            except Exception as e:
                rows.append(dict(sample_id=pt.sample_id, X=pt.X, Y=pt.Y, year=int(pt.Year),
                                 error=f"{type(e).__name__}: {e}"))
        done = sum("error" not in r for r in rows)
        print(f"  [{gi}/{len(groups)}] {cell} {year}: {ok_cell}/{len(g)} -- "
              f"{done} written, {(time.time()-t0)/60:.1f} min", flush=True)

    man = pd.DataFrame(rows)
    man.to_csv(outdir / "manifest.csv", index=False)
    ok = man[man.get("error").isna()] if "error" in man else man
    print(f"\n{len(ok)} windows written of {len(pts)} attempted")
    if "landcover" in ok:
        print("\nby land cover:")
        print(ok.groupby(["landcover", "anthropic"]).size().to_string())
        print(f"\nanthropic share: {100*ok.anthropic.mean():.0f} %")
    (outdir / "sampling.json").write_text(json.dumps(
        dict(vars(args)) | {"envelope": env}, indent=2, default=str))
    print(f"\n  -> {outdir}/  (one .nc per sample, manifest.csv, sampling.json)")
    print("\nThen, on the local machine: adapt scripts/31_pretrain_mae.py to read this "
          "directory,\nand pretrain three ways -- natural only, everything, and natural "
          "plus a capped share of\nanthropic -- so the choice is measured rather than assumed.")


if __name__ == "__main__":
    main()
