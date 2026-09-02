#!/usr/bin/env python3
"""Supplementary figure: the tiling of the study extent, and which tiles are worth visiting.

The extent is the plot envelope + 20 km, which at 10 km tiles is 17,019 tiles covering
600 x 2,783 km. Most of that rectangle is ocean, Argentina or Atacama: only about a third of
the tiles contain any MapBiomas native vegetation, and `scripts/73` pays a full ~300 s
Landsat load before it ever consults the mask. Pre-filtering the grid against a decimated
MapBiomas read is therefore not an optimisation, it is the difference between a feasible run
and an infeasible one -- the empty tiles alone would cost ~930 h of pure I/O.

Writes the figure (pdf + png) and the retained tile list that `scripts/73 --tiles-file`
consumes.

Usage:
    python scripts/76_tile_grid_figure.py --out results/figures --tile-km 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.transform import rowcol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapinfer as mi          # noqa: E402
from biodiv.mapbiomas import NATIVE        # noqa: E402

STEP = 8            # MapBiomas decimation: ~30 m -> ~240 m
UTM = "EPSG:32719"


def native_grid(years: list[int], step: int = STEP):
    """Union of the native mask over the MapBiomas map of **each prediction year**.

    Not the 2024 map alone. A tile that carried native forest in 2000 and was cleared before
    2024 is exactly the tile whose trend we want to measure; filtering on the last year only
    would drop it and bias the run towards places that never changed. `scripts/73` masks per
    pixel with `mapbiomas.year_map(y)` already, so this is the tile-level counterpart of the
    same rule: visit a tile if ANY prediction year has native cover in it.

    `year_map` resolves each year to its nearest available annual map (MapBiomas Chile
    collection 2 runs 1999-2024, so 2026 falls back to 2024).
    """
    from biodiv import mapbiomas as mb
    union, tr, used = None, None, {}
    for y in years:
        path, got, _ = mb.year_map(y)
        used[y] = int(got)
        if got in {v for k, v in used.items() if k != y}:
            continue                                   # same raster already folded in
        with rasterio.open(path) as src:
            a = src.read(1, out_shape=(src.height // step, src.width // step))
            if tr is None:
                tr = src.transform * src.transform.scale(src.width / a.shape[1],
                                                         src.height / a.shape[0])
        n = np.isin(a, list(NATIVE))
        union = n if union is None else (union | n)
        print(f"  MapBiomas {got} (for {y}): native {n.sum():,} px, "
              f"union {union.sum():,}")
    return union, tr, used


def classify(tiles: pd.DataFrame, native: np.ndarray, tr) -> np.ndarray:
    """True where the tile contains at least one native pixel (integral-image lookup)."""
    back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    lon0, lat0 = back.transform(tiles["xmin"].to_numpy(), tiles["ymin"].to_numpy())
    lon1, lat1 = back.transform(tiles["xmax"].to_numpy(), tiles["ymax"].to_numpy())
    r0, c0 = rowcol(tr, lon0, lat1)
    r1, c1 = rowcol(tr, lon1, lat0)
    h, w = native.shape
    r0, r1 = np.clip(r0, 0, h - 1), np.clip(r1, 0, h - 1)
    c0, c1 = np.clip(c0, 0, w - 1), np.clip(c1, 0, w - 1)
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(native.astype(np.int64), axis=0), axis=1)
    rr0, rr1 = np.minimum(r0, r1), np.maximum(r0, r1) + 1
    cc0, cc1 = np.minimum(c0, c1), np.maximum(c0, c1) + 1
    return (ii[rr1, cc1] - ii[rr0, cc1] - ii[rr1, cc0] + ii[rr0, cc0]) > 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-km", type=float, default=10.0, dest="tile_km")
    p.add_argument("--margin-km", type=float, default=20.0, dest="margin_km")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--years", default="2000,2003,2006,2009,2012,2015,2018,2021,2024,2026",
                   help="prediction years; the tile filter is the UNION of their native masks")
    args = p.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    plots = pd.read_parquet(ROOT / args.derived / "plots_unified.parquet")
    fwd = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    px, py = fwd.transform(plots["lon"].to_numpy(), plots["lat"].to_numpy())
    m = args.margin_km * 1000
    ext = (px.min() - m, py.min() - m, px.max() + m, py.max() + m)

    years = [int(v) for v in args.years.split(",")]
    tiles = mi.tile_grid(ext, args.tile_km * 1000, 30)
    print(f"native mask = union over the prediction years {years}")
    native, tr, used = native_grid(years)
    keep = classify(tiles, native, tr)
    tiles["has_native"] = keep
    print(f"{len(tiles):,} tiles of {args.tile_km:g} km over "
          f"{(ext[2]-ext[0])/1000:.0f} x {(ext[3]-ext[1])/1000:.0f} km")
    print(f"  with native: {keep.sum():,} ({100*keep.mean():.1f} %)")
    print(f"  empty:       {(~keep).sum():,} -- {(~keep).sum()*300/3600:.0f} h of load avoided")

    kept = tiles.loc[keep, ["tile_id", "xmin", "ymin", "xmax", "ymax"]]
    kept.to_csv(out / f"tiles_native_{args.tile_km:g}km.csv", index=False)
    print(f"  -> {out / f'tiles_native_{args.tile_km:g}km.csv'}")

    # ---- figure -----------------------------------------------------------------------
    # The extent is ~4.6x taller than wide, so a single map panel would be unreadable at
    # journal width. Panel (a) is the map, squeezed; (b) and (c) give the latitudinal
    # profile that the squeezed map cannot show.
    lat_p = plots["lat"].to_numpy()
    back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    _, tlat = back.transform((tiles["xmin"] + tiles["xmax"]).to_numpy() / 2,
                             (tiles["ymin"] + tiles["ymax"]).to_numpy() / 2)
    tiles["lat_c"] = tlat

    fig = plt.figure(figsize=(11.5, 13))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    km = 1000.0
    dropped = tiles.loc[~keep]
    ax.add_collection(PatchCollection(
        [Rectangle((r.xmin / km, r.ymin / km), (r.xmax - r.xmin) / km,
                   (r.ymax - r.ymin) / km) for r in dropped.itertuples()],
        facecolor="0.90", edgecolor="none"))
    ax.add_collection(PatchCollection(
        [Rectangle((r.xmin / km, r.ymin / km), (r.xmax - r.xmin) / km,
                   (r.ymax - r.ymin) / km) for r in tiles.loc[keep].itertuples()],
        facecolor="#2e7d32", edgecolor="none"))
    ax.scatter(px / km, py / km, s=1.1, c="#d81b60", lw=0, zorder=3, label="plots (3,102)")
    ax.set_xlim(ext[0] / km, ext[2] / km)
    ax.set_ylim(ext[1] / km, ext[3] / km)
    ax.set_aspect("equal")
    ax.set_xlabel("UTM 19S easting (km)")
    ax.set_ylabel("UTM 19S northing (km)")
    ax.set_title(f"(a) {args.tile_km:g} km tiling of the study extent", fontsize=10,
                 loc="left")
    ax.legend(handles=[
        Patch(facecolor="#2e7d32", label=f"native present, run ({keep.sum():,})"),
        Patch(facecolor="0.90", label=f"no native, skipped ({(~keep).sum():,})"),
        plt.Line2D([], [], marker="o", ls="", ms=3, c="#d81b60", label="plots (3,102)")],
        loc="upper right", fontsize=8, frameon=True)

    edges = np.arange(np.floor(tiles["lat_c"].min()), np.ceil(tiles["lat_c"].max()) + 1, 1.0)
    ax = fig.add_subplot(gs[0, 1])
    h_all, _ = np.histogram(tiles["lat_c"], bins=edges)
    h_keep, _ = np.histogram(tiles.loc[keep, "lat_c"], bins=edges)
    c = (edges[:-1] + edges[1:]) / 2
    ax.barh(c, h_all, height=0.9, color="0.88", label="all tiles")
    ax.barh(c, h_keep, height=0.9, color="#2e7d32", label="with native")
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel("tiles per 1° band")
    ax.set_ylabel("latitude (°)")
    ax.set_title("(b) tiles by latitude", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    hp, _ = np.histogram(lat_p, bins=edges)
    ax.barh(c, hp, height=0.9, color="#d81b60")
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel("plots per 1° band")
    ax.set_title("(c) training plots by latitude", fontsize=10, loc="left")
    ax.set_yticklabels([])

    fig.suptitle(
        f"Tiling and native-vegetation pre-filter for the multitemporal facet maps\n"
        f"extent = plot envelope + {args.margin_km:g} km; a tile is run if ANY prediction "
        f"year has native cover\n"
        f"MapBiomas maps {sorted(set(used.values()))}, classes {sorted(NATIVE)}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for e in ("pdf", "png"):
        fp = out / f"fig_tile_grid.{e}"
        fig.savefig(fp, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  -> {fp}")


if __name__ == "__main__":
    main()
