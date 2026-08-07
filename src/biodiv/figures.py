"""Publication-figure helpers: consistent style, Chile basemap, and format policy.

Export policy, applied by :func:`save_figure`:

- **PDF** when the figure is vector content — lines, points, polygons, text. Stays sharp at
  any zoom and at any print size, and journals prefer it.
- **PNG at 300 dpi** when the figure contains genuine raster imagery — pixel patches,
  hillshades, image panels. Vectorising those would embed one path per pixel and produce an
  enormous file for no gain.

Heatmaps are drawn with ``pcolormesh`` rather than ``imshow`` so a correlation matrix counts
as vector content and goes to PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SHAPEFILE = Path(__file__).resolve().parents[2] / "shapefiles" / "regiones_chile.shp"

# Study area: central Chile, matching the subset filter in scripts/01_build_subset.py
STUDY_LAT = (-38.0, -30.0)
STUDY_LON = (-73.5, -69.5)

# Colour-blind-safe qualitative palette (Okabe-Ito), used wherever series are categorical.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

INDEX_COLOURS = {
    "ndvi": "#0072B2", "evi": "#D55E00", "kndvi": "#009E73",
    "savi": "#CC79A7", "nbr": "#E69F00",
}
INDEX_LABELS = {
    "ndvi": "NDVI", "evi": "EVI", "kndvi": "kNDVI", "savi": "SAVI", "nbr": "NBR",
}


def set_paper_style(base_size: int = 9) -> None:
    """Typography and line weights sized for a single journal column.

    Fonts stay as text in the PDF (``pdf.fonttype = 42``) so the typesetter can restyle or
    search them; the default Type 3 encoding breaks both.
    """
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_figure(fig, name: str, out_dir: str | Path, raster: bool = False,
                also_png: bool = False) -> Path:
    """Write a figure under the export policy. Returns the path written.

    ``raster=True`` for figures containing real pixel imagery; everything else goes to PDF.
    ``also_png=True`` additionally writes a 300-dpi PNG of a vector figure, for slides or
    for journals that reject PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if raster:
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=300)
    else:
        path = out_dir / f"{name}.pdf"
        fig.savefig(path)
        if also_png:
            fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    return path


_CHILE_CACHE: dict = {}


def load_chile(clip_study_area: bool = False, simplify: float | None = 0.005):
    """Chile region polygons (EPSG:4326) for use as a basemap.

    ``simplify`` is a Douglas-Peucker tolerance in degrees. The source coastline is stored
    at full resolution (a 7.8 MB shapefile); drawing it as vector puts every vertex into the
    PDF and produced a 16 MB figure for a three-panel map. At 0.005° (~500 m) the outline is
    visually identical at journal print size and the file drops by two orders of magnitude.
    Pass ``None`` to keep full resolution for a large-format map.

    Result is cached: the basemap is drawn many times across a figure set.
    """
    import geopandas as gpd

    key = (clip_study_area, simplify)
    if key in _CHILE_CACHE:
        return _CHILE_CACHE[key]

    g = gpd.read_file(SHAPEFILE)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    if clip_study_area:
        g = g.cx[STUDY_LON[0]:STUDY_LON[1], STUDY_LAT[0]:STUDY_LAT[1]]
    if simplify:
        g = g.copy()
        g["geometry"] = g.geometry.simplify(simplify, preserve_topology=True)
    _CHILE_CACHE[key] = g
    return g


def basemap(ax, extent=None, lw: float = 0.35, face: str = "#F2F2F2",
            edge: str = "#9A9A9A", label_axes: bool = True):
    """Draw the Chile outline into ``ax`` and set a geographic aspect ratio.

    The aspect is set so one degree of latitude and one of longitude occupy the same ground
    distance at the centre of the extent — without it, Chile appears stretched east-west.
    """
    if extent is None:
        extent = (*STUDY_LON, *STUDY_LAT)
    # Clip the geometry to the visible window before drawing. Setting xlim/ylim alone hides
    # the rest of the country but still writes every vertex into the PDF -- with Chile's
    # southern fjords that is most of the file. Clipping first cut fig01 from 16 MB to ~0.1 MB.
    m = 0.5
    chile = load_chile().cx[extent[0] - m:extent[1] + m, extent[2] - m:extent[3] + m]
    chile.plot(ax=ax, facecolor=face, edgecolor=edge, linewidth=lw, zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    mid_lat = np.deg2rad((extent[2] + extent[3]) / 2)
    ax.set_aspect(1.0 / np.cos(mid_lat))
    if label_axes:
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
    return ax


def add_inset_locator(fig, ax, extent=None, size: str = "28%"):
    """Small all-Chile inset with the study extent boxed, for orientation."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Rectangle

    if extent is None:
        extent = (*STUDY_LON, *STUDY_LAT)
    axin = inset_axes(ax, width=size, height=size, loc="upper right", borderpad=0.4)
    # The locator inset shows all of Chile at ~3 cm, so it can be simplified much harder
    chile = load_chile(simplify=0.05)
    chile.plot(ax=axin, facecolor="#EDEDED", edgecolor="#9A9A9A", linewidth=0.2)
    axin.add_patch(Rectangle((extent[0], extent[2]), extent[1] - extent[0],
                             extent[3] - extent[2], fill=False, edgecolor="#D55E00",
                             linewidth=0.9))
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_linewidth(0.4)
    axin.set_aspect(1.0 / np.cos(np.deg2rad(-38)))
    return axin


def scalebar(ax, length_km: float = 100, loc=(0.06, 0.06)):
    """Scale bar in kilometres, converted to degrees of longitude at the axis centre."""
    lat_mid = np.mean(ax.get_ylim())
    deg = length_km / (111.320 * np.cos(np.deg2rad(lat_mid)))
    x0 = ax.get_xlim()[0] + loc[0] * np.diff(ax.get_xlim())[0]
    y0 = ax.get_ylim()[0] + loc[1] * np.diff(ax.get_ylim())[0]
    ax.plot([x0, x0 + deg], [y0, y0], color="k", lw=1.6, solid_capstyle="butt", zorder=5)
    ax.text(x0 + deg / 2, y0 + 0.012 * np.diff(ax.get_ylim())[0], f"{length_km:g} km",
            ha="center", va="bottom", fontsize=7)


def heatmap(ax, M, xlabels, ylabels, cmap="RdBu_r", vmin=-1, vmax=1,
            annot: bool = True, fmt: str = "{:.2f}", fontsize: int = 6):
    """Vector heatmap via pcolormesh, so the figure can still be exported as PDF."""
    M = np.asarray(M, dtype=float)
    mesh = ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1),
                         M[::-1], cmap=cmap, vmin=vmin, vmax=vmax,
                         edgecolors="white", linewidth=0.3)
    ax.set_xticks(np.arange(M.shape[1]) + 0.5, xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(M.shape[0]) + 0.5, ylabels[::-1])
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[::-1][i, j]
                if np.isfinite(v):
                    ax.text(j + 0.5, i + 0.5, fmt.format(v), ha="center", va="center",
                            fontsize=fontsize,
                            color="white" if abs(v) > 0.6 * max(abs(vmin), abs(vmax)) else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return mesh
