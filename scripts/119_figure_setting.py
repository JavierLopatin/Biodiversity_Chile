#!/usr/bin/env python3
"""Figura 1: el escenario, y por que la geografia predice tanto.

No es un mapa de ubicacion. Es el montaje del control central del paper: TODOS los paneles
comparten la latitud en el eje y, asi que el lector ve en una sola escala donde estan las
parcelas, de que inventario vienen, y como cada faceta se mueve con la latitud. El gradiente
que el modelo nulo geografico explota queda a la vista antes de que el texto lo nombre, y con
el la razon de centrar dentro de bandas de 2 grados.

    a  mapa, parcelas por inventario
    b  parcelas por banda de 2 grados, apiladas por inventario
    c  cuatro facetas contra la latitud, una por familia

El panel b no es decorativo: muestra que los dos inventarios solo se solapan entre 30 y 38 S y
que el resto del pais es Living Trees solo. Esa es la confusion declarada entre procedencia y
latitud, y es la segunda razon del centrado por banda -- la primera es el gradiente del panel c.

Las facetas del panel c son una por familia y las mismas que el texto discute:

    riqueza cruda q0   el mayor margen sobre el nulo de las 22 facetas
    LCBD Sorensen      unicidad composicional, la otra que gana sin clima
    Isomap 1 (p/a)     ordenacion, donde la geografia gana
    MPD                estructura filogenetica, donde la geografia gana

Uso:
    python scripts/119_figure_setting.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, DARK, GREY = "#4C78A8", "#E8832A", "#333333", "#9A9A9A"
DPI = 300
BANDA = 2.0
LAT_MIN, LAT_MAX = -56.0, -29.0
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8.5})

FUENTES = [("parcelas_cl", BLUE, "Parcelas-CL"), ("living_trees", ORANGE, "Living Trees Chile")]

#: (columna, fichero, rotulo, escala del eje x). Una faceta por familia.
FACETAS = [
    ("hill_q0_unified", "unified_diversity_responses_woody.parquet", "Richness $q_0$", 1.0),
    ("lcbd_count_sorensen", "lcbd_count_sorensen_unified_padded_woody.parquet",
     "LCBD Sørensen ($\\times 10^{-4}$)", 1e4),
    ("isomap1_pa_unified", "unified_diversity_responses_woody.parquet", "Isomap 1 (p/a)", 1.0),
    ("mpd_unified", "unified_phylo_responses_woody.parquet", "MPD", 1.0),
]


def datos() -> pd.DataFrame:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lon", "lat", "source"]]
    for col, src, _, _esc in FACETAS:
        d = pd.read_parquet(DERIVED / src)
        if col not in d.columns:
            raise KeyError(f"{col} no esta en {src}")
        p = p.merge(d[["PlotObservationID", col]], on="PlotObservationID", how="left")
    p["banda"] = np.floor(p.lat / BANDA) * BANDA
    return p


def lineas_banda(ax) -> None:
    for b in np.arange(LAT_MIN, LAT_MAX + BANDA, BANDA):
        ax.axhline(b, color="#E4E4E4", lw=0.5, zorder=0)


def main() -> None:
    d = datos()
    chile = gpd.read_file(ROOT / "shapefiles" / "regiones_chile.shp").to_crs("EPSG:4326")

    fig = plt.figure(figsize=(13.0, 8.4))
    gs = GridSpec(1, 6, width_ratios=[2.5, 1.15, 1.25, 1.25, 1.25, 1.25],
                  wspace=0.16, figure=fig)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1], sharey=ax_map)
    axes_f = [fig.add_subplot(gs[0, 2 + i], sharey=ax_map) for i in range(len(FACETAS))]

    # --- a  mapa
    chile.plot(ax=ax_map, facecolor="#F4F4F4", edgecolor="#BBBBBB", linewidth=0.3, zorder=1)
    lineas_banda(ax_map)
    for src, color, etiqueta in FUENTES:
        s = d[d.source == src]
        ax_map.scatter(s.lon, s.lat, s=5, alpha=0.55, color=color, lw=0, zorder=3,
                       label=f"{etiqueta} ($n$ = {len(s):,})")
    ax_map.set(xlabel="Longitude (°)", ylabel="Latitude (°)", xlim=(-76.5, -65.5))
    ax_map.set_title("a  Plot network", loc="left", fontweight="bold")
    ax_map.legend(loc="lower left", fontsize=8, framealpha=0.95, markerscale=2.2)

    # --- b  parcelas por banda, apiladas por inventario
    lineas_banda(ax_n)
    bandas = np.sort(d.banda.unique())
    izq = np.zeros(len(bandas))
    for src, color, etiqueta in FUENTES:
        n = (d[d.source == src].groupby("banda").size().reindex(bandas, fill_value=0).to_numpy())
        ax_n.barh(bandas + BANDA / 2, n, height=BANDA * 0.78, left=izq, color=color,
                  alpha=0.85, lw=0, zorder=3)
        izq = izq + n
    ax_n.set_xlabel("Plots per 2° bin")
    ax_n.set_title("b  Sampling", loc="left", fontweight="bold")
    # la franja donde los dos inventarios coexisten: la confusion declarada del pool
    ax_n.axhspan(-38, -30, color=DARK, alpha=0.05, zorder=1)
    ax_n.text(ax_n.get_xlim()[1] * 0.96, -34, "both\ninventories", fontsize=7.5, style="italic",
              color="#777777", ha="right", va="center", zorder=4)

    # --- c  facetas contra la latitud
    for ax, (col, _, etiqueta, esc) in zip(axes_f, FACETAS):
        lineas_banda(ax)
        s = d[np.isfinite(d[col])]
        for src, color, _ in FUENTES:
            t = s[s.source == src]
            ax.scatter(t[col] * esc, t.lat, s=3, alpha=0.30, color=color, lw=0, zorder=2)
        m = s.groupby("banda")[col].mean() * esc
        ax.step(m.to_numpy(), m.index.to_numpy() + BANDA / 2, where="mid", color=DARK,
                lw=1.6, zorder=4)
        ax.set_xlabel(etiqueta)
        ax.tick_params(labelleft=False)

    axes_f[0].set_title("c  Each facet against latitude", loc="left", fontweight="bold")
    for ax in [ax_n, *axes_f]:
        ax.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
        ax.set_axisbelow(True)
    ax_map.set_ylim(LAT_MIN, LAT_MAX)

    h = [Line2D([], [], color=DARK, lw=1.6, label="mean within each 2° latitude bin")]
    axes_f[-1].legend(handles=h, fontsize=7.5, loc="lower right", framealpha=0.95)

    # `tight_layout` no es compatible con este GridSpec compartido; los margenes van a mano
    fig.subplots_adjust(left=0.055, right=0.995, top=0.945, bottom=0.075)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig01_setting.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig01_setting'}.{{png,pdf}}")
    print(f"\nparcelas por inventario: {d.source.value_counts().to_dict()}")
    print(f"bandas de {BANDA:g} grados: {len(bandas)}  "
          f"(mediana {int(d.groupby('banda').size().median())} parcelas, "
          f"minimo {int(d.groupby('banda').size().min())})")


if __name__ == "__main__":
    main()
