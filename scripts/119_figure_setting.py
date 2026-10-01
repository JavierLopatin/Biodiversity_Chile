#!/usr/bin/env python3
"""Figura 1: el escenario, y por que la geografia predice tanto.

No es un mapa de ubicacion. Es el montaje del control central del paper: TODOS los paneles
comparten la latitud en el eje y, asi que el lector ve en una sola escala donde estan las
parcelas, de que inventario vienen, y como cada faceta se mueve con la latitud. El gradiente
que el modelo nulo geografico explota queda a la vista antes de que el texto lo nombre, y con
el la razon de centrar dentro de bandas de 2 grados.

    a  mapa, parcelas por inventario
    b  parcelas por banda de 2 grados, apiladas por inventario
    c  seis facetas contra la latitud, en un solo panel y en unidades de desviacion estandar

El panel b no es decorativo: muestra que los dos inventarios solo se solapan entre 30 y 38 S y
que el resto del pais es Living Trees solo. Esa es la confusion declarada entre procedencia y
latitud, y es la segunda razon del centrado por banda -- la primera es el gradiente del panel c.

El panel c lleva una linea por faceta y no un panel por faceta. Las facetas estan en escalas
incomparables -- riqueza de 0 a 26, LCBD del orden de 1e-4, MPD en millones de anos-- asi que
cada una se tipifica sobre todas sus parcelas y lo que se dibuja es la MEDIA POR BANDA en
unidades de desviacion estandar. Eso convierte el panel en una medida de cuan latitudinal es
cada faceta, y el recorrido de cada linea en DE va en la leyenda.

El resultado medido es que TODAS lo son, y parecido: de 1,7 DE la riqueza cruda a 2,5 la
diversidad oscura. Importa decirlo asi y no al reves. Un panel por faceta en escala original
sugiere que las facetas donde la reflectancia aporta son mas planas que las otras, y eso es un
artefacto de comparar escalas distintas lado a lado: tipificadas, no lo son. Entonces el
gradiente latitudinal NO explica que facetas gana el sensor, y esta figura justifica el modelo
nulo geografico y el centrado por banda sin adelantar el resultado.

No hay ejes de Isomap: una ordenacion no es una faceta de diversidad sino una coordenada
derivada de la matriz de comunidad, y su signo y escala son arbitrarios.

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

#: (columna, fichero, rotulo, color). Dos por familia donde el contraste importa.
RIQ, FIL, COMP, OSC = "#4C78A8", "#54A24B", "#E8832A", "#B279A2"
FACETAS = [
    ("hill_q0_unified", "unified_diversity_responses_woody.parquet", "Richness $q_0$ (raw)", RIQ),
    ("td_inext_q0", "td_inext_coverage_unified_padded_woody.parquet",
     "TD $q_0$ (cov.-std.)", RIQ),
    ("lcbd_count_sorensen", "lcbd_count_sorensen_unified_padded_woody.parquet",
     "LCBD Sørensen", COMP),
    ("mpd_unified", "unified_phylo_responses_woody.parquet", "MPD", FIL),
    ("ses_pd_unified", "unified_phylo_responses_woody.parquet", "SES PD", FIL),
    ("dark_n_unified", "unified_dark_diversity_woody.parquet", "Dark diversity", OSC),
]
ESTILO = ["-", "--"]        #: dos facetas de la misma familia comparten color, no trazo


def datos() -> pd.DataFrame:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lon", "lat", "source"]]
    for col, src, _, _c in FACETAS:
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

    fig = plt.figure(figsize=(11.2, 8.4))
    gs = GridSpec(1, 3, width_ratios=[2.3, 1.0, 2.6], wspace=0.14, figure=fig)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1], sharey=ax_map)
    ax_f = fig.add_subplot(gs[0, 2], sharey=ax_map)

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

    # --- c  todas las facetas en un panel, tipificadas
    lineas_banda(ax_f)
    ax_f.axvline(0, color=DARK, lw=0.8, zorder=2)
    vistos, handles = {}, []
    for col, _, etiqueta, color in FACETAS:
        s = d[np.isfinite(d[col])]
        z = (s[col] - s[col].mean()) / s[col].std()
        m = z.groupby(s.banda).mean()
        ls = ESTILO[vistos.get(color, 0)]
        vistos[color] = vistos.get(color, 0) + 1
        ax_f.step(m.to_numpy(), m.index.to_numpy() + BANDA / 2, where="mid", color=color,
                  lw=2.0, ls=ls, zorder=4, solid_capstyle="round")
        rango = float(m.max() - m.min())
        handles.append((rango, Line2D([], [], color=color, lw=2.0, ls=ls,
                                      label=f"{etiqueta}  ({rango:.1f} SD)")))
    ax_f.set_xlabel("Bin mean, in standard deviations of the facet")
    ax_f.set_title("c  How latitudinal each facet is", loc="left", fontweight="bold")
    ax_f.tick_params(labelleft=False)
    # a la derecha y al centro: entre 38 y 48 S el lado derecho esta vacio, y abajo a la
    # izquierda la leyenda tapaba las lineas del extremo austral
    handles = [h for _, h in sorted(handles, key=lambda t: -t[0])]
    ax_f.legend(handles=handles, fontsize=8, loc="center right", framealpha=0.95,
                labelspacing=0.4, handlelength=2.4,
                title="latitudinal range", title_fontsize=8)
    ax_f.set_xlim(right=2.05)

    for ax in (ax_n, ax_f):
        ax.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
        ax.set_axisbelow(True)
    ax_map.set_ylim(LAT_MIN, LAT_MAX)

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
