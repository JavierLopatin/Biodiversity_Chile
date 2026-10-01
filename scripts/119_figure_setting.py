#!/usr/bin/env python3
"""Figura 1: el escenario, y por que la geografia predice tanto.

No es un mapa de ubicacion. Es el montaje del control central del paper: los dos paneles
comparten la latitud en el eje y, asi que el lector ve en una sola escala donde estan las
parcelas y como cada faceta se mueve con la latitud. El gradiente que el modelo nulo
geografico explota queda a la vista antes de que el texto lo nombre, y con el la razon de
centrar dentro de bandas de 2 grados.

    a  mapa, parcelas por inventario
    b  seis facetas contra la latitud, tipificadas, con suavizado local

El mapa ya no sombrea la franja donde los dos inventarios coexisten: sobrecargaba un panel que
tiene que leerse de un vistazo. El dato sigue siendo parte del argumento y va al pie de figura
y a Metodos -- los dos inventarios solo coexisten entre 30 y 38 S, cuatro bandas y 1.584
parcelas, y de 38 S al sur no hay ninguna parcela de Parcelas-CL. Esa es la confusion declarada
entre procedencia y latitud, y es la segunda razon del centrado por banda; la primera es el
gradiente del panel b. El script la imprime al correr para que no se pierda.

El panel b lleva una linea por faceta y no un panel por faceta. Las facetas estan en escalas
incomparables -- riqueza de 0 a 26, LCBD del orden de 1e-4, MPD en millones de anos-- asi que
cada una se tipifica sobre todas sus parcelas y se suaviza contra la latitud con LOWESS. El
recorrido de cada curva, en desviaciones estandar, va en la leyenda.

El resultado medido es que TODAS son fuertemente latitudinales: el recorrido va de 1,5 DE en
TD q0 estandarizada a 2,9 en la diversidad oscura, con la riqueza cruda en 1,9. Importa decirlo
asi y no al reves. Un panel por faceta en escala original sugiere que las facetas donde la
reflectancia aporta son mas planas que las otras, y eso es un artefacto de comparar escalas
distintas lado a lado: tipificadas, no lo son. Entonces el gradiente latitudinal NO explica que
facetas gana el sensor, y esta figura justifica el modelo nulo geografico y el centrado por
banda sin adelantar el resultado.

El recorrido depende de FRAC y hay que leerlo como una magnitud, no como una cifra: con 0,25
iba de 1,2 a 2,7 y con 0,15 va de 1,5 a 2,9, porque LOWESS tiene pocos puntos en los extremos.
Lo que no cambia con la ventana es la conclusion: las seis estan en el mismo orden de magnitud.

Entre 31 y 37 S las curvas se ondulan. Ahi esta la mayor densidad de parcelas Y la franja donde
los dos inventarios coexisten, asi que parte de esa estructura puede ser el limite entre
inventarios y no ecologia. Con FRAC = 0,25 desaparece. Se deja a 0,15 porque el giro del extremo
austral es real y una ventana gruesa lo aplana, pero la ondulacion del centro no se interpreta.

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
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, DARK = "#4C78A8", "#E8832A", "#333333"
DPI = 300
BANDA = 2.0
LAT_MIN, LAT_MAX = -56.0, -29.0
#: Ventana del suavizado, como fraccion de las parcelas. 0,15 sobre 25 grados de recorrido da
#: una ventana de unos 4 grados, el doble de la banda de centrado. Mas gruesa aplana el giro
#: del extremo austral, que es real; mas fina empieza a seguir el ruido de las bandas con 19
#: parcelas.
FRAC = 0.15
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8.5})

#: Rotulos cortos a proposito: el panel es angosto y la leyenda va dentro de sus ejes.
FUENTES = [("parcelas_cl", BLUE, "Parcelas-CL"), ("living_trees", ORANGE, "Living Trees")]
SOLAPE = (-38.0, -30.0)     #: bandas donde los dos inventarios coexisten, medido en `datos()`

#: (columna, fichero, rotulo, color). Una o dos por familia, donde el contraste importa.
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

    # Alta y angosta a proposito. Con cajas iguales y el mapa sin distorsionar, el rango de
    # longitud que se dibuja sale de la forma de la caja: una figura mas baja obliga a estirar
    # la longitud y Chile queda chico dentro de mucho oceano. 25 grados de latitud sobre una
    # caja de ~7,9 por ~3,4 pulgadas pide unos 11 de longitud, que es el ancho real del pais.
    fig = plt.figure(figsize=(7.8, 9.9))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.025, figure=fig)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1], sharey=ax_map)

    # --- a  mapa
    lineas_banda(ax_map)
    chile.plot(ax=ax_map, facecolor="#F4F4F4", edgecolor="#BBBBBB", linewidth=0.3, zorder=2)
    marcas = []
    for src, color, etiqueta in FUENTES:
        s = d[d.source == src]
        ax_map.scatter(s.lon, s.lat, s=5, alpha=0.55, color=color, lw=0, zorder=3)
        marcas.append(Line2D([], [], marker="o", ls="none", ms=6, color=color,
                             label=f"{etiqueta} ($n$={len(s):,})"))
    # `adjustable="datalim"` en vez de encoger la caja: con aspecto igual, matplotlib por
    # defecto achica el EJE hasta que la forma del dato le calce, y eso dejaba un hueco grande
    # entre los dos paneles que ningun `wspace` cerraba. Asi la caja se queda del tamano que le
    # dio el GridSpec y es el rango de longitud el que se estira para llenarla.
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set(xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax_map.set_title("a  Plot network", loc="left", fontweight="bold")
    # arriba a la izquierda: con el panel angosto la leyenda no cabia abajo sin salirse, y
    # esa esquina es oceano en todo el tramo de 30 a 36 S



    # --- b  facetas tipificadas, suavizadas contra la latitud
    lineas_banda(ax_f)
    ax_f.axvline(0, color=DARK, lw=0.8, zorder=2)
    vistos, handles = {}, []
    for col, _, etiqueta, color in FACETAS:
        s = d[np.isfinite(d[col])]
        z = ((s[col] - s[col].mean()) / s[col].std()).to_numpy()
        sm_fit = sm.nonparametric.lowess(z, s.lat.to_numpy(), frac=FRAC, return_sorted=True)
        lat_s, z_s = sm_fit[:, 0], sm_fit[:, 1]
        ls = ESTILO[vistos.get(color, 0)]
        vistos[color] = vistos.get(color, 0) + 1
        ax_f.plot(z_s, lat_s, color=color, lw=2.2, ls=ls, zorder=4, solid_capstyle="round")
        rango = float(z_s.max() - z_s.min())
        handles.append((rango, Line2D([], [], color=color, lw=2.2, ls=ls,
                                      label=f"{etiqueta}  ({rango:.1f} SD)")))

    ax_f.set_xlabel("Facet value (SD), smoothed against latitude")
    ax_f.set_title("b  Latitudinal biodiversity distribution", loc="left", fontweight="bold")
    ax_f.tick_params(labelleft=False)
    ax_f.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
    ax_f.set_axisbelow(True)
    handles = [h for _, h in sorted(handles, key=lambda t: -t[0])]

    ax_map.set_ylim(LAT_MIN, LAT_MAX)
    ax_map.set_xlim(float(d.lon.median()) - 0.1, float(d.lon.median()) + 0.1)

    # Una sola leyenda al pie, para los dos paneles: son una figura, no dos, y repetir el marco
    # de la leyenda dos veces dentro de los ejes lo negaba. Ademas libera las dos esquinas que
    # las leyendas ocupaban, que es justo donde caen los datos.
    fig.subplots_adjust(left=0.085, right=0.99, top=0.955, bottom=0.092)
    # Anclada por ARRIBA y justo debajo del rotulo del eje x, no al pie de la figura: anclarla
    # abajo dejaba el sobrante entre el rotulo y la leyenda, que es donde mas se nota.
    fig.legend(handles=marcas + handles, ncol=4, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.040), frameon=False, columnspacing=1.8,
               handletextpad=0.5, handlelength=2.2, labelspacing=0.35)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig01_setting.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig01_setting'}.{{png,pdf}}")

    t = d.pivot_table(index="banda", columns="source", aggfunc="size", fill_value=0)
    sol = t[(t.parcelas_cl > 0) & (t.living_trees > 0)]
    print(f"\nparcelas por inventario: {d.source.value_counts().to_dict()}")
    print(f"bandas con los dos inventarios: {list(sol.index)} -> {int(sol.sum().sum())} parcelas")


if __name__ == "__main__":
    main()
