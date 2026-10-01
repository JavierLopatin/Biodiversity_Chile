#!/usr/bin/env python3
"""Figura 3: que facetas gana el sensor remoto a saber donde estas.

El resultado central del paper, y la figura alrededor de la cual se escribe Resultados.

La pregunta no es "cuanto predice el modelo" sino "cuanto predice MAS QUE tres coordenadas".
Casi toda la literatura de este campo reporta R2 agrupados sin ese control, y aqui tres numeros
-- lon, lat, elevacion-- reproducen el 0,702 del PCoA 1 contra el 0,704 del modelo completo.

Terminologia: "bin" y no "band" para los grupos de latitud. En teledeteccion "band" es banda
espectral, y la leyenda de esta misma figura dice "six-band geometric median" -- dos
centimetros debajo pondria "2 deg latitude bands". "Block" y "window" tampoco servian: el
manuscrito los usa ya para el CV espacial de 20 km y para las ventanas causales y de pixeles.
"Bin" esta libre y nombra la operacion, que es agrupar por latitud; "belt" tambien estaba
libre pero sugiere una zona ecologica real, y estas no lo son.

Dos paneles, las mismas 20 facetas en el mismo orden:

    a  el piso de coordenadas contra la REFLECTANCIA SOLA (sin clima)
    b  el piso de coordenadas contra reflectancia + clima

Los titulos dicen "reflectance" y no "geomedian": que el resumen temporal sea un geomediano de
seis bandas es un detalle de implementacion que va en la leyenda y en Metodos, no en el titulo
de un panel, donde solo compite con el mensaje.

El contraste entre los dos paneles ES el resultado, y cada panel se colorea con SUS PROPIOS
numeros: en (a) cinco facetas de veintidos superan al modelo nulo geografico, en (b) diecinueve.
Esa diferencia es el argumento -- pero el clima sale de una grilla interpolada desde estaciones y
dentro de una banda de latitud sigue siendo en buena parte posicion, asi que un bloque que solo
gana con clima no ha demostrado que el sensor aporte. Eso se dice en el texto y en el pie, no
escondido en el color: colorear (b) con el criterio de (a) hacia que la leyenda mintiera, porque
mancuernas que caen claramente a la derecha del nulo salian en gris.

La separacion tiene lectura directa: lo que gana en (a) son escalares DEL RODAL -- cuan inusual
es esta parcela, cuantas especies hay. Lo que pierde son posiciones en un pool regional: que
tipo de comunidad, que estructura filogenetica, que falta del pool. Un sensor ve el rodal, no la
historia biogeografica.

La ordenacion del texto principal es ISOMAP (decision del 2026-10-01); los ejes de PCoA van al
suplemento (`figS8`), con los diez juntos.

Los asteriscos salen de `results/tables/significancia_margen.csv` (`scripts/118`): bootstrap
por bloques de 20 km sobre la diferencia pareada de errores cuadraticos contra el nulo, con
Benjamini-Hochberg sobre las 52 pruebas. No es un F-test ni un t-test; por que ninguno de los
dos aplica esta en `scripts/118`. En el panel (b) va el asterisco solo, sin el numero, porque
ese panel no rotula margenes.

Todo lo demas sale de `results/tables/margen_sin_clima.csv` (`scripts/107`). Ningun numero esta
escrito a mano.

Uso:
    python scripts/115_figure_remote_vs_geography.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables" / "margen_sin_clima.csv"
SIG = ROOT / "results" / "tables" / "significancia_margen.csv"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, GREY = "#4C78A8", "#888888"
DARK, WIN = "#333333", "#B4451F"
DPI = 300
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9.5})

#: Un solo color para "gana", no uno por grupo: la familia de cada faceta ya la dice el rotulo
#: del bloque, asi que el color por grupo era informacion repetida que alargaba la leyenda a
#: cinco entradas -- y el diccionario por grupo no cubria "otra", el grupo de los ejes de
#: Isomap, que ahora si puede ganar en el panel de clima.
#: Orden por FAMILIA de faceta, no por margen. Ordenar por el resultado agrupa facetas que no
#: se comparan entre si (una TD junto a un eje de ordenacion) y rompe la lectura: el lector
#: quiere ver las tres TD juntas, las tres PD juntas, y despues las de composicion. El orden
#: dentro de cada familia es q0, q1, q2, que es el orden del parametro de Hill.
#: La ordenacion del texto principal es ISOMAP (decision del 2026-10-01). Se eligio sobre PCoA
#: por los dos criterios que el suplemento reporta en detalle:
#:   acumulacion de informacion  Isomap con 2 ejes supera a PCoA con 8 en el techo de
#:                               reconstruccion del Jaccard observado (0,577 contra 0,502)
#:   desempeno                   eje 1 contra eje 1 sobre el modelo nulo geografico:
#:                               +0,051 contra -0,003 en p/a, -0,009 contra -0,030 en frecuencia
#: El eje 1 de Isomap es ademas el estable al parametro k (rho 0,94 entre k=30 y k=80), asi
#: que la eleccion no se apoya en los ejes 2 y 3, que si se mueven (0,70 y 0,83).
#: Los cuatro ejes de PCoA van al suplemento (figS8), con los diez juntos para que el lector
#: pueda comprobar cualquier criterio.
FAMILIAS = [
    ("Taxonomic richness", ["hill_q0_unified", "td_inext_q0", "td_inext_q1", "td_inext_q2"]),
    ("Phylogenetic", ["pd_inext_q0", "pd_inext_q1", "pd_inext_q2",
                      "mpd_unified", "mntd_unified",
                      "ses_pd_unified", "ses_mpd_unified", "ses_mntd_unified"]),
    ("Compositional uniqueness", ["lcbd_count_sorensen", "lcbd_pa_unified",
                                  "lcbd_freq_unified"]),
    ("Dark diversity", ["dark_n_unified"]),
    ("Floristic composition (ordination)", ["isomap1_pa_unified", "isomap2_pa_unified",
                                            "isomap3_pa_unified", "isomap1_freq_unified",
                                            "isomap2_freq_unified", "isomap3_freq_unified"]),
]
ORDEN = [f for _, fs in FAMILIAS for f in fs]

#: Hueco entre familias, en unidades de fila. Sin el, el rotulo de familia no cabe entre la
#: linea separadora y la primera mancuerna del grupo, y pisa los datos.
HUECO = 1.0


def posiciones() -> tuple[dict, list]:
    """y de cada faceta y de cada separador, con un hueco entre familias."""
    y, sep, cur = {}, [], 0.0
    for k, (_, fs) in enumerate(FAMILIAS):
        if k:
            sep.append(cur - HUECO / 2)
            cur += HUECO
        for fa in fs:
            y[fa] = cur
            cur += 1.0
    return y, sep

#: Rotulos en ingles.
LAB = {
    "lcbd_count_sorensen": "LCBD Sørensen", "lcbd_pa_unified": "LCBD (p/a)",
    "lcbd_freq_unified": "LCBD (freq.)", "pcoa1_pa_unified": "PCoA 1 (p/a)",
    "pcoa2_pa_unified": "PCoA 2 (p/a)", "pcoa1_freq_unified": "PCoA 1 (freq.)",
    "pcoa2_freq_unified": "PCoA 2 (freq.)",
    "isomap1_pa_unified": "Isomap 1 (p/a)", "isomap2_pa_unified": "Isomap 2 (p/a)",
    "isomap3_pa_unified": "Isomap 3 (p/a)", "isomap1_freq_unified": "Isomap 1 (freq.)",
    "isomap2_freq_unified": "Isomap 2 (freq.)", "isomap3_freq_unified": "Isomap 3 (freq.)",
    "hill_q0_unified": "Richness $q_0$ (raw)", "dark_n_unified": "Dark diversity",
    "td_inext_q0": "TD $q_0$ (cov.-std.)", "td_inext_q1": "TD $q_1$ (cov.-std.)",
    "td_inext_q2": "TD $q_2$ (cov.-std.)",
    "mpd_unified": "MPD", "mntd_unified": "MNTD", "ses_pd_unified": "SES PD",
    "ses_mpd_unified": "SES MPD", "ses_mntd_unified": "SES MNTD",
    "pd_inext_q0": "PD $q_0$ (cov.-std.)", "pd_inext_q1": "PD $q_1$ (cov.-std.)",
    "pd_inext_q2": "PD $q_2$ (cov.-std.)",
}


def significancia() -> dict:
    """`(faceta, modelo) -> asteriscos` de `scripts/118`. Sin la tabla la figura sale igual."""
    try:
        sg = pd.read_csv(SIG).fillna({"sig": ""})
    except FileNotFoundError:
        print("[aviso] falta significancia_margen.csv: figura sin asteriscos")
        return {}
    return {(r.faceta, r.modelo): str(r.sig) for _, r in sg.iterrows()}


def panel(ax, d: pd.DataFrame, col_r2: str, col_margen: str, titulo: str,
          marca_ganadores: bool, est: dict) -> None:
    """Mancuerna: piso de coordenadas -> bloque, una fila por faceta.

    La mancuerna y no barras porque lo que importa es la DISTANCIA entre dos numeros, no el
    valor de ninguno: una barra invita a leer la altura, que aqui no significa nada sin el piso
    al lado.
    """
    Y, SEP = posiciones()
    y = [Y[k] for k in d.faceta]
    for i, (_, r) in zip(y, d.iterrows()):
        # Dos estados: gana o no, decidido con las columnas DE ESTE PANEL. El flag
        # `remota_propia` de la tabla mira solo los bloques sin clima, asi que servia para (a)
        # y no para (b): usarlo en los dos dejaba en gris mancuernas del panel de clima que
        # caen a la derecha del nulo, justo lo que la leyenda llama ganancia.
        #
        # "Gana" sigue exigiendo las dos condiciones: margen positivo Y R2 absoluto positivo.
        # Si el nulo geografico es negativo, superarlo solo dice "menos malo que la geografia"
        # y el modelo sigue sin predecir. Con clima, TD q1/q2 cruzan el cero por +0,004: pasan
        # el criterio, pero la mancuerna queda pegada al cero y eso se ve.
        gana = bool(r[col_margen] > 0 and r[col_r2] > 0)
        c = BLUE if gana else GREY
        ax.plot([r.piso_coords, r[col_r2]], [i, i], color=c,
                lw=2.8 if gana else 1.6, alpha=1.0 if gana else 0.5, zorder=2,
                solid_capstyle="round")
        ax.plot(r.piso_coords, i, "o", ms=5.5, color="white", mec=DARK, mew=1.4, zorder=3)
        ax.plot(r[col_r2], i, "o", ms=7 if gana else 5, color=c,
                mec="white", mew=0.8, zorder=4)

    for i, (_, r) in zip(y, d.iterrows()):
        if not (r[col_margen] > 0 and r[col_r2] > 0):
            continue
        a = est.get((r.faceta, col_margen), "")
        x = max(r[col_r2], r.piso_coords) + 0.02
        if marca_ganadores:
            ax.text(x, i, f"+{r[col_r2] - r.piso_coords:.3f}{a}", va="center", fontsize=8.5,
                    color=WIN, fontweight="bold")
        elif a:
            # panel sin rotulos de margen: el asterisco solo, pegado al marcador
            ax.text(x, i, a, va="center", fontsize=9.5, color=WIN, fontweight="bold")

    # separador entre familias: sin el, veinte filas seguidas se leen como una lista plana
    for sy in SEP:
        ax.axhline(sy, color="#BBBBBB", lw=0.8, ls="-", zorder=1)

    ax.axvline(0, color=DARK, lw=0.7, zorder=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LAB.get(k, k) for k in d.faceta])
    ax.set_xlabel("out-of-fold $R^2$, centred within 2° latitude bins")
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    # 0,75 y no 0,60: con los ejes de Isomap el maximo sube a 0,695 (Isomap 2 p/a con
    # clima), asi que 0,60 truncaba datos
    ax.set_xlim(-0.12, 0.75)


def main() -> None:
    t = pd.read_csv(TAB)
    t = t[t.faceta.isin(ORDEN)].copy()        # PCoA queda fuera: va al suplemento
    t["R2_gm_clima"] = t.piso_coords + t.gm_clima
    t["orden"] = t.faceta.map({k: i for i, k in enumerate(ORDEN)})
    t = t.sort_values("orden").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), sharey=True)
    est = significancia()
    panel(axes[0], t, "R2_gm", "gm", "a  Reflectance alone", True, est)
    panel(axes[1], t, "R2_gm_clima", "gm_clima", "b  Reflectance + climate", False, est)

    # una sola inversion, despues de dibujar los dos paneles: con sharey=True, invertir
    # dentro de cada panel se aplica dos veces y se cancela
    Y, _ = posiciones()
    axes[0].invert_yaxis()
    axes[0].set_ylim(max(Y.values()) + 0.6, -1.0)

    # el rotulo va en el hueco, debajo del separador y encima de la primera fila del grupo:
    # con el eje invertido "debajo" es y mayor
    for nombre, fs in FAMILIAS:
        axes[0].text(-0.108, Y[fs[0]] - 0.52, nombre.upper(), fontsize=8, style="italic",
                     color="#777777", va="center", ha="left", zorder=5)

    # El conteo "n de 22" va en el pie de figura, no dentro de los ejes: es una lectura del
    # grafico, no un dato, y dentro compite con los margenes rotulados.
    gana = {c: ((t[m] > 0) & (t[c] > 0)) for c, m in (("R2_gm", "gm"),
                                                      ("R2_gm_clima", "gm_clima"))}

    h = [Line2D([], [], marker="o", ls="-", ms=6, color=BLUE, lw=2.4,
                label="model gain over geographic null model"),
         Line2D([], [], marker="o", ls="-", ms=5, color=GREY, lw=1.6,
                label="model loss over geographic null model")]
    # dentro del panel b, abajo a la izquierda: con dos estados la leyenda es corta y ahi
    # queda hueco, porque las facetas de esa zona (los ejes de ordenacion) caen a la derecha
    # etiquetas cortas a proposito: el detalle (lon/lat/elevacion, geomediano de seis
    # bandas) va al pie de figura. Con los textos largos la leyenda tapaba las mancuernas
    # de los ejes de ordenacion, que en el panel b caen justo en esa esquina.
    axes[1].legend(handles=h, fontsize=8.5, loc="lower left", framealpha=0.95,
                   handletextpad=0.5, borderpad=0.6, labelspacing=0.45)

    fig.text(0.008, -0.015,
             "Significance of the margin over the null model: * $q$ < 0.05   ** $q$ < 0.01   "
             "*** $q$ < 0.001.  One-sided block bootstrap over the same 20 km blocks that "
             "define the cross-validation,\non the paired difference in squared error; "
             "Benjamini–Hochberg across the 52 tests. See Methods.",
             fontsize=7.5, color="#555555", ha="left", va="top")

    # sin titulo general: el mensaje va en el pie de figura, y los titulos de panel ya
    # dicen que contrasta cada uno
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig03_remote_vs_geography.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig03_remote_vs_geography'}.{{png,pdf}}")
    for col, mar, que in (("R2_gm", "gm", "reflectancia sola"),
                          ("R2_gm_clima", "gm_clima", "reflectancia + clima")):
        g = gana[col]
        print(f"\n{int(g.sum())} de {len(t)} facetas le ganan al modelo nulo geografico "
              f"con {que}:")
        print(t.loc[g, ["grupo", "faceta", "piso_coords", mar, col]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
