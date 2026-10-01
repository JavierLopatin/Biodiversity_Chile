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

El contraste entre los dos paneles ES el resultado. En (a) solo cuatro facetas superan a la
geografia: las tres variantes de LCBD y la riqueza cruda. En (b) muchas mas, porque el clima
las levanta -- pero el clima sale de una grilla interpolada desde estaciones y dentro de una
banda de latitud sigue siendo en buena parte posicion. Un bloque que solo gana con clima no ha
demostrado que el sensor aporte.

La separacion tiene lectura directa: lo que gana en (a) son escalares DEL RODAL -- cuan inusual
es esta parcela, cuantas especies hay. Lo que pierde son posiciones en un pool regional: que
tipo de comunidad, que estructura filogenetica, que falta del pool. Un sensor ve el rodal, no la
historia biogeografica.

Los ejes de Isomap quedan fuera: por decision del 2026-09-30 la ordenacion del texto principal
es PCoA, e Isomap va al suplemento.

Todo sale de `results/tables/margen_sin_clima.csv` (`scripts/107`). Ningun numero esta escrito
a mano.

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
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, PURPLE, GREY = "#4C78A8", "#54A24B", "#B279A2", "#888888"
DARK, WIN = "#333333", "#B4451F"
DPI = 300
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9.5})

COL = {"composicion": BLUE, "riqueza": GREEN, "filogenetica": PURPLE}
GRUPO_EN = {"composicion": "Composition", "riqueza": "Richness",
            "filogenetica": "Phylogenetic"}
#: Orden por FAMILIA de faceta, no por margen. Ordenar por el resultado agrupa facetas que no
#: se comparan entre si (una TD junto a un eje de ordenacion) y rompe la lectura: el lector
#: quiere ver las tres TD juntas, las tres PD juntas, y despues las de composicion. El orden
#: dentro de cada familia es q0, q1, q2, que es el orden del parametro de Hill.
#: Los seis ejes de Isomap no aparecen: por decision del 2026-09-30 la ordenacion del texto
#: principal es PCoA e Isomap va al suplemento.
FAMILIAS = [
    ("Taxonomic richness", ["hill_q0_unified", "td_inext_q0", "td_inext_q1", "td_inext_q2"]),
    ("Phylogenetic", ["pd_inext_q0", "pd_inext_q1", "pd_inext_q2",
                      "mpd_unified", "mntd_unified",
                      "ses_pd_unified", "ses_mpd_unified", "ses_mntd_unified"]),
    ("Compositional uniqueness", ["lcbd_count_sorensen", "lcbd_pa_unified",
                                  "lcbd_freq_unified"]),
    ("Dark diversity", ["dark_n_unified"]),
    ("Floristic composition (ordination)", ["pcoa1_pa_unified", "pcoa2_pa_unified",
                                            "pcoa1_freq_unified", "pcoa2_freq_unified"]),
]
ORDEN = [f for _, fs in FAMILIAS for f in fs]

#: Rotulos en ingles.
LAB = {
    "lcbd_count_sorensen": "LCBD Sørensen", "lcbd_pa_unified": "LCBD (p/a)",
    "lcbd_freq_unified": "LCBD (freq.)", "pcoa1_pa_unified": "PCoA 1 (p/a)",
    "pcoa2_pa_unified": "PCoA 2 (p/a)", "pcoa1_freq_unified": "PCoA 1 (freq.)",
    "pcoa2_freq_unified": "PCoA 2 (freq.)",
    "hill_q0_unified": "Richness $q_0$ (raw)", "dark_n_unified": "Dark diversity",
    "td_inext_q0": "TD $q_0$ (cov.-std.)", "td_inext_q1": "TD $q_1$ (cov.-std.)",
    "td_inext_q2": "TD $q_2$ (cov.-std.)",
    "mpd_unified": "MPD", "mntd_unified": "MNTD", "ses_pd_unified": "SES PD",
    "ses_mpd_unified": "SES MPD", "ses_mntd_unified": "SES MNTD",
    "pd_inext_q0": "PD $q_0$ (cov.-std.)", "pd_inext_q1": "PD $q_1$ (cov.-std.)",
    "pd_inext_q2": "PD $q_2$ (cov.-std.)",
}


def panel(ax, d: pd.DataFrame, col_r2: str, titulo: str, marca_ganadores: bool) -> None:
    """Mancuerna: piso de coordenadas -> bloque, una fila por faceta.

    La mancuerna y no barras porque lo que importa es la DISTANCIA entre dos numeros, no el
    valor de ninguno: una barra invita a leer la altura, que aqui no significa nada sin el piso
    al lado.
    """
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        # Tres estados, no dos. Superar al piso NO basta: si el piso de coordenadas es
        # negativo, un margen positivo solo dice "menos malo que la geografia" y el modelo
        # sigue sin predecir. Pasa en TD q1/q2 y PD q0/q2, con piso de -0,005 a -0,073.
        # Pintarlas como ganadoras seria reintroducir en la figura la trampa que la tabla
        # evita exigiendo margen positivo Y R2 absoluto positivo.
        gana = bool(r.remota_propia)
        margen_vacio = (r[col_r2] > r.piso_coords) and not gana
        c = COL[r.grupo] if gana else GREY
        ax.plot([r.piso_coords, r[col_r2]], [i, i], color=c,
                lw=2.8 if gana else 1.6, alpha=1.0 if gana else 0.5, zorder=2,
                ls=":" if margen_vacio else "-", solid_capstyle="round")
        ax.plot(r.piso_coords, i, "o", ms=5.5, color="white", mec=DARK, mew=1.4, zorder=3)
        ax.plot(r[col_r2], i, "o", ms=7 if gana else 5, color=c,
                mec="white", mew=0.8, zorder=4)

    if marca_ganadores:
        for i, (_, r) in enumerate(d.iterrows()):
            if r.remota_propia:
                ax.text(max(r[col_r2], r.piso_coords) + 0.02, i,
                        f"+{r[col_r2] - r.piso_coords:.3f}", va="center", fontsize=8.5,
                        color=WIN, fontweight="bold")

    # separador entre familias: sin el, veinte filas seguidas se leen como una lista plana
    i = 0
    for nombre, fs in FAMILIAS:
        i += len(fs)
        if i < len(d):
            ax.axhline(i - 0.5, color="#BBBBBB", lw=0.8, ls="-", zorder=1)

    ax.axvline(0, color=DARK, lw=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LAB.get(k, k) for k in d.faceta])
    ax.set_xlabel("out-of-fold $R^2$, centred within 2° latitude bins")
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.12, 0.78)


def main() -> None:
    t = pd.read_csv(TAB)
    t = t[t.faceta.isin(LAB)].copy()          # fuera los ejes Isomap
    t["R2_gm_clima"] = t.piso_coords + t.gm_clima
    t["orden"] = t.faceta.map({k: i for i, k in enumerate(ORDEN)})
    t = t.sort_values("orden").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), sharey=True)
    panel(axes[0], t, "R2_gm", "a  Reflectance alone", True)
    panel(axes[1], t, "R2_gm_clima", "b  Reflectance + climate", False)

    # una sola inversion, despues de dibujar los dos paneles: con sharey=True, invertir
    # dentro de cada panel se aplica dos veces y se cancela
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(t) - 0.4, -1.3)      # hueco arriba para el primer rotulo

    i = 0
    for nombre, fs in FAMILIAS:
        axes[0].text(-0.105, i - 0.62, nombre.upper(), fontsize=7.8, style="italic",
                     color="#777777", va="center", ha="left", zorder=5)
        i += len(fs)

    # El conteo "n de 20" va en el pie de figura, no dentro de los ejes: es una lectura del
    # grafico, no un dato, y dentro compite con los margenes rotulados.
    n = int(t.remota_propia.sum())

    h = [Line2D([], [], marker="o", ls="none", ms=6, color="white", mec=DARK, mew=1.4,
                label="geography floor: longitude, latitude, elevation"),
         Line2D([], [], ls="none", label="reflectance: six-band geometric median")]
    h += [Line2D([], [], marker="o", ls="none", ms=6, color=COL[g],
                 label=GRUPO_EN[g] + " — reflectance wins") for g in COL]
    h += [Line2D([], [], marker="o", ls=":", ms=5, color=GREY, lw=1.6,
                 label="margin over a negative floor ($R^2<0$)"),
          Line2D([], [], marker="o", ls="-", ms=5, color=GREY, lw=1.6,
                 label="reflectance loses to geography")]
    axes[1].legend(handles=h, fontsize=8.5, loc="lower right", framealpha=0.95)

    fig.suptitle("What reflectance adds over knowing where you are",
                 fontsize=14.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig03_remote_vs_geography.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig03_remote_vs_geography'}.{{png,pdf}}")
    print(f"\n{n} de {len(t)} facetas le ganan a tres coordenadas sin clima:")
    print(t.loc[t.remota_propia, ["grupo", "faceta", "piso_coords", "gm", "R2_gm"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
