#!/usr/bin/env python3
"""Variante de la figura 3: los dos modelos en un solo panel, faceta por faceta.

Misma pregunta y mismos numeros que `scripts/115`, otra puesta en pagina. Alla los dos modelos
viven en paneles separados y el lector compara saltando la vista entre ejes; aqui comparten la
fila, asi que la comparacion es local -- se lee de un vistazo cuanto de la ganancia aparece solo
al sumar clima.

El modelo nulo geografico deja de ser un punto y pasa a ser una MARCA VERTICAL que cruza la
fila entera: con dos modelos por faceta el circulo blanco competia con ellos por la atencion,
y lo que el nulo tiene que hacer es ser la referencia, no un tercer dato.

    marca vertical    modelo nulo geografico (lon, lat, elevacion)
    circulo lleno     reflectancia sola
    triangulo lleno   reflectancia + clima

El color dice si ese modelo le gana al nulo, y es distinto para cada uno -- azul la
reflectancia, naranjo el clima-- porque la pregunta del paper no es cuantas facetas ganan sino
CUAL de los dos modelos las gana. Gris cuando pierde, para los dos.

Aqui "gana" es solo el signo del margen, a diferencia de `scripts/115`, que ademas exige R2
absoluto positivo. Ese segundo criterio existe porque cuando el nulo es negativo superarlo solo
dice "menos malo que la geografia", y el modelo sigue sin predecir (TD q1/q2 y PD q2 sin clima).
Pero codificarlo en el color producia mancuernas grises que se extienden a la DERECHA del nulo,
que es exactamente lo que la leyenda llama ganancia: una regla escondida contradiciendo a la
leyenda. La zona muerta se dibuja en vez de codificarse -- el fondo sombreado marca R2 <= 0,
donde el modelo no explica nada aunque le gane al nulo-- y asi las dos cosas se ven por separado
en vez de colapsarse en un color.

Uso:
    python scripts/117_figure_fig03_variante.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables" / "margen_sin_clima.csv"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREY, DARK = "#4C78A8", "#E8832A", "#9A9A9A", "#333333"
DPI = 300
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9.5})

#: (columna de R2 absoluto, columna de margen, marcador, color de ganancia, desplazamiento en y)
#: El desplazamiento separa las dos mancuernas dentro de la fila; sin el se pisan y la figura
#: deja de poder leerse justo donde los dos modelos difieren poco, que es la mitad de los casos.
MODELOS = [
    ("R2_gm", "gm", "o", BLUE, -0.20),
    ("R2_gm_clima", "gm_clima", "^", ORANGE, +0.20),
]

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
HUECO = 1.2          #: mas hueco que en `scripts/115`: cada fila ocupa ahora dos mancuernas

LAB = {
    "lcbd_count_sorensen": "LCBD Sørensen", "lcbd_pa_unified": "LCBD (p/a)",
    "lcbd_freq_unified": "LCBD (freq.)",
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


def posiciones() -> tuple[dict, list]:
    y, sep, cur = {}, [], 0.0
    for k, (_, fs) in enumerate(FAMILIAS):
        if k:
            sep.append(cur - HUECO / 2)
            cur += HUECO
        for fa in fs:
            y[fa] = cur
            cur += 1.0
    return y, sep


def main() -> None:
    t = pd.read_csv(TAB)
    t = t[t.faceta.isin(ORDEN)].copy()
    t["R2_gm_clima"] = t.piso_coords + t.gm_clima
    t["orden"] = t.faceta.map({k: i for i, k in enumerate(ORDEN)})
    t = t.sort_values("orden").reset_index(drop=True)

    Y, SEP = posiciones()
    fig, ax = plt.subplots(figsize=(9.2, 9.6))

    for _, r in t.iterrows():
        i = Y[r.faceta]
        # la marca del nulo cruza la fila entera, por detras de las dos mancuernas
        ax.plot([r.piso_coords] * 2, [i - 0.42, i + 0.42], color=DARK, lw=1.7, zorder=3,
                solid_capstyle="butt")
        for col, mar, mk, c_win, dy in MODELOS:
            gana = bool(r[mar] > 0)
            c = c_win if gana else GREY
            ax.plot([r.piso_coords, r[col]], [i + dy] * 2, color=c,
                    lw=2.6 if gana else 1.5, alpha=1.0 if gana else 0.55, zorder=2,
                    solid_capstyle="round")
            ax.plot(r[col], i + dy, mk, ms=7 if gana else 5.5, color=c,
                    mec="white", mew=0.8, zorder=4)
            if gana:
                ax.text(max(r[col], r.piso_coords) + 0.018, i + dy, f"+{r[mar]:.3f}",
                        va="center", fontsize=8, color=c, fontweight="bold")

    # zona muerta: a la izquierda de 0 el modelo no explica nada, le gane o no al nulo
    ax.axvspan(-0.13, 0.0, color="#E9E9E9", zorder=0)

    for sy in SEP:
        ax.axhline(sy, color="#BBBBBB", lw=0.8, zorder=1)
    ax.axvline(0, color=DARK, lw=0.7, zorder=1)

    ax.set_yticks([Y[k] for k in t.faceta])
    ax.set_yticklabels([LAB.get(k, k) for k in t.faceta])
    ax.set_xlabel("out-of-fold $R^2$, centred within 2° latitude bins")
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.13, 0.82)
    ax.text(-0.125, max(Y.values()) + 0.45, "$R^2 \\leq 0$: no predictive skill",
            fontsize=7.5, style="italic", color="#8A8A8A", va="center", ha="left",
            rotation=0, zorder=5)
    ax.invert_yaxis()
    ax.set_ylim(max(Y.values()) + 0.8, -1.2)

    for nombre, fs in FAMILIAS:
        ax.text(-0.122, Y[fs[0]] - 0.75, nombre.upper(), fontsize=8, style="italic",
                color="#777777", va="center", ha="left", zorder=5)

    h = [Line2D([], [], ls="-", lw=1.7, color=DARK,
                label="geographic null model (longitude, latitude, elevation)"),
         Line2D([], [], marker="o", ls="-", ms=6.5, lw=2.4, color=BLUE,
                label="reflectance alone — gain"),
         Line2D([], [], marker="^", ls="-", ms=6.5, lw=2.4, color=ORANGE,
                label="reflectance + climate — gain"),
         Line2D([], [], marker="s", ls="-", ms=5.5, lw=1.5, color=GREY, alpha=0.55,
                label="loss over the null model")]
    # arriba a la derecha pero una fila mas abajo: abajo tapaba las tres ultimas filas de
    # Isomap y pegada al techo tapaba el margen de la riqueza cruda, la fila mas larga de la
    # figura. A la altura de las TD el lado derecho esta vacio.
    ax.legend(handles=h, fontsize=8.5, loc="upper right", bbox_to_anchor=(0.998, 0.935),
              framealpha=0.95, handletextpad=0.6, borderpad=0.6, labelspacing=0.45)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig03alt_remote_vs_geography.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig03alt_remote_vs_geography'}.{{png,pdf}}")
    for col, mar, _, _, _ in MODELOS:
        g = (t[mar] > 0) & (t[col] > 0)
        print(f"{col:14s} gana en {int(g.sum()):2d} de {len(t)}")


if __name__ == "__main__":
    main()
