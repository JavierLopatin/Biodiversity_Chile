#!/usr/bin/env python3
"""Figura del suplemento: Isomap contra PCoA como ordenacion de la composicion.

Isomap preserva mucha mas estructura de la disimilitud que el PCoA -- con DOS ejes supera a un
PCoA de OCHO (techo de reconstruccion 0,577 contra 0,502; `scripts/113`). La pregunta de esta
figura es otra: si esa ventaja de representacion se traduce en ventaja de PREDICCION.

No se traduce de forma robusta, y por eso el texto principal se queda con PCoA (decision del
2026-09-30). El criterio prerregistrado era que el margen de Isomap sobre el modelo nulo
geografico superara al de PCoA; segun se calcule el R2 dentro de bin sobre el pool entero o
solo sobre Living Trees, Isomap gana o pierde, y el desacuerdo (0,04) es del mismo orden que
los margenes comparados. A eso se suma que los ejes 2 y 3 de p/a son inestables al parametro k
(rho 0,70 y 0,83 entre k=30 y k=80).

Mismo lenguaje visual que fig03: mancuerna del modelo nulo geografico al bloque, dos paneles
(reflectancia sola y reflectancia + clima), y las mismas reglas de color.

Uso:
    python scripts/116_figure_isomap_supplement.py
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

BLUE, GREY, DARK, WIN = "#4C78A8", "#888888", "#333333", "#B4451F"
ORANGE = "#F58518"
DPI = 300
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9.5})

FAMILIAS = [
    ("Isomap, presence/absence", ["isomap1_pa_unified", "isomap2_pa_unified",
                                  "isomap3_pa_unified"]),
    ("PCoA, presence/absence", ["pcoa1_pa_unified", "pcoa2_pa_unified"]),
    ("Isomap, frequency", ["isomap1_freq_unified", "isomap2_freq_unified",
                           "isomap3_freq_unified"]),
    ("PCoA, frequency", ["pcoa1_freq_unified", "pcoa2_freq_unified"]),
]
ORDEN = [f for _, fs in FAMILIAS for f in fs]
LAB = {f"{m}{i}_{v}_unified": f"{m.capitalize() if m == 'isomap' else 'PCoA'} {i}"
       for m in ("isomap", "pcoa") for i in (1, 2, 3) for v in ("pa", "freq")}
HUECO = 1.0


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


def panel(ax, d: pd.DataFrame, col: str, titulo: str, rotula: bool) -> None:
    Y, SEP = posiciones()
    y = [Y[k] for k in d.faceta]
    for i, (_, r) in zip(y, d.iterrows()):
        gana = r[col] > r.piso_coords
        c = (ORANGE if r.faceta.startswith("isomap") else BLUE) if gana else GREY
        ax.plot([r.piso_coords, r[col]], [i, i], color=c, lw=2.8 if gana else 1.6,
                alpha=1.0 if gana else 0.5, zorder=2, solid_capstyle="round")
        ax.plot(r.piso_coords, i, "o", ms=5.5, color="white", mec=DARK, mew=1.4, zorder=3)
        ax.plot(r[col], i, "o", ms=7 if gana else 5, color=c, mec="white", mew=0.8, zorder=4)
        if rotula:
            dif = r[col] - r.piso_coords
            hi, lo = max(r[col], r.piso_coords), min(r[col], r.piso_coords)
            # cerca del borde derecho la etiqueta se sale; va a la izquierda de la mancuerna
            fuera = hi + 0.02 > 0.68
            ax.text(lo - 0.02 if fuera else hi + 0.02, i, f"{dif:+.3f}", va="center",
                    ha="right" if fuera else "left", fontsize=8.5,
                    color=WIN if dif > 0 else GREY,
                    fontweight="bold" if dif > 0 else "normal")
    for sy in SEP:
        ax.axhline(sy, color="#BBBBBB", lw=0.8, zorder=1)
    ax.axvline(0, color=DARK, lw=0.7, zorder=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LAB.get(k, k) for k in d.faceta])
    ax.set_xlabel("out-of-fold $R^2$, centred within 2° latitude bins")
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.05, 0.75)


def main() -> None:
    t = pd.read_csv(TAB)
    t = t[t.faceta.isin(ORDEN)].copy()
    t["R2_gm_clima"] = t.piso_coords + t.gm_clima
    t["orden"] = t.faceta.map({k: i for i, k in enumerate(ORDEN)})
    t = t.sort_values("orden").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
    panel(axes[0], t, "R2_gm", "a  Reflectance alone", True)
    panel(axes[1], t, "R2_gm_clima", "b  Reflectance + climate", True)

    Y, _ = posiciones()
    axes[0].invert_yaxis()
    axes[0].set_ylim(max(Y.values()) + 0.6, -1.0)
    for nombre, fs in FAMILIAS:
        axes[0].text(-0.042, Y[fs[0]] - 0.52, nombre.upper(), fontsize=8, style="italic",
                     color="#777777", va="center", ha="left", zorder=5)

    h = [Line2D([], [], marker="o", ls="none", ms=6, color="white", mec=DARK, mew=1.4,
                label="geographic null model: longitude, latitude, elevation"),
         Line2D([], [], marker="o", ls="none", ms=6, color=ORANGE, label="Isomap axis"),
         Line2D([], [], marker="o", ls="none", ms=6, color=BLUE, label="PCoA axis"),
         Line2D([], [], marker="o", ls="none", ms=5, color=GREY,
                label="loses to the geographic null model")]
    fig.legend(handles=h, fontsize=9, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=False, columnspacing=2.2)

    fig.suptitle("Isomap preserves more of the dissimilarity structure, "
                 "but neither ordination beats geography on reflectance alone",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figS8_isomap_vs_pcoa.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'figS8_isomap_vs_pcoa'}.{{png,pdf}}")
    print(t[["faceta", "piso_coords", "R2_gm", "gm", "gm_clima"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
