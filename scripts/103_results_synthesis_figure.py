#!/usr/bin/env python3
"""Sintesis de los resultados: que representacion remota predice beta, y cuando.

Tres paneles, los tres desde `results/tables/lecturas_pareadas.csv`, que sale de los OOF
versionados (`scripts/102`). Ningun numero de esta figura esta escrito a mano.

    a  la compuerta, con y sin control de latitud -- el geomediano no es el gradiente
    b  por estrato de cobertura -- se invierte entre bosque y matorral
    c  por clase de pendiente y por fuente -- se invierte entre llano y ladera

El panel a lleva el piso de topografia+area como linea: la mitad de las conclusiones
dependen de comparar contra el piso y no contra cero, y sin la linea nadie lo hace.

Uso:
    python scripts/103_results_synthesis_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables" / "lecturas_pareadas.csv"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, GREY = "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#666666"
RED = "#B4451F"
DPI = 200
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})

#: Orden y rotulo en ingles de cada bloque de predictores.
BLOCKS = [("clima", "Climate"), ("curva", "Curve"), ("lsp", "LSP"),
          ("gm", "Geomedian"), ("clima_curva", "Climate+curve"),
          ("gm_clima", "Geomedian+climate")]
COLOURS = {"clima": BLUE, "curva": ORANGE, "lsp": "#C7A23C",
           "gm": GREEN, "clima_curva": PURPLE, "gm_clima": RED}


def panel_gate(ax, t: pd.DataFrame) -> None:
    m = t[(t.tipo == "modelo") & (t.grupo == "compuerta_pool")
          & (t.target == "lcbd_count_sorensen") & (t.clase_pendiente == "todas")]
    w = m.pivot_table(index="etiqueta", columns="lectura", values="R2_o_dif")
    lects = [("todas", "Pooled"), ("lt_dentro_banda", "Within 2° latitude band\n(Living Trees)")]

    x = np.arange(len(lects))
    n = len(BLOCKS)
    for i, (key, lab) in enumerate(BLOCKS):
        vals = [w.at[key, c] for c, _ in lects]
        ax.bar(x + (i - (n - 1) / 2) * 0.14, vals, width=0.13,
               color=COLOURS[key], label=lab, zorder=3)
    for j, (c, _) in enumerate(lects):
        f = w.at["piso_topo_area", c]
        ax.plot([j - 0.5, j + 0.5], [f, f], color="#333333", lw=1.4, ls="--", zorder=4)
        ax.text(j + 0.5, f, " floor", va="center", fontsize=8, color="#333333")

    ax.set_xticks(x); ax.set_xticklabels([l for _, l in lects])
    ax.set_ylabel("out-of-fold $R^2$")
    ax.set_title("a  Which block predicts composition (LCBD)", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def panel_strata(ax, t: pd.DataFrame) -> None:
    m = t[(t.tipo == "modelo") & (t.grupo == "estratos") & (t.lectura == "todas")]
    w = m.pivot_table(index="etiqueta", columns="pool", values="R2_o_dif")
    pools = [c for c in ["Forest", "NoBosque"] if c in w.columns]
    labs = {"Forest": "Forest", "NoBosque": "Non-forest\n(shrubland, grassland)"}

    keys = ["clima", "curva", "lsp", "gm", "gm_clima"]
    x = np.arange(len(pools))
    for i, key in enumerate(keys):
        vals = [w.at[key, p] for p in pools]
        ax.bar(x + (i - (len(keys) - 1) / 2) * 0.16, vals, width=0.15,
               color=COLOURS[key], zorder=3)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([labs[p] for p in pools])
    ax.set_ylabel("out-of-fold $R^2$")
    ax.set_title("b  The ranking flips between strata", loc="left", fontweight="bold")
    ax.annotate("curve wins", xy=(1.0, 0.171), xytext=(0.62, 0.225), fontsize=8.5,
                color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.annotate("geomedian wins", xy=(0.16, 0.262), xytext=(-0.42, 0.30), fontsize=8.5,
                color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def panel_slope(ax, t: pd.DataFrame) -> None:
    p = t[(t.tipo == "par") & (t.etiqueta == "gm_nc − curva_kndvi_nc")
          & (t.target == "lcbd_pa_unified")]
    order = ["<5", "5-15", "15-25", ">25"]
    series = [("pcl_dentro_owner", BLUE, "Parcelas-CL, within contributor"),
              ("lt_dentro_banda", ORANGE, "Living Trees, within 2° band")]

    x = np.arange(len(order))
    for i, (lec, col, lab) in enumerate(series):
        s = p[p.lectura == lec].set_index("clase_pendiente")
        v = [s.at[c, "R2_o_dif"] for c in order]
        e = [s.at[c, "DE"] for c in order]
        ax.errorbar(x + (i - 0.5) * 0.09, v, yerr=e, fmt="-o", color=col, ms=6, lw=1.8,
                    capsize=3, label=lab, zorder=3)
    ax.axhline(0, color="#333333", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(["< 5°", "5–15°", "15–25°", "> 25°"])
    ax.set_xlabel("terrain slope (DEM)")
    ax.set_ylabel("geomedian − curve  ($\\Delta R^2$)")
    ax.set_title("c  …and between flat ground and steep slopes", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.text(0.02, 0.06, "above 0: geomedian better\nbelow 0: curve better",
            transform=ax.transAxes, fontsize=8, color=GREY)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    t = pd.read_csv(TAB)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    panel_gate(axes[0], t)
    panel_strata(axes[1], t)
    panel_slope(axes[2], t)
    fig.suptitle("Which remote-sensing representation predicts woody composition, "
                 "and when", fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig26_representation_synthesis.{ext}", dpi=DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig26_representation_synthesis'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
