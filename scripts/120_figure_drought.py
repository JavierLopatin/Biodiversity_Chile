#!/usr/bin/env python3
"""Figura del contraste de estres hidrico: cuando funciona el sensor.

El margen sobre el modelo nulo geografico, medido por separado en el tercil SECO y en el
HUMEDO del indice de precipitacion estandarizado. Se mide solo sobre Living Trees, que es el
inventario con cobertura en todo el pais, asi que el contraste no mezcla procedencias.

    a  margen seco -> margen humedo, faceta por faceta
    b  robustez: el mismo contraste bajo cuatro especificaciones

El panel b existe porque el resultado NO es igual de solido en sus dos lecturas, y la figura
tiene que decirlo en vez de dejar que el lector suponga:

    DIRECCION, robusta.    Entre 18 y 21 de 26 facetas predicen mejor en el tercil humedo
                           segun la especificacion, con p entre 0,038 y 0,001 y neto medio
                           entre +0,025 y +0,063. Las cuatro apuntan al mismo lado.
    ORDEN, no robusto.     Que faceta mejora mas depende del nulo que se descuente: entre el
                           nulo geografico y el piso de topografia + area la correlacion de
                           rangos cae a 0,13-0,29. La metrica, en cambio, casi no importa
                           (0,94 entre R2 agrupado y R2 dentro de banda).

Por eso el panel a se lee como "hacia donde se mueven las facetas" y no como un ranking, y el
texto no debe nombrar cual mejora mas.

Todo sale de `results/tables/sintesis_lenoso.csv` (`scripts/107`). Ningun numero esta escrito
a mano: las cuatro especificaciones se recalculan aqui llamando a la misma funcion que produce
la tabla del paper.

Uso:
    python scripts/120_figure_drought.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables" / "sintesis_lenoso.csv"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, GREY, DARK = "#4C78A8", "#9A9A9A", "#333333"
DPI = 300
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9.5})

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
HUECO = 1.0

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

#: (nulo, metrica, rotulo). La primera es la del texto principal.
ESPECIFICACIONES = [
    ("coords", "R2_en_banda", "geographic null,\nwithin bins"),
    ("coords", "R2", "geographic null,\npooled"),
    ("piso", "R2_en_banda", "topography + area,\nwithin bins"),
    ("piso", "R2", "topography + area,\npooled"),
]


def cargar_107():
    spec = importlib.util.spec_from_file_location("s107", ROOT / "scripts" /
                                                  "107_sintesis_lenoso.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s107"] = m
    spec.loader.exec_module(m)
    return m


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
    m = cargar_107()
    f = pd.read_csv(TAB)

    specs = {}
    for nulo, met, _ in ESPECIFICACIONES:
        o = m.contraste_estres(f, nulo=nulo, metrica=met,
                               escribir=False).set_index("faceta")
        specs[(nulo, met)] = o
    principal = specs[("coords", "R2_en_banda")]
    t = principal.reindex([x for x in ORDEN if x in principal.index]).dropna(subset=["neto"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.6),
                             gridspec_kw={"width_ratios": [2.3, 1.0]})
    ax, axb = axes
    Y, SEP = posiciones()

    # --- a  margen seco -> margen humedo
    for fa, r in t.iterrows():
        i = Y[fa]
        gana = r.neto > 0
        c = BLUE if gana else GREY
        ax.annotate("", xy=(r.margen_humedo, i), xytext=(r.margen_seco, i),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4 if gana else 1.5,
                                    alpha=1.0 if gana else 0.55, shrinkA=0, shrinkB=0,
                                    mutation_scale=11))
        ax.plot(r.margen_seco, i, "o", ms=5, color="white", mec=c, mew=1.5,
                alpha=1.0 if gana else 0.6, zorder=4)
    for sy in SEP:
        ax.axhline(sy, color="#BBBBBB", lw=0.8, zorder=1)
    ax.axvline(0, color=DARK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks([Y[k] for k in t.index])
    ax.set_yticklabels([LAB.get(k, k) for k in t.index])
    ax.set_xlabel("margin over the geographic null model, within 2° latitude bins")
    ax.set_title("a  From the dry to the wet tercile", loc="left", fontweight="bold")
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    # margen explicito: con `annotate` las flechas no entran en el autoescalado y se cortaban
    lo = min(t.margen_seco.min(), t.margen_humedo.min())
    hi = max(t.margen_seco.max(), t.margen_humedo.max())
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.invert_yaxis()
    ax.set_ylim(max(Y.values()) + 0.6, -1.0)
    for nombre, fs in FAMILIAS:
        ax.text(ax.get_xlim()[0] + 0.01, Y[fs[0]] - 0.52, nombre.upper(), fontsize=8,
                style="italic", color="#777777", va="center", ha="left", zorder=5)

    h = [Line2D([], [], marker="o", ls="none", ms=5, color="white", mec=DARK, mew=1.5,
                label="dry tercile"),
         Line2D([], [], color=BLUE, lw=2.4, label="better in the wet tercile"),
         Line2D([], [], color=GREY, lw=1.5, alpha=0.6, label="better in the dry tercile")]
    ax.legend(handles=h, fontsize=8.5, loc="lower left", framealpha=0.95, borderpad=0.6)

    # --- b  robustez: la misma pregunta bajo cuatro especificaciones
    rng = np.random.default_rng(0)
    for j, (nulo, met, etiqueta) in enumerate(ESPECIFICACIONES):
        v = specs[(nulo, met)].neto.to_numpy()
        k, n = int((v > 0).sum()), len(v)
        p = binomtest(k, n, 0.5, alternative="greater").pvalue
        axb.scatter(v, j + rng.uniform(-0.13, 0.13, n), s=18, lw=0,
                    color=np.where(v > 0, BLUE, GREY), alpha=0.75, zorder=3)
        axb.plot([v.mean()] * 2, [j - 0.26, j + 0.26], color=DARK, lw=2.2, zorder=4)
        axb.text(0.985, j - 0.34, f"{k}/{n}   $p$ = {p:.3f}", transform=
                 axb.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                 color="#555555")
    axb.axvline(0, color=DARK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    axb.set_yticks(range(len(ESPECIFICACIONES)))
    axb.set_yticklabels([e[2] for e in ESPECIFICACIONES], fontsize=8.5)
    axb.invert_yaxis()
    axb.set_xlabel("change in margin, dry $\\rightarrow$ wet")
    axb.set_title("b  Does the direction survive the choice of null?", loc="left",
                  fontweight="bold")
    axb.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    axb.set_axisbelow(True)
    axb.set_ylim(len(ESPECIFICACIONES) - 0.45, -0.75)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig03_drought_contrast.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig03_drought_contrast'}.{{png,pdf}}")
    for (nulo, met, et) in ESPECIFICACIONES:
        v = specs[(nulo, met)].neto
        k, n = int((v > 0).sum()), len(v)
        print(f"{nulo:7s} {met:12s}  {k:2d}/{n}  p={binomtest(k, n, 0.5, 'greater').pvalue:.4f}"
              f"  neto medio {v.mean():+.3f}")


if __name__ == "__main__":
    main()
