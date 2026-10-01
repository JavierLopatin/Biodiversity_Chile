#!/usr/bin/env python3
"""Contraste de estres hidrico: el efecto que no sobrevive su propio control.

El margen sobre el modelo nulo geografico, medido por separado en el tercil SECO y en el
HUMEDO del indice de precipitacion estandarizado. Solo sobre Living Trees, el inventario con
cobertura en todo el pais, asi que el contraste no mezcla procedencias.

    a  margen seco -> margen humedo, faceta por faceta
    b  el control que mata el efecto, y el mecanismo

QUE PASO. El tercil se define como residuo de SPI dentro del ano de censo, para que sea una
anomalia y no un ano. Pero el residuo se tomaba solo respecto a la LATITUD, y en Chile la
precipitacion tiene un gradiente oeste-este tan fuerte como el norte-sur porque la cordillera
lo impone. Medido sobre las 1.815 parcelas con SPI, el tercil seco quedaba 318 m mas alto que
el humedo (729 contra 411) y mas al este: rho del tercil con la elevacion -0,286 y con la
longitud -0,198. Eso no es un contraste entre estados hidricos sino entre la cordillera y el
valle.

Descontando los tres ejes del modelo nulo -- lon, lat, elevacion-- esas correlaciones caen a
-0,029 y +0,044, el contraste de SPI se conserva casi entero (media -0,56 contra -0,20, antes
-0,60 contra -0,16), y el efecto DESAPARECE:

    margen sobre el nulo, detrend solo latitud      19 de 26 mejoran en humedo, media +0,038
    margen sobre el nulo, detrend lon+lat+elevacion  8 de 26,                   media -0,018
    R2 crudo, sin descontar el nulo                 17 de 26, p = 0,17,         media +0,013

EL MECANISMO. Con los terciles limpios, el MODELO NULO mejora en el tercil humedo en 18 de 26
facetas (media +0,032). Cuando llueve mas de lo esperado predice mejor todo, incluidas tres
coordenadas, asi que el margen del sensor no crece. El resultado positivo anterior venia de que
el nulo EMPEORABA en el tercil humedo, y eso era artefacto de que "humedo" significaba
"tierras bajas".

Lo que no se puede hacer es el diseno que compararia anos dentro de un mismo bloque espacial:
de los 531 bloques de 20 km con parcelas, 263 tienen dos o mas anos de censo pero solo DOS
llegan a veinte parcelas, y un R2 por bloque y tercil necesita decenas.

Todo sale de `results/tables/sintesis_lenoso.csv` (`scripts/107`), que ahora guarda las filas
de tercil bajo los dos detrends. Ningun numero esta escrito a mano.

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

BLUE, RED, GREY, DARK = "#4C78A8", "#C0392B", "#9A9A9A", "#333333"
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

#: (detrend, columna, rotulo). Las dos primeras son el efecto; las dos ultimas, su mecanismo.
FILAS_B = [
    ("lat", "neto", "margin gain\ndetrended on latitude only"),
    ("geo", "neto", "margin gain\ndetrended on lon + lat + elevation"),
    ("lat", "piso_delta", "the null model's own gain\ndetrended on latitude only"),
    ("geo", "piso_delta", "the null model's own gain\ndetrended on lon + lat + elevation"),
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

    specs = {dt: m.contraste_estres(f, detrend=dt, escribir=False).set_index("faceta")
             for dt in ("geo", "lat")}
    principal = specs["geo"]
    t = principal.reindex([x for x in ORDEN if x in principal.index]).dropna(subset=["neto"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.6),
                             gridspec_kw={"width_ratios": [2.3, 1.0]})
    ax, axb = axes
    Y, SEP = posiciones()

    # --- a  margen seco -> margen humedo
    for fa, r in t.iterrows():
        i = Y[fa]
        gana = r.neto > 0
        c = BLUE if gana else RED
        ax.annotate("", xy=(r.margen_humedo, i), xytext=(r.margen_seco, i),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2,
                                    alpha=0.9, shrinkA=0, shrinkB=0,
                                    mutation_scale=11))
        ax.plot(r.margen_seco, i, "o", ms=5.5, color=RED, mec="white", mew=0.9, zorder=4)
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

    h = [Line2D([], [], marker="o", ls="none", ms=6, color=RED, label="dry tercile (start)"),
         Line2D([], [], color=BLUE, lw=2.2, label="gains in the wet tercile"),
         Line2D([], [], color=RED, lw=2.2, label="loses in the wet tercile")]
    ax.legend(handles=h, fontsize=8.5, loc="lower left", framealpha=0.95, borderpad=0.6)

    # --- b  robustez: la misma pregunta bajo cuatro especificaciones
    rng = np.random.default_rng(0)
    for j, (dt, col, etiqueta) in enumerate(FILAS_B):
        v = specs[dt][col].to_numpy()
        k, n = int((v > 0).sum()), len(v)
        p = binomtest(k, n, 0.5).pvalue
        axb.scatter(v, j + rng.uniform(-0.13, 0.13, n), s=18, lw=0,
                    color=np.where(v > 0, BLUE, RED), alpha=0.75, zorder=3)
        axb.plot([v.mean()] * 2, [j - 0.26, j + 0.26], color=DARK, lw=2.2, zorder=4)
        axb.text(0.985, j - 0.36, f"{k}/{n}   $p$ = {p:.2f}", transform=
                 axb.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                 color="#555555")
    axb.axvline(0, color=DARK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    axb.set_yticks(range(len(FILAS_B)))
    axb.set_yticklabels([e[2] for e in FILAS_B], fontsize=8)
    axb.invert_yaxis()
    axb.set_xlabel("change from the dry to the wet tercile")
    axb.set_title("b  The effect is the elevation confound", loc="left", fontweight="bold")
    axb.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    axb.set_axisbelow(True)
    axb.set_ylim(len(FILAS_B) - 0.45, -0.75)
    axb.axhline(1.5, color="#BBBBBB", lw=0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig03_drought_contrast.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig03_drought_contrast'}.{{png,pdf}}")
    for dt, col, et in FILAS_B:
        v = specs[dt][col]
        k, n = int((v > 0).sum()), len(v)
        print(f"detrend={dt:4s} {col:11s} {k:2d}/{n}  p={binomtest(k, n, 0.5).pvalue:.3f}"
              f"  media {v.mean():+.4f}")
    b = specs["geo"].bruto
    print(f"detrend=geo  bruto       {int((b>0).sum()):2d}/{len(b)}  "
          f"p={binomtest(int((b>0).sum()), len(b), 0.5).pvalue:.3f}  media {b.mean():+.4f}")


if __name__ == "__main__":
    main()
