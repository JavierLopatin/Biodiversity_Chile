#!/usr/bin/env python3
"""Biplot del PCA de LSP+estabilidad interanual, con las facetas de diversidad
como vectores suplementarios (`scripts/48_lsp_stability_pca.R`).

Circulo de correlaciones (radio 1): las cargas de `prcomp` se reescalan por `sdev` para
quedar en la misma escala [-1,1] que las correlaciones de las facetas -- asi ambos grupos
de flechas son comparables en el mismo grafico. Solo se muestran las cargas LSP+estabilidad
de mayor magnitud (de 94) para que siga siendo legible; las facetas de diversidad (15) se
muestran todas.

Texto de la figura en ingles (titulo, ejes, etiquetas) para reuso directo en el paper --
codigo/comentarios quedan en espanol, mismo criterio que el resto del repo.

Uso:
    python scripts/49_pca_biplot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import figures as fg   # noqa: E402

TABLES = ROOT / "results" / "tables"
TOP_N_LOADINGS = 10
#: mostrar solo kNDVI + estabilidad (no index-especifica) -- los 5 indices dan vectores casi
#: duplicados para la misma metrica LSP (sos_ndvi ~ sos_evi ~ sos_kndvi ...), mostrarlos
#: todos amontona las etiquetas sin agregar nada. kNDVI es el indice foco de la sesion.
LOADING_PREFIX_KEEP = "kndvi_"
STABILITY_COLS = {"rmse_total", "rmse_sos", "rmse_pos", "rmse_eos"}

#: abreviaturas estandar de fenologia (TIMESAT/PhenoSensing), sin el prefijo de indice --
#: la figura ya deja claro en el titulo que es kNDVI, repetirlo en cada etiqueta solo suma
#: ruido visual.
LSP_LABELS = {
    "sos": "SOS", "pos": "POS", "eos": "EOS",
    "vsos": "vSOS", "vpos": "vPOS", "veos": "vEOS",
    "los": "LOS", "msp": "MSP", "mau": "MAU",
    "vmsp": "vMSP", "vmau": "vMAU", "ampl": "Ampl",
    "ios": "IOS", "rog": "ROG", "ros": "ROS",
    "sw": "SW", "trough": "Trough", "mos": "MOS",
}
STABILITY_LABELS = {
    "rmse_total": "RMSE", "rmse_sos": "RMSEsos",
    "rmse_pos": "RMSEpos", "rmse_eos": "RMSEeos",
}
FACET_LABELS = {
    "hill_q0": "Richness (q0)", "hill_q1": "Diversity (q1)", "hill_q2": "Diversity (q2)",
    "pcoa1_pa": "PCoA1 (P/A)", "pcoa2_pa": "PCoA2 (P/A)", "lcbd_pa": "LCBD (P/A)",
    "pcoa1_cover": "PCoA1 (cover)", "pcoa2_cover": "PCoA2 (cover)",
    "lcbd_cover": "LCBD (cover)",
    "mpd": "MPD", "mntd": "MNTD",
    "ses_pd": "SES.PD", "ses_mpd": "SES.MPD", "ses_mntd": "SES.MNTD",
    "dark_n": "Dark diversity",
}


def loading_label(var: str) -> str:
    if var in STABILITY_LABELS:
        return STABILITY_LABELS[var]
    metric = var.removeprefix(LOADING_PREFIX_KEEP)
    return LSP_LABELS.get(metric, var)


def main() -> None:
    scores = pd.read_parquet(TABLES / "pca_lsp_stability_scores.parquet")
    loadings = pd.read_parquet(TABLES / "pca_lsp_stability_loadings.parquet")
    supp = pd.read_parquet(TABLES / "pca_lsp_stability_supp_facets.parquet")
    var = pd.read_csv(TABLES / "pca_lsp_stability_variance.csv", header=None,
                      names=["pc", "var"])
    var_pc1 = var.loc[var["pc"] == "PC1", "var"].iloc[0] * 100
    var_pc2 = var.loc[var["pc"] == "PC2", "var"].iloc[0] * 100

    # circulo de correlaciones: reescalar cargas por sdev(PC) para que queden en [-1,1],
    # igual que las correlaciones de las facetas (ya en esa escala por construccion).
    # sdev(PC) = desvio de los scores, que prcomp ya centra/escala antes de proyectar.
    sdev_pc1 = scores["PC1"].std()
    sdev_pc2 = scores["PC2"].std()
    loadings = loadings.copy()
    loadings["PC1_corr"] = loadings["PC1"] * sdev_pc1
    loadings["PC2_corr"] = loadings["PC2"] * sdev_pc2
    loadings["len"] = np.hypot(loadings["PC1_corr"], loadings["PC2_corr"])
    keep = loadings["variable"].str.startswith(LOADING_PREFIX_KEEP) | \
        loadings["variable"].isin(STABILITY_COLS)
    top = loadings[keep].sort_values("len", ascending=False).head(TOP_N_LOADINGS)

    fg.set_paper_style()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    ax.scatter(scores["PC1"], scores["PC2"], s=8, alpha=0.25, color="#999999",
              linewidths=0, zorder=1)

    circ = plt.Circle((0, 0), 1, fill=False, color="#cccccc", lw=0.8, zorder=2)
    ax.add_patch(circ)

    for _, r in top.iterrows():
        ax.annotate("", xy=(r["PC1_corr"], r["PC2_corr"]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", color="#2f6f7f", lw=1.0), zorder=3)
        ax.text(r["PC1_corr"] * 1.08, r["PC2_corr"] * 1.08, loading_label(r["variable"]),
               fontsize=7, color="#2f6f7f", ha="center", va="center")

    for _, r in supp.iterrows():
        ax.annotate("", xy=(r["PC1"], r["PC2"]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", color="#c1553b", lw=1.3), zorder=4)
        ax.text(r["PC1"] * 1.1, r["PC2"] * 1.1, FACET_LABELS.get(r["facet"], r["facet"]),
               fontsize=7, color="#c1553b", weight="bold", ha="center", va="center")

    lim = 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)")
    ax.set_title(f"Phenology (kNDVI) + interannual stability "
                f"(green, top {len(top)} of {keep.sum()}) vs. "
                f"diversity facets (red, supplementary)", fontsize=8)

    fig.tight_layout()
    out = fg.save_figure(fig, "fig18_pca_lsp_stability_biplot",
                         ROOT / "results" / "figures", also_png=True)
    print(f"escrito {out} (+ .png)")


if __name__ == "__main__":
    main()
