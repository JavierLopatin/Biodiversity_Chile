#!/usr/bin/env python3
"""Curvas de acumulacion de las facetas para el manuscrito, separadas en dos figuras:
q=0 (las tres facetas con senal) al texto principal, q=1 y q=2 al suplemento.

Reusa las mismas salidas que `scripts/60_unified_summary_figures.py` (fig20/fig21) pero
reparte los paneles segun lo que el texto principal reporta, en vez de mostrar la rejilla
completa de q. Mismo estilo de ploteo: paleta, DPI y salida PNG+PDF.

Taxonomica y filogenetica vienen de la rarefaccion/extrapolacion de iNEXT sobre el pool
(`unified_hill_curves.csv`); el tercer panel es la componente beta de la particion de Hill
(`unified_beta_freq_curve.csv`), que es el recambio regional que LCBD reparte entre
parcelas -- LCBD es por parcela y no tiene curva de acumulacion propia.

Uso:
    python scripts/79_hill_curve_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE = "#4C78A8", "#F58518", "#54A24B", "#B279A2"
DPI = 200

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

HILL = {"taxonomica": ("Taxonomic diversity (TD)", GREEN),
        "filogenetica_meanPD": ("Phylogenetic diversity (PD)", BLUE)}


def hill_panel(ax, hc: pd.DataFrame, metric: str, q: int) -> None:
    """Rarefaccion en linea continua, extrapolacion punteada, banda de confianza."""
    label, color = HILL[metric]
    z = hc[(hc["q"] == q) & (hc["metric"] == metric)].sort_values("n")
    interp = z[z["method"].isin(["Rarefaction", "Observed"])]
    extrap = z[z["method"].isin(["Observed", "Extrapolation"])]
    obs = z[z["method"] == "Observed"]
    ax.plot(interp["n"], interp["value"], "-", color=color, lw=2)
    ax.plot(extrap["n"], extrap["value"], "--", color=color, lw=2)
    ax.fill_between(z["n"], z["lo"], z["hi"], color=color, alpha=0.15)
    ax.scatter(obs["n"], obs["value"], color=color, s=45, zorder=5)
    ax.set(title=f"{label}, $q={q}$", xlabel="plots pooled", ylabel="effective species")


def beta_panel(ax, bc: pd.DataFrame, q: int) -> None:
    z = bc[bc["q"] == q].sort_values("n")
    ax.plot(z["n"], z["beta_mean"], "-", color=ORANGE, lw=2)
    ax.fill_between(z["n"], z["beta_lo"], z["beta_hi"], color=ORANGE, alpha=0.15)
    ax.set(title=f"Hill beta, $q={q}$", xlabel="plots pooled",
           ylabel="effective communities")
    ax.set_xscale("log")


def main() -> None:
    hc = pd.read_csv(DERIVED / "unified_hill_curves.csv")
    hc = hc[hc["dataset"] == "unificado completo"]
    bc = pd.read_csv(DERIVED / "unified_beta_freq_curve.csv")

    # Texto principal: solo q=0, las tres facetas que el manuscrito reporta.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    hill_panel(axes[0], hc, "taxonomica", 0)
    hill_panel(axes[1], hc, "filogenetica_meanPD", 0)
    beta_panel(axes[2], bc, 0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig24_hill_curves_q0.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"-> {FIG_DIR / 'fig24_hill_curves_q0'}.{{png,pdf}}")

    # Suplemento: q=1 y q=2.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, q in enumerate([1, 2]):
        hill_panel(axes[row, 0], hc, "taxonomica", q)
        hill_panel(axes[row, 1], hc, "filogenetica_meanPD", q)
        beta_panel(axes[row, 2], bc, q)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figS6_hill_curves_q12.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"-> {FIG_DIR / 'figS6_hill_curves_q12'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
