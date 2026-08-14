#!/usr/bin/env python3
"""Figuras resumen de la base unificada (Parcelas-CL + Living Trees Chile) para
`docs/19_unified_facets_methodology.md`. Reusa el mismo codigo de ploteo del
notebook (`notebooks/02_explore_biodiversity_responses.ipynb` SS7a/7e/7f/7g), guardado
como PNG independiente en vez de solo en el notebook -- mismo criterio que
`scripts/26_rarefaction_inext.R`/`scripts/27_compute_dark_diversity.R` (fig15/fig16).

Uso:
    python scripts/60_unified_summary_figures.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE = "#4C78A8", "#F58518", "#54A24B", "#B279A2"
DPI = 200


def fig19_coverage_map() -> None:
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")
    chile = gpd.read_file(ROOT / "shapefiles" / "regiones_chile.shp").to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(6, 11))
    chile.plot(ax=ax, facecolor="#F2F2F2", edgecolor="#9A9A9A", linewidth=0.35)
    for src, color in [("parcelas_cl", BLUE), ("living_trees", ORANGE)]:
        sub = plots[plots["source"] == src]
        ax.scatter(sub["lon"], sub["lat"], s=8, alpha=0.6, color=color,
                   label=f"{src} (n={len(sub)})")
    ax.set(title="Unified coverage: Parcelas-CL + Living Trees Chile",
          xlabel="lon", ylabel="lat", aspect="equal")
    ax.legend(loc="lower left")
    fig.tight_layout()
    stem = FIG_DIR / "fig19_unified_coverage_map"
    fig.savefig(f"{stem}.png", dpi=DPI)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"-> {stem}.{{png,pdf}}")


def fig20_hill_curves() -> None:
    hc = pd.read_csv(DERIVED / "unified_hill_curves.csv")
    hc = hc[hc["dataset"] == "unificado completo"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    metrics = ["taxonomica", "filogenetica_meanPD"]
    titles = {"taxonomica": "Taxonomic Hill diversity",
             "filogenetica_meanPD": "Phylogenetic Hill diversity (meanPD)"}

    for i, q in enumerate([0, 1, 2]):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            z = hc[(hc["q"] == q) & (hc["metric"] == metric)].sort_values("n")
            obs = z[z["method"] == "Observed"]
            interp = z[z["method"].isin(["Rarefaction", "Observed"])]
            extrap = z[z["method"].isin(["Observed", "Extrapolation"])]
            ax.plot(interp["n"], interp["value"], "-", color=GREEN, lw=1.5)
            ax.plot(extrap["n"], extrap["value"], "--", color=GREEN, lw=1.5)
            ax.fill_between(z["n"], z["lo"], z["hi"], color=GREEN, alpha=0.15)
            ax.scatter(obs["n"], obs["value"], color=GREEN, s=20, zorder=5)
            if i == 0:
                ax.set_title(titles[metric])
            if j == 0:
                ax.set_ylabel(f"Hill diversity (q={q})")
            if i == 2:
                ax.set_xlabel("sampling units (plots)")
    fig.tight_layout()
    stem = FIG_DIR / "fig20_unified_hill_curves"
    fig.savefig(f"{stem}.png", dpi=DPI)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"-> {stem}.{{png,pdf}}")


def fig21_beta_partition_curve() -> None:
    bc = pd.read_csv(DERIVED / "unified_beta_freq_curve.csv").sort_values("n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, comp, color in zip(axes, ["alpha", "beta", "gamma"], [BLUE, GREEN, ORANGE]):
        for q, ls in zip([0, 1, 2], ["-", "--", ":"]):
            z = bc[bc["q"] == q]
            ax.plot(z["n"], z[f"{comp}_mean"], ls, color=color, lw=1.6, label=f"q={q}")
            ax.fill_between(z["n"], z[f"{comp}_lo"], z[f"{comp}_hi"], color=color, alpha=0.12)
        ax.set(title=f"Hill {comp}", xlabel="plots pooled (n)",
              ylabel="effective species" if comp != "beta" else "effective communities")
        ax.set_xscale("log")
        ax.legend()
    fig.tight_layout()
    stem = FIG_DIR / "fig21_beta_partition_curve"
    fig.savefig(f"{stem}.png", dpi=DPI)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"-> {stem}.{{png,pdf}}")


def fig22_dark_diversity_curve() -> None:
    dc = pd.read_csv(DERIVED / "dark_diversity_curve.csv").sort_values("n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dc["n"], dc["dark_n_mean"], "-", color=PURPLE, lw=1.8)
    ax.fill_between(dc["n"], dc["dark_n_mean_lo"], dc["dark_n_mean_hi"], color=PURPLE, alpha=0.15)
    ax.set(title="dark_n stability vs co-occurrence pool size", xlabel="plots in pool (n)",
          ylabel="mean dark_n per plot")
    ax.set_xscale("log")
    fig.tight_layout()
    stem = FIG_DIR / "fig22_dark_diversity_curve"
    fig.savefig(f"{stem}.png", dpi=DPI)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"-> {stem}.{{png,pdf}}")


if __name__ == "__main__":
    fig19_coverage_map()
    fig20_hill_curves()
    fig21_beta_partition_curve()
    fig22_dark_diversity_curve()
