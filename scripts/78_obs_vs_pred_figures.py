#!/usr/bin/env python3
"""Observado contra predicho fuera de fold para el modelo que se despliega en los mapas
(C1D01, red unidimensional, kNDVI serie cruda, validacion por bloque espacial de 20 km).

Lee `oof_predictions.csv` del run block20 en `results/models_unified_topofix/`, promedia
las cinco semillas por parcela -- que es el objeto que se despliega, el ensemble de
semillas -- y dibuja el ajuste facet por facet. Los R2 que anota reproducen exactamente
las columnas `ens.` y `seeds` de la Tabla 3 del manuscrito, que salen del mismo archivo.

Mismo estilo de ploteo que `scripts/60_unified_summary_figures.py` (paleta, DPI, PNG+PDF).

Uso:
    python scripts/78_obs_vs_pred_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Corrida por defecto. `--run` apunta a cualquier otra; `--tag` cambia el sufijo de
#: las figuras para no pisar las de otra variante.
RUN = (ROOT / "results" / "models_unified_topofix"
       / "C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr"
       / "kfold5_block20_unified" / "oof_predictions.csv")

BLUE, ORANGE = "#4C78A8", "#F58518"
DPI = 200

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

#: Facetas en el orden del manuscrito. `signal` marca las tres que conservan senal y
#: que van a la figura del texto principal; las de q=1,2 solo al suplemento.
FACETS = [
    ("lcbd_count_sorensen", "LCBD", True),
    ("pd_inext_q0", r"PD$_0$", True),
    ("td_inext_q0", r"TD$_0$", True),
    ("pd_inext_q1", r"PD$_1$", False),
    ("pd_inext_q2", r"PD$_2$", False),
    ("td_inext_q1", r"TD$_1$", False),
    ("td_inext_q2", r"TD$_2$", False),
]


def r2(obs: pd.Series, pred: pd.Series) -> float:
    return 1.0 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()


def load(run: Path = RUN) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Ensemble por parcela, y el R2 medio entre semillas con su desviacion."""
    raw = pd.read_csv(run)
    ens = raw.groupby("PlotObservationID").mean(numeric_only=True)
    ens["living_trees"] = ens.index.astype(str).str.startswith("LT_")

    seed_r2: dict[str, tuple[float, float]] = {}
    for facet, _, _ in FACETS:
        scores = []
        for _, g in raw.groupby("seed"):
            m = g[f"{facet}_obs"].notna() & g[f"{facet}_pred"].notna()
            scores.append(r2(g.loc[m, f"{facet}_obs"], g.loc[m, f"{facet}_pred"]))
        seed_r2[facet] = (float(np.mean(scores)), float(np.std(scores)))
    return ens, seed_r2


def panel(ax, ens: pd.DataFrame, facet: str, label: str,
          seed_r2: tuple[float, float]) -> None:
    obs, pred = ens[f"{facet}_obs"], ens[f"{facet}_pred"]
    keep = obs.notna() & pred.notna()
    obs, pred, lt = obs[keep], pred[keep], ens.loc[keep, "living_trees"]

    for mask, color, name in [(~lt, BLUE, "Parcelas-CL"), (lt, ORANGE, "Living Trees")]:
        ax.scatter(obs[mask], pred[mask], s=24, alpha=0.45, linewidths=0,
                   color=color, label=f"{name} (n={int(mask.sum())})")

    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=0.8, color="#555555")
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), aspect="equal",
           xlabel="observed", ylabel="predicted", title=label)

    mean, sd = seed_r2
    ax.text(0.04, 0.96,
            f"$R^2_{{ens}}$ = {r2(obs, pred):.3f}\n"
            f"seeds = {mean:.3f} $\\pm$ {sd:.3f}\n"
            f"n = {len(obs)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11)
    if facet.startswith("lcbd"):
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")


def figure(ens: pd.DataFrame, seed_r2: dict, facets: list, stem: str,
           ncols: int, figsize: tuple[float, float]) -> None:
    nrows = int(np.ceil(len(facets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, (facet, label, _) in zip(axes, facets):
        panel(ax, ens, facet, label, seed_r2[facet])
    for ax in axes[len(facets):]:
        ax.set_visible(False)
    axes[0].legend(loc="lower right", fontsize=10, markerscale=1.6, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=DPI)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"-> {FIG_DIR / stem}.{{png,pdf}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN, help="oof_predictions.csv a dibujar")
    ap.add_argument("--tag", default="", help="sufijo de las figuras, p.ej. _woody")
    args = ap.parse_args()

    ens, seed_r2 = load(args.run)
    figure(ens, seed_r2, [f for f in FACETS if f[2]],
           f"fig23_obs_vs_pred_c1d_block20{args.tag}", ncols=3, figsize=(14, 5.2))
    figure(ens, seed_r2, FACETS,
           f"figS5_obs_vs_pred_c1d_all_facets{args.tag}", ncols=4, figsize=(17, 9.5))


if __name__ == "__main__":
    main()
