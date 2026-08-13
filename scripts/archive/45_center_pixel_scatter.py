#!/usr/bin/env python3
"""Observado vs. predicho, el ganador de cada faceta (matrix de pixel central, docs/17).

Un panel por faceta, un target representativo, el run_id que gano esa faceta en la tabla de
docs/17_center_pixel_kndvi_block20.md. Usa `oof_predictions.csv` ya en disco -- no reajusta
nada.

Uso:
    python scripts/45_center_pixel_scatter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import figures as fg   # noqa: E402

SCHEME = "kfold5_block20"

# faceta -> (run_id ganador, target representativo)
PANELS = {
    "alfa": ("RF03c_curve-topo_ctr-area_kndvi", "hill_q0"),
    "beta_pa": ("RF01c_lsp_ctr-topo_ctr-area_kndvi", "pcoa1_pa"),
    "beta_cover": ("MLP06_curve_kndvi_ctr", "pcoa1_cover"),
    "filo": ("RF02c_lsp_ctr-qc-topo_ctr-area_kndvi", "ses_pd"),
    "dark": ("RF03c_curve-topo_ctr-area_kndvi", "dark_n"),
}


def main() -> None:
    fg.set_paper_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.1 * len(PANELS), 3.2))

    for ax, (facet, (run_id, target)) in zip(axes, PANELS.items()):
        path = ROOT / "results" / "models" / run_id / SCHEME / "oof_predictions.csv"
        df = pd.read_csv(path)
        # 3 filas por parcela (una por seed); promediar antes de graficar, si no cada punto
        # se dibuja 3 veces con una prediccion ligeramente distinta.
        ens = df.groupby("PlotObservationID")[[f"{target}_obs", f"{target}_pred"]].mean()
        obs, pred = ens[f"{target}_obs"], ens[f"{target}_pred"]
        r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()

        ax.scatter(obs, pred, s=6, alpha=0.35, color="#2f6f7f", linewidths=0)
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, ls="--", zorder=0)
        ax.set_title(f"{facet}\n{run_id.split('_')[0]} / {target}\nR2={r2:.3f}", fontsize=8)
        ax.set_xlabel("observado")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("predicho")
    fig.suptitle("Observado vs. predicho -- ganador de cada faceta "
                 "(kNDVI, pixel central, kfold5_block20)", fontsize=9, y=1.04)
    fig.tight_layout()

    out = fg.save_figure(fig, "fig17_center_obs_vs_pred", ROOT / "results" / "figures",
                         also_png=True)
    print(f"escrito {out} (+ .png)")


if __name__ == "__main__":
    main()
