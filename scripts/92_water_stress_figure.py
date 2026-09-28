#!/usr/bin/env python3
"""El efecto del estres hidrico sobre la estabilidad de la prediccion de riqueza.

Solo resultados de analisis: donde caen las parcelas de cada inventario y de cada estrato
MapBiomas pertenece al mapa de la zona de estudio (`fig01_study_area`), no aqui.

Los cuatro paneles: (A) el SPI de cada ventana censal, que queda siempre bajo cero porque
todo el periodo cae dentro de la megasequia; (B) el confusor, el SPI es en gran parte un
gradiente latitudinal, por eso se residualiza contra latitud antes de partir en terciles;
(C, D) el resultado, R2 fuera de fold por tercil de estres dentro de cada ano de censo.

Uso:
    python scripts/92_divisions_and_drought_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, GREY = "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#666666"
DPI = 200
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})

L2 = {3: "Forest", 59: "Forest", 60: "Forest", 61: "Forest",
      66: "NoBosque", 12: "NoBosque"}
RUNS = {"climate": "RFG1c_clim-topo_ctr-area_raw100_unified-all_unified_woody",
        "phenology": "RFG2c_curve-topo_ctr-area_kndvi_raw100_unified-all_unified_woody"}
OBS, PRE = "hill_q0_unified_obs", "hill_q0_unified_pred"


def r2(a, b):
    return 1 - ((b - a) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def load():
    d = pd.read_parquet(DERIVED / "plots_unified.parquet").set_index("PlotObservationID")
    mb = pd.read_parquet(DERIVED / "mapbiomas_class_unified.parquet").set_index("PlotObservationID")
    spi = pd.read_parquet(DERIVED / "spi_unified.parquet").set_index("PlotObservationID")
    j = d.join(mb[["mb_code"]]).join(spi[["spi12_win_mean"]])
    j["lvl2"] = j.mb_code.map(L2)
    return j


def tercile_result(j: pd.DataFrame, run: str) -> pd.DataFrame:
    """R2 por tercil de SPI residualizado contra latitud, dentro de cada ano de censo."""
    f = ROOT / "results" / "models_gate" / run / "kfold5_block20_unified" / "oof_predictions.csv"
    o = pd.read_csv(f).groupby("PlotObservationID").mean(numeric_only=True)
    e = o.join(j[["source", "Year", "lat", "spi12_win_mean"]])
    z = e[(e.source == "living_trees") & e[OBS].notna() & e.spi12_win_mean.notna()]
    rows = []
    for y, g in z.groupby("Year"):
        if len(g) < 60:
            continue
        g = g.copy()
        # el SPI dentro de ano es 67% gradiente latitudinal; sin quitarlo, partir por
        # tercil de SPI es partir por latitud
        g["res"] = g.spi12_win_mean - np.polyval(np.polyfit(g.lat, g.spi12_win_mean, 1), g.lat)
        g["t"] = pd.qcut(g.res, 3, labels=[0, 1, 2])
        for t in range(3):
            s = g[g.t == t]
            rows.append(dict(year=int(y), tercil=t, n=len(s), r2=r2(s[OBS], s[PRE])))
    return pd.DataFrame(rows)


def main() -> None:
    j = load()
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.26)

    # --- A: estres hidrico por ano de censo
    ax = fig.add_subplot(gs[0, 0])
    z = j[j.spi12_win_mean.notna()]
    yrs = sorted(int(y) for y, g in z.groupby("Year") if len(g) >= 30)
    ax.boxplot([z[z.Year == y].spi12_win_mean for y in yrs], positions=yrs,
               widths=0.6, showfliers=False,
               boxprops=dict(color=GREY), medianprops=dict(color=ORANGE, lw=2),
               whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set(title="A. Water stress over the census window", xlabel="census year",
           ylabel="SPI, 12 months")
    ax.set_xticks(yrs[::2]); ax.set_xticklabels(yrs[::2], rotation=45)
    ax.text(0.03, 0.06, "all windows below 0: megadrought", transform=ax.transAxes, fontsize=9,
            color=GREY)

    # --- B: SPI contra latitud, el confusor
    ax = fig.add_subplot(gs[1, 0])
    for val, color, lab in [("parcelas_cl", BLUE, "Parcelas-CL"),
                            ("living_trees", ORANGE, "Living Trees")]:
        s = z[z.source == val]
        ax.scatter(s.lat, s.spi12_win_mean, s=4, alpha=0.4, linewidths=0, color=color, label=lab)
    ax.set(title="B. SPI tracks latitude", xlabel="latitude", ylabel="SPI, 12 months")
    ax.legend(fontsize=8, markerscale=2.5)
    ax.text(0.03, 0.9, r"$\rho$(SPI, lat) = $-$0.67 within year", transform=ax.transAxes,
            fontsize=9, color=GREY)

    # --- C y D: el resultado, por tercil
    for k, (lab, run) in enumerate(RUNS.items()):
        ax = fig.add_subplot(gs[k, 1])
        t = tercile_result(j, run)
        for tercil, color, name in [(0, "#B4451F", "dry"), (1, GREY, "mid"),
                                    (2, "#2C7FB8", "wet")]:
            s = t[t.tercil == tercil].sort_values("year")
            ax.plot(s.year, s.r2, "-o", color=color, ms=5, lw=1.8, label=name)
        d0 = t[t.tercil == 0].set_index("year").r2 - t[t.tercil == 2].set_index("year").r2
        ax.set(title=f"{'CD'[k]}. Richness prediction, {lab}",
               xlabel="census year" if k else "", ylabel="out-of-fold $R^2$")
        ax.axhline(0, color="#333333", lw=0.8, ls=":")
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        ax.text(0.03, 0.9, f"dry − wet = {d0.mean():+.3f}  ({int((d0>0).sum())}/{len(d0)} years)",
                transform=ax.transAxes, fontsize=9, color="#B4451F")

    fig.suptitle("Water stress and the stability of richness prediction",
                 fontsize=15, y=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig25_water_stress.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig25_water_stress'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
