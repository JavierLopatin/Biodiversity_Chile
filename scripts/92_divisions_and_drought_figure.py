#!/usr/bin/env python3
"""Las divisiones del pool y el resultado del estres hidrico, en una figura.

El analisis fue partiendo el pool por seis criterios sucesivos -- fuente, forma de
crecimiento, estrato de cobertura, contribuyente, bloque temporal y tercil de estres --
y cada particion cambio alguna conclusion. Esta figura muestra donde cae cada una en el
territorio y que sale del contraste por estres.

Uso:
    python scripts/92_divisions_and_drought_figure.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
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
    chile = gpd.read_file(ROOT / "shapefiles" / "regiones_chile.shp").to_crs("EPSG:4326")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.4, 1.4], hspace=0.38, wspace=0.34)

    # --- A y B: los dos cortes espaciales
    for k, (col, groups, title) in enumerate([
            (0, [("parcelas_cl", BLUE, "Parcelas-CL"), ("living_trees", ORANGE, "Living Trees")],
             "A. Inventory"),
            (1, [("NoBosque", GREEN, "Non-forest"), ("Forest", PURPLE, "Forest")],
             "B. MapBiomas stratum")]):
        ax = fig.add_subplot(gs[:, col])
        chile.plot(ax=ax, facecolor="#F2F2F2", edgecolor="#BBBBBB", linewidth=0.3)
        key = "source" if k == 0 else "lvl2"
        for val, color, lab in groups:
            s = j[j[key] == val]
            ax.scatter(s.lon, s.lat, s=4, alpha=0.55, linewidths=0, color=color,
                       label=f"{lab} ({len(s)})")
        ax.set(title=title, aspect="equal", xlabel="lon", ylabel="lat" if k == 0 else "")
        ax.set_xlim(-76, -66)
        ax.legend(loc="lower left", fontsize=8, markerscale=2.5, framealpha=0.9)

    # --- C: estres hidrico por ano de censo
    ax = fig.add_subplot(gs[0, 2])
    z = j[j.spi12_win_mean.notna()]
    yrs = sorted(int(y) for y, g in z.groupby("Year") if len(g) >= 30)
    ax.boxplot([z[z.Year == y].spi12_win_mean for y in yrs], positions=yrs,
               widths=0.6, showfliers=False,
               boxprops=dict(color=GREY), medianprops=dict(color=ORANGE, lw=2),
               whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set(title="C. Water stress over the census window", xlabel="census year",
           ylabel="SPI, 12 months")
    ax.set_xticks(yrs[::2]); ax.set_xticklabels(yrs[::2], rotation=45)
    ax.text(0.03, 0.06, "all windows below 0: megadrought", transform=ax.transAxes, fontsize=9,
            color=GREY)

    # --- D: SPI contra latitud, el confusor
    ax = fig.add_subplot(gs[1, 2])
    for val, color, lab in [("parcelas_cl", BLUE, "Parcelas-CL"),
                            ("living_trees", ORANGE, "Living Trees")]:
        s = z[z.source == val]
        ax.scatter(s.lat, s.spi12_win_mean, s=4, alpha=0.4, linewidths=0, color=color, label=lab)
    ax.set(title="D. SPI tracks latitude", xlabel="latitude", ylabel="SPI, 12 months")
    ax.legend(fontsize=8, markerscale=2.5)
    ax.text(0.03, 0.9, r"$\rho$(SPI, lat) = $-$0.67 within year", transform=ax.transAxes,
            fontsize=9, color=GREY)

    # --- E y F: el resultado, por tercil
    for k, (lab, run) in enumerate(RUNS.items()):
        ax = fig.add_subplot(gs[k, 3])
        t = tercile_result(j, run)
        for tercil, color, name in [(0, "#B4451F", "dry"), (1, GREY, "mid"),
                                    (2, "#2C7FB8", "wet")]:
            s = t[t.tercil == tercil].sort_values("year")
            ax.plot(s.year, s.r2, "-o", color=color, ms=5, lw=1.8, label=name)
        d0 = t[t.tercil == 0].set_index("year").r2 - t[t.tercil == 2].set_index("year").r2
        ax.set(title=f"{'EF'[k]}. Richness prediction, {lab}",
               xlabel="census year" if k else "", ylabel="out-of-fold $R^2$")
        ax.axhline(0, color="#333333", lw=0.8, ls=":")
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        ax.text(0.03, 0.9, f"dry − wet = {d0.mean():+.3f}  ({int((d0>0).sum())}/{len(d0)} years)",
                transform=ax.transAxes, fontsize=9, color="#B4451F")

    fig.suptitle("Pool divisions and the effect of water stress on prediction",
                 fontsize=15, y=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig25_divisions_and_drought.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig25_divisions_and_drought'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
