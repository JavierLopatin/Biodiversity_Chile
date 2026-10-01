#!/usr/bin/env python3
"""Suplemento: el efecto de sequia y por que no sobrevive sus controles.

Tres paneles para tres afirmaciones, en orden:

    a  el diseno ingenuo esta confundido        balance de covariables en los tres disenos
    b  corregirlo mata el efecto                contraste por terciles y su mecanismo
    c  un segundo diseno tampoco lo encuentra   pares ecologicamente gemelos

Van juntos a proposito. Es la misma pregunta contestada de tres formas, y separarlas obliga al
lector a rearmar el argumento.

El panel a es el que decide, y no se mira lo suficiente en este tipo de analisis: muestra que
el tercil definido como residuo de SPI respecto a la LATITUD deja la elevacion desbalanceada en
-0,29 de diferencia de medias estandarizada y la longitud en -0,20. Con ese desbalance, el
"tercil seco" es la cordillera. Descontar los tres ejes del modelo nulo lo arregla, y el
emparejamiento lo arregla ademas para la composicion, que el detrend no toca.

Fuentes: `scripts/107` (terciles y contraste) y `scripts/121` (emparejamiento). Ningun numero
esta escrito a mano.

Uso:
    python scripts/122_figura_sequia_control.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
BLUE, RED, DARK, GREY = "#4C78A8", "#C0392B", "#333333", "#9A9A9A"
DPI = 300
CALIPERS = (0.25, 0.5, 1.0, 2.0)
CAL_PRINCIPAL = 0.5
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 11,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})

LAB_COV = {"isomap1_pa_unified": "Isomap 1 (p/a)", "isomap2_pa_unified": "Isomap 2 (p/a)",
           "isomap3_pa_unified": "Isomap 3 (p/a)", "hill_q0_unified": "Richness $q_0$",
           "elevation": "Elevation", "lat": "Latitude", "lon": "Longitude",
           "Year": "Census year", "spi": "SPI  (the contrast)"}
ORDEN_COV = list(LAB_COV)

#: Rotulos de faceta del panel c, los mismos que la figura central del paper.
LAB_FAC = {
    "lcbd_count_sorensen": "LCBD Sørensen", "lcbd_pa_unified": "LCBD (p/a)",
    "lcbd_freq_unified": "LCBD (freq.)", "hill_q0_unified": "Richness $q_0$ (raw)",
    "dark_n_unified": "Dark diversity", "mpd_unified": "MPD", "mntd_unified": "MNTD",
    "ses_pd_unified": "SES PD", "ses_mpd_unified": "SES MPD", "ses_mntd_unified": "SES MNTD",
    "td_inext_q0": "TD $q_0$", "td_inext_q1": "TD $q_1$", "td_inext_q2": "TD $q_2$",
    "pd_inext_q0": "PD $q_0$", "pd_inext_q1": "PD $q_1$", "pd_inext_q2": "PD $q_2$",
    **{f"{m}{i}_{v}_unified": f"{'Isomap' if m == 'isomap' else 'PCoA'} {i} "
       f"({'p/a' if v == 'pa' else 'freq.'})"
       for m in ("isomap", "pcoa") for i in (1, 2, 3) for v in ("pa", "freq")},
}


def cargar(nombre: str, archivo: str):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / archivo)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    m7 = cargar("s107", "107_sintesis_lenoso.py")
    m21 = cargar("s121", "121_sequia_pareada.py")

    d = m21.candidatos(m7)                      # terciles con detrend geografico
    d_lat = d.copy()
    d_lat["t"] = m7.terciles(d_lat, ejes=("lat",))
    pares = m21.emparejar(d, CAL_PRINCIPAL)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})
    axa, axb, axc = axes

    # ---------------------------------------------------------------- a  balance
    disenos = [
        ("terciles, latitude detrend", RED, "o",
         (d_lat[d_lat.t == 0], d_lat[d_lat.t == 2])),
        ("terciles, lon+lat+elev detrend", "#E8832A", "s", (d[d.t == 0], d[d.t == 2])),
        (f"matched pairs ({len(pares)} pairs)", BLUE, "^",
         (d.loc[pares.i_seco], d.loc[pares.i_humedo])),
    ]
    for j, (etq, color, mk, (a, b)) in enumerate(disenos):
        v = [m21.smd(a[c].reset_index(drop=True), b[c].reset_index(drop=True))
             for c in ORDEN_COV]
        axa.plot(v, range(len(ORDEN_COV)), mk, ms=6.5, color=color, lw=0, label=etq,
                 alpha=0.9, zorder=3)
    for x in (-0.1, 0.1):
        axa.axvline(x, color="#CCCCCC", lw=0.8, ls=(0, (3, 3)), zorder=1)
    axa.axvline(0, color=DARK, lw=0.9, zorder=2)
    axa.set_yticks(range(len(ORDEN_COV)))
    axa.set_yticklabels([LAB_COV[c] for c in ORDEN_COV])
    axa.invert_yaxis()
    axa.axhline(len(ORDEN_COV) - 1.5, color="#BBBBBB", lw=0.8)
    axa.set_xlim(-1.55, 0.95)
    axa.set_xlabel("standardised mean difference, dry $-$ wet")
    axa.set_title("a  What each design actually compares", loc="left", fontweight="bold")
    # abajo a la derecha: abajo a la izquierda tapaba los marcadores de SPI, que caen en -1,1
    axa.legend(fontsize=8, loc="lower right", framealpha=0.95)
    axa.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
    axa.set_axisbelow(True)

    # ------------------------------------------------- b  terciles y su mecanismo
    f = pd.read_csv(ROOT / "results" / "tables" / "sintesis_lenoso.csv")
    filas_b = [("lat", "neto", "margin gain\nlatitude detrend"),
               ("geo", "neto", "margin gain\nlon+lat+elev detrend"),
               ("lat", "piso_delta", "null model's own gain\nlatitude detrend"),
               ("geo", "piso_delta", "null model's own gain\nlon+lat+elev detrend")]
    rng = np.random.default_rng(0)
    for j, (dt, col, etq) in enumerate(filas_b):
        v = m7.contraste_estres(f, detrend=dt, escribir=False)[col].to_numpy()
        k, n = int((v > 0).sum()), len(v)
        axb.scatter(v, j + rng.uniform(-0.14, 0.14, n), s=17, lw=0,
                    color=np.where(v > 0, BLUE, RED), alpha=0.75, zorder=3)
        axb.plot([v.mean()] * 2, [j - 0.27, j + 0.27], color=DARK, lw=2.2, zorder=4)
        axb.text(0.985, j - 0.37, f"{k}/{n}   $p$ = {binomtest(k, n, 0.5).pvalue:.2f}",
                 transform=axb.get_yaxis_transform(), ha="right", va="center",
                 fontsize=8, color="#555555")
    axb.axvline(0, color=DARK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    axb.axhline(1.5, color="#BBBBBB", lw=0.8)
    axb.set_yticks(range(len(filas_b)))
    axb.set_yticklabels([e[2] for e in filas_b], fontsize=8)
    axb.invert_yaxis()
    axb.set_ylim(len(filas_b) - 0.45, -0.8)
    axb.set_xlabel("change from the dry to the wet tercile")
    axb.set_title("b  Correcting it removes the effect", loc="left", fontweight="bold")
    axb.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
    axb.set_axisbelow(True)

    # ------------------------------------------------------- c  pares emparejados
    e = m21.errores(pd.Index(d.PlotObservationID))
    facetas = [c for c in e.columns if not c.startswith("__obs__")]
    ids_s = d.loc[pares.i_seco, "PlotObservationID"].to_numpy()
    ids_h = d.loc[pares.i_humedo, "PlotObservationID"].to_numpy()
    filas = []
    for t in facetas:
        es, eh = e[t].reindex(ids_s).to_numpy(), e[t].reindex(ids_h).to_numpy()
        ok = np.isfinite(es) & np.isfinite(eh)
        if ok.sum() < 40:
            continue
        dif = es[ok] - eh[ok]
        sd2 = float(e[f"__obs__{t}"].std()) ** 2
        filas.append(dict(faceta=t, dif=dif.mean() / sd2,
                          p=wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0))
    c = pd.DataFrame(filas).sort_values("dif").reset_index(drop=True)
    col = np.where(c.dif > 0, RED, BLUE)
    axc.barh(c.index, c.dif, color=col, alpha=0.85, height=0.72, lw=0, zorder=3)
    for i, r in c.iterrows():
        if r.p < 0.05:
            axc.text(r.dif + (0.012 if r.dif > 0 else -0.012), i, "*", va="center",
                     ha="left" if r.dif > 0 else "right", fontsize=12, color=DARK)
    axc.axvline(0, color=DARK, lw=0.9, zorder=2)
    axc.set_yticks(range(len(c)))
    axc.set_yticklabels([LAB_FAC.get(x, x) for x in c.faceta], fontsize=7.5)
    axc.set_xlabel("squared error, dry $-$ wet (facet variance units)")
    axc.set_title("c  Ecologically matched pairs find nothing either", loc="left",
                  fontweight="bold")
    axc.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
    axc.set_axisbelow(True)
    k, n = int((c.dif > 0).sum()), len(c)
    barrido = []
    for cal in CALIPERS:
        pr = m21.emparejar(d, cal)
        si = d.loc[pr.i_seco, "PlotObservationID"].to_numpy()
        hi = d.loc[pr.i_humedo, "PlotObservationID"].to_numpy()
        kk = nn = 0
        for t in facetas:
            a1, a2 = e[t].reindex(si).to_numpy(), e[t].reindex(hi).to_numpy()
            o = np.isfinite(a1) & np.isfinite(a2)
            if o.sum() < 40:
                continue
            nn += 1
            kk += int((a1[o] - a2[o]).mean() > 0)
        barrido.append(f"{cal:.2f}: {kk}/{nn}")
    axc.text(0.015, 0.015, f"worse in dry: {k}/{n}, $p$ = {binomtest(k, n, 0.5).pvalue:.2f}\n"
             f"caliper sweep  " + "   ".join(barrido) + "\n* Wilcoxon $p$ < 0.05",
             transform=axc.transAxes, fontsize=7.5, color="#555555", va="bottom",
             bbox=dict(fc="white", ec="#DDDDDD", boxstyle="round,pad=0.4"))

    h = [Line2D([], [], color=RED, lw=5, alpha=0.85, label="model does worse when dry"),
         Line2D([], [], color=BLUE, lw=5, alpha=0.85, label="model does worse when wet")]
    axc.legend(handles=h, fontsize=8, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figS9_drought_control.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'figS9_drought_control'}.{{png,pdf}}")
    print(f"pares: {len(pares)}   peor en seco {k}/{n}   barrido {barrido}")


if __name__ == "__main__":
    main()
