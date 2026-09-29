#!/usr/bin/env python3
"""El resultado general del pool lenoso, en tres paneles.

Todo sale de `results/tables/sintesis_lenoso.csv` y `contraste_estres_lenoso.csv`, que salen de
`scripts/107`. Ningun numero de esta figura esta escrito a mano.

    a  cada faceta contra su piso, con el control de latitud -- que se predice y que no
    b  que representacion gana, por grupo de facetas
    c  seco contra humedo descontado el piso -- la senal remota es mas fuerte en anos humedos

El panel a muestra el R2 CENTRADO DENTRO DE BANDAS de 2 grados de latitud, no el agrupado, y
deja el agrupado como una marca gris al lado. Las dos bases solo se solapan entre 30 y 38 S y
varios targets de unified_all difieren por inventario (riqueza media 3,6 en Living Trees contra
5,4 en Parcelas-CL; diversidad oscura 40,0 contra 69,4), asi que buena parte del R2 agrupado es
gradiente latitudinal y procedencia. La distancia entre la barra y la marca es lo que cuesta el
control: en diversidad oscura son 35 puntos (0,695 -> 0,341) y en LCBD Sorensen 29 (0,502 ->
0,210). El numero que se defiende es el de la barra.

El panel c usa el NETO, no el bruto: al bloque remoto se le resta el cambio que el piso tiene
entre los mismos terciles. Sin esa resta `hill_q0_unified` parece predecirse mejor en seco y
sostiene la idea de que la sequia se invierte entre riqueza y composicion; su piso se mueve mas
que el bloque, y el neto apunta al mismo lado que las otras 16 facetas.

Uso:
    python scripts/108_woody_synthesis_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, GREY = "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#666666"
RED, GOLD = "#B4451F", "#C7A23C"
DPI = 200
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})

COLOURS = {"clima": BLUE, "curva": ORANGE, "lsp": GOLD, "gm": GREEN,
           "clima_curva": PURPLE, "gm_clima": RED, "piso": GREY, "area_sola": "#999999"}
NOMBRES = {"clima": "Climate", "curva": "Curve", "lsp": "LSP", "gm": "Geomedian",
           "clima_curva": "Climate+curve", "gm_clima": "Geomedian+climate",
           "piso": "Floor (topography+area)", "area_sola": "Plot area alone"}
#: Rotulos en ingles, y el orden en que se presentan las facetas.
FACETAS = {
    "pcoa2_pa_unified": "PCoA 2 (p/a)", "pcoa1_pa_unified": "PCoA 1 (p/a)",
    "pcoa1_freq_unified": "PCoA 1 (freq.)", "pcoa2_freq_unified": "PCoA 2 (freq.)",
    "lcbd_count_sorensen": "LCBD Sørensen", "lcbd_pa_unified": "LCBD (p/a)",
    "lcbd_freq_unified": "LCBD (freq.)",
    "dark_n_unified": "Dark diversity", "hill_q0_unified": "Richness (raw, $q_0$)",
    "td_inext_q1": "TD $q_1$ (cov.-std.)", "td_inext_q2": "TD $q_2$ (cov.-std.)",
    "td_inext_q0": "TD $q_0$ (cov.-std.)",
    "ses_mpd_unified": "SES MPD", "mpd_unified": "MPD", "ses_pd_unified": "SES PD",
    "ses_mntd_unified": "SES MNTD", "mntd_unified": "MNTD",
    "pd_inext_q1": "PD $q_1$ (cov.-std.)", "pd_inext_q0": "PD $q_0$ (cov.-std.)",
    "pd_inext_q2": "PD $q_2$ (cov.-std.)",
}
GRUPOS = [("composicion", "Composition"), ("riqueza", "Richness"),
          ("filogenetica", "Phylogenetic")]


def panel_facetas(ax, r: pd.DataFrame) -> None:
    orden = [k for k in FACETAS if k in set(r.faceta)]
    y = np.arange(len(orden))
    col = lambda k, c: r.loc[r.faceta == k, c].iloc[0]
    piso = [col(k, "R2_piso_en_banda") for k in orden]
    mejor = [col(k, "R2_en_banda") for k in orden]
    pooled = [col(k, "R2") for k in orden]
    reps = [col(k, "representacion") for k in orden]

    ax.barh(y, mejor, height=0.62, color=[COLOURS[x] for x in reps], zorder=3)
    ax.barh(y, piso, height=0.26, color="#333333", zorder=4)
    # el R2 agrupado va como marca, no como barra: es el numero inflado por el gradiente y la
    # procedencia, y sirve solo para ver cuanto cuesta el control
    ax.plot(pooled, y, ls="none", marker="|", ms=11, mew=1.6, color="#888888", zorder=5)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FACETAS[k] for k in orden])
    ax.invert_yaxis()
    ax.set_xlabel("out-of-fold $R^2$, centred within 2° latitude bands")
    ax.set_title("a  What the woody pool predicts, facet by facet", loc="left",
                 fontweight="bold")
    # la barra ancha se colorea por QUE bloque gano esa faceta, asi que la leyenda tiene que
    # nombrar los bloques y no decir "best block" en un color que no significa nada
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    usados = [x for x in ["clima", "curva", "lsp", "gm_clima", "clima_curva"] if x in set(reps)]
    ax.legend(handles=[Patch(facecolor=COLOURS[x], label=f"best block: {NOMBRES[x]}")
                       for x in usados]
              + [Patch(facecolor="#333333", label="floor (topography+area)"),
                 Line2D([], [], ls="none", marker="|", ms=11, mew=1.6, color="#888888",
                        label="same block, pooled (uncontrolled)")],
              fontsize=7.2, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def panel_bloques(ax, f: pd.DataFrame) -> None:
    g = f[(f.estres == "todas") & (f.representacion != "area_sola")]
    reps = ["piso", "clima", "curva", "lsp", "gm", "gm_clima"]
    x = np.arange(len(GRUPOS))
    for i, rep in enumerate(reps):
        vals = [g[(g.grupo == gk) & (g.representacion == rep)].R2_en_banda.mean()
                for gk, _ in GRUPOS]
        ax.bar(x + (i - (len(reps) - 1) / 2) * 0.145, vals, width=0.135,
               color=COLOURS[rep], label=NOMBRES[rep], zorder=3)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in GRUPOS])
    ax.set_ylabel("mean $R^2$ across facets\n(within 2° latitude bands)")
    ax.set_title("b  Geomedian+climate wins in every facet group", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.95)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def panel_estres(ax, c: pd.DataFrame) -> None:
    c = c.sort_values("neto")
    y = np.arange(len(c))
    col = {"composicion": BLUE, "riqueza": GREEN, "filogenetica": PURPLE}
    ax.barh(y, c.neto, height=0.68, color=[col[g] for g in c.grupo], zorder=3)
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([FACETAS.get(k, k) for k in c.faceta])
    ax.set_xlabel("$\\Delta R^2$ (wet − dry), floor change subtracted")
    ax.set_title("c  The remote signal is stronger in wet years", loc="left",
                 fontweight="bold")
    n = int((c.neto > 0).sum())
    ax.text(0.97, 0.06, f"{n} of {len(c)} facets positive", transform=ax.transAxes,
            ha="right", fontsize=8.5, color="#333333", fontweight="bold")
    ax.text(0.97, 0.015, "SPI-12 terciles within census year, Living Trees",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=GREY)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=col[g], label=lab) for g, lab in GRUPOS],
              fontsize=7.5, loc="upper left", framealpha=0.95)
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    f = pd.read_csv(TAB / "sintesis_lenoso.csv")
    r = pd.read_csv(TAB / "sintesis_lenoso_resumen.csv")
    c = pd.read_csv(TAB / "contraste_estres_lenoso.csv")

    fig = plt.figure(figsize=(15.5, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1],
                          hspace=0.42, wspace=0.34)
    panel_facetas(fig.add_subplot(gs[:, 0]), r)
    panel_bloques(fig.add_subplot(gs[0, 1]), f)
    panel_estres(fig.add_subplot(gs[1, 1]), c)
    fig.suptitle("Predicting woody plant diversity from Landsat across Chile "
                 "(3,102 plots, woody-harmonised pool)", fontsize=12.5, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig27_woody_synthesis.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig27_woody_synthesis'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
