#!/usr/bin/env python3
"""Figura: donde falla el modelo, y por que no es regresion a la media.

El paper tiene R2 y le falta ecologia. Esta figura es la capa que falta, y es deliberadamente
una pregunta de RESIDUOS y no una exploracion: "en que zonas funciona" se convierte en pesca;
"que distingue lo bien predicho de lo mal predicho" esta acotado por construccion.

Las 2.499 parcelas con LCBD se parten en cuartiles del residuo absoluto del mejor bloque
(geomediano + clima), y cada cuartil se describe. Seis paneles, en el orden del argumento:

    a  el eje        el residuo absoluto crece diez veces de Q1 a Q4
    b  el motor      la media del LCBD observado es plana pero la dispersion no
    c  EL CONTROL    que patrones sobreviven al condicionar por esa dispersion
    d  estructura    el peor cuartil tiene una dominante que se lleva mas
    e  nicho         y mas especialistas de rango altitudinal estrecho
    f  quienes       Nothofagus pasa del 55 % al 77 % de las parcelas

EL PANEL B ES EL QUE DECIDE, Y NO DICE LO QUE PARECE. El residuo absoluto mezcla dos cosas:
parcelas cuyo LCBD es extremo y el modelo regresa a la media, y parcelas donde el predictor no
informa. La MEDIA del LCBD observado es plana entre cuartiles -- 4,04e-4 en Q1 contra 3,99e-4 en
Q4-- pero la DISPERSION no: la desviacion absoluta respecto de la media global pasa de 0,143 a
0,246 (en 1e-4), y la fraccion de parcelas en las colas del 10 % global pasa de 0,090 a 0,354.
O sea la regresion a la media SI esta, y de hecho es el correlato mas fuerte del residuo:
rho = +0,347, contra -0,173 del numero de especies y +0,171 de la dominancia.

El panel c es ese control y es lo que la figura tenia que mostrar y no mostraba. Condicionando
por quintiles de |LCBD - media|:

    numero de especies        NO sobrevive: -0,21, -0,06, +0,06, +0,01, -0,17 -- cambia de signo
    dominancia                sobrevive debil y uniforme: +0,13, +0,10, +0,02, +0,08, +0,09
    especialistas estrechos   sobrevive y SE FORTALECE: +0,04, +0,08, +0,08, +0,25, +0,34
    Nothofagus presente       igual: +0,02, -0,01, +0,02, +0,19, +0,62

Los dos ultimos no son efectos principales sino INTERACCIONES con la extremidad, y eso cambia
la lectura. No es que el modelo falle donde hay menos especies -- ese patron es marginal y se
disuelve-- sino que DONDE LA COMUNIDAD ES INUSUAL falla especificamente en los rodales de
Nothofagus con especialistas de rango estrecho. En el quintil mas extremo el residuo medio es
2,4e-5 con Nothofagus contra 1,0e-5 sin el. Y no es el confundido al reves: la extremidad media
es MENOR en las parcelas con Nothofagus (1,68e-5 contra 2,03e-5), asi que el condicionamiento
revela el patron en vez de crearlo.

Por eso el panel de "menos especies" se fue: el numero de especies aparece ahora solo en el
panel c, como la linea que cruza el cero. Los paneles con distribucion son los patrones que
sobreviven.

El panel e usa el rasgo de rango altitudinal del catalogo de Rodriguez et al. 2018
(`scripts/81`) como DESCRIPCION, no como respuesta nueva a modelar. El rasgo cubre el 90-93 %
de las ocurrencias y la cobertura SUBE hacia el peor cuartil, asi que el patron no es un hueco
de datos.

Fuentes: `results/tables/residuos_por_parcela.csv` y `residuos_taxones.csv` (`scripts/111`).
Ningun numero esta escrito a mano.

Uso:
    python scripts/123_figure_where_it_fails.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

#: De bien predicho a mal predicho. Un degradado y no cuatro colores sueltos: el cuartil es
#: una variable ordenada y el color tiene que decirlo.
COLORES = ["#3B6E9E", "#8BA9C4", "#D89B91", "#C0392B"]
DARK, GREY = "#333333", "#9A9A9A"
ORDEN = ["Q1 mejor", "Q2", "Q3", "Q4 peor"]
ETIQ = ["Q1\nbest", "Q2", "Q3", "Q4\nworst"]
DPI = 300
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})


def violines(ax, d: pd.DataFrame, col: str, escala: float = 1.0, log: bool = False) -> None:
    """Violin por cuartil con la mediana marcada. Violin y no barra: con 625 parcelas por
    cuartil lo que importa es si la DISTRIBUCION se corre, no si la media lo hace."""
    datos = [d.loc[d.cuartil == q, col].dropna().to_numpy() * escala for q in ORDEN]
    v = ax.violinplot(datos, positions=range(4), widths=0.78, showextrema=False)
    for cuerpo, c in zip(v["bodies"], COLORES):
        cuerpo.set_facecolor(c)
        cuerpo.set_alpha(0.75)
        cuerpo.set_linewidth(0)
    for i, x in enumerate(datos):
        ax.plot([i - 0.22, i + 0.22], [np.median(x)] * 2, color=DARK, lw=2, zorder=4)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(ETIQ)
    ax.grid(axis="y", lw=0.4, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    d = pd.read_csv(TAB / "residuos_por_parcela.csv")
    d["cuartil"] = pd.Categorical(d.cuartil, ORDEN, ordered=True)
    occ = pd.read_parquet(ROOT / "data" / "derived" /
                          "occurrences_unified_counts_woody.parquet")
    noto = set(occ.loc[occ.species.str.startswith("Nothofagus"), "PlotObservationID"])
    d["noto"] = d.PlotObservationID.isin(noto).astype(float)
    tx = pd.read_csv(TAB / "residuos_taxones.csv")
    print(f"{len(d)} parcelas, {d.groupby('cuartil', observed=True).size().to_dict()}")

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4))
    (a, b, c), (e, f, g) = axes

    violines(a, d, "res", escala=1e5, log=True)
    a.set_ylabel("absolute residual ($\\times 10^{-5}$)")
    a.set_title("a  The sorting axis", loc="left", fontweight="bold")

    violines(b, d, "obs", escala=1e4)
    b.set_ylabel("observed LCBD ($\\times 10^{-4}$)")
    b.set_title("b  Flat mean, wider spread", loc="left", fontweight="bold")
    gm = d.obs.mean()
    ext = d.groupby("cuartil", observed=True).obs.apply(lambda v: (v - gm).abs().mean()) * 1e4
    b.annotate(f"mean |deviation| {ext.iloc[0]:.3f} $\\to$ {ext.iloc[-1]:.3f}",
               xy=(0.5, 0.03), xycoords="axes fraction", ha="center", fontsize=8,
               color="#555555")

    # --- c  EL CONTROL: que sobrevive al condicionar por la extremidad del LCBD
    gm2 = d.obs.mean()
    d["ext"] = (d.obs - gm2).abs()
    d["q_ext"] = pd.qcut(d.ext, 5, labels=[f"E{i+1}" for i in range(5)])
    series = [("n_sp", "species per plot", "#4C78A8", "-"),
              ("dom", "dominance", "#54A24B", "-"),
              ("frac_estrecho", "narrow-range fraction", "#B279A2", "-"),
              ("noto", "$\\it{Nothofagus}$ present", COLORES[-1], "-")]
    c.axhline(0, color=DARK, lw=0.9, zorder=2)
    for col, etq, color, ls in series:
        r = [spearmanr(g.res, g[col], nan_policy="omit").statistic
             for _, g in d.groupby("q_ext", observed=True)]
        c.plot(range(5), r, ls, color=color, lw=2.2, marker="o", ms=5, label=etq, zorder=3)
    c.set_xticks(range(5))
    c.set_xticklabels([f"E{i+1}" for i in range(5)])
    c.set_xlabel("quintile of $|$LCBD $-$ mean$|$")
    c.set_ylabel(r"$\rho$ with the residual")
    c.set_title("c  The control: what survives", loc="left", fontweight="bold")
    c.legend(fontsize=7.5, loc="upper left", framealpha=0.95, labelspacing=0.3)
    c.grid(axis="y", lw=0.4, color="#EEEEEE", zorder=0)
    c.set_axisbelow(True)

    violines(e, d, "dom")
    e.set_ylabel("share of the dominant species")
    e.set_title("d  More dominated", loc="left", fontweight="bold")
    e.set_xlabel("quartile of absolute residual")

    violines(f, d, "frac_estrecho")
    f.set_ylabel("narrow-range species in the plot")
    f.set_title("e  More narrow-range specialists", loc="left", fontweight="bold")

    # --- f  quienes: generos mas frecuentes, Q1 contra Q4
    w = (tx[tx.nivel == "genero"]
         .pivot_table(index="taxon", columns="cuartil", values="frac_parcelas")
         .reindex(columns=["Q1 mejor", "Q4 peor"]).fillna(0.0))
    w["dif"] = w["Q4 peor"] - w["Q1 mejor"]
    w = w.sort_values("dif")
    y = np.arange(len(w))
    for i, (nombre, r) in enumerate(w.iterrows()):
        col = COLORES[-1] if r.dif > 0 else COLORES[0]
        g.plot([r["Q1 mejor"], r["Q4 peor"]], [i, i], color=col, lw=2.2, alpha=0.85, zorder=3)
        g.plot(r["Q1 mejor"], i, "o", ms=5, color="white", mec=col, mew=1.6, zorder=4)
        g.plot(r["Q4 peor"], i, "o", ms=6, color=col, mec="white", mew=0.8, zorder=5)
    g.set_yticks(y)
    g.set_yticklabels([f"$\\it{{{n}}}$" for n in w.index], fontsize=8)
    g.set_xlabel("fraction of plots where the genus occurs")
    g.set_title("f  Who dominates the worst quartile", loc="left", fontweight="bold")
    g.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
    g.set_axisbelow(True)
    g.legend(handles=[Line2D([], [], marker="o", ls="none", ms=6, color="white", mec=DARK,
                             mew=1.6, label="Q1 best"),
                      Line2D([], [], marker="o", ls="none", ms=6, color=DARK, label="Q4 worst")],
             fontsize=8, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig04_where_it_fails.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'fig04_where_it_fails'}.{{png,pdf}}")
    r = d.groupby("cuartil", observed=True).agg(
        residuo=("res", "mean"), lcbd_obs=("obs", "mean"), n_sp=("n_sp", "mean"),
        dominancia=("dom", "mean"), estrecho=("frac_estrecho", "mean"))
    print(r.to_string(float_format=lambda v: f"{v:.5g}"))
    print(f"\nNothofagus: Q1 {w.loc['Nothofagus', 'Q1 mejor']:.3f} -> "
          f"Q4 {w.loc['Nothofagus', 'Q4 peor']:.3f}")


if __name__ == "__main__":
    main()
