#!/usr/bin/env python3
"""La linea base de ubicacion contra el radio de exclusion, en las dos redes de parcelas.

Por que esta figura existe. El paper mide el margen con bloques de 20 km, y el numero 20 es
nominal: medido en `scripts/153`, un bloque de 20 km deja el dato de entrenamiento mas cercano
a 9,6 km en Parcelas-CL y a 9,3 km en Living Trees, no a 20. B-LOO sustituye ese parametro
nominal por la distancia real, y su radio 0 ES la validacion cruzada aleatoria, asi que una
sola curva cubre el rango entero entre el protocolo que Wadoux et al. (2021) defienden y el
que critican.

Lo que la figura muestra, y que no es lo que yo esperaba. Las dos redes NO recorren la misma
curva en puntos de operacion distintos: recorren curvas de FORMA distinta.

    Parcelas-CL   0,686 a radio 0  ->  -0,017 a 80 km   (caida 0,703)
    Living Trees  0,511 a radio 0  ->  +0,430 a 80 km   (caida 0,081)

La prueba es pareada por (faceta, grupo), 6 celdas por red, y no se solapan: la caida menor de
Parcelas-CL (0,424) es 3,5 veces la mayor de Living Trees (0,121).

La lectura. Las dos lineas base de ubicacion predicen por mecanismos distintos. Parcelas-CL
cubre 7,6 grados en una nube apretada --- vecino mas cercano a 0,15 km --- y su linea base es
INTERPOLACION LOCAL: al quitarle los vecinos se muere. Living Trees cubre 24,3 grados repartido
--- vecino a 3,07 km --- y su linea base es el GRADIENTE LATITUDINAL, que un buffer de 80 km no
toca porque no es estructura local. De ahi que el optimismo del protocolo aleatorio sea seis
veces mayor en la red compacta: lo que el protocolo puede regalar es la parte interpolable, y
solo una de las dos redes la tiene en cantidad.

Consecuencia para el paper, en el otro sentido del que suele suponerse: MAS extension hace la
conclusion MENOS sensible al protocolo, porque desplaza la linea base de interpolacion a
gradiente.

El panel b es la robustez del resultado del paper: el margen del diseno sobre la linea base en
Living Trees es plano contra el radio (0,056 a radio 0, 0,063 a 20 km), asi que la conclusion
no depende de haber elegido 20 km.

Reservas declaradas, las dos del lado de Parcelas-CL:
  - el tope comun de entrenamiento (800) no se alcanza ahi: a 80 km solo sobreviven 716-752,
    o sea un 9 % menos de datos que en Living Trees. Sube el R2 de Living Trees por igual en
    todos los radios, no le cambia la forma, que es lo que se compara.
  - `lcbd_count_sorensen` queda FUERA de la figura: en Parcelas-CL corre con 479 parcelas y 30
    de test, contra 2.001 y 80 en Living Trees, y un grupo no alcanzo el minimo.

Uso:
    python scripts/154_figura_bloo_redes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from biodiv import figures as fg              # noqa: E402

DPI = 300
#: Las dos redes caen, cada una, en una de las extensiones del registro: Parcelas-CL mide 7,6
#: grados y la extension `regional` del paper 8,0; Living Trees mide 24,3 y la `national` 24,3.
#: Asi que el color y el marcador son los que esas extensiones ya tienen en el resto del paper.
#: El rotulo nombra las DOS cosas --- inventario y extension --- justamente porque confundirlas
#: es el error que esta figura corrige.
REDES = {"parcelas_cl":  ("regional", "Parcelas-CL, 7.6$^\\circ$"),
         "living_trees": ("national", "Living Trees, 24.3$^\\circ$")}
#: Esta faceta no entra: en Parcelas-CL corre con 479 parcelas y 30 de test (ver docstring).
FUERA = "lcbd_count_sorensen"
#: Lo que cada protocolo deja como distancia al entrenamiento mas cercano, de `scripts/153`.
#: Es el puente entre esta figura y las que usan bloques: marca DONDE cae cada protocolo
#: sobre el eje que esta figura recorre.
OPERACION = {"parcelas_cl":  dict(aleatorio=0.165, bloques=9.64),
             "living_trees": dict(aleatorio=3.54, bloques=9.26)}
#: El radio 0 no existe en un eje logaritmico; se dibuja en esta posicion y se
#: rotula "0 (random)". Queda a la izquierda del primer radio real, 0,5 km.
R0 = 0.25
fg.set_paper_style()


def carga() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(TAB / f"bloo_{s}.csv") for s in ("pcl", "lt")],
                  ignore_index=True)
    assert "inventario" in d.columns, "la tabla no trae inventario: corrida vieja"
    d = d[d.faceta != FUERA]
    assert set(d.inventario) == set(REDES), f"inventarios inesperados: {set(d.inventario)}"
    return d


def main() -> None:
    d = carga()
    # Un panel POR FACETA y no una media de las dos: las dos facetas tienen niveles distintos
    # de R2, asi que promediarlas convierte esa diferencia de nivel en una barra de error de
    # +-0,21 y entierra justamente lo que hay que juzgar, que es la FORMA de cada curva. Por
    # faceta, la barra es la dispersion entre los tres grupos y vale 0,04--0,11. Ademas el
    # contraste aparece dos veces de forma independiente, que es mejor evidencia que una media.
    facetas = [f for f in ("hill_q0_unified", "isomap1_pa_unified") if f in set(d.faceta)]
    rotulo = {"hill_q0_unified": "richness $q_0$", "isomap1_pa_unified": "Isomap 1 (p/a)"}
    fig, axes = plt.subplots(1, len(facetas) + 1, figsize=(13.2, 4.0),
                             gridspec_kw=dict(wspace=0.28))

    def serie(ax, t, red, col, ms_extra=0.0):
        e = fg.EXTENT[REDES[red][0]]
        g = t.groupby("radio_km")[col]
        m, sd = g.mean(), g.std()
        x = m.index.to_numpy()
        comun = dict(color=e["color"], marker=e["marker"], elinewidth=0.8, capsize=0,
                     mfc=e["color"] if e["fill"] else "none", mec=e["color"])
        # El radio 0 se dibuja aparte porque el eje es logaritmico y el 0 no vive en el. Va
        # mas grande: es la CV aleatoria, el extremo del rango que la figura recorre.
        ax.errorbar(x[1:], m.to_numpy()[1:], yerr=sd.to_numpy()[1:], ls=e["ls"],
                    ms=e["ms"] + ms_extra, lw=1.4, mew=0.9, **comun)
        ax.errorbar([R0], m.to_numpy()[:1], yerr=sd.to_numpy()[:1], ls="none",
                    ms=e["ms"] + 1.6 + ms_extra, mew=1.3, **comun)

    for ax, fac in zip(axes, facetas):
        for red in REDES:
            serie(ax, d[(d.inventario == red) & (d.faceta == fac)], red, "R2_nulo")
        # Donde cae cada protocolo sobre ESTE eje: la distancia real que deja, medida en
        # `scripts/153`, no su parametro nominal. Es el puente con las figuras de bloques.
        for red, (ext, _) in REDES.items():
            c = fg.EXTENT[ext]["color"]
            for cual, est in (("aleatorio", (0, (1, 1.6))), ("bloques", (0, (4, 2)))):
                ax.axvline(OPERACION[red][cual], color=c, lw=0.9, ls=est, alpha=0.5)
        ax.set_ylabel("$R^2$ of the location baseline")
        ax.annotate(rotulo[fac], xy=(0.5, 0.97), xycoords="axes fraction",
                    ha="center", va="top", fontsize=7.5, color=fg.NEUTRAL["ink"])

    axm = axes[-1]
    for red in REDES:
        serie(axm, d[d.inventario == red], red, "margen")
    axm.set_ylabel("margin of the design over the baseline")
    axm.annotate("both response variables", xy=(0.5, 0.97), xycoords="axes fraction",
                 ha="center", va="top", fontsize=7.5, color=fg.NEUTRAL["ink"])

    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xticks([R0, 0.5, 1, 2, 5, 10, 20, 40, 80])
        ax.set_xticklabels(["0\n(random)", "0.5", "1", "2", "5", "10", "20", "40", "80"])
        ax.axhline(0, color=fg.NEUTRAL["zero"], lw=1.0, ls="--")
        ax.set_xlabel("exclusion radius (km)")
        ax.set_title("abc"[i], loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(handles=[Line2D([], [], color=fg.EXTENT[e]["color"],
                                   marker=fg.EXTENT[e]["marker"], ls=fg.EXTENT[e]["ls"],
                                   ms=fg.EXTENT[e]["ms"], label=lab)
                            for _, (e, lab) in REDES.items()]
                   + [Line2D([], [], color=fg.NEUTRAL["grid"], ls=(0, (1, 1.6)), lw=0.9,
                             label="where random CV sits"),
                      Line2D([], [], color=fg.NEUTRAL["grid"], ls=(0, (4, 2)), lw=0.9,
                             label="where 20 km blocks sit")],
                   loc="lower left", frameon=False, fontsize=6.5)

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig10_bloo_redes.{ext}", dpi=DPI, bbox_inches="tight")
    print(f"-> {FIG}/fig10_bloo_redes.pdf\n")
    for fac in facetas:
        for red in REDES:
            t = (d[(d.inventario == red) & (d.faceta == fac)]
                 .pivot_table(index="grupo", columns="radio_km", values="R2_nulo"))
            caida = t[0.0] - t[80.0]
            print(f"{rotulo[fac]:16s} {red:13s} {t[0.0].mean():+.3f} -> "
                  f"{t[80.0].mean():+.3f}   caida {caida.mean():+.3f}   "
                  f"sd entre grupos {t.std().mean():.3f}")
    print("\nmargen contra radio (media de 2 facetas x 3 grupos):")
    print(d.groupby(["inventario", "radio_km"]).margen.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
