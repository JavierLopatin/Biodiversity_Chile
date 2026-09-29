#!/usr/bin/env python3
"""El espectro del PCoA: por que la composicion no se ordena en este sistema.

`scripts/52` guarda dos ejes (`K_AXES <- 2`) y avisa si explican menos del 20 %. Explican el
13,0 %, o sea que el aviso salta -- pero el aviso no dice cuanto peor es el problema, y la
pregunta "y si usamos PCoA 3" solo se contesta con el espectro completo.

Este script lo calcula replicando la receta de `scripts/52` (Jaccard binario sobre la matriz
p/a lenosa, correccion de Cailliez) y emite los autovalores. La replica da 12,98 % en dos ejes
contra el 13,0 % que imprime el log de R, asi que el metodo coincide.

Lo que sale no es "faltan ejes", es que no hay estructura de baja dimension:

    PCoA 1  7,03 %      10 ejes   34 %
    PCoA 2  5,95 %      20 ejes   46 %
    PCoA 3  4,54 %      50 ejes   62 %
    PCoA 4  3,44 %     100 ejes   72 %
    PCoA 5  2,85 %     260 ejes   80 %

No hay codo: 7,0 -> 6,0 -> 4,5 -> 3,4 -> 2,9 es una rampa, no un escalon. Hacen falta 26 ejes
para la mitad del recambio floristico. La causa es que la matriz de distancias esta saturada --
la mayoria de los pares de parcelas no comparte ninguna especie, con Jaccard 1-- y una matriz
casi constante no se embebe en pocas dimensiones.

De ahi que el paper use LCBD y no una ordenacion: LCBD es un escalar por parcela que no exige
que la disimilitud sea ordenable. Y de ahi tambien que el aporte del satelite sobre el clima sea
mayor en LCBD (+0,067 sobre 0,144) que en PCoA 1 (+0,019 sobre 0,544): PCoA 1 es justamente la
parte de la composicion que SI es de baja dimension, el gradiente termico, y esa las grillas de
clima ya la tienen.

Salidas:
    results/tables/pcoa_spectrum.csv
    results/figures/figS7_pcoa_scree.{png,pdf}

Uso:
    python scripts/109_pcoa_spectrum.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
N_MUESTRA_PARES = 700      #: parcelas muestreadas para la fraccion de pares sin especies comunes


def comunidad() -> np.ndarray:
    """Matriz p/a lenosa, el mismo sustrato y el mismo `has_data` que `scripts/52`."""
    occ = pd.read_parquet(DERIVED / "occurrences_unified.parquet")
    lk = pd.read_csv(DERIVED / "growth_form_lookup.csv")
    occ = occ[occ.species.isin(set(lk.loc[lk.forma == "lenosa", "species"]))]
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")["PlotObservationID"]
    m = occ.assign(v=1).pivot_table(index="PlotObservationID", columns="species",
                                    values="v", aggfunc="max", fill_value=0)
    m = m.reindex(plots).fillna(0)
    return (m[m.sum(1) > 0].values > 0).astype(float)


def espectro(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Autovalores relativos del PCoA de Jaccard con correccion de Cailliez.

    Cailliez (Legendre & Legendre 2012, eq. 9.29) suma una constante a las distancias fuera de
    la diagonal hasta hacerlas euclideas; la constante es el mayor autovalor real de la matriz
    en bloques de abajo. Sin la correccion, los autovalores negativos que Jaccard produce hacen
    que las fracciones no sumen 1 y el porcentaje que se reporta deje de ser comparable.
    """
    n = X.shape[0]
    D = squareform(pdist(X, metric="jaccard"))
    J = np.eye(n) - np.ones((n, n)) / n
    cen = lambda A: -0.5 * J @ A @ J                                       # noqa: E731
    M = np.block([[np.zeros((n, n)), 2 * cen(D ** 2)], [-np.eye(n), -4 * cen(D)]])
    c = float(np.real(np.linalg.eigvals(M)).max())
    Dc = D + c
    np.fill_diagonal(Dc, 0.0)
    ev = np.linalg.eigvalsh(cen(Dc ** 2))[::-1]
    pos = ev[ev > 0]
    return pos / pos.sum(), c


def pares_sin_especies_comunes(X: np.ndarray, seed: int = 0) -> float:
    """Fraccion de pares de parcelas con Jaccard = 1, sobre una muestra.

    El mecanismo detras del espectro plano, y se mide sobre una muestra porque el conteo exacto
    son ~4,8 millones de pares y el numero no cambia la lectura.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], min(N_MUESTRA_PARES, X.shape[0]), replace=False)
    S = X[idx] > 0
    comunes = S.astype(float) @ S.astype(float).T
    iu = np.triu_indices(len(idx), k=1)
    return float((comunes[iu] == 0).mean())


def figura(rel: np.ndarray, frac: float, n: int, p: int) -> None:
    cum = np.cumsum(rel)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    k = 30
    a.bar(np.arange(1, k + 1), 100 * rel[:k], color="#4C78A8", zorder=3)
    a.set_xlabel("PCoA axis")
    a.set_ylabel("variance explained (%)")
    a.set_title("a  No dominant axis, no elbow", loc="left", fontweight="bold")
    a.annotate(f"axes 1–2 kept\nby the analysis: {100*cum[1]:.1f}%",
               xy=(2, 100 * rel[1]), xytext=(7.5, 100 * rel[0] * 0.82), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    a.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    a.set_axisbelow(True)

    b.plot(np.arange(1, len(cum) + 1), 100 * cum, color="#B4451F", lw=1.8, zorder=3)
    for q, col in [(50, "#666666"), (80, "#999999")]:
        need = int(np.searchsorted(cum, q / 100) + 1)
        b.axhline(q, color=col, lw=0.9, ls="--")
        b.annotate(f"{q}% needs {need} axes", xy=(need, q), xytext=(need * 1.5, q - 9),
                   fontsize=8.5, color="#333333",
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    b.set_xscale("log")
    b.set_xlabel("number of PCoA axes (log scale)")
    b.set_ylabel("cumulative variance explained (%)")
    b.set_title("b  Half the floristic turnover needs 26 axes", loc="left", fontweight="bold")
    b.grid(lw=0.4, color="#DDDDDD", zorder=0)
    b.set_axisbelow(True)

    fig.suptitle(f"Why woody composition is not ordinated: {n:,} plots, {p} species, "
                 f"{100*frac:.0f}% of plot pairs share no species (Jaccard = 1)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figS7_pcoa_scree.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIG / 'figS7_pcoa_scree'}.{{png,pdf}}")


def main() -> None:
    X = comunidad()
    n, p = X.shape
    print(f"matriz p/a: {n} parcelas x {p} especies "
          f"(media {X.sum(1).mean():.2f} especies/parcela)")
    rel, c = espectro(X)
    frac = pares_sin_especies_comunes(X)
    print(f"constante de Cailliez c = {c:.4f}")
    print(f"pares sin ninguna especie en comun: {100*frac:.1f} %")

    cum = np.cumsum(rel)
    t = pd.DataFrame({"eje": np.arange(1, len(rel) + 1),
                      "varianza": rel, "acumulada": cum})
    TAB.mkdir(parents=True, exist_ok=True)
    t.to_csv(TAB / "pcoa_spectrum.csv", index=False)
    print(f"-> {TAB / 'pcoa_spectrum.csv'}")
    print()
    for i in range(6):
        print(f"  PCoA {i+1}: {100*rel[i]:6.2f} %   acumulado {100*cum[i]:6.2f} %")
    for q in (0.5, 0.8):
        print(f"  ejes para {int(100*q)} %: {int(np.searchsorted(cum, q) + 1)}")
    figura(rel, frac, n, p)


if __name__ == "__main__":
    main()
