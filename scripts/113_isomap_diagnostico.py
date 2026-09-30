#!/usr/bin/env python3
"""Por que Isomap y no PCoA: el techo de reconstruccion, y que representa cada eje.

Los dos diagnosticos que sostienen la decision de cambiar la ordenacion. Se corren aqui, en
Python, sobre una submuestra, porque su papel es DECIDIR -- la ordenacion que entra al paper la
produce `scripts/52` en R, con `vegan::isomap`, sobre el pool entero.

1. TECHO DE RECONSTRUCCION. No es un modelo: es el limite superior de cualquier diseno que
   regrese k ejes. Si conocieras los ejes exactos de las parcelas de test, que tan bien
   reproducirias su Jaccard observado. Es la medida que decide, porque comparar la varianza
   explicada de dos ordenaciones distintas no dice cual conserva mas de la disimilitud real.

       k ejes      PCoA    Isomap k=10   k=30   k=80
            2     0,419          0,582  0,577  0,557
            3     0,444          0,588  0,606  0,593
            5     0,480          0,609  0,620  0,616
            8     0,502          0,619  0,636  0,632

   Isomap con DOS ejes le gana a PCoA con OCHO. Y los dos riesgos que la literatura marca no
   aparecen aqui: el grafo de k vecinos queda conexo con k = 10, 30 y 80, y k importa poco
   (0,577-0,606 con 3 ejes), asi que la sensibilidad a k que Feilhauer et al. 2011 declara como
   limitacion -- ellos optimizaron k sobre los mismos datos que despues mapearon-- no es un
   problema en este pool.

   El mecanismo: el PCoA embebe las distancias directas, y con el 77,6 % de los pares saturados
   en Jaccard = 1 la geometria global es casi un simplex, que no se representa en pocas
   dimensiones. Isomap mide distancias GEODESICAS sobre el grafo de k vecinos, asi que dos
   parcelas sin ninguna especie en comun nunca se comparan directamente: su distancia va por un
   camino de parcelas intermedias que si comparten.

2. QUE REPRESENTA CADA EJE, y cuanto de el da la geografia sola. Necesario para no vender como
   composicion lo que es un gradiente climatico:

       eje 1   precipitacion del trimestre seco +0,885, latitud -0,825, MAP +0,817
       eje 2   RIQUEZA +0,523, latitud solo +0,101
       eje 3   elevacion +0,507, latitud -0,020

   El eje 2 es el interesante: es un eje de riqueza casi ortogonal a la latitud. Pero la
   geografia sola (lon, lat, elevacion, con RF y CV por bloque de latitud) predice los tres a
   R2 0,778 / 0,726 / 0,578, asi que un compuesto RGB de estos ejes seria en buena parte una
   imagen de latitud y elevacion.

Salida: `results/tables/isomap_diagnostico.csv` y `isomap_ejes.csv`.

Uso:
    python scripts/113_isomap_diagnostico.py [--n 1200]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "results" / "tables" / "isomap_diagnostico.csv"

KS = (10, 30, 80)           #: vecinos del grafo, para la sensibilidad
EJES = (2, 3, 5, 8)         #: cuantos ejes se prueban en el techo
#: Submuestra. El pdist completo sobre las 3.094 son 4,8 millones de pares y el diagnostico no
#: cambia: es una comparacion entre dos ordenaciones sobre las MISMAS parcelas.
N_DEFAULT = 1200


def comunidad(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Matriz p/a lenosa y los ids, el mismo sustrato y el mismo `has_data` que `scripts/52`."""
    occ = pd.read_parquet(DERIVED / "occurrences_unified.parquet")
    lk = pd.read_csv(DERIVED / "growth_form_lookup.csv")
    occ = occ[occ.species.isin(set(lk.loc[lk.forma == "lenosa", "species"]))]
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")["PlotObservationID"]
    m = occ.assign(v=1).pivot_table(index="PlotObservationID", columns="species",
                                    values="v", aggfunc="max", fill_value=0)
    m = m.reindex(plots).fillna(0)
    m = m[m.sum(1) > 0]
    sel = np.sort(np.random.default_rng(seed).choice(len(m), min(n, len(m)), replace=False))
    return (m.values[sel] > 0).astype(float), m.index.to_numpy()[sel]


def _cen(A: np.ndarray) -> np.ndarray:
    n = len(A)
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ A @ J


def _emb(G: np.ndarray, k: int) -> np.ndarray:
    w, v = np.linalg.eigh(G)
    o = np.argsort(w)[::-1][:k]
    return v[:, o] * np.sqrt(np.clip(w[o], 0, None))


def gram_pcoa(D: np.ndarray) -> np.ndarray:
    """Gram del PCoA con correccion de Cailliez, la receta de `scripts/52`."""
    n = len(D)
    M = np.block([[np.zeros((n, n)), 2 * _cen(D ** 2)], [-np.eye(n), -4 * _cen(D)]])
    c = float(np.real(np.linalg.eigvals(M)).max())
    Dc = D + c
    np.fill_diagonal(Dc, 0.0)
    return _cen(Dc ** 2)


def gram_isomap(D: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Gram de Isomap: distancias geodesicas sobre el grafo de k vecinos."""
    g = kneighbors_graph(D, k, metric="precomputed", mode="distance")
    ncomp, _ = connected_components(g, directed=False)
    gd = shortest_path(g, method="D", directed=False)
    if not np.isfinite(gd).all():
        # grafo desconexo: los pares inalcanzables se cierran al maximo observado por 1,5, que
        # es lo que hace `vegan::isomap(fragmentedOK=TRUE)` en espiritu. Si ncomp > 1 hay que
        # decirlo, no taparlo.
        gd = np.where(np.isfinite(gd), gd, np.nanmax(gd[np.isfinite(gd)]) * 1.5)
    return _cen(gd ** 2), int(ncomp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    a = ap.parse_args()

    X, ids = comunidad(a.n)
    n = len(X)
    D = squareform(pdist(X, metric="jaccard"))
    obs = D[np.triu_indices(n, 1)]
    print(f"{n} parcelas, {len(obs):,} pares, Jaccard = 1 en {100 * (obs == 1).mean():.1f} %")

    grams = {"PCoA": (gram_pcoa(D), np.nan)}
    for k in KS:
        grams[f"Isomap k={k}"] = gram_isomap(D, k)

    filas = []
    for nombre, (G, ncomp) in grams.items():
        for k in EJES:
            filas.append(dict(ordenacion=nombre, n_ejes=k, componentes=ncomp,
                              techo=float(spearmanr(pdist(_emb(G, k)), obs).statistic)))
    t = pd.DataFrame(filas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(f"-> {OUT}")
    print("\nTecho de reconstruccion (rho de Spearman contra el Jaccard observado):")
    print(t.pivot_table(index="n_ejes", columns="ordenacion", values="techo").round(3).to_string())
    print("\ncomponentes conexas del grafo kNN:",
          {k: int(grams[f'Isomap k={k}'][1]) for k in KS})

    ejes(grams["Isomap k=30"][0], ids)


def ejes(G: np.ndarray, ids: np.ndarray) -> None:
    """Que representa cada eje, y cuanto de el da la geografia sola."""
    E = _emb(G, 3)
    P = pd.read_parquet(DERIVED / "plots_unified.parquet").set_index("PlotObservationID")
    ctx = P.loc[ids, ["lat", "lon", "elevation", "richness"]].copy()
    c = pd.read_parquet(DERIVED / "climate_unified.parquet")
    if c.index.name:
        c = c.reset_index()
    ctx = ctx.join(c.set_index("PlotObservationID"))
    vs = [v for v in ["lat", "lon", "elevation", "richness", "clim_mat", "clim_map",
                      "clim_p_driest_quarter", "clim_tmax_warmest"] if v in ctx.columns]

    Xg = ctx[["lon", "lat", "elevation"]].to_numpy(float)
    ok = np.isfinite(Xg).all(1)
    # bloques de ~20 km en latitud, el mismo espiritu que el CV espacial del proyecto: sin
    # agrupar, un RF sobre coordenadas interpola entre vecinos y el R2 no significa nada.
    grp = (np.floor(ctx.lat / 0.2) * 0.2).astype(str)

    filas = []
    for i in range(3):
        f = {"eje": i + 1}
        for v in vs:
            f[f"rho_{v}"] = float(spearmanr(E[:, i], ctx[v], nan_policy="omit").statistic)
        p = cross_val_predict(RandomForestRegressor(400, random_state=0, n_jobs=-1),
                              Xg[ok], E[ok, i], cv=GroupKFold(5), groups=grp[ok])
        y = E[ok, i]
        f["R2_geografia_sola"] = float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        filas.append(f)
    e = pd.DataFrame(filas)
    e.to_csv(OUT.with_name("isomap_ejes.csv"), index=False)
    print(f"\n-> {OUT.with_name('isomap_ejes.csv')}")
    print("\nQue representa cada eje, y cuanto da la geografia sola:")
    print(e.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
