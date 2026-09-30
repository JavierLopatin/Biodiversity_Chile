#!/usr/bin/env python3
"""A donde se va la capacidad predictiva, y por que ni regularizar ni adaptar la recuperan.

Tres preguntas que un revisor hace, contestadas con medidas y no con argumentos. Todas sobre
el pool lenoso, `kfold5_block20_unified`, con geomediano + clima + topografia.

1. PRESUPUESTO DE R2. El ajuste en muestra de un Random Forest es casi decorativo -- arboles
   completos memorizan-- asi que la comparacion que informa no es train contra test sino
   OOB contra test: OOB generaliza a parcelas NUEVAS de la MISMA zona, test a una zona nueva.

       LCBD              train 0,917   OOB 0,540   test 0,475   en banda ~0,21
       riqueza cruda     train 0,907   OOB 0,537   test 0,403   en banda ~0,39

   De los 0,92 que el bosque captura, 0,38 no generaliza ni a parcelas vecinas, y de lo que
   queda una buena parte es gradiente geografico. La perdida por CAMBIAR DE ZONA es solo
   0,065 y 0,134 -- eso acota el techo de cualquier adaptacion de dominio.

2. REGULARIZAR NO AYUDA. Subir `min_samples_leaf` de 2 a 50 cierra la brecha train-OOB de
   0,38 a 0,065 y BAJA las tres metricas fuera de muestra (test 0,473 -> 0,439, en banda
   0,208 -> 0,159). La configuracion por defecto ya es la mejor. El 0,917 en muestra no
   estaba costando nada: el bagging con submuestreo de variables ya autorregula.

3. CORAL SUPERFICIAL EMPEORA. Sun, Feng & Saenko (2016, AAAI) blanquea las features de
   entrenamiento y las recolorea con la covarianza del bloque de test. Su implementacion de
   referencia usa `cov + eye()`, o sea lambda = 1, que es el valor donde medimos -0,073.
   Consistente en todo el barrido (-0,054 a -0,082), y el dano DISMINUYE al subir lambda,
   es decir al acercarse la transformacion a la identidad: la firma de algo que no aporta.

   Y no es por falta de desplazamiento: la distancia CORAL entre folds espaciales es 7-11x
   la de particiones aleatorias del mismo tamano. Desplazamiento hay. Lo que falla es el
   supuesto: CORAL asume P(y|x) estable bajo la transformacion, y aqui la diferencia de
   covarianza entre regiones es SENAL -- la relacion entre bandas y clima cambia entre el
   mediterraneo y el bosque templado porque cambia la ecologia. Mover X hacia la covarianza
   del destino dejando y fijo corrompe el emparejamiento en vez de corregirlo.

Las tres convergen en lo mismo: el limite no es el modelo ni el protocolo, son los
predictores. Con geomediano, clima y topografia a 30 m, la senal de composicion que no es
geografia es ~0,21, y ningun cambio de modelo la mueve.

Salida: `results/tables/diagnostico_generalizacion.csv`.

Uso:
    python scripts/114_diagnostico_generalizacion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "results" / "tables" / "diagnostico_generalizacion.csv"

RF = dict(n_estimators=300, min_samples_leaf=2, max_features=0.33, bootstrap=True, n_jobs=-1)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)      #: barrido de CORAL; 1.0 es el default del paper
LEAVES = (2, 5, 10, 20, 50)                  #: barrido de regularizacion
BANDA = 2.0
TARGETS = {
    "lcbd_count_sorensen": "lcbd_count_sorensen_unified_padded_woody.parquet",
    "hill_q0_unified": "unified_diversity_responses_woody.parquet",
}


def diseno() -> tuple[np.ndarray, pd.Index, np.ndarray]:
    """gm + clima + topografia. No pasa por `features.build_design` a proposito: esa carga
    la tabla de curvas, que no esta en todas las maquinas, y este diagnostico no la usa."""
    gm = pd.read_parquet(DERIVED / "gm_center_nc.parquet")
    cl = pd.read_parquet(DERIVED / "climate_unified.parquet")
    if cl.index.name:
        cl = cl.reset_index()
    tp = (pd.read_parquet(DERIVED / "topography_unified.parquet")
          .rename(columns={"plot_id": "PlotObservationID"}))
    pl = pd.read_parquet(DERIVED / "plots_unified.parquet")[["PlotObservationID", "lat"]]
    d = (gm.merge(cl, on="PlotObservationID").merge(tp, on="PlotObservationID")
         .merge(pl, on="PlotObservationID"))
    cols = [c for c in d.columns if c not in ("PlotObservationID", "lat")
            and pd.api.types.is_numeric_dtype(d[c])]
    X = d[cols].to_numpy(float)
    ok = np.isfinite(X).all(1)
    return X[ok], pd.Index(d.PlotObservationID)[ok], d.lat.to_numpy()[ok]


def r2(o, p) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def r2_en_banda(o, p, b) -> float:
    d = pd.DataFrame({"o": o, "p": p, "b": b})
    oc = (d.o - d.groupby("b").o.transform("mean")).to_numpy()
    pc = (d.p - d.groupby("b").p.transform("mean")).to_numpy()
    return float(1 - ((pc - oc) ** 2).sum() / (oc ** 2).sum())


def coral_map(S: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """Blanquea S y lo recolorea con la covarianza de T (Sun, Feng & Saenko 2016, AAAI).

    `lam` es el ridge que su codigo aplica como `cov + eye()`, o sea lam = 1 por defecto.
    """
    d = S.shape[1]
    W = np.real(fractional_matrix_power(np.cov(S, rowvar=False) + lam * np.eye(d), -0.5)
                @ fractional_matrix_power(np.cov(T, rowvar=False) + lam * np.eye(d), 0.5))
    return (S - S.mean(0)) @ W + T.mean(0)


def coral_dist(A: np.ndarray, B: np.ndarray) -> float:
    """La propia perdida CORAL, para ver si hay algo que alinear antes de intentar alinearlo."""
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    d = A.shape[1]
    return float(((np.cov(A, rowvar=False) - np.cov(B, rowvar=False)) ** 2).sum() / (4 * d * d))


def main() -> None:
    X, ids, lat = diseno()
    banda = np.floor(lat / BANDA) * BANDA
    folds = pd.read_parquet(DERIVED / "cv_folds_unified_block20.parquet")
    print(f"diseno: {X.shape[0]} parcelas x {X.shape[1]} predictores (gm + clima + topo)")

    # --- hay desplazamiento que alinear? CORAL entre folds contra el nulo de particiones al azar
    rng = np.random.default_rng(0)
    esp, azar = [], []
    for _, g in folds.groupby("held_out"):
        te = ids.isin(g.loc[g.split == "test", "PlotObservationID"])
        tr = ids.isin(g.loc[g.split == "train", "PlotObservationID"])
        if te.sum() < 50:
            continue
        esp.append(coral_dist(X[tr], X[te]))
        p = rng.permutation(len(X))
        k = int(te.sum())
        azar.append(coral_dist(X[p[k:]], X[p[:k]]))
    print(f"\ndistancia CORAL entre folds espaciales: {np.mean(esp):.2e}")
    print(f"                  nulo (particiones al azar): {np.mean(azar):.2e}"
          f"   -> {np.mean(esp) / np.mean(azar):.1f}x")

    filas = []
    for t, src in TARGETS.items():
        y = (pd.read_parquet(DERIVED / src).set_index("PlotObservationID")[t]
             .reindex(ids).to_numpy(float))
        m = np.isfinite(y)
        part = [(ids.isin(g.loc[g.split == "train", "PlotObservationID"]) & m,
                 ids.isin(g.loc[g.split == "test", "PlotObservationID"]) & m)
                for _, g in folds.groupby("held_out")]
        part = [(a, b) for a, b in part if a.sum() >= 50 and b.sum() >= 30]

        # 1 y 2: presupuesto de R2, y si regularizar lo mejora
        for leaf in LEAVES:
            tr_, ob_, te_, bd_ = [], [], [], []
            for a, b in part:
                rf = RandomForestRegressor(random_state=0, oob_score=True,
                                           **(RF | {"min_samples_leaf": leaf})).fit(X[a], y[a])
                p = rf.predict(X[b])
                tr_.append(r2(y[a], rf.predict(X[a])))
                ob_.append(r2(y[a], rf.oob_prediction_))
                te_.append(r2(y[b], p))
                bd_.append(r2_en_banda(y[b], p, banda[b]))
            filas.append(dict(target=t, experimento="regularizacion", ajuste=f"leaf={leaf}",
                              train=np.mean(tr_), oob=np.mean(ob_), test=np.mean(te_),
                              en_banda=np.mean(bd_)))

        # 3: CORAL superficial, barrido de lambda, contra el mismo camino sin CORAL
        for lam in (None, *LAMBDAS):
            te_, bd_ = [], []
            for a, b in part:
                mu, sd = X[a].mean(0), X[a].std(0) + 1e-12
                S, T = (X[a] - mu) / sd, (X[b] - mu) / sd
                Su = S if lam is None else coral_map(S, T, lam)
                p = RandomForestRegressor(random_state=0, **RF).fit(Su, y[a]).predict(T)
                te_.append(r2(y[b], p))
                bd_.append(r2_en_banda(y[b], p, banda[b]))
            filas.append(dict(target=t, experimento="coral",
                              ajuste="sin CORAL" if lam is None else f"lambda={lam:g}",
                              train=np.nan, oob=np.nan, test=np.mean(te_),
                              en_banda=np.mean(bd_)))

    f = pd.DataFrame(filas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    for exp in ("regularizacion", "coral"):
        print(f"\n--- {exp} ---")
        print(f[f.experimento == exp].drop(columns="experimento").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
