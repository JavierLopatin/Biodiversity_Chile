#!/usr/bin/env python3
"""Validacion cruzada dejar-uno-fuera con buffer (B-LOO CV), siguiendo a Ploton et al. (2020).

La CV por bloques da el margen sobre el modelo nulo geografico a UNA escala de separacion:
20 km. B-LOO lo da como funcion continua del radio de exclusion, que contesta dos preguntas
que el paper hoy no puede contestar: a que distancia muere el aporte del sensor, y si la
conclusion depende de haber elegido 20 km.

Es ademas lo que le responde empiricamente a Wadoux et al. (2021): la curva a radio 0 ES la
validacion cruzada aleatoria, asi que una sola figura cubre todo el rango entre el protocolo
que ellos defienden y el que critican.

PORTADO DEL CODIGO ORIGINAL (`Ploton_code/R-script_Ploton-et-al_2020.R`), no de la prosa del
articulo, porque el codigo trae dos controles que la seccion de Metodos no menciona y que
cambian el resultado:

1. TAMANO DE ENTRENAMIENTO CONSTANTE. El numero de parcelas de entrenamiento se fija al que
   sobrevive en el radio MAS GRANDE, y en todos los radios menores se submuestrea hasta ese
   numero. Sin esto la curva confunde "menos datos" con "datos mas lejanos", y el error
   creceria con el radio por la razon equivocada. En el original:

       nb_training = nrow(training_tmp)                  # al radio maximo
       if(Ri == 0) training_tmp = training_tmp_j[sample(..., nb_training),]
       if(Ri > 0)  ... training_tmp[-sample(..., nb_training_dif),]

2. BLINDAJE CONTRA EXTRAPOLACION. Antes de aceptar una parcela de test se verifica que sus
   valores caigan dentro del rango de CADA predictor en el entrenamiento que queda al radio
   maximo; si no, se descarta y se sortea otra ("RETRY test pixel selection"). Esto es
   precisamente la objecion de Wadoux -- que la CV espacial fuerza extrapolacion en el
   espacio de covariables-- anticipada en el codigo, no solo en el texto.

Diferencias deliberadas con el original, y por que:

    radios      0-80 km en vez de 0-150. Su rango de autocorrelacion era ~120 km para biomasa
                en Africa central; el nuestro es mucho mas corto, y los bloques del paper son
                de 20 km. Un radio de 150 km aqui dejaria sin entrenamiento a media muestra.
    nulo        lon + lat + ELEVACION, que es el modelo nulo del paper, no solo X e Y.
    metrica     R2 agrupado sobre los puntos de test del grupo. El R2 dentro de banda del
                paper no se puede calcular sobre 100 puntos dispersos, asi que la curva vive
                en otra escala que las tablas; lo que se lee es su FORMA, no su altura.

Salida: `results/tables/bloo_cv.csv`, una fila por (faceta, radio, grupo, modelo).

Uso:
    python scripts/135_bloo_cv.py [--radios 0 5 10 20 30 40 60 80] [--test 100] [--grupos 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "results" / "tables" / "bloo_cv.csv"
RF = dict(n_estimators=300, min_samples_leaf=2, max_features=0.33, n_jobs=1, random_state=0)

#: Una faceta por familia: la de mayor margen, la composicional que gana, y una que pierde.
FACETAS = {
    "hill_q0_unified": "unified_diversity_responses_woody.parquet",
    "lcbd_count_sorensen": "lcbd_count_sorensen_unified_padded_woody.parquet",
    "isomap1_pa_unified": "unified_diversity_responses_woody.parquet",
}
NULO = ["lon", "lat", "elevation"]
#: Contra que se mide el margen. `geo` es la linea base de ubicacion (Ploton); `clima` es el
#: diseno SIN el bloque de reflectancia, o sea clima y topografia, que responde la pregunta
#: central del paper -- que agrega la reflectancia sobre lo que el clima ya sabe.
def referencia(pred: list[str], cual: str) -> list[str]:
    return NULO if cual == "geo" else [c for c in pred if not c.startswith("gm_")]


def datos(inventario: str = "ambos") -> tuple[pd.DataFrame, list[str]]:
    gm = pd.read_parquet(DERIVED / "gm_center_nc.parquet")
    cl = pd.read_parquet(DERIVED / "climate_unified.parquet")
    if cl.index.name:
        cl = cl.reset_index()
    tp = (pd.read_parquet(DERIVED / "topography_unified.parquet")
          .rename(columns={"plot_id": "PlotObservationID"}))
    # `elevation` ya viene en la tabla de topografia; de plots solo se toman las coordenadas,
    # porque un merge con las dos la duplicaria en `elevation_x`/`elevation_y`
    pl = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lon", "lat", "X_m", "Y_m", "source"]]
    d = (gm.merge(cl, on="PlotObservationID").merge(tp, on="PlotObservationID")
         .merge(pl, on="PlotObservationID"))
    for col, src in FACETAS.items():
        t = pd.read_parquet(DERIVED / src)
        d = d.merge(t[["PlotObservationID", col]], on="PlotObservationID", how="left")
    reservadas = {"PlotObservationID", "X_m", "Y_m", "source", *FACETAS}
    pred = [c for c in d.columns if c not in reservadas
            and pd.api.types.is_numeric_dtype(d[c])]
    # El inventario es un parametro porque la curva POR RED es la medicion que separa la
    # geometria del muestreo de la extension: las dos redes caen en puntos distintos del eje
    # de radio --- su vecino mas cercano esta a 0,15 y a 3,07 km --- y lo que hay que saber es
    # si recorren la MISMA curva. Ver `scripts/153_geometria_muestreo.py`.
    sel = d.source.unique() if inventario == "ambos" else [inventario]
    out = d[d.source.isin(sel)]
    assert len(out), f"el inventario {inventario!r} no dejo parcelas"
    return out.reset_index(drop=True), pred


def r2(o, p) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radios", type=float, nargs="+",
                    default=[0, 5, 10, 20, 30, 40, 60, 80])
    ap.add_argument("--test", type=int, default=100, help="parcelas de test por grupo")
    ap.add_argument("--grupos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-test", type=int, default=20,
                    help="minimo de puntos para calcular un R2 por grupo")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--referencia", choices=("geo", "clima"), default="geo")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--inventario", choices=("living_trees", "parcelas_cl", "ambos"),
                    default="ambos")
    #: Tope al tamano de entrenamiento. Sin tope, cada red usa lo que le sobrevive al radio
    #: maximo --- 830 parcelas en Parcelas-CL y 1.900 en Living Trees --- y las dos curvas
    #: dejan de ser comparables en ALTURA, que es la mitad de lo que hay que comparar. Con un
    #: tope comun las dos corren con el mismo n y solo difiere la geometria.
    ap.add_argument("--n-entren", type=int, default=0,
                    help="tope de parcelas de entrenamiento, igual en todas las redes; "
                         "0 = lo que sobreviva al radio maximo")
    ap.add_argument("--sobremuestra", type=float, default=1.6,
                    help="candidatos por parcela de test; el blindaje descarta algunos")
    a = ap.parse_args()
    radios = sorted(a.radios)
    r_max = radios[-1] * 1000.0

    d, pred = datos(a.inventario)
    print(f"{a.inventario}: {len(d)} parcelas, {len(pred)} predictores; radios {radios} km; "
          f"{a.grupos} grupos x {a.test} parcelas de test"
          + (f"; entrenamiento topado a {a.n_entren}" if a.n_entren else ""))
    XY = d[["X_m", "Y_m"]].to_numpy(float)
    rng = np.random.default_rng(a.seed)
    filas, t0 = [], time.time()

    for faceta, _ in FACETAS.items():
        y_all = d[faceta].to_numpy(float)
        disp = np.isfinite(y_all) & np.isfinite(d[pred].to_numpy(float)).all(1)
        idx_disp = np.flatnonzero(disp)
        Xf = d[pred].to_numpy(float)
        Xn = d[referencia(pred, a.referencia)].to_numpy(float)
        print(f"\n=== {faceta}: {len(idx_disp)} parcelas con faceta y predictores")

        def una_parcela(i: int, semilla: int):
            """Un punto de test: el blindaje, y luego los dos modelos en cada radio."""
            rr = np.random.default_rng(semilla)
            dist = np.linalg.norm(XY[idx_disp] - XY[i], axis=1)
            # entrenamiento al radio MAXIMO: define nb_training y el blindaje
            tr_max = idx_disp[dist > r_max]
            if len(tr_max) < 100:
                return None
            # blindaje: el punto de test cae dentro del rango de cada predictor
            lo, hi = Xf[tr_max].min(0), Xf[tr_max].max(0)
            if not np.all((Xf[i] >= lo) & (Xf[i] <= hi)):
                return None
            nb_training = len(tr_max) if not a.n_entren else min(len(tr_max), a.n_entren)
            out = []
            for r in radios:
                cand = idx_disp[dist > r * 1000.0]
                cand = cand[cand != i]
                # TAMANO CONSTANTE: submuestrear al nb_training del radio maximo
                if len(cand) > nb_training:
                    cand = rr.choice(cand, nb_training, replace=False)
                pf = RandomForestRegressor(**RF).fit(Xf[cand], y_all[cand]).predict(Xf[[i]])
                pn = RandomForestRegressor(**RF).fit(Xn[cand], y_all[cand]).predict(Xn[[i]])
                out.append((r, y_all[i], float(pf[0]), float(pn[0]),
                            float(y_all[cand].mean()), len(cand)))
            return out

        for grupo in range(a.grupos):
            acc = {r: {"obs": [], "full": [], "nulo": [], "media": [], "n_tr": []}
                   for r in radios}
            # se sortean mas candidatos de los necesarios porque el blindaje descarta algunos
            cand_i = rng.choice(idx_disp, min(int(a.test * a.sobremuestra), len(idx_disp)),
                                replace=False)
            res = Parallel(n_jobs=a.jobs, verbose=0)(
                delayed(una_parcela)(int(i), a.seed * 1000 + k)
                for k, i in enumerate(cand_i))
            elegidas = 0
            for rr_ in res:
                if rr_ is None or elegidas >= a.test:
                    continue
                elegidas += 1
                for (r, o, pf, pn, me, ntr) in rr_:
                    acc[r]["obs"].append(o); acc[r]["full"].append(pf)
                    acc[r]["nulo"].append(pn); acc[r]["media"].append(me)
                    acc[r]["n_tr"].append(ntr)
            intentos = len(cand_i)
            for r in radios:
                A = acc[r]
                if len(A["obs"]) < a.min_test:
                    continue
                rf, rn = r2(A["obs"], A["full"]), r2(A["obs"], A["nulo"])
                filas.append(dict(faceta=faceta, inventario=a.inventario,
                                  radio_km=r, grupo=grupo,
                                  n_test=len(A["obs"]), n_train=int(np.mean(A["n_tr"])),
                                  R2_diseno=rf, R2_nulo=rn, margen=rf - rn,
                                  R2_media_entren=r2(A["obs"], A["media"])))
            print(f"  grupo {grupo}: {elegidas} parcelas aceptadas de {intentos} intentos "
                  f"({time.time()-t0:.0f}s)")
            pd.DataFrame(filas).to_csv(a.salida or OUT, index=False)   # guardado incremental

    f = pd.DataFrame(filas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(a.salida or OUT, index=False)
    print(f"\n-> {OUT}   ({time.time()-t0:.0f}s)")
    print(f.groupby(["faceta", "radio_km"])[["R2_nulo", "R2_diseno", "margen"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
