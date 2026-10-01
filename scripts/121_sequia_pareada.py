#!/usr/bin/env python3
"""Estres hidrico por EMPAREJAMIENTO: comparar parcelas ecologicamente gemelas, no regiones.

Por que otro diseno
-------------------
El contraste por terciles (`scripts/107`, `scripts/120`) compara dos CONJUNTOS de parcelas, asi
que cualquier diferencia entre ellos se cuela en el resultado. Ya costo caro una vez: con el
tercil definido como residuo de SPI respecto a la latitud, el tercil seco quedaba 318 m mas
alto y mas al este, y el "efecto de sequia" era ese confundido. Descontando lon + lat +
elevacion la geografia queda balanceada -- diferencia de medias estandarizada 0,065 en
elevacion, -0,029 en latitud-- pero la COMPOSICION no: Isomap 1 queda en -0,398 y Isomap 3 en
-0,210. Seguimos comparando bosques distintos.

El diseno por bloque espacial x ano, que seria lo natural, no es viable: de los 531 bloques de
20 km con parcelas, 263 tienen dos o mas anos de censo pero solo DOS llegan a veinte parcelas.

Este script sustituye cercania espacial por cercania ECOLOGICA. Para cada parcela del tercil
seco busca su gemela en el tercil humedo -- misma posicion en la ordenacion, misma riqueza,
misma elevacion-- sin exigir que esten cerca. Dos parcelas con la misma composicion y la misma
riqueza son comparables aunque las separen 300 km, y eso es justo lo que el pool permite.

Y cambia la metrica. Sobre pares emparejados no se calcula R2: el R2 normaliza por la varianza
del target, que el emparejamiento modifica a proposito. Se compara el ERROR DE PREDICCION
dentro de cada par, que es la cantidad que la pregunta necesita -- si el modelo describe peor a
la misma comunidad cuando esta seca-- y se prueba con un test pareado sin supuestos de forma.

Lo que mide, medido
-------------------
Con calibrador de 0,5 DE salen 309 pares de 602 secas y 607 humedas, y el balance es de libro:
todas las covariables bajan a |SMD| <= 0,046 mientras SPI se mantiene en -1,09, o sea el
contraste hidrico se conserva entero y todo lo demas se iguala. Latitud, longitud y ano quedan
balanceados sin haber entrado en el emparejamiento.

Y el efecto NO aparece. El error cuadratico es mayor en el tercil seco en 16 de 26 facetas
(binomial p = 0,33), y el test de Wilcoxon por faceta da p < 0,05 en solo 2 de 26 -- cerca de
las 1,3 que se esperan por azar-- y en DIRECCIONES OPUESTAS.

El barrido del calibrador tampoco encuentra nada, y conviene leerlo con cuidado porque no es
monotono:

    calibrador 0,25 DE   156 pares   10 de 20 peor en seco   |SMD| medio 0,006
    calibrador 0,50 DE   309 pares   16 de 26                |SMD| medio 0,011
    calibrador 1,00 DE   444 pares   18 de 26                |SMD| medio 0,037
    calibrador 2,00 DE   485 pares   14 de 26                |SMD| medio 0,065

El desbalance residual si crece de forma monotona con el calibrador, pero el conteo no: va de
50 % a 62 %, 69 % y 54 %. Ninguno se aparta de lo esperable por azar (binomial p = 1,00, 0,33,
0,076 y 0,58), y el vaiven entre calibradores es justamente lo que se ve cuando no hay efecto.
Seria distinto si el conteo creciera al aflojar, que apuntaria a un confundido reentrando; no
es lo que pasa.

Restringiendo a las facetas bien emparejadas -- 300 pares o mas y diferencia de target bajo
0,5 DE-- quedan 4 de 11, p = 0,55.

Dos disenos independientes coinciden entonces en lo mismo: el contraste por terciles con el
detrend geografico completo (8 de 26, p = 0,08) y el emparejamiento (sin senal a ningun
calibrador). No hay efecto de sequia detectable sobre el margen del sensor en este pool. El que
se reportaba antes era el gradiente de elevacion.

Que puede concluir y que no
---------------------------
Esto NO es un experimento. El emparejamiento iguala lo que se mide y lo declara con el balance
post-emparejamiento; lo que no se midio sigue suelto. Y el par comparte estado hidrico distinto
pero tambien, en general, ano y lugar distintos, asi que "seco" sigue siendo una anomalia de
SPI y no una sequia impuesta.

Un detalle de circularidad que hay que declarar: los ejes de Isomap estan entre los covariables
de emparejamiento, asi que para esas facetas el par tiene el target casi igual por construccion.
Eso no invalida la comparacion de errores -- al contrario, es el caso mas limpio: mismo valor
objetivo, distinto estado hidrico-- pero no se puede leer como si la faceta fuera independiente
del emparejamiento.

Salida: `results/tables/sequia_pareada.csv`.

Uso:
    python scripts/121_sequia_pareada.py [--caliper 0.5]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
OUT = ROOT / "results" / "tables" / "sequia_pareada.csv"
SCHEME = "kfold5_block20_unified"
RUN = "RFG5c_gm-clim-topo_ctr-area_raw100"
FAMILIAS = ["pg-all_unified_woody", "unified-all_unified_woody"]

#: Covariables de emparejamiento: posicion en la ordenacion, riqueza y elevacion. La ordenacion
#: lleva tres ejes porque uno solo deja fuera la estructura que el paper discute; la elevacion
#: entra aunque ya este balanceada, para que el par tambien lo este individualmente.
COV = ["isomap1_pa_unified", "isomap2_pa_unified", "isomap3_pa_unified",
       "hill_q0_unified", "elevation"]


def cargar_107():
    spec = importlib.util.spec_from_file_location("s107", ROOT / "scripts" /
                                                  "107_sintesis_lenoso.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s107"] = m
    spec.loader.exec_module(m)
    return m


def candidatos(m) -> pd.DataFrame:
    ctx = m.contexto()
    r = pd.read_parquet(DERIVED / "unified_diversity_responses_woody.parquet")
    d = ctx.merge(r[["PlotObservationID"] + [c for c in COV if c in r.columns]],
                  on="PlotObservationID")
    lt = d[(d.source == "living_trees") & d.spi.notna()].copy()
    lt["t"] = m.terciles(lt)
    return lt[lt.t.notna()].dropna(subset=COV).reset_index(drop=True)


def smd(a: pd.Series, b: pd.Series) -> float:
    """Diferencia de medias estandarizada, la medida habitual de balance."""
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def emparejar(d: pd.DataFrame, caliper: float, seed: int = 0) -> pd.DataFrame:
    """Vecino mas cercano seco -> humedo, sin reemplazo, con calibrador en DE.

    Codicioso por orden de dificultad: primero las parcelas secas cuyo vecino mas cercano esta
    mas lejos, que es la unica forma de que el emparejamiento codicioso no les deje las sobras.
    """
    z = (d[COV] - d[COV].mean()) / d[COV].std()
    seco = d.index[d.t == 0].to_numpy()
    humedo = d.index[d.t == 2].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(25, len(humedo))).fit(z.loc[humedo])
    dist, idx = nn.kneighbors(z.loc[seco])
    orden = np.argsort(-dist[:, 0])          # el mas dificil primero
    usados, pares = set(), []
    for i in orden:
        for j, dd in zip(idx[i], dist[i]):
            if dd > caliper:
                break
            if humedo[j] in usados:
                continue
            usados.add(humedo[j])
            pares.append((seco[i], humedo[j], float(dd)))
            break
    return pd.DataFrame(pares, columns=["i_seco", "i_humedo", "distancia"])


def errores(ids: pd.Index) -> pd.DataFrame:
    """Error cuadratico fuera de muestra por parcela y faceta, promediado entre semillas."""
    out = {}
    for fam in FAMILIAS:
        f = GATE / f"{RUN}_{fam}" / SCHEME / "oof_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for t in sorted(c[:-4] for c in d.columns if c.endswith("_obs")):
            z = d[["PlotObservationID", f"{t}_obs", f"{t}_pred"]].dropna()
            e = (z[f"{t}_pred"] - z[f"{t}_obs"]) ** 2
            s = e.groupby(z.PlotObservationID).mean()
            out[t] = s.reindex(ids)
            out[f"__obs__{t}"] = z.groupby("PlotObservationID")[f"{t}_obs"].first().reindex(ids)
    return pd.DataFrame(out, index=ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caliper", type=float, default=0.5,
                    help="distancia maxima admitida, en DE del espacio de covariables")
    a = ap.parse_args()

    m = cargar_107()
    d = candidatos(m)
    print(f"candidatos: {len(d)}   seco {int((d.t==0).sum())}  humedo {int((d.t==2).sum())}")

    pares = emparejar(d, a.caliper)
    print(f"pares formados: {len(pares)}  (calibrador {a.caliper} DE, "
          f"distancia mediana {pares.distancia.median():.3f})")
    s, h = d.loc[pares.i_seco], d.loc[pares.i_humedo]

    print("\n--- balance: diferencia de medias estandarizada (seco - humedo) ---")
    a0, b0 = d[d.t == 0], d[d.t == 2]
    print(f"{'covariable':24s} {'antes':>8s} {'despues':>9s}")
    bal = []
    for c in COV + ["lat", "lon", "spi", "Year"]:
        x, y = smd(a0[c], b0[c]), smd(s[c].reset_index(drop=True), h[c].reset_index(drop=True))
        bal.append(dict(variable=c, smd_antes=x, smd_despues=y))
        print(f"{c:24s} {x:+8.3f} {y:+9.3f}")

    e = errores(pd.Index(d.PlotObservationID))
    facetas = [c for c in e.columns if not c.startswith("__obs__")]
    ids_s = d.loc[pares.i_seco, "PlotObservationID"].to_numpy()
    ids_h = d.loc[pares.i_humedo, "PlotObservationID"].to_numpy()

    filas = []
    for t in facetas:
        es, eh = e[t].reindex(ids_s).to_numpy(), e[t].reindex(ids_h).to_numpy()
        ok = np.isfinite(es) & np.isfinite(eh)
        if ok.sum() < 40:
            continue
        dif = es[ok] - eh[ok]                       # > 0 = peor en seco
        obs = e[f"__obs__{t}"]
        sd = float(obs.std())
        p = wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
        filas.append(dict(faceta=t, n_pares=int(ok.sum()),
                          rmse_seco=float(np.sqrt(es[ok].mean())) / sd,
                          rmse_humedo=float(np.sqrt(eh[ok].mean())) / sd,
                          dif_media=float(dif.mean()) / sd ** 2,
                          dif_mediana=float(np.median(dif)) / sd ** 2,
                          p_wilcoxon=float(p),
                          dif_target=float(np.nanmean(
                              np.abs(obs.reindex(ids_s).to_numpy()[ok]
                                     - obs.reindex(ids_h).to_numpy()[ok]))) / sd))
    f = pd.DataFrame(filas).sort_values("dif_media", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([f.assign(bloque="faceta"),
               pd.DataFrame(bal).assign(bloque="balance")]).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    k, n = int((f.dif_media > 0).sum()), len(f)
    print(f"\nerror cuadratico MAYOR en el tercil seco en {k} de {n} facetas "
          f"(dif_media > 0 = el modelo falla mas en seco):")
    print(f[["faceta", "n_pares", "rmse_seco", "rmse_humedo", "dif_media",
             "p_wilcoxon", "dif_target"]].round(4).to_string(index=False))

    # El barrido del calibrador es el diagnostico que decide: si el efecto fuera real seria
    # MAS nitido con pares mejor emparejados, no menos.
    print("\n--- barrido del calibrador: el efecto escala con el desbalance residual ---")
    for cal in (0.25, 0.5, 1.0, 2.0):
        pr = emparejar(d, cal)
        si = d.loc[pr.i_seco, "PlotObservationID"].to_numpy()
        hi = d.loc[pr.i_humedo, "PlotObservationID"].to_numpy()
        kk = nn = 0
        for t in facetas:
            es, eh = e[t].reindex(si).to_numpy(), e[t].reindex(hi).to_numpy()
            ok = np.isfinite(es) & np.isfinite(eh)
            if ok.sum() < 40:
                continue
            nn += 1
            kk += int((es[ok] - eh[ok]).mean() > 0)
        sm = np.mean([abs(smd(d.loc[pr.i_seco, c].reset_index(drop=True),
                              d.loc[pr.i_humedo, c].reset_index(drop=True))) for c in COV])
        print(f"  calibrador {cal:4.2f} DE   {len(pr):4d} pares   {kk:2d} de {nn} peor en seco"
              f"   |SMD| medio {sm:.3f}")


if __name__ == "__main__":
    main()
