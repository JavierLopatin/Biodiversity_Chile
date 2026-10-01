#!/usr/bin/env python3
"""Significancia del margen sobre el modelo nulo geografico, con bootstrap por bloques.

Por que NO un F-test ni un t-test
---------------------------------
El F-test clasico de R2 -- F = (R2/k) / ((1-R2)/(n-k-1)) -- supone un ajuste OLS DENTRO de
muestra, residuos i.i.d. normales y k grados de libertad conocidos. Aqui no se cumple ninguno
de los tres: el R2 es FUERA de muestra (predicciones out-of-fold de un Random Forest, que no
tiene grados de libertad definidos), los residuos estan espacialmente autocorrelacionados, y la
metrica se calcula despues de centrar dentro de bandas de latitud. Aplicarlo daria p
ridiculamente pequenos para todo: con n = 3.094 el denominador trata parcelas vecinas de un
mismo bosque como observaciones independientes, y el n EFECTIVO de este diseno es el numero de
bloques espaciales, no el de parcelas.

Un t-test sobre los cinco folds es honesto pero casi sin potencia (4 grados de libertad) y
ademas prueba el objeto equivocado: la variabilidad entre folds mezcla la dificultad de cada
region con el error de estimacion.

Que se hace en cambio
---------------------
La afirmacion de la figura no es "R2 > 0" sino "este modelo le gana al modelo nulo geografico".
Eso es una comparacion PAREADA de errores de prediccion, y la herramienta estandar es un test
sobre la diferencia de errores cuadraticos (la familia de Diebold-Mariano en evaluacion de
pronosticos), con un bootstrap por BLOQUES para no suponer independencia espacial.

Para cada parcela i, con observado y predicho centrados dentro de su banda de 2 grados:

    margen = [ sum_i e2_nulo,i  -  sum_i e2_modelo,i ] / sum_i oc_i^2

que es exactamente la diferencia de R2 dentro de banda, porque el denominador es el mismo. El
bootstrap remuestrea con reemplazo los MISMOS bloques de 20 km que definen la validacion
cruzada, no las parcelas: remuestrear parcelas trataria vecinas como independientes y daria
intervalos demasiado angostos. Como todo son sumas sobre parcelas, basta precalcular la suma
por bloque y cada replica es una suma de bloques.

Dos p-valores por modelo, los dos de una cola:

    p_margen   fraccion de replicas con margen <= 0   -> "no le gana al nulo"
    p_r2       fraccion de replicas con R2 <= 0       -> "no predice nada"

El asterisco de la figura usa `q_margen`: `p_margen` corregido por Benjamini-Hochberg sobre
las 52 pruebas de la tabla (26 facetas x 2 modelos; las 22 de la figura mas los cuatro ejes de
PCoA, que van al suplemento pero son parte del mismo analisis). Sin corregir, con 52 pruebas se
esperan ~3 falsos positivos a 0,05, y esto se lee como una bateria de pruebas, no como una sola.
BH y no Bonferroni porque las facetas estan fuertemente correlacionadas entre si -- varias son
transformaciones de la misma matriz de comunidad-- y controlar la tasa de error por familia
seria absurdamente conservador.

Promedio entre semillas: el R2 reportado en todo el proyecto es la MEDIA de los R2 por semilla.
Como el denominador sum(oc^2) no depende de la semilla, promediar los errores cuadraticos por
parcela y dividir una vez da exactamente ese mismo numero, no una aproximacion.

Lo que el bootstrap NO cubre: mantiene fijas las predicciones out-of-fold, asi que mide
incertidumbre sobre la MUESTRA de parcelas, no sobre la particion de la validacion cruzada ni
sobre la semilla del bosque. Es lo habitual en este tipo de evaluacion y hay que declararlo.
Tambien se centra una sola vez sobre la muestra completa: las medias por banda salen de ~3.000
parcelas y se tratan como conocidas.

Salida: `results/tables/significancia_margen.csv`.

Uso:
    python scripts/118_significancia_margen.py [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biodiv import cv as cvmod          # noqa: E402

DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
OUT = ROOT / "results" / "tables" / "significancia_margen.csv"
SCHEME = "kfold5_block20_unified"
BANDA = 2.0
BLOQUE_KM = 20.0

NULO = "B03c_coords_raw100"
MODELOS = {"gm": "RFG4c_gm-topo_ctr-area_raw100",
           "gm_clima": "RFG5c_gm-clim-topo_ctr-area_raw100"}
FAMILIAS = ["pg-all_unified_woody", "unified-all_unified_woody"]


def contexto() -> pd.DataFrame:
    """Banda de latitud y bloque de 20 km, los dos ejes del test."""
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")
    g = pd.DataFrame({"X": p["X_m"], "Y": p["Y_m"]})
    return pd.DataFrame({"PlotObservationID": p.PlotObservationID,
                         "banda": np.floor(p.lat / BANDA) * BANDA,
                         "bloque": cvmod.add_block_key(g, BLOQUE_KM)})


def oof(run: str, fam: str) -> pd.DataFrame | None:
    f = GATE / f"{run}_{fam}" / SCHEME / "oof_predictions.csv"
    return pd.read_csv(f) if f.exists() else None


def por_bloque(d: pd.DataFrame, t: str, ctx: pd.DataFrame) -> pd.DataFrame:
    """Suma por bloque de e2 (promediada entre semillas) y de oc^2.

    Centrar y promediar aqui, no en el bootstrap: el estimando es el R2 dentro de banda sobre
    la muestra completa, y las medias por banda se tratan como conocidas.
    """
    oc_, pc_ = f"{t}_obs", f"{t}_pred"
    if oc_ not in d.columns:
        return None
    z = d[["PlotObservationID", "seed", oc_, pc_]].dropna().merge(ctx, on="PlotObservationID")
    if z.empty:
        return None
    k = ["seed", "banda"]
    oc = z[oc_] - z.groupby(k)[oc_].transform("mean")
    pc = z[pc_] - z.groupby(k)[pc_].transform("mean")
    z = z.assign(e2=(pc - oc) ** 2, oc2=oc ** 2)
    # media entre semillas por parcela; oc2 es identico en todas, asi que `mean` lo deja igual
    por_parcela = z.groupby(["PlotObservationID", "bloque"], as_index=False)[["e2", "oc2"]].mean()
    return por_parcela.groupby("bloque", as_index=False)[["e2", "oc2"]].sum()


def test(nulo: pd.DataFrame, mod: pd.DataFrame, n_boot: int, rng) -> dict:
    j = nulo.merge(mod, on="bloque", suffixes=("_n", "_m"))
    en, em, o2 = j.e2_n.to_numpy(), j.e2_m.to_numpy(), j.oc2_n.to_numpy()
    nb = len(j)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    sn, sm, so = en[idx].sum(1), em[idx].sum(1), o2[idx].sum(1)
    margen_b, r2_b = (sn - sm) / so, 1 - sm / so
    return dict(R2=float(1 - em.sum() / o2.sum()),
                R2_nulo=float(1 - en.sum() / o2.sum()),
                margen=float((en.sum() - em.sum()) / o2.sum()),
                p_margen=float((margen_b <= 0).mean()),
                p_r2=float((r2_b <= 0).mean()),
                margen_lo=float(np.percentile(margen_b, 2.5)),
                margen_hi=float(np.percentile(margen_b, 97.5)),
                n_bloques=nb)


def bh(p: np.ndarray) -> np.ndarray:
    """q-valores de Benjamini-Hochberg, con la imposicion de monotonia habitual."""
    n = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()

    ctx = contexto()
    rng = np.random.default_rng(0)
    filas = []
    for fam in FAMILIAS:
        dn = oof(NULO, fam)
        if dn is None:
            print(f"[salto] sin corrida nula para {fam}")
            continue
        targets = sorted(c[:-4] for c in dn.columns if c.endswith("_obs"))
        for nombre, run in MODELOS.items():
            dm = oof(run, fam)
            if dm is None:
                print(f"[salto] sin {run} para {fam}")
                continue
            for t in targets:
                bn, bm = por_bloque(dn, t, ctx), por_bloque(dm, t, ctx)
                if bn is None or bm is None:
                    continue
                filas.append(dict(familia=fam, faceta=t, modelo=nombre,
                                  **test(bn, bm, a.n_boot, rng)))

    f = pd.DataFrame(filas)
    # BH sobre el conjunto de pruebas que la figura muestra: las dos columnas por separado
    # serian dos familias distintas, y la figura las presenta juntas
    f["q_margen"] = bh(f.p_margen.to_numpy())
    f["q_r2"] = bh(f.p_r2.to_numpy())
    f["sig"] = pd.cut(f.q_margen, [-0.01, 0.001, 0.01, 0.05, 1.0],
                      labels=["***", "**", "*", ""]).astype(str).replace("nan", "")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"-> {OUT}   ({a.n_boot} replicas de bootstrap por bloques de "
          f"{int(BLOQUE_KM)} km)\n")
    for m in MODELOS:
        s = f[f.modelo == m].sort_values("margen", ascending=False)
        print(f"--- {m} ---")
        print(s[["faceta", "R2_nulo", "R2", "margen", "margen_lo", "margen_hi",
                 "p_margen", "q_margen", "sig", "p_r2", "q_r2"]].round(4).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
