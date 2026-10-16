#!/usr/bin/env python3
"""Que fija cuanto infla la validacion aleatoria: la geometria del muestreo, no la extension.

El paper reporta que los folds aleatorios suben el R2 en 171 de 171 comparaciones, por una
mediana de +0,158 (`optimism_gap.csv`). Pero la corrida de las cuatro extensiones
(`scripts/136`, `escalas_completo_*.csv`) mide +0,022 a extension NACIONAL, sobre la misma
rejilla de 20 km. Un factor de siete entre dos numeros que el texto llama los dos
"nacionales".

Las dos medidas no son del mismo pool: `optimism_gap` sale de la fase de tamizaje, que es
Parcelas-CL sola (1.082 parcelas, -37,9 a -30,2 grados), y la corrida de escalas es Living
Trees sola (2.020 parcelas, 24,3 grados). Este script muestra que lo que separa a las dos no
es la extension --- la de Parcelas-CL es TRES VECES MAS CHICA y su optimismo es SEIS VECES
MAYOR, o sea al revés de lo que predice un argumento de extension --- sino a que distancia
deja cada protocolo el dato de entrenamiento mas cercano a cada parcela evaluada.

Eso es lo unico que un protocolo de validacion controla, y es el diagnostico de Mila et al.
2022: un R2 fuera de fold mide interpolacion en la medida en que esa distancia caiga dentro
del rango de autocorrelacion residual. El rango residual tras descontar ubicacion esta medido
en `autocorrelacion_rango.csv` y va de 2 a 25 km, con mediana cerca de 10 km.

Tres productos:

1. **Geometria de cada red.** Vecino etiquetado mas cercano, bloques de 20 km ocupados y
   parcelas por bloque, por inventario.
2. **Lo que hace cada protocolo.** Distancia de cada parcela de test a su parcela de
   entrenamiento mas cercana, bajo folds aleatorios y bajo bloques de 20 km. Los bloques dan
   casi la misma geometria en las dos redes; el aleatorio no.
3. **La descomposicion del gap de `optimism_gap`.** `kfold5_window` es un split aleatorio
   SALVO que mantiene juntas las parcelas cuyas ventanas de extraccion de 5x5 pixeles se
   solapan. La resta aisla la pseudorreplica de pixel del optimismo espacial real.

Salida: `results/tables/geometria_muestreo.csv`, `results/tables/optimismo_descompuesto.csv`.

Uso:
    python scripts/153_geometria_muestreo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "src"))

BLOQUE_M = 20_000.0
SEMILLA = 42
#: los dos inventarios del pool unificado, mas el pool entero
REDES = {"parcelas_cl": "Parcelas-CL (pool de tamizaje, optimism_gap)",
         "living_trees": "Living Trees (corrida de escalas)",
         "ambos": "pool unificado"}


def clave_bloque(xy: np.ndarray) -> pd.Series:
    return pd.Series(np.floor(xy[:, 0] / BLOQUE_M).astype(int).astype(str) + "_"
                     + np.floor(xy[:, 1] / BLOQUE_M).astype(int).astype(str))


def vecino_mas_cercano(xy: np.ndarray) -> np.ndarray:
    """Distancia en km de cada parcela a OTRA parcela, no a si misma (de ahi el k=2)."""
    return cKDTree(xy).query(xy, k=2)[0][:, 1] / 1000.0


def dist_a_entrenamiento(xy: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Distancia en km de cada parcela de test a su parcela de ENTRENAMIENTO mas cercana.

    Es la cantidad que un protocolo de validacion fija, y la unica. Se acumula sobre los
    cinco folds, asi que cada parcela aparece exactamente una vez.
    """
    out = []
    for f in np.unique(fold):
        te, tr = xy[fold == f], xy[fold != f]
        if len(te) and len(tr):
            out.append(cKDTree(tr).query(te, k=1)[0] / 1000.0)
    return np.concatenate(out)


def folds(xy: np.ndarray, protocolo: str, rng: np.random.Generator) -> np.ndarray:
    if protocolo == "aleatorio":
        return rng.permutation(len(xy)) % 5
    cl = clave_bloque(xy)
    bl = cl.unique()
    return cl.map(dict(zip(bl, rng.permutation(len(bl)) % 5))).to_numpy()


def geometria() -> pd.DataFrame:
    pu = pd.read_parquet(DERIVED / "plots_unified.parquet")
    filas = []
    for red, rotulo in REDES.items():
        d = pu if red == "ambos" else pu[pu.source == red]
        xy = d[["X_m", "Y_m"]].to_numpy(float)
        vmc = vecino_mas_cercano(xy)
        nb = clave_bloque(xy).nunique()
        fila = dict(red=red, rotulo=rotulo, n=len(xy), bloques_20km=nb,
                    parcelas_por_bloque=len(xy) / nb,
                    grados_latitud=float(d.lat.max() - d.lat.min()),
                    vmc_mediana_km=float(np.median(vmc)),
                    vmc_p25_km=float(np.percentile(vmc, 25)),
                    vmc_p75_km=float(np.percentile(vmc, 75)),
                    frac_vmc_bajo_1km=float((vmc < 1).mean()))
        # Una semilla fija por red: lo que se compara entre protocolos es la geometria que
        # cada uno impone, no el sorteo.
        for protocolo in ("aleatorio", "bloques"):
            rng = np.random.default_rng(SEMILLA)
            dte = dist_a_entrenamiento(xy, folds(xy, protocolo, rng))
            fila[f"dist_train_{protocolo}_mediana_km"] = float(np.median(dte))
            fila[f"dist_train_{protocolo}_p90_km"] = float(np.percentile(dte, 90))
        fila["razon_bloques_sobre_aleatorio"] = (fila["dist_train_bloques_mediana_km"]
                                                 / fila["dist_train_aleatorio_mediana_km"])
        filas.append(fila)
    return pd.DataFrame(filas)


def descompone_gap() -> pd.DataFrame:
    """Separa la pseudorreplica de pixel del optimismo espacial, dentro de `optimism_gap`.

    `kfold5_window` agrupa por componente conexa del grafo "estas dos parcelas comparten
    pixeles de la ventana de extraccion de 5x5". Medido en `biodiv.cv.window_components`, el
    split aleatorio parte 126 de 135 componentes y deja 557 parcelas (51,5 %) con pixeles en
    los dos lados. Fuera de esos componentes el esquema es tan libre como el aleatorio --- los
    componentes miden 150 m --- asi que la resta aisla el leak de pixel.
    """
    d = (pd.read_csv(TAB / "optimism_gap.csv")
         .dropna(subset=["kfold5_random", "kfold5_block20", "kfold5_window"]))
    d["leak_pixel"] = d.kfold5_random - d.kfold5_window
    d["espacial_20km"] = d.kfold5_window - d.kfold5_block20
    d["total"] = d.kfold5_random - d.kfold5_block20
    return d


def main() -> None:
    g = geometria()
    g.to_csv(TAB / "geometria_muestreo.csv", index=False)
    print("GEOMETRIA DE CADA RED Y DE CADA PROTOCOLO\n")
    col = ["red", "n", "bloques_20km", "parcelas_por_bloque", "grados_latitud",
           "vmc_mediana_km", "frac_vmc_bajo_1km", "dist_train_aleatorio_mediana_km",
           "dist_train_bloques_mediana_km", "razon_bloques_sobre_aleatorio"]
    print(g[col].round(3).to_string(index=False))
    print(f"\n-> {TAB}/geometria_muestreo.csv")

    d = descompone_gap()
    d.to_csv(TAB / "optimismo_descompuesto.csv", index=False)
    print("\n\nEL GAP DE optimism_gap, DESCOMPUESTO "
          f"({len(d)} comparaciones modelo x variable)\n")
    print(d[["leak_pixel", "espacial_20km", "total"]].median().round(3).to_string())
    print(f"\n  pseudorreplica de pixel como fraccion del gap (mediana de la razon): "
          f"{(d.leak_pixel / d.total).median():.3f}")
    print(f"  comparaciones con leak de pixel > 0: {(d.leak_pixel > 0).sum()} de {len(d)}")
    print(f"\n-> {TAB}/optimismo_descompuesto.csv")

    # El rango residual es la vara contra la que se leen las distancias de arriba: una
    # distancia muy por dentro del rango significa que el R2 mide interpolacion.
    r = pd.read_csv(TAB / "autocorrelacion_rango.csv")
    ru = r[r.version == "residual_ubicacion"].rango_emp_km.dropna()
    print(f"\n\nVARA DE COMPARACION: rango empirico tras descontar ubicacion, sobre "
          f"{len(ru)} variables")
    print(f"  p25 {ru.quantile(0.25):.1f} km   mediana {ru.median():.1f} km   "
          f"p75 {ru.quantile(0.75):.1f} km")


if __name__ == "__main__":
    main()
