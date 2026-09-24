#!/usr/bin/env python3
"""Geomediano espectral de las parcelas de Living Trees, en el formato de `cube_predictors`.

Por que existe. `data/derived/cube_predictors.parquet` solo cubre las 1.082 parcelas de
Parcelas-CL, porque salio de los cubos `.nc` que `scripts/02` extrajo para ellas. Sobre el
pool unificado eso deja el bloque `gm` en NaN para las 2.020 parcelas de Living Trees, y el
`Preprocessor` lo rellena con la mediana del fold: para 3 de cada 4 filas el bloque espectral
seria una constante, que ademas funciona como etiqueta de fuente. Cualquier contraste entre
espectro y fenologia medido asi confunde "espectro contra curva" con "sin datos contra curva".

Por que NO hace falta volver al Data Cube. `scripts/35_extract_living_trees.py` ya extrajo las
series por la misma via: mismo modulo `biodiv.cube`, mismos flags de `qa_pixel` y el mismo
`clear_mask`, misma ventana causal de tres años aplicada en la extraccion, y la misma escala
de reflectancia. La paridad es por construccion, no por replicacion: este script llama a las
mismas funciones de `src/biodiv/geomedian.py` que usa `scripts/18`.

Lo unico que cambia es el origen del arreglo (n_obs, 6): alli se lee de un cubo, aqui de las
seis tablas de banda ya extraidas.

Uso:
    python scripts/84_living_trees_geomedian.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import geomedian as gmod  # noqa: E402

LT_DIR = ROOT / "data" / "derived" / "living_trees"
#: `_center` y no `_median`. Living Trees guarda las dos agregaciones que dejo
#: `scripts/35` -- `_center` y `_mean5x5` -- y Parcelas-CL expone siete en
#: `cube_predictors` (`center`, `mean`, `median`, `trimmed`, `sd`, `cv`, `npx`). De esas,
#: `center` es la unica que significa lo mismo en ambas fuentes: un solo pixel, el que
#: contiene la coordenada.
#:
#: `median` es la mediana sobre los 25 pixeles y Living Trees no la tiene, asi que usarla
#: dejaria 2.020 de 3.102 filas imputadas. `mean5x5` contra `mean` seria la otra pareja
#: comparable y queda pendiente evaluarla: promedia sobre la misma ventana pero es una
#: estadistica menos robusta que la mediana ante un pixel que falla. Mezclar `mean5x5` en
#: una fuente con `median` en la otra seria un metodo distinto por fuente, que es el
#: artefacto que este bloque existe para evitar.
PX = "center"
SUFFIX = "_center"


def load_bands() -> pd.DataFrame:
    """Las seis bandas alineadas por (site_id, time, sensor), sin filas incompletas."""
    parts = []
    for b in gmod.BANDS:
        f = LT_DIR / f"series_band_{b}_{PX}.parquet"
        if not f.exists():
            raise SystemExit(f"falta {f}")
        d = pd.read_parquet(f).set_index(["site_id", "time", "sensor"])[f"band_{b}"]
        parts.append(d)
    M = pd.concat(parts, axis=1)
    M.columns = gmod.BANDS
    return M[M.notna().all(axis=1)]


def site_to_plot(site_id: str) -> str:
    """LT0000 -> LT_0, que es el esquema que da `scripts/50`."""
    return f"LT_{int(str(site_id)[2:])}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/cube_predictors_living_trees.parquet")
    args = ap.parse_args()

    M = load_bands()
    rows = []
    for site, g in M.groupby(level="site_id"):
        X = g.to_numpy(dtype=float)
        row = {"PlotObservationID": site_to_plot(site), f"gm_count{SUFFIX}": float(len(X))}
        if len(X) >= gmod.MIN_OBS:
            gm = gmod.geometric_median(X)
            for b, v in zip(gmod.BANDS, gm):
                row[f"gm_band_{b}{SUFFIX}"] = float(v)
            for name, v in gmod.indices_from_bands(dict(zip(gmod.BANDS, gm))).items():
                row[f"gm_{name}{SUFFIX}"] = v
            e, s, bc = gmod.mads(X, gm)
            row[f"gm_emad{SUFFIX}"], row[f"gm_smad{SUFFIX}"], row[f"gm_bcmad{SUFFIX}"] = e, s, bc
        rows.append(row)

    out = pd.DataFrame(rows)
    f = ROOT / args.out
    out.to_parquet(f, index=False)
    print(f"-> {f}  ({len(out)} parcelas, {out.shape[1] - 1} columnas)")
    n_ok = out.filter(like="gm_band_").notna().all(axis=1).sum()
    print(f"   con geomediano: {int(n_ok)}  (el resto no alcanza {gmod.MIN_OBS} observaciones)")
    print(f"\n   escala de las bandas, para comparar con Parcelas-CL:")
    print(out.filter(like="gm_band_").describe().loc[["min", "50%", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
