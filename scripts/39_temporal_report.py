#!/usr/bin/env python3
"""Qué pasa con el R² cuando se retiene el tiempo, y cuando se retienen sitio y tiempo.

Cuatro esquemas, ordenados por la distancia a la que evalúan (`sampling_pattern.json`):

    kfold5_window    0,5 km   interpolar en un territorio muestreado -- el titular hasta ahora
    kfold5_block20  12,1 km   sitio nuevo, cualquier ano
    kfold_time      10,1 km   ano nuevo, cualquier sitio
    kfold_loc_time  14,1 km   sitio nuevo Y ano nuevo -- lo mas parecido a un mapa proyectado

**La comparacion contra `kfold5_window` no es "cuanto empeora el modelo"**, sino cuanto del
R² titular venia de predecir cerca. Los cuatro puntuan las mismas 1.082 parcelas con el mismo
modelo; lo unico que cambia es de donde puede aprender.

Y `kfold_time` esta ahi para poder interpretar `kfold_loc_time`: si el LLTO cae y el LTO no,
la caida es espacial; si caen los dos, hay un componente temporal de verdad. Sin esa
descomposicion, un LLTO bajo no dice cual de las dos cosas fallo.

Uso:
    python scripts/39_temporal_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import metrics as mx                 # noqa: E402
from biodiv import targets as tg                 # noqa: E402

SCHEMES = ["kfold5_window", "kfold5_block20", "kfold_time", "kfold_loc_time",
           "time_within_owner"]

#: Los cabezas de familia. Se fijan por nombre porque la comparacion tiene que ser el mismo
#: modelo bajo esquemas distintos: dejar que cada esquema elija su propio ganador compararia
#: modelos distintos y el efecto del esquema quedaria mezclado con el de la seleccion.
HEADS = {
    "MLP06_curve_kndvi_raw100": "MLP",
    "RF06_curve_all-topo-area_raw100": "RF",
    "C2D02_serpentine_kndvi_raw36_ctxclim-topo-area": "C2D",
    "C1D01_curve1d_kndvi_raw36_ctxclim-topo-area": "C1D",
    "B03_coords": "BASE",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/tables/temporal_validation.csv")
    args = p.parse_args()

    s = pd.read_csv(ROOT / "results" / "models" / "summary.csv")
    rows = []
    for sc in SCHEMES:
        sub = s[s["scheme"] == sc]
        if sub.empty:
            continue
        m = mx.by_facet(sub)
        for rid, fam in HEADS.items():
            if rid in m.index:
                rows.append(dict(scheme=sc, familia=fam, run_id=rid,
                                 **{f: m.loc[rid, f] for f in tg.FACETS},
                                 media=m.loc[rid, "media"]))
    if not rows:
        raise SystemExit("todavia no hay corridas bajo los esquemas temporales")
    d = pd.DataFrame(rows)

    piv = d.pivot_table(index="familia", columns="scheme", values="media")
    piv = piv[[c for c in SCHEMES if c in piv.columns]]
    print("media sobre las cinco facetas, mismo modelo bajo cada esquema:\n")
    print(piv.round(3).to_string())

    if {"kfold5_window", "kfold_loc_time"} <= set(piv.columns):
        print("\ncaida respecto de kfold5_window:")
        drop = piv.sub(piv["kfold5_window"], axis=0).drop(columns="kfold5_window")
        print(drop.round(3).to_string())

        need = ["kfold5_window", "kfold_time", "kfold_loc_time", "kfold5_block20"]
        missing = [c for c in need if c not in piv.columns]
        if missing:
            print(f"\ndescomposicion sitio/tiempo: falta {', '.join(missing)}. "
                  "Sin el termino espacial puro no se puede separar cuanto de la caida del "
                  "LLTO es el sitio y cuanto el ano; correr esos esquemas para los cabezas "
                  "de familia.")
        else:
            print("\ndescomposicion, para las familias con los tres esquemas:")
            for fam in piv.index:
                r = piv.loc[fam]
                if r[need].isna().any():
                    continue
                dt = r["kfold_time"] - r["kfold5_window"]
                ds = r["kfold5_block20"] - r["kfold5_window"]
                dst = r["kfold_loc_time"] - r["kfold5_window"]
                print(f"  {fam:5s} tiempo {dt:+.3f}  sitio {ds:+.3f}  ambos {dst:+.3f}"
                      f"   (interaccion {dst - dt - ds:+.3f})")

    # por faceta, sólo para el esquema más exigente
    hard = d[d.scheme == "kfold_loc_time"]
    if len(hard):
        print("\npor faceta bajo kfold_loc_time (sitio nuevo y ano nuevo):")
        print(hard.set_index("familia")[list(tg.FACETS) + ["media"]]
              .sort_values("media", ascending=False).round(3).to_string())

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
