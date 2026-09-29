#!/usr/bin/env python3
"""Lecturas del OOF por banda de latitud, sobre el pool tratado como uno solo.

`scripts/100` parte por fuente porque la fuente fue el confusor que hubo que descartar.
Una vez descartado, la fuente es procedencia y no ecologia: el eje que significa algo es la
latitud. Este script rehace las mismas lecturas sobre el pool unificado, sin mirar de que
inventario viene cada parcela.

Dos cosas que conviene tener presentes al leerlo:

- Las dos fuentes solo se solapan entre 30 y 38 S. Al sur de 38 todas las parcelas son de
  Living Trees, asi que una banda austral no es una mezcla de inventarios: es un inventario.
  La banda sigue siendo la unidad correcta -- no se compara entre bandas-- pero la
  independencia respecto de la fuente vale para el conjunto, no dentro de cada banda.
- El R2 "dentro de banda" centra observado y predicho por banda y toma la referencia en 0,
  igual que `scripts/90`. No es el R2 de un modelo reajustado por banda.

Salida: `results/tables/lecturas_por_latitud.csv`.

Uso:
    python scripts/104_lecturas_por_latitud.py [--band 2.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
OUT = ROOT / "results" / "tables" / "lecturas_por_latitud.csv"
SCHEME = "kfold5_block20_unified"

#: Bloque -> corrida, sobre el pool leñoso. Las dos familias de target van por separado
#: porque `pg_all` y `unified_all` no cubren las mismas parcelas.
RUNS_PG = {
    "piso_topo_area": "B01c_topo_ctr-area_raw100_pg-all_unified_woody",
    "clima":          "RFG1c_clim-topo_ctr-area_raw100_pg-all_unified_woody",
    "curva":          "RFG2c_curve-topo_ctr-area_kndvi_raw100_pg-all_unified_woody",
    "lsp":            "RFG7c_lspu-topo_ctr-area_kndvi_raw100_pg-all_unified_woody",
    "gm":             "RFG4c_gm-topo_ctr-area_raw100_pg-all_unified_woody",
    "clima_curva":    "RFG3c_clim-curve-topo_ctr-area_kndvi_raw100_pg-all_unified_woody",
    "gm_clima":       "RFG5c_gm-clim-topo_ctr-area_raw100_pg-all_unified_woody",
}
RUNS_UNI = {k: v.replace("pg-all", "unified-all") for k, v in RUNS_PG.items()}
TARGETS = {"lcbd_count_sorensen": RUNS_PG, "lcbd_pa_unified": RUNS_UNI,
           "hill_q0_unified": RUNS_UNI}

MIN_N = 40          #: banda x clase con menos parcelas que esto no se reporta

#: Zonas para el contraste geomediano-curva. Tres y no dos a proposito: 30-38 mezcla los dos
#: inventarios, y 38-46 y 46-56 son Living Trees solo pero ecologicamente distintas. Si el
#: efecto aparece en la zona mixta Y en las dos que no lo son, no puede ser de la fuente.
ZONAS = [(-56, -46, "46-56 S  Patagonia"), (-46, -38, "38-46 S  templado"),
         (-38, -29, "30-38 S  mediterraneo")]
PENDS = [(-1, 5, "<5"), (5, 15, "5-15"), (15, 25, "15-25"), (25, 90, ">25")]


def r2(o: np.ndarray, p: np.ndarray) -> float:
    return 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def r2_centrado(o: np.ndarray, p: np.ndarray) -> float:
    """R2 de valores ya centrados por grupo: la referencia es 0, no la media global."""
    return 1 - ((p - o) ** 2).sum() / (o ** 2).sum()


def sitios() -> pd.DataFrame:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lat", "source"]]
    t = pd.read_parquet(DERIVED / "topography_unified.parquet")[["plot_id", "slope"]]
    return p.merge(t, left_on="PlotObservationID", right_on="plot_id", how="left")


def oof(run: str, target: str) -> pd.DataFrame | None:
    f = GATE / run / SCHEME / "oof_predictions.csv"
    if not f.exists():
        return None
    cols = ["PlotObservationID", "seed", f"{target}_obs", f"{target}_pred"]
    d = pd.read_csv(f, usecols=lambda c: c in cols)
    if f"{target}_obs" not in d:
        return None
    return d.rename(columns={f"{target}_obs": "o", f"{target}_pred": "p"}).dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=2.0)
    a = ap.parse_args()

    s = sitios()
    rows = []
    for target, runs in TARGETS.items():
        for bloque, run in runs.items():
            d = oof(run, target)
            if d is None:
                print(f"  falta {run} / {target}")
                continue
            d = d.merge(s, on="PlotObservationID", how="left").dropna(subset=["lat"])
            d["banda"] = (np.floor(d.lat / a.band) * a.band).astype(int)

            for seed, g in d.groupby("seed"):
                rows.append(dict(target=target, bloque=bloque, seed=int(seed),
                                 banda="todas", n=len(g), lectura="global", R2=r2(g.o, g.p)))
                # centrado por banda: quita el escalon entre bandas de las dos series
                oc = g.o - g.groupby("banda").o.transform("mean")
                pc = g.p - g.groupby("banda").p.transform("mean")
                rows.append(dict(target=target, bloque=bloque, seed=int(seed),
                                 banda="todas", n=len(g), lectura="dentro_de_banda",
                                 R2=r2_centrado(oc.to_numpy(), pc.to_numpy())))
                # y banda por banda, sin centrar: el R2 que se logra dentro de cada una
                for b, gb in g.groupby("banda"):
                    if len(gb) < MIN_N:
                        continue
                    rows.append(dict(target=target, bloque=bloque, seed=int(seed),
                                     banda=int(b), n=len(gb), lectura="en_la_banda",
                                     R2=r2(gb.o, gb.p)))

    # --- contraste geomediano contra curva, por zona latitudinal y clase de pendiente
    pares = []
    for target, runs in TARGETS.items():
        a_, b_ = oof(runs["gm"], target), oof(runs["curva"], target)
        if a_ is None or b_ is None:
            continue
        j = (a_.merge(b_, on=["PlotObservationID", "seed"], suffixes=("_gm", "_cv"))
               .merge(s, on="PlotObservationID", how="left").dropna(subset=["lat", "slope"]))
        for lo, hi, zona in ZONAS:
            for slo, shi, pend in PENDS:
                g = j[(j.lat >= lo) & (j.lat < hi) & (j.slope > slo) & (j.slope <= shi)]
                n = int(len(g) / max(g.seed.nunique(), 1))
                if n < MIN_N:
                    continue
                d = [r2(x.o_gm, x.p_gm) - r2(x.o_cv, x.p_cv) for _, x in g.groupby("seed")]
                pares.append(dict(target=target, zona=zona, pendiente=pend, n=n,
                                  n_seeds=g.seed.nunique(), dif_mean=float(np.mean(d)),
                                  dif_sd=float(np.std(d, ddof=1))))
    pd.DataFrame(pares).to_csv(OUT.with_name("gm_menos_curva_por_zona.csv"), index=False)
    print(f"-> {OUT.with_name('gm_menos_curva_por_zona.csv')}  ({len(pares)} filas)")

    t = pd.DataFrame(rows)
    agg = (t.groupby(["target", "bloque", "banda", "lectura"])
             .agg(n=("n", "first"), n_seeds=("seed", "nunique"),
                  R2_mean=("R2", "mean"), R2_sd=("R2", "std"))
             .reset_index())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"-> {OUT}  ({len(agg)} filas)")

    w = agg[(agg.target == "lcbd_count_sorensen") & (agg.banda == "todas")]
    print(w.pivot_table(index="bloque", columns="lectura", values="R2_mean").round(3))


if __name__ == "__main__":
    main()
