#!/usr/bin/env python3
"""R² por clase de pendiente del DEM: reflectancia corregida (SCS+C) contra original.

Lee los OOF de results/models_topocorr/ (scripts/09 con BIODIV_CURVES=_raw100<v> y
BIODIV_LSPU=<v>) y compara cada versión corregida contra nc, que es la reflectancia
original pasada por el mismo código (scripts/97). La comparación va pareada por semilla y
sobre las mismas parcelas, con los mismos folds.

El contraste que importa no es el R² global sino el R² por clase de pendiente. La
corrección solo debería mover las parcelas con pendiente; si llano y pendiente mejoran
parejo, sospechar un bug antes de celebrar. R² dentro de cada clase, contra la media de esa
clase: mide si el modelo ordena parcelas de pendiente parecida, no si separa llano de ladera.

Uso:
    python scripts/98_topocorr_by_slope.py
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "models_topocorr"
SCHEME = "kfold5_block20_unified"
ID = "PlotObservationID"
TARGETS = {"pg-all": ["lcbd_count_sorensen"], "unified-all": ["lcbd_pa_unified", "hill_q0_unified"]}
BINS, LABELS = [-0.01, 5, 15, 25, 90], ["<5", "5-15", "15-25", ">25"]


def r2(o, p):
    o, p = np.asarray(o), np.asarray(p)
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def main() -> None:
    topo = pd.read_parquet(ROOT / "data" / "derived" / "topography_unified.parquet")
    topo = topo.rename(columns={"plot_id": ID})[[ID, "slope"]]
    topo["slope_cls"] = pd.cut(topo.slope, BINS, labels=LABELS).astype(str)
    rows = []
    for f in glob.glob(str(RUNS / "*" / SCHEME / "oof_predictions.csv")):
        run = Path(f).parts[-3]
        spec = "curva" if run.startswith("RFG2") else "lsp"
        v = re.search(r"_raw100(nc|tcfe|tccsall|tccs)", run).group(1)
        tset = "unified-all" if "unified-all" in run else "pg-all"
        oof = pd.read_csv(f).merge(topo, on=ID, how="left")
        for t in TARGETS[tset]:
            d = oof[oof[f"{t}_obs"].notna()]
            for sc, g in [("todas", d)] + list(d.groupby("slope_cls")):
                for seed, h in g.groupby("seed"):
                    rows.append(dict(spec=spec, version=v, target=t, slope_cls=sc, seed=seed,
                                     n=h[ID].nunique(), R2=r2(h[f"{t}_obs"], h[f"{t}_pred"])))
    r = pd.DataFrame(rows)
    base = r[r.version == "nc"].set_index(["spec", "target", "slope_cls", "seed"]).R2
    r["dR2"] = r.R2 - r.set_index(["spec", "target", "slope_cls", "seed"]).index.map(base)
    out = (r.groupby(["target", "spec", "slope_cls", "version"])
             .agg(n=("n", "first"), R2=("R2", "mean"), dR2=("dR2", "mean"), dR2_sd=("dR2", "std"))
             .reset_index())
    RUNS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RUNS / "r2_by_slope.csv", index=False)
    order = ["todas"] + LABELS
    for t in out.target.unique():
        for sp in ["curva", "lsp"]:
            x = out[(out.target == t) & (out.spec == sp)]
            p = x.pivot_table(index="slope_cls", columns="version", values=["R2", "dR2"]).reindex(order)
            n = x.groupby("slope_cls").n.first().reindex(order)
            print(f"\n== {t} / {sp} ==  (n por clase: {n.to_dict()})")
            print(p.round(3).to_string())
    print(f"\n-> {RUNS / 'r2_by_slope.csv'}")


if __name__ == "__main__":
    main()
