#!/usr/bin/env python3
"""Deep CORAL sobre la 1D-CNN (C1D01): los cinco brazos, pareados por semilla contra el control.

Brazos (pg_all leñoso, kfold5_block20_unified, 3 semillas, results/models_coral), todos por el
mismo camino de código (pasada del destino a todo lambda):
  A  lambda = 0, BatchNorm congelado en la pasada del destino   control limpio
  B  lambda = 0, BatchNorm actualizado por el destino           AdaBN sola
  C  lambda = 1                                                 sensibilidad (inerte por escala)
  D  lambda = 10                                                sensibilidad (inerte por escala)
  E  lambda* = mediana de train_loss / coral_loss en el mejor epoch de A (regla de Sun &
     Saenko 2016, calibrada solo con pérdidas de entrenamiento del control)    PRIMARIO

Criterio prerregistrado: E sube el R² de lcbd_count_sorensen dentro de banda de 2° (pool
entero) al menos 0,02 sobre A. Además da la trayectoria de la pérdida CORAL: si en E cae a
casi cero, no había nada que alinear; si se queda alta, alinear compite con predecir.

Escribe results/tables/coral_brazos.csv y results/tables/coral_perdida.csv.

Uso:
    python scripts/120_coral_arms.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
B = "kfold5_block20_unified"
_s = importlib.util.spec_from_file_location("l100", ROOT / "scripts" / "100_paired_lenses.py")
l100 = importlib.util.module_from_spec(_s); _s.loader.exec_module(l100)

C = "results/models_coral/C1D01_curve1d_kndvi_raw100_pg-all_unified_woody_coral{}_ctr"
TARGETS = ["lcbd_count_sorensen", "td_inext_q0", "pd_inext_q0"]


def arms() -> dict[str, Path]:
    out = {"A": ROOT / C.format("0"), "B": ROOT / C.format("0bnupd"),
           "C": ROOT / C.format("1"), "D": ROOT / C.format("10")}
    e = [p for p in (ROOT / "results/models_coral").glob("C1D01_*_coral*_ctr")
         if p not in out.values() and (p / B / "oof_predictions.csv").exists()]
    if len(e) == 1:
        out["E"] = e[0]
    return out


def pool_band(o: pd.DataFrame, t: str, band: float = 2.0) -> pd.Series:
    d = o[o[f"{t}_obs"].notna()].copy()
    d["b"] = np.floor(d.lat / band)
    res = {}
    for sd, g in d.groupby("seed"):
        oc = g[f"{t}_obs"] - g.groupby("b")[f"{t}_obs"].transform("mean")
        pc = g[f"{t}_pred"] - g.groupby("b")[f"{t}_pred"].transform("mean")
        res[sd] = l100.cr2(oc, pc)
    return pd.Series(res)


def main() -> None:
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "Owner", "source", "lat"]]
    A = arms()
    L, rows, loss_rows = {}, [], []
    for k, p in A.items():
        o = pd.read_csv(p / B / "oof_predictions.csv").merge(plots, on=ID, how="left")
        L[k] = {t: l100.lenses(o, t, 2.0).assign(pool_dentro_banda=pool_band(o, t)) for t in TARGETS}
        h = pd.read_csv(p / B / "history.csv")
        best = h.loc[h.groupby(["seed", "fold"]).val_loss.idxmin()]
        first = h[h.epoch == 0]
        loss_rows.append(dict(brazo=k, run=p.name, lambda_max=h.coral_lambda.max(),
                              coral_epoch0=first.coral_loss.median(),
                              coral_mejor_epoch=best.coral_loss.median(),
                              huber_mejor_epoch=best.train_loss.median() if k in "AB" else np.nan,
                              lambda_x_coral_mejor=(best.coral_lambda * best.coral_loss).median(),
                              mejor_epoch=best.epoch.median(), train_loss_mejor=best.train_loss.median()))
    lect = ["todas", "pool_dentro_banda", "lt_dentro_banda", "pcl_dentro_owner"]
    for k in A:
        for t in TARGETS:
            x, d = L[k][t], L[k][t] - L["A"][t]
            for le in lect:
                rows.append(dict(brazo=k, run=A[k].name, target=t, lectura=le, R2=x[le].mean(),
                                 R2_DE=x[le].std(ddof=1), dif_vs_A=d[le].mean(),
                                 dif_vs_A_DE=d[le].std(ddof=1)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/coral_brazos.csv", index=False)
    ls = pd.DataFrame(loss_rows)
    ls.to_csv(ROOT / "results/tables/coral_perdida.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.pivot_table(index=["target", "brazo"], columns="lectura", values="dif_vs_A").round(3).to_string())
    print(out[out.brazo == "A"].pivot_table(index="target", columns="lectura", values="R2").round(3).to_string())
    print(ls.round(5).to_string())


if __name__ == "__main__":
    main()
