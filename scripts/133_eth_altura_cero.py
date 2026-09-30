#!/usr/bin/env python3
"""¿La ganancia de riqueza de la altura ETH en el matorral de 30-35°S viene de las parcelas con altura 0?

La compuerta (scripts/132) dio RFE1 − RFG5a = +0,017 ± 0,007 en hill_q0 dentro de banda en
NoBosque 30-35°S, justo donde la SD del ensemble supera a la altura. Hipótesis (sin afirmar):
altura == 0 separaría suelo desnudo o vegetación rala de vegetación leñosa. El bloque eth no
tiene una columna "altura == 0" que quitar, así que se mira sobre el mismo OOF: la diferencia
pareada por semilla restringida a las parcelas con y sin altura 0, y la riqueza de cada grupo.

Escribe results/tables/eth_altura_cero.csv.

Uso:
    python scripts/133_eth_altura_cero.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID, B, T = "PlotObservationID", "kfold5_block20_unified", "hill_q0_unified"


def cr2(o, q):
    return 1 - ((o - q) ** 2).sum() / (o ** 2).sum()


def band(g):
    b = np.floor(g.lat / 2)
    return cr2(g.o - g.groupby(b).o.transform("mean"), g.q - g.groupby(b).q.transform("mean"))


def main() -> None:
    e = pd.read_parquet(ROOT / "data/derived/eth_canopy_plot.parquet")[[ID, "eth_h"]]
    p = (pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lat"]]
         .merge(pd.read_parquet(ROOT / "data/derived/mapbiomas_class_unified.parquet")[[ID, "mb_class"]], how="left")
         .merge(e))
    cell = p[~p.mb_class.fillna("").str.contains("Forest") & (p.lat >= -35)].copy()
    cell["cero"] = cell.eth_h == 0
    y = pd.read_parquet(ROOT / "data/derived/unified_diversity_responses_woody.parquet")[[ID, T]]
    rows = [dict(grupo=f"altura_0={k}", medida="riqueza_media", valor=g[T].mean(), n=len(g))
            for k, g in cell.merge(y).groupby("cero")]
    oo = {}
    for m in ("RFE1", "RFG5a"):
        f = next((ROOT / "results/models_eth").glob(f"{m}c_*unified-all_unified_woody*")) / B / "oof_predictions.csv"
        oo[m] = (pd.read_csv(f)[[ID, "seed", f"{T}_obs", f"{T}_pred"]]
                 .rename(columns={f"{T}_obs": "o", f"{T}_pred": "q"}).merge(cell[[ID, "lat", "cero"]]))
    for name, sel in (("toda_la_celda", lambda d: d), ("sin_altura_0", lambda d: d[~d.cero]),
                      ("solo_altura_0", lambda d: d[d.cero])):
        r = {m: pd.Series({s: band(sel(g)) for s, g in oo[m].groupby("seed")}) for m in oo}
        d = r["RFE1"] - r["RFG5a"]
        rows.append(dict(grupo=name, medida="RFE1-RFG5a_banda", valor=d.mean(), DE=d.std(),
                         n=sel(oo["RFE1"])[ID].nunique()))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/eth_altura_cero.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
