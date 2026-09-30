#!/usr/bin/env python3
"""Compuerta de la altura ETH sobre el pool leñoso: dentro de banda y por estrato desde el principio.

Corridas en results/models_eth (kfold5_block20_unified, 3 semillas, BIODIV_TARGETS=_woody):
B03 (coordenadas), B01 (topo+área), RFG5a (gm+clima), RFE1 (gm+clima+eth), RFE2 (eth solo),
RFE3 (gm+clima+dispersión de altura en ventanas). Pares por semilla:
  RFE1 − RFG5a   ¿la altura aporta sobre gm + clima?      <- la pregunta
  RFE3 − RFG5a   ¿aporta la dispersión estructural sola?
  RFE2 − B01     ¿la altura sola informa?
  RFE1 − B03, RFG5a − B03, RFE2 − B03   contra el piso de coordenadas

Lecturas: pool entero y dentro de banda de 2° (la del paper; la agrupada no significa nada
porque D1 dio 0,833 agrupado contra 0,540 en banda), y lo mismo DENTRO de cada estrato
MapBiomas (Forest / NoBosque): el producto mide en un estrato y falla en el matorral de
30-35°S, así que un resultado agrupado mezclaría las dos cosas.

Escribe results/tables/eth_compuerta.csv.

Uso:
    python scripts/132_eth_compuerta.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
B = "kfold5_block20_unified"
M = ROOT / "results/models_eth"
MODELS = ["B03", "B01", "RFG5a", "RFE1", "RFE2", "RFE3"]
PAIRS = [("RFE1", "RFG5a"), ("RFE3", "RFG5a"), ("RFE2", "B01"), ("RFE1", "B03"),
         ("RFG5a", "B03"), ("RFE2", "B03")]


def r2(o, p):
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def cr2(o, p):
    return float(1 - ((o - p) ** 2).sum() / (o ** 2).sum())


def band_r2(g: pd.DataFrame) -> float:
    b = np.floor(g.lat / 2)
    return cr2(g.o - g.groupby(b).o.transform("mean"), g.p - g.groupby(b).p.transform("mean"))


def lenses(o: pd.DataFrame, t: str) -> pd.DataFrame:
    d = o[o[f"{t}_obs"].notna()].rename(columns={f"{t}_obs": "o", f"{t}_pred": "p"})
    rows = []
    for seed, g in d.groupby("seed"):
        r = {"seed": seed, "todas": r2(g.o, g.p), "banda": band_r2(g)}
        for s, h in g.groupby("estrato"):
            if len(h) > 30:
                r[f"{s}"] = r2(h.o, h.p)
                r[f"{s}_banda"] = band_r2(h)
                r[f"n_{s}"] = len(h)
        r["n"] = len(g)
        rows.append(r)
    return pd.DataFrame(rows).set_index("seed")


def run_dir(model: str, ts: str) -> Path | None:
    hits = [p for p in M.glob(f"{model}c_*_{ts}_unified_woody*") if (p / B / "oof_predictions.csv").exists()]
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lat"]]
    mb = pd.read_parquet(ROOT / "data/derived/mapbiomas_class_unified.parquet")[[ID, "mb_class"]]
    plots = plots.merge(mb, on=ID, how="left")
    forest = plots.mb_class.fillna("").str.contains("Forest")
    # NoBosque se parte en 30-35°S (donde D2 mostró SD > altura en el 57 %) y el resto (>35°S,
    # razón 0,52-0,86): leerlo agrupado mezclaría una zona donde la altura es ruido con otra
    # donde es señal. Forest es homogéneo (razón 0,40-0,75) y va entero.
    plots["estrato"] = np.where(forest, "Forest", np.where(plots.lat >= -35, "NoBosque_30-35", "NoBosque_>35"))
    fams = {"pg-all": ["lcbd_count_sorensen", "td_inext_q0", "pd_inext_q0"],
            "unified-all": ["hill_q0_unified", "lcbd_pa_unified", "lcbd_freq_unified", "pcoa1_pa_unified",
                            "pcoa2_pa_unified", "mpd_unified", "mntd_unified", "ses_mpd_unified",
                            "ses_pd_unified", "dark_n_unified"]}
    rows = []
    for ts, targets in fams.items():
        oofs = {m: pd.read_csv(d / B / "oof_predictions.csv").merge(plots, on=ID, how="left")
                for m in MODELS if (d := run_dir(m, ts)) is not None}
        for t in targets:
            L = {m: lenses(o, t) for m, o in oofs.items() if f"{t}_obs" in o}
            cols = [c for c in next(iter(L.values())).columns if not c.startswith("n")]
            for m, x in L.items():
                for c in cols:
                    rows.append(dict(tipo="modelo", familia=ts, target=t, etiqueta=m, lectura=c,
                                     valor=x[c].mean(), DE=x[c].std(ddof=1)))
            for a, b in PAIRS:
                if a in L and b in L:
                    dd = L[a][cols] - L[b][cols]
                    for c in cols:
                        rows.append(dict(tipo="par", familia=ts, target=t, etiqueta=f"{a} − {b}", lectura=c,
                                         valor=dd[c].mean(), DE=dd[c].std(ddof=1)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/eth_compuerta.csv", index=False)
    pd.set_option("display.width", 250)
    x = out[out.tipo == "par"].pivot_table(index=["familia", "target", "etiqueta"], columns="lectura", values="valor")
    print(x[[c for c in ["todas", "banda", "Forest", "Forest_banda", "NoBosque_30-35", "NoBosque_30-35_banda",
             "NoBosque_>35", "NoBosque_>35_banda"] if c in x]].round(3).to_string())


if __name__ == "__main__":
    main()
