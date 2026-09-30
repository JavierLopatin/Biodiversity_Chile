#!/usr/bin/env python3
"""Altura ETH: ¿el bloque es un indicador geográfico por la puerta de atrás? Control previo a la compuerta.

Igual que scripts/121 con la disponibilidad de GEDI. Desde (altura en el píxel, altura == 0,
SD del ensemble, n de píxeles enmascarados en la ventana 9x9) un RF predice (a) la latitud,
R² con CV de bloques (kfold5_block20_unified), y (b) el estrato MapBiomas Forest/NoBosque,
AUC con los mismos folds. Como referencia, lo mismo desde el geomediano (gmoall) y desde
coordenadas, porque cualquier predictor espectral predice latitud y estrato en parte, y lo que
importa es si eth es peor que lo que ya usamos. Además, las 34 parcelas enmascaradas: su
indicador contra fuente y latitud.

Escribe results/tables/eth_proxy_geografico.csv.

Uso:
    python scripts/130_eth_proxy_geografico.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biodiv import features as feat                                          # noqa: E402

ID = "PlotObservationID"


def main() -> None:
    P = pd.read_parquet(ROOT / "data/derived/eth_canopy_plot.parquet")
    z = np.load(ROOT / "data/derived/eth_canopy_windows.npz")
    w = z["windows"]
    P["eth_h_cero"] = (P.eth_h == 0).astype(float)
    P["eth_n_mask9"] = np.isnan(w[:, 0, 6:15, 6:15]).reshape(len(w), -1).sum(1)
    P["eth_isna"] = P.eth_h.isna().astype(float)
    pl = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lat", "lon", "source"]]
    mb = pd.read_parquet(ROOT / "data/derived/mapbiomas_class_unified.parquet")[[ID, "mb_class"]]
    P = P.merge(pl, on=ID).merge(mb, on=ID, how="left")
    P["forest"] = P.mb_class.fillna("").str.contains("Forest").astype(int)
    folds = pd.read_parquet(ROOT / "data/derived/cv_folds_unified_block20.parquet")
    fold = P[ID].map(folds[folds.split == "test"].set_index(ID).fold).to_numpy()
    gm, _ = feat.build_design("gmoall", derived=str(ROOT / "data/derived"), px="center", ids=pd.Index(P[ID]))
    sets = {
        "eth_control (h, h==0, SD, n_mask)": P[["eth_h", "eth_h_cero", "eth_sd", "eth_n_mask9", "eth_isna"]],
        "eth_bloque (h, SD, ventanas)": P[[c for c in P.columns if c.startswith("eth_") and c != "eth_desfase_anios"]],
        "solo eth_h == 0": P[["eth_h_cero"]],
        "gm (gmoall) referencia": gm.reset_index(drop=True),
    }
    rows = []
    for name, X in sets.items():
        X = X.fillna(X.median()).to_numpy(float)
        for tgt in ("lat", "forest"):
            y = P[tgt].to_numpy()
            pred = np.full(len(P), np.nan)
            for f in np.unique(fold[~pd.isna(fold)]):
                te = fold == f
                if tgt == "lat":
                    m = RandomForestRegressor(500, min_samples_leaf=5, n_jobs=-1, random_state=0)
                    pred[te] = m.fit(X[~te], y[~te]).predict(X[te])
                else:
                    m = RandomForestClassifier(500, min_samples_leaf=5, n_jobs=-1, random_state=0)
                    pred[te] = m.fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
            ok = np.isfinite(pred)
            if tgt == "lat":
                v = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
                rows.append(dict(predictores=name, respuesta="latitud", metrica="R2_cv_bloques", valor=v))
            else:
                rows.append(dict(predictores=name, respuesta="Forest/NoBosque", metrica="AUC_cv_bloques",
                                 valor=roc_auc_score(y[ok], pred[ok])))
    # las enmascaradas
    na = P.eth_isna == 1
    rows.append(dict(predictores="eth_isna", respuesta="fuente", metrica="frac_LT_entre_enmascaradas",
                     valor=(P.source[na] == "living_trees").mean()))
    rows.append(dict(predictores="eth_isna", respuesta="fuente", metrica="frac_LT_pool", valor=(P.source == "living_trees").mean()))
    rows.append(dict(predictores="eth_isna", respuesta="latitud", metrica="mannwhitney_p",
                     valor=mannwhitneyu(P.lat[na], P.lat[~na]).pvalue))
    rows.append(dict(predictores="eth_isna", respuesta="latitud", metrica="lat_mediana_enmascaradas", valor=P.lat[na].median()))
    rows.append(dict(predictores="eth_h == 0", respuesta="estrato", metrica="frac_NoBosque_entre_ceros",
                     valor=1 - P.forest[P.eth_h_cero == 1].mean()))
    rows.append(dict(predictores="eth_h == 0", respuesta="n", metrica="n_ceros", valor=(P.eth_h_cero == 1).sum()))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/eth_proxy_geografico.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
