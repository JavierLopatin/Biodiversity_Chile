#!/usr/bin/env python3
"""Altura de dosel ETH (Lang et al. 2023): los dos tests de descarte y el contraste con GEDI.

Lee las ventanas de 21x21 por parcela (scripts/126 --modo parcelas o scripts/128).

D1. ¿Lang es redundante con lo que ya tenemos? RF que predice la altura de Lang en el píxel de
    la parcela (y la media 3x3) desde gm (gmoall) + clima + topografía, con los folds de bloques
    de siempre (kfold5_block20_unified), 3 semillas. R² en el pool, por fuente y dentro de banda
    de 2° (pool entero). Lectura graduada: ~0,6 ya es redundancia sustancial, ~0,8 casi total.

D2. ¿Sirve en vegetación baja? Por zona latitudinal (las de scripts/122) x estrato MapBiomas
    (Forest si la clase contiene "Forest", NoBosque si no): n, mediana de la altura, de la SD y
    de SD/altura. La SD es la del ENSEMBLE (incertidumbre predictiva): detecta incertidumbre,
    no sesgo.

GEDI. Validación, no predicción: cada footprint L2A 2022 que cae dentro de la ventana de una
    parcela se empareja con el píxel de Lang que lo contiene; rh98 contra altura de Lang por
    zona y estrato (sesgo mediano Lang − rh98, error absoluto mediano, Spearman). Reporta la
    dirección medida del sesgo, sin suponer ninguna.

Además el desfase: 2020 − año del censo.

Escribe results/tables/eth_d1_redundancia.csv, eth_d2_incertidumbre.csv, eth_gedi_validacion.csv
y data/derived/eth_canopy_plot.parquet (altura y SD en el píxel, media/sd/cuantiles en 3x3,
5x5 y 9x9).

Uso:
    python scripts/129_eth_descarte.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biodiv import features as feat                                          # noqa: E402

ID = "PlotObservationID"
WIN = 21
C = WIN // 2
ZONES = [(-30, -17, "norte_>30S"), (-35, -30, "30-35S"), (-38, -35, "35-38S"),
         (-42, -38, "38-42S"), (-47, -42, "42-47S"), (-56, -47, "sur_<47S")]


def zone(lat: pd.Series) -> pd.Series:
    out = pd.Series("fuera", index=lat.index)
    for lo, hi, n in ZONES:
        out[(lat >= lo) & (lat < hi)] = n
    return out


def r2(o, p):
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def cr2(o, p):
    return float(1 - ((o - p) ** 2).sum() / (o ** 2).sum())


def plot_table(w: np.ndarray, ids: pd.Series) -> pd.DataFrame:
    h, s = w[:, 0], w[:, 1]
    out = pd.DataFrame({ID: ids, "eth_h": h[:, C, C], "eth_sd": s[:, C, C]})
    for k in (3, 5, 9):
        r = k // 2
        sub = h[:, C - r:C + r + 1, C - r:C + r + 1].reshape(len(h), -1)
        out[f"eth_h{k}_mean"] = np.nanmean(sub, 1)
        out[f"eth_h{k}_sd"] = np.nanstd(sub, 1)
        for q in (10, 50, 90):
            out[f"eth_h{k}_q{q}"] = np.nanpercentile(sub, q, axis=1)
        out[f"eth_sd{k}_mean"] = np.nanmean(s[:, C - r:C + r + 1, C - r:C + r + 1].reshape(len(h), -1), 1)
    return out


def main() -> None:
    z = np.load(ROOT / "data/derived/eth_canopy_windows.npz")
    w, dx, dy = z["windows"].copy(), float(z["dx"]), float(z["dy"])
    w[~np.isfinite(w)] = np.nan        # enmascarados de EE (-inf) o de relleno: fuera del producto
    idx = pd.read_parquet(ROOT / "data/derived/eth_canopy_windows.parquet")
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lat", "lon", "source", "Year"]]
    P = plot_table(w, idx[ID]).merge(plots, on=ID)
    mb = pd.read_parquet(ROOT / "data/derived/mapbiomas_class_unified.parquet")[[ID, "mb_class"]]
    P = P.merge(mb, on=ID, how="left")
    P["estrato"] = np.where(P.mb_class.fillna("").str.contains("Forest"), "Forest", "NoBosque")
    P["zona"] = zone(P.lat)
    P["eth_desfase_anios"] = 2020 - P.Year
    P.drop(columns=["lat", "lon", "source", "Year", "mb_class", "zona", "estrato"]).to_parquet(
        ROOT / "data/derived/eth_canopy_plot.parquet", index=False)
    print(f"parcelas con altura en el píxel: {P.eth_h.notna().sum()}/{len(P)}")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # D1 -----------------------------------------------------------------------------------
    X, _ = feat.build_design("gmoall+clim+topo_ctr", derived=str(ROOT / "data/derived"), px="center",
                             ids=pd.Index(P[ID]))
    X = X.fillna(X.median())
    folds = pd.read_parquet(ROOT / "data/derived/cv_folds_unified_block20.parquet")
    te_fold = folds[folds.split == "test"].set_index(ID).fold
    fold = P[ID].map(te_fold).to_numpy()
    rows = []
    for yname in ("eth_h", "eth_h3_mean", "eth_sd"):
        y = P[yname].to_numpy(float)
        for seed in range(3):
            pred = np.full(len(P), np.nan)
            for f in np.unique(fold[~pd.isna(fold)]):
                te = fold == f
                tr = ~te & np.isfinite(y)
                rf = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.33, n_jobs=-1,
                                           random_state=seed).fit(X[tr], y[tr])
                pred[te] = rf.predict(X[te])
            d = P.assign(o=y, p=pred)
            d = d[np.isfinite(d.o) & np.isfinite(d.p)]
            d["b"] = np.floor(d.lat / 2)
            oc = d.o - d.groupby("b").o.transform("mean")
            pc = d.p - d.groupby("b").p.transform("mean")
            rec = {"todas": r2(d.o, d.p), "pool_dentro_banda": cr2(oc, pc)}
            for s, g in d.groupby("source"):
                rec[s] = r2(g.o, g.p)
            for s, g in d.groupby("estrato"):
                rec[f"estrato_{s}"] = r2(g.o, g.p)
            for k, v in rec.items():
                rows.append(dict(variable=yname, predictores="gmoall+clim+topo_ctr", seed=seed,
                                 lectura=k, R2=v, n=len(d)))
    D1 = pd.DataFrame(rows).groupby(["variable", "predictores", "lectura", "n"]).R2.agg(["mean", "std"]).reset_index()
    D1.to_csv(ROOT / "results/tables/eth_d1_redundancia.csv", index=False)

    # D2 -----------------------------------------------------------------------------------
    P["ratio_sd_h"] = P.eth_sd / P.eth_h.where(P.eth_h > 0)
    D2 = (P.groupby(["zona", "estrato"])
          .agg(n=("eth_h", "size"), h_mediana=("eth_h", "median"), sd_mediana=("eth_sd", "median"),
               ratio_sd_h_mediana=("ratio_sd_h", "median"), frac_ratio_gt1=("ratio_sd_h", lambda s: (s > 1).mean()),
               frac_h0=("eth_h", lambda s: (s == 0).mean()), h9_sd_mediana=("eth_h9_sd", "median"),
               desfase_mediana=("eth_desfase_anios", "median"))
          .reset_index())
    D2.to_csv(ROOT / "results/tables/eth_d2_incertidumbre.csv", index=False)

    # GEDI ---------------------------------------------------------------------------------
    _s = importlib.util.spec_from_file_location("g121", ROOT / "scripts" / "121_gedi_disponibilidad.py")
    g121 = importlib.util.module_from_spec(_s); _s.loader.exec_module(g121)
    g = g121.load_gedi(g121.GEDI, "L2A", ["lat", "lon", "rh98"])
    lat0 = P.lat.min() - 0.01
    g = g[(g.lat > lat0) & (g.lat < P.lat.max() + 0.01)]
    pairs = []
    # esquina superior izquierda de cada ventana en la grilla nativa (EPSG:4326)
    x0 = -180 + np.floor((P.lon + 180) / dx) * dx - C * dx
    y0 = 84 + np.floor((P.lat - 84) / dy) * dy - C * dy
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(g[["lat", "lon"]].to_numpy()), metric="haversine")
    ind = tree.query_radius(np.radians(P[["lat", "lon"]].to_numpy()), 150 / 6_371_000.0)
    for k, ii in enumerate(ind):
        if not len(ii):
            continue
        gg = g.iloc[ii]
        col = np.floor((gg.lon.to_numpy() - x0.iloc[k]) / dx).astype(int)
        row = np.floor((gg.lat.to_numpy() - y0.iloc[k]) / dy).astype(int)
        ok = (col >= 0) & (col < WIN) & (row >= 0) & (row < WIN)
        for rh, r_, c_ in zip(gg.rh98.to_numpy()[ok], row[ok], col[ok]):
            pairs.append((P[ID].iloc[k], rh, w[k, 0, r_, c_], w[k, 1, r_, c_]))
    V = pd.DataFrame(pairs, columns=[ID, "rh98", "eth_h", "eth_sd"]).dropna()
    V = V.merge(P[[ID, "zona", "estrato"]], on=ID)
    V["dif"] = V.eth_h - V.rh98

    def summ(d):
        return pd.Series({"n_footprints": len(d), "n_parcelas": d[ID].nunique(),
                          "rh98_mediana": d.rh98.median(), "eth_mediana": d.eth_h.median(),
                          "sesgo_mediano_eth_menos_rh98": d.dif.median(),
                          "error_abs_mediano": d.dif.abs().median(),
                          "spearman": d.rh98.corr(d.eth_h, method="spearman")})
    GV = pd.concat([V.groupby(["zona", "estrato"]).apply(summ).reset_index(),
                    V.groupby("estrato").apply(summ).reset_index().assign(zona="todas"),
                    summ(V).to_frame().T.assign(zona="todas", estrato="todos")], ignore_index=True)
    GV.to_csv(ROOT / "results/tables/eth_gedi_validacion.csv", index=False)

    pd.set_option("display.width", 220)
    print(D1.round(3).to_string())
    print(D2.round(2).to_string())
    print(GV.round(2).to_string())
    print(P.eth_desfase_anios.describe().round(1).to_string())


if __name__ == "__main__":
    main()
