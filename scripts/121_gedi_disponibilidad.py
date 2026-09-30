#!/usr/bin/env python3
"""GEDI, medición previa A1: ¿la disponibilidad de footprints es un proxy de geografía?

Para cada parcela: distancia al footprint L2A más cercano (QC de clean_gedi.py: degrade 0,
quality 1, sensitivity > 0,9; año 2022) y número de footprints a 100, 250 y 500 m. Después,
rho de Spearman de esas cifras contra latitud y elevación, y el R² (CV por bloques, los mismos
folds de siempre) de un RF que predice latitud solo desde ellas. Si predicen latitud, un
indicador de disponibilidad sería meter coordenadas por la puerta de atrás.

También el desfase temporal: años entre el censo (Year) y la pasada GEDI (2022).

Usa todos los footprints con QC, no los filtrados por cobertura de suelo (unified/), porque
ese filtro es a su vez una selección geográfica.

Escribe data/derived/gedi_disponibilidad.parquet y results/tables/gedi_a1_disponibilidad.csv.

Uso:
    python scripts/121_gedi_disponibilidad.py
    python scripts/121_gedi_disponibilidad.py --gedi /mnt/rapidita_4T/datos/GEDI/clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
R_EARTH = 6_371_000.0
RADII = (100, 250, 500)
GEDI = "/mnt/rapidita_4T/datos/GEDI/clean"


def load_gedi(root: str, product: str, cols: list[str]) -> pd.DataFrame:
    """Footprints con QC de clean_gedi.py, un parquet por gránulo; la fecha sale del nombre
    (AAAADDD), porque las tablas no la traen."""
    import re
    parts = []
    for f in sorted(Path(root, product).glob("*.parquet")):
        d = pd.read_parquet(f, columns=cols)
        m = re.search(r"_(\d{7})\d{6}_O(\d+)_", f.name)
        d["date"] = pd.to_datetime(m.group(1), format="%Y%j")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gedi", default=GEDI)
    a = ap.parse_args()
    g = load_gedi(a.gedi, "L2A", ["lat", "lon"])
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[
        [ID, "lat", "lon", "Year", "source", "elevation"]]
    tree = BallTree(np.radians(g[["lat", "lon"]].to_numpy()), metric="haversine")
    P = np.radians(plots[["lat", "lon"]].to_numpy())
    dist, _ = tree.query(P, k=1)
    out = plots.copy()
    out["gedi_dist_m"] = dist[:, 0] * R_EARTH
    for r in RADII:
        out[f"gedi_n{r}"] = tree.query_radius(P, r / R_EARTH, count_only=True)
    out["gedi_lag_anios"] = 2022 - out.Year
    out.drop(columns=["lat", "lon", "Year", "source", "elevation"]).to_parquet(
        ROOT / "data/derived/gedi_disponibilidad.parquet", index=False)

    cols = ["gedi_dist_m"] + [f"gedi_n{r}" for r in RADII]
    rows = []
    for c in cols:
        for v in ("lat", "elevation"):
            m = out[[c, v]].dropna()
            rho = spearmanr(m[c], m[v]).statistic
            rows.append(dict(medida="spearman", variable=c, contra=v, valor=rho, n=len(m)))
        for s, h in out.groupby("source"):
            rows.append(dict(medida="spearman_lat_por_fuente", variable=c, contra=s,
                             valor=spearmanr(h[c], h.lat).statistic, n=len(h)))
    # RF que predice latitud solo desde la disponibilidad, con los folds por bloques
    folds = pd.read_parquet(ROOT / "data/derived/cv_folds_unified_block20.parquet")
    te_fold = folds[folds.split == "test"].set_index(ID).fold
    d = out.assign(fold=out[ID].map(te_fold)).reset_index(drop=True)
    pred = np.full(len(d), np.nan)
    for f in sorted(d.fold.dropna().unique()):
        te = (d.fold == f).to_numpy()
        rf = RandomForestRegressor(500, min_samples_leaf=5, n_jobs=-1, random_state=0)
        rf.fit(d.loc[~te, cols], d.loc[~te, "lat"])
        pred[te] = rf.predict(d.loc[te, cols])
    ok = ~np.isnan(pred)
    r2 = 1 - ((d.lat[ok] - pred[ok]) ** 2).sum() / ((d.lat[ok] - d.lat[ok].mean()) ** 2).sum()
    rows.append(dict(medida="R2_rf_lat_cv_bloques", variable="+".join(cols), contra="lat",
                     valor=r2, n=int(ok.sum())))
    for c in cols + ["gedi_lag_anios"]:
        q = out[c].describe(percentiles=[.1, .25, .5, .75, .9])
        for k, v in q.items():
            rows.append(dict(medida=f"resumen_{k}", variable=c, contra="todas", valor=v, n=len(out)))
    for r in RADII:
        rows.append(dict(medida="frac_con_footprint", variable=f"gedi_n{r}", contra="todas",
                         valor=(out[f"gedi_n{r}"] > 0).mean(), n=len(out)))
        for s, h in out.groupby("source"):
            rows.append(dict(medida="frac_con_footprint", variable=f"gedi_n{r}", contra=s,
                             valor=(h[f"gedi_n{r}"] > 0).mean(), n=len(h)))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/tables/gedi_a1_disponibilidad.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[~res.medida.str.startswith("resumen")].round(3).to_string())
    print(out[cols + ["gedi_lag_anios"]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
