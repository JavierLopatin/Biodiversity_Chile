#!/usr/bin/env python3
"""GEDI, medición previa A2: ¿a qué distancia deja de valer un footprint?

Semivariograma empírico de la altura de dosel (rh98, L2A 2022 con QC) sobre los footprints
mismos, por zona latitudinal, y ajuste de un modelo exponencial con pepita
gamma(h) = c0 + c (1 - exp(-3h / a)); `a` es el rango práctico (95 % de la meseta). Ese rango
fija el radio de agregación alrededor de cada parcela.

Solo footprints a menos de 5 km de alguna parcela: el paisaje que importa es el de las
parcelas, no el desierto del norte, que dominaría cualquier semivariograma nacional. Pares
por vecindad: para una muestra de anclas por zona, todos los footprints a menos de 3 km
(BallTree haversine). Las distancias bajo ~60 m (espaciado a lo largo de la traza) solo salen
de cruces de trazas y tienen pocos pares; se reporta n por bin.

Escribe results/tables/gedi_a2_semivariograma.csv (empírico) y gedi_a2_rango.csv (ajuste por zona).

Uso:
    python scripts/122_gedi_semivariograma.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
R_EARTH = 6_371_000.0
_s = importlib.util.spec_from_file_location("g121", ROOT / "scripts" / "121_gedi_disponibilidad.py")
g121 = importlib.util.module_from_spec(_s); _s.loader.exec_module(g121)
ZONES = [(-30, -17, "norte_>30S"), (-35, -30, "30-35S"), (-38, -35, "35-38S"),
         (-42, -38, "38-42S"), (-47, -42, "42-47S"), (-56, -47, "sur_<47S")]
BINS = np.r_[0, 30, 60, 90, 120, 150, 200, 250, 300, 400, 500, 650, 800, 1000, 1300, 1600,
             2000, 2500, 3000]


def expo(h, c0, c, a):
    return c0 + c * (1 - np.exp(-3 * h / a))


def expo2(h, c0, c1, a1, c2, a2):
    """Anidado: estructura corta (rodal) más larga (paisaje). Con una sola exponencial el
    rango sale en 1,7-3,3 km porque lo arrastra la tendencia de paisaje, y no dice nada del
    radio útil alrededor de una parcela."""
    return c0 + c1 * (1 - np.exp(-3 * h / a1)) + c2 * (1 - np.exp(-3 * h / a2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gedi", default=g121.GEDI)
    ap.add_argument("--anchors", type=int, default=20000)
    a = ap.parse_args()
    g = g121.load_gedi(a.gedi, "L2A", ["lat", "lon", "rh98"])
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[["lat", "lon"]]
    pt = BallTree(np.radians(plots.to_numpy()), metric="haversine")
    d0, _ = pt.query(np.radians(g[["lat", "lon"]].to_numpy()), k=1)
    g = g[d0[:, 0] * R_EARTH < 5000].reset_index(drop=True)
    print(f"footprints a < 5 km de una parcela: {len(g):,}")
    rng = np.random.default_rng(0)
    emp, fits = [], []
    for lo, hi, name in ZONES:
        z = g[(g.lat >= lo) & (g.lat < hi)].reset_index(drop=True)
        if len(z) < 1000:
            continue
        X = np.radians(z[["lat", "lon"]].to_numpy())
        tree = BallTree(X, metric="haversine")
        anc = rng.choice(len(z), min(a.anchors, len(z)), replace=False)
        ind, dist = tree.query_radius(X[anc], BINS[-1] / R_EARTH, return_distance=True)
        h = np.concatenate([dd * R_EARTH for dd in dist])
        zi = np.repeat(z.rh98.to_numpy()[anc], [len(i) for i in ind])
        zj = z.rh98.to_numpy()[np.concatenate(ind)]
        keep = h > 0
        h, sq = h[keep], 0.5 * (zi[keep] - zj[keep]) ** 2
        b = np.digitize(h, BINS) - 1
        e = pd.DataFrame({"b": b, "h": h, "g": sq}).groupby("b").agg(
            h=("h", "mean"), gamma=("g", "mean"), n=("g", "size")).reset_index()
        e = e[(e.b >= 0) & (e.b < len(BINS) - 1)]
        e["zona"], e["var_total"], e["n_footprints"] = name, z.rh98.var(), len(z)
        e["rh98_media"] = z.rh98.mean()
        emp.append(e)
        w = e[e.n >= 100]
        try:
            p, _ = curve_fit(expo, w.h, w.gamma, p0=[w.gamma.iloc[0], w.gamma.max(), 300],
                             sigma=1 / np.sqrt(w.n), bounds=([0, 0, 10], [np.inf, np.inf, 20000]))
        except RuntimeError:
            p = [np.nan] * 3
        w1 = e[(e.n >= 100) & (e.b >= 1)]
        try:
            q, _ = curve_fit(expo2, w1.h, w1.gamma, p0=[20, 20, 200, 20, 3000],
                             sigma=1 / np.sqrt(w1.n), maxfev=20000,
                             bounds=([0, 0, 20, 0, 500], [np.inf, np.inf, 1500, np.inf, 50000]))
        except RuntimeError:
            q = [np.nan] * 5
        fits.append(dict(anidado_pepita=q[0], anidado_c_corto=q[1], anidado_rango_corto_m=q[2],
                         anidado_c_largo=q[3], anidado_rango_largo_m=q[4], zona=name, n_footprints=len(z), rh98_media=z.rh98.mean(),
                         pepita=p[0], meseta_parcial=p[1], rango_practico_m=p[2],
                         pepita_frac=p[0] / (p[0] + p[1]), var_total=z.rh98.var(),
                         gamma_30_60=e.loc[e.b == 1, "gamma"].squeeze() if (e.b == 1).any() else np.nan))
    E = pd.concat(emp)
    F = pd.DataFrame(fits)
    E.to_csv(ROOT / "results/tables/gedi_a2_semivariograma.csv", index=False)
    F.to_csv(ROOT / "results/tables/gedi_a2_rango.csv", index=False)
    pd.set_option("display.width", 220)
    print(F.round(2).to_string())
    print(E.pivot(index="b", columns="zona", values="gamma").round(1).assign(
        h=BINS[:-1][E.pivot(index="b", columns="zona", values="gamma").index]).to_string())
    print(E.pivot(index="b", columns="zona", values="n").to_string())


if __name__ == "__main__":
    main()
