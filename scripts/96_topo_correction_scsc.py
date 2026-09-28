#!/usr/bin/env python3
"""Corrección topográfica SCS+C (Soenen et al. 2005) de la reflectancia del píxel central,
con tres estimadores de C comparados lado a lado.

    IL       = cos(θs)·cos(β) + sin(θs)·sin(β)·cos(φs − α)
    ρ_corr   = ρ · (cos(β)·cos(θs) + C) / (IL + C)
    C        = b / m,   de   ρ = m·IL + b

θs, φs: cenit y acimut solar; β, α: pendiente y aspect del DEM. Todo en grados, con
acimut y aspect desde el norte en sentido horario (scripts/95).

EL PROBLEMA CON EL C CLÁSICO. C se estima normalmente con una regresión transversal: muchos
píxeles de una escena. Ahí la iluminación está confundida con la vegetación, porque la
orientación cambia la vegetación: en Chile mediterráneo las laderas sur (IL bajo en
invierno) son más húmedas y verdes. Eso atenúa m. Tener la escena completa no lo arregla.

ESTIMADOR TEMPORAL (fe). m se identifica con la vegetación fija:
    ρ_{p,t} = a_p + g_{pasada(t)} + m·IL_{p,t} + e
con efectos fijos de parcela y de pasada orbital, por (sensor, banda, clase MapBiomas nivel
2). Lo que queda moviendo IL es el sol recorriendo el año sobre un terreno que no cambia.
El intercepto consistente con ese m, b = media(ρ − m·IL), se calcula por grupo pasada ×
clase, con caída a pasada y luego a sensor × clase × estación.

EL SOL BAJO ROMPE EL MODELO LAMBERTIANO. Estimado sobre todas las observaciones, la m
temporal sale con m/ρ̄ ≈ 1,8-2,1 en bosque, por encima del 1/IL̄ ≈ 1,53 de un coseno puro.
La reflectancia cae más que proporcionalmente con IL, y C sale negativo en 27-76% de las
observaciones según la banda. Es parejo en rojo y NIR (bosque L8: 2,05 y 2,01), así que no
es fenología, que los movería en sentidos opuestos. La causa probable es la sombra
proyectada por el relieve vecino con sol bajo, que IL (solo sombra propia) no modela.
Recortando a cenit ≤55° e IL >0,5, m/ρ̄ cae bajo el coseno y C pasa a positivo. Por eso:

- REGIMEN_EST: las m y b de fe y cs se estiman solo con cenit ≤ 55° e IL > 0,5.
- REGIMEN_APP: la corrección se aplica con cenit ≤ 65° e IL > 0,3. El resto se CONSERVA sin
  corregir, con bandera low_sun. No se descarta: sacar el sol bajo elimina el invierno
  preferentemente en latitud alta, y la curva perdería su ancla invernal en el sur.

Estimadores (sufijo de las columnas de salida):
  fe     temporal con efectos fijos, en el régimen restringido.        [principal]
  cs     clásico transversal por pasada × clase, en el régimen restringido.
  csall  clásico transversal por pasada × clase, sobre todas las obs con IL > 0 y
         aplicado a todas: la práctica habitual, sin recorte geométrico.
fe contra cs mide el estimador en la misma geometría; cs contra csall mide la geometría.

Guardas, sin imputar nada. Banderas: m_nonpos (m ≤ 0 o sin estimar), c_range (C fuera de
[0, 10]), self_shadow (IL ≤ 0), low_sun (fuera del régimen de aplicación), sin_grupo_b.
β < 0,5°: aspect indefinido; la corrección vale 1 exacto (IL = cos θs), y se verifica.

Salida: data/derived/obs_center_unified_tc.parquet (bandas e índices originales _nc y
corregidos _tc<est>, con C, nivel y bandera por banda y estimador) y
results/topo_correction/ (m por grupo, C por banda y estimador, diagnóstico de corr(ρ, IL)
dentro de pasada por clase de pendiente).

Uso:
    python scripts/96_topo_correction_scsc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT_DIR = ROOT / "results" / "topo_correction"
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
ESTIMATORS = ["fe", "cs", "csall"]
ID = "PlotObservationID"
MIN_PLOTS, MIN_IL_RANGE, C_MAX, FLAT = 20, 0.15, 10.0, 0.5
EST_ZEN, EST_IL = 55.0, 0.5
APP_ZEN, APP_IL = 65.0, 0.3
SAVI_L = 0.5


def indices(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Las fórmulas de biodiv.cube.to_indices, con sus clips."""
    blue, red, nir, swir2 = b["blue"], b["red"], b["nir"], b["swir2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.clip((nir - red) / (nir + red), -1, 1)
        evi = np.clip(2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0), -1, 1)
        nbr = np.clip((nir - swir2) / (nir + swir2), -1, 1)
        savi = np.clip(((nir - red) / (nir + red + SAVI_L)) * (1.0 + SAVI_L), -1.5, 1.5)
    return {"ndvi": ndvi, "evi": evi, "kndvi": np.tanh(ndvi ** 2), "nbr": nbr, "savi": savi}


def twoway_demean(y, x, g1, g2, tol: float = 1e-10, max_iter: int = 200):
    """Proyecciones alternadas para dos efectos fijos."""
    y, x = np.asarray(y, float).copy(), np.asarray(x, float).copy()
    for _ in range(max_iter):
        dmax = 0.0
        for g in (g1, g2):
            my = pd.Series(y).groupby(g).transform("mean").to_numpy()
            mx = pd.Series(x).groupby(g).transform("mean").to_numpy()
            y -= my; x -= mx
            dmax = max(dmax, np.abs(my).max(), np.abs(mx).max())
        if dmax < tol:
            break
    return y, x


def season(month: pd.Series) -> pd.Series:
    return month.map({12: "DEF", 1: "DEF", 2: "DEF", 3: "MAM", 4: "MAM", 5: "MAM",
                      6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})


LEVELS = (("pasada_clase", ["pass_id", "cls"]), ("pasada", ["pass_id"]),
          ("sensor_clase_estacion", ["sensor", "cls", "season"]))


def c_fe(o: pd.DataFrame, rho: np.ndarray, est: np.ndarray, m_rows: list, band: str):
    """C por observación con m de efectos fijos y b por grupo jerárquico."""
    m_obs = np.full(len(o), np.nan)
    for (sen, cls), gi in o[est].groupby(["sensor", "cls"]).groups.items():
        gi = np.asarray(gi)
        n_p = o.loc[gi, ID].nunique()
        if n_p < MIN_PLOTS:
            continue
        yd, xd = twoway_demean(rho[gi], o.IL.to_numpy()[gi], o.loc[gi, ID].to_numpy(),
                               o.loc[gi, "pass_id"].to_numpy())
        m = float((xd * yd).sum() / (xd * xd).sum())
        rho_bar, il_bar = float(rho[gi].mean()), float(o.IL.to_numpy()[gi].mean())
        m_rows.append(dict(band=band, sensor=sen, cls=cls, m=m, k=m / rho_bar,
                           k_coseno=1 / il_bar, n=len(gi), parcelas=n_p))
        if m > 0:
            m_obs[(o.sensor == sen).to_numpy() & (o.cls == cls).to_numpy()] = m
    resid = rho - m_obs * o.IL.to_numpy()
    C = np.full(len(o), np.nan); level = np.full(len(o), "", dtype=object)
    base = est & np.isfinite(m_obs)
    for lvl, keys in LEVELS:
        d = o.loc[base, keys].copy()
        d["r"], d["IL"], d["pid"] = resid[base], o.IL.to_numpy()[base], o.loc[base, ID]
        st = d.groupby(keys).agg(b=("r", "mean"), n_p=("pid", "nunique"),
                                 q95=("IL", lambda s: s.quantile(.95)),
                                 q05=("IL", lambda s: s.quantile(.05)))
        good = st[(st.n_p >= MIN_PLOTS) & ((st.q95 - st.q05) >= MIN_IL_RANGE)].b
        todo = np.isfinite(m_obs) & (level == "")
        j = o.loc[todo, keys].join(good, on=keys)
        hit = j.b.notna().to_numpy(); idx = np.flatnonzero(todo)[hit]
        C[idx] = j.b.to_numpy()[hit] / m_obs[idx]; level[idx] = lvl
    return C, level, np.isfinite(m_obs)


def c_cs(o: pd.DataFrame, rho: np.ndarray, est: np.ndarray):
    """C clásico: OLS ρ ~ IL transversal dentro de cada grupo, jerárquico."""
    C = np.full(len(o), np.nan); level = np.full(len(o), "", dtype=object)
    for lvl, keys in LEVELS:
        d = o.loc[est, keys].copy()
        d["rho"], d["IL"], d["pid"] = rho[est], o.IL.to_numpy()[est], o.loc[est, ID]

        def fit(g):
            if g.pid.nunique() < MIN_PLOTS or (g.IL.quantile(.95) - g.IL.quantile(.05)) < MIN_IL_RANGE:
                return np.nan
            m, b = np.polyfit(g.IL.to_numpy(), g.rho.to_numpy(), 1)
            return b / m if m > 0 else np.nan
        cg = d.groupby(keys).apply(fit).rename("C").dropna()
        todo = level == ""
        j = o.loc[todo, keys].join(cg, on=keys)
        hit = j.C.notna().to_numpy(); idx = np.flatnonzero(todo)[hit]
        C[idx] = j.C.to_numpy()[hit]; level[idx] = lvl
    return C, level, np.ones(len(o), bool)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    o = pd.read_parquet(DERIVED / "obs_center_unified.parquet")
    o["cls"] = np.where(o.mb_native.fillna(False), o.mb_level2.fillna("otra"), "no_nativa")
    o["season"] = season(o.time.dt.month)
    zs, sl = np.radians(o.sun_zenith.to_numpy()), np.radians(o.slope.to_numpy())
    asp = np.radians(np.nan_to_num(o.aspect.to_numpy()))
    o["IL"] = np.cos(zs) * np.cos(sl) + np.sin(zs) * np.sin(sl) * np.cos(
        np.radians(o.sun_azimuth.to_numpy()) - asp)
    cos_ref = np.cos(sl) * np.cos(zs)
    IL = o.IL.to_numpy(); zen = o.sun_zenith.to_numpy()
    flat = o.slope.to_numpy() < FLAT
    o["slope_cls"] = pd.cut(o.slope, [-0.01, 5, 15, 25, 90], labels=["<5", "5-15", "15-25", ">25"])

    regimes = {"fe": ((zen <= EST_ZEN) & (IL > EST_IL), (zen <= APP_ZEN) & (IL > APP_IL)),
               "cs": ((zen <= EST_ZEN) & (IL > EST_IL), (zen <= APP_ZEN) & (IL > APP_IL)),
               "csall": (IL > 0, IL > 0)}
    m_rows, c_rows = [], []
    for b in BANDS:
        rho = o[b].to_numpy()
        valid = np.isfinite(rho) & (rho > 0) & (rho <= 1.5)
        for e in ESTIMATORS:
            est, app = regimes[e]
            est = est & valid
            C, level, has_m = (c_fe(o, rho, est, m_rows, b) if e == "fe" else c_cs(o, rho, est))
            flag = np.full(len(o), "", dtype=object)
            flag[valid & ~has_m] = "m_nonpos"
            flag[valid & has_m & ~np.isfinite(C)] = "sin_grupo_b"
            bad_c = np.isfinite(C) & ~((C >= 0) & (C <= C_MAX))
            flag[valid & bad_c] = "c_range"
            flag[valid & ~app] = "low_sun"
            flag[valid & (IL <= 0)] = "self_shadow"
            do = valid & app & np.isfinite(C) & ~bad_c
            factor = np.ones(len(o))
            factor[do] = (cos_ref[do] + C[do]) / (IL[do] + C[do])
            factor[flat] = 1.0
            flag[flat & valid] = "llano"
            o[f"{b}_tc{e}"] = np.where(np.isfinite(rho), rho * factor, np.nan)
            o[f"C_{b}_{e}"], o[f"flag_{b}_{e}"] = C, flag
            cc = C[do]
            c_rows.append(dict(band=b, est=e, obs_corregidas=float(do[valid].mean()),
                               C_mediana=float(np.median(cc)) if cc.size else np.nan,
                               C_p10=float(np.quantile(cc, .1)) if cc.size else np.nan,
                               C_p90=float(np.quantile(cc, .9)) if cc.size else np.nan,
                               frac_C_neg=float((C[valid & np.isfinite(C)] < 0).mean())))

    z0 = o.slope.to_numpy() == 0
    if z0.any():
        f0 = (cos_ref[z0] + 1.0) / (IL[z0] + 1.0)
        assert np.allclose(f0, 1.0, atol=1e-12), "la corrección no vale 1 en pendiente 0"

    nc = indices({b: o[b].to_numpy() for b in BANDS})
    for i in INDICES:
        o[f"{i}_nc"] = nc[i]
    for e in ESTIMATORS:
        tc = indices({b: o[f"{b}_tc{e}"].to_numpy() for b in BANDS})
        for i in INDICES:
            o[f"{i}_tc{e}"] = tc[i]

    diag = []
    for b in ["red", "nir", "swir1"]:
        for sc, g in o.groupby("slope_cls", observed=True):
            for col, v in [(b, "nc")] + [(f"{b}_tc{e}", e) for e in ESTIMATORS]:
                x = g[[col, "IL", "pass_id"]].dropna()
                x = x[x.groupby("pass_id")[col].transform("size") >= 10]
                yd = x[col] - x.groupby("pass_id")[col].transform("mean")
                xd = x.IL - x.groupby("pass_id").IL.transform("mean")
                diag.append(dict(band=b, slope_cls=str(sc), version=v,
                                 r_dentro_pasada=float(np.corrcoef(yd, xd)[0, 1])))
    mt, ct, dg = pd.DataFrame(m_rows), pd.DataFrame(c_rows), pd.DataFrame(diag)
    mt.to_csv(OUT_DIR / "m_fe_by_group.csv", index=False)
    ct.to_csv(OUT_DIR / "c_by_estimator.csv", index=False)
    dg.to_csv(OUT_DIR / "diag_r_il.csv", index=False)
    print("== m temporal (fe, régimen restringido): k = m/ρ̄ contra 1/IL̄ ==")
    print(mt.pivot_table(index=["sensor", "cls"], columns="band", values="k").round(2).to_string())
    print("\n== C por estimador ==\n", ct.round(3).to_string(index=False))
    print("\n== corr(ρ, IL) dentro de pasada ==")
    print(dg.pivot_table(index=["band", "slope_cls"], columns="version",
                         values="r_dentro_pasada")[["nc"] + ESTIMATORS].round(3).to_string())

    keep = [ID, "source", "time", "sensor", "pass_id", "cls", "slope", "aspect",
            "sun_zenith", "sun_azimuth", "IL", "slope_cls"] + BANDS + \
        [f"{b}_tc{e}" for b in BANDS for e in ESTIMATORS] + \
        [f"{i}_nc" for i in INDICES] + [f"{i}_tc{e}" for i in INDICES for e in ESTIMATORS] + \
        [f"{p}_{b}_{e}" for b in BANDS for e in ESTIMATORS for p in ("C", "flag")]
    f = DERIVED / "obs_center_unified_tc.parquet"
    o[keep].to_parquet(f, index=False)
    print(f"-> {f}")


if __name__ == "__main__":
    main()
