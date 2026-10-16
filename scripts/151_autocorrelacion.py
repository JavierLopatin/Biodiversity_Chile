#!/usr/bin/env python3
"""Hasta donde llega la senal: el rango de autocorrelacion contra el que medir los 20 km.

Ya sabiamos cuanto SEPARAN los bloques (`141_vecindad_block_cv.py`: mediana 8,9 km entre una
parcela de test y su vecino de entrenamiento, 59 % por debajo de 10 km). Lo que faltaba es
contra que comparar ese numero. Este script lo mide.

POR QUE EL VARIOGRAMA DE LA RESPUESTA CRUDA NO SIRVE
----------------------------------------------------
El pool cubre 25 grados de latitud y las respuestas tienen un gradiente a esa escala. Un
variograma sobre la respuesta cruda no mide autocorrelacion sino TENDENCIA: la semivarianza
crece monotonamente hasta el largo del pais y el "rango" ajustado sale del orden de 1.000 km,
que diria que ningun bloque realizable descorrelaciona nada. Es el mismo problema que separa
el R2 agrupado del centrado dentro de bandas, visto en el segundo momento en vez del primero.

El variograma que dimensiona un bloque es el de los RESIDUOS DE LA LINEA BASE, porque lo que
el bloque tiene que descorrelacionar es lo que el modelo puede explotar, y la linea base de
ubicacion (lon, lat, elevacion) es exactamente el modelo que explota posicion. Se reportan las
tres versiones y la comparacion entre ellas es el resultado, no un control:

    cruda        tendencia + autocorrelacion   -> rango inutil para dimensionar
    centrada     dentro de bandas de 2 grados  -> la metrica del texto principal
    residual     fuera de muestra del nulo     -> el rango que decide el tamano de bloque

TAMBIEN EL rSAC DE LOS MODELOS
------------------------------
Bouchard et al. (2024) y Sabatini et al. (2022) reportan "sin patron en los residuos" como
diagnostico; Tordoni et al. (2024) lo omite. Aqui se reporta de una forma que dice algo mas:
la I de Moran por banda de distancia para los residuos del NULO DE UBICACION y para los del
DISENO. Si el nulo deja autocorrelacion residual y el diseno no, eso no es un chequeo de
cajita --es el mecanismo del paper medido en los residuos.

AREA PARA CLARK-EVANS
---------------------
`biodiv.spatial_diag.sampling_pattern` toma por omision la caja envolvente del DOMINIO de
prediccion, y esa eleccion es correcta cuando hay mapa. Aqui no la podemos usar: el pool no
etiquetado se extrajo para `plots_subset` y cubre la mitad norte de la extension del pool
unificado, asi que su caja subestima el area y infla R (da 0,678 contra 0,144 del pool viejo,
y la diferencia es el area, no el patron). Como este paper no publica mapa, el dominio no esta
definido, y el denominador honesto es la ENVOLVENTE CONVEXA DE LAS PROPIAS PARCELAS. Queda
declarado aqui porque cambia el numero que se cita.

Salidas:
    results/tables/autocorrelacion_rango.csv   rango por respuesta y version
    results/tables/autocorrelacion_variograma.csv   el variograma empirico, por si se grafica
    results/tables/rsac_moran.csv              I de Moran por banda, por modelo
    results/tables/patron_muestreo_paper.json  Clark-Evans y Gj del pool del paper
    results/figures/figS11_autocorrelacion.{pdf,png}

Uso:
    python scripts/151_autocorrelacion.py [--jobs 4] [--max-km 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, cKDTree
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "src"))
from biodiv import cv as cvmod                   # noqa: E402
from biodiv import facetas_paper as FP           # noqa: E402
from biodiv import figures as fg                 # noqa: E402

#: Misma procedencia que `scripts/136`: la lista de facetas vive en `facetas_paper`, pero el
#: archivo de cada una no, porque dependen del pool leñoso con relleno.
FUENTE = {
    **{f: "unified_diversity_responses_woody.parquet" for f in
       ["hill_q0_unified", "lcbd_pa_unified", "lcbd_freq_unified",
        *[f"isomap{i}_{v}_unified" for i in (1, 2, 3) for v in ("pa", "freq")]]},
    "lcbd_count_sorensen": "lcbd_count_sorensen_unified_padded_woody.parquet",
    **{f"td_inext_q{q}": "td_inext_coverage_unified_padded_woody.parquet" for q in (0, 1, 2)},
    **{f"pd_inext_q{q}": "pd_inext_coverage_unified_padded_woody.parquet" for q in (0, 1, 2)},
    **{f: "unified_phylo_responses_woody.parquet" for f in
       ["mpd_unified", "mntd_unified", "ses_pd_unified"]},
}
NULO = ["lon", "lat", "elevation"]
BANDA_DEG = 2.0                                  # el centrado del texto principal


# ---------------------------------------------------------------------------- datos

def datos() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """El pool del paper con sus predictores, y los tres bloques como listas de columnas."""
    gm = pd.read_parquet(DERIVED / "gm_center_nc.parquet")
    cl = pd.read_parquet(DERIVED / "climate_unified.parquet")
    if cl.index.name:
        cl = cl.reset_index()
    tp = (pd.read_parquet(DERIVED / "topography_unified.parquet")
          .rename(columns={"plot_id": "PlotObservationID"}))
    pl = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lon", "lat", "X_m", "Y_m", "source"]]
    d = (gm.merge(cl, on="PlotObservationID").merge(tp, on="PlotObservationID")
         .merge(pl, on="PlotObservationID"))
    for f, src in FUENTE.items():
        t = pd.read_parquet(DERIVED / src)
        if f in t.columns:
            d = d.merge(t[["PlotObservationID", f]], on="PlotObservationID", how="left")

    num = lambda cols: [c for c in cols if c in d.columns  # noqa: E731
                        and pd.api.types.is_numeric_dtype(d[c])]
    reserv = {"PlotObservationID", "X_m", "Y_m", "source", "lon", "lat", *FUENTE}
    gm_c = num([c for c in gm.columns if c not in reserv])
    tp_c = num([c for c in tp.columns if c not in reserv])
    cl_c = num([c for c in cl.columns if c not in reserv])
    bloques = {
        "ubicacion": NULO,                       # el nulo del paper
        "reflectancia": gm_c + tp_c,             # RFG4c: geomediana + topografia
        "diseno": gm_c + tp_c + cl_c,            # RFG5c: y clima
    }
    return d.reset_index(drop=True), bloques


def oof(X: np.ndarray, y: np.ndarray, folds: pd.DataFrame, ids: pd.Index,
        jobs: int) -> np.ndarray:
    """Prediccion fuera de muestra con los folds del paper, no con una particion nueva.

    Importa que sean los del paper (`cv_folds_unified_block20.parquet`): el rango que se va a
    comparar contra 20 km tiene que salir de los residuos del mismo modelo que produjo los
    margenes, o estariamos dimensionando el bloque de otro experimento.
    """
    pos = pd.Series(np.arange(len(ids)), index=ids)
    p = np.full(len(y), np.nan)
    for _, _, tr, te in cvmod.iter_folds(folds, "kfold5_block20_unified"):
        itr = pos.reindex(tr).dropna().astype(int).to_numpy()
        ite = pos.reindex(te).dropna().astype(int).to_numpy()
        itr = itr[np.isfinite(y[itr])]
        ite = ite[np.isfinite(y[ite])]
        if len(ite) < 5 or len(itr) < 50:
            continue
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33,
                                  n_jobs=jobs, random_state=0).fit(X[itr], y[itr])
        p[ite] = m.predict(X[ite])
    return p


# ------------------------------------------------------------------- variograma

def bins_km(max_km: float, n: int = 18) -> np.ndarray:
    """Bordes log-espaciados de 1 km a `max_km`.

    Logaritmicos y no regulares porque lo que se quiere resolver esta abajo: entre 1 y 30 km
    es donde cae el tamano de bloque, y un binning regular hasta 200 km pondria tres puntos
    ahi y quince en la cola donde no se decide nada.
    """
    return np.unique(np.r_[0.0, np.geomspace(1.0, max_km, n)])


def variograma(xy_km: np.ndarray, z: np.ndarray, bordes: np.ndarray) -> pd.DataFrame:
    """Semivariograma empirico: gamma(h) = media de (z_i - z_j)^2 / 2 en cada banda."""
    ok = np.isfinite(z)
    xy_km, z = xy_km[ok], z[ok]
    i, j = np.triu_indices(len(z), k=1)
    h = np.hypot(*(xy_km[i] - xy_km[j]).T)
    dz2 = (z[i] - z[j]) ** 2
    k = np.digitize(h, bordes) - 1
    val = (bordes[:-1] + bordes[1:]) / 2
    rows = []
    for b in range(len(bordes) - 1):
        m = k == b
        if m.sum() < 30:                         # bandas con pocos pares no se reportan
            continue
        rows.append(dict(h_km=float(h[m].mean()), h_centro_km=float(val[b]),
                         gamma=float(dz2[m].mean() / 2), n_pares=int(m.sum())))
    return pd.DataFrame(rows)


def _exp_nugget(h, c0, c, a):
    """gamma(h) = c0 + c (1 - exp(-3h/a)); `a` es el rango practico (95 % de la meseta)."""
    return c0 + c * (1.0 - np.exp(-3.0 * h / a))


def rango(vg: pd.DataFrame, max_km: float) -> dict:
    """Rango practico por tres caminos, porque el ajuste parametrico falla en silencio.

    El exponencial con pepita es el modelo habitual, pero cuando el variograma no tiene
    estructura --- pura pepita --- el parametro `a` no esta identificado y `curve_fit` devuelve
    el valor INICIAL sin avisar. Medido: en la primera corrida, los tres LCBD y TD q0 salieron
    con 35,7-35,8 km identicos en las cinco versiones, y 35,8 km es `h.mean()`, que es
    justamente el `p0` que se le paso. Un rango que no depende de los datos no es un rango.

    Asi que se reportan tres cosas y se declara cual usar:

    ``rango_km``        el ajuste, **solo si esta identificado** (ver abajo); si no, NaN.
    ``rango_emp_km``    sin supuesto de forma: la distancia a la que el variograma empirico
                        alcanza el 95 % de su propia meseta, interpolando entre bandas.
    ``estructura``      (meseta - pepita) / meseta, o sea cuanta de la varianza es espacial.
                        Con `estructura` baja no hay rango que medir y da igual el metodo.

    El ajuste se considera identificado si la estructura supera 0,15 y `a` se movio del inicio.
    """
    out = dict(rango_km=np.nan, rango_emp_km=np.nan, pepita=np.nan, meseta=np.nan,
               estructura=np.nan, acotado=False, ajuste_identificado=False)
    if len(vg) < 5:
        return out
    h, g = vg["h_km"].to_numpy(), vg["gamma"].to_numpy()

    # --- meseta y pepita empiricas: el tercio de distancias mayores contra la banda mas corta
    k = max(3, len(g) // 3)
    meseta, pepita = float(np.median(g[-k:])), float(g[0])
    out["meseta"], out["pepita"] = meseta, pepita
    out["estructura"] = float((meseta - pepita) / meseta) if meseta > 0 else np.nan

    if meseta > pepita:
        obj = pepita + 0.95 * (meseta - pepita)
        i = int(np.argmax(g >= obj)) if (g >= obj).any() else -1
        if i > 0:                                # interpolacion lineal dentro de la banda
            g0, g1, h0, h1 = g[i - 1], g[i], h[i - 1], h[i]
            frac = (obj - g0) / (g1 - g0) if g1 > g0 else 0.0
            out["rango_emp_km"] = float(h0 + frac * (h1 - h0))
        elif i == 0:
            out["rango_emp_km"] = float(h[0])
        else:
            out["rango_emp_km"] = np.nan
            out["acotado"] = True                # no alcanza su meseta dentro del alcance

    p0_a = max(h.mean(), 1.0)
    try:
        (c0, c, a), _ = curve_fit(_exp_nugget, h, g, p0=[pepita, max(meseta - pepita, 1e-12),
                                                         p0_a],
                                  maxfev=20000,
                                  bounds=([0, 0, 1e-3], [np.inf, np.inf, 5 * max_km]))
    except Exception:
        return out
    movio = abs(a - p0_a) > 0.01 * p0_a
    ident = bool((out["estructura"] or 0) > 0.15 and movio)
    out["ajuste_identificado"] = ident
    if ident:
        out["rango_km"] = float(min(a, max_km))
        out["acotado"] = bool(a > max_km)
    return out


def cruce_moran(mo: pd.DataFrame, umbral: float = 0.05) -> dict:
    """Distancia a la que la I de Moran por banda cae bajo `umbral`, e I a escala de bloque.

    Es el estimador mas directo de "hasta donde llega la senal" y no supone forma funcional
    ninguna. `I_10_30km` es el promedio en las bandas que caen dentro del rango del tamano de
    bloque del paper, que es el numero que de verdad hay que mirar.
    """
    if mo.empty:
        return dict(cruce_km=np.nan, I_10_30km=np.nan)
    h, I = mo["h_centro_km"].to_numpy(), mo["moran_I"].to_numpy()
    cruce = np.nan
    for i in range(1, len(I)):
        if I[i] < umbral <= I[i - 1]:
            frac = (I[i - 1] - umbral) / (I[i - 1] - I[i])
            cruce = float(h[i - 1] + frac * (h[i] - h[i - 1]))
            break
    if np.isnan(cruce) and I[0] < umbral:
        cruce = 0.0                              # sin autocorrelacion ya en la primera banda
    m = (h >= 10) & (h <= 30)
    return dict(cruce_km=cruce, I_10_30km=float(I[m].mean()) if m.any() else np.nan)


def moran_bandas(xy_km: np.ndarray, z: np.ndarray, bordes: np.ndarray) -> pd.DataFrame:
    """I de Moran con pesos binarios por banda de distancia.

    Sin supuesto de forma funcional, al contrario del variograma ajustado: dice en que bandas
    hay autocorrelacion sin pedirle a los datos que se parezcan a un exponencial.
    """
    ok = np.isfinite(z)
    xy_km, z = xy_km[ok], z[ok]
    n = len(z)
    zc = z - z.mean()
    denom = (zc ** 2).sum()
    i, j = np.triu_indices(n, k=1)
    h = np.hypot(*(xy_km[i] - xy_km[j]).T)
    prod = zc[i] * zc[j]
    k = np.digitize(h, bordes) - 1
    rows = []
    for b in range(len(bordes) - 1):
        m = k == b
        if m.sum() < 30:
            continue
        W = 2 * m.sum()                          # pesos simetricos: cada par cuenta dos veces
        rows.append(dict(h_centro_km=float((bordes[b] + bordes[b + 1]) / 2),
                         moran_I=float((n / W) * 2 * prod[m].sum() / denom),
                         esperado=-1.0 / (n - 1), n_pares=int(m.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- patron de muestreo

def patron(d: pd.DataFrame) -> dict:
    """Clark-Evans con el area de la envolvente convexa, y Gj, para el pool del paper."""
    xy = d[["X_m", "Y_m"]].to_numpy(float)
    hull = ConvexHull(xy)
    area_m2 = float(hull.volume)                 # en 2D, `volume` es el area
    dd, _ = cKDTree(xy).query(xy, k=2)
    gj = dd[:, 1]
    esperada = 0.5 / np.sqrt(len(xy) / area_m2)
    return dict(n=int(len(xy)), area_km2=round(area_m2 / 1e6, 1),
                clark_evans_R=round(float(gj.mean() / esperada), 3),
                Gj_p5_km=round(float(np.percentile(gj, 5)) / 1000, 3),
                Gj_p50_km=round(float(np.median(gj)) / 1000, 3),
                Gj_p95_km=round(float(np.percentile(gj, 95)) / 1000, 3))


# ---------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--max-km", type=float, default=200.0, dest="max_km")
    p.add_argument("--fig-dir", default="results/figures", dest="fig_dir")
    a = p.parse_args()

    d, bloques = datos()
    folds = cvmod.load_schemes(DERIVED / "cv_folds_unified_block20.parquet")
    ids = pd.Index(d["PlotObservationID"])
    xy_km = d[["X_m", "Y_m"]].to_numpy(float) / 1000.0
    bordes = bins_km(a.max_km)

    pat = patron(d)
    print(f"patron del pool del paper: Clark-Evans R = {pat['clark_evans_R']} "
          f"(envolvente convexa, {pat['area_km2']:,.0f} km2)")
    print(f"  Gj mediana {pat['Gj_p50_km']:.2f} km  (p5 {pat['Gj_p5_km']:.2f}, "
          f"p95 {pat['Gj_p95_km']:.2f})\n")
    (TAB / "patron_muestreo_paper.json").write_text(json.dumps(pat, indent=2))

    banda = np.floor(d["lat"].to_numpy() / BANDA_DEG)
    rangos, vgs, morans = [], [], []
    for fac in FP.ORDEN:
        if fac not in d.columns:
            print(f"  [falta] {fac}")
            continue
        y = d[fac].to_numpy(float)
        if np.isfinite(y).sum() < 200:
            print(f"  [pocas] {fac}: n={int(np.isfinite(y).sum())}")
            continue

        # centrada dentro de banda: la metrica del texto principal, en el segundo momento
        cen = y - pd.Series(y).groupby(banda).transform("mean").to_numpy()

        versiones = {"cruda": y, "centrada": cen}
        for nombre, cols in bloques.items():
            X = d[cols].to_numpy(float)
            fin = np.isfinite(X).all(axis=1)
            yy = np.where(fin, y, np.nan)
            pred = oof(np.nan_to_num(X), yy, folds, ids, a.jobs)
            versiones[f"residual_{nombre}"] = yy - pred

        for ver, z in versiones.items():
            vg = variograma(xy_km, z, bordes)
            r = rango(vg, a.max_km)
            rangos.append(dict(faceta=fac, version=ver, n=int(np.isfinite(z).sum()), **r))
            vgs.append(vg.assign(faceta=fac, version=ver))
            mo = moran_bandas(xy_km, z, bordes)
            morans.append(mo.assign(faceta=fac, version=ver))
            rangos[-1].update(cruce_moran(mo))
        print(f"  {FP.LAB.get(fac, fac):24s} "
              + "  ".join(f"{v}={rangos[-len(versiones) + i]['rango_emp_km']:6.1f}"
                          for i, v in enumerate(versiones)))

    R = pd.DataFrame(rangos)
    R.to_csv(TAB / "autocorrelacion_rango.csv", index=False)
    pd.concat(vgs, ignore_index=True).to_csv(TAB / "autocorrelacion_variograma.csv",
                                             index=False)
    M = pd.concat(morans, ignore_index=True)
    M.to_csv(TAB / "rsac_moran.csv", index=False)

    print("\nresumen por version (mediana sobre las 16 respuestas):")
    res = R.groupby("version").agg(
        rango_emp_km=("rango_emp_km", "median"),
        rango_ajuste_km=("rango_km", "median"),
        estructura=("estructura", "median"),
        cruce_moran_km=("cruce_km", "median"),
        I_10_30km=("I_10_30km", "median"),
        ajustes_ok=("ajuste_identificado", "sum"),
        sin_meseta=("acotado", "sum"))
    print(res.round(3).to_string())
    print("\n  rango_emp_km es el estimador a citar; rango_ajuste_km solo donde "
          "ajustes_ok lo respalda.")

    # --- figura: el variograma mediano por version, y la I de Moran por banda
    import matplotlib.pyplot as plt
    fg.set_paper_style()
    VERS = [("cruda", "#b2182b", "raw"),
            ("centrada", "#ef8a62", "within-band centred"),
            ("residual_reflectancia", "#f6c141", "residual, reflectance only"),
            ("residual_ubicacion", "#2166ac", "residual, location baseline"),
            ("residual_diseno", "#4d9221", "residual, full design")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    # Panel a: el rango por respuesta contra el tamano de bloque. Se dibuja este y no el
    # variograma mediano porque el variograma normalizado deja las cinco versiones casi
    # superpuestas --- la forma es la misma, lo que cambia es la escala--- y la decision que
    # el panel tiene que sostener es por respuesta: cuales exceden los 20 km.
    orden = [f for f in FP.ORDEN if f in set(R["faceta"])][::-1]
    y = np.arange(len(orden))
    for ver, col, lab in [v for v in VERS if v[0].startswith("residual")]:
        v = (R[R["version"] == ver].set_index("faceta")["rango_emp_km"].reindex(orden))
        ax1.scatter(v.to_numpy(), y, s=16, color=col, label=lab, zorder=3)
    ax1.axvline(20, color="0.3", ls=":", lw=1.0, zorder=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([FP.LAB.get(f, f) for f in orden], fontsize=6)
    ax1.set_xlabel("autocorrelation range (km)")
    ax1.set_xscale("log")
    ax1.legend(fontsize=5.6, loc="lower right")
    ax1.set_title("Range of what each baseline leaves", fontsize=8)
    ax1.annotate("20 km block", xy=(20, 0.01), xycoords=("data", "axes fraction"),
                 xytext=(3, 0), textcoords="offset points", fontsize=6, color="0.3",
                 rotation=90, va="bottom")

    # Panel b: la I de Moran por banda, que es el estimador sin supuesto de forma.
    for ver, col, lab in VERS:
        s_ = M[M["version"] == ver]
        if s_.empty:
            continue
        g = s_.groupby("h_centro_km")["moran_I"].median()
        ax2.plot(g.index, g.values, color=col, lw=1.6, label=lab)
    ax2.axhline(0, color="0.5", lw=0.8, zorder=1)
    ax2.axvline(20, color="0.3", ls=":", lw=1.0, zorder=1)
    ax2.annotate("20 km block", xy=(20, 0.02), xycoords=("data", "axes fraction"),
                 xytext=(3, 0), textcoords="offset points", fontsize=6, color="0.3",
                 rotation=90, va="bottom")
    ax2.set_xscale("log")
    ax2.set_xlabel("distance band (km)")
    ax2.set_ylabel("Moran's $I$")
    ax2.legend(fontsize=5.6)
    ax2.set_title("Residual spatial autocorrelation, median over 16", fontsize=8)
    fig.tight_layout()
    fg.save_figure(fig, "figS11_autocorrelacion", a.fig_dir)
    print(f"\n  -> {a.fig_dir}/figS11_autocorrelacion.pdf")


if __name__ == "__main__":
    main()
