#!/usr/bin/env python3
"""Métricas LSP para el pool unificado (Parcelas-CL + Living Trees), por la MISMA vía en las
dos fuentes.

Por qué hace falta. `lsp_all_auto.parquet` solo cubre Parcelas-CL: sale de
`scripts/04_recompute_lsp.py`, que lee los `.nc` por parcela de la adquisición (curvas
PhenoShape por píxel), y Living Trees no tiene `.nc`. Un bloque `lsp` sobre el pool mixto
dejaba las 2.020 filas de LT imputadas con la mediana del fold, que es el mismo artefacto
que ya tuvieron `clim` y `gm`.

Por qué no basta con calcular LT y juntarlo con `lsp_all_auto`. Esas métricas de PCL se
calcularon sobre las curvas originales de la adquisición, con los defectos A y B de
`docs/13_phenology_year_boundary.md` (orden de llegada no reproducible en 392 parcelas y
colas sin suavizar). Juntarlas con LT calculado ahora pondría un método por fuente, y el RF
lo aprendería como etiqueta de fuente. Acá las dos fuentes pasan por el mismo código:

  observaciones crudas del píxel  ->  PhenoShape(linear, rollWindow=5, nGS=52, shrink)
                                   ->  PhenoLSP(hemisphere="auto")

- PCL: `obs_<index>` de cada `.nc`, con `doy` reconstruido desde `time` (la misma trampa
  que resuelve `scripts/29.observations`).
- LT: `living_trees/series_<index>_<px>.parquet`, cruzado a `PlotObservationID` por
  (lat, lon) contra `sites.parquet`, como en `scripts/68`.
- Reconstructor `linear` + `shrink`: la variante `_v2lin` que `docs/13` propone como
  default. Es reproducible, suaviza bien las colas y conserva la tendencia interanual. No
  se usa `harmonic`: fuerza el cierre del año y bajó el R² cuando se probó.
- `px`: `center` (el píxel central) y `mean5x5` (la media de los 25 píxeles por
  adquisición). En PCL la media se calcula acá desde los 25 píxeles de `obs_<index>`; en LT
  viene hecha en la serie `_mean5x5` de la extracción. Es la misma estadística, media por
  adquisición antes del ajuste. No es la media de 25 LSP por píxel que guarda
  `lsp_all_auto` en sus columnas `_mean5x5`.

Qué se pierde. `anchor_circ_sd` (dispersión circular del anclaje entre los 25 píxeles) no se
puede calcular: cada ajuste es de un solo píxel o de una media. `season_phase` se reporta
igual por ajuste (`phase_anchor`, `phase_strength`, `phase_aseasonal`).

Uso:
    python scripts/92_lsp_unified.py --limit 20            # prueba
    python scripts/92_lsp_unified.py --workers 12
    python scripts/92_lsp_unified.py --parity              # PCL nuevo vs lsp_all_auto

Escribe data/derived/lsp_unified_v2lin.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
PHENO_DIR = DERIVED / "phenology"
LT_DIR = DERIVED / "living_trees"
PHENO_PATH = "/mnt/rapidita_4T/GitHub/PhenoSensing"

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
PXS = ["center", "mean5x5"]
NGS, ROLL, RECON, ROLL_MODE, HEMI = 52, 5, "linear", "shrink", "auto"
MIN_OBS = 5
LSP_METRICS = ["sos", "pos", "eos", "vsos", "vpos", "veos", "los", "msp", "mau",
               "vmsp", "vmau", "ampl", "ios", "rog", "ros", "sw", "trough", "mos"]
PHASE_COORDS = ("phase_offset", "aseasonal", "multi_season")


def _init() -> None:
    sys.path.insert(0, PHENO_PATH)
    import phenosensing  # noqa: F401  -- registra el accessor .pheno
    warnings.filterwarnings("ignore")


def lsp_from_obs(t: np.ndarray, v: np.ndarray) -> dict:
    """Una serie de observaciones (tiempos, valores) -> métricas LSP y fase."""
    import xarray as xr
    from phenosensing.phase import season_phase

    ok = np.isfinite(v)
    if ok.sum() < MIN_OBS:
        return {"lsp_error": f"n_obs<{MIN_OBS}"}
    t, v = pd.to_datetime(t[ok]), v[ok].astype(float)
    da = xr.DataArray(v.reshape(-1, 1, 1), dims=("time", "y", "x"),
                      coords={"time": t.values, "y": [0], "x": [0]})
    da = da.assign_coords(doy=("time", da["time"].dt.dayofyear.values))
    try:
        shape = da.pheno.PhenoShape(interpolType=RECON, rollWindow=ROLL, nGS=NGS,
                                    rollMode=ROLL_MODE)
        shape = shape.sortby("doy")
        lsp = shape.pheno.PhenoLSP(nGS=NGS, hemisphere=HEMI)
    except Exception as e:  # noqa: BLE001 -- se registra, no se oculta
        return {"lsp_error": type(e).__name__ + ": " + str(e)[:120]}
    r = {m: float(np.asarray(lsp[m].values).ravel()[0]) for m in LSP_METRICS if m in lsp}
    curve = shape.values.ravel()
    r["peak_doy_observed"] = float(shape["doy"].values[np.nanargmax(curve)]) \
        if np.isfinite(curve).any() else np.nan
    try:
        clean = shape.drop_vars([c for c in PHASE_COORDS if c in shape.coords], errors="ignore")
        sp = season_phase(clean)
        r["phase_anchor"] = float(np.asarray(sp["anchor"].values).ravel()[0])
        r["phase_strength"] = float(np.asarray(sp["strength"].values).ravel()[0])
        r["phase_aseasonal"] = bool(np.asarray(sp["aseasonal"].values).ravel()[0])
    except Exception as e:  # noqa: BLE001
        r["phase_error"] = type(e).__name__
    r["n_obs"] = int(ok.sum())
    r["lsp_error"] = ""
    return r


def pcl_plot(path: str) -> list[dict]:
    import xarray as xr
    rows = []
    with xr.open_dataset(path) as ds:
        pid = "PCL_" + str(ds.attrs["plot_id"])
        t = ds["time"].values
        for ix in INDICES:
            if f"obs_{ix}" not in ds:
                continue
            a = ds[f"obs_{ix}"].values                          # (time, y, x)
            cy, cx = a.shape[1] // 2, a.shape[2] // 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                series = {"center": a[:, cy, cx],
                          "mean5x5": np.nanmean(a.reshape(a.shape[0], -1), axis=1)}
            for px, v in series.items():
                rows.append({"PlotObservationID": pid, "source": "parcelas_cl",
                             "index": ix, "px": px, **lsp_from_obs(t, v)})
    return rows


def lt_plot(args: tuple) -> list[dict]:
    pid, per_key = args
    rows = []
    for (ix, px), (t, v) in per_key.items():
        rows.append({"PlotObservationID": pid, "source": "living_trees",
                     "index": ix, "px": px, **lsp_from_obs(t, v)})
    return rows


def lt_tasks(limit: int | None) -> list[tuple]:
    plots = pd.read_parquet(DERIVED / "living_trees_plots.parquet")
    sites = pd.read_parquet(LT_DIR / "sites.parquet")
    cw = plots.merge(sites[["site_id", "lat", "lon"]], on=["lat", "lon"], how="left")
    if cw.site_id.isna().any():
        raise SystemExit(f"{cw.site_id.isna().sum()} parcelas LT sin site_id")
    cw = cw.set_index("site_id").PlotObservationID
    by_plot: dict[str, dict] = {}
    for ix in INDICES:
        for px in PXS:
            s = pd.read_parquet(LT_DIR / f"series_{ix}_{px}.parquet")
            s["pid"] = s.site_id.map(cw)
            for pid, g in s.dropna(subset=["pid"]).groupby("pid"):
                g = g.sort_values("time")
                by_plot.setdefault(pid, {})[(ix, px)] = (g.time.values, g[ix].values)
    tasks = sorted(by_plot.items())
    return tasks[:limit] if limit else tasks


def parity(out: pd.DataFrame) -> None:
    """PCL por la vía nueva contra lsp_all_auto (curvas de la adquisición, píxel central)."""
    old = pd.read_parquet(DERIVED / "lsp_all_auto.parquet")
    old["PlotObservationID"] = "PCL_" + old.plot_id.astype(str)
    new = out[(out.source == "parcelas_cl") & (out.px == "center") & (out.lsp_error == "")]
    m = new.merge(old, on=["PlotObservationID", "index"], suffixes=("_new", "_old"))
    print(f"\n== paridad PCL, píxel central: {len(m)} pares parcela x índice ==")
    for c in ["sos", "pos", "eos", "los", "ampl", "msp", "mau", "trough", "rog", "ros"]:
        a, b = m[f"{c}_new"], m[f"{c}_old"]
        ok = a.notna() & b.notna()
        if c in ("sos", "pos", "eos"):
            d = np.abs(((a - b + 182.5) % 365) - 182.5)[ok]     # distancia circular, días
            print(f"  {c:7s} n={ok.sum():5d}  |dif| mediana {d.median():6.1f} d   "
                  f"p90 {d.quantile(.9):6.1f} d   <=16 d: {(d <= 16).mean():.0%}")
        else:
            rho = a[ok].corr(b[ok], method="spearman")
            print(f"  {c:7s} n={ok.sum():5d}  Spearman {rho:.3f}   "
                  f"mediana nueva {a[ok].median():.4g} / vieja {b[ok].median():.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parity", action="store_true", help="solo el contraste, sobre la salida")
    ap.add_argument("--out", default=str(DERIVED / "lsp_unified_v2lin.parquet"))
    a = ap.parse_args()

    if a.parity:
        parity(pd.read_parquet(a.out))
        return

    t0 = time.time()
    files = sorted(str(p) for p in PHENO_DIR.glob("*.nc"))
    lt = lt_tasks(a.limit)
    if a.limit:
        files = files[:a.limit]
    print(f"PCL {len(files)} .nc | LT {len(lt)} parcelas | {RECON}+{ROLL_MODE}, "
          f"nGS={NGS}, rollWindow={ROLL}, hemisphere={HEMI}")
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for r in ex.map(pcl_plot, files, chunksize=4):
            rows.extend(r)
        for r in ex.map(lt_plot, lt, chunksize=8):
            rows.extend(r)
    out = pd.DataFrame(rows)
    out["recon"], out["roll_mode"], out["hemisphere"] = RECON, ROLL_MODE, HEMI
    bad = out.lsp_error != ""
    print(f"filas {len(out)}  errores {bad.sum()} "
          f"({out[bad].lsp_error.str.split(':').str[0].value_counts().to_dict()})")
    print(out[~bad].groupby(["source", "px"]).PlotObservationID.nunique().to_string())
    out.to_parquet(a.out, index=False)
    print(f"-> {a.out}  ({(time.time() - t0) / 60:.1f} min)")
    if not a.limit:
        parity(out)


if __name__ == "__main__":
    main()
