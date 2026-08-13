#!/usr/bin/env python3
"""Estabilidad fenologica interanual (Lopatin et al. 2023, IEEE GRSL) -- RMSE segmentado.

Referenciado en `docs/01_state_of_the_art.md:39` pero nunca calculado en este proyecto.
`PhenoSensing.get_timeseries_metrics` tenia un bug de una linea (Riesgo R9,
`docs/02_innovation_and_impact.md:59`): el RMSE se comparaba contra el cubo completo `ds`
en vez de contra la ventana `sample` ya recortada -- corregido en PhenoSensing
(commit ddaa3bc, `ds` -> `sample` en `accessor.py:480`) antes de este script.

Que mide, con precision: cada parcela tiene UNA sola ventana causal de 3 anios (no una
serie larga con multiples ventanas deslizantes, que es el uso tipico de Lopatin 2023). Con
`window_length=3` sobre un cubo que ya abarca exactamente 3 anios, `get_timeseries_metrics`
da una sola ventana -- el RMSE mide que tan bien la curva ajustada a los 3 anios predice
las observaciones crudas dentro de esa misma ventana (consistencia/bondad de ajuste
interna), no una comparacion explicita anio-1-contra-anio-2-contra-anio-3. Sigue siendo una
metrica de estabilidad fenologica legitima -- mas residuo, fenologia menos predecible en el
periodo -- pero no es literalmente "diferencia entre anios".

Agregacion pixel -> parcela: `np.nanmean` sobre los 25 pixeles, mismo criterio que
`mean5x5` ya usa para las metricas LSP (`scripts/04_recompute_lsp.py:234`).

Uso:
    python scripts/47_interannual_stability.py
    python scripts/47_interannual_stability.py --limit 20 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PHENO = Path("/mnt/rapidita_4T/GitHub/PhenoSensing")
INDEX = "kndvi"
NGS = 52
ROLL = 5
WINDOW_LENGTH = 3
METRICS = ["rmse", "rmse_sos", "rmse_pos", "rmse_eos"]


def stability_for_plot(path: str, pheno_path: str) -> dict:
    """RMSE total/SOS/POS/EOS por parcela, promediado sobre los 25 pixeles.

    NaN (no un valor forzado) si la parcela no tiene observaciones en al menos
    `WINDOW_LENGTH` anios calendario distintos -- `get_timeseries_metrics` no produce
    ninguna ventana en ese caso (`accessor.py:453-456`).
    """
    sys.path.insert(0, pheno_path)
    import phenosensing  # noqa: F401  (registra el accessor .pheno)
    import xarray as xr

    warnings.filterwarnings("ignore")
    path = Path(path)
    row = {"plot_id": path.stem}
    try:
        with xr.open_dataset(path) as ds:
            obs_name = f"obs_{INDEX}"
            if obs_name not in ds:
                row.update({m: np.nan for m in METRICS})
                row["status"] = "sin obs_" + INDEX
                return row
            # El `.nc` guarda `doy` como la dimension de 52 pasos de `phenoshape`, que tapa
            # la coordenada por-observacion que PhenoShape necesita -- misma trampa
            # documentada en `scripts/29_refit_curves_from_cubes.py::observations()`.
            # Reconstruida desde `time`, que es exacta.
            da = ds[obs_name].assign_coords(
                year=("time", ds["year"].values if "year" in ds.coords
                     else ds["time"].dt.year.values),
                doy=("time", ds["time"].dt.dayofyear.values))
            n_years = len(np.unique(da["year"].values))
            if n_years < WINDOW_LENGTH:
                row.update({m: np.nan for m in METRICS})
                row["status"] = f"solo {n_years} anios, faltan {WINDOW_LENGTH}"
                return row

            out = da.pheno.get_timeseries_metrics(
                window_length=WINDOW_LENGTH, metric=METRICS,
                rollWindow=ROLL, nGS=NGS, hemisphere="auto")
            for m in METRICS:
                arr = out[m].values  # (window, y, x); una sola ventana esperada
                row[m if m != "rmse" else "rmse_total"] = float(np.nanmean(arr))
            row["status"] = "ok"
    except Exception as e:
        row.update({m if m != "rmse" else "rmse_total": np.nan for m in METRICS})
        row["status"] = f"{type(e).__name__}: {e}"
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cubes", default="data/derived/phenology")
    p.add_argument("--out", default="data/derived/interannual_stability.parquet")
    p.add_argument("--phenosensing", default=str(DEFAULT_PHENO))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cubes = sorted(Path(args.cubes).glob("*.nc"))
    if args.limit:
        cubes = cubes[: args.limit]
    if not cubes:
        raise SystemExit(f"no hay .nc en {args.cubes}")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(stability_for_plot, str(c), args.phenosensing): c for c in cubes}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(cubes):
                print(f"  [{i}/{len(cubes)}] {(time.time()-t0)/60:.1f} min", flush=True)

    out = pd.DataFrame(rows).rename(columns={"plot_id": "PlotObservationID"})
    n_ok = (out["status"] == "ok").sum()
    n_nan = len(out) - n_ok
    print(f"\n{len(out)} parcelas, {n_ok} ok, {n_nan} sin valor (ver 'status')")
    if n_nan:
        print(out.loc[out["status"] != "ok", "status"].value_counts().to_string())

    rmse_cols = ["rmse_total", "rmse_sos", "rmse_pos", "rmse_eos"]
    neg = (out[rmse_cols] < 0).any(axis=1).sum()
    if neg:
        raise SystemExit(f"{neg} parcelas con RMSE negativo -- RMSE no puede ser negativo, "
                         f"revisar antes de escribir el parquet")

    cols = ["PlotObservationID"] + rmse_cols
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_parquet(out_path, index=False)
    print(f"escrito {out_path}")


if __name__ == "__main__":
    main()
