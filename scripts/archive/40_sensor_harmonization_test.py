#!/usr/bin/env python3
"""Sesgo cross-sensor (TM/ETM+ vs OLI) medido en los residuos de la curva ya ajustada.

No se vuelve a consultar el datacube. Cada `.nc` de `data/derived/phenology/` ya trae, por
parcela, las observaciones crudas etiquetadas con el sensor de origen (`obs_{index}`,
coordenada `sensor` sobre `time`) y la curva PhenoShape ya ajustada (`phenoshape`). Muchas
parcelas censadas despues de ~2015 tienen, dentro de su propia ventana causal de 3 anios,
observaciones tanto de Landsat 7 (ETM+, sigue volando pese al SLC-off) como de Landsat 8/9
(OLI) -- vuelo simultaneo real, no epocas distintas.

Por que residuos y no niveles crudos: comparar reflectancia cruda entre parcelas mezcla
sesgo de sensor con efecto de sitio y estacion. Usando la curva PhenoShape de la propia
parcela como linea de base, lo unico que deberia quedar sistematicamente distinto entre
ETM+ y OLI es el sesgo de sensor.

**El sesgo NO es un corrimiento constante -- escala con el nivel de vegetacion.** La primera
version de este script promediaba el residuo sobre todo el anio y encontraba un offset unico
(ej. +0.04 en kNDVI). Aplicado, eso sobrecorregia la temporada baja (donde el sesgo real es
chico, casi nulo) y dejaba casi intacta la brecha real del pico de crecimiento (donde es
grande) -- visible a ojo en `notebooks/08_sensor_harmonization.ipynb` antes de este fix.
Confirmado con una regresion residuo ~ nivel_de_curva por familia: TM/ETM+ tiene pendiente
negativa, OLI tiene pendiente positiva (signos opuestos, ambas con p<1e-110, n=100k
observaciones) -- un sesgo de tipo ganancia/amplitud, no un corrimiento fijo.

Por eso el offset se mide por TERCIL del nivel de la curva (bajo/medio/alto), no como un solo
numero. Dentro de cada tercil se repite exactamente la misma logica pareada de antes: agregar
a una media de residuo por parcela y por familia, y testear la diferencia PAREADA (misma
parcela, misma curva, mismo sitio) con Wilcoxon signed-rank (no asume normalidad) y t-test
pareado como chequeo secundario. `biodiv.harmonize.apply_offset` interpola linealmente entre
los 3 puntos (nivel_medio_tercil, offset) para corregir cualquier observacion OLI segun su
propio nivel de curva.

Solo se opera sobre indices (`kndvi`, `ndvi`, ...), que son lo que efectivamente alimenta
las curvas C1D/C2D/MLP -- las bandas crudas no tienen una curva PhenoShape ajustada contra
la cual residualizar.

Uso:
    python scripts/40_sensor_harmonization_test.py
    python scripts/40_sensor_harmonization_test.py --phenology-dir data/derived/phenology
    python scripts/40_sensor_harmonization_test.py --n-bins 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# No se importa `biodiv.cube`: solo hace falta la lista de indices y el agrupamiento de
# sensores, y `cube.py` importa `datacube` (el paquete ODC) a nivel de modulo -- una
# dependencia pesada, ligada a la maquina con acceso S3, que este script existe justamente
# para no necesitar. Los valores de abajo son la misma tabla que `cube.SENSOR_YEARS` /
# `cube.INDEX_NAMES`; si esos cambian, actualizar aqui tambien.
INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]

# TM (L5) y ETM+ (L7) son una generacion de sensor; OLI (L8/L9) es otra.
SENSOR_FAMILY = {
    "landsat5": "tm_etm", "landsat7": "tm_etm",
    "landsat8": "oli", "landsat9": "oli",
}
MIN_OBS_PER_FAMILY_BIN = 2
MIN_PLOTS_PER_BIN = 5
N_BINS_DEFAULT = 3


def plot_level_residuals(path: Path) -> dict[str, list[tuple]]:
    """Por indice: lista de (nivel_de_curva, familia, residuo) por observacion, una parcela.

    Vacia si la parcela no tiene observaciones de ambas familias dentro de su ventana.
    """
    ds = xr.open_dataset(path)
    sensor = np.asarray(ds["sensor"].values)
    family = np.array([SENSOR_FAMILY.get(s, "other") for s in sensor])
    if len(set(family) & {"tm_etm", "oli"}) < 2:
        return {}

    doy_obs = ds["time"].dt.dayofyear.values.astype(float)
    grid_doy = ds["doy"].values.astype(float)
    order = np.argsort(grid_doy)
    grid_doy_sorted = grid_doy[order]

    out = {}
    for idx in INDICES:
        obs_name = f"obs_{idx}"
        if obs_name not in ds or "phenoshape" not in ds:
            continue
        obs = ds[obs_name].mean(dim=["y", "x"], skipna=True).values
        curve = ds["phenoshape"].sel(index=idx).mean(dim=["y", "x"], skipna=True).values
        curve_sorted = curve[order]
        if np.all(np.isnan(curve_sorted)) or np.all(np.isnan(obs)):
            continue
        curve_at_obs = np.interp(doy_obs, grid_doy_sorted, curve_sorted, period=365)
        resid = obs - curve_at_obs
        rows = [(lvl, fam, r) for lvl, fam, r in zip(curve_at_obs, family, resid)
               if fam in ("tm_etm", "oli") and np.isfinite(lvl) and np.isfinite(r)]
        if rows:
            out[idx] = rows
    return out


def bin_and_pair(df: pd.DataFrame, edges: np.ndarray) -> list[dict]:
    """Para un indice: offset pareado por parcela, uno por tercil de nivel."""
    df = df.copy()
    df["bin"] = np.clip(np.digitize(df["level"], edges[1:-1]), 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        g = df[df["bin"] == b]
        if g.empty:
            bins.append({"status": "PENDIENTE", "reason": "sin observaciones en este tercil"})
            continue
        per_plot = g.groupby(["plot_id", "family"])["residual"].mean().unstack()
        per_plot = per_plot.dropna(subset=["tm_etm", "oli"]) if set(
            ("tm_etm", "oli")).issubset(per_plot.columns) else per_plot.iloc[0:0]
        n = len(per_plot)
        level_mid = float(g["level"].median())
        if n < MIN_PLOTS_PER_BIN:
            bins.append({"status": "PENDIENTE", "n_parcelas": n, "level_mediana": level_mid,
                        "reason": f"muestra insuficiente (<{MIN_PLOTS_PER_BIN} parcelas)"})
            continue
        diff = (per_plot["oli"] - per_plot["tm_etm"]).to_numpy()
        w_stat, w_p = stats.wilcoxon(diff)
        t_stat, t_p = stats.ttest_1samp(diff, 0.0)
        bins.append({
            "n_parcelas": n, "level_mediana": level_mid,
            "level_lo": float(edges[b]), "level_hi": float(edges[b + 1]),
            "offset_oli_menos_tm_etm_mediana": float(np.median(diff)),
            "wilcoxon_p": float(w_p), "ttest_pareado_p": float(t_p),
        })
    return bins


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phenology-dir", default="data/derived/phenology")
    p.add_argument("--out", default="results/tables/sensor_harmonization.json")
    p.add_argument("--n-bins", type=int, default=N_BINS_DEFAULT,
                   help="terciles (3) por defecto; el sesgo escala con el nivel, no es "
                        "constante -- ver docstring")
    args = p.parse_args()

    files = sorted(Path(args.phenology_dir).glob("*.nc"))
    print(f"parcelas en {args.phenology_dir}: {len(files)}")

    per_index_rows = {ix: [] for ix in INDICES}
    failed = 0
    for f in files:
        try:
            r = plot_level_residuals(f)
        except Exception:
            failed += 1
            continue
        for idx, rows in r.items():
            for lvl, fam, resid in rows:
                per_index_rows[idx].append(
                    {"plot_id": f.stem, "level": lvl, "family": fam, "residual": resid})
    if failed:
        print(f"  ({failed} parcelas fallaron al abrir/procesar, ignoradas)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_eligible_files = sum(1 for ix in INDICES if per_index_rows[ix])
    if not n_eligible_files:
        result = {"status": "PENDIENTE",
                  "reason": "ninguna parcela con observaciones de ambas familias "
                            "de sensor dentro de su ventana causal"}
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print("PENDIENTE: sin parcelas elegibles -- ver campo 'reason'")
        return

    result = {"status": "ok", "n_files": len(files), "n_bins": args.n_bins,
              "min_plots_per_bin": MIN_PLOTS_PER_BIN, "per_index": {}}
    for idx in INDICES:
        rows = per_index_rows[idx]
        if not rows:
            result["per_index"][idx] = {"status": "PENDIENTE",
                                        "reason": "sin observaciones elegibles"}
            print(f"  {idx}: PENDIENTE (sin observaciones elegibles)")
            continue
        df = pd.DataFrame(rows)
        n_plots = df["plot_id"].nunique()
        edges = np.quantile(df["level"], np.linspace(0, 1, args.n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bins = bin_and_pair(df, edges)
        result["per_index"][idx] = {"n_parcelas": n_plots, "bins": bins}
        print(f"  {idx}: n_parcelas={n_plots}")
        for b in bins:
            if b.get("status") == "PENDIENTE":
                print(f"    tercil [{b.get('level_lo', '?')}, {b.get('level_hi', '?')}]: "
                      f"PENDIENTE ({b['reason']})")
            else:
                print(f"    nivel~{b['level_mediana']:+.4f}  n={b['n_parcelas']}  "
                      f"offset(mediana)={b['offset_oli_menos_tm_etm_mediana']:+.4f}  "
                      f"wilcoxon p={b['wilcoxon_p']:.4g}")

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nescrito {out_path}")


if __name__ == "__main__":
    main()
