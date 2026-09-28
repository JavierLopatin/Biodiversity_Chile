#!/usr/bin/env python3
"""Observaciones del píxel central de las dos fuentes, con su geometría solar y su terreno.

Insumo de la corrección topográfica SCS+C (scripts/96). Una fila por observación Landsat
del píxel central de cada parcela del pool unificado:

- reflectancia de superficie de las 6 bandas, en 0-1, tal como las guardó la adquisición.
  PCL sale de obs_band_* de los .nc (píxel central del parche 5x5); LT de
  living_trees/series_band_*_center.parquet, unidas por (site_id, time, sensor).
- sensor e instante de adquisición (UTC).
- geometría solar calculada con pysolar para (lat, lon, instante, elevación DEM):
  sun_zenith y sun_azimuth en grados, con el acimut medido desde el norte en sentido
  horario. Es la misma convención que el aspect del DEM, así cos(sun_azimuth - aspect)
  sale sin conversión. pysolar da ~0,1° de error. Una implementación NOAA vectorizada daba
  0,48° y no alcanza para esto.
- slope y aspect del DEM (Copernicus 30 m, píxel central, grados; aspect = hacia dónde
  mira la ladera, 0 = N, horario). Salen de topography_unified.parquet y NO de
  plots_unified.slope_field, que es la pendiente de campo de LT (no está en grados).
  La convención de aspect se verificó con física: en laderas >20°, el NIR de mayo a agosto
  correlaciona +0,87 (LT) y +0,91 (PCL) con el IL calculado con este aspect, y -0,85 /
  -0,90 con el aspect girado 180°.
- clase MapBiomas nivel 2 (scripts/89) y pasada orbital: sensor + instante redondeado a
  5 minutos, que equivale a un path, para agrupar la estimación de C.

Escribe data/derived/obs_center_unified.parquet.

Uso:
    python scripts/95_obs_center_solar.py --workers 12
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
LT_DIR = DERIVED / "living_trees"
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
ID = "PlotObservationID"


def _pcl_one(path: str) -> pd.DataFrame:
    with xr.open_dataset(path) as d:
        cy, cx = d.sizes["y"] // 2, d.sizes["x"] // 2
        out = pd.DataFrame({
            ID: "PCL_" + str(d.attrs["plot_id"]),
            "time": pd.to_datetime(d["time"].values),
            "sensor": np.asarray(d["sensor"].values).astype(str),
            **{b: d[f"obs_band_{b}"].isel(y=cy, x=cx).values.astype(float) for b in BANDS},
        })
    return out


def pcl_obs(workers: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(DERIVED / "phenology" / "*.nc")))
    with ProcessPoolExecutor(workers) as ex:
        return pd.concat(ex.map(_pcl_one, files, chunksize=16), ignore_index=True)


def lt_obs() -> pd.DataFrame:
    plots = pd.read_parquet(DERIVED / "living_trees_plots.parquet")
    sites = pd.read_parquet(LT_DIR / "sites.parquet")
    cw = plots.merge(sites[["site_id", "lat", "lon"]], on=["lat", "lon"])[[ID, "site_id"]]
    out = None
    for b in BANDS:
        s = pd.read_parquet(LT_DIR / f"series_band_{b}_center.parquet").rename(
            columns={f"band_{b}": b})
        out = s if out is None else out.merge(s, on=["site_id", "time", "sensor"], how="outer")
    out = out.merge(cw, on="site_id", how="inner").drop(columns="site_id")
    out["time"] = pd.to_datetime(out["time"])
    return out


def _sun(args: tuple) -> tuple[list, list]:
    from pysolar.solar import get_altitude, get_azimuth
    lat, lon, elev, times = args
    z, a = [], []
    for t in times:
        w = pd.Timestamp(t).to_pydatetime().replace(tzinfo=dt.timezone.utc)
        z.append(90.0 - get_altitude(lat, lon, w, elevation=elev))
        a.append(get_azimuth(lat, lon, w, elevation=elev) % 360.0)
    return z, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    obs = pd.concat([pcl_obs(a.workers).assign(source="parcelas_cl"),
                     lt_obs().assign(source="living_trees")], ignore_index=True)
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")[[ID, "lat", "lon"]]
    topo = pd.read_parquet(DERIVED / "topography_unified.parquet")
    idc = ID if ID in topo.columns else "plot_id"
    topo = topo.rename(columns={idc: ID})[[ID, "elevation", "slope", "aspect"]]
    mb = pd.read_parquet(DERIVED / "mapbiomas_class_unified.parquet")
    lg = pd.read_csv(ROOT / "MapBiomas" / "legend.csv")
    mb["mb_level2"] = mb.mb_code.map(dict(zip(lg.code, lg.level2)))
    obs = (obs.merge(plots, on=ID, how="left").merge(topo, on=ID, how="left")
              .merge(mb[[ID, "mb_level2", "mb_native"]], on=ID, how="left"))
    assert obs.lat.notna().all() and obs.slope.notna().all(), "parcelas sin lat/lon o sin DEM"

    # geometría solar por (parcela, instante) único
    obs = obs.sort_values([ID, "time"]).reset_index(drop=True)
    tasks, keys = [], []
    for pid, g in obs.groupby(ID, sort=False):
        tasks.append((float(g.lat.iloc[0]), float(g.lon.iloc[0]),
                      float(np.nan_to_num(g.elevation.iloc[0])), g.time.values))
        keys.append(g.index.values)
    z = np.empty(len(obs)); az = np.empty(len(obs))
    with ProcessPoolExecutor(a.workers) as ex:
        for idx, (zz, aa) in zip(keys, ex.map(_sun, tasks, chunksize=8)):
            z[idx] = zz; az[idx] = aa
    obs["sun_zenith"], obs["sun_azimuth"] = z, az
    obs["pass_id"] = obs.sensor.astype(str) + "_" + obs.time.dt.floor("5min").dt.strftime("%Y%m%dT%H%M")

    f = DERIVED / "obs_center_unified.parquet"
    obs.to_parquet(f, index=False)
    print(obs.groupby("source").agg(obs=("time", "size"), parcelas=(ID, "nunique"),
                                    pasadas=("pass_id", "nunique")).to_string())
    print(f"cenit solar: {obs.sun_zenith.min():.1f}-{obs.sun_zenith.max():.1f}°; "
          f"obs con las 6 bandas: {obs[BANDS].notna().all(axis=1).mean():.3f}")
    print(f"-> {f}")


if __name__ == "__main__":
    main()
