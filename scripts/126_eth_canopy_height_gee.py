#!/usr/bin/env python3
"""Altura de dosel de Lang et al. 2023 (ETH, 10 m, 2020) desde Google Earth Engine.

Dos activos públicos:
    users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1     altura (m)
    users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1   SD del ensemble = incertidumbre predictiva,
                                                       NO dispersión de altura dentro del píxel

Dos modos:

  --modo parcelas   (el que necesita el análisis) una ventana de 21x21 píxeles de 10 m
                    (210 m) centrada en cada una de las 3.102 parcelas, las dos bandas,
                    descargada directo con ee.data.computePixels, sin pasar por Drive. Cubre
                    el píxel, 3x3, 5x5, 9x9 y la GLCM 5x5 con margen. Escribe
                    data/derived/eth_canopy_windows.npz (alturas y SD, forma (n, 2, 21, 21)) y
                    un índice .parquet con PlotObservationID y la esquina de cada ventana.

  --modo chile      Chile continental entero a 10 m, exportado a Google Drive en teselas
                    GeoTIFF (Export.image.toDrive, fileDimensions). Son del orden de 7-8 mil
                    millones de píxeles por banda: varios GB por banda en Drive, y la
                    descarga desde Drive es aparte. Límite: FAO GAUL 2015 nivel 0, "Chile",
                    recortado a Chile continental. Grilla nativa del activo.

Credenciales: el entorno `phenopy` ya trae earthengine-api y ~/.config/earthengine/credentials
existe, pero ee.Initialize() sin proyecto falla ("Not signed up for Earth Engine or project is
not registered"). Hace falta un proyecto de Google Cloud registrado para Earth Engine:

    conda activate phenopy
    earthengine authenticate          # solo si el token caducó
    python scripts/126_eth_canopy_height_gee.py --project ee-javierlopatin --modo parcelas

Uso:
    python scripts/126_eth_canopy_height_gee.py --project P --modo parcelas
    python scripts/126_eth_canopy_height_gee.py --project P --modo chile --drive-folder ETH_CH_Chile
    python scripts/126_eth_canopy_height_gee.py --project P --probar      # solo verifica acceso
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
H = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
SD = "users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1"
WIN = 21


def image(ee):
    return ee.Image(H).rename("height").addBands(ee.Image(SD).rename("sd"))


def probar(ee) -> None:
    img = image(ee)
    print("bandas:", img.bandNames().getInfo())
    print("proyección:", ee.Image(H).projection().getInfo())
    pt = ee.Geometry.Point([-72.6, -39.8])      # bosque templado, prueba
    print("muestra:", img.sample(pt, scale=10).first().getInfo())


def parcelas(ee, out: Path, limit: int = 0, threads: int = 12) -> None:
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lon", "lat"]]
    if limit:
        plots = plots.sample(limit, random_state=0).reset_index(drop=True)
    proj = ee.Image(H).projection().getInfo()
    crs, tr = proj["crs"], proj["transform"]           # grilla nativa del activo
    dx, dy = tr[0], tr[4]
    img = image(ee).toFloat()
    arr = np.full((len(plots), 2, WIN, WIN), np.nan, np.float32)
    corners = [None] * len(plots)

    def one(k: int) -> None:
        pid, lon, lat = plots.iloc[k]
        # esquina de la ventana alineada a la grilla nativa (EPSG:4326 en el activo)
        cx = tr[2] + np.floor((lon - tr[2]) / dx) * dx
        cy = tr[5] + np.floor((lat - tr[5]) / dy) * dy
        x0, y0 = cx - (WIN // 2) * dx, cy - (WIN // 2) * dy
        req = {"expression": img, "fileFormat": "NPY",
               "grid": {"dimensions": {"width": WIN, "height": WIN},
                        "affineTransform": {"scaleX": dx, "shearX": 0, "translateX": x0,
                                            "shearY": 0, "scaleY": dy, "translateY": y0},
                        "crsCode": crs}}
        for intento in range(6):
            try:
                a = np.load(io.BytesIO(ee.data.computePixels(req)))
                arr[k, 0], arr[k, 1] = a["height"], a["sd"]
                break
            except Exception as e:                     # cuotas o cortes: reintenta
                if intento == 5:
                    print(f"  {pid}: {str(e)[:120]}", flush=True)
                time.sleep(2 ** intento)
        corners[k] = (pid, x0, y0)

    # computePixels es una petición por ventana; en paralelo baja de ~50 a ~5 minutos
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(threads) as ex:
        for n, _ in enumerate(ex.map(one, range(len(plots))), 1):
            if n % 500 == 0:
                print(f"  {n}/{len(plots)}", flush=True)
    arr[~np.isfinite(arr)] = np.nan                   # EE entrega los píxeles enmascarados como -inf
    np.savez_compressed(out, windows=arr, dx=dx, dy=dy, crs=crs)
    pd.DataFrame(corners, columns=[ID, "x0", "y0"]).to_parquet(out.with_suffix(".parquet"), index=False)
    ok = np.isfinite(arr[:, 0, WIN // 2, WIN // 2]).sum()
    print(f"-> {out}: {ok}/{len(plots)} parcelas con altura en el píxel central")


#: Franjas de latitud de la exportación de Chile. Una sola tarea por banda para todo Chile se
#: quedó congelada al 55 % tras 18 h (2026-09-30); en franjas de 4° las tareas corren en
#: paralelo, se siguen por separado y una que se trabe se relanza sola.
STRIPS = [(-56, -52), (-52, -48), (-48, -44), (-44, -40), (-40, -36), (-36, -32), (-32, -28),
          (-28, -24), (-24, -20), (-20, -17)]


def chile(ee, folder: str, only: list[str] | None = None) -> None:
    """Chile continental en franjas: el polígono GAUL incluye Isla de Pascua y Juan Fernández,
    así que se intersecta con -76..-66 de longitud. Grilla nativa del activo (EPSG:4326,
    1/12000°), sin remuestrear. ``only`` relanza solo esas descripciones de tarea."""
    chile_geom = (ee.FeatureCollection("FAO/GAUL/2015/level0")
                  .filter(ee.Filter.eq("ADM0_NAME", "Chile")).geometry())
    tr = ee.Image(H).projection().getInfo()["transform"]
    for lo, hi in STRIPS:
        cont = ee.Geometry.Rectangle([-76.0, lo, -66.0, hi], "EPSG:4326", False)
        region = chile_geom.intersection(cont, 100)
        for band, asset in (("height", H), ("sd", SD)):
            name = f"ETH_CH_{band}_Chile_S{abs(lo)}-{abs(hi)}"
            if only and name not in only:
                continue
            task = ee.batch.Export.image.toDrive(
                image=ee.Image(asset).clip(region), description=name, folder=folder,
                fileNamePrefix=name, region=region.bounds(), crs="EPSG:4326", crsTransform=tr,
                maxPixels=1e11, fileDimensions=32768, fileFormat="GeoTIFF",
                formatOptions={"cloudOptimized": True})
            task.start()
            print(f"tarea {name}: {task.id}", flush=True)
    print("seguir en https://code.earthengine.google.com/tasks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="proyecto de Google Cloud registrado en Earth Engine")
    ap.add_argument("--modo", choices=["parcelas", "chile"], default="parcelas")
    ap.add_argument("--probar", action="store_true")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="modo parcelas: solo N parcelas al azar (prueba)")
    ap.add_argument("--drive-folder", default="ETH_CH_Chile")
    ap.add_argument("--only", nargs="+", default=None, help="modo chile: relanzar solo estas tareas")
    ap.add_argument("--out", default=str(ROOT / "data/derived/eth_canopy_windows.npz"))
    a = ap.parse_args()
    import ee
    # el endpoint de alto volumen es para computePixels; las tareas de exportación van por el normal
    if a.modo == "chile" and not a.probar:
        ee.Initialize(project=a.project)
    else:
        ee.Initialize(project=a.project, opt_url="https://earthengine-highvolume.googleapis.com")
    if a.probar:
        probar(ee)
    elif a.modo == "parcelas":
        parcelas(ee, Path(a.out), a.limit, a.threads)
    else:
        chile(ee, a.drive_folder, a.only)


if __name__ == "__main__":
    main()
