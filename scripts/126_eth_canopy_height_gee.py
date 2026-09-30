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


def parcelas(ee, out: Path, limit: int = 0) -> None:
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lon", "lat"]]
    if limit:
        plots = plots.sample(limit, random_state=0).reset_index(drop=True)
    proj = ee.Image(H).projection().getInfo()
    crs, tr = proj["crs"], proj["transform"]           # grilla nativa del activo
    dx, dy = tr[0], tr[4]
    img = image(ee).toFloat()
    arr = np.full((len(plots), 2, WIN, WIN), np.nan, np.float32)
    corners = []
    for k, r in enumerate(plots.itertuples()):
        # esquina de la ventana alineada a la grilla nativa (EPSG:4326 en el activo)
        cx = tr[2] + np.floor((r.lon - tr[2]) / dx) * dx
        cy = tr[5] + np.floor((r.lat - tr[5]) / dy) * dy
        x0, y0 = cx - (WIN // 2) * dx, cy - (WIN // 2) * dy
        req = {"expression": img, "fileFormat": "NPY",
               "grid": {"dimensions": {"width": WIN, "height": WIN},
                        "affineTransform": {"scaleX": dx, "shearX": 0, "translateX": x0,
                                            "shearY": 0, "scaleY": dy, "translateY": y0},
                        "crsCode": crs}}
        for intento in range(5):
            try:
                a = np.load(io.BytesIO(ee.data.computePixels(req)))
                arr[k, 0], arr[k, 1] = a["height"], a["sd"]
                break
            except Exception as e:                     # cuotas o cortes: reintenta
                if intento == 4:
                    print(f"  {r[1]}: {str(e)[:120]}")
                time.sleep(2 ** intento)
        corners.append((r[1], x0, y0))
        if k % 200 == 0:
            print(f"  {k}/{len(plots)}")
    np.savez_compressed(out, windows=arr, dx=dx, dy=dy, crs=crs)
    pd.DataFrame(corners, columns=[ID, "x0", "y0"]).to_parquet(out.with_suffix(".parquet"), index=False)
    ok = np.isfinite(arr[:, 0, WIN // 2, WIN // 2]).sum()
    print(f"-> {out}: {ok}/{len(plots)} parcelas con altura en el píxel central")


def chile(ee, folder: str) -> None:
    """Chile continental: el polígono GAUL incluye Isla de Pascua y Juan Fernández, cuyo
    recuadro llegaría a -109° de longitud, así que se intersecta con -76..-66 / -56..-17.
    Grilla nativa del activo (EPSG:4326, 1/12000°), sin remuestrear."""
    cont = ee.Geometry.Rectangle([-76.0, -56.0, -66.0, -17.0], "EPSG:4326", False)
    region = (ee.FeatureCollection("FAO/GAUL/2015/level0")
              .filter(ee.Filter.eq("ADM0_NAME", "Chile")).geometry().intersection(cont, 100))
    tr = ee.Image(H).projection().getInfo()["transform"]
    for band, asset in (("height", H), ("sd", SD)):
        task = ee.batch.Export.image.toDrive(
            image=ee.Image(asset).clip(region), description=f"ETH_CH_{band}_Chile",
            folder=folder, fileNamePrefix=f"ETH_CH_{band}_Chile", region=region.bounds(),
            crs="EPSG:4326", crsTransform=tr, maxPixels=1e11, fileDimensions=32768,
            fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})
        task.start()
        print(f"tarea {band}: {task.id}  (seguir en https://code.earthengine.google.com/tasks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="proyecto de Google Cloud registrado en Earth Engine")
    ap.add_argument("--modo", choices=["parcelas", "chile"], default="parcelas")
    ap.add_argument("--probar", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="modo parcelas: solo N parcelas al azar (prueba)")
    ap.add_argument("--drive-folder", default="ETH_CH_Chile")
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
        parcelas(ee, Path(a.out), a.limit)
    else:
        chile(ee, a.drive_folder)


if __name__ == "__main__":
    main()
