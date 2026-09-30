#!/usr/bin/env python3
"""Ventanas de altura de dosel ETH alrededor de cada parcela, leídas de los TIFF locales.

Lee los VRT que arma scripts/127 (ETH_CH_height_Chile.vrt, ETH_CH_sd_Chile.vrt, grilla nativa
EPSG:4326 de 1/12000°) y recorta una ventana de 21x21 píxeles centrada en el píxel que
contiene cada parcela: alcanza para el píxel, 3x3, 5x5, 9x9 y la GLCM 5x5 con margen. Mismo
formato que scripts/126 --modo parcelas, así que las dos vías son intercambiables.

Los valores de relleno del producto (255 en altura y SD) pasan a NaN.

Escribe data/derived/eth_canopy_windows.npz (windows: (n, 2, 21, 21), banda 0 altura y 1 SD)
y eth_canopy_windows.parquet (PlotObservationID, fila y columna del píxel central).

Uso:
    python scripts/128_eth_local_windows.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
SRC = "/mnt/rapidita_4T/datos/ETH_GlobalCanopyHeight_10m_2020_version1"
WIN = 21
NODATA = 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=str(ROOT / "data/derived/eth_canopy_windows.npz"))
    a = ap.parse_args()
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "lon", "lat"]]
    arr = np.full((len(plots), 2, WIN, WIN), np.nan, np.float32)
    rc = []
    for b, band in enumerate(("height", "sd")):
        with rasterio.open(Path(a.src) / f"ETH_CH_{band}_Chile.vrt") as ds:
            rows, cols = rasterio.transform.rowcol(ds.transform, plots.lon.to_numpy(), plots.lat.to_numpy())
            h = WIN // 2
            for k, (r, c) in enumerate(zip(rows, cols)):
                w = ds.read(1, window=Window(c - h, r - h, WIN, WIN), boundless=True,
                            fill_value=NODATA).astype(np.float32)
                nd = ds.nodata if ds.nodata is not None else NODATA
                w[(w == nd) | (w == NODATA)] = np.nan
                arr[k, b] = w
            if b == 0:
                rc = list(zip(plots[ID], rows, cols))
                transform, crs = ds.transform, ds.crs.to_string()
    np.savez_compressed(a.out, windows=arr, dx=transform.a, dy=transform.e, crs=crs)
    pd.DataFrame(rc, columns=[ID, "row", "col"]).to_parquet(Path(a.out).with_suffix(".parquet"), index=False)
    ok = np.isfinite(arr[:, 0, WIN // 2, WIN // 2])
    print(f"-> {a.out}: {ok.sum()}/{len(plots)} parcelas con altura en el píxel central")
    print(pd.Series(arr[ok, 0, WIN // 2, WIN // 2]).describe().round(1).to_string())


if __name__ == "__main__":
    main()
