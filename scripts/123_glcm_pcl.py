#!/usr/bin/env python3
"""Textura GLCM sobre el geomediano por píxel, solo Parcelas-CL: el test de descarte.

Los cubos locales ``data/derived/phenology/{plot_id}.nc`` guardan la serie de observaciones
de los 5x5 píxeles de cada parcela PCL (``obs_band_*``, ya con máscara de nubes). Living Trees
solo guarda centro y media 5x5, así que la textura de LT exige reextraer en el pod. Antes de
pedir eso, este script mide si la textura aporta algo dentro de PCL: si no aporta ahí, no hace
falta el viaje.

Pasos:
  1. Geomediano de las seis bandas por píxel (``biodiv.geomedian``, >= MIN_OBS obs).
  2. Cuantización a 32 niveles con cortes GLOBALES por banda (percentiles 1 y 99 de todos los
     píxeles de todas las parcelas). Con cortes por ventana la textura sería relativa a cada
     parcela y no comparable entre parcelas.
  3. GLCM simétrica, distancia 1, ángulos 0 y pi/2, promediada entre ángulos. Ventana 5x5
     (40 pares) como PRIMARIA y 3x3 central (12 pares) como sensibilidad: con 12 pares ASM,
     entropía e IDM salen casi discretas. Los pares con un píxel sin geomediano se descartan.
  4. Siete propiedades (las de lib-samsara): ASM, contraste, correlación, varianza, IDM, media
     de suma y entropía. La media de suma de una GLCM simétrica es 2 x la media de niveles: es
     NIVEL, no textura, y se reporta así.
  5. El nivel de la ventana (media del geomediano de la banda sobre los píxeles válidos),
     porque la textura está confundida con el nivel y el modelo debe llevar los dos.

Escala: 5x5 a 30 m son 22.500 m² (3x3: 8.100 m²), de 16 a 45 veces la parcela mediana. Es
contexto de paisaje, no textura de la parcela.

Escribe data/derived/gm_pixels_pcl.parquet (geomediano por píxel) y
data/derived/glcm_pcl.parquet (una fila por parcela: tex{5,3}_{banda}_{prop}, lvl{5,3}_{banda},
n_px{5,3}).

Uso:
    python scripts/123_glcm_pcl.py
    python scripts/123_glcm_pcl.py --limit 20     # prueba
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biodiv import geomedian as gmod                                      # noqa: E402

ID = "PlotObservationID"
LEVELS = 32
PROPS = ["asm", "contrast", "correlation", "variance", "idm", "sumavg", "entropy"]


def gm_pixels(path: Path) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        pid = ds.attrs["plot_id"]
        X = np.stack([ds[f"obs_band_{b}"].values for b in gmod.BANDS], axis=-1)  # t,y,x,b
    ny, nx = X.shape[1:3]
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            x = X[:, iy, ix, :]
            ok = np.isfinite(x).all(axis=1)
            r = {"plot_id": pid, "y": iy, "x": ix, "n_obs": int(ok.sum())}
            if ok.sum() >= gmod.MIN_OBS:
                r.update({b: float(v) for b, v in zip(gmod.BANDS, gmod.geometric_median(x[ok]))})
            rows.append(r)
    return pd.DataFrame(rows)


def glcm_props(q: np.ndarray) -> dict[str, float]:
    """``q``: ventana cuantizada (enteros 0..LEVELS-1, -1 = sin dato)."""
    P = np.zeros((LEVELS, LEVELS))
    for a, b in ((q[:, :-1], q[:, 1:]), (q[:-1, :], q[1:, :])):      # 0 y pi/2, distancia 1
        a, b = a.ravel(), b.ravel()
        ok = (a >= 0) & (b >= 0)
        np.add.at(P, (a[ok], b[ok]), 1)
        np.add.at(P, (b[ok], a[ok]), 1)                               # simétrica
    n = P.sum()
    if n == 0:
        return {k: np.nan for k in PROPS} | {"npairs": 0}
    P /= n
    i, j = np.indices(P.shape)
    mu = (i * P).sum()
    var = ((i - mu) ** 2 * P).sum()
    nz = P[P > 0]
    return {"asm": (P ** 2).sum(),
            "contrast": ((i - j) ** 2 * P).sum(),
            "correlation": ((i - mu) * (j - mu) * P).sum() / var if var > 0 else np.nan,
            "variance": var,
            "idm": (P / (1 + (i - j) ** 2)).sum(),
            "sumavg": ((i + j) * P).sum(),
            "entropy": -(nz * np.log(nz)).sum(),
            "npairs": int(n / 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", default="data/derived/phenology")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    files = sorted((ROOT / a.cubes).glob("*.nc"))[: a.limit or None]
    px = pd.concat(Parallel(n_jobs=a.jobs)(delayed(gm_pixels)(f) for f in files), ignore_index=True)
    px[ID] = "PCL_" + px.plot_id.astype(str)
    if not a.limit:
        px.to_parquet(ROOT / "data/derived/gm_pixels_pcl.parquet", index=False)
    print(f"parcelas: {px.plot_id.nunique()}, píxeles con geomediano: "
          f"{px[gmod.BANDS[0]].notna().sum():,} / {len(px):,}")

    cuts = {b: np.nanpercentile(px[b], [1, 99]) for b in gmod.BANDS}
    rows = []
    for pid, g in px.groupby(ID):
        ny, nx = g.y.max() + 1, g.x.max() + 1
        r = {ID: pid}
        for b in gmod.BANDS:
            grid = np.full((ny, nx), np.nan)
            grid[g.y, g.x] = g[b]
            lo, hi = cuts[b]
            q = np.floor((np.clip(grid, lo, hi) - lo) / (hi - lo) * LEVELS)
            q = np.where(np.isfinite(grid), np.minimum(q, LEVELS - 1), -1).astype(int)
            cy, cx = ny // 2, nx // 2
            for w in (5, 3):
                h = w // 2
                sl = (slice(cy - h, cy + h + 1), slice(cx - h, cx + h + 1))
                pr = glcm_props(q[sl])
                for k in PROPS:
                    r[f"tex{w}_{b}_{k}"] = pr[k]
                r[f"lvl{w}_{b}"] = np.nanmean(grid[sl]) if np.isfinite(grid[sl]).any() else np.nan
                if b == gmod.BANDS[0]:
                    r[f"n_px{w}"] = int(np.isfinite(grid[sl]).sum())
                    r[f"npairs{w}"] = pr["npairs"]
        rows.append(r)
    out = pd.DataFrame(rows)
    if not a.limit:
        out.to_parquet(ROOT / "data/derived/glcm_pcl.parquet", index=False)
    print(out[["n_px5", "npairs5", "n_px3", "npairs3"]].describe().round(1).to_string())
    print(out.filter(like="tex5_nir").describe().round(3).T.to_string())


if __name__ == "__main__":
    main()
