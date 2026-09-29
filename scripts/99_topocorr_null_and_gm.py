#!/usr/bin/env python3
"""Control de perturbación nula para la corrección topográfica, y geomediano por versión.

1. NULO. ¿Un ΔR² de ~0,02 al corregir es efecto de la corrección, o lo produce cualquier
   perturbación de ese tamaño de los predictores (el RF se reordena)? Para responder se
   construyen versiones nulas: el vector de factores de corrección de las 6 bandas de cada
   observación corregida (ρ_tcfe / ρ), tratado como unidad, se PERMUTA al azar entre las
   observaciones corregidas. La magnitud y la distribución de la corrección se conservan
   exactas, y también la correlación entre bandas dentro de una observación; lo único que
   se destruye es la alineación con el terreno y el sol. Las observaciones no corregidas
   quedan iguales. Los índices se recalculan desde las bandas nulas con las fórmulas de
   biodiv.cube, igual que en scripts/96. Hay N_DRAWS sorteos con semillas fijas (null1..).

2. GEOMEDIANO POR VERSIÓN. gm del píxel central (geomedian.geometric_median, mads,
   indices_from_bands, MIN_OBS; las mismas funciones que scripts/18 y 84) sobre las
   observaciones con las 6 bandas presentes, para nc (bandas originales), las tres
   versiones corregidas y las nulas.
   -> data/derived/gm_center_<v>.parquet. Lo lee el bloque gmo de features.py con
      BIODIV_GMO=<v>.

Escribe data/derived/obs_center_unified_tcnull.parquet con las columnas nulas; scripts/97
lo lee con --versions null1 null2 null3.

Uso:
    python scripts/99_topocorr_null_and_gm.py
    python scripts/99_topocorr_null_and_gm.py --draws 4 10 --suffix _b   # sorteos 4..10, sin gm
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
sys.path.insert(0, str(ROOT / "src"))
from biodiv import geomedian as gmod  # noqa: E402

_s = importlib.util.spec_from_file_location("tc96", ROOT / "scripts" / "96_topo_correction_scsc.py")
tc96 = importlib.util.module_from_spec(_s); _s.loader.exec_module(tc96)

BANDS, INDICES, ID = tc96.BANDS, tc96.INDICES, "PlotObservationID"
N_DRAWS, REF = 3, "tcfe"
VERSIONS_GM = ["nc", "tcfe", "tccs", "tccsall"] + [f"null{k}" for k in range(1, N_DRAWS + 1)]


def gm_one(args):
    pid, X = args
    row = {ID: pid, "gm_count": float(len(X))}
    if len(X) >= gmod.MIN_OBS:
        gm = gmod.geometric_median(X)
        for b, v in zip(gmod.BANDS, gm):
            row[f"gm_band_{b}"] = float(v)
        row.update({f"gm_{k}": v for k, v in gmod.indices_from_bands(dict(zip(gmod.BANDS, gm))).items()})
        row["gm_emad"], row["gm_smad"], row["gm_bcmad"] = gmod.mads(X, gm)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", nargs=2, type=int, default=[1, N_DRAWS],
                    help="primer y último sorteo (semilla 1000 + k); la semilla fija el sorteo")
    ap.add_argument("--suffix", default="", help="sufijo del parquet de salida")
    ap.add_argument("--no-gm", action="store_true")
    a = ap.parse_args()
    draws = range(a.draws[0], a.draws[1] + 1)
    o = pd.read_parquet(DERIVED / "obs_center_unified_tc.parquet")
    assert gmod.BANDS == BANDS
    F = np.column_stack([o[f"{b}_{REF}"].to_numpy() / o[b].to_numpy() for b in BANDS])
    corrected = np.isfinite(F).all(axis=1) & (np.abs(F - 1) > 0).any(axis=1)
    print(f"observaciones corregidas por {REF}: {corrected.sum()} de {len(o)}")
    idx = np.flatnonzero(corrected)
    for k in draws:
        rng = np.random.default_rng(1000 + k)
        Fk = np.ones_like(F)
        Fk[idx] = F[rng.permutation(idx)]
        bands = {b: o[b].to_numpy() * Fk[:, j] for j, b in enumerate(BANDS)}
        for b in BANDS:
            o[f"{b}_null{k}"] = bands[b]
        for i, v in tc96.indices(bands).items():
            o[f"{i}_null{k}"] = v
        d = np.abs(o[f"kndvi_null{k}"] - o["kndvi_nc"])[corrected]
        d0 = np.abs(o["kndvi_tcfe"] - o["kndvi_nc"])[corrected]
        print(f"  null{k}: |Δ kNDVI| mediana {np.nanmedian(d):.4f} (real {np.nanmedian(d0):.4f})")
    keep = [ID, "source", "time"] + [f"{c}_null{k}" for k in draws for c in BANDS + INDICES]
    o[keep].to_parquet(DERIVED / f"obs_center_unified_tcnull{a.suffix}.parquet", index=False)
    if a.no_gm:
        return

    for v in VERSIONS_GM:
        cols = BANDS if v == "nc" else [f"{b}_{v}" for b in BANDS]
        tasks = []
        for pid, g in o[[ID] + cols].groupby(ID):
            X = g[cols].to_numpy(float)
            tasks.append((pid, X[np.isfinite(X).all(axis=1)]))
        with ProcessPoolExecutor(12) as ex:
            rows = list(ex.map(gm_one, tasks, chunksize=32))
        out = pd.DataFrame(rows)
        out.to_parquet(DERIVED / f"gm_center_{v}.parquet", index=False)
        print(f"gm {v}: {out.gm_band_nir.notna().sum()} parcelas con geomediano")


if __name__ == "__main__":
    main()
