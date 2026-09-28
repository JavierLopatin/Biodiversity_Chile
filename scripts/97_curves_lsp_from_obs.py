#!/usr/bin/env python3
"""Curvas raw100 y LSP del píxel central desde la tabla de observaciones, para la reflectancia
original y para las tres versiones con corrección topográfica (scripts/96).

Un solo camino para las cuatro versiones y las dos fuentes, así el contraste corregido
contra original queda pareado y no mezcla el efecto de la corrección con diferencias de
código:
  nc       índices desde las bandas originales (control; debe reproducir lo existente)
  tcfe     SCS+C con C temporal de efectos fijos        [principal]
  tccs     SCS+C con C clásico, mismo régimen geométrico
  tccsall  SCS+C con C clásico sobre todas las observaciones

- Curva raw100: biodiv.curves.interp_grid (la función de scripts/29 y 67) con ngs=100 y
  roll=5, sobre la ventana [primer, último] instante de la parcela.
  -> phenoshape_by_index_raw100<v>_unified.parquet (px = center), y una copia de
     phenoshape_doy_grid_raw100.parquet con el mismo sufijo, que features.py exige.
- LSP: scripts/92.lsp_from_obs (PhenoShape linear + shrink, PhenoLSP auto).
  -> lsp_unified_v2lin<v>.parquet (px = center). Se lee con BIODIV_LSPU=<v>.

Uso:
    python scripts/97_curves_lsp_from_obs.py --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
sys.path.insert(0, str(ROOT / "src"))
from biodiv.curves import interp_grid  # noqa: E402

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
VERSIONS = ["nc", "tcfe", "tccs", "tccsall"]
ID = "PlotObservationID"
NGS, ROLL = 100, 5

_spec = importlib.util.spec_from_file_location("lsp92", ROOT / "scripts" / "92_lsp_unified.py")
lsp92 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lsp92)


def one_plot(args: tuple) -> tuple[list, list]:
    pid, source, t, cols = args
    td = t.astype("datetime64[D]").astype(float)
    curves, lsps = [], []
    for v, per_ix in cols.items():
        for ix, vals in per_ix.items():
            c = interp_grid(td, vals, NGS, roll=ROLL, t_min=td.min(), t_max=td.max())
            curves.append({"v": v, "plot_id": pid, "index": ix, "px": "center",
                           **{f"s{k:02d}": float(x) for k, x in enumerate(c)}})
            lsps.append({"v": v, ID: pid, "source": source, "index": ix, "px": "center",
                         **lsp92.lsp_from_obs(t, vals)})
    return curves, lsps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--obs", default="obs_center_unified_tc.parquet",
                    help="obs_center_unified_tcnull.parquet para las versiones nulas (scripts/99)")
    a = ap.parse_args()
    versions = a.versions

    o = pd.read_parquet(DERIVED / a.obs).sort_values([ID, "time"])
    tasks = []
    for pid, g in o.groupby(ID, sort=True):
        cols = {v: {ix: g[f"{ix}_{v}"].to_numpy(float) for ix in INDICES} for v in versions}
        tasks.append((pid, g.source.iloc[0], g.time.to_numpy(), cols))
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"{len(tasks)} parcelas x {len(versions)} versiones x {len(INDICES)} índices")
    curves, lsps = [], []
    with ProcessPoolExecutor(a.workers, initializer=lsp92._init) as ex:
        for c, l in ex.map(one_plot, tasks, chunksize=8):
            curves.extend(c); lsps.extend(l)
    cu, ls = pd.DataFrame(curves), pd.DataFrame(lsps)
    if a.limit:
        print(ls.groupby("v").lsp_error.apply(lambda s: (s != "").sum()))
        return
    for v in versions:
        sfx = f"_raw100{v}"
        cu[cu.v == v].drop(columns="v").to_parquet(DERIVED / f"phenoshape_by_index{sfx}_unified.parquet", index=False)
        shutil.copyfile(DERIVED / "phenoshape_doy_grid_raw100.parquet", DERIVED / f"phenoshape_doy_grid{sfx}.parquet")
        l = ls[ls.v == v].drop(columns="v")
        l.to_parquet(DERIVED / f"lsp_unified_v2lin{v}.parquet", index=False)
        print(f"{v}: curvas {len(cu[cu.v == v])}, LSP con error {(l.lsp_error != '').sum()}")


if __name__ == "__main__":
    main()
