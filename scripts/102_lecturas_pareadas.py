#!/usr/bin/env python3
"""Las lecturas controladas que sostienen las conclusiones, a una tabla versionada.

scripts/101 resume pooled_metrics (el R² del pool). Las afirmaciones del manuscrito salen
de otra cosa: el mismo OOF leído por fuente, dentro de contribuyente (PCL), dentro de banda
de latitud (LT) y por clase de pendiente del DEM, con diferencias pareadas por semilla
entre corridas. Este script declara EN CÓDIGO qué corridas y qué comparaciones se citan, las
calcula con las funciones de scripts/100 y escribe results/tables/lecturas_pareadas.csv:

  tipo           'modelo' (R² de una corrida) o 'par' (R²_a − R²_b, pareado por semilla)
  grupo          qué pregunta responde (compuerta_pool, estratos, pendiente, ...)
  etiqueta       'gm' o 'gm − curva'; run_a/run_b y esquema
  target, clase_pendiente (todas, <5, 5-15, 15-25, >25), lectura
  n, R2_o_dif (media sobre semillas), DE (entre semillas, o de la diferencia pareada)

Las lecturas son: todas · parcelas_cl · living_trees · pcl_dentro_owner (observado y
predicho centrados por contribuyente) · lt_dentro_banda (centrados por banda de 2° de
latitud).

Uso:
    python scripts/102_lecturas_pareadas.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
_s = importlib.util.spec_from_file_location("l100", ROOT / "scripts" / "100_paired_lenses.py")
l100 = importlib.util.module_from_spec(_s); _s.loader.exec_module(l100)

G, S, T = "results/models_gate", "results/models_strata", "results/models_topocorr"
B = "kfold5_block20_unified"


def pool_runs(ts: str) -> dict[str, str]:
    return {
        "piso_topo_area": f"{G}/B01c_topo_ctr-area_raw100_{ts}_unified_woody",
        "clima": f"{G}/RFG1c_clim-topo_ctr-area_raw100_{ts}_unified_woody",
        "curva": f"{G}/RFG2c_curve-topo_ctr-area_kndvi_raw100_{ts}_unified_woody",
        "lsp": f"{G}/RFG7c_lspu-topo_ctr-area_kndvi_raw100_{ts}_unified_woody",
        "gm": f"{G}/RFG4ac_gmoall-topo_ctr-area_raw100_{ts}_unified_woody_gmonc",
        "gm_clima": f"{G}/RFG5ac_gmoall-clim-topo_ctr-area_raw100_{ts}_unified_woody_gmonc",
        "clima_curva": f"{G}/RFG3c_clim-curve-topo_ctr-area_kndvi_raw100_{ts}_unified_woody",
    }


def strata_runs(k: str) -> dict[str, str]:
    return {
        "clima": f"{S}/RFG1c_clim-topo_ctr-area_raw100_pg-all_unified_strat_{k}",
        "curva": f"{S}/RFG2c_curve-topo_ctr-area_kndvi_raw100_pg-all_unified_strat_{k}",
        "lsp": f"{S}/RFG7c_lspu-topo_ctr-area_kndvi_raw100_pg-all_unified_strat_{k}",
        "gm": f"{S}/RFG4ac_gmoall-topo_ctr-area_raw100_pg-all_unified_strat_{k}_gmonc",
        "gm_clima": f"{S}/RFG5ac_gmoall-clim-topo_ctr-area_raw100_pg-all_unified_strat_{k}_gmonc",
    }


def topo_runs(ts: str) -> dict[str, str]:
    return {
        "gm_nc": f"{T}/RFG4ac_gmoall-topo_ctr-area_raw100nc_{ts}_unified_woody_gmonc",
        "gm_csall": f"{T}/RFG4ac_gmoall-topo_ctr-area_raw100tccsall_{ts}_unified_woody_gmotccsall",
        "gm_clima_nc": f"{T}/RFG5ac_gmoall-clim-topo_ctr-area_raw100nc_{ts}_unified_woody_gmonc",
        "gm_clima_csall": f"{T}/RFG5ac_gmoall-clim-topo_ctr-area_raw100tccsall_{ts}_unified_woody_gmotccsall",
        "curva_kndvi_nc": f"{T}/RFG2c_curve-topo_ctr-area_kndvi_raw100nc_{ts}_unified_woody",
        "curva_evi_nc": f"{T}/RFG2c_curve-topo_ctr-area_evi_raw100nc_{ts}_unified_woody",
        "curva_evi_csall": f"{T}/RFG2c_curve-topo_ctr-area_evi_raw100tccsall_{ts}_unified_woody",
    }


POOL_PAIRS = [("gm", "curva"), ("gm", "clima"), ("gm", "piso_topo_area"), ("curva", "clima"),
              ("gm_clima", "clima"), ("gm_clima", "clima_curva"), ("gm_clima", "gm")]
STRATA_PAIRS = [("curva", "gm"), ("curva", "clima"), ("gm", "clima"), ("gm_clima", "clima"),
                ("curva", "gm_clima")]
TOPO_PAIRS = [("gm_nc", "curva_kndvi_nc"), ("gm_nc", "curva_evi_nc"),
              ("gm_csall", "curva_evi_csall"), ("gm_csall", "gm_nc"),
              ("gm_clima_csall", "gm_clima_nc")]
SPECS = [
    ("compuerta_pool", "pg_all_lenoso", "lcbd_count_sorensen", pool_runs("pg-all"), POOL_PAIRS, False),
    ("compuerta_pool", "unified_all_lenoso", "lcbd_pa_unified", pool_runs("unified-all"), POOL_PAIRS, False),
    ("compuerta_pool", "unified_all_lenoso", "hill_q0_unified", pool_runs("unified-all"), POOL_PAIRS, False),
    ("estratos", "Forest", "lcbd_count_sorensen", strata_runs("11forest"), STRATA_PAIRS, False),
    ("estratos", "NoBosque", "lcbd_count_sorensen", strata_runs("nobosque"), STRATA_PAIRS, False),
    ("pendiente", "pg_all_lenoso", "lcbd_count_sorensen", topo_runs("pg-all"), TOPO_PAIRS, True),
    ("pendiente", "unified_all_lenoso", "lcbd_pa_unified", topo_runs("unified-all"), TOPO_PAIRS, True),
]


def main() -> None:
    plots = pd.read_parquet(ROOT / "data" / "derived" / "plots_unified.parquet")[[ID, "Owner", "source", "lat"]]
    topo = pd.read_parquet(ROOT / "data" / "derived" / "topography_unified.parquet").rename(
        columns={"plot_id": ID})[[ID, "slope"]]
    rows = []
    for grupo, pool, t, runs, pairs, by_slope in SPECS:
        per = {}
        for lab, path in runs.items():
            oof = (pd.read_csv(ROOT / path / B / "oof_predictions.csv")
                     .merge(plots, on=ID, how="left").merge(topo, on=ID, how="left"))
            L = l100.lenses(oof, t, 2.0, by_slope=by_slope)
            if not by_slope:
                L = L.assign(slope_cls="todas").reset_index().set_index(["slope_cls", "seed"])
            per[lab] = L
        lect = ["todas", "parcelas_cl", "living_trees", "pcl_dentro_owner", "lt_dentro_banda"]
        ncol = {"todas": "n_todas", "parcelas_cl": "n_parcelas_cl", "living_trees": "n_living_trees",
                "pcl_dentro_owner": "n_parcelas_cl", "lt_dentro_banda": "n_living_trees"}
        for lab, L in per.items():
            for sc, g in L.groupby(level="slope_cls"):
                for le in lect:
                    if le not in g:
                        continue
                    rows.append(dict(tipo="modelo", grupo=grupo, pool=pool, etiqueta=lab,
                                     run_a=Path(runs[lab]).name, run_b="", esquema=B, target=t,
                                     clase_pendiente=sc, lectura=le, n=int(g[ncol[le]].iloc[0]),
                                     R2_o_dif=g[le].mean(), DE=g[le].std(ddof=1)))
        for a, b in pairs:
            d = per[a][lect] - per[b][lect]
            for sc, g in d.groupby(level="slope_cls"):
                na = per[a].xs(sc, level="slope_cls")
                for le in lect:
                    rows.append(dict(tipo="par", grupo=grupo, pool=pool, etiqueta=f"{a} − {b}",
                                     run_a=Path(runs[a]).name, run_b=Path(runs[b]).name, esquema=B,
                                     target=t, clase_pendiente=sc, lectura=le,
                                     n=int(na[ncol[le]].iloc[0]), R2_o_dif=g[le].mean(),
                                     DE=g[le].std(ddof=1)))
    out = pd.DataFrame(rows)
    f = ROOT / "results" / "tables" / "lecturas_pareadas.csv"
    out.to_csv(f, index=False)
    print(f"-> {f}  ({len(out)} filas: {(out.tipo == 'modelo').sum()} de modelos, {(out.tipo == 'par').sum()} pareadas)")
    chk = out[(out.tipo == "par") & (out.clase_pendiente == "todas")]
    print(chk[chk.lectura.isin(["todas", "lt_dentro_banda", "pcl_dentro_owner"])]
          .pivot_table(index=["pool", "target", "etiqueta"], columns="lectura", values="R2_o_dif").round(3).to_string())


if __name__ == "__main__":
    main()
