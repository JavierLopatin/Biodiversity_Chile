#!/usr/bin/env python3
"""¿Ayudan las coordenadas y el SPI? Test por tercil de sequía y con las lecturas de control.

Compara, sobre el mismo OOF (mismos folds y semillas), la base gm+clima (RFG5a) contra
gm+clima+coordenadas (RFC1), gm+clima+SPI (RFC2) y las dos (RFC3); y la curva (RFG2) contra
curva+coordenadas (RFC4). Para cada par da la diferencia de R² pareada por semilla:
  - global y con las lecturas de scripts/100 (por fuente, PCL dentro de Owner, LT dentro de
    banda de 2°). Las coordenadas interpolan el gradiente norte-sur; si su ganancia
    desaparece dentro de banda, es posición y no ecología.
  - por TERCIL DE SEQUÍA: SPI-12 medio de la ventana causal (scripts/91) residualizado contra
    latitud, porque el SPI es en buena parte un gradiente latitudinal (mismo criterio que
    scripts/94). Solo parcelas con SPI; los censos posteriores a 2021 no lo tienen.

Escribe results/tables/test_coords_sequia.csv.

Uso:
    python scripts/107_drought_coords_test.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
B = "kfold5_block20_unified"
_s = importlib.util.spec_from_file_location("l100", ROOT / "scripts" / "100_paired_lenses.py")
l100 = importlib.util.module_from_spec(_s); _s.loader.exec_module(l100)

G, S = "results/models_gate", "results/models_strata"
PG = ["lcbd_count_sorensen", "pd_inext_q0", "pd_inext_q1", "pd_inext_q2", "td_inext_q0",
      "td_inext_q1", "td_inext_q2"]
UNI = ["hill_q0_unified", "lcbd_pa_unified", "lcbd_freq_unified", "pcoa1_pa_unified",
       "pcoa2_pa_unified", "pcoa1_freq_unified", "pcoa2_freq_unified", "mpd_unified",
       "mntd_unified", "ses_pd_unified", "ses_mpd_unified", "ses_mntd_unified", "dark_n_unified"]


def runs(root: str, suf: str, gmo: str) -> dict[str, str]:
    return {
        "gm_clima": f"{root}/RFG5ac_gmoall-clim-topo_ctr-area_raw100_{suf}{gmo}",
        "gm_clima_coords": f"{root}/RFC1c_gmoall-clim-coords-topo_ctr-area_raw100_{suf}{gmo}",
        "gm_clima_spi": f"{root}/RFC2c_gmoall-clim-spi-topo_ctr-area_raw100_{suf}{gmo}",
        "gm_clima_coords_spi": f"{root}/RFC3c_gmoall-clim-coords-spi-topo_ctr-area_raw100_{suf}{gmo}",
        "curva": f"{root}/RFG2c_curve-topo_ctr-area_kndvi_raw100_{suf}",
        "curva_coords": f"{root}/RFC4c_curve-coords-topo_ctr-area_kndvi_raw100_{suf}",
    }


POOLS = [("pg_all_lenoso", runs(G, "pg-all_unified_woody", "_gmonc"), PG),
         ("unified_all_lenoso", runs(G, "unified-all_unified_woody", "_gmonc"), UNI),
         ("Forest", runs(S, "pg-all_unified_strat_11forest", "_gmonc"), ["lcbd_count_sorensen", "pd_inext_q0", "td_inext_q0"]),
         ("NoBosque", runs(S, "pg-all_unified_strat_nobosque", "_gmonc"), ["lcbd_count_sorensen", "pd_inext_q0", "td_inext_q0"])]
PAIRS = [("gm_clima_coords", "gm_clima"), ("gm_clima_spi", "gm_clima"),
         ("gm_clima_coords_spi", "gm_clima"), ("curva_coords", "curva")]


def r2(o, p):
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def main() -> None:
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "Owner", "source", "lat"]]
    spi = pd.read_parquet(ROOT / "data/derived/spi_unified.parquet")[[ID, "spi12_win_mean"]]
    s = spi.merge(plots[[ID, "lat"]], on=ID).dropna()
    b = np.polyfit(s.lat, s.spi12_win_mean, 2)
    s["spi_res"] = s.spi12_win_mean - np.polyval(b, s.lat)
    s["tercil_sequia"] = pd.qcut(s.spi_res, 3, labels=["seco", "medio", "humedo"]).astype(str)
    rows = []
    for pool, rr, targets in POOLS:
        oofs = {}
        for k, p in rr.items():
            f = ROOT / p / B / "oof_predictions.csv"
            if f.exists():
                oofs[k] = pd.read_csv(f).merge(plots, on=ID, how="left").merge(s[[ID, "tercil_sequia"]], on=ID, how="left")
        for t in targets:
            per = {}
            for k, o in oofs.items():
                if f"{t}_obs" not in o:
                    continue
                L = l100.lenses(o, t, 2.0)[["todas", "parcelas_cl", "living_trees", "pcl_dentro_owner", "lt_dentro_banda"]]
                d = o[o[f"{t}_obs"].notna() & o.tercil_sequia.notna()]
                for tc, g in d.groupby("tercil_sequia"):
                    L[f"sequia_{tc}"] = pd.Series({sd: r2(h[f"{t}_obs"], h[f"{t}_pred"]) for sd, h in g.groupby("seed")})
                per[k] = L
                for col in L.columns:
                    rows.append(dict(tipo="modelo", pool=pool, target=t, etiqueta=k, lectura=col,
                                     valor=L[col].mean(), DE=L[col].std(ddof=1)))
            for a, bb in PAIRS:
                if a in per and bb in per:
                    dd = per[a] - per[bb]
                    for col in dd.columns:
                        rows.append(dict(tipo="par", pool=pool, target=t, etiqueta=f"{a} − {bb}",
                                         lectura=col, valor=dd[col].mean(), DE=dd[col].std(ddof=1)))
    out = pd.DataFrame(rows)
    f = ROOT / "results/tables/test_coords_sequia.csv"
    out.to_csv(f, index=False)
    print(f"-> {f} ({len(out)} filas)")
    x = out[(out.tipo == "par") & out.lectura.isin(["todas", "lt_dentro_banda", "pcl_dentro_owner",
                                                     "sequia_seco", "sequia_humedo"])]
    key = ["lcbd_count_sorensen", "pd_inext_q0", "td_inext_q0", "hill_q0_unified", "lcbd_pa_unified"]
    print(x[x.target.isin(key)].pivot_table(index=["pool", "target", "etiqueta"], columns="lectura",
                                            values="valor").round(3).to_string())


if __name__ == "__main__":
    main()
