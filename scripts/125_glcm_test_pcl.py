#!/usr/bin/env python3
"""¿La textura GLCM aporta sobre el geomediano dentro de Parcelas-CL? Lectura del test de descarte.

Corridas (results/models_glcm, kfold5_block20_unified restringido a PCL porque los targets
de las LT están en NaN, 3 semillas):
  pg_all con BIODIV_TARGETS=_woodypcl   LCBD e iNEXT recalculados dentro de PCL (n 226-479)
  unified_all con _woodypclm            facetas leñosas del pool amplio en las 1.082 PCL
Modelos: B03 (coordenadas), RFG4a (gm), RFG5a (gm+clima), RFT0 (gm+nivel 5x5), RFT1
(gm+nivel+textura 5x5), RFT2 (gm+nivel+textura 3x3), RFT3 (gm+clima+nivel+textura 5x5),
RFT4 (gm+clima+nivel 5x5).

Pares, diferencia de R² pareada por semilla, en dos lecturas: PCL entero y dentro de
contribuyente (Owner, observado y predicho centrados):
  textura sobre nivel      RFT1 − RFT0, RFT3 − RFT4     <- la pregunta
  nivel de ventana sobre gm RFT0 − RFG4a
  textura 3x3              RFT2 − RFG4a
  todo contra coordenadas  RFT1 − B03, RFT3 − B03

Y la tabla de confusión con el nivel: por banda, propiedad y ventana, rho de Spearman entre
la textura y el nivel de la ventana, y la correlación parcial (Spearman, por rangos) de la
textura con cada target controlando por ese nivel.

Escribe results/tables/glcm_pcl_modelos.csv y glcm_pcl_correlaciones.csv.

Uso:
    python scripts/125_glcm_test_pcl.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"
B = "kfold5_block20_unified"
M = ROOT / "results/models_glcm"
_s = importlib.util.spec_from_file_location("l100", ROOT / "scripts" / "100_paired_lenses.py")
l100 = importlib.util.module_from_spec(_s); _s.loader.exec_module(l100)

PAIRS = [("RFT1", "RFT0"), ("RFT3", "RFT4"), ("RFT0", "RFG4a"), ("RFT2", "RFG4a"),
         ("RFT1", "RFG4a"), ("RFT1", "B03"), ("RFT3", "B03"), ("RFG5a", "B03")]
FAMILIES = {"pg_all_pcl": ("pg-all_unified_woodypcl", ["lcbd_count_sorensen", "td_inext_q0", "pd_inext_q0"]),
            "unified_all_pclm": ("unified-all_unified_woodypclm",
                                 ["hill_q0_unified", "lcbd_pa_unified", "pcoa1_pa_unified",
                                  "pcoa2_pa_unified", "mpd_unified", "ses_mpd_unified"])}


def run_dir(model: str, suf: str) -> Path | None:
    hits = [p for p in M.glob(f"{model}c_*_{suf}*") if (p / B / "oof_predictions.csv").exists()]
    hits = [p for p in hits if p.name.split("_raw100_")[-1].split("_gmo")[0] == suf]
    return hits[0] if len(hits) == 1 else None


def partial_rank(x, y, z) -> float:
    """Spearman parcial: correlación de los residuos de rank(x) y rank(y) contra rank(z)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 20:
        return np.nan
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    Z = np.c_[np.ones(ok.sum()), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> None:
    plots = pd.read_parquet(ROOT / "data/derived/plots_unified.parquet")[[ID, "Owner", "source", "lat"]]
    rows = []
    for fam, (suf, targets) in FAMILIES.items():
        oofs = {}
        for m in ("B03", "RFG4a", "RFG5a", "RFT0", "RFT1", "RFT2", "RFT3", "RFT4"):
            d = run_dir(m, suf)
            if d is not None:
                oofs[m] = pd.read_csv(d / B / "oof_predictions.csv").merge(plots, on=ID, how="left")
        for t in targets:
            L = {m: l100.lenses(o[o.source == "parcelas_cl"], t, 2.0)[["parcelas_cl", "pcl_dentro_owner", "n_parcelas_cl"]]
                 for m, o in oofs.items() if f"{t}_obs" in o}
            for m, x in L.items():
                for le in ("parcelas_cl", "pcl_dentro_owner"):
                    rows.append(dict(tipo="modelo", familia=fam, target=t, etiqueta=m, lectura=le,
                                     n=int(x.n_parcelas_cl.iloc[0]), valor=x[le].mean(), DE=x[le].std(ddof=1)))
            for a, b in PAIRS:
                if a in L and b in L:
                    dd = L[a] - L[b]
                    for le in ("parcelas_cl", "pcl_dentro_owner"):
                        rows.append(dict(tipo="par", familia=fam, target=t, etiqueta=f"{a} − {b}",
                                         lectura=le, n=int(L[a].n_parcelas_cl.iloc[0]),
                                         valor=dd[le].mean(), DE=dd[le].std(ddof=1)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/glcm_pcl_modelos.csv", index=False)

    g = pd.read_parquet(ROOT / "data/derived/glcm_pcl.parquet")
    y = g[[ID]].merge(pd.read_parquet(ROOT / "data/derived/unified_diversity_responses_woody.parquet")
                      [[ID, "hill_q0_unified", "lcbd_pa_unified"]], on=ID, how="left") \
               .merge(pd.read_parquet(ROOT / "data/derived/lcbd_count_sorensen_unified_padded_woodypcl.parquet"),
                      on=ID, how="left")
    crow = []
    for c in [c for c in g.columns if c.startswith("tex")]:
        w, band, prop = c[3], c.split("_")[1], c.split("_", 2)[2]
        lvl = g[f"lvl{w}_{band}"].to_numpy(float)
        x = g[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(lvl)
        r = dict(ventana=f"{w}x{w}", banda=band, propiedad=prop,
                 rho_con_nivel=pd.Series(x[ok]).corr(pd.Series(lvl[ok]), method="spearman"))
        for t in ("hill_q0_unified", "lcbd_pa_unified", "lcbd_count_sorensen"):
            v = y[t].to_numpy(float)
            okt = ok & np.isfinite(v)
            r[f"rho_{t}"] = pd.Series(x[okt]).corr(pd.Series(v[okt]), method="spearman")
            r[f"parcial_{t}"] = partial_rank(x, v, lvl)
        crow.append(r)
    C = pd.DataFrame(crow)
    C.to_csv(ROOT / "results/tables/glcm_pcl_correlaciones.csv", index=False)

    pd.set_option("display.width", 220)
    p = out[out.tipo == "par"].pivot_table(index=["familia", "target", "etiqueta"], columns="lectura",
                                           values="valor").round(3)
    print(p.to_string())
    print(out[out.tipo == "modelo"].pivot_table(index=["familia", "target"], columns=["lectura", "etiqueta"],
                                                values="valor").round(3).T.to_string())
    print(C.groupby(["ventana", "propiedad"])[["rho_con_nivel", "rho_hill_q0_unified",
                                              "parcial_hill_q0_unified", "parcial_lcbd_pa_unified"]]
          .agg(lambda s: s.abs().max()).round(2).to_string())


if __name__ == "__main__":
    main()
