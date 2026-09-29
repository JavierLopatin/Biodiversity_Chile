#!/usr/bin/env python3
"""El control que decide hierbas contra lenoso: 60 parcelas cargan todo el R2 de riqueza.

Con hierbas, la riqueza parecia predecible: TD0 0,675 con la curva y 0,678 en la tanda anterior,
contra ~0 en lenoso. De ahi salio la idea de que "con hierbas la fenologia predice riqueza", y
esa idea estuvo a punto de decidir el alcance del paper.

Es falso, y este script lo mide. Tres hechos encadenados:

1. El AREA SOLA es la mejor spec de todo el factorial con hierbas para riqueza: TD0 0,825, por
   encima del clima (0,821), de gm+clima (0,807) y de la curva (0,675). Un predictor que no mira
   el satelite le gana a todos los que si.
2. El area sola funciona porque etiqueta un protocolo. Las unicas parcelas de Parcelas-CL de
   exactamente 500 m2 son las 60 de Becerra, y su TD0 medio es 49,5 contra 6,6-8,9 en los demas
   tamanos de PCL y 7,3-7,4 en Living Trees. Es censo de flora completa contra censo lenoso.
3. Y las 60 estan en UNA sola localidad, Coquimbo. Por eso el clima tambien las encuentra: no
   necesita fenologia, le basta la posicion. El area etiqueta el protocolo y el protocolo
   coincide con un punto del mapa.

La prueba es quitarlas. Son el 6,7 % de las parcelas con TD0 y al sacarlas TODAS las
representaciones caen a cero o por debajo -- el area a -0,006, el clima a -0,085, la curva a
-1,998. Un R2 que vive entero en el 6,7 % de la muestra no es una relacion ecologica.

En lenoso el artefacto no existe, y no porque se corrigiera: al armonizar a lenoso las parcelas
de Becerra pasan a tener riqueza lenosa normal, el contraste de protocolo desaparece y el R2 de
riqueza es ~0 desde el principio. El ~0 del pool lenoso es el resultado honesto; el 0,68 con
hierbas era el artefacto.

Salida: `results/tables/control_becerra.csv`.

Uso:
    python scripts/106_becerra_control.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
OUT = ROOT / "results" / "tables" / "control_becerra.csv"
SCHEME = "kfold5_block20_unified"

#: El dueno cuyo protocolo difiere. No se elige por tener R2 alto: se elige porque es el unico
#: grupo de PCL con censo de flora completa, y eso se ve en `PlotSize_m2` antes de mirar ningun
#: modelo.
OWNER = "Becerra, P.I."

SPECS = {
    "area_sola":      "B02c_area_raw100",
    "piso_topo_area": "B01c_topo_ctr-area_raw100",
    "clima":          "RFG1c_clim-topo_ctr-area_raw100",
    "curva":          "RFG2c_curve-topo_ctr-area_kndvi_raw100",
    "gm":             "RFG4c_gm-topo_ctr-area_raw100",
    "gm_clima":       "RFG5c_gm-clim-topo_ctr-area_raw100",
    "lsp":            "RFG7c_lspu-topo_ctr-area_kndvi_raw100",
}
TARGETS = ["td_inext_q0", "pd_inext_q0", "lcbd_count_sorensen"]


def r2(o, p) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def tabla_tamanos(ids: set) -> pd.DataFrame:
    """TD0 observado por fuente y tamano de parcela: el mecanismo, antes de cualquier modelo."""
    d = pd.read_csv(GATE / f"{SPECS['area_sola']}_pg-all_unified" / SCHEME /
                    "oof_predictions.csv")
    d = (d[d.td_inext_q0_obs.notna()].groupby("PlotObservationID")
         .td_inext_q0_obs.first().rename("TD0").reset_index())
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "Owner", "PlotSize_m2", "source", "Location"]]
    m = d.merge(p, on="PlotObservationID", how="left")
    g = (m.groupby(["source", "PlotSize_m2"])
         .agg(n=("TD0", "size"), TD0_mean=("TD0", "mean"), TD0_median=("TD0", "median"))
         .reset_index())
    g["es_becerra"] = [
        bool(len(m[(m.source == s) & (m.PlotSize_m2 == a) & (m.Owner == OWNER)]) == n)
        for s, a, n in zip(g.source, g.PlotSize_m2, g.n)]
    return g


def main() -> None:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[["PlotObservationID", "Owner"]]
    ids = set(p.loc[p.Owner == OWNER, "PlotObservationID"])
    if not ids:
        raise SystemExit(f"Ninguna parcela con Owner == {OWNER!r}: revisa la grafia en "
                         "plots_unified.parquet antes de leer este control.")

    rows = []
    for nombre, spec in SPECS.items():
        for variante, sfx in [("con_hierbas", "_pg-all_unified"),
                              ("lenoso", "_pg-all_unified_woody")]:
            f = GATE / f"{spec}{sfx}" / SCHEME / "oof_predictions.csv"
            if not f.exists():
                rows.append(dict(spec=nombre, variante=variante, target="", n=np.nan,
                                 n_becerra=np.nan, R2_todas=np.nan, R2_sin_becerra=np.nan,
                                 delta=np.nan, estado="FALTA"))
                continue
            d = pd.read_csv(f)
            for t in TARGETS:
                if f"{t}_obs" not in d:
                    continue
                z = d[d[f"{t}_obs"].notna()]
                if not len(z):
                    continue
                w = z[~z.PlotObservationID.isin(ids)]
                a = float(np.mean([r2(g[f"{t}_obs"], g[f"{t}_pred"])
                                   for _, g in z.groupby("seed")]))
                b = float(np.mean([r2(g[f"{t}_obs"], g[f"{t}_pred"])
                                   for _, g in w.groupby("seed")]))
                rows.append(dict(spec=nombre, variante=variante, target=t,
                                 n=z.PlotObservationID.nunique(),
                                 n_becerra=len(set(z.PlotObservationID) & ids),
                                 R2_todas=a, R2_sin_becerra=b, delta=b - a, estado="ok"))

    f = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT, index=False)
    tabla_tamanos(ids).to_csv(OUT.with_name("td0_por_tamano_parcela.csv"), index=False)
    print(f"-> {OUT}  ({len(f)} filas)")
    print(f"-> {OUT.with_name('td0_por_tamano_parcela.csv')}")

    w = f[(f.target == "td_inext_q0") & (f.variante == "con_hierbas") & (f.estado == "ok")]
    print("\nTD0 con hierbas -- el R2 vive en el 6,7 % de la muestra:")
    print(w[["spec", "n", "n_becerra", "R2_todas", "R2_sin_becerra", "delta"]]
          .sort_values("R2_todas", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
