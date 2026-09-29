#!/usr/bin/env python3
"""El factorial completo: representacion x hierbas x cobertura x estres hidrico.

Una sola tabla larga con las cuatro dimensiones que el analisis fue abriendo, y con los
huecos declarados como filas y no como ausencias. Un hueco que no aparece es un hueco que
nadie ve: la variante con hierbas solo se corrio con la curva, y esa ausencia estuvo
sosteniendo la idea de que con hierbas la fenologia predice PD0 y TD0 -- cuando en lenoso,
para PD0, el clima le gana a la curva.

Dimensiones:
    representacion   piso | clima | curva | lsp | gm | gm+clima | clima+curva
    hierbas          con_hierbas | lenoso
    cobertura        pool | Forest | NoBosque
    estres           todas | seco | medio | humedo   (tercil de SPI residualizado
                     contra latitud dentro de cada ano de censo, solo Living Trees)

El estres sale del OOF, no de corridas aparte: es una particion de las predicciones ya
hechas. Por eso se puede cruzar con todo lo demas sin multiplicar el computo.

Salida: `results/tables/factorial.csv`.

Uso:
    python scripts/105_factorial.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
STRATA = ROOT / "results" / "models_strata"
UNIW = ROOT / "results" / "models_unified_woody"
OUT = ROOT / "results" / "tables" / "factorial.csv"
SCHEME = "kfold5_block20_unified"

REPS = {
    "piso":        ("B01c",  "topo_ctr-area_raw100"),
    "clima":       ("RFG1c", "clim-topo_ctr-area_raw100"),
    "curva":       ("RFG2c", "curve-topo_ctr-area_kndvi_raw100"),
    "clima_curva": ("RFG3c", "clim-curve-topo_ctr-area_kndvi_raw100"),
    "gm":          ("RFG4c", "gm-topo_ctr-area_raw100"),
    "gm_clima":    ("RFG5c", "gm-clim-topo_ctr-area_raw100"),
    "lsp":         ("RFG7c", "lspu-topo_ctr-area_kndvi_raw100"),
}
#: (carpeta, sufijo del run_id, cobertura, variante de hierbas). La unica corrida RF con
#: hierbas vive en models_unified_woody y con prefijo RF03pc, no RFG2c: es de la tanda
#: anterior a la compuerta espectral y nunca se rehizo con los demas bloques.
POOLS = [(GATE, "pg-all_unified_woody", "pool", "lenoso"),
         (UNIW, "pg-all_unified", "pool", "con_hierbas"),
         (STRATA, "pg-all_unified_strat_11forest", "Forest", "lenoso"),
         (STRATA, "pg-all_unified_strat_nobosque", "NoBosque", "lenoso")]
TARGETS = ["lcbd_count_sorensen", "pd_inext_q0", "td_inext_q0"]
MIN_N_TERCIL = 40


def r2(o, p) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def contexto() -> pd.DataFrame:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lat", "Year", "source"]]
    spi = pd.read_parquet(DERIVED / "spi_unified.parquet")
    if spi.index.name:
        spi = spi.reset_index()
    col = [c for c in spi.columns if "spi12" in c and "mean" in c][0]
    return p.merge(spi[["PlotObservationID", col]], on="PlotObservationID",
                   how="left").rename(columns={col: "spi"})


def terciles(g: pd.DataFrame) -> pd.Series:
    """Tercil de SPI dentro del ano, quitandole antes el gradiente latitudinal.

    El SPI dentro de un ano es en dos tercios latitud (rho -0,67): partir por tercil de SPI
    crudo seria partir por latitud y volver a medir el gradiente que este proyecto ya
    descarto cuatro veces."""
    out = pd.Series(index=g.index, dtype="float64")
    for _, gy in g.groupby("Year"):
        if len(gy) < 3 * MIN_N_TERCIL or gy.lat.nunique() < 3:
            continue
        res = gy.spi - np.polyval(np.polyfit(gy.lat, gy.spi, 1), gy.lat)
        out.loc[gy.index] = pd.qcut(res, 3, labels=[0, 1, 2]).astype(float)
    return out


def main() -> None:
    ctx = contexto()
    rows = []
    for carpeta, sufijo, pool, hierbas in POOLS:
        for rep, (fam, feats) in REPS.items():
            fam_ = "RF03pc" if (hierbas == "con_hierbas" and rep == "curva") else fam
            d = carpeta / f"{fam_}_{feats}_{sufijo}" / SCHEME / "oof_predictions.csv"
            if not d.exists():
                for t in TARGETS:
                    rows.append(dict(representacion=rep, hierbas=hierbas, cobertura=pool,
                                     estres="todas", target=t, n=np.nan, R2=np.nan,
                                     estado="FALTA"))
                continue
            o = pd.read_csv(d).groupby("PlotObservationID").mean(numeric_only=True)
            e = o.join(ctx.set_index("PlotObservationID"))
            for t in TARGETS:
                oc, pc = f"{t}_obs", f"{t}_pred"
                if oc not in e:
                    continue
                z = e[e[oc].notna()]
                rows.append(dict(representacion=rep, hierbas=hierbas, cobertura=pool,
                                 estres="todas", target=t, n=len(z), R2=r2(z[oc], z[pc]),
                                 estado="ok"))
                lt = z[(z.source == "living_trees") & z.spi.notna()].copy()
                lt["t"] = terciles(lt)
                for k, nombre in enumerate(["seco", "medio", "humedo"]):
                    s = lt[lt.t == k]
                    if len(s) < MIN_N_TERCIL:
                        continue
                    rows.append(dict(representacion=rep, hierbas=hierbas, cobertura=pool,
                                     estres=nombre, target=t, n=len(s),
                                     R2=r2(s[oc], s[pc]), estado="ok"))

    f = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"-> {OUT}  ({len(f)} filas, {int((f.estado == 'FALTA').sum())} celdas faltantes)")

    w = f[(f.estres == "todas") & (f.target == "lcbd_count_sorensen")]
    print("\nLCBD, R2 global (NaN = corrida que falta):")
    print(w.pivot_table(index="representacion", columns=["hierbas", "cobertura"],
                        values="R2", dropna=False).round(3).to_string())


if __name__ == "__main__":
    main()
