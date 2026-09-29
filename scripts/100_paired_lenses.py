#!/usr/bin/env python3
"""R² de varias corridas bajo las lecturas de scripts/90, con diferencias pareadas por semilla.

Para cada corrida y target, R² por semilla en cinco lecturas del mismo OOF:
  todas                     el pool completo
  parcelas_cl / living_trees cada fuente contra su propia media
  pcl_dentro_owner          PCL con observado y predicho centrados por contribuyente
  lt_dentro_banda           LT centrado por bandas de 2° de latitud

Después, para cada par (a, b) pedido, la diferencia R²_a − R²_b pareada por semilla (media
± DE de las 3 diferencias). Con los mismos folds y semillas es el test correcto: la DE de la
diferencia pareada es mucho más ajustada que la dispersión de cada media.

Todo predictor que se veía fuerte a escala de pool resultó ser gradiente entre fuentes o
latitud (dark_n, pcoa2, hill_q0, riqueza PCL). Este script aplica el mismo control a
cualquier corrida nueva antes de leerla como hallazgo.

Uso:
    python scripts/100_paired_lenses.py --target lcbd_count_sorensen \\
        --run gm=results/models_gate/RFG4ac_... --run clima=results/models_gate/RFG1c_... \\
        --pair gm:clima
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"


def r2(o, p):
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def cr2(o, p):
    return float(1 - ((o - p) ** 2).sum() / (o ** 2).sum())


SLOPE_BINS, SLOPE_LABELS = [-0.01, 5, 15, 25, 90], ["<5", "5-15", "15-25", ">25"]


def lenses(oof: pd.DataFrame, t: str, band: float, by_slope: bool = False) -> pd.DataFrame:
    """R² por semilla en las cinco lecturas. Con ``by_slope`` se repite dentro de cada clase
    de pendiente del DEM (columna ``slope``), y el índice pasa a (slope_cls, seed)."""
    d = oof[oof[f"{t}_obs"].notna()].rename(columns={f"{t}_obs": "o", f"{t}_pred": "p"})
    if by_slope:
        d = d.assign(slope_cls=pd.cut(d.slope, SLOPE_BINS, labels=SLOPE_LABELS).astype(str))
        parts = [lenses_plain(d, band).assign(slope_cls="todas")] + [
            lenses_plain(g, band).assign(slope_cls=sc) for sc, g in d.groupby("slope_cls")]
        return pd.concat(parts).reset_index().set_index(["slope_cls", "seed"])
    return lenses_plain(d, band)


def lenses_plain(d: pd.DataFrame, band: float) -> pd.DataFrame:
    rows = []
    for seed, g in d.groupby("seed"):
        pcl, lt = g[g.source == "parcelas_cl"].copy(), g[g.source == "living_trees"].copy()
        out = {"todas": r2(g.o, g.p), "parcelas_cl": r2(pcl.o, pcl.p) if len(pcl) > 2 else np.nan,
               "living_trees": r2(lt.o, lt.p) if len(lt) > 2 else np.nan,
               "n_todas": g[ID].nunique(), "n_parcelas_cl": pcl[ID].nunique(),
               "n_living_trees": lt[ID].nunique()}
        if len(pcl) > 2:
            for c in ("o", "p"):
                pcl[c + "c"] = pcl[c] - pcl.groupby("Owner")[c].transform("mean")
            out["pcl_dentro_owner"] = cr2(pcl.oc, pcl.pc)
        if len(lt) > 2:
            lt["b"] = np.floor(lt.lat / band)
            for c in ("o", "p"):
                lt[c + "c"] = lt[c] - lt.groupby("b")[c].transform("mean")
            out["lt_dentro_banda"] = cr2(lt.oc, lt.pc)
        rows.append({"seed": seed, **out})
    return pd.DataFrame(rows).set_index("seed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--run", action="append", required=True, help="etiqueta=directorio del run")
    ap.add_argument("--pair", action="append", default=[], help="a:b para R²_a − R²_b pareado")
    ap.add_argument("--scheme", default="kfold5_block20_unified")
    ap.add_argument("--band", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    plots = pd.read_parquet(ROOT / "data" / "derived" / "plots_unified.parquet")[[ID, "Owner", "source", "lat"]]
    per = {}
    for spec in a.run:
        lab, path = spec.split("=", 1)
        oof = pd.read_csv(Path(path) / a.scheme / "oof_predictions.csv").merge(plots, on=ID, how="left")
        per[lab] = lenses(oof, a.target, a.band)
    mean = pd.DataFrame({k: v.mean() for k, v in per.items()}).T
    sd = pd.DataFrame({k: v.std() for k, v in per.items()}).T
    print(f"== {a.target}: R² medio (DE entre semillas) ==")
    print((mean.round(3).astype(str) + " (" + sd.round(3).astype(str) + ")").to_string())
    rows = []
    for pr in a.pair:
        x, y = pr.split(":")
        d = per[x] - per[y]
        for lens in d.columns:
            rows.append(dict(par=f"{x} - {y}", lectura=lens, dif=d[lens].mean(), dif_sd=d[lens].std()))
    if rows:
        pt = pd.DataFrame(rows)
        pt["txt"] = pt.dif.map("{:+.3f}".format) + " ± " + pt.dif_sd.map("{:.3f}".format)
        print("\n== diferencias pareadas por semilla ==")
        print(pt.pivot(index="par", columns="lectura", values="txt").to_string())
        if a.out:
            pt.to_csv(a.out, index=False)
    if a.out:
        mean.to_csv(Path(a.out).with_name(Path(a.out).stem + "_medias.csv"))


if __name__ == "__main__":
    main()
