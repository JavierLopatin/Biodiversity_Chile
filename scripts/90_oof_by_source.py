"""R² del OOF evaluado por fuente, dentro de contribuyente y dentro de bandas de latitud.

Una validación por bloque espacial sobre un pool que agrega inventarios heterogéneos puede
dar R² alto porque el modelo aprende de qué fuente o de qué contribuyente viene cada
parcela, no la relación ecológica. Este script lee los oof_predictions.csv ya escritos,
sin reajustar nada, y separa tres lecturas del mismo modelo:

1. Por fuente: R² y rho de Spearman sobre las filas de Parcelas-CL y de Living Trees por
   separado, cada una contra su propia media.
2. Dentro de contribuyente (solo Parcelas-CL): observado y predicho centrados por Owner
   dentro de cada semilla. Mide lo que el modelo predice una vez quitada la diferencia de
   nivel entre estudios (protocolo, tamaño de parcela, región). Reporta además la fracción
   de varianza del target que explica el Owner.
3. Dentro de bandas de latitud (solo Living Trees): el mismo centrado, por bandas de
   `--band` grados. Quita el gradiente norte-sur grueso.

Métricas por semilla, promediadas (media ± DE). No son las del ensemble, que difieren en
el tercer decimal.

Uso:
    python scripts/90_oof_by_source.py \\
        --run results/models_gate/RFG1c_clim-topo_ctr-area_raw100_unified-all_unified_woody \\
        --targets hill_q0_unified dark_n_unified pcoa2_pa_unified
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "PlotObservationID"


def r2(o: pd.Series, p: pd.Series) -> float:
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def centred_r2(o: pd.Series, p: pd.Series) -> float:
    """R² de valores ya centrados por grupo: la referencia es 0, no la media global."""
    return float(1 - ((o - p) ** 2).sum() / (o ** 2).sum())


def per_seed(df: pd.DataFrame, fn, cols=("obs", "pred")) -> tuple[float, float]:
    vals = [fn(g[cols[0]], g[cols[1]]) for _, g in df.groupby("seed")]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def rho(o: pd.Series, p: pd.Series) -> float:
    return float(o.corr(p, method="spearman"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path,
                    help="directorio del run (el que contiene <scheme>/oof_predictions.csv)")
    ap.add_argument("--scheme", default="kfold5_block20_unified")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--band", type=float, default=2.0, help="ancho de banda de latitud, grados")
    ap.add_argument("--derived", default="data/derived")
    args = ap.parse_args()

    oof = pd.read_csv(args.run / args.scheme / "oof_predictions.csv")
    plots = pd.read_parquet(Path(args.derived) / "plots_unified.parquet")[
        [ID_COL, "Owner", "source", "lat"]]
    oof = oof.merge(plots, on=ID_COL, how="left")
    assert oof.source.notna().all(), "parcelas del OOF sin fila en plots_unified"

    rows = []
    for t in args.targets:
        d = oof[oof[f"{t}_obs"].notna()].rename(columns={f"{t}_obs": "obs", f"{t}_pred": "pred"})
        for src, g in [("todas", d), ("parcelas_cl", d[d.source == "parcelas_cl"]),
                       ("living_trees", d[d.source == "living_trees"])]:
            m, s = per_seed(g, r2)
            rows.append(dict(target=t, lectura=src, n=g[ID_COL].nunique(),
                             R2=m, R2_sd=s, rho=per_seed(g, rho)[0]))

        pcl = d[d.source == "parcelas_cl"].copy()
        one = pcl.drop_duplicates(ID_COL)
        eta = 1 - ((one.obs - one.groupby("Owner").obs.transform("mean")) ** 2).sum() \
            / ((one.obs - one.obs.mean()) ** 2).sum()
        for c in ("obs", "pred"):
            pcl[c + "_c"] = pcl[c] - pcl.groupby(["seed", "Owner"])[c].transform("mean")
        m, s = per_seed(pcl, centred_r2, ("obs_c", "pred_c"))
        rows.append(dict(target=t, lectura="parcelas_cl dentro de Owner",
                         n=one[ID_COL].nunique(), R2=m, R2_sd=s,
                         rho=per_seed(pcl, rho, ("obs_c", "pred_c"))[0],
                         var_owner=float(eta)))

        lt = d[d.source == "living_trees"].copy()
        lt["banda"] = np.floor(lt.lat / args.band)
        for c in ("obs", "pred"):
            lt[c + "_c"] = lt[c] - lt.groupby(["seed", "banda"])[c].transform("mean")
        m, s = per_seed(lt, centred_r2, ("obs_c", "pred_c"))
        rows.append(dict(target=t, lectura=f"living_trees dentro de bandas de {args.band:g} grados",
                         n=lt[ID_COL].nunique(), R2=m, R2_sd=s,
                         rho=per_seed(lt, rho, ("obs_c", "pred_c"))[0]))

    out = pd.DataFrame(rows)
    print(f"{args.run.name} / {args.scheme}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    f = args.run / args.scheme / "oof_by_source.csv"
    out.to_csv(f, index=False)
    print(f"-> {f}")


if __name__ == "__main__":
    main()
