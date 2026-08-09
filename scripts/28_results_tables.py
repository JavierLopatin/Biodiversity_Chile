#!/usr/bin/env python3
"""Las tablas de resultados: R2, %RMSE y sesgo, por faceta y por combinacion.

`12_model_report.py` compara modelos y hace los tests pareados; esto es lo otro que un
paper necesita: la tabla completa de metricas, para cada corrida y cada target, bajo el
esquema primario.

**Un solo esquema.** `kfold5_window` es la validacion del proyecto (docs/10_findings.md 1b).
Las filas de `kfold5_owner`, `kfold5_random` y `kfold5_block20` que quedan en `summary.csv`
son de la tanda anterior y solo se usan para la brecha de optimismo, que sigue viviendo en
`12_model_report.py`. Aqui no aparecen.

**Tres metricas, no una.** Un R2 alto con sesgo grande es un modelo que acierta la forma de
la nube y falla el nivel -- y con targets que se transforman con Yeo-Johnson y se
retransforman con el estimador de Duan, el sesgo es exactamente donde eso se rompe. El
%RMSE dice cuanto error hay en unidades del rango del target, que es lo unico comparable
entre facetas cuando una vive en 1-50 especies y otra en 1e-3.

**El sesgo se normaliza igual que el RMSE.** `Bias` esta en unidades del target, asi que
promediarlo entre targets no significa nada. `nBias = Bias / (p99 - p1)` usa el mismo
denominador que `nRMSE`, se recupera de `summary.csv` como `Bias * nRMSE / RMSE`, y ya es
promediable. Se reporta ademas su valor absoluto: promediar sesgos con signo entre targets
los cancela y hace parecer insesgado un conjunto de modelos que no lo es.

Salidas en `results/tables/`:

  results_full.csv         una fila por (corrida, target): R2, %RMSE, sesgo, n, Spearman
  results_by_facet.csv     una fila por (corrida, faceta): las tres metricas promediadas
  results_best.csv         el mejor modelo de cada target, con las tres metricas
  results_tables.md        las mismas, en markdown, para pegar en el doc

Uso:
    python scripts/28_results_tables.py
    python scripts/28_results_tables.py --scheme kfold5_owner   # solo para comparar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import targets as tg                # noqa: E402

PRIMARY = "kfold5_window"
KEYS = ["run_id", "family", "features", "index", "substrate", "fusion", "model"]


def load(summary: Path, scheme: str) -> pd.DataFrame:
    s = pd.read_csv(summary)
    s = s[s["scheme"] == scheme].copy()
    if s.empty:
        raise SystemExit(f"no hay filas con scheme={scheme!r} en {summary}")
    # rango del target = RMSE / nRMSE; se recupera para poder normalizar el sesgo con el
    # mismo denominador y que sea comparable entre facetas
    with np.errstate(divide="ignore", invalid="ignore"):
        s["nBias"] = s["Bias"] * s["nRMSE"] / s["RMSE"]
    s["facet"] = s["target"].map(tg.FACET_OF)
    unknown = sorted(set(s.loc[s["facet"].isna(), "target"]))
    if unknown:
        raise SystemExit(f"targets sin faceta asignada en biodiv.targets.FACETS: {unknown}")
    return s


def full_table(s: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (corrida, target). Media y sd entre semillas."""
    g = s.groupby(KEYS + ["facet", "target"], dropna=False)
    out = g.agg(n=("n", "first"),
                seeds=("seed", "nunique"),
                R2=("R2", "mean"), R2_sd=("R2", "std"),
                nRMSE=("nRMSE", "mean"), RMSE=("RMSE", "mean"),
                Bias=("Bias", "mean"), nBias=("nBias", "mean"),
                spearman=("spearman", "mean")).reset_index()
    out["R2_sd"] = out["R2_sd"].fillna(0.0)
    return out.sort_values(["facet", "target", "R2"], ascending=[True, True, False])


def by_facet(full: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (corrida, faceta).

    `nBias_abs` promedia |sesgo|, no el sesgo: un modelo que sobreestima un target y
    subestima otro por la misma cantidad tiene sesgo medio cero y no es insesgado.
    """
    f = full.assign(nBias_abs=full["nBias"].abs())
    g = f.groupby(KEYS + ["facet"], dropna=False)
    out = g.agg(n_targets=("target", "nunique"),
                R2=("R2", "mean"), R2_min=("R2", "min"), R2_max=("R2", "max"),
                nRMSE=("nRMSE", "mean"),
                nBias=("nBias", "mean"), nBias_abs=("nBias_abs", "mean"),
                spearman=("spearman", "mean")).reset_index()
    return out.sort_values(["facet", "R2"], ascending=[True, False])


def best_per_target(full: pd.DataFrame) -> pd.DataFrame:
    idx = full.groupby("target")["R2"].idxmax()
    cols = ["facet", "target", "run_id", "family", "features", "index", "substrate",
            "n", "seeds", "R2", "R2_sd", "nRMSE", "nBias", "spearman"]
    return full.loc[idx, cols].sort_values(["facet", "target"])


#: columnas que ya vienen en porcentaje de rango y se leen con un decimal, no con tres
PCT_COLS = {"%RMSE", "sesgo %", "|sesgo|"}


def _md(df: pd.DataFrame, floats: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            nd = 1 if c in PCT_COLS else floats
            d[c] = d[c].map(lambda v, nd=nd: "" if pd.isna(v) else f"{v:.{nd}f}")
    d = d.fillna("")
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write_markdown(path: Path, scheme: str, full: pd.DataFrame, facet: pd.DataFrame,
                   best: pd.DataFrame) -> None:
    n_runs = full["run_id"].nunique()
    parts = [
        f"# Resultados — `{scheme}`",
        "",
        f"{n_runs} corridas x {full['target'].nunique()} targets. Generado por "
        "`scripts/28_results_tables.py`; no editar a mano.",
        "",
        "`%RMSE` es el RMSE dividido por el rango 1–99 del target, y `sesgo` la media del "
        "residuo con el mismo denominador. Los dos son porcentajes de rango, así que se "
        "pueden comparar entre facetas; el R² y el Spearman no necesitan normalización.",
        "",
        "## El mejor modelo de cada target",
        "",
        _md(best.assign(**{"%RMSE": (100 * best["nRMSE"]).round(1),
                           "sesgo %": (100 * best["nBias"]).round(1)})
                .drop(columns=["nRMSE", "nBias"])),
        "",
        "## Por faceta — las 10 mejores corridas de cada una",
        "",
    ]
    for fname in tg.FACETS:
        sub = facet[facet["facet"] == fname].head(10)
        if sub.empty:
            continue
        parts += [f"### `{fname}`", "",
                  _md(sub[["run_id", "family", "index", "substrate", "n_targets",
                           "R2", "R2_min", "R2_max", "nRMSE", "nBias_abs", "spearman"]]
                      .rename(columns={"nRMSE": "%RMSE", "nBias_abs": "|sesgo|"})
                      .assign(**{"%RMSE": (100 * sub["nRMSE"]).round(1),
                                 "|sesgo|": (100 * sub["nBias_abs"]).round(1)})),
                  ""]
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default=PRIMARY)
    p.add_argument("--models", default="results/models")
    p.add_argument("--tables", default="results/tables")
    args = p.parse_args()

    tdir = ROOT / args.tables
    tdir.mkdir(parents=True, exist_ok=True)
    s = load(ROOT / args.models / "summary.csv", args.scheme)

    full = full_table(s)
    facet = by_facet(full)
    best = best_per_target(full)

    suffix = "" if args.scheme == PRIMARY else f"_{args.scheme}"
    full.to_csv(tdir / f"results_full{suffix}.csv", index=False)
    facet.to_csv(tdir / f"results_by_facet{suffix}.csv", index=False)
    best.to_csv(tdir / f"results_best{suffix}.csv", index=False)
    write_markdown(tdir / f"results_tables{suffix}.md", args.scheme, full, facet, best)

    print(f"esquema {args.scheme}: {full['run_id'].nunique()} corridas, "
          f"{full['target'].nunique()} targets, {len(full)} filas\n")
    cov = (full.groupby("facet")
           .agg(corridas=("run_id", "nunique"), targets=("target", "nunique"),
                R2_max=("R2", "max"), R2_mediana=("R2", "median")))
    print(cov.round(3).to_string())
    rel = tdir.relative_to(ROOT) if tdir.is_relative_to(ROOT) else tdir
    print(f"\n  -> {rel}/results_{{full,by_facet,best}}{suffix}.csv")
    print(f"  -> {rel}/results_tables{suffix}.md")


if __name__ == "__main__":
    main()
