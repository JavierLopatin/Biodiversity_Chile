#!/usr/bin/env python3
"""Emite en LaTeX las tablas de resultados del manuscrito desde los `pooled_metrics.csv`.

Se escriben a mano una sola vez y despues se regeneran: la Tabla 3 cambio tres veces en
un dia (reordenamiento de familias, arreglo de semilla, target armonizado a lenosas) y
cada vez habia que reescribir 28 celdas sin equivocarse. Generarlas evita el error de
transcripcion, que en una tabla de R2 no se nota al leer.

Uso:
    python scripts/83_emit_results_tables.py              # al stdout
    python scripts/83_emit_results_tables.py --supplement # las siete facetas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "models_unified_woody"

FAMILIES = {
    "Random Forest": "RF03pc_curve-topo_ctr-area_kndvi_raw100_pg-all_unified{}",
    "1D-CNN": "C1D01_curve1d_kndvi_raw100_pg-all_unified{}_ctr",
    "2D-CNN": "C2D02_serpentine_kndvi_raw100_pg-all_unified{}_ctr",
    "2D-CNN + MAE": "C2D02_serpentine_kndvi_raw100_pg-all_unified{}_maekndvi_m06_ctr",
}

#: Las tres variantes del target. `herbcommon` es la que aisla el efecto: mismas parcelas
#: que la lenosa, pero con el target calculado sobre la flora completa.
VARIANTS = [
    ("all vascular", ""),
    ("all vascular, common plots", "_herbcommon"),
    ("woody only", "_woody"),
]

SIGNAL = [("LCBD", "lcbd_count_sorensen"), (r"PD$_0$", "pd_inext_q0"),
          (r"TD$_0$", "td_inext_q0")]
ALL_FACETS = [("LCBD", "lcbd_count_sorensen"),
              (r"PD$_0$", "pd_inext_q0"), (r"PD$_1$", "pd_inext_q1"),
              (r"PD$_2$", "pd_inext_q2"), (r"TD$_0$", "td_inext_q0"),
              (r"TD$_1$", "td_inext_q1"), (r"TD$_2$", "td_inext_q2")]


def cell(run: str, key: str, bold: bool = False) -> tuple[str, float]:
    f = BASE / run / "kfold5_block20_unified" / "pooled_metrics.csv"
    if not f.exists():
        return "--", float("nan")
    m = pd.read_csv(f).set_index("target")
    if key not in m.index:
        return "--", float("nan")
    r = m.loc[key]
    body = f"{r.R2_mean:.3f}{{\\pm}}{r.R2_sd:.3f}"
    return (f"$\\mathbf{{{body}}}$" if bold else f"${body}$"), float(r.R2_mean)


def emit(facets, label: str) -> str:
    rows = []
    for vlabel, tag in VARIANTS:
        for i, (fname, key) in enumerate(facets):
            runs = [stem.format(tag) for stem in FAMILIES.values()]
            means = [cell(r, key)[1] for r in runs]
            # la negrita marca la media mas alta por fila; NaN nunca gana
            best = max(range(len(means)),
                       key=lambda j: means[j] if means[j] == means[j] else float("-inf"))
            cells = [cell(r, key, bold=(i == best))[0] for i, r in enumerate(runs)]
            first = f"\\multirow{{{len(facets)}}}{{*}}{{{vlabel}}}" if i == 0 else ""
            rows.append(f"{first} & {fname} & " + " & ".join(cells) + r" \\")
        if vlabel != VARIANTS[-1][0]:
            rows.append(r"\midrule")
    body = "\n".join(rows)
    head = " & ".join(FAMILIES)
    return (f"% generado por scripts/83_emit_results_tables.py -- no editar a mano\n"
            f"\\begin{{tabular}}{{ll rrrr}}\n\\toprule\n"
            f"Target & Facet & {head} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supplement", action="store_true")
    a = ap.parse_args()
    print(emit(ALL_FACETS if a.supplement else SIGNAL,
               "supplement" if a.supplement else "main"))


if __name__ == "__main__":
    main()
