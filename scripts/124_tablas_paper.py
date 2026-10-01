#!/usr/bin/env python3
"""Las dos tablas del texto principal, en CSV y en LaTeX listo para \\input.

    T2  que aporta cada bloque de predictores sobre el modelo nulo geografico, faceta por
        faceta. Es la tabla que acompana a la figura central y reemplaza a `tab:families`.
    T3  arquitectura: el Random Forest contra las tres redes convolucionales, las cuatro
        contra el mismo nulo. Reemplaza a `tab:curveformat`.

Las dos reportan el MARGEN sobre el nulo geografico (lon, lat, elevacion) con el R2 centrado
dentro de bandas de 2 grados, no el R2 agrupado, porque un R2 agrupado sobre 26 grados de
latitud es en buena parte el gradiente. La columna del nulo va al lado para que cada margen se
lea contra algo.

Los asteriscos son los q de Benjamini-Hochberg de `scripts/118`, corregidos dentro de la
familia de pruebas que corresponde a cada tabla -- T2 usa la familia `margen` para las dos
columnas de reflectancia y `representacion` para las otras tres; T3 usa `arquitectura`. El
criterio y por que no se corrigen las 151 juntas estan en `scripts/118`.

Ningun numero esta escrito a mano: todo sale de `sintesis_lenoso.csv` y `significancia_margen.csv`.

Uso:
    python scripts/124_tablas_paper.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import facetas_paper as FP          # noqa: E402


#: Columnas de T2: clave interna -> encabezado. El orden va de lo que no es teledeteccion a lo
#: que si, para que la tabla se lea como una escalera.
COLS_T2 = [("clima", "Climate"), ("curva", "Phenology"), ("lsp", "LSP"),
           ("gm", "Reflectance"), ("gm_clima", "Reflectance + climate")]
COLS_T3 = [("gm_clima", "Random Forest"), ("cnn_1d", "CNN 1D"),
           ("cnn_2d", "CNN 2D"), ("cnn_2d_mae", "CNN 2D (MAE)")]

FAMILIAS = FP.FAMILIAS
LAB = FP.LAB


def celda(r: pd.Series | None) -> str:
    if r is None or pd.isna(r.margen):
        return "---"
    e = "" if pd.isna(r.sig) else str(r.sig)
    return f"${r.margen:+.3f}$\\textsuperscript{{{e}}}" if e else f"${r.margen:+.3f}$"


def tabla(sig: pd.DataFrame, cols: list[tuple[str, str]], facetas: list[str],
          por_familia: bool, titulo: str, etiqueta: str, nota: str) -> str:
    idx = sig.set_index(["faceta", "modelo"])
    nulo = sig.drop_duplicates("faceta").set_index("faceta").R2_nulo
    enc = " & ".join(["Facet", "Null $R^2$"] + [c[1] for c in cols])
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           f"\\caption{{{titulo}}}", f"\\label{{{etiqueta}}}",
           r"\begin{tabular}{l" + "r" * (len(cols) + 1) + "}", r"\toprule",
           enc + r" \\", r"\midrule"]
    grupos = FAMILIAS if por_familia else [("", facetas)]
    primero = True
    for nombre, fs in grupos:
        fs = [f for f in fs if f in facetas]
        if not fs:
            continue
        if nombre:
            if not primero:
                out.append(r"\addlinespace")
            out.append(f"\\multicolumn{{{len(cols) + 2}}}{{l}}{{\\textit{{{nombre}}}}} \\\\")
        primero = False
        for f in fs:
            fila = [LAB.get(f, f), f"${nulo.get(f, float('nan')):.3f}$"]
            for k, _ in cols:
                fila.append(celda(idx.loc[(f, k)] if (f, k) in idx.index else None))
            out.append(" & ".join(fila) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{\textwidth}\footnotesize\vspace{2pt}" + nota + r"\end{minipage}",
            r"\end{table}"]
    return "\n".join(out)


def main() -> None:
    sig = pd.read_csv(TAB / "significancia_margen.csv")
    orden = [f for _, fs in FAMILIAS for f in fs]

    nota = (r"Margin in out-of-fold $R^2$ over the geographic null model (longitude, latitude, "
            r"elevation), with observed and predicted values centred within 2\textdegree{} "
            r"latitude bins. Positive values mean the block adds over knowing where the plot "
            r"is. Significance from a one-sided block bootstrap over the same 20\,km blocks "
            r"that define the cross-validation, Benjamini--Hochberg within each question: "
            r"\textsuperscript{*}$q<0.05$, \textsuperscript{**}$q<0.01$, "
            r"\textsuperscript{***}$q<0.001$.")

    t2 = tabla(sig, COLS_T2, orden, True,
               "What each predictor block adds over knowing where the plot is.",
               "tab:margins", nota)
    (TAB / "T2_margenes.tex").write_text(t2 + "\n")

    con_cnn = sorted(sig.loc[sig.modelo == "cnn_1d", "faceta"].unique(),
                     key=lambda f: orden.index(f) if f in orden else 99)
    t3 = tabla(sig, COLS_T3, con_cnn, False,
               "Architecture: the Random Forest against three convolutional networks, all "
               "against the same geographic null model.",
               "tab:architecture", nota)
    (TAB / "T3_arquitectura.tex").write_text(t3 + "\n")

    for nombre, cols, fs in (("T2", COLS_T2, orden), ("T3", COLS_T3, con_cnn)):
        w = (sig[sig.faceta.isin(fs)].pivot_table(index="faceta", columns="modelo",
                                                  values="margen")
             .reindex(index=fs, columns=[c[0] for c in cols]))
        w.insert(0, "nulo", sig.drop_duplicates("faceta").set_index("faceta").R2_nulo.reindex(fs))
        w.to_csv(TAB / f"{nombre}_margenes.csv" if nombre == "T2"
                 else TAB / f"{nombre}_arquitectura.csv")
        print(f"\n=== {nombre} ===")
        print(w.round(3).to_string())
    print(f"\n-> {TAB / 'T2_margenes.tex'}\n-> {TAB / 'T3_arquitectura.tex'}")


if __name__ == "__main__":
    main()
