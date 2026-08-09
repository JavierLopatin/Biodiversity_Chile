#!/usr/bin/env python3
"""Read the screening and GDM tables together and say what the evidence supports.

Two tables answer two different questions and must not be averaged into one ranking:

  ``results/tables/block_screen.csv``   which predictor block, judged by RF R2 on the beta
                                        targets, under a given CV scheme
  ``results/tables/gdm_comparison.csv`` which *model family*, judged by rank agreement with
                                        the observed dissimilarity on held-out pairs

The report prints each on its own terms and then states the three comparisons that decide
what to build next:

  **X03 vs X01** — does phenological shape beat an annual composite?
  **X03 vs X04** — does the composite add anything *on top of* the curve?
  **X17 vs X03 and X16** — does phenology survive once climate is in the model, and does it
  add over climate alone? Geographic distance alone already reaches Spearman +0.435 against
  observed dissimilarity, so this is the contrast that decides whether the remote sensing is
  carrying an environmental signal or standing in for one.

Usage:
    python scripts/23_screen_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)

#: Reference row of the screening matrix: the 52-step curve plus context.
REF = "X03"


def contrast(df: pd.DataFrame, a: str, b: str, scheme: str, label: str) -> None:
    """Print one head-to-head with its seed-level spread, so the sign is readable."""
    s = df[df["scheme"] == scheme].set_index("row")
    if a not in s.index or b not in s.index:
        print(f"  {label:52s} (falta {a if a not in s.index else b})")
        return
    ra, rb = s.loc[a], s.loc[b]
    d = ra["R2_beta"] - rb["R2_beta"]
    noise = float(max(ra.get("R2_beta_sd", 0) or 0, rb.get("R2_beta_sd", 0) or 0))
    verdict = "sin diferencia" if abs(d) <= noise else ("gana " + (a if d > 0 else b))
    print(f"  {label:52s} {a}={ra['R2_beta']:+.3f}  {b}={rb['R2_beta']:+.3f}  "
          f"dif={d:+.3f} (ruido entre semillas {noise:.3f})  -> {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screen", default="results/tables/block_screen.csv")
    p.add_argument("--gdm", default="results/tables/gdm_comparison.csv")
    args = p.parse_args()

    screen = Path(args.screen)
    if screen.exists():
        df = pd.read_csv(screen)
        for scheme in df["scheme"].unique():
            s = df[df["scheme"] == scheme].sort_values("R2_beta", ascending=False)
            ref = s[s["row"] == REF]["R2_beta"]
            head = f"===== cribado RF — {scheme}"
            if len(ref):
                head += f"   (referencia {REF}, la curva = {float(ref.iloc[0]):+.3f})"
            print(f"\n{head}")
            cols = ["row", "n_features", "R2_beta", "R2_beta_sd", "R2_alpha",
                    "lcbd_pa", "pcoa1_pa", "pcoa2_pa", "question"]
            cols = [c for c in cols if c in s.columns]
            print(s[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

        print("\n===== los contrastes que deciden")
        for scheme in df["scheme"].unique():
            print(f"\n  -- {scheme}")
            contrast(df, "X03", "X01", scheme,
                     "la forma supera al compuesto anual?")
            contrast(df, "X04", "X03", scheme,
                     "el compuesto aporta SOBRE la curva?")
            contrast(df, "X17", "X16", scheme,
                     "la fenologia aporta SOBRE el clima?")
            contrast(df, "X17", "X03", scheme,
                     "el clima aporta SOBRE la fenologia?")
            contrast(df, "X07", "X03", scheme,
                     "la geomediana + MADs alcanzan a la curva?")
            contrast(df, "X10", "X03", scheme,
                     "4 numeros estacionales alcanzan a los 52 pasos?")
    else:
        print(f"(sin {screen})")

    gdm = Path(args.gdm)
    if gdm.exists():
        g = pd.read_csv(gdm)
        print("\n\n===== familias de modelo — rho de Spearman contra el Jaccard observado,")
        print("      sobre los pares entre parcelas excluidas de cada fold")
        rho = [c for c in g.columns if c.endswith("_rho")]
        agg = (g.groupby(["scheme", "spec"])[rho].mean()
               .rename(columns=lambda c: c[:-4]))
        print(agg.to_string(float_format=lambda v: f"{v:+.3f}"))
        print("\n  'oracle2' y 'oracle' no son modelos: son los ejes PCoA verdaderos de las")
        print("  parcelas excluidas, es decir el techo que cualquier regresion sobre ejes")
        print("  esta persiguiendo. 'geo' es GDM con solo distancia geografica: el numero")
        print("  que la teledeteccion tiene que superar, no el modelo nulo.")
    else:
        print(f"\n(sin {gdm})")


if __name__ == "__main__":
    main()
