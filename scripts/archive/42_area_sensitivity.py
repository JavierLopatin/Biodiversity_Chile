#!/usr/bin/env python3
"""Cuanto cambia la riqueza predicha segun que area de referencia se elija.

RF06 y C2D02 usan `PlotSize_m2` (columna `area_log10`, `src/biodiv/features.py:235-240`)
como covariable de entrada. Un mapa se predice sobre pixeles continuos, que no tienen
"tamano de parcela" -- para generarlo hay que fijar un valor constante de area en todos
lados. La pregunta que importa (planteada por D. Craven): cuanto se mueve la riqueza
predicha segun que valor arbitrario se elija. Si se mueve mucho, el modelo no aprendio una
relacion area-riqueza estable, aprendio ruido -- coincide con el hallazgo de hoy de que la
correccion SAR (que remueve la varianza ligada a area del target) empeora el ajuste: el
modelo dependia de area como atajo.

Alcance: solo `kfold5_block20` (el esquema espacial "honesto" en distancia de hoy), y solo
hill_q0. Usa las parcelas reales de test de cada fold, sustituyendo unicamente su
covariable de area -- no construye infraestructura de mapa sobre el pool no etiquetado
(ese pool no tiene PlotSize_m2/stratum/topo).

Uso:
    python scripts/42_area_sensitivity.py --model rf
    python scripts/42_area_sensitivity.py --model cnn
    python scripts/42_area_sensitivity.py --model both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cv as cvmod              # noqa: E402
from biodiv import features as feat         # noqa: E402
from biodiv import models_tabular as mt     # noqa: E402
from biodiv import targets as tg            # noqa: E402

SCHEME = "kfold5_block20"
SPEC = "curve_all+topo+area"
TARGET = "hill_q0"
REF_AREAS_M2 = [100, 400, 1000, 10000]


def run_rf(derived: str, out_dir: Path) -> pd.DataFrame:
    ids = feat.plot_ids(derived)
    X_full, _ = feat.build_design(SPEC, index=None, derived=derived, ids=ids, px="mean5x5")
    _, Y_full, names = tg.load_targets(derived, "all", plot_ids=ids)
    j = names.index(TARGET)
    y = Y_full[:, j]

    cv = cvmod.load_schemes(Path(derived) / "cv_folds_modelling.parquet")
    pos = pd.Series(np.arange(len(ids)), index=ids)

    pre = feat.Preprocessor(standardise=False).fit(X_full, ids)
    area_col = pre.feature_names.index("area_log10")
    ref_log10 = {a: np.log10(a) for a in REF_AREAS_M2}

    rows = []
    for fold, held, tr_ids, te_ids in cvmod.iter_folds(cv, SCHEME):
        cvmod.assert_no_leak(tr_ids, te_ids, f"({SCHEME} fold {fold})")
        itr, ite = pos[tr_ids].to_numpy(), pos[te_ids].to_numpy()
        ytr = y[itr]
        ok = np.isfinite(ytr)
        if ok.sum() < 20:
            continue

        pre_f = feat.Preprocessor(standardise=False).fit(X_full, tr_ids)
        Xtr = pre_f.transform(X_full.loc[tr_ids])[ok]
        Xte_real = pre_f.transform(X_full.loc[te_ids])

        rf = mt.make_rf(seed=0, oob_score=False).fit(Xtr, ytr[ok])
        pred_real = rf.predict(Xte_real)

        pred_by_area = {}
        for a, la in ref_log10.items():
            Xte_swap = Xte_real.copy()
            Xte_swap[:, area_col] = la
            pred_by_area[a] = rf.predict(Xte_swap)

        block = pd.DataFrame({"plot_id": te_ids, "fold": fold,
                              "hill_q0_obs": y[ite],
                              "hill_q0_pred_real_area": pred_real})
        for a in REF_AREAS_M2:
            block[f"hill_q0_pred_area{a}"] = pred_by_area[a]
        rows.append(block)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_dir / "area_sensitivity_rf.csv", index=False)
    return out


def summarize(df: pd.DataFrame, model_label: str) -> None:
    ref_cols = [f"hill_q0_pred_area{a}" for a in REF_AREAS_M2]
    across_ref = df[ref_cols].std(axis=1)
    across_ref_range = df[ref_cols].max(axis=1) - df[ref_cols].min(axis=1)
    natural_spread = df["hill_q0_pred_real_area"].std()

    print(f"\n=== {model_label}: sensibilidad al area de referencia (n={len(df)}) ===")
    print(f"  desvio de la prediccion ENTRE areas de referencia, por parcela: "
         f"mediana={across_ref.median():.3f}  p90={across_ref.quantile(0.9):.3f}")
    print(f"  rango (max-min) entre areas de referencia, por parcela: "
         f"mediana={across_ref_range.median():.3f}  p90={across_ref_range.quantile(0.9):.3f}")
    print(f"  desvio NATURAL de la prediccion entre parcelas (area real): "
         f"{natural_spread:.3f}")
    print(f"  proporcion: {across_ref.median() / natural_spread:.1%} "
         f"(que fraccion del rango natural entre parcelas se mueve por elegir un area "
         f"de referencia distinta)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/tables")
    p.add_argument("--model", choices=["rf", "cnn", "both"], default="both")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("rf", "both"):
        df_rf = run_rf(args.derived, out_dir)
        summarize(df_rf, "RF06")

    if args.model in ("cnn", "both"):
        print("\n(CNN: correr por separado, ver scripts/42b_area_sensitivity_cnn.py)")


if __name__ == "__main__":
    main()
