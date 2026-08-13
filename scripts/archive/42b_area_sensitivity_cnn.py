#!/usr/bin/env python3
"""C2D02, sensibilidad al area de referencia -- fold 0 de kfold5_block20 unicamente.

Companero de `scripts/42_area_sensitivity.py` (ver su docstring para el porque). No hay
checkpoint reutilizable: `dl_runner.run_dl` guarda pesos para seed[0]/fold=0 pero no el
normalizador del contexto ni el escalador del target, y ninguno existe bajo
`kfold5_block20` (solo `kfold5_window`). Replica en vivo el tramo de
`src/biodiv/dl_runner.py::run_dl` para fold=0/seed=0 con la config de C2D02 (defaults de
`scripts/11_run_conv.py`: substrate=serpentine, index=kndvi, context=topo+area,
arch=sep, width=B, fusion=late) -- entrena una vez, predice dos veces (area real y area
sustituida) con el mismo modelo en memoria. No toca `dl_runner.py`.

Uso:
    python scripts/42b_area_sensitivity_cnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cv as cvmod                              # noqa: E402
from biodiv import features as feat                          # noqa: E402
from biodiv import substrates as sub                          # noqa: E402
from biodiv import targets as tg                              # noqa: E402
from biodiv.dl_runner import substrate_builder                # noqa: E402
from biodiv.models_conv import build_model                    # noqa: E402
from biodiv.trainer import CurveDataset, TrainCfg, predict, train_one_fold  # noqa: E402

SCHEME = "kfold5_block20"
SUBSTRATE = "serpentine"
INDEX = "kndvi"
CTX_SPEC = "topo+area"
TARGET = "hill_q0"
REF_AREAS_M2 = [100, 400, 1000, 10000]
DERIVED = "data/derived"


def main() -> None:
    ids = feat.plot_ids(DERIVED)
    _, Y_full, names = tg.load_targets(DERIVED, "all", plot_ids=ids)
    j = names.index(TARGET)
    n_out = len(names)

    Xctx, _ = feat.build_design(CTX_SPEC, index=None, derived=DERIVED, ids=ids, px="mean5x5")
    area_col = list(Xctx.columns).index("area_log10")

    s = sub.make_substrate(SUBSTRATE, index=INDEX, derived=DERIVED, px="mean5x5",
                           normalize="none", rotation="trough", ids=ids)
    curves, images, pad_mode, rows = s.curves, s.X, s.pad_mode, s.rows
    builder = substrate_builder(SUBSTRATE, normalize="none")

    cv = cvmod.load_schemes(Path(DERIVED) / "cv_folds_modelling.parquet")
    plots = feat.load_tables(DERIVED).plots
    group_col = cvmod.SCHEME_GROUP[SCHEME]
    plots = cvmod.ensure_group_col(plots, group_col)
    pos = pd.Series(np.arange(len(ids)), index=ids)

    fold, held, tr_ids, te_ids = next(iter(cvmod.iter_folds(cv, SCHEME)))
    cvmod.assert_no_leak(tr_ids, te_ids, f"({SCHEME} fold {fold})")
    seed = 0
    fit_ids, val_ids = cvmod.inner_split(tr_ids, plots, group_col, seed=seed)
    cvmod.assert_no_leak(fit_ids, val_ids, "(inner split)")

    ifit, ival, ite = pos[fit_ids].to_numpy(), pos[val_ids].to_numpy(), pos[te_ids].to_numpy()
    itr = pos[tr_ids].to_numpy()

    pre_ctx = feat.Preprocessor(standardise=True).fit(Xctx, fit_ids)
    ctx_all = pre_ctx.transform(Xctx)

    scaler = tg.fit_target_scaler(Y_full[ifit])
    Ys = tg.apply_target_scaler(Y_full, scaler)
    mask = tg.target_mask(Y_full)

    model = build_model("C2D", arch="sep", p_conv=0.1, p_head=0.3,
                        c_in=images.shape[1], n_out=n_out, n_ctx=ctx_all.shape[1],
                        width="B", pad_mode=pad_mode, fusion="late")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"C2D02 fold 0: {n_params:,} parametros, input {tuple(images.shape[1:])}")

    def ds(rows_idx, ctx, augment):
        return CurveDataset(curves[rows_idx], images[rows_idx], ctx[rows_idx],
                            Ys[rows_idx], mask[rows_idx], patch=None,
                            build=builder, augment=augment, seed=seed)

    cfg_t = TrainCfg()
    print(f"entrenando (max_epochs={cfg_t.max_epochs}, patience={cfg_t.patience})...")
    model, hist, resid_val = train_one_fold(
        model, ds(ifit, ctx_all, True), ds(ival, ctx_all, False), cfg_t, seed=seed)
    print(f"  {len(hist)} epocas")

    pred_s_real = predict(model, ds(ite, ctx_all, False), cfg_t)
    pred_real = tg.inverse_with_smearing(pred_s_real, scaler, resid_val,
                                         y_train=Y_full[itr], seed=seed)[:, j]

    out = pd.DataFrame({"plot_id": te_ids, "fold": fold,
                        "hill_q0_obs": Y_full[ite, j],
                        "hill_q0_pred_real_area": pred_real})

    for a in REF_AREAS_M2:
        Xctx_swap = Xctx.copy()
        Xctx_swap.iloc[:, area_col] = np.log10(a)
        ctx_swap = pre_ctx.transform(Xctx_swap)
        pred_s = predict(model, ds(ite, ctx_swap, False), cfg_t)
        pred = tg.inverse_with_smearing(pred_s, scaler, resid_val,
                                        y_train=Y_full[itr], seed=seed)[:, j]
        out[f"hill_q0_pred_area{a}"] = pred

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "area_sensitivity_cnn.csv", index=False)

    ref_cols = [f"hill_q0_pred_area{a}" for a in REF_AREAS_M2]
    across_ref = out[ref_cols].std(axis=1)
    across_ref_range = out[ref_cols].max(axis=1) - out[ref_cols].min(axis=1)
    natural_spread = out["hill_q0_pred_real_area"].std()
    print(f"\n=== C2D02 (fold 0, {SCHEME}): sensibilidad al area (n={len(out)}) ===")
    print(f"  desvio ENTRE areas de referencia, por parcela: mediana={across_ref.median():.3f}")
    print(f"  rango (max-min) entre areas: mediana={across_ref_range.median():.3f}")
    print(f"  desvio natural entre parcelas (area real): {natural_spread:.3f}")
    print(f"  proporcion: {across_ref.median() / natural_spread:.1%}")


if __name__ == "__main__":
    main()
