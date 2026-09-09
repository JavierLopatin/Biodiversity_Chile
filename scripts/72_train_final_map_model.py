#!/usr/bin/env python3
"""Fit the deployable map model: the 2D-CNN+MAE config, trained on 100% of the unified
pool -- no held-out test fold, because a model meant for map inference has no "held-out
pixel", every plot is training signal for it.

SUPERSEDED as the deployed map model (2026-09-08): under the corrected topography this
config does not top any facet any more (`docs/20_pg_facets_unified.md` section 9.7) -- its
apparent wins on `pd_inext_q0`/`td_inext_q0` below were an artefact of the mixed aspect
convention section 9 documents, not real signal. The deployed model is now the 1D-CNN,
`scripts/77_train_final_map_model_c1d.py`. This script is kept to reproduce the 2D-CNN+MAE
path, not as instructions for what to deploy.

Winner, per `docs/20_pg_facets_unified.md` section 6 (pre-topofix, see above): kNDVI, raw
curves (`raw100`), centre pixel, `topo+area` context (real Living Trees topography,
`scripts/70`), `serpentine` substrate 2D-CNN initialised from the masked-autoencoder
checkpoint (`results/mae/mae_serpentine_kndvi_sep_wB_p2_m0.6.pt`), `pg_all` target set. It
topped `pd_inext_q0` (0.613) and `td_inext_q0` (0.786) under `kfold5_block20_unified` --
the two facets with real signal -- and was competitive (not worst) everywhere else.

Reuses the exact training call the CV runs use (`cv.inner_split`, `trainer.train_one_fold`,
the checkpoint dict shape from `dl_runner.run_dl`) so a checkpoint this script writes loads
identically to a fold checkpoint from `scripts/11_run_conv.py --save-state` (same keys:
state_dict/ctx_preprocessor/target_scaler/rows/targets), which is what
`scripts/13_interpretability.py` already knows how to reload for inference.

The only structural difference from `run_dl`: no outer test split. `inner_split` still
carves an internal validation set (grouped by `block20_unified`, same spatial grouping the
outer CV scheme itself uses) purely for early stopping -- so the final model never trains
on the epoch its own validation loss would have flagged as overfit, even though there is no
outer held-out score to report for this run.

Writes one checkpoint per seed to `results/models_unified/<run_id>/final/model_seed<k>.pt` --
multiple seeds so map inference can ensemble (mean) predictions, the same way OOF scoring
already does across folds.

`--all-data` fits on the full 3,102-plot pool with NO inner validation split at all -- the
preprocessor and target scaler are also fit on all 3,102 plots, not the ~80% `inner_split`
would carve out. Multitemporal map inference needs one model with no plot excluded from its
training set, so this mode drops early stopping and instead trains each seed for a FIXED
epoch budget (`--epochs`), with the cosine LR schedule set to complete over exactly that
budget (`biodiv.trainer.train_fixed_epochs`). Default budget (33) is the median *best*
epoch across the five early-stopped seeds in `logs/80_train_final_map_model.log`
(33/58/44/75/58 total epochs run before the patience=25 stop => best epoch = total -
patience = 8/33/19/50/33 => median 33) -- NOT the median of the raw logged totals (58),
which is how long training ran including the post-best patience wait, not how long it took
to reach the optimum. Pass `--epochs` explicitly to override. Writes to
`results/models_unified/<run_id>_alldata/final/` so the early-stopped `_FINAL` checkpoints
are untouched.

Usage:
    python scripts/72_train_final_map_model.py
    python scripts/72_train_final_map_model.py --seeds 3
    python scripts/72_train_final_map_model.py --all-data --epochs 33
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import cv as cvmod                                    # noqa: E402
from biodiv import features as feat                                # noqa: E402
from biodiv import substrates as sub                                # noqa: E402
from biodiv import targets as tg                                    # noqa: E402
from biodiv.dl_runner import substrate_builder                      # noqa: E402
from biodiv.mae import load_trunk                                   # noqa: E402
from biodiv.models_conv import build_model, count_params            # noqa: E402
from biodiv.trainer import (                                        # noqa: E402
    CurveDataset, TrainCfg, train_fixed_epochs, train_one_fold,
)

ID_COL = "PlotObservationID"

SUBSTRATE = "serpentine"
INDEX = "kndvi"
PX = "center"
TARGET_SET = "pg_all"
CTX_SPEC_BASE = feat.CONTEXT_SPEC       # "topo+area"
INIT_FROM = "results/mae/mae_serpentine_kndvi_sep_wB_p2_m0.6.pt"
GROUP_COL = "block20_unified"           # inner-split grouping, matches the outer CV scheme
RUN_TAG = "C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL"

# Median *best* epoch (not median of the raw logged totals) across the 5 early-stopped
# seeds in logs/80_train_final_map_model.log -- see module docstring for the derivation.
DEFAULT_ALLDATA_EPOCHS = 33


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/models_unified")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=300, dest="max_epochs")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--all-data", action="store_true", dest="all_data",
                   help="fit on 100%% of plots, no inner validation split, fixed epoch "
                        "budget instead of early stopping -- for the deployment/map model")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"fixed epoch budget for --all-data (default {DEFAULT_ALLDATA_EPOCHS}, "
                        "see module docstring for how it was derived)")
    args = p.parse_args()
    if args.epochs is not None and not args.all_data:
        raise SystemExit("--epochs only applies to --all-data")

    if not feat.unified_flag():
        raise SystemExit("set BIODIV_UNIFIED=1 -- this model is trained on the unified pool")
    if feat.curve_suffix() != "_raw100":
        raise SystemExit("set BIODIV_CURVES=_raw100 -- this is the winning raw-curve config")

    if args.all_data:
        epochs = args.epochs if args.epochs is not None else DEFAULT_ALLDATA_EPOCHS
        run_tag = RUN_TAG + "_alldata"
    else:
        run_tag = RUN_TAG

    derived = Path(args.derived)
    out_root = Path(args.out) / run_tag / "final"
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ids = feat.plot_ids(derived)
    _, Y_full, names = tg.load_targets(derived, TARGET_SET, plot_ids=ids)
    n_out = len(names)
    print(f"pool: {len(ids)} parcelas, {n_out} facetas objetivo: {names}")

    ctx_spec = CTX_SPEC_BASE.replace("topo", "topo_ctr")  # PX == "center"
    Xctx, _ = feat.build_design(ctx_spec, index=INDEX, derived=derived, ids=ids, px=PX)

    s = sub.make_substrate(SUBSTRATE, index=INDEX, derived=derived, px=PX,
                           normalize="none", rotation="trough", ids=ids)
    curves, images, pad_mode, rows = s.curves, s.X, s.pad_mode, s.rows
    builder = substrate_builder(SUBSTRATE, normalize="none")

    plots = feat.load_tables(derived).plots
    plots = cvmod.ensure_group_col(plots, GROUP_COL)

    tr_ids = ids  # every plot is training data -- no outer held-out fold
    if args.all_data:
        cfg_t = TrainCfg(max_epochs=epochs, patience=args.patience)
        print(f"--all-data: fit on all {len(tr_ids)} plots, fixed epoch budget={epochs}, "
             f"no inner validation split")
    else:
        cfg_t = TrainCfg(max_epochs=args.max_epochs, patience=args.patience)

    for seed in range(args.seeds):
        if args.all_data:
            # No inner split at all -- fit_ids IS every plot, preprocessor/scaler included.
            fit_ids, val_ids = tr_ids, []
            pos = pd.Series(np.arange(len(ids)), index=ids)
            ifit = pos[fit_ids].to_numpy()
        else:
            fit_ids, val_ids = cvmod.inner_split(tr_ids, plots, GROUP_COL, seed=seed)
            cvmod.assert_no_leak(fit_ids, val_ids, "(inner split)")
            pos = pd.Series(np.arange(len(ids)), index=ids)
            ifit, ival = pos[fit_ids].to_numpy(), pos[val_ids].to_numpy()

        pre_ctx = feat.Preprocessor(standardise=True).fit(Xctx, fit_ids)
        ctx_all = pre_ctx.transform(Xctx)

        scaler = tg.fit_target_scaler(Y_full[ifit])
        Ys = tg.apply_target_scaler(Y_full, scaler)
        mask = tg.target_mask(Y_full)

        imgs = images  # normalize="none" -> no per-fold standardisation of the image
        model = build_model("C2D", arch="sep", p_conv=0.1, p_head=0.3,
                            c_in=imgs.shape[1], n_out=n_out, n_ctx=ctx_all.shape[1],
                            width="B", pad_mode=pad_mode, fusion="late")

        ck = torch.load(INIT_FROM, map_location="cpu", weights_only=False)
        n_loaded = load_trunk(model, ck["trunk"])
        if seed == 0:
            n_params = count_params(model)
            print(f"{run_tag}: {n_params:,} parametros, input {tuple(imgs.shape[1:])}, "
                 f"init from {Path(INIT_FROM).name}: {n_loaded} trunk tensors")

        def ds(rows_idx, augment):
            return CurveDataset(curves[rows_idx], imgs[rows_idx], ctx_all[rows_idx],
                                Ys[rows_idx], mask[rows_idx], patch=None, build=builder,
                                augment=augment and cfg_t.augment, aug=cfg_t.aug, seed=seed)

        if args.all_data:
            model, hist = train_fixed_epochs(
                model, ds(ifit, True), cfg_t, seed=seed, verbose=False)
            final_loss = hist[-1]["train_loss"]
            print(f"  seed {seed}: {len(hist)} epocas (fijo), train_loss final={final_loss:.4f}, "
                 f"fit={len(fit_ids)}")
        else:
            model, hist, resid_val = train_one_fold(
                model, ds(ifit, True), ds(ival, False), cfg_t, seed=seed, verbose=False)
            best_val = min(h["val_loss"] for h in hist)
            print(f"  seed {seed}: {len(hist)} epocas, mejor val_loss={best_val:.4f}, "
                 f"fit={len(fit_ids)} val={len(val_ids)}")

        torch.save({"state_dict": model.state_dict(), "n_params": count_params(model),
                    "input_shape": tuple(imgs.shape[1:]), "rows": rows,
                    "targets": names, "ctx_preprocessor": pre_ctx,
                    "target_scaler": scaler, "fit_ids": list(fit_ids),
                    "train_ids": list(tr_ids), "test_ids": []},
                   out_root / f"model_seed{seed}.pt")

    dt = (time.time() - t0) / 60
    print(f"\n-> {out_root}  ({args.seeds} checkpoints, {dt:.1f}m)")
    print("Para inferencia: cargar cada model_seed*.pt, predecir, promediar (ensemble) "
         "sobre las semillas -- mismo patron que mx.ensemble_oof usa entre folds.")


if __name__ == "__main__":
    main()
