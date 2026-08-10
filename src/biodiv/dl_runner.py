"""The shared run loop for every neural family: MLP, 1-D CNN, 2-D CNN.

One function, :func:`run_dl`, because the three families must differ **only** in how they
read the curve. Same targets, same masked loss, same target transform, same context vector,
same fusion point, same folds, same inner split, same early stopping, same scoring. If the
2-D model wins, that has to be attributable to the second dimension and not to a training
detail that happened to differ.

Per fold the sequence is fixed:

1. grouped inner split of the training fold, using the outer scheme's own grouping column;
2. fit the context scaler and the target power transform on ``fit`` only, never on ``val``,
   because the val loss decides when to stop;
3. train with early stopping, restore best weights;
4. back-transform test predictions with Duan smearing driven by the **inner-validation**
   residuals — the model did not fit those, so their spread is honest;
5. clip to the training range and record.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import cv as cvmod
from . import features as feat
from . import metrics as mx
from . import runlog
from . import substrates as sub
from . import targets as tg
from .models_conv import build_model, count_params
from .trainer import CurveDataset, TrainCfg, predict, train_one_fold

ID_COL = "PlotObservationID"


def substrate_builder(name: str, normalize: str = "none", **tf_kw):
    """``(B, C, 52) curves -> (B, C', ...)`` so augmented curves can be re-imaged per batch."""
    if name in ("curve1d", "curve5"):
        return lambda c: c
    if name in ("stack5", "pxcube"):
        return lambda c: c[:, None, :, :]

    from .transforms1d import make_transform
    tf = make_transform(name, normalize=normalize, **tf_kw)

    def build(c):
        imgs = np.stack([tf.transform(x) for x in c[:, 0, :]])
        return imgs[:, None] if imgs.ndim == 3 else np.transpose(imgs, (0, 3, 1, 2))
    return build


def _global_standardise(train_imgs: np.ndarray):
    """Per-channel standardisation with training-fold statistics.

    The alternative the vendored transforms ship with is per-sample min-max, which removes
    exactly the amplitude and mean-greenness information this project is trying to use. Doing
    it globally keeps between-plot differences intact while still giving the optimiser a
    well-conditioned input.
    """
    axes = (0,) + tuple(range(2, train_imgs.ndim))
    m = train_imgs.mean(axis=axes, keepdims=True)
    s = train_imgs.std(axis=axes, keepdims=True)
    s = np.where(s > 1e-8, s, 1.0)
    return lambda a: ((a - m) / s).astype(np.float32)


def run_dl(*, family: str, run_id: str, scheme: str, substrate: str = "curve1d",
           index: str | None = None, features_spec: str = "",
           ctx_spec: str = feat.CONTEXT_SPEC, width: str = "B", fusion: str = "late",
           rotation: str = "trough", normalize: str = "none", target_set: str = "all",
           px: str = "mean5x5",
           seeds: tuple[int, ...] = (0, 1, 2, 3, 4), derived: str = "data/derived",
           out_root: Path = runlog.RESULTS, train_cfg: TrainCfg | None = None,
           force: bool = False, notes: str = "", verbose: bool = False,
           save_state: bool = False, arch: str = "sep",
           p_conv: float = 0.1, p_head: float = 0.3) -> pd.DataFrame | None:
    cfg_t = train_cfg or TrainCfg()
    cfg = runlog.RunConfig(
        run_id=run_id, family=family, scheme=scheme,
        features=features_spec or ctx_spec, substrate=substrate, index=index or "",
        target_set=target_set, seeds=tuple(seeds), fusion=fusion,
        model=f"{family}-{width}",
        params=dict(rotation=rotation, normalize=normalize, ctx=ctx_spec, px=px,
                    loss=cfg_t.loss, lr=cfg_t.lr, batch_size=cfg_t.batch_size,
                    max_epochs=cfg_t.max_epochs, patience=cfg_t.patience,
                    augment=cfg_t.augment, mixup=cfg_t.mixup),
        notes=notes)
    if runlog.already_done(cfg, out_root) and not force:
        print(f"  [skip] {run_id} / {scheme}")
        return None

    t0 = time.time()
    ids = feat.plot_ids(derived)
    _, Y_full, names = tg.load_targets(derived, target_set, plot_ids=ids)
    n_out = len(names)

    # context vector: identical for every family, so the families stay comparable
    ctx_use = ctx_spec.replace("topo", "topo_ctr") if px == "center" else ctx_spec
    Xctx, _ = feat.build_design(ctx_use, index=index, derived=derived, ids=ids, px=px)

    if family == "MLP":
        Xtab, _ = feat.build_design(features_spec, index=index, derived=derived,
                                    ids=ids, px=px)
        curves = images = None
        builder = None
        pad_mode = "zeros"
        rows: list[str] = []
    else:
        s = sub.make_substrate(substrate, index=index, derived=derived, px=px,
                               normalize=normalize, rotation=rotation, ids=ids)
        curves, images, pad_mode, rows = s.curves, s.X, s.pad_mode, s.rows
        builder = substrate_builder(substrate, normalize=normalize)

    patches = None
    if fusion in ("patch", "patchctx"):
        patches, _ = sub.load_topo_patches(derived, ids)

    cv = cvmod.load_schemes(Path(derived) / "cv_folds_modelling.parquet")
    plots = feat.load_tables(derived).plots
    group_col = cvmod.SCHEME_GROUP[scheme]
    plots = cvmod.ensure_group_col(plots, group_col)
    pos = pd.Series(np.arange(len(ids)), index=ids)

    per_fold, n_params, hist_rows = [], None, []
    for seed in seeds:
        for fold, held, tr_ids, te_ids in cvmod.iter_folds(cv, scheme):
            cvmod.assert_no_leak(tr_ids, te_ids, f"({scheme} fold {fold})")
            fit_ids, val_ids = cvmod.inner_split(tr_ids, plots, group_col, seed=seed)
            cvmod.assert_no_leak(fit_ids, val_ids, "(inner split)")

            ifit, ival, ite = pos[fit_ids].to_numpy(), pos[val_ids].to_numpy(), pos[te_ids].to_numpy()
            itr = pos[tr_ids].to_numpy()

            pre_ctx = feat.Preprocessor(standardise=True).fit(Xctx, fit_ids)
            ctx_all = pre_ctx.transform(Xctx)

            scaler = tg.fit_target_scaler(Y_full[ifit])
            Ys = tg.apply_target_scaler(Y_full, scaler)
            mask = tg.target_mask(Y_full)

            if family == "MLP":
                pre_tab = feat.Preprocessor(standardise=True).fit(Xtab, fit_ids)
                Xin = pre_tab.transform(Xtab)
                # the MLP takes features and context as one flat vector
                Xin = np.concatenate([Xin, ctx_all], axis=1)
                ctx_in = np.zeros((len(ids), 0), dtype=np.float32)
                src = Xin[:, None, :]                 # dummy channel axis for the dataset
                imgs = Xin
                model = build_model("MLP", c_in=1, n_out=n_out, n_ctx=0, width=width,
                                    d_in=Xin.shape[1])
            else:
                norm = _global_standardise(images[ifit]) if normalize == "global" else (lambda a: a)
                imgs = norm(images)
                src = curves
                ctx_in = ctx_all if fusion in ("late", "film") else np.zeros((len(ids), 0), np.float32)
                model = build_model(family, arch=arch, p_conv=p_conv, p_head=p_head,
                                    c_in=imgs.shape[1], n_out=n_out,
                                    n_ctx=ctx_in.shape[1], width=width,
                                    pad_mode=pad_mode, fusion=fusion)

            if n_params is None:
                n_params = count_params(model)
                print(f"    {run_id}: {n_params:,} parameters, input {tuple(imgs.shape[1:])}")

            def ds(rows_idx, augment):
                return CurveDataset(
                    src[rows_idx], imgs[rows_idx], ctx_in[rows_idx], Ys[rows_idx],
                    mask[rows_idx],
                    patch=None if patches is None else patches[rows_idx],
                    build=(builder if family != "MLP" else None),
                    augment=augment and cfg_t.augment, aug=cfg_t.aug, seed=seed)

            model, hist, resid_val = train_one_fold(
                model, ds(ifit, True), ds(ival, False), cfg_t, seed=seed,
                use_patch=(fusion in ("patch", "patchctx")), verbose=verbose)
            pred_s = predict(model, ds(ite, False), cfg_t,
                             use_patch=(fusion in ("patch", "patchctx")))

            pred = tg.inverse_with_smearing(pred_s, scaler, resid_val,
                                            y_train=Y_full[itr], seed=seed)

            block = pd.DataFrame({ID_COL: te_ids, "fold": fold, "held_out": held, "seed": seed})
            for j, t in enumerate(names):
                block[f"{t}_obs"] = Y_full[ite, j]
                block[f"{t}_pred"] = pred[:, j]
            per_fold.append(block)
            hist_rows.append(pd.DataFrame(hist).assign(seed=seed, fold=fold))

            if save_state and seed == seeds[0] and fold == 0:
                out = cfg.outdir(out_root)
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "n_params": n_params,
                            "input_shape": tuple(imgs.shape[1:]), "rows": rows,
                            "targets": names},
                           out / f"model_seed{seed}_fold{fold}.pt")

    oof = cvmod.collect_oof(per_fold, scheme, names)
    per_seed = mx.pooled_metrics(oof, names, by=["seed"])
    ens_oof = mx.ensemble_oof(oof, names)
    ens = mx.compute_metrics(np.column_stack([ens_oof[f"{t}_pred"] for t in names]),
                             np.column_stack([ens_oof[f"{t}_obs"] for t in names]), names)
    pooled = runlog.summarise_pooled(per_seed, ens)
    cfg.params["n_params"] = n_params

    runlog.write_run(cfg, oof, per_seed, pooled,
                     extra={"history": pd.concat(hist_rows, ignore_index=True)},
                     root=out_root)
    dt = time.time() - t0
    r2 = " ".join(f"{r.target}={r.R2_mean:+.3f}" for r in pooled.itertuples())
    print(f"  {run_id:40s} {scheme:16s} [{dt / 60:5.1f}m] {r2}")
    return pooled
