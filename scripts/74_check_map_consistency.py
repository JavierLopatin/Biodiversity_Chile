#!/usr/bin/env python3
"""Gate before any map: the map-inference path must reproduce the training path.

Runs on the machine that holds the unified curves and the all-data checkpoints (no
datacube needed). Three checks, each printed with a pass/fail line:

1. **Scaler round trip.** The pickled `PowerTransformer` was written by an older
   scikit-learn; ``inverse_transform(transform(y)) == y`` on the training targets shows
   the unpickled object still computes what it did when fitted (LCBD, with lambda about
   -4200, is the sensitive channel).
2. **Same inputs.** For every training plot, the context row built by
   `mapinfer.context_frame` from the topography table equals the row `features.build_design`
   produced for training, and the serpentine image built by `mapinfer.images_from_curves`
   equals the one `substrates.make_substrate` produced.
3. **Same outputs.** `FacetEnsemble.predict_scaled` on those inputs equals a direct call
   of the reloaded model on the training tensors, and the seed-mean back-transformed
   prediction correlates with the observed facets at the level the block-CV run reported
   (an in-sample number, so it should be at least as high; a collapse means the pipeline,
   not the model, is broken).

Usage (on the pod / rapidita):
    BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python scripts/74_check_map_consistency.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import features as feat        # noqa: E402
from biodiv import mapinfer as mi          # noqa: E402
from biodiv import substrates as sub       # noqa: E402
from biodiv import targets as tg           # noqa: E402

DEFAULT_CKPT = ("results/models_unified/"
                "C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL_alldata/final")


def ok(flag: bool, msg: str) -> bool:
    print(("PASS  " if flag else "FAIL  ") + msg)
    return flag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT, dest="ckpt_dir")
    args = p.parse_args()
    derived = Path(args.derived)
    if not feat.unified_flag() or feat.curve_suffix() != "_raw100":
        raise SystemExit("set BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100")

    ckpts = sorted(Path(args.ckpt_dir).glob("model_seed*.pt"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ens = mi.FacetEnsemble(ckpts)
    vers = [str(x.message)[:90] for x in w if "Version" in type(x.message).__name__]
    print(f"{len(ckpts)} checkpoints; targets {ens.targets}")
    if vers:
        print(f"note: {len(vers)} unpickling warnings, e.g. {vers[0]}")

    # ---- training inputs, the training way ------------------------------------------
    ids = feat.plot_ids(derived)
    _, Y, names = tg.load_targets(derived, list(ens.targets), plot_ids=ids)
    Xctx, _ = feat.build_design("topo_ctr+area", index="kndvi", derived=derived, ids=ids,
                                px="center")
    s = sub.make_substrate("serpentine", index="kndvi", derived=derived, px="center",
                           normalize="none", rotation="trough", ids=ids)
    images_train = np.ascontiguousarray(s.X, dtype=np.float32)
    curves_train = s.curves[:, 0, :]
    print(f"{len(ids)} plots, images {images_train.shape}, ctx {Xctx.shape}")

    all_ok = True

    # 1. scaler round trip
    sc = ens.members[0].scaler
    yf = np.where(np.isfinite(Y), Y, np.nanmedian(Y, axis=0))
    back = sc.inverse_transform(sc.transform(yf))
    rel = np.nanmax(np.abs(back - yf) / (np.abs(yf) + 1e-12), axis=0)
    all_ok &= ok(bool((rel < 1e-6).all()), f"scaler round trip, max rel err per target {rel}")

    # 2a. context
    tables = feat.load_tables(derived)
    topo = tables.topo.reindex(ids)
    plots = tables.plots.set_index(feat.ID_COL).reindex(ids)
    frames = []
    for st in feat.STRATA:
        m = (plots["stratum"].astype(str) == st).to_numpy()
        if not m.any():
            continue
        tp = {v: topo[v].to_numpy(float)[m] for v in feat.TOPO_VARS}
        fr = mi.context_frame(ens.context_columns, tp, 1.0, st)
        fr["area_log10"] = plots["log10_area"].to_numpy(float)[m]
        fr.index = ids[m]
        frames.append(fr)
    ctx_map = pd.concat(frames).reindex(ids)
    diff = (ctx_map.to_numpy(float) - Xctx[ens.context_columns].to_numpy(float))
    same_nan = np.isnan(ctx_map.to_numpy(float)) == np.isnan(Xctx[ens.context_columns].to_numpy(float))
    all_ok &= ok(bool(np.nanmax(np.abs(diff)) < 1e-9 and same_nan.all()),
                 f"context rows identical (max abs diff {np.nanmax(np.abs(diff)):.2e}, "
                 f"NaN pattern equal {same_nan.all()})")

    # 2b. images
    perm = mi.serpentine_perm(mi.NGS)
    images_map = mi.images_from_curves(curves_train.astype(np.float32), perm)
    finite = np.isfinite(images_train).all(axis=(1, 2, 3))
    d_img = np.nanmax(np.abs(images_map[finite] - images_train[finite]))
    all_ok &= ok(bool(d_img < 1e-6), f"serpentine images identical (max abs diff {d_img:.2e}, "
                                     f"{int(finite.sum())} plots with a curve)")

    # 3a. model outputs
    scaled_map = ens.predict_scaled(images_map, ctx_map)
    with torch.no_grad():
        m0 = ens.members[0]
        c0 = torch.from_numpy(m0.pre.transform(Xctx[ens.context_columns].iloc[np.flatnonzero(finite)]))
        x0 = torch.from_numpy(images_train[finite])
        direct = m0.model(x0, c0).numpy()
    d_out = np.nanmax(np.abs(scaled_map[0][finite] - direct))
    all_ok &= ok(bool(d_out < 1e-4), f"seed-0 forward pass identical (max abs diff {d_out:.2e})")

    # 3b. in-sample agreement with observed facets
    pred = ens.predict(images_map, ctx_map, y_train=Y)
    rows = []
    for j, t in enumerate(names):
        okm = np.isfinite(pred[:, j]) & np.isfinite(Y[:, j])
        r2 = 1 - np.sum((pred[okm, j] - Y[okm, j]) ** 2) / np.sum((Y[okm, j] - Y[okm, j].mean()) ** 2)
        rows.append(dict(target=t, n=int(okm.sum()), r2_insample=round(float(r2), 3),
                         pred_min=float(np.nanmin(pred[:, j])), pred_max=float(np.nanmax(pred[:, j])),
                         obs_min=float(np.nanmin(Y[:, j])), obs_max=float(np.nanmax(Y[:, j]))))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    q0 = df[df.target.isin(["pd_inext_q0", "td_inext_q0"])]["r2_insample"]
    all_ok &= ok(bool((q0 > 0.5).all()),
                 "in-sample R2 on PD0/TD0 above 0.5 (block-CV was 0.61/0.79; in-sample must not be lower)")
    print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED -- do not produce maps")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
