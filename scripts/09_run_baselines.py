#!/usr/bin/env python3
"""Tier 0 controls and Tier 1 Random Forest — the first numbers this project produces.

Nothing in the deep-learning tiers is interpretable without these. In order of importance:

  B02  `area` alone. Plot size ranges 78.5-10,000 m2 and its Spearman correlation with
       `hill_q0` is +0.33 (risk R3). Whatever R2 a model reaches for richness has to be read
       against this number, so it is computed first and quoted beside every richness result.

  B03  `coords` (lon, lat, elevation) alone. If coordinates predict as well as phenology
       under `kfold5_random` and collapse under `kfold5_block20`, that single contrast is
       the whole spatial-autocorrelation story (risk R8).

  B01  `topo+area`. The no-phenology reference every fused model must beat.

  B00  training mean. Must return pooled R2 ~ 0. Anything else means the partition or the
       target scaling is broken, and no other result is trustworthy.

  RF01 vs RF03  the field's baseline (18 LSP metrics) against the full 52-step curve. This
       is gap G1 of docs/01_state_of_the_art.md, answered before any neural network exists.

Usage:
    python scripts/09_run_baselines.py --model RF01 --index ndvi --scheme kfold5_window
    python scripts/09_run_baselines.py --all --scheme kfold5_window --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import cv as cvmod                                          # noqa: E402
from biodiv import features as feat                                     # noqa: E402
from biodiv import metrics as mx                                        # noqa: E402
from biodiv import runlog, targets as tg                                # noqa: E402
from biodiv.models_tabular import fit_predict_rf, fit_predict_rf_multioutput  # noqa: E402

ID_COL = "PlotObservationID"

MODELS: dict[str, dict] = {
    "B00": dict(family="BASE", spec="area", per_index=False, mean_only=True,
                note="training-mean predictor; pooled R2 must be ~0"),
    "B01": dict(family="BASE", spec="topo+area", per_index=False,
                note="no-phenology reference"),
    "B02": dict(family="BASE", spec="area", per_index=False,
                note="sampling-effort baseline for risk R3"),
    "B03": dict(family="BASE", spec="coords", per_index=False,
                note="spatial-autocorrelation control for risk R8"),
    "RF01": dict(family="RF", spec="lsp+topo+area", per_index=True,
                 note="the field's baseline: 18 LSP metrics"),
    "RF02": dict(family="RF", spec="lsp+qc+topo+area", per_index=True,
                 note="does LSP reliability information help?"),
    "RF03": dict(family="RF", spec="curve+topo+area", per_index=True,
                 note="the full 52-week curve (gap G1)"),
    "RF04": dict(family="RF", spec="lsp+curve+topo+area", per_index=True,
                 note="curve and its scalar summaries together"),
    "RF05": dict(family="RF", spec="lsp_all+topo+area", per_index=False,
                 note="LSP of all five indices stacked"),
    "RF06": dict(family="RF", spec="curve_all+topo+area", per_index=False,
                 note="curves of all five indices stacked"),
    "RF08": dict(family="RF", spec="lsp+curve+topo", per_index=True,
                 note="--no-area ablation of RF04"),
    # Single-block pixel-level ablations. Everything except the named block stays at the 5x5
    # patch summary, so the contrast against RF01 / RF03 isolates that block's footprint. This
    # is a different question from `--px center`, which moves the whole design at once: here
    # the comparison is controlled, there it is about the design as a whole.
    "RF01p": dict(family="RF", spec="lsp_ctr+topo+area", per_index=True,
                  note="RF01 with the LSP block at the centre pixel only"),
    "RF03p": dict(family="RF", spec="curve+topo+area", per_index=True, curve_px="center",
                  note="RF03 with the curve at the centre pixel only"),
}


#: Switching the whole design between pixel levels, not one block at a time. Comparing an
#: LSP block at the centre pixel against a topography block at the 5x5 mean would confound
#: the two choices; the ablation is only interpretable if every predictor describes the same
#: footprint.
PX_BLOCKS = {"lsp": "lsp_ctr", "topo": "topo_ctr"}


def _at_pixel(spec: str, px: str) -> str:
    if px != "center":
        return spec
    return "+".join(PX_BLOCKS.get(b, b) for b in spec.split("+"))


def run_one(model: str, scheme: str, index: str | None, seeds: list[int],
            derived: str, target_set: str, root: Path,
            multioutput: bool = False, force: bool = False,
            px: str = "mean5x5") -> pd.DataFrame | None:
    meta = MODELS[model]
    spec = _at_pixel(meta["spec"], px)
    tag = spec.replace("+", "-")
    idx = index if meta["per_index"] else ""
    cfg = runlog.RunConfig(
        run_id=runlog.make_run_id(model + ("c" if px == "center" else ""), tag, idx),
        family=meta["family"], scheme=scheme, features=spec, index=idx,
        target_set=target_set, seeds=tuple(seeds),
        model="RF-multioutput" if multioutput else ("mean" if meta.get("mean_only") else "RF"),
        fusion="none", notes=meta["note"], params=dict(px=px),
    )
    if runlog.already_done(cfg, root) and not force:
        print(f"  [skip] {cfg.run_id} / {scheme}")
        return None

    t0 = time.time()
    ids = feat.plot_ids(derived)
    X_full, _ = feat.build_design(spec, index=index, derived=derived, ids=ids,
                                  px=meta.get("curve_px", px))
    _, Y_full, names = tg.load_targets(derived, target_set, plot_ids=ids)
    if multioutput:
        keep = [j for j, n in enumerate(names) if n in tg.TARGETS_MAIN]
        Y_full, names = Y_full[:, keep], [names[j] for j in keep]

    cv = cvmod.load_schemes(Path(derived) / "cv_folds_modelling.parquet")
    pos = pd.Series(np.arange(len(ids)), index=ids)

    per_fold, imp_rows = [], []
    for seed in seeds:
        for fold, held, tr_ids, te_ids in cvmod.iter_folds(cv, scheme):
            cvmod.assert_no_leak(tr_ids, te_ids, f"({scheme} fold {fold})")
            itr, ite = pos[tr_ids].to_numpy(), pos[te_ids].to_numpy()

            pre = feat.Preprocessor(standardise=False).fit(X_full, tr_ids)
            Xtr, Xte = pre.transform(X_full.loc[tr_ids]), pre.transform(X_full.loc[te_ids])

            scaler = tg.fit_target_scaler(Y_full[itr])
            Ytr = tg.apply_target_scaler(Y_full[itr], scaler)

            if meta.get("mean_only"):
                mu = np.nanmean(Ytr, axis=0)
                pred_s = np.tile(mu, (len(ite), 1))
                resid_s = Ytr - mu           # a mean predictor has no skill; marginal is exact
                imps = None
            elif multioutput:
                pred_s = fit_predict_rf_multioutput(Xtr, Ytr, Xte, seed=seed)
                resid_s = Ytr - fit_predict_rf_multioutput(Xtr, Ytr, Xtr, seed=seed)
                imps = None
            else:
                pred_s, imps, resid_s = fit_predict_rf(Xtr, Ytr, Xte, seed=seed,
                                                       importance=True)

            pred = tg.inverse_with_smearing(pred_s, scaler, resid_s,
                                            y_train=Y_full[itr], seed=seed)

            block = pd.DataFrame({ID_COL: te_ids, "fold": fold, "held_out": held, "seed": seed})
            for j, t in enumerate(names):
                block[f"{t}_obs"] = Y_full[ite, j]
                block[f"{t}_pred"] = pred[:, j]
            per_fold.append(block)

            if imps is not None and seed == seeds[0]:
                for j, t in enumerate(names):
                    imp_rows.append(pd.DataFrame({
                        "fold": fold, "target": t,
                        "feature": pre.feature_names, "importance": imps[:, j]}))

    oof = cvmod.collect_oof(per_fold, scheme, names)
    per_seed = mx.pooled_metrics(oof, names, by=["seed"])
    ens_oof = mx.ensemble_oof(oof, names)
    ens = mx.compute_metrics(
        np.column_stack([ens_oof[f"{t}_pred"] for t in names]),
        np.column_stack([ens_oof[f"{t}_obs"] for t in names]), names)
    pooled = runlog.summarise_pooled(per_seed, ens)

    extra = {}
    if imp_rows:
        imp = pd.concat(imp_rows, ignore_index=True)
        extra["importance"] = (imp.groupby(["target", "feature"], as_index=False)["importance"]
                               .mean()
                               .sort_values(["target", "importance"], ascending=[True, False]))
    runlog.write_run(cfg, oof, per_seed, pooled, extra=extra, root=root)

    dt = time.time() - t0
    r2 = " ".join(f"{r.target}={r.R2_mean:+.3f}" for r in pooled.itertuples())
    print(f"  {cfg.run_id:40s} {scheme:16s} [{dt:5.1f}s] {r2}")
    return pooled


def _best_block(root: Path, scheme: str) -> tuple[str, str]:
    """Pick the (index, feature spec) with the best mean R2 so far, for the follow-up runs."""
    path = Path(root) / "summary.csv"
    if not path.exists():
        return ("ndvi", "lsp+topo+area")
    s = pd.read_csv(path)
    s = s[(s["scheme"] == scheme) & (s["family"] == "RF")]
    if s.empty:
        return ("ndvi", "lsp+topo+area")
    rank = s.groupby(["index", "features"], dropna=False)["R2"].mean().sort_values(ascending=False)
    ix, spec = rank.index[0]
    ix = "" if not isinstance(ix, str) or ix in ("", "nan") else ix
    return (ix, str(spec))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS) + ["RF07"], default=None)
    p.add_argument("--all", action="store_true", help="run every Tier 0 and Tier 1 model")
    p.add_argument("--index", default=None, help="vegetation index; omit to sweep all five")
    p.add_argument("--scheme", default="kfold5_window")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--target-set", default="all")
    p.add_argument("--out", default="results/models")
    p.add_argument("--px", default="mean5x5", choices=["mean5x5", "center"],
                   help="pixel level of the WHOLE design: 5x5 patch summary, or centre pixel")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seeds = list(range(args.seeds))
    root = Path(args.out)
    todo = sorted(MODELS) if args.all else ([args.model] if args.model else [])
    if not todo and args.model != "RF07":
        p.error("give --model or --all")

    print(f"scheme={args.scheme}  seeds={seeds}  targets={args.target_set}\n")
    for model in todo:
        if model == "RF07":
            continue
        meta = MODELS[model]
        idxs = ([args.index] if args.index else feat.INDICES) if meta["per_index"] else [None]
        for ix in idxs:
            run_one(model, args.scheme, ix, seeds, args.derived, args.target_set, root,
                    force=args.force, px=args.px)

    if args.all or args.model == "RF07":
        ix, spec = _best_block(root, args.scheme)
        print(f"\nRF07 joint forest on features={spec!r} index={ix or '-'}")
        MODELS["RF07"] = dict(family="RF", spec=spec, per_index=bool(ix),
                              note="single joint forest over the six complete targets")
        run_one("RF07", args.scheme, ix or None, seeds, args.derived, "main", root,
                multioutput=True, force=args.force, px=args.px)


if __name__ == "__main__":
    main()
