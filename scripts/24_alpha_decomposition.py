#!/usr/bin/env python3
"""Is the alpha R2 under a given CV scheme ecological signal or contributor lookup?

Motivation. Under ``kfold5_window`` the same Random Forest that scores R2_alpha = -0.60
under ``kfold5_owner`` scores +0.57, and climate alone (18 columns on a 0.05 degree grid)
reaches hill_q0 = +0.76. The obvious explanation is that richness carries eta2 = 0.72
between contributors and that ``kfold5_window`` puts every contributor on both sides of the
split, so any smooth spatial field works as a lookup key for the contributor's level.

But that explanation was argued from a *different* scheme's numbers, which is not evidence
about this one: under ``kfold5_window`` the model does see other plots of the held-out
plot's contributor, so it could legitimately learn a within-contributor relationship —
"in this contributor's plots, greener means richer" — and that would be real ecology, not
lookup. Nothing measured so far separates the two.

This script separates them, entirely inside whichever scheme is passed. It never refits
anything under a second scheme, so the answer cannot be an artefact of the comparison.

**Test 1 — decompose the out-of-fold predictions.** Every plot's observed and predicted
value is split into its contributor's mean and its deviation from that mean:

    between:  the owner means, obs vs pred      -- "does it know who surveyed this?"
    within:   deviations from the owner mean    -- "does it rank plots inside a survey?"

Only the *within* term can be ecological. A model that nails the level of every contributor
and ranks plots randomly inside each one scores a high total R2 and a within R2 of zero.
Both terms are computed from the same OOF table that produced the headline number, so this
is a decomposition of that number, not a different experiment.

**Test 2 — the coordinate control.** Refit with longitude, latitude and elevation as the
only predictors, same scheme, same folds, same seeds. Three columns cannot carry
phenology, aridity or spectral information; whatever R2 they reach is the ceiling of what
pure position buys under this scheme. If the full predictor set does not beat it, the
satellite data is not what is doing the work.

Usage:
    python scripts/24_alpha_decomposition.py --scheme kfold5_window
    python scripts/24_alpha_decomposition.py --scheme kfold5_window --spec curve+clim+topo+area
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import cv as cvmod            # noqa: E402
from biodiv import features as feat       # noqa: E402
from biodiv import metrics as mx          # noqa: E402
from biodiv import targets as tg          # noqa: E402
from biodiv.models_tabular import fit_predict_rf   # noqa: E402

ID = "PlotObservationID"
ALPHA = ["hill_q0", "hill_q1", "hill_q2"]
BETA = ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"]


def oof_table(spec: str, index: str | None, scheme: str, seeds: list[int],
              derived: str) -> tuple[pd.DataFrame, list[str], int]:
    """Out-of-fold observed/predicted for every target, exactly as script 19 builds it."""
    ids = feat.plot_ids(derived)
    X_full, _ = feat.build_design(spec, index=index, derived=derived, ids=ids)
    _, Y_full, names = tg.load_targets(derived, "all", plot_ids=ids)
    cv = cvmod.load_schemes(Path(derived) / "cv_folds_modelling.parquet")
    pos = pd.Series(np.arange(len(ids)), index=ids)

    per_fold = []
    for seed in seeds:
        for fold, held, tr_ids, te_ids in cvmod.iter_folds(cv, scheme):
            cvmod.assert_no_leak(tr_ids, te_ids, f"({scheme} fold {fold})")
            itr, ite = pos[tr_ids].to_numpy(), pos[te_ids].to_numpy()

            pre = feat.Preprocessor(standardise=False).fit(X_full, tr_ids)
            Xtr = pre.transform(X_full.loc[tr_ids])
            Xte = pre.transform(X_full.loc[te_ids])

            scaler = tg.fit_target_scaler(Y_full[itr])
            Ytr = tg.apply_target_scaler(Y_full[itr], scaler)
            pred_s, _, resid_s = fit_predict_rf(Xtr, Ytr, Xte, seed=seed, importance=False)
            pred = tg.inverse_with_smearing(pred_s, scaler, resid_s,
                                            y_train=Y_full[itr], seed=seed)

            blk = pd.DataFrame({ID: te_ids, "fold": fold, "held_out": held, "seed": seed})
            for j, t in enumerate(names):
                blk[f"{t}_obs"], blk[f"{t}_pred"] = Y_full[ite, j], pred[:, j]
            per_fold.append(blk)

    return cvmod.collect_oof(per_fold, scheme, names), names, X_full.shape[1]


def decompose(oof: pd.DataFrame, owner: pd.Series, targets: list[str]) -> pd.DataFrame:
    """Split R2 into what the owner means explain and what survives inside an owner.

    ``within`` centres observed and predicted on *their own* owner means, so a model that
    reproduces every contributor's level but ranks plots randomly inside a contributor gets
    within = 0. R2 is computed the usual way against the variance of the centred target, and
    averaged over seeds so it is on the same footing as the headline number.
    """
    rows = []
    o = oof.copy()
    o["owner"] = o[ID].map(owner)
    for t in targets:
        obs, pred = f"{t}_obs", f"{t}_pred"
        per_seed = {"total": [], "within": [], "between": []}
        for _, s in o.groupby("seed"):
            g = s.groupby("owner")
            obs_c = s[obs] - g[obs].transform("mean")
            pred_c = s[pred] - g[pred].transform("mean")

            per_seed["total"].append(1 - ((s[obs] - s[pred]) ** 2).sum()
                                     / ((s[obs] - s[obs].mean()) ** 2).sum())
            per_seed["within"].append(1 - ((obs_c - pred_c) ** 2).sum()
                                      / (obs_c ** 2).sum())
            m = s.groupby("owner")[[obs, pred]].mean()
            per_seed["between"].append(1 - ((m[obs] - m[pred]) ** 2).sum()
                                       / ((m[obs] - m[obs].mean()) ** 2).sum())
        rows.append({"target": t,
                     "R2_total": np.mean(per_seed["total"]),
                     "R2_within_owner": np.mean(per_seed["within"]),
                     "R2_between_owner": np.mean(per_seed["between"]),
                     "var_within_frac": float(
                         (o.groupby("owner")[obs].transform(lambda x: x - x.mean()) ** 2).sum()
                         / ((o[obs] - o[obs].mean()) ** 2).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="kfold5_window")
    p.add_argument("--spec", default="curve+clim+topo+area")
    p.add_argument("--index", default="kndvi")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/tables/alpha_decomposition.csv")
    args = p.parse_args()

    seeds = list(range(args.seeds))
    plots = pd.read_parquet(Path(args.derived) / "plots_subset.parquet")
    owner = plots.set_index(ID)["Owner"]

    out = []
    for label, spec, index in [("predictores", args.spec, args.index),
                               ("solo coordenadas", "coords", None)]:
        oof, names, p_n = oof_table(spec, index, args.scheme, seeds, args.derived)
        d = decompose(oof, owner, ALPHA + BETA)
        d.insert(0, "block", label)
        d.insert(1, "spec", spec)
        d.insert(2, "n_features", p_n)
        d.insert(3, "scheme", args.scheme)
        out.append(d)

        print(f"\n===== {label}  ({spec}, p={p_n})   esquema={args.scheme}")
        print("        R2_total = el numero que se reporta")
        print("       R2_within = lo que sobrevive DENTRO de un contribuyente (lo ecologico)")
        print("      R2_between = acertar el nivel de cada contribuyente\n")
        show = d[["target", "R2_total", "R2_within_owner", "R2_between_owner",
                  "var_within_frac"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        for grp, cols in [("alfa", ALPHA), ("beta", BETA)]:
            s = d[d.target.isin(cols)]
            print(f"  {grp:5s}  total={s.R2_total.mean():+.3f}  "
                  f"within={s.R2_within_owner.mean():+.3f}  "
                  f"between={s.R2_between_owner.mean():+.3f}")

    res = pd.concat(out, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
