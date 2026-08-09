#!/usr/bin/env python3
"""Screen predictor blocks with a Random Forest before any deep model touches them.

RF is the judge because it costs minutes, has no hyperparameters that would confound the
comparison, and — measured across the 79 published runs — already matches or beats every
network in this project. If a block cannot beat the curve under RF, nothing downstream will
rescue it.

**The screen is scored on beta, not on the mean of all nine targets.** Two things established
before this matrix was built make that the only defensible choice:

  1. Alpha diversity is not predictable across contributors *at all*. Richness has
     eta^2 = 0.72 between owners — Dobbs & Miranda average 1.6 species per plot, Becerra 28.2,
     at comparable plot sizes — so a held-out contributor sits at a level no predictor can
     know. Removing that level from the training target moves R2_alpha from -0.56 to -0.07,
     i.e. 87% of the deficit was level error; but -0.07 is also what predicting the global
     mean gets, so what remains is no within-contributor signal. Alpha is reported, never
     optimised.

  2. Beta is the opposite: R2_beta is unchanged by removing the owner level (+0.250 vs
     +0.206), so its signal is real, and its analytic ceiling under owner-grouped CV is
     +0.667 against +0.250 achieved. That is where the entire remaining margin lives.

The central row is **X04 against X03**: an annual composite carries level and spread but no
date and no shape, and permuting the 52 steps leaves it unchanged. If adding the curve on top
of the composite does not beat the curve alone, the project's claim needs restating — not as
"phenology predicts composition" but as something more precise about which part of the
temporal signal is doing the work. That has to be reportable either way.

Usage:
    python scripts/19_screen_blocks.py                       # the full matrix
    python scripts/19_screen_blocks.py --rows X03 X04        # only these
    python scripts/19_screen_blocks.py --schemes kfold5_owner kfold5_window
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import cv as cvmod                                           # noqa: E402
from biodiv import features as feat                                      # noqa: E402
from biodiv import metrics as mx                                         # noqa: E402
from biodiv import targets as tg                                         # noqa: E402
from biodiv.models_tabular import fit_predict_rf                         # noqa: E402

ID = "PlotObservationID"

ALPHA = ["hill_q0", "hill_q1", "hill_q2"]
BETA = ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"]
BETA_COVER = ["lcbd_cover", "pcoa1_cover", "pcoa2_cover"]

#: The screening matrix. ``index`` is the vegetation index for the blocks that need one;
#: kNDVI throughout, because it is the best-performing index for beta in the published table
#: and holding it fixed keeps the block contrast clean.
MATRIX: dict[str, dict] = {
    "X00": dict(spec="composite_all", index=None,
                q="compuesto anual solo: cuanto da sin nada de forma ni fecha"),
    "X01": dict(spec="composite_all+topo+area", index=None,
                q="el control no fenologico completo"),
    "X02": dict(spec="lsp+topo+area", index="kndvi",
                q="LSP actual, la linea base del campo (= RF01)"),
    "X03": dict(spec="curve+topo+area", index="kndvi",
                q="LA REFERENCIA: la curva de 52 pasos (= RF03)"),
    "X04": dict(spec="composite_all+curve+topo+area", index="kndvi",
                q="EL CONTRASTE CENTRAL: la forma aporta SOBRE el compuesto?"),
    "X05": dict(spec="svh+topo+area", index=None,
                q="heterogeneidad espectral entre pixeles, sola"),
    "X06": dict(spec="composite_all+svh+topo+area", index=None,
                q="la heterogeneidad aporta sobre el nivel?"),
    "X07": dict(spec="gm+topo+area", index=None,
                q="geomediana + las tres MAD: composite sin forma temporal"),
    "X08": dict(spec="gm+curve+topo+area", index="kndvi",
                q="fenologia sobre geomediana"),
    "X09": dict(spec="obscomp_all+topo+area", index=None,
                q="composites sobre observaciones reales, sin el suavizado"),
    "X10": dict(spec="seas_all+topo+area", index=None,
                q="fenologia reducida a 4 numeros: mediana por estacion austral"),
    "X11": dict(spec="contrast+topo+area", index=None,
                q="contrastes entre indices (ndvi-savi mide exposicion de suelo)"),
    "X12": dict(spec="gm+obscomp_all+seas_all+contrast+topo+area", index=None,
                q="todo lo que NO tiene forma, junto"),
    "X13": dict(spec="curve_all+topo+area", index=None,
                q="las 5 curvas apiladas (= RF06, el mejor publicado)"),
    "X14": dict(spec="lsp+curve+composite_all+gm+obscomp_all+seas_all+contrast+svh+topo+area",
                index="kndvi", q="todo junto: el techo empirico"),
    # Climate. Run only after scripts/22_extract_climate.py. The GDM comparison is what
    # motivates these: geographic distance alone reaches rho +0.435 against observed
    # dissimilarity and every spectral block together adds +0.057, which says the missing
    # axis is the one distance is standing in for.
    "X15": dict(spec="clim+area", index=None,
                q="clima solo: normales bioclimaticas 1971-2000"),
    "X16": dict(spec="clim+topo+area", index=None,
                q="clima + topografia, sin nada de teledeteccion"),
    "X17": dict(spec="curve+clim+topo+area", index="kndvi",
                q="EL CONTRASTE: la fenologia aporta SOBRE el clima?"),
    "X18": dict(spec="curve_all+clim+topo+area", index=None,
                q="las 5 curvas + clima"),
    "X19": dict(spec="lsp+curve+gm+seas_all+contrast+clim+topo+area", index="kndvi",
                q="el mejor candidato completo"),
}

DEFAULT_SCHEMES = ["kfold5_owner", "kfold5_window"]


def run_block(spec: str, index: str | None, scheme: str, seeds: list[int],
              agg: str, derived: str) -> dict:
    ids = feat.plot_ids(derived)
    X_full, _ = feat.build_design(spec, index=index, derived=derived, ids=ids, agg=agg)
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

    oof = cvmod.collect_oof(per_fold, scheme, names)
    r2 = mx.pooled_metrics(oof, names, by=["seed"]).groupby("target")["R2"].mean()
    sd = mx.pooled_metrics(oof, names, by=["seed"]).groupby("target")["R2"].std()

    out = {t: float(r2[t]) for t in names}
    out["R2_beta"] = float(r2[BETA].mean())
    out["R2_beta_sd"] = float(sd[BETA].mean())
    out["R2_alpha"] = float(r2[ALPHA].mean())
    out["R2_beta_cover"] = float(r2[BETA_COVER].mean())
    out["n_features"] = int(X_full.shape[1])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", nargs="+", default=sorted(MATRIX))
    p.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--agg", default=feat.DEFAULT_AGG,
                   choices=["median", "mean", "trimmed", "center"])
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/tables/block_screen.csv")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seeds = list(range(args.seeds))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_path) if out_path.exists() and not args.force else pd.DataFrame()
    seen = set(zip(done["row"], done["scheme"], done["agg"])) if len(done) else set()

    rows = list(done.to_dict("records"))
    for name in args.rows:
        meta = MATRIX[name]
        for scheme in args.schemes:
            if (name, scheme, args.agg) in seen:
                print(f"  [skip] {name} / {scheme} / {args.agg}")
                continue
            t0 = time.time()
            res = run_block(meta["spec"], meta["index"], scheme, seeds, args.agg,
                            args.derived)
            rec = dict(row=name, scheme=scheme, agg=args.agg, spec=meta["spec"],
                       index=meta["index"] or "", question=meta["q"],
                       minutes=round((time.time() - t0) / 60, 2), **res)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  {name} {scheme:20s} p={rec['n_features']:4d} "
                  f"[{rec['minutes']:5.1f}m]  R2_beta={res['R2_beta']:+.3f} "
                  f"R2_alpha={res['R2_alpha']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n-> {out_path}")
    for scheme in args.schemes:
        s = df[(df["scheme"] == scheme) & (df["agg"] == args.agg)]
        if s.empty:
            continue
        s = s.sort_values("R2_beta", ascending=False)
        ref = s[s["row"] == "X03"]["R2_beta"]
        print(f"\n===== {scheme} — ordenado por R2_beta"
              + (f"  (X03, la referencia = {float(ref.iloc[0]):+.3f})" if len(ref) else ""))
        cols = ["row", "n_features", "R2_beta", "R2_beta_sd", "R2_alpha", "lcbd_pa",
                "pcoa1_pa", "pcoa2_pa", "question"]
        print(s[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
