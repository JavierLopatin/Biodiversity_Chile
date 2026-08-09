#!/usr/bin/env python3
"""GDM and SGDM against the axis-regression design, scored on the same held-out pairs.

The comparison this script exists to make is not "which R-squared is bigger" — the two
families do not even predict the same quantity. It is: **given the same predictors, the same
folds and the same held-out plots, which approach reproduces the observed compositional
dissimilarity better?** So everything is scored in dissimilarity space, on the pairs among
held-out plots only:

  ``gdm``     fit GDM on the training pairs, predict the test pairs directly
  ``sgdm``    the same after a fold-local sparse CCA reduction of the predictors
  ``rf_axes`` fit the project's Random Forest on the first k PCoA axes, predict them for the
              test plots, and reconstruct pairwise Euclidean distance from the predictions
  ``oracle``  the *true* PCoA axes of the held-out plots, reconstructed the same way. Not a
              model: the ceiling that any k-axis regression is working against.
  ``geo``     GDM on geographic distance alone. Composition turns over with distance whether
              or not anything is measured, and a satellite predictor has to beat that.

Everything is fitted inside the fold — the standardisation, the spline knots, the sparse CCA
loadings (which read the species matrix and would leak outright otherwise) and the forest.

Usage:
    python scripts/21_run_gdm.py --scheme kfold5_owner --spec gm+obscomp_all+seas_all+topo
    python scripts/21_run_gdm.py --scheme kfold5_owner --spec curve+topo --index kndvi
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import cv as cvmod                                           # noqa: E402
from biodiv import features as feat                                      # noqa: E402
from biodiv import gdm as gmod                                           # noqa: E402
from biodiv import io_parcelas as iop                                    # noqa: E402
from biodiv.models_tabular import fit_predict_rf                         # noqa: E402

ID = "PlotObservationID"

#: Training pairs drawn per fold. All 375k pairs of an 866-plot training fold fit in memory,
#: but the deviance surface is flat long before that and the subsample keeps a run to
#: minutes. Test pairs are never subsampled — every held-out pair is scored.
MAX_TRAIN_PAIRS = 250_000


def species_matrix(zip_path: str, ids: list[str]) -> np.ndarray:
    """Hellinger-transformed presence/absence matrix, the response side of the sparse CCA.

    Built once over all plots, but only the training rows are ever passed to
    :func:`biodiv.gdm.sparse_cca`. What crosses the fold boundary is the *column vocabulary* —
    which species exist in the region — and not any held-out plot's occurrences. That is the
    same transductive assumption the fixed PCoA axes already make, and it is stated here so
    it is a choice rather than an oversight.
    """
    long = iop.load_long(zip_path)
    long[ID] = long[ID].astype(str)
    long = long[long[ID].isin(set(ids))]
    agg = long.groupby([ID, "Accepted_species"], as_index=False)["Value"].sum()
    w = (agg.pivot(index=ID, columns="Accepted_species", values="Value")
         .reindex(ids).fillna(0.0).to_numpy())
    b = (w > 0).astype(float)
    tot = b.sum(axis=1, keepdims=True)
    return np.sqrt(np.divide(b, tot, out=np.zeros_like(b), where=tot > 0))


#: L1 budgets tried by ``--sgdm-grid``, as fractions of sqrt(ncol) — Witten et al.'s
#: parameterisation. The published SGDM grid is of this shape and size.
PENALTY_GRID = [0.1, 0.2, 0.4, 0.7, 1.0]


def _choose_penalties(Xall: np.ndarray, Z: np.ndarray, itr: np.ndarray,
                      ti: np.ndarray, tj: np.ndarray, y_tr: np.ndarray, args
                      ) -> tuple[float, float]:
    """Grid-search the sparse CCA penalties inside the training fold.

    The inner split is a random half of the *training* plots. It only ever sees training
    rows, so the held-out fold stays untouched; the alternative — tuning against the outer
    test pairs — is the single easiest way to make a penalised method look good.
    """
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(itr))
    fit, val = itr[perm[: len(itr) // 2]], itr[perm[len(itr) // 2:]]
    fit_set = set(fit.tolist())
    m = np.array([a in fit_set and b in fit_set for a, b in zip(ti, tj)])
    if m.sum() < 5000:
        return args.cx, args.cz

    vi, vj = all_pairs(np.sort(val), rng, 40_000)
    y_val = np.load(Path(args.derived) / "dissimilarity_pa.npy")[vi, vj].astype(float)

    best, best_rho = (args.cx, args.cz), -np.inf
    for cx in PENALTY_GRID:
        for cz in PENALTY_GRID:
            try:
                W = gmod.sparse_cca(Xall[fit], Z[fit] - Z[fit].mean(axis=0),
                                    n_components=args.n_components, cx=cx, cz=cz)
                P = Xall @ W
                S, k = gmod.spline_transform(P[fit])
                Pa = np.zeros((Xall.shape[0], S.shape[1]))
                Pa[fit] = S
                Pa[val] = gmod.spline_transform(P[val], k)[0]
                beta = gmod.fit_gdm(gmod.pair_design(Pa, ti[m], tj[m]), y_tr[m])
                rho = float(spearmanr(
                    y_val, gmod.predict_gdm(gmod.pair_design(Pa, vi, vj), beta)).statistic)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(rho) and rho > best_rho:
                best, best_rho = (cx, cz), rho
    return best


def all_pairs(idx: np.ndarray, rng: np.random.Generator | None = None,
              cap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.array(list(combinations(range(len(idx)), 2))).T
    if cap is not None and len(i) > cap:
        sel = rng.choice(len(i), cap, replace=False)
        i, j = i[sel], j[sel]
    return idx[i], idx[j]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="kfold5_owner")
    p.add_argument("--spec", default="gm+obscomp_all+seas_all+contrast+topo")
    p.add_argument("--index", default=None)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--zip", default="data/20602096.zip")
    p.add_argument("--k-axes", type=int, default=8, dest="k_axes")
    p.add_argument("--n-components", type=int, default=10, dest="n_components")
    p.add_argument("--cx", type=float, default=0.3, help="L1 budget on the predictor side")
    p.add_argument("--cz", type=float, default=0.3, help="L1 budget on the species side")
    p.add_argument("--sgdm-grid", action="store_true", dest="sgdm_grid",
                   help="choose cx/cz by grid search inside each training fold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/tables/gdm_comparison.csv")
    args = p.parse_args()

    d = Path(args.derived)
    ids = feat.plot_ids(args.derived)
    X_full, _ = feat.build_design(args.spec, index=args.index, derived=args.derived, ids=ids)
    D_obs = np.load(d / "dissimilarity_pa.npy").astype(np.float64)
    axes = pd.read_parquet(d / "composition_axes.parquet")
    axes[ID] = axes[ID].astype(ids.dtype)
    axcols = [f"pcoa{j+1}_pa" for j in range(args.k_axes)]
    A_true = axes.set_index(ID).reindex(ids)[axcols].to_numpy(float)
    Z = species_matrix(args.zip, [str(i) for i in ids])

    plots = pd.read_parquet(d / "plots_subset.parquet").set_index(ID).reindex(ids)
    geo = plots[["X", "Y"]].to_numpy(float) / 1000.0          # km

    cv = cvmod.load_schemes(d / "cv_folds_modelling.parquet")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rng = np.random.default_rng(args.seed)

    print(f"esquema={args.scheme}  spec={args.spec}  p={X_full.shape[1]}  "
          f"k_axes={args.k_axes}  sgdm_comp={args.n_components}\n")

    rows = []
    for fold, held, tr_ids, te_ids in cvmod.iter_folds(cv, args.scheme):
        t0 = time.time()
        itr = np.sort(pos[tr_ids].to_numpy())
        ite = np.sort(pos[te_ids].to_numpy())

        pre = feat.Preprocessor(standardise=True).fit(X_full, tr_ids)
        Xtr_s, Xte_s = pre.transform(X_full.loc[tr_ids]), pre.transform(X_full.loc[te_ids])
        Xall = np.zeros((len(ids), Xtr_s.shape[1]), dtype=np.float64)
        Xall[itr], Xall[ite] = Xtr_s, Xte_s

        ti, tj = all_pairs(itr, rng, MAX_TRAIN_PAIRS)
        ei, ej = all_pairs(ite)
        y_tr, y_te = D_obs[ti, tj], D_obs[ei, ej]

        res: dict[str, float] = {}

        def score(name: str, pred_te: np.ndarray, as_dissim: bool) -> None:
            rho = float(spearmanr(y_te, pred_te).statistic)
            res[f"{name}_rho"] = rho
            if as_dissim:
                res[f"{name}_dev"] = gmod.deviance_explained(y_te, pred_te)

        # --- GDM on the full predictor set -------------------------------------------
        S, knots = gmod.spline_transform(Xall[itr])
        S_all = np.zeros((len(ids), S.shape[1]))
        S_all[itr] = S
        S_all[ite] = gmod.spline_transform(Xall[ite], knots)[0]
        beta = gmod.fit_gdm(gmod.pair_design(S_all, ti, tj), y_tr)
        score("gdm", gmod.predict_gdm(gmod.pair_design(S_all, ei, ej), beta), True)

        # --- SGDM: fold-local sparse CCA, then GDM on the components ------------------
        # The L1 budgets are chosen by grid search on an inner split of the *training* fold,
        # which is what Leitao et al. (2015) do and what the first run of this script did
        # not: with the default penalties SGDM scored below plain GDM, and an untuned
        # penalised method losing to its unpenalised parent says nothing about the method.
        cx, cz = _choose_penalties(Xall, Z, itr, ti, tj, y_tr, args) if args.sgdm_grid \
            else (args.cx, args.cz)
        W = gmod.sparse_cca(Xall[itr], Z[itr] - Z[itr].mean(axis=0),
                            n_components=args.n_components, cx=cx, cz=cz)
        res["sgdm_cx"], res["sgdm_cz"] = cx, cz
        P = Xall @ W
        Ps, pk = gmod.spline_transform(P[itr])
        P_all = np.zeros((len(ids), Ps.shape[1]))
        P_all[itr] = Ps
        P_all[ite] = gmod.spline_transform(P[ite], pk)[0]
        beta_s = gmod.fit_gdm(gmod.pair_design(P_all, ti, tj), y_tr)
        score("sgdm", gmod.predict_gdm(gmod.pair_design(P_all, ei, ej), beta_s), True)

        # --- geographic distance alone -----------------------------------------------
        # Composition turns over with distance whether or not anything is measured, so this
        # is the number the satellite predictors have to beat, not the null deviance.
        Gs, gk = gmod.spline_transform(geo[itr])
        G_all = np.zeros((len(ids), Gs.shape[1]))
        G_all[itr] = Gs
        G_all[ite] = gmod.spline_transform(geo[ite], gk)[0]
        beta_g = gmod.fit_gdm(gmod.pair_design(G_all, ti, tj), y_tr)
        score("geo", gmod.predict_gdm(gmod.pair_design(G_all, ei, ej), beta_g), True)

        # --- predictors AND geography: does the satellite add over pure distance? -----
        GX_all = np.hstack([S_all, G_all])
        beta_gx = gmod.fit_gdm(gmod.pair_design(GX_all, ti, tj), y_tr)
        score("gdm_geo", gmod.predict_gdm(gmod.pair_design(GX_all, ei, ej), beta_gx), True)

        # --- the project's design: RF on k PCoA axes, read back as distance -----------
        # `ite` is sorted above, so a pair's global row index maps back to its position in
        # the prediction block by searchsorted.
        ok = np.isfinite(A_true[itr]).all(axis=1)
        pred_ax, _, _ = fit_predict_rf(Xtr_s[ok], A_true[itr][ok], Xte_s, seed=args.seed)
        pi, pj = np.searchsorted(ite, ei), np.searchsorted(ite, ej)
        score("rf_axes", np.linalg.norm(pred_ax[pi] - pred_ax[pj], axis=1), False)

        # --- the ceiling: the true axes of the held-out plots -------------------------
        score("oracle", np.linalg.norm(A_true[ei] - A_true[ej], axis=1), False)
        score("oracle2", np.linalg.norm(A_true[ei][:, :2] - A_true[ej][:, :2], axis=1), False)

        rows.append(dict(scheme=args.scheme, spec=args.spec, fold=fold, held_out=held,
                         n_test=len(ite), n_pairs=len(ei),
                         minutes=round((time.time() - t0) / 60, 2), **res))
        print(f"  fold {fold} ({held[:22]:22s}) n={len(ite):4d} pares={len(ei):6d} "
              f"[{rows[-1]['minutes']:4.1f}m]  "
              f"gdm={res['gdm_rho']:+.3f} +geo={res['gdm_geo_rho']:+.3f} "
              f"sgdm={res['sgdm_rho']:+.3f} geo={res['geo_rho']:+.3f} "
              f"rf={res['rf_axes_rho']:+.3f} oracle={res['oracle_rho']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = not out.exists()
    df.to_csv(out, mode="a", header=header, index=False)

    print(f"\n-> {out}")
    print("\nmedia sobre folds (Spearman contra la disimilitud de Jaccard observada,")
    print("sobre los pares entre parcelas excluidas):")
    for name, label in [("geo", "GDM solo distancia geografica"),
                        ("gdm", "GDM con los predictores"),
                        ("gdm_geo", "GDM con predictores + geografia"),
                        ("sgdm", "SGDM (sCCA fold-local + GDM)"),
                        ("rf_axes", f"RF sobre {args.k_axes} ejes PCoA -> distancia"),
                        ("oracle2", "TECHO: los 2 ejes verdaderos"),
                        ("oracle", f"TECHO: los {args.k_axes} ejes verdaderos")]:
        c = f"{name}_rho"
        if c in df:
            dv = f"{name}_dev"
            extra = f"   dev_expl={df[dv].mean():+.3f}" if dv in df else ""
            print(f"  {label:42s} rho = {df[c].mean():+.3f} "
                  f"(sd {df[c].std():.3f}){extra}")


if __name__ == "__main__":
    main()
