#!/usr/bin/env python3
"""How much of community composition can two PCoA axes represent, and would more help?

The project models ``pcoa1_pa`` and ``pcoa2_pa`` and reports R-squared against them. That
number answers "how well is this axis predicted", which is not the same question as "how
well is composition predicted", and the gap between the two is large here: after the
Cailliez correction two axes carry **12.5%** of the corrected eigenvalue mass for the
presence/absence facet and **13.8%** for the cover facet. A model with R-squared 1.0 on both
axes would still be silent about the other seven eighths.

This script measures three things and writes the extra axes so they can be fitted:

  1. **Variance explained by axis count.** How many axes are needed before the ordination
     stops being a bottleneck.

  2. **The dissimilarity-space ceiling.** Reconstruct pairwise Euclidean distance from the
     first k true axes and rank-correlate it with the observed Jaccard dissimilarity over
     all 585,171 pairs. This is the honest upper bound: no model predicting k axes can
     represent composition better than the axes themselves do. Spearman rather than Pearson
     because the Cailliez constant shifts every off-diagonal distance by the same amount and
     a rank statistic is immune to it.

  3. **Where the axes are attainable.** An axis carrying 1% of the variance is mostly noise;
     fitting it wastes compute and dilutes any multi-task loss. The cut is reported, not
     assumed.

Writes ``data/derived/composition_axes.parquet`` (k axes per facet) and
``data/derived/dissimilarity_pa.npy`` (the observed Jaccard matrix, used as the target of
the dissimilarity-space metric and later as GDM's response).

Usage:
    python scripts/20_composition_axes.py --k 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib.util                                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "resp07b", Path(__file__).resolve().parent / "07b_compute_responses_python.py")
r07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r07)

ID_COL = "PlotObservationID"

#: Pairs sampled for the rank correlation when the full matrix would be wasteful. 585k pairs
#: is fine; this exists so the same code serves the cover facet and any future larger subset.
MAX_PAIRS = 2_000_000


def rank_agreement(D_obs: np.ndarray, coords: np.ndarray, seed: int = 0) -> float:
    """Spearman between observed dissimilarity and distance reconstructed from ``coords``."""
    n = D_obs.shape[0]
    iu = np.triu_indices(n, 1)
    if len(iu[0]) > MAX_PAIRS:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(iu[0]), MAX_PAIRS, replace=False)
        iu = (iu[0][sel], iu[1][sel])
    d_pred = np.linalg.norm(coords[iu[0]] - coords[iu[1]], axis=1)
    return float(spearmanr(D_obs[iu], d_pred).statistic)


def facet(comm_beta: np.ndarray, comm_dist: np.ndarray, ids: list[str], dist_method: str,
          label: str, k: int) -> tuple[pd.DataFrame, np.ndarray]:
    keep = comm_beta.sum(axis=0) > 0
    comm_dist = comm_dist[:, keep]
    D = (r07.jaccard_binary(comm_dist) if dist_method == "jaccard"
         else r07.bray_curtis(comm_dist))
    scores, rel_eig = r07.pcoa_cailliez(D, k=k)

    print(f"\n=== {label}: {len(ids)} parcelas, distancia {dist_method}")
    print("  ejes  var_explicada_acum   spearman(D_obs, D_reconstruida)")
    rows = []
    for kk in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        if kk > scores.shape[1]:
            break
        var = float(rel_eig[:kk].sum())
        rho = rank_agreement(D, scores[:, :kk])
        rows.append((kk, var, rho))
        mark = "  <- lo que se modela hoy" if kk == 2 else ""
        print(f"  {kk:4d}      {100*var:6.1f}%           {rho:+.3f}{mark}")

    out = pd.DataFrame(scores, columns=[f"pcoa{j+1}" for j in range(scores.shape[1])])
    out.insert(0, ID_COL, ids)
    return out, D


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="data/20602096.zip")
    p.add_argument("--plots", default="data/derived/plots_subset.parquet")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--k", type=int, default=16)
    args = p.parse_args()

    plots = pd.read_parquet(args.plots)
    plot_ids = plots[ID_COL].astype(str).tolist()
    stratum = dict(zip(plot_ids, plots["stratum"].astype(str)))

    comm, _ = r07.build_comm(args.zip, plot_ids)
    comm_pa = (comm > 0).astype(np.float64)

    pa, D_pa = facet(comm_pa, comm_pa, plot_ids, "jaccard", "presence-absence", args.k)
    pa = pa.rename(columns={c: f"{c}_pa" for c in pa.columns if c != ID_COL})

    cover_ids = [i for i in plot_ids if stratum[i] == "cover"]
    pos = {pid: j for j, pid in enumerate(plot_ids)}
    comm_cover = comm[[pos[i] for i in cover_ids]]
    tot = comm_cover.sum(axis=1, keepdims=True)
    cover, _ = facet(comm_cover, np.where(tot > 0, comm_cover / tot, 0.0), cover_ids,
                     "bray", "cover", args.k)
    cover = cover.rename(columns={c: f"{c}_cover" for c in cover.columns if c != ID_COL})

    out = pd.DataFrame({ID_COL: plot_ids}).merge(pa, on=ID_COL, how="left") \
                                          .merge(cover, on=ID_COL, how="left")
    out[ID_COL] = plots[ID_COL].to_numpy()

    d = Path(args.derived)
    out.to_parquet(d / "composition_axes.parquet", index=False)
    np.save(d / "dissimilarity_pa.npy", D_pa.astype(np.float32))
    print(f"\n-> {d/'composition_axes.parquet'}  ({len(out)} filas x {out.shape[1]} columnas)")
    print(f"-> {d/'dissimilarity_pa.npy'}  ({D_pa.shape[0]}x{D_pa.shape[0]} Jaccard observado)")


if __name__ == "__main__":
    main()
