#!/usr/bin/env python3
"""Python port of ``scripts/07_compute_taxonomic_beta_responses.R``.

Same nine targets, same two-tier design, same numbers. It exists because the analysis
machine that holds the ``.nc`` observation cubes has no R installed, and without
``biodiversity_responses.parquet`` no model in this project can be fitted at all.

Every step below mirrors a named R function. The mapping, with the source each formula was
taken from, because "equivalent" is a claim that has to be checkable:

``hillR::hill_taxa(comm, q, MARGIN=1)``
    Rows relativised to sum 1, then ``q0`` = number of non-zero species, ``q1`` =
    ``exp(-sum p log p)``, ``q2`` = ``1 / sum(p^2)``. The general form is
    ``(sum p^q)^(1/(1-q))``.

``adespatial::beta.div(Y, method="hellinger")``
    Y is Hellinger-transformed (``sqrt(y_ij / rowsum_i)``), then ``s = (Y - colmean)^2``,
    ``SStotal = sum(s)`` and ``LCBD_i = rowsum_i(s) / SStotal``. No distance matrix is
    formed: for the five transform-based methods beta.div takes this direct route.

``adespatial::beta.div(Y, method="jaccard")``
    The distance-based branch. ``dist.ldc(Y, "jaccard")`` returns ``sqrt(D_jaccard)``, which
    is the Euclidean version, so ``D^2 == D_jaccard``. Then
    ``SStotal = sum_{i<j} D_jaccard[i,j] / n``, ``delta1 = Gower(-0.5 * D_jaccard)`` and
    ``LCBD_i = delta1[i,i] / SStotal``. Both branches make ``sum(LCBD) == 1``, which is
    asserted here rather than assumed.

``ape::pcoa(d, correction="cailliez")``
    Classical PCoA on the Gower-centred ``-0.5 D^2``; when the smallest eigenvalue is
    negative, Cailliez's (1983) constant ``c1`` is the largest real eigenvalue of the
    ``2n x 2n`` block matrix ``[[0, 2*delta1], [-I, -4*delta2]]`` with
    ``delta2 = Gower(-0.5 D)``. The decomposition is then redone on ``-0.5 (D + c1)^2``
    with a zero diagonal.

**Eigenvector sign is arbitrary and does not match R.** LAPACK fixes no convention, so
``pcoa1`` here may be the negative of ``pcoa1`` there. It changes no R-squared, no rank
correlation and no split a tree can make. A deterministic convention is imposed anyway
(the loading of largest absolute value is made positive) so that reruns agree with each
other.

Usage:
    python scripts/07b_compute_responses_python.py \
        --zip data/20602096.zip --plots data/derived/plots_subset.parquet \
        --out data/derived/biodiversity_responses.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import io_parcelas as iop                                    # noqa: E402

ID_COL = "PlotObservationID"
EPS = float(np.sqrt(np.finfo(float).eps))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# community matrix
# --------------------------------------------------------------------------------------

def build_comm(zip_path: str, plot_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """(n_plots, n_species) abundance matrix in the canonical plot order.

    A handful of (plot, species) pairs carry two records — the same taxon recorded in two
    strata or growth forms — and their ``Value`` is summed, the standard aggregation in
    vegetation ecology and what ``aggregate(Value ~ ...)`` does on the R side.
    """
    long = iop.load_long(zip_path)
    long[ID_COL] = long[ID_COL].astype(str)
    long = long[long[ID_COL].isin(set(plot_ids))]
    _log(f"registros del subset: {len(long)} filas, {long['Accepted_species'].nunique()} taxones")

    agg = long.groupby([ID_COL, "Accepted_species"], as_index=False)["Value"].sum()
    wide = agg.pivot(index=ID_COL, columns="Accepted_species", values="Value").fillna(0.0)
    wide = wide.reindex(plot_ids).fillna(0.0)
    return wide.to_numpy(dtype=np.float64), list(wide.columns)


# --------------------------------------------------------------------------------------
# Hill numbers
# --------------------------------------------------------------------------------------

def hill_taxa(comm: np.ndarray, q: int) -> np.ndarray:
    """``hillR::hill_taxa`` with ``MARGIN = 1``: rows relativised, then the Hill number."""
    total = comm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(total > 0, comm / total, 0.0)
    if q == 0:
        return (p > 0).sum(axis=1).astype(float)
    if q == 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            terms = np.where(p > 0, -p * np.log(p), 0.0)
        return np.exp(terms.sum(axis=1))
    return (p ** q).sum(axis=1) ** (1.0 / (1.0 - q))


# --------------------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------------------

def jaccard_binary(Y: np.ndarray) -> np.ndarray:
    """``vegan::vegdist(Y, "jaccard", binary=TRUE)`` as a full square matrix.

    ``(b + c) / (a + b + c)`` on presence/absence, computed from the shared-species inner
    product so the 999 permutations stay in BLAS rather than in Python.
    """
    B = (Y > 0).astype(np.float64)
    shared = B @ B.T
    r = np.diag(shared).copy()
    union = r[:, None] + r[None, :] - shared
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(union > 0, 1.0 - shared / union, 0.0)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def bray_curtis(Y: np.ndarray) -> np.ndarray:
    """``vegan::vegdist(Y, "bray")``: ``sum|x - y| / sum(x + y)``."""
    n = Y.shape[0]
    D = np.zeros((n, n))
    tot = Y.sum(axis=1)
    for i in range(n):
        num = np.abs(Y[i][None, :] - Y).sum(axis=1)
        den = tot[i] + tot
        with np.errstate(invalid="ignore", divide="ignore"):
            D[i] = np.where(den > 0, num / den, 0.0)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def gower_centre(M: np.ndarray) -> np.ndarray:
    """``(I - 11'/n) M (I - 11'/n)`` without materialising the projector."""
    rm = M.mean(axis=1, keepdims=True)
    cm = M.mean(axis=0, keepdims=True)
    return M - rm - cm + M.mean()


# --------------------------------------------------------------------------------------
# LCBD
# --------------------------------------------------------------------------------------

def lcbd_hellinger(Y: np.ndarray) -> np.ndarray:
    total = Y.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = np.sqrt(np.where(total > 0, Y / total, 0.0))
    s = (H - H.mean(axis=0, keepdims=True)) ** 2
    return s.sum(axis=1) / s.sum()


def lcbd_from_distance(Dsq: np.ndarray) -> np.ndarray:
    """LCBD from the *squared* distances of beta.div's distance-based branch.

    For ``method="jaccard"`` the squared distance is the plain Jaccard dissimilarity,
    because ``dist.ldc`` already returns its square root.
    """
    n = Dsq.shape[0]
    ss_total = Dsq[np.triu_indices(n, 1)].sum() / n
    delta1 = gower_centre(-0.5 * Dsq)
    return np.diag(delta1) / ss_total


def _lcbd_jaccard_rowform(D: np.ndarray) -> np.ndarray:
    """Same quantity, in closed form — used to permute cheaply and to cross-check.

    ``M = -0.5 D`` has a zero diagonal, so ``Gower(M)[i,i] = -2*rowmean_i(M) + mean(M)``,
    which reduces to ``rowmean_i(D) - mean(D)/2``; and ``SStotal = n * mean(D) / 2``.
    """
    n = D.shape[0]
    gm = D.mean()
    return (D.mean(axis=1) - 0.5 * gm) / (n * gm / 2.0)


def permutation_p_lcbd(Y: np.ndarray, observed: np.ndarray, kind: str,
                       nperm: int, seed: int) -> np.ndarray:
    """``beta.div``'s test: permute each species column independently, ``nperm`` times."""
    rng = np.random.default_rng(seed)
    n, p = Y.shape
    ge = np.ones(n)
    for _ in range(nperm):
        Yp = np.empty_like(Y)
        for j in range(p):
            Yp[:, j] = rng.permutation(Y[:, j])
        perm = (_lcbd_jaccard_rowform(jaccard_binary(Yp)) if kind == "jaccard"
                else lcbd_hellinger(Yp))
        ge += (perm >= observed)
    return ge / (nperm + 1)


# --------------------------------------------------------------------------------------
# PCoA
# --------------------------------------------------------------------------------------

def pcoa_cailliez(D: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``ape::pcoa(D, correction="cailliez")``. Returns ``(scores[:, :k], rel_eig)``.

    ``rel_eig`` is ``Rel_corr_eig`` when the correction fired and ``Relative_eig`` when it
    did not, matching which column the R script reads.
    """
    D = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(D, 0.0)
    n = D.shape[0]

    delta1 = gower_centre(-0.5 * D ** 2)
    vals, vecs = np.linalg.eigh(delta1)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    if vals.min() > -EPS:
        trace = np.trace(delta1)
        pos = vals > EPS
        return _scores(vals, vecs, pos, k), vals[pos] / trace

    _log(f"    autovalor minimo {vals.min():.4g} < 0 -> correccion de Cailliez")
    delta2 = gower_centre(-0.5 * D)
    top = np.hstack([np.zeros((n, n)), 2.0 * delta1])
    bot = np.hstack([-np.eye(n), -4.0 * delta2])
    c1 = float(np.max(np.real(np.linalg.eigvals(np.vstack([top, bot])))))
    _log(f"    c1 = {c1:.6f}")

    Dc = -0.5 * (D + c1) ** 2
    np.fill_diagonal(Dc, 0.0)
    delta1c = gower_centre(Dc)
    trace = np.trace(delta1c)
    vals, vecs = np.linalg.eigh(delta1c)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = vals > EPS
    return _scores(vals, vecs, pos, k), vals[pos] / trace


def _scores(vals: np.ndarray, vecs: np.ndarray, pos: np.ndarray, k: int) -> np.ndarray:
    """Principal coordinates: eigenvectors scaled by the square root of the eigenvalue.

    The sign convention — largest absolute loading made positive — is ours, not R's. It
    exists so two runs of this script agree; it cannot be made to agree with ape, and does
    not need to be (see the module docstring).
    """
    kk = min(k, int(pos.sum()))
    V = vecs[:, :kk] * np.sqrt(vals[:kk])
    for j in range(kk):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1.0
    return V


# --------------------------------------------------------------------------------------
# one facet
# --------------------------------------------------------------------------------------

def run_facet(comm_beta: np.ndarray, comm_dist: np.ndarray, ids: list[str],
              method_beta: str, dist_method: str, label: str,
              k_axes: int, nperm: int, seed: int) -> pd.DataFrame:
    keep = comm_beta.sum(axis=0) > 0
    comm_beta, comm_dist = comm_beta[:, keep], comm_dist[:, keep]
    _log(f"[{label}] {comm_beta.shape[0]} parcelas x {comm_beta.shape[1]} especies")

    _log(f"[{label}] beta.div (method={method_beta}, nperm={nperm}) ...")
    if method_beta == "jaccard":
        Djac = jaccard_binary(comm_beta)
        lcbd = lcbd_from_distance(Djac)
        closed = _lcbd_jaccard_rowform(Djac)
        assert np.allclose(lcbd, closed, atol=1e-12), "LCBD closed form disagrees with Gower"
    elif method_beta == "hellinger":
        lcbd = lcbd_hellinger(comm_beta)
    else:
        raise ValueError(method_beta)
    assert abs(lcbd.sum() - 1.0) < 1e-8, f"sum(LCBD) = {lcbd.sum():.6f}, debe ser 1"

    p_lcbd = (permutation_p_lcbd(comm_beta, lcbd, method_beta, nperm, seed)
              if nperm > 0 else np.full(len(ids), np.nan))
    _log(f"[{label}] beta.div listo")

    _log(f"[{label}] PCoA (distance={dist_method}) ...")
    D = jaccard_binary(comm_dist) if dist_method == "jaccard" else bray_curtis(comm_dist)
    scores, rel_eig = pcoa_cailliez(D, k=k_axes)
    _log(f"[{label}] PCoA listo")

    var_explained = float(rel_eig[:scores.shape[1]].sum())
    print(f"[{label}] PCoA varianza explicada por {scores.shape[1]} ejes: "
          f"{100 * var_explained:.1f}%")
    if var_explained < 0.2:
        print(f"AVISO: [{label}] PCoA explica poca varianza "
              f"({100 * var_explained:.1f}% en {scores.shape[1]} ejes)")

    out = pd.DataFrame({ID_COL: ids, "lcbd": lcbd, "p_lcbd": p_lcbd})
    for j in range(scores.shape[1]):
        out[f"pcoa{j + 1}"] = scores[:, j]
    return out


# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="data/20602096.zip")
    p.add_argument("--plots", default="data/derived/plots_subset.parquet")
    p.add_argument("--out", default="data/derived/biodiversity_responses.parquet")
    p.add_argument("--k-axes", type=int, default=2, dest="k_axes")
    p.add_argument("--nperm", type=int, default=999)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    plots = pd.read_parquet(args.plots)
    plot_ids = plots[ID_COL].astype(str).tolist()
    stratum = dict(zip(plot_ids, plots["stratum"].astype(str)))
    print(f"plots_subset: {len(plot_ids)} parcelas")

    comm_full, _ = build_comm(args.zip, plot_ids)

    hill = {f"hill_q{q}": hill_taxa(comm_full, q) for q in (0, 1, 2)}
    r = np.corrcoef(hill["hill_q0"], plots["richness"].to_numpy(float))[0, 1]
    print(f"cor(hill_q0, richness) = {r:.4f} (chequeo de sanidad, debiera ser ~1)")
    if r < 0.999:
        raise SystemExit(f"hill_q0 no reproduce richness (r = {r:.4f}); revisar la agregacion")

    comm_pa = (comm_full > 0).astype(np.float64)
    pa = run_facet(comm_pa, comm_pa, plot_ids, "jaccard", "jaccard", "presence-absence",
                   args.k_axes, args.nperm, args.seed)
    pa = pa.rename(columns={c: f"{c}_pa" for c in pa.columns if c != ID_COL})

    # Abundance-weighted, `cover` stratum only. Rows are relativised before Bray-Curtis
    # because Cover and Cover_st are on different scales (~100 vs ~4700); beta.div's
    # hellinger already relativises internally, so the untransformed matrix goes there.
    cover_ids = [i for i in plot_ids if stratum[i] == "cover"]
    pos = {pid: j for j, pid in enumerate(plot_ids)}
    comm_cover = comm_full[[pos[i] for i in cover_ids]]
    tot = comm_cover.sum(axis=1, keepdims=True)
    comm_cover_rel = np.where(tot > 0, comm_cover / tot, 0.0)
    cover = run_facet(comm_cover, comm_cover_rel, cover_ids, "hellinger", "bray", "cover",
                      args.k_axes, args.nperm, args.seed)
    cover = cover.rename(columns={c: f"{c}_cover" for c in cover.columns if c != ID_COL})

    resp = pd.DataFrame({ID_COL: plot_ids, **hill})
    resp = resp.merge(pa, on=ID_COL, how="left").merge(cover, on=ID_COL, how="left")
    resp = resp.set_index(ID_COL).reindex(plot_ids).reset_index()
    resp[ID_COL] = plots[ID_COL].to_numpy()          # restore the original dtype

    n_na = int(resp["lcbd_cover"].isna().sum())
    expected = sum(1 for i in plot_ids if stratum[i] != "cover")
    print(f"filas: {len(resp)}, NA en columnas *_cover: {n_na} "
          f"(esperado = parcelas fuera de stratum 'cover' = {expected})")
    if n_na != expected:
        raise SystemExit("el patron de NA de las columnas *_cover no es el esperado")

    for col in [c for c in resp.columns if c.startswith("pcoa")]:
        x = resp[col].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() < 8:
            continue
        mad = np.median(np.abs(x[ok] - np.median(x[ok]))) * 1.4826
        if mad <= 0:
            continue
        z = np.abs(x - np.median(x[ok])) / mad
        extreme = np.where(ok & (z > 10))[0]
        if len(extreme):
            print(f"AVISO: {len(extreme)} parcela(s) con {col} extremo "
                  f"(parcela con poca informacion composicional): "
                  f"{', '.join(str(resp[ID_COL].iloc[i]) for i in extreme)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    resp.to_parquet(args.out, index=False)
    print(f"escrito {args.out} ({len(resp)} filas, {resp.shape[1]} columnas)")


if __name__ == "__main__":
    main()
