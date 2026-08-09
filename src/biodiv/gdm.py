"""Generalized Dissimilarity Modelling, and the sparse reduction that makes it usable here.

GDM (Ferrier, Manion, Elith & Richardson 2007, *Diversity and Distributions* 13:252-264,
doi:10.1111/j.1472-4642.2007.00341.x) regresses **pairwise compositional dissimilarity**
directly on pairwise environmental difference. It exists because the two things this project
runs into are exactly what it was designed for:

  1. **Saturation.** 57.6% of the 584,821 plot pairs here share no species at all and sit at
     Jaccard 1.0 exactly; the median pair is saturated. A linear ordination has to spend its
     axes representing a distance that has stopped varying, which is why two PCoA axes carry
     only 12.5% of the corrected eigenvalue mass. GDM's link, ``d = 1 - exp(-eta)``, is
     asymptotic at 1 by construction: saturation is the model's shape, not its error.

  2. **A varying rate of turnover along a gradient.** A degree of aridity does not buy the
     same amount of species replacement everywhere. GDM gives each predictor a monotone
     I-spline so the rate can vary along it, while keeping the response to *more* difference
     monotone — coefficients are constrained non-negative, so no predictor can be fitted to
     make two sites more similar the further apart they are.

**No ordination bottleneck.** The dissimilarity ceiling measured in
``scripts/20_composition_axes.py`` — a perfect model of the two axes reproduces observed
Jaccard at Spearman +0.305 — applies to the axis-regression design, not to this one. GDM is
scored on the dissimilarities themselves.

:func:`sparse_cca` implements the reduction of Leitao, Schwieder, Suess et al. (2015),
*Methods in Ecology and Evolution* 6:1204-1214, doi:10.1111/2041-210X.12378 — "Sparse
Generalised Dissimilarity Modelling" — via the penalised matrix decomposition of Witten,
Tibshirani & Hastie (2009). With 376 candidate predictors the pair design matrix would be
~1.3 GB and hopelessly collinear; SGDM projects onto a handful of sparse canonical
components first.

**The reduction is supervised and therefore leaks unless it is fold-local.** ``sparse_cca``
reads the species matrix. Fitting it once on all plots and then cross-validating the GDM
would report a number that no held-out site could reproduce. Every function here takes an
explicit set of training rows for that reason.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

#: Splines per predictor and the quantiles their knots sit at. Three knots at the 0th, 50th
#: and 100th percentile is the GDM default and is what the published applications use.
N_SPLINES = 3
KNOT_QUANTILES = (0.0, 0.5, 1.0)


def ispline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Monotone I-spline basis, (n, 3), each column rising from 0 to 1 over the knot range.

    Order-2 I-splines: piecewise quadratic, continuous, non-decreasing. Written out rather
    than pulled from a spline library because GDM needs the *integrated* (monotone) form and
    the standard bases return the derivative form, which is not what the non-negativity
    constraint is meant to act on.
    """
    k0, k1, k2 = knots
    x = np.asarray(x, dtype=float)
    out = np.zeros((len(x), 3))

    # first: rises over [k0, k1], flat at 1 after
    if k1 > k0:
        t = np.clip((x - k0) / (k1 - k0), 0, 1)
        out[:, 0] = t * (2 - t)
    # second: rises over the whole range, quadratic through the middle knot
    if k2 > k0:
        t = np.clip((x - k0) / (k2 - k0), 0, 1)
        out[:, 1] = t * t * (3 - 2 * t)
    # third: flat at 0 until k1, rises over [k1, k2]
    if k2 > k1:
        t = np.clip((x - k1) / (k2 - k1), 0, 1)
        out[:, 2] = t * t
    return out


def spline_transform(X: np.ndarray, knots: list[np.ndarray] | None = None
                     ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Expand (n, p) predictors into (n, 3p) I-spline coordinates.

    ``knots`` are returned so a test fold is expanded on the *training* fold's knots. Placing
    them on the test data would let the basis adapt to the held-out sites.
    """
    if knots is None:
        knots = [np.quantile(X[:, j], KNOT_QUANTILES) for j in range(X.shape[1])]
    return np.hstack([ispline_basis(X[:, j], knots[j]) for j in range(X.shape[1])]), knots


def pair_design(S: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """``|I(x_i) - I(x_j)|`` for the given pairs, with a leading intercept column."""
    d = np.abs(S[i] - S[j])
    return np.hstack([np.ones((len(i), 1), dtype=d.dtype), d])


def _deviance(beta: np.ndarray, A: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Binomial deviance of ``y`` under ``d = 1 - exp(-A beta)``, and its gradient.

    Written in terms of ``eta`` rather than of the fitted dissimilarity: ``log(1 - d)`` is
    exactly ``-eta``, so the ``d -> 1`` arm — which is where 58% of the pairs live — costs
    nothing numerically. Evaluating it as ``log(1 - d)`` instead would underflow on the
    majority of this dataset.
    """
    eta = np.clip(A @ beta, 1e-9, 50.0)
    expm = -np.expm1(-eta)                       # = 1 - exp(-eta), accurate for small eta
    dev = -2.0 * np.sum(y * np.log(np.maximum(expm, 1e-300)) - (1.0 - y) * eta)
    g_eta = -2.0 * (y * np.exp(-eta) / np.maximum(expm, 1e-300) - (1.0 - y))
    return float(dev), A.T @ g_eta


def fit_gdm(A: np.ndarray, y: np.ndarray, max_iter: int = 300) -> np.ndarray:
    """Fit the non-negative coefficients by L-BFGS-B on the binomial deviance.

    Non-negativity is the modelling assumption, not a numerical convenience: a negative
    coefficient would say two sites become *more* similar as they grow further apart on that
    predictor, which is not a turnover model.
    """
    beta0 = np.full(A.shape[1], 0.01)
    beta0[0] = 0.1
    res = minimize(_deviance, beta0, args=(A, y), jac=True, method="L-BFGS-B",
                   bounds=[(1e-8, None)] + [(0.0, None)] * (A.shape[1] - 1),
                   options=dict(maxiter=max_iter))
    return res.x


def predict_gdm(A: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return -np.expm1(-np.clip(A @ beta, 0.0, 50.0))


def binomial_deviance(y: np.ndarray, p: np.ndarray) -> float:
    """Deviance of a proportion response, measured from the saturated model.

    ``2 * sum[ y log(y/p) + (1-y) log((1-y)/(1-p)) ]``. The saturated term is what makes this
    zero at ``p == y``; dropping it — as the plain log-likelihood does — leaves the entropy of
    ``y`` inside the residual, so a perfect fit would still report a large deviance and the
    explained fraction would top out well below 1. With 58% of the pairs sitting at ``y = 1``
    exactly, that entropy term is not small.
    """
    y = np.clip(y, 0.0, 1.0)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(y > 0, y * np.log(y / p), 0.0)
        b = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - p)), 0.0)
    return float(2.0 * np.sum(a + b))


def deviance_explained(y: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of null deviance explained — GDM's own reported statistic.

    0 when the model does no better than the mean dissimilarity, 1 at a perfect fit.
    """
    null = binomial_deviance(y, np.full_like(y, float(np.clip(y.mean(), 1e-12, 1 - 1e-12))))
    return float(1.0 - binomial_deviance(y, pred) / null) if null > 0 else np.nan


# --------------------------------------------------------------------------------------
# sparse CCA — the SGDM reduction
# --------------------------------------------------------------------------------------

def _soft(a: np.ndarray, delta: float) -> np.ndarray:
    return np.sign(a) * np.maximum(np.abs(a) - delta, 0.0)


def _l1_project(a: np.ndarray, budget: float) -> np.ndarray:
    """Soft-threshold ``a`` so that ``||u||_1 <= budget`` with ``||u||_2 = 1``.

    Binary search on the threshold, exactly as in Witten et al.'s penalised matrix
    decomposition. If the unpenalised direction already satisfies the budget it is returned
    untouched.
    """
    u = a / max(np.linalg.norm(a), 1e-12)
    if np.abs(u).sum() <= budget:
        return u
    lo, hi = 0.0, np.abs(a).max()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = _soft(a, mid)
        nrm = np.linalg.norm(s)
        if nrm < 1e-12:
            hi = mid
            continue
        u = s / nrm
        if np.abs(u).sum() > budget:
            lo = mid
        else:
            hi = mid
    s = _soft(a, hi)
    return s / max(np.linalg.norm(s), 1e-12)


def sparse_cca(X: np.ndarray, Z: np.ndarray, n_components: int = 8,
               cx: float = 0.3, cz: float = 0.3, n_iter: int = 50) -> np.ndarray:
    """Sparse canonical vectors of ``X`` against the species matrix ``Z``.

    Returns the (p, k) loading matrix; project new data with ``X_new @ W``. ``cx`` and ``cz``
    are the L1 budgets as fractions of ``sqrt(ncol)``, which is how Witten et al. parameterise
    them; the SGDM paper picks them by a grid search on cross-validated GDM performance and
    the caller here is expected to do the same rather than trust these defaults.

    ``X`` and ``Z`` must be column-standardised **on the training rows only**.
    """
    K = X.T @ Z
    p, q = K.shape
    bx, bz = cx * np.sqrt(p), cz * np.sqrt(q)
    W = np.zeros((p, n_components))
    for c in range(n_components):
        v = np.random.default_rng(c).normal(size=q)
        v /= max(np.linalg.norm(v), 1e-12)
        u = np.zeros(p)
        for _ in range(n_iter):
            u_new = _l1_project(K @ v, bx)
            v_new = _l1_project(K.T @ u_new, bz)
            if np.allclose(u_new, u, atol=1e-8) and np.allclose(v_new, v, atol=1e-8):
                u, v = u_new, v_new
                break
            u, v = u_new, v_new
        W[:, c] = u
        K = K - float(u @ K @ v) * np.outer(u, v)      # deflate
    return W
