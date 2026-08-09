"""Geometric median composites and the three MADs, computed on the stored observation cubes.

A geomedian is the multivariate median of the six surface-reflectance bands over time: a
composite that carries **no notion of temporal shape at all**. That is exactly why it is
worth computing here. The project's whole claim is that phenological shape predicts
composition; the honest control is a composite built from the same pixels, the same 150 m
window and the same cloud mask, differing only in that it throws the time axis away.

Because ``scripts/02_extract_phenology.py`` stored ``obs_band_*`` alongside the fitted curve,
this needs no datacube connection, no requester-pays and no S3 — the cubes on disk are the
whole input.

**Why the geometric median rather than a per-band median.** A band-wise median mixes
reflectances from different dates and can land on a spectrum no observation ever had — it is
not a valid surface. The geometric median minimises the sum of Euclidean distances in the
full 6-band space, so the result stays inside the observed spectral cloud and band ratios
computed from it (NDVI, SAVI, ...) remain physically meaningful.

**The three MADs are the point, not a by-product.** They measure how far the observations
scatter around that composite, which is within-window variability *without any temporal
shape*. Against the phenological curve they are the cleanest available contrast: if the
curve beats them, the gain is attributable to shape; if it does not, it never was.

  - ``emad``  Euclidean distance, sensitive to overall brightness change
  - ``smad``  cosine distance, sensitive to change in spectral *shape* at constant brightness
  - ``bcmad`` Bray-Curtis, bounded and dominated by the brighter bands

Definitions follow Roberts, Mueller & McIntyre (2017), *High-dimensional pixel composites
from Earth observation time series*, IEEE TGRS 55(11), which is what ``odc-algo``'s
``geomedian_with_mads`` implements.
"""

from __future__ import annotations

import numpy as np

#: Reflectance bands, in the order the geomedian is computed over. Fixed because ``emad`` is
#: a distance in this space and would change meaning if the set changed.
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]

#: Below this many cloud-free observations a pixel gets NaN rather than a geomedian. A
#: median over five dates is not a composite, it is one of the five dates.
MIN_OBS = 10

SAVI_L = 0.5


def geometric_median(X: np.ndarray, eps: float = 1e-7, max_iter: int = 200) -> np.ndarray:
    """Weiszfeld's algorithm with the Vardi & Zhang (2000) modification.

    ``X`` is (n_obs, n_bands) and must already be free of NaN. The plain Weiszfeld iteration
    divides by the distance to the current iterate, so it breaks down the moment the iterate
    lands exactly on a data point — which happens routinely with few observations and
    quantised reflectance. The modification handles that case in closed form instead of
    producing an infinity.
    """
    if X.shape[0] == 1:
        return X[0].copy()
    y = np.median(X, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        on_point = d < eps
        n_on = int(on_point.sum())
        if n_on == X.shape[0]:
            return y
        w = 1.0 / d[~on_point]
        T = (X[~on_point] * w[:, None]).sum(axis=0) / w.sum()
        if n_on == 0:
            y_next = T
        else:
            # y coincides with n_on data points: shrink towards T by the amount the
            # subgradient allows, per Vardi & Zhang eq. (2.5).
            R = np.linalg.norm((X[~on_point] - y) * w[:, None], axis=0).sum()
            r = min(1.0, n_on / max(R, 1e-12))
            y_next = (1.0 - r) * T + r * y
        if np.linalg.norm(y_next - y) < eps:
            return y_next
        y = y_next
    return y


def mads(X: np.ndarray, gm: np.ndarray) -> tuple[float, float, float]:
    """The three median absolute deviations of ``X`` about the geomedian ``gm``."""
    diff = X - gm
    emad = float(np.median(np.linalg.norm(diff, axis=1)))

    nx = np.linalg.norm(X, axis=1)
    ng = float(np.linalg.norm(gm))
    denom = nx * ng
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(denom > 0, (X @ gm) / denom, 1.0)
    smad = float(np.median(1.0 - np.clip(cos, -1.0, 1.0)))

    num = np.abs(diff).sum(axis=1)
    den = np.abs(X + gm).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bc = np.where(den > 0, num / den, 0.0)
    bcmad = float(np.median(bc))
    return emad, smad, bcmad


def indices_from_bands(b: dict[str, float]) -> dict[str, float]:
    """The project's five indices, from the geomedian spectrum.

    Formulas are copied from :func:`biodiv.cube.to_indices` including the clips, so a
    geomedian NDVI and an observation NDVI are the same quantity computed on different
    inputs — otherwise the composite-versus-curve contrast would confound the definition
    with the compositing step.
    """
    blue, red, nir, swir2 = b["blue"], b["red"], b["nir"], b["swir2"]
    ndvi = float(np.clip((nir - red) / (nir + red), -1, 1)) if (nir + red) else np.nan
    den_evi = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = float(np.clip(2.5 * (nir - red) / den_evi, -1, 1)) if den_evi else np.nan
    nbr = float(np.clip((nir - swir2) / (nir + swir2), -1, 1)) if (nir + swir2) else np.nan
    den_savi = nir + red + SAVI_L
    savi = (float(np.clip(((nir - red) / den_savi) * (1.0 + SAVI_L), -1.5, 1.5))
            if den_savi else np.nan)
    return {"ndvi": ndvi, "evi": evi, "kndvi": float(np.tanh(ndvi ** 2)), "nbr": nbr,
            "savi": savi}


# --------------------------------------------------------------------------------------
# observation-level composites
# --------------------------------------------------------------------------------------

#: Percentiles kept from the observation distribution. p10 and p90 are the useful ends: the
#: true min and max of a Landsat series are usually one undetected cloud edge or one shadow,
#: so they are recorded but never relied on.
PCTLS = [10, 25, 50, 75, 90]

#: Austral seasons by calendar month. DJF is the growing season in central Chile and JJA the
#: wet winter; splitting on the southern-hemisphere calendar rather than the northern one is
#: what makes a "seasonal contrast" mean anything here.
SEASONS = {"djf": (12, 1, 2), "mam": (3, 4, 5), "jja": (6, 7, 8), "son": (9, 10, 11)}


def composite_stats(v: np.ndarray) -> dict[str, float]:
    """Distribution summary of one index at one pixel: level, spread, and nothing else.

    No date and no ordering enters any of these — that is the whole point. ``cv`` is the
    temporal coefficient of variation, which measured on this dataset (NBR, Spearman +0.53
    against ``pcoa1_pa``) carries signal that no level statistic does.
    """
    v = v[np.isfinite(v)]
    if v.size < 3:
        return {}
    q = np.percentile(v, PCTLS)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {
        "mean": mean, "sd": sd,
        **{f"p{p}": float(x) for p, x in zip(PCTLS, q)},
        "min": float(v.min()), "max": float(v.max()),
        "range": float(v.max() - v.min()),
        "iqr": float(q[3] - q[1]),
        "cv": float(sd / abs(mean)) if abs(mean) > 1e-9 else np.nan,
        "n": float(v.size),
    }


def seasonal_medians(v: np.ndarray, month: np.ndarray) -> dict[str, float]:
    """Median of one index within each austral season, plus the summer-minus-winter contrast.

    A seasonal contrast is the crudest possible phenology: two numbers instead of a fitted
    52-step curve. If it recovers most of what the curve does, that is the result — it says
    the useful part of the temporal signal is an amplitude, not a shape.
    """
    out: dict[str, float] = {}
    for name, months in SEASONS.items():
        sel = np.isin(month, months) & np.isfinite(v)
        out[name] = float(np.median(v[sel])) if sel.sum() >= 3 else np.nan
    out["djf_jja"] = out["djf"] - out["jja"]
    out["amp_season"] = (np.nanmax([out[s] for s in SEASONS])
                         - np.nanmin([out[s] for s in SEASONS]))
    return out
