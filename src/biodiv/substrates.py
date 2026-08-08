"""Model inputs built from the stored 52-week PhenoShape curves.

Three kinds of substrate, and the distinction is the point of the benchmark:

``curve1d``
    The curve itself, ``(C, 52)``. The 1-D control: whatever a 2-D model gains has to be
    gained over this.

``stack5``, ``pxcube``
    **Images whose vertical axis is not manufactured.** ``stack5`` stacks the five vegetation
    indices as rows, so a 3x3 kernel reads the contrast between indices within the same week —
    greenness against moisture, at green-up. ``pxcube`` stacks the 25 pixels of the plot's 5x5
    window as rows, sorted by their own mean, so the vertical axis is *within-plot
    phenological heterogeneity*: a mechanistic predictor of beta diversity and of LCBD, which
    are three of the nine targets. The 25 rows are one sample, so there is no fold-leakage
    question — this uses the pixels as a representation, not as augmentation.

the transform catalogue
    ``reshape``, ``gaf``, ``mtf``, ``cwt`` and the rest, from :mod:`biodiv.transforms1d`. Here
    the second axis *is* manufactured, which is exactly the comparison being made.

``docs/03_cnn_architecture.md`` proposed a fourth, the year x DOY *phenocube*. It is not built:
the stored curve is a single PhenoShape fit pooled over a causal 3-year window, by design, to
give a stable ecosystem-level seasonal profile. Splitting that window back into per-year
curves would need 9-30 clear observations to constrain 52 weekly steps, which
``docs/05_data_acquisition.md`` already identifies as insufficient. ``stack5`` and ``pxcube``
serve the same argumentative role with data that exists.

Padding is declared per substrate because the two axes do not always mean the same thing —
see :data:`PAD_MODE`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from .transforms1d import NGS, make_transform

STEP_COLS = feat.STEP_COLS
INDICES = feat.INDICES

#: Vertical/horizontal padding for each substrate's convolutions.
#:
#: The DOY axis is circular: week 51 is adjacent to week 0, and zero padding invents a trough
#: there that a kernel reads as a real phenological event. But the *vertical* axis is only
#: circular for the transforms whose both axes are DOY (gaf, mtf, ndi, cos2d). For `stack5`
#: the rows are vegetation indices and for `pxcube` they are pixels — wrapping those would
#: make the first index a neighbour of the last, which means nothing. For the folded `reshape`
#: images both axes are artefacts of the folding, so neither wraps.
PAD_MODE: dict[str, str | tuple[str, str]] = {
    "curve1d": "circular",
    "curve5": "circular",
    "stack5": ("zeros", "circular"),
    "pxcube": ("zeros", "circular"),
    "cwt": ("zeros", "circular"),
    "spectrogram": "zeros",
    "reshape": "zeros",
    "serpentine": "zeros",
    "hilbert": "zeros",
    "gaf": "circular",
    "mtf": "circular",
    "ndi": "circular",
    "cos2d": "zeros",
}

NATIVE = {"curve1d", "curve5", "stack5", "pxcube"}


@dataclass
class Substrate:
    """A materialised input tensor plus everything needed to interpret it later."""
    name: str
    X: np.ndarray                     # (N, C, H, W) or (N, C, W) for curve1d
    plot_ids: pd.Index
    pad_mode: str | tuple[str, str]
    curves: np.ndarray                # (N, C, 52) source curves, for augmentation
    rows: list[str]                   # meaning of the vertical axis, for interpretability

    @property
    def c_in(self) -> int:
        return self.X.shape[1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.X.shape[1:]


# --------------------------------------------------------------------------------------
# curve loading
# --------------------------------------------------------------------------------------

def load_curves(index: str | list[str], derived: str = "data/derived",
                px: str = "mean5x5", ids: pd.Index | None = None) -> tuple[np.ndarray, pd.Index]:
    """Return ``(N, C, 52)`` curves in canonical plot order. C = number of indices requested."""
    t = feat.load_tables(derived)
    ids = pd.Index(t.plots[feat.ID_COL]) if ids is None else pd.Index(ids)
    names = [index] if isinstance(index, str) else list(index)
    out = np.stack(
        [t.curves.xs((ix, px), level=("index", "px")).reindex(ids)[STEP_COLS].to_numpy(np.float32)
         for ix in names], axis=1)
    return out, ids


def load_pixel_curves(index: str, derived: str = "data/derived",
                      ids: pd.Index | None = None) -> tuple[np.ndarray, pd.Index]:
    """Return ``(N, 25, 52)`` — the 25 pixels of each plot, rows sorted by their own mean.

    Sorting makes the vertical ordering deterministic and meaningful (low-greenness pixels at
    the top). The raw (y, x) order would make the axis an arbitrary permutation that differs
    in meaning between plots, which a convolution cannot use.
    """
    t = feat.load_tables(derived)
    ids = pd.Index(t.plots[feat.ID_COL]) if ids is None else pd.Index(ids)
    px = pd.read_parquet(t.pixels)
    px = px[px["index"] == index]
    arr = (px.set_index(["plot_id", "y", "x"])[STEP_COLS]
             .sort_index().to_numpy(np.float32).reshape(-1, 25, NGS))
    order = pd.Index(sorted(px["plot_id"].unique()))
    cube = pd.Series(list(arr), index=order).reindex(ids)
    out = np.stack(cube.to_numpy())
    means = out.mean(axis=2)                       # (N, 25)
    rank = np.argsort(means, axis=1)
    out = np.take_along_axis(out, rank[:, :, None], axis=1)
    return out, ids


# --------------------------------------------------------------------------------------
# substrate construction
# --------------------------------------------------------------------------------------

def make_substrate(name: str, index: str | None = None, derived: str = "data/derived",
                   px: str = "mean5x5", normalize: str = "none",
                   rotation: str = "trough", ids: pd.Index | None = None,
                   **tf_kw) -> Substrate:
    """Build ``(N, C, ...)`` model input.

    ``rotation``: ``trough`` keeps the stored global anchor (DOY 108, applied by script 04);
    ``calendar`` undoes it with a circular roll so the array boundary falls at 1 January
    again. The rotation is *global* — the same constant for every plot — because between-plot
    phase differences are signal (winter-peaking matorral against summer-peaking sclerophyll)
    and centring each plot on its own peak would delete them.
    """
    if name == "curve5":
        # The five indices as *channels* of a 1-D signal, not as rows of an image. Paired with
        # `stack5` this isolates what the second dimension adds: same information, one read by
        # channel mixing and the other by a kernel that spans indices and weeks together.
        curves, ids = load_curves(INDICES, derived, px, ids)
        curves = _rotate(curves, rotation)
        return Substrate(name, curves, ids, "circular", curves, rows=list(INDICES))

    if name == "stack5":
        curves, ids = load_curves(INDICES, derived, px, ids)
        curves = _rotate(curves, rotation)
        return Substrate(name, curves[:, None, :, :], ids, PAD_MODE[name], curves,
                         rows=list(INDICES))

    if name == "pxcube":
        if index is None:
            raise ValueError("pxcube needs --index")
        cube, ids = load_pixel_curves(index, derived, ids)
        cube = _rotate(cube, rotation)
        return Substrate(name, cube[:, None, :, :], ids, PAD_MODE[name], cube,
                         rows=[f"px{i:02d}" for i in range(cube.shape[1])])

    if index is None:
        raise ValueError(f"substrate {name!r} needs --index")
    curves, ids = load_curves(index, derived, px, ids)
    curves = _rotate(curves, rotation)

    if name == "curve1d":
        return Substrate(name, curves, ids, PAD_MODE[name], curves, rows=[index])

    tf = make_transform(name, normalize=normalize, **tf_kw)
    imgs = np.stack([tf.transform(c) for c in curves[:, 0, :]])
    if imgs.ndim == 3:
        imgs = imgs[:, None, :, :]
    else:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    return Substrate(name, np.ascontiguousarray(imgs, dtype=np.float32), ids,
                     PAD_MODE[name], curves, rows=[f"row{i}" for i in range(imgs.shape[2])])


def _rotate(x: np.ndarray, rotation: str) -> np.ndarray:
    """Curves are stored trough-anchored at DOY 108; ``calendar`` rolls them back.

    108 / 7 = 15.4 steps, so the calendar roll is +15 steps. Fractional accuracy is not
    needed: the point of the factor is whether the array boundary sits in the dormant season
    or inside the growing season, and 3 days either way does not change that.
    """
    if rotation == "trough":
        return x
    if rotation == "calendar":
        return np.roll(x, shift=15, axis=-1)
    raise ValueError(f"unknown rotation {rotation!r}")


def load_topo_patches(derived: str = "data/derived",
                      ids: pd.Index | None = None) -> tuple[np.ndarray, list[str]]:
    """(N, 9, 5, 5) terrain patches for the `patch` fusion option."""
    import xarray as xr
    t = feat.load_tables(derived)
    ids = pd.Index(t.plots[feat.ID_COL]) if ids is None else pd.Index(ids)
    path = Path(derived) / "topography" / "topography_patches.nc"
    with xr.open_dataset(path) as ds:
        varnames = [v for v in ds.data_vars]
        arr = np.stack([ds[v].sel(plot_id=ids.to_numpy()).to_numpy() for v in varnames], axis=1)
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    m = arr.mean(axis=(0, 2, 3), keepdims=True)
    s = arr.std(axis=(0, 2, 3), keepdims=True)
    return (arr - m) / np.where(s > 1e-8, s, 1.0), varnames


# --------------------------------------------------------------------------------------
# interpretability: fold a 2-D attribution map back onto the week axis
# --------------------------------------------------------------------------------------

def unfold_to_doy(attr: np.ndarray, substrate: str, n: int = NGS) -> np.ndarray:
    """Map an attribution image back to a length-52 profile over the DOY axis.

    This is what turns "the model looks at this pixel" into "the model looks at this week",
    which is the only form in which the result is ecologically readable. Each substrate needs
    its own inverse; ``reshape`` and ``serpentine`` are exact, the square transforms use the
    row marginal (row i corresponds to step i and the maps are symmetric).
    """
    a = np.asarray(attr, dtype=np.float64)
    if a.ndim == 3:                        # (C, H, W) -> collapse channels
        a = np.abs(a).mean(axis=0)

    if substrate in ("reshape", "hilbert"):
        if substrate == "hilbert":
            from .transforms1d import _d2xy
            order = int(np.log2(a.shape[0]))
            flat = np.array([a[y, x] for x, y in (_d2xy(order, d) for d in range(a.size))])
        else:
            flat = a.reshape(-1)
        return flat[:n]
    if substrate == "serpentine":
        b = a.copy()
        b[1::2] = b[1::2, ::-1]
        return b.reshape(-1)[:n]
    if substrate in ("gaf", "mtf", "ndi", "cos2d"):
        return np.abs(a).mean(axis=0)[:n]
    if substrate in ("cwt", "spectrogram", "stack5", "pxcube"):
        col = np.abs(a).mean(axis=0)
        if col.size == n:
            return col
        idx = np.linspace(0, col.size - 1, n)
        return np.interp(idx, np.arange(col.size), col)
    if substrate == "curve1d":
        return np.abs(a).reshape(-1)[:n]
    raise ValueError(f"no inverse map defined for substrate {substrate!r}")


def step_to_doy(derived: str = "data/derived") -> np.ndarray:
    """Median real day-of-year of each of the 52 steps, for labelling attribution plots."""
    t = feat.load_tables(derived)
    return t.doy_grid[STEP_COLS].median(axis=0).to_numpy()
