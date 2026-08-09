"""Consuming the cross-validation schemes, and the nested inner split.

The fold tables are fully materialised long tables — one row per (scheme, fold, plot) with a
``split`` label — so nothing here re-derives a partition. It only reads, iterates and asserts.

**The inner split is where leakage comes back in.** Early stopping needs a validation set
carved out of the training fold. If that carve is random, plots from the same owner (or the
same 20 km block) land on both sides of it, the stopping epoch is chosen against a leaky
signal, and the careful outer grouping is quietly undone. :func:`inner_split` therefore
always groups by the *same* column as the outer scheme.

**Pixel rows never define a fold.** ``phenoshape_pixels.parquet`` has 25 rows per plot that
share a 150 m footprint. Folds are assigned by ``plot_id`` and pixel rows are selected by
membership; :func:`assert_no_leak` enforces it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import cv_groups

ID_COL = "PlotObservationID"

#: Grouping column behind each scheme. Used by :func:`inner_split` so the nested validation
#: split respects the same structure as the outer one. ``kfold5_random`` deliberately maps to
#: ``block20``: the outer split is random by design, but letting early stopping peek at a
#: spatially adjacent plot would corrupt even the optimistic reference.
SCHEME_GROUP = {
    "kfold5_owner": "Owner",
    "kfold5_block20": "block20",
    "kfold5_dataset": "metadata_id",
    "kfold5_location": "Location",
    "kfold5_random": "block20",
    "lodo_owner": "Owner",
    "lodo_dataset": "metadata_id",
    "kfold5_window": "window_component",
    "kfold5_owner_window": "owner_window",
}

#: LODO schemes do not partition the plots — groups below the minimum size are never a test
#: fold — so a pooled out-of-fold table is undefined for them and metrics are per-fold.
NON_PARTITIONING = {"lodo_owner", "lodo_dataset"}


def load_schemes(path: Path | str = "data/derived/cv_folds_modelling.parquet") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python scripts/08_build_modelling_folds.py` first."
        )
    return pd.read_parquet(path)


def scheme_folds(cv: pd.DataFrame, scheme: str) -> pd.DataFrame:
    if scheme not in set(cv["scheme"]):
        raise KeyError(f"unknown scheme {scheme!r}; available: {sorted(set(cv['scheme']))}")
    return cv[cv["scheme"] == scheme]


def iter_folds(cv: pd.DataFrame, scheme: str) -> Iterator[tuple[int, str, pd.Index, pd.Index]]:
    """Yield ``(fold, held_out, train_ids, test_ids)`` as pandas Index objects."""
    sub = scheme_folds(cv, scheme)
    for fold, g in sub.groupby("fold", sort=True):
        tr = pd.Index(g.loc[g["split"] == "train", ID_COL])
        te = pd.Index(g.loc[g["split"] == "test", ID_COL])
        yield int(fold), str(g["held_out"].iloc[0]), tr, te


def ensure_group_col(plots: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Materialise a grouping column that is derived rather than stored.

    `block20`, `window_component` and `owner_window` are computed from coordinates, not
    read from the plots table, and `scripts/08_build_modelling_folds.py` only keeps them
    long enough to write the fold assignments. Any consumer that needs the *same* grouping
    for its inner split has to recompute it, and doing that in one place is the difference
    between a nested split that respects the outer grouping and one that silently does not.

    Returns ``plots`` unchanged if the column is already there, a copy with the column
    added otherwise.
    """
    if group_col in plots.columns:
        return plots
    out = plots.copy()
    if group_col.startswith("block"):
        out[group_col] = add_block_key(out, float(group_col.replace("block", "")))
    elif group_col == "window_component":
        out[group_col] = window_components(out).to_numpy()
    elif group_col == "owner_window":
        w = window_components(out).to_numpy()
        out[group_col] = [f"{o}|{x}" for o, x in zip(out["Owner"], w)]
    else:
        raise KeyError(
            f"{group_col!r} is neither a column of the plots table nor a derived grouping "
            "this function knows how to build")
    return out


def inner_split(train_ids: pd.Index, plots: pd.DataFrame, group_col: str,
                seed: int, val_frac: float = 0.2) -> tuple[pd.Index, pd.Index]:
    """Carve a grouped validation set out of the training fold, for early stopping.

    Grouped by ``group_col`` — the outer scheme's own grouping column — so the stopping
    epoch is not chosen against plots that are neighbours or same-protocol siblings of the
    training data. Returns ``(fit_ids, val_ids)``; scalers are fitted on ``fit_ids`` only.
    """
    sub = plots[plots[ID_COL].isin(train_ids)]
    k = max(2, int(round(1.0 / val_frac)))
    stratify = "richness" if "richness" in sub.columns else None
    cv = cv_groups.grouped_kfold(sub, group_col, k=k, seed=seed, stratify_on=stratify)
    val = pd.Index(cv.loc[(cv["fold"] == seed % k) & (cv["split"] == "test"), ID_COL])
    fit = pd.Index(train_ids).difference(val)
    if len(val) == 0 or len(fit) == 0:
        raise RuntimeError(f"degenerate inner split for group_col={group_col!r}, seed={seed}")
    return fit, val


def assert_no_leak(train_ids, test_ids, context: str = "") -> None:
    overlap = pd.Index(train_ids).intersection(pd.Index(test_ids))
    if len(overlap):
        raise AssertionError(
            f"{len(overlap)} plots in both train and test {context}: {list(overlap[:5])}"
        )


def pixel_rows(pixels: pd.DataFrame, plot_ids, id_col: str = "plot_id") -> pd.DataFrame:
    """Select the 25-per-plot rows belonging to a set of plots. Never the other way round."""
    return pixels[pixels[id_col].isin(pd.Index(plot_ids))]


def collect_oof(per_fold: list[pd.DataFrame], scheme: str,
                names: list[str]) -> pd.DataFrame:
    """Concatenate per-fold prediction frames into one out-of-fold table.

    For a partitioning scheme each plot appears exactly once per seed; that invariant is
    asserted, because a violated one means the fold table is malformed and every downstream
    score is wrong. LODO schemes are exempt and flagged.
    """
    oof = pd.concat(per_fold, ignore_index=True)
    if scheme not in NON_PARTITIONING:
        dup = oof.groupby([ID_COL, "seed"]).size()
        if (dup > 1).any():
            raise AssertionError(
                f"scheme {scheme!r} is not partitioning: {(dup > 1).sum()} plot/seed "
                "combinations predicted more than once"
            )
    oof.attrs["scheme"] = scheme
    oof.attrs["partitioning"] = scheme not in NON_PARTITIONING
    oof.attrs["targets"] = names
    return oof


def add_block_key(plots: pd.DataFrame, km: float) -> pd.Series:
    """Geometric block key from UTM coordinates, for a spatially blocked scheme.

    ``Location`` is a nominal site *label*, not geometry, and the 23 datasets overlap each
    other spatially (103 overlapping bounding-box pairs measured on this subset). Neither
    controls spatial autocorrelation, which is what risk R8 is about — hence a real grid.
    """
    size = km * 1000.0
    return (np.floor(plots["X"] / size).astype(int).astype(str) + "_"
            + np.floor(plots["Y"] / size).astype(int).astype(str))


def window_components(plots: pd.DataFrame, half_m: float = 150.0) -> pd.Series:
    """Connected components of the "plots share an extraction window" graph.

    Every predictor in this project is read from a 5x5 Landsat window centred on the plot,
    so two plots whose windows overlap are described by *partly the same pixels*. Measured
    on this subset: 767 overlapping pairs, 575 plots in 135 components, the largest holding
    17 plots (a recensus laid over a grid). Splitting those across a fold boundary is
    pseudoreplication — the model is scored on pixels it was trained on.

    Measured on the realised partitions (test side, ``cv_folds_modelling.parquet``), the
    leak is much smaller than the component count suggests, and grouping by component is
    not the only way to close it:

    ============================  ==================  ===================
    scheme                        components split    plots on both sides
    ============================  ==================  ===================
    ``kfold5_random``             126 of 135          557 (51.5%)
    ``kfold5_owner``              7 of 135            47 (4.3%)
    ``kfold5_block20``            0                   0
    ``kfold5_window``             0                   0
    ``kfold5_owner_window``       0                   0
    ============================  ==================  ===================

    ``kfold5_owner`` closes 128 of the 135 components *incidentally*, because contributors
    are spatially clustered — grouping by person groups by ground almost for free. The 7
    that survive are shared ground under two owner labels; five of them are the same
    Altamirano/Miranda recensus pair. Dropping the minority side of each closes the leak
    at a cost of 12 plots (1.1%).

    ``kfold5_block20`` also reaches zero, but incidentally rather than by construction: a
    20 km grid line does not care where a 150 m window falls, and a different grid offset
    could cut one. Only grouping by component is zero **by construction**.

    The windows are axis-aligned squares of side ``2 * half_m`` in UTM, so overlap is the
    Chebyshev condition ``|dX| < 2*half_m and |dY| < 2*half_m``... except that the window is
    ``half_m`` on each side of the plot, which makes the overlap threshold ``half_m`` in each
    axis. Union-find over the resulting edges, sorted-sweep on X so this stays O(n log n)
    rather than O(n^2).
    """
    xs = plots["X"].to_numpy(float)
    ys = plots["Y"].to_numpy(float)
    n = len(plots)

    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    order = np.argsort(xs, kind="stable")
    for pi in range(n):
        i = order[pi]
        for pj in range(pi + 1, n):
            j = order[pj]
            if xs[j] - xs[i] >= half_m:
                break
            if abs(ys[j] - ys[i]) < half_m:
                union(i, j)

    roots = np.array([find(i) for i in range(n)])
    return pd.Series([f"w{r}" for r in roots], index=plots.index, name="window_component")


def random_kfold(plots: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Ungrouped K-fold — the deliberately optimistic reference.

    Reported so the gap against the grouped schemes can be quoted. That gap is a result in
    itself (risk R8), which is the only reason a random split appears in this project at all.
    """
    rng = np.random.default_rng(seed)
    fold_of_plot = pd.Series(rng.permutation(len(plots)) % k, index=plots.index)
    rows = []
    for fold in range(k):
        rows.append(pd.DataFrame({
            "fold": fold,
            "held_out": f"fold{fold}",
            ID_COL: plots[ID_COL],
            "split": np.where(fold_of_plot == fold, "test", "train"),
        }))
    return pd.concat(rows, ignore_index=True)
