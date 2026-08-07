#!/usr/bin/env python3
"""Recompute LSP metrics from the stored PhenoShape curves under a chosen phase anchoring.

No datacube access. `phenoshape` and the raw observations are anchoring-independent, so
changing how seasons are anchored costs one pass over local files rather than a re-download.
That separation is the reason acquisition stores curves rather than metrics.

Why anchoring matters here. Central Chile carries vegetation with opposite seasonal phase
in the same scene: sclerophyll forest greening in austral spring, and matorral greening on
the winter rains and senescing through the summer drought. A single global rotation
("south") serves the majority phase and mis-anchors the winter-peaking minority
*systematically* — and since that minority is an ecologically distinct community type, the
resulting bias would correlate with the target being predicted. Per-pixel anchoring
("auto") adapts to each pixel instead.

The known risk of "auto" is that on a weak seasonal signal the harmonic fit chases noise
and assigns different anchors to neighbouring pixels of the same plot, which would break
the 5x5 augmentation. This script measures that directly: `anchor_circ_sd` is the circular
standard deviation of the per-pixel anchor within a plot, and `aseasonal_frac` the share of
pixels below the seasonality-strength threshold. Both are written out so plots with
incoherent anchoring can be flagged rather than silently trusted.

Usage:
    python scripts/04_recompute_lsp.py --hemisphere auto --compare
    python scripts/04_recompute_lsp.py --hemisphere auto --write-back
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/home/jovyan/PhenoSensing")
import phenosensing  # noqa: E402,F401  -- registers the `.pheno` accessor
from phenosensing.phase import season_phase  # noqa: E402

LSP_METRICS = [
    "sos", "pos", "eos", "vsos", "vpos", "veos", "los", "msp", "mau",
    "vmsp", "vmau", "ampl", "ios", "rog", "ros", "sw", "trough", "mos",
]

# PhenoLSP(hemisphere="auto") attaches these as *coordinates* on its output. If they are
# carried back onto `phenoshape` and the file is reprocessed, `season_phase` tries to create
# variables with the same names and raises. Strip them so every pass is idempotent.
PHASE_COORDS = ("phase_offset", "aseasonal", "multi_season")

# Where to cut the year when rolling curves for the 2D transform.
#
# NOT the textbook austral anchor (DOY 183, ~1 July). Measured over all 1,082 plots, central
# Chile does not follow the austral-summer growing season that `hemisphere="south"` assumes:
# it greens on the winter rains, peaks in late winter to spring (median peak DOY 243, 68% of
# plots peaking Jul-Oct) and troughs in autumn (median trough DOY 108). Rotating to DOY 183
# therefore moves the array boundary INTO the growing season:
#
#   rotation                    peak lands at   plots with peak <6 steps from an edge
#   none (calendar)             34/52                     16%
#   austral DOY 183             11/52                     23%   <- worse than doing nothing
#   regional trough DOY 108     19/52                      5%
#
# The anchor is the measured regional trough, and `scripts/04_recompute_lsp.py --anchor`
# lets it be re-derived if the study area changes.
TROUGH_ANCHOR = 108
AUSTRAL_ANCHOR = TROUGH_ANCHOR  # backwards-compatible alias


def strip_phase_coords(obj):
    return obj.drop_vars([c for c in PHASE_COORDS if c in obj.coords], errors="ignore")


def to_calendar_order(curve: xr.DataArray) -> xr.DataArray:
    """Restore ascending calendar DOY before computing LSP.

    ``PhenoLSP`` locates SOS/POS/EOS by array position and maps them back to days assuming
    the axis runs monotonically through the year. On a curve rolled for the 2D transform
    that assumption is false and every DOY-valued metric comes out shifted — measured at a
    median of 106 days, with none of 40 test plots matching the pre-rotation values.

    Sorting by the ``doy`` coordinate undoes any roll, whatever anchor produced it, and is a
    no-op on curves that were never rolled. Always call this before ``PhenoLSP``.
    """
    return curve.sortby("doy")


def austral_shift(doy, anchor: int = TROUGH_ANCHOR) -> int:
    """Roll amount that puts the step nearest ``anchor`` at array position 0."""
    return -int(np.argmin(np.abs(np.asarray(doy) - anchor)))


def rotate_to_trough(obj, anchor: int = TROUGH_ANCHOR):
    """Roll a PhenoShape curve (or a whole Dataset) so the array starts at the regional
    seasonal trough (DOY 108, mid-April), placing the growing season mid-array.

    Must be applied to the **Dataset**, never assigned back as a DataArray. Assigning a
    rolled DataArray into a Dataset makes xarray realign it against the existing ``doy``
    index and silently undo the roll, leaving a file stamped as rotated that is not. Rolling
    the Dataset touches only variables carrying a ``doy`` dim — `phenoshape` — and leaves
    `lsp_*` (index, y, x) and `obs_*` (time, y, x) alone.

    This reorders the array only. The ``doy`` coordinate is rolled with the data, so every
    step still carries its true calendar day: no information is lost and nothing downstream
    has to guess which convention a file uses.

    Purpose is the 2D transform, not the metrics. The DOY axis is circular but a 52-step
    array has two free ends, and whatever sits at those ends is cut in half: in an 8x8
    reshape it lands in the last and first rows, adjacent in time but maximally distant in
    the image, where no 3x3 kernel can bridge it. Putting the cut at the trough moves that
    damage to the least informative part of the year (16% -> 5% of plots with their peak
    near an edge).

    The shift is a single constant for every plot, deliberately. A global roll is a
    relabelling of the axis and preserves the phase *differences between* plots, which are
    signal: winter-peaking matorral versus summer-peaking sclerophyll is a real community
    distinction. Per-plot centring would delete exactly that.

    Note for anyone computing LSP afterwards: on a curve already rolled here, use
    ``hemisphere="north"`` — asking for ``"south"`` would rotate a second time.
    """
    return obj.roll(doy=austral_shift(obj["doy"].values, anchor), roll_coords=True)


def circular_sd(doy: np.ndarray, period: float = 365.0) -> float:
    """Circular sd in days. A plain sd is meaningless on a wrapped axis: anchors of 5 and
    360 are 10 days apart, not 355."""
    d = np.asarray(doy, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan
    a = 2 * np.pi * d / period
    R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    R = min(max(R, 1e-12), 1.0)
    return float(np.sqrt(-2 * np.log(R)) * period / (2 * np.pi))


def phase_diagnostics(curve: xr.DataArray) -> dict:
    """Per-plot coherence of the per-pixel anchoring."""
    try:
        sp = season_phase(to_calendar_order(strip_phase_coords(curve)))
    except Exception as e:
        return dict(anchor_circ_sd=np.nan, aseasonal_frac=np.nan,
                    strength_median=np.nan, phase_error=str(e))
    anchor = np.asarray(sp["anchor"].values).ravel()
    strength = np.asarray(sp["strength"].values).ravel()
    aseason = np.asarray(sp["aseasonal"].values).ravel()
    return dict(
        anchor_circ_sd=circular_sd(anchor),
        anchor_median=float(np.nanmedian(anchor)),
        aseasonal_frac=float(np.nanmean(aseason)),
        strength_median=float(np.nanmedian(strength)),
        phase_error="",
    )


def observed_peak(curve: xr.DataArray) -> float:
    """Calendar DOY of the observed peak at the centre pixel.

    Read straight off the array, so it is independent of anchoring and of whether the file
    has already been rolled (the doy coordinate rolls with the data). This is the one phase
    descriptor that is always comparable, and it is what makes per-group anchoring possible.
    """
    cy, cx = curve.sizes["y"] // 2, curve.sizes["x"] // 2
    v = curve.isel(y=cy, x=cx).values
    if not np.isfinite(v).any():
        return np.nan
    return float(np.asarray(curve["doy"].values)[np.nanargmax(v)])


def phase_group_of(peak_doy: float) -> str:
    """Austral winter-peaking (matorral greening on the winter rains) vs summer-peaking."""
    if not np.isfinite(peak_doy):
        return "unknown"
    return "winter_peaking" if 120 <= peak_doy <= 260 else "summer_peaking"


# Measured over 153 extracted plots (NDVI, centre pixel), median metrics per group:
#
#   group            anchoring   SOS   POS   EOS    verdict
#   summer_peaking   south       264   307    89    coherent, season wraps the year end
#   summer_peaking   north       214   307   322    EOS truncated at the array boundary
#   winter_peaking   south       200   163   162    POS precedes SOS -- incoherent
#   winter_peaking   north        79   158   247    coherent, no wrap
#
# The two anchorings are complementary, and `degenerate_px_frac` is 0 in every cell above,
# so the usual sos==pos check does NOT catch the failure. Hence "by_phase": pick the
# anchoring per plot from its observed peak.
BY_PHASE = {"summer_peaking": "south", "winter_peaking": "north", "unknown": "south"}


def lsp_table(files, hemisphere: str, index: str, ref_index: str = "ndvi",
              per_index_phase: bool = False) -> tuple[pd.DataFrame, dict]:
    """Long-format LSP table. ``index="all"`` covers every vegetation index in the files.

    The phase group is decided **once per plot** from ``ref_index`` and then applied to all
    of that plot's indices. Deciding it per index would let the same plot be anchored one
    way in NDVI and another in EVI, so its metrics would no longer be comparable across
    indices — which is exactly what a multi-index benchmark needs them to be.
    """
    rows, deg, tot = [], 0, 0
    for f in files:
        try:
            with xr.open_dataset(f) as d:
                available = [str(v) for v in np.atleast_1d(d["index"].values)]
                wanted = available if index == "all" else (
                    [index] if index in available else [])
                if not wanted:
                    continue

                ref = ref_index if ref_index in available else wanted[0]
                pk_ref = observed_peak(to_calendar_order(
                    strip_phase_coords(d["phenoshape"].sel(index=ref))))

                for ix in wanted:
                    c = to_calendar_order(strip_phase_coords(d["phenoshape"].sel(index=ix)))
                    pk_own = observed_peak(c)
                    # Default: one frame per plot, taken from ref_index, so metrics stay
                    # comparable across indices. Opt out with per_index_phase when each
                    # index should sit in its own frame -- NBR in particular tracks moisture
                    # rather than greenness and peaks at a different time of year, so the
                    # NDVI-derived frame does not fit it.
                    pk = pk_own if per_index_phase else pk_ref
                    grp = phase_group_of(pk)
                    hemi = BY_PHASE[grp] if hemisphere == "by_phase" else hemisphere
                    lsp = c.pheno.PhenoLSP(nGS=c.sizes["doy"], hemisphere=hemi)
                    cy, cx = c.sizes["y"] // 2, c.sizes["x"] // 2
                    r = {"plot_id": d.attrs["plot_id"], "index": ix}
                    for m in LSP_METRICS:
                        if m in lsp:
                            a = lsp[m].values
                            r[m] = float(a[cy, cx])
                            r[f"{m}_mean5x5"] = float(np.nanmean(a))
                    sv, pv = lsp["sos"].values, lsp["pos"].values
                    deg += int(np.nansum(sv == pv)); tot += sv.size
                    r["degenerate_px_frac"] = float(np.nanmean(sv == pv))

                # Observed peak of the curve, read straight off the array. Independent of
                # anchoring and of whether the file has been rolled (the doy coordinate
                # rolls with the data), so it is the one phase descriptor that is always
                # comparable. This is what lets a global "south" anchoring be applied while
                # still identifying the plots it mis-anchors, instead of hiding them.
                    r["peak_doy_observed"] = pk_own
                    r["peak_doy_ref"] = pk_ref
                    r["phase_group"] = grp
                    # Coherent seasons run SOS -> POS -> EOS, allowing one wrap of the year.
                    # Surfaced as a column because degenerate_px_frac does NOT catch this:
                    # a mis-anchored season can have sos != pos and still be nonsense.
                    def _fwd(a, b):
                        return (b - a) % 365.0
                    r["order_coherent"] = bool(
                        np.isfinite(r.get("sos", np.nan)) and np.isfinite(r.get("eos", np.nan))
                        and _fwd(r["sos"], r["pos"]) + _fwd(r["pos"], r["eos"]) <= 365.0
                        and _fwd(r["sos"], r["pos"]) > 0 and _fwd(r["pos"], r["eos"]) > 0
                    )
                    r["anchoring_used"] = hemi
                    r["ref_index"] = ref
                    if hemi == "auto":
                        r.update(phase_diagnostics(c))
                    rows.append(r)
        except Exception:
            continue
    return pd.DataFrame(rows), dict(degenerate_pct=100 * deg / max(tot, 1), n_plots=len(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pheno-dir", default="data/derived/phenology")
    p.add_argument("--out-dir", default="data/derived")
    # Measured over ALL 1,082 plots (NDVI, centre pixel):
    #
    #   anchoring   degenerate (sos==pos)   median LOS   IQR LOS
    #   north               3.0%                156         29
    #   south               2.7%                149         43
    #   auto                1.8%                164         29
    #
    # "auto" wins outright. It also works as intended here, contrary to what a smaller,
    # geographically biased subset suggested: median seasonality strength over the full set
    # is 0.170 and 55% of plots clear the 0.15 threshold, so most pixels do get anchored
    # rather than silently falling back to no rotation.
    #
    # "by_phase" is kept for inspection but is NOT the default: its 120-260 threshold cuts
    # straight through the dominant Sep-Oct peak, splitting one ecological group in two
    # rather than separating two. Peak timing over the full set is unimodal
    # (circular R = 0.56), so a hand-made two-population scheme is not warranted.
    p.add_argument("--hemisphere", default="auto",
                   choices=["auto", "north", "south", "by_phase"],
                   help="auto (default, per-pixel harmonic): best on the full dataset")
    p.add_argument("--index", default="all",
                   help="vegetation index, or 'all' for every index in the files")
    p.add_argument("--ref-index", default="ndvi",
                   help="index whose observed peak decides the plot's phase group")
    p.add_argument("--per-index-phase", action="store_true",
                   help="give each vegetation index its own phase frame instead of sharing "
                        "the ref-index frame; trades cross-index comparability for "
                        "per-index coherence (relevant for NBR, which tracks moisture)")
    p.add_argument("--compare", action="store_true",
                   help="also score the other two anchorings, for the record")
    p.add_argument("--write-back", action="store_true",
                   help="rewrite lsp_* inside each .nc under this anchoring and stamp "
                        "attrs['phase_anchoring']; leaves phenoshape and obs_* untouched")
    p.add_argument("--anchor", type=int, default=TROUGH_ANCHOR,
                   help="DOY at which to cut the year when rolling (default: the measured "
                        "regional trough, 108)")
    p.add_argument("--rotate-austral", dest="rotate_austral", action="store_true",
                   help="roll phenoshape so the array starts at --anchor "
                        "(the regional trough); the doy coordinate rolls with it, so "
                        "calendar days "
                        "stay attached. LSP is computed BEFORE the roll, so metrics remain "
                        "in calendar DOY and no double rotation occurs")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.pheno_dir).glob("*.nc"))
    if args.limit:
        files = files[: args.limit]
    print(f"plots on disk: {len(files)}  |  index: {args.index}")

    if args.compare:
        print("\n=== anchoring comparison ===")
        print(f"{'anchoring':10s} {'degenerate%':>12s} {'med SOS':>8s} {'med LOS':>8s} {'IQR LOS':>8s}")
        for h in ["north", "south", "auto"]:
            t, s = lsp_table(files, h, args.index, args.ref_index,
                             args.per_index_phase)
            if t.empty:
                continue
            print(f"{h:10s} {s['degenerate_pct']:11.1f}% {t['sos'].median():8.0f} "
                  f"{t['los'].median():8.0f} "
                  f"{t['los'].quantile(.75) - t['los'].quantile(.25):8.0f}")

    if args.write_back:
        n_ok = n_err = 0
        for f in files:
            tmp = f.with_suffix(".nc.tmp")
            try:
                with xr.open_dataset(f) as d:
                    ds = d.load()          # detach from the file before overwriting it
                if str(ds.attrs.get("phase_anchoring", "")) == args.hemisphere:
                    n_ok += 1
                    continue
                idx_present = list(np.atleast_1d(ds["index"].values))
                new = {}
                for m in LSP_METRICS:
                    new[f"lsp_{m}"] = []
                keep_idx = []
                for ix in idx_present:
                    c = to_calendar_order(strip_phase_coords(ds["phenoshape"].sel(index=ix)))
                    lsp = c.pheno.PhenoLSP(nGS=c.sizes["doy"], hemisphere=args.hemisphere)
                    keep_idx.append(ix)
                    for m in LSP_METRICS:
                        new[f"lsp_{m}"].append(lsp[m] if m in lsp else None)
                for m in LSP_METRICS:
                    vals = [v for v in new[f"lsp_{m}"] if v is not None]
                    if len(vals) == len(keep_idx) and vals:
                        ds[f"lsp_{m}"] = xr.concat(vals, dim="index").assign_coords(
                            index=keep_idx)
                ds.attrs["phase_anchoring"] = args.hemisphere
                ds = strip_phase_coords(ds)
                if args.rotate_austral and ds.attrs.get("doy_order") != "trough_anchored":
                    ds = rotate_to_trough(ds, args.anchor)
                    ds.attrs["doy_order"] = "trough_anchored"
                    ds.attrs["doy_anchor"] = args.anchor
                elif "doy_order" not in ds.attrs:
                    ds.attrs["doy_order"] = "calendar"
                for c in list(ds.coords):
                    ds[c].attrs.pop("units", None)
                    ds[c].encoding.pop("units", None)
                ds.to_netcdf(tmp)          # write then rename, so an interruption cannot
                tmp.replace(f)             # leave a half-written file behind
                n_ok += 1
            except Exception as e:
                n_err += 1
                tmp.unlink(missing_ok=True)
                print(f"  write-back failed for {f.name}: {type(e).__name__}: {e}")
        print(f"\nwrite-back under '{args.hemisphere}': {n_ok} files updated, {n_err} failed")

    tab, stats = lsp_table(files, args.hemisphere, args.index, args.ref_index,
                           args.per_index_phase)
    out = Path(args.out_dir) / f"lsp_{args.index}_{args.hemisphere}.parquet"
    tab.to_parquet(out, index=False)
    npl = tab["plot_id"].nunique() if len(tab) else 0
    print(f"\nwritten: {out}  ({len(tab)} rows, {npl} plots, "
          f"degenerate pixels {stats['degenerate_pct']:.1f}%)")
    if "order_coherent" in tab and len(tab):
        print("\nseason order coherence (SOS -> POS -> EOS, one wrap allowed):")
        for (ix,), grp in tab.groupby(["index"]):
            print(f"  {ix:6s} coherent in {100 * grp['order_coherent'].mean():5.1f}% of plots")

    if args.hemisphere == "auto" and "anchor_circ_sd" in tab:
        cs = tab["anchor_circ_sd"].dropna()
        print("\n=== within-plot anchoring coherence (the risk of per-pixel anchoring) ===")
        print(f"circular sd of the anchor across the 25 pixels, in days:")
        print(f"  median={cs.median():.0f}  q75={cs.quantile(.75):.0f}  q90={cs.quantile(.90):.0f}")
        for thr in (15, 30, 60):
            print(f"  plots with circ sd > {thr:3d} d: {int((cs > thr).sum()):4d} "
                  f"({100 * (cs > thr).mean():.0f}%)")
        print(f"  median aseasonal pixel fraction: {tab['aseasonal_frac'].median():.2f}")
        print(f"  median seasonality strength:     {tab['strength_median'].median():.3f}")
        print("\nPlots with a large circular sd have pixels anchored to different"
              "\nphenological years; their 5x5 window is not internally comparable and"
              "\nshould be flagged before being used for augmentation.")


if __name__ == "__main__":
    main()


rotate_to_austral = rotate_to_trough  # backwards-compatible alias
