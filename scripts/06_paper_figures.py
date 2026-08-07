#!/usr/bin/env python3
"""Render the paper and supplementary figure set.

Format follows the policy in `biodiv.figures`: PDF for vector content, PNG at 300 dpi for
figures carrying real pixel imagery. Every figure is written twice-nothing: one file, one
format, chosen by content.

Point maps use the Chile region polygons in `shapefiles/regiones_chile.shp` as a basemap,
with a locator inset and a scale bar.

Usage:
    python scripts/06_paper_figures.py --out-dir results/figures
    python scripts/06_paper_figures.py --only map_study_area
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import figures as F  # noqa: E402

DERIVED = ROOT / "data" / "derived"
PHENO = DERIVED / "phenology"


# --------------------------------------------------------------------------- data
def load_all() -> dict:
    plots = pd.read_parquet(DERIVED / "plots_subset.parquet")
    d = dict(plots=plots)

    mp = PHENO / "manifest.csv"
    if mp.exists():
        man = pd.read_csv(mp)
        dup = [c for c in ("metadata_id", "Location", "lat", "lon") if c in man.columns]
        d["manifest"] = man
        d["db"] = plots.merge(man.drop(columns=dup), left_on="PlotObservationID",
                              right_on="plot_id", how="inner")
    tp = DERIVED / "topography" / "topography.parquet"
    if tp.exists():
        d["topo"] = pd.read_parquet(tp)
        if "db" in d:
            d["db"] = d["db"].merge(d["topo"], on="plot_id", how="left")
    lp = DERIVED / "lsp_all_by_phase.parquet"
    if lp.exists():
        d["lsp"] = pd.read_parquet(lp)
    cp = DERIVED / "cv_fold_report.csv"
    if cp.exists():
        d["cv"] = pd.read_csv(cp)
    fp = DERIVED / "filter_cascade.csv"
    if fp.exists():
        d["cascade"] = pd.read_csv(fp)
    return d


def curve_matrix(plot_ids, index="ndvi"):
    """{plot_id: 52-step centre-pixel curve} for one index."""
    out, doy = {}, None
    for pid in plot_ids:
        f = PHENO / f"{pid}.nc"
        if not f.exists():
            continue
        try:
            with xr.open_dataset(f) as d:
                if index not in [str(v) for v in np.atleast_1d(d["index"].values)]:
                    continue
                c = d["phenoshape"].sel(index=index)
                out[pid] = c.isel(y=c.sizes["y"] // 2, x=c.sizes["x"] // 2).values
                doy = np.asarray(c["doy"].values)
        except Exception:
            continue
    return pd.DataFrame(out).T, doy


# --------------------------------------------------------------------------- figures
def fig_map_study_area(D, out):
    """Fig 1. Plot locations over the Chile basemap, panelled by census year and richness."""
    db = D.get("db", D["plots"].assign(plot_id=D["plots"]["PlotObservationID"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.6))

    F.basemap(axes[0])
    for i, (g, sub) in enumerate(db.groupby("metadata_id")):
        axes[0].scatter(sub["lon"], sub["lat"], s=7, lw=0,
                        color=F.PALETTE[i % len(F.PALETTE)], alpha=.85)
    axes[0].set_title(f"Source project (n = {db['metadata_id'].nunique()})")
    F.add_inset_locator(fig, axes[0])
    F.scalebar(axes[0])

    F.basemap(axes[1], label_axes=False)
    s = axes[1].scatter(db["lon"], db["lat"], c=db["Year"], s=7, lw=0, cmap="viridis")
    fig.colorbar(s, ax=axes[1], shrink=.55, label="Census year")
    axes[1].set_title("Census year")
    axes[1].set_xlabel("Longitude (°)")

    F.basemap(axes[2], label_axes=False)
    s = axes[2].scatter(db["lon"], db["lat"], c=np.log10(db["richness"]),
                        s=7, lw=0, cmap="magma_r")
    cb = fig.colorbar(s, ax=axes[2], shrink=.55, label="Species richness")
    cb.set_ticks(np.log10([1, 2, 5, 10, 25, 50]))
    cb.set_ticklabels([1, 2, 5, 10, 25, 50])
    axes[2].set_title("Species richness")
    axes[2].set_xlabel("Longitude (°)")

    fig.tight_layout()
    return F.save_figure(fig, "fig01_study_area", out)


def fig_sampling(D, out):
    """Fig 2. Filter cascade, census-year distribution, and richness."""
    plots = D["plots"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

    if "cascade" in D:
        c = D["cascade"]
        ax[0].barh(range(len(c)), c["n_plots"], color=F.PALETTE[0], height=.62)
        ax[0].set_yticks(range(len(c)), [s.split(". ", 1)[-1] for s in c["step"]], fontsize=7)
        ax[0].invert_yaxis()
        for i, v in enumerate(c["n_plots"]):
            ax[0].text(v + 15, i, f"{v:,}", va="center", fontsize=7)
        ax[0].set_xlabel("Plots retained")
        ax[0].set_xlim(0, c["n_plots"].max() * 1.18)
    ax[0].set_title("a  Filter cascade", loc="left", fontweight="bold")

    yr = plots["Year"].value_counts().sort_index()
    ax[1].bar(yr.index, yr.values, color=F.PALETTE[0], width=.75)
    ax[1].set(xlabel="Census year", ylabel="Plots")
    ax[1].set_title("b  Temporal distribution", loc="left", fontweight="bold")

    ax[2].hist(plots["richness"], bins=np.arange(0, plots["richness"].max() + 2),
               color=F.PALETTE[2])
    ax[2].axvline(plots["richness"].median(), color=F.PALETTE[1], ls="--", lw=1,
                  label=f"median = {plots['richness'].median():.0f}")
    ax[2].set(xlabel="Species richness", ylabel="Plots")
    ax[2].legend()
    ax[2].set_title("c  Woody species richness", loc="left", fontweight="bold")

    fig.tight_layout()
    return F.save_figure(fig, "fig02_sampling", out)


def fig_predictor_quality(D, out):
    """Fig 3. Observation density and DOY-gap structure of the 3-year window."""
    if "db" not in D:
        return None
    db = D["db"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

    ax[0].hist(db["n_obs"], bins=40, color=F.PALETTE[0])
    ax[0].set(xlabel="Clear observations per 3-year window", ylabel="Plots")
    ax[0].set_title("a", loc="left", fontweight="bold")

    ax[1].hist(db["max_doy_gap"].dropna(), bins=40, color=F.PALETTE[4])
    ax[1].axvline(45, color=F.PALETTE[1], ls="--", lw=1, label="45-day threshold")
    ax[1].set(xlabel="Largest DOY gap (days)", ylabel="Plots")
    ax[1].legend()
    ax[1].set_title("b", loc="left", fontweight="bold")

    s = ax[2].scatter(db["n_obs"], db["max_doy_gap"], c=db["Year"], s=8, lw=0, cmap="viridis")
    fig.colorbar(s, ax=ax[2], label="Census year")
    ax[2].set(xlabel="Observations", ylabel="Largest DOY gap (days)")
    ax[2].set_title("c", loc="left", fontweight="bold")

    fig.tight_layout()
    return F.save_figure(fig, "fig03_predictor_quality", out)


def fig_curves_by_index(D, out):
    """Fig 4. Mean phenological curve per index, and between-index correlation."""
    ids = (D["db"]["plot_id"] if "db" in D else D["plots"]["PlotObservationID"]).tolist()
    mats, doy = {}, None
    for ix in F.INDEX_LABELS:
        m, dd = curve_matrix(ids, ix)
        if len(m):
            mats[ix], doy = m, dd
    if not mats:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ix, m in mats.items():
        med = m.median()
        q1, q3 = m.quantile(.25), m.quantile(.75)
        ax[0].fill_between(doy, q1, q3, color=F.INDEX_COLOURS[ix], alpha=.15, lw=0)
        ax[0].plot(doy, med, color=F.INDEX_COLOURS[ix], label=F.INDEX_LABELS[ix])
    ax[0].set(xlabel="Day of year", ylabel="Index value")
    ax[0].legend(ncol=2)
    ax[0].set_title("a  Median curve (IQR shaded)", loc="left", fontweight="bold")

    common = sorted(set.intersection(*(set(m.index) for m in mats.values())))
    flat = pd.DataFrame({ix: m.loc[common].to_numpy().ravel() for ix, m in mats.items()})
    corr = flat.corr()
    lab = [F.INDEX_LABELS[c] for c in corr.columns]
    mesh = F.heatmap(ax[1], corr.values, lab, lab, vmin=-1, vmax=1, fontsize=7)
    fig.colorbar(mesh, ax=ax[1], label="Pearson r")
    ax[1].set_title(f"b  Between-index correlation (n = {len(common)} plots)",
                    loc="left", fontweight="bold")

    fig.tight_layout()
    return F.save_figure(fig, "fig04_curves_by_index", out)


def fig_curves_by_project(D, out):
    """Fig S3. Mean curve per source project — the protocol/site effect made visible.

    Worth its own figure: median richness runs from 1 (md006) to 27 (md023) across projects,
    so anything that separates projects is a candidate confounder rather than ecology, and
    it is the reason cross-validation holds out whole datasets.
    """
    if "db" not in D:
        return None
    db = D["db"]
    m, doy = curve_matrix(db["plot_id"].tolist(), "ndvi")
    if not len(m):
        return None
    meta = db.set_index("plot_id")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    joined = m.join(meta["metadata_id"])
    big = [g for g, sub in joined.groupby("metadata_id") if len(sub) >= 20]
    for i, g in enumerate(sorted(big)):
        sub = joined[joined["metadata_id"] == g].drop(columns="metadata_id")
        ax[0].plot(doy, sub.median(), color=F.PALETTE[i % len(F.PALETTE)],
                   label=f"{g} (n={len(sub)})")
    ax[0].set(xlabel="Day of year", ylabel="NDVI")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("a  Median NDVI curve by source project", loc="left", fontweight="bold")

    # Richness differs by an order of magnitude between projects: a random CV split would
    # let a model learn the project fingerprint instead of the ecology.
    order = meta.groupby("metadata_id")["richness"].median().sort_values()
    order = order[order.index.isin(big)]
    ax[1].barh(range(len(order)), order.values, color=F.PALETTE[2], height=.6)
    ax[1].set_yticks(range(len(order)), order.index, fontsize=7)
    ax[1].set_xlabel("Median species richness")
    ax[1].set_title("b  Richness by project", loc="left", fontweight="bold")

    fig.tight_layout()
    return F.save_figure(fig, "figS3_project_effect", out)


def fig_environment_maps(D, out):
    """Fig S4. Terrain and season length over the basemap."""
    if "db" not in D:
        return None
    db = D["db"]
    panels = [("elevation", "Elevation (m)", "terrain"),
              ("heat_load", "Heat load index", "inferno"),
              ("n_obs", "Clear observations", "viridis")]
    if "lsp" in D:
        los = (D["lsp"][D["lsp"]["index"] == "ndvi"][["plot_id", "los"]]
               .rename(columns={"los": "los_ndvi"}))
        db = db.merge(los, on="plot_id", how="left")
        panels.append(("los_ndvi", "Length of season, NDVI (d)", "cividis"))

    panels = [p for p in panels if p[0] in db]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 5.4))
    axes = np.atleast_1d(axes)
    for a, (col, lab, cmap) in zip(axes, panels):
        F.basemap(a, label_axes=(a is axes[0]))
        s = a.scatter(db["lon"], db["lat"], c=db[col], s=7, lw=0, cmap=cmap)
        fig.colorbar(s, ax=a, shrink=.5, label=lab)
        a.set_xlabel("Longitude (°)")
    fig.tight_layout()
    return F.save_figure(fig, "figS4_environment_maps", out)


def fig_phase(D, out):
    """Fig 5. Seasonal phase: peak timing, and where each phase group sits."""
    if "lsp" not in D:
        return None
    t = D["lsp"].drop_duplicates("plot_id")
    if "peak_doy_observed" not in t:
        return None
    db = D.get("db")
    fig = plt.figure(figsize=(10, 3.8))
    axp = fig.add_subplot(1, 3, 1, projection="polar")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Peak timing is a circular quantity; a linear histogram would split the season at 1 Jan
    ang = 2 * np.pi * t["peak_doy_observed"].dropna() / 365.0
    counts, edges = np.histogram(ang, bins=24, range=(0, 2 * np.pi))
    axp.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
            color=F.PALETTE[0], edgecolor="white", linewidth=.4)
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    axp.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=6)
    axp.set_yticklabels([])
    axp.set_title("a  Peak timing", loc="left", fontweight="bold", pad=14)

    grp = t["phase_group"].value_counts()
    ax2.bar(range(len(grp)), grp.values,
            color=[F.PALETTE[1] if "winter" in g else F.PALETTE[0] for g in grp.index])
    ax2.set_xticks(range(len(grp)), [g.replace("_", "\n") for g in grp.index], fontsize=7)
    ax2.set_ylabel("Plots")
    for i, v in enumerate(grp.values):
        ax2.text(i, v, f"{v}\n({100*v/grp.sum():.0f}%)", ha="center", va="bottom", fontsize=7)
    ax2.set_ylim(0, grp.max() * 1.25)
    ax2.set_title("b  Phase group", loc="left", fontweight="bold")

    if db is not None:
        m = db.merge(t[["plot_id", "phase_group"]], on="plot_id", how="inner")
        F.basemap(ax3, label_axes=False)
        for g, sub in m.groupby("phase_group"):
            ax3.scatter(sub["lon"], sub["lat"], s=7, lw=0, label=g.replace("_", " "),
                        color=F.PALETTE[1] if "winter" in g else F.PALETTE[0])
        ax3.legend(fontsize=6, loc="lower left")
        ax3.set_xlabel("Longitude (°)")
    ax3.set_title("c  Spatial pattern", loc="left", fontweight="bold")

    fig.tight_layout()
    return F.save_figure(fig, "fig05_seasonal_phase", out)


def fig_lsp_topography(D, out):
    """Fig 6. LSP metric distributions and their association with terrain, per index."""
    if "lsp" not in D or "db" not in D:
        return None
    lsp, db = D["lsp"], D["db"]
    METRICS = ["sos", "pos", "eos", "los", "ampl", "vpos"]
    TOPO = ["elevation", "slope", "northness", "heat_load", "tpi"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for a, m in zip(axes.ravel(), METRICS):
        if m not in lsp:
            continue
        for ix, g in lsp.groupby("index"):
            v = g[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v):
                a.hist(v, bins=30, histtype="step", lw=1.1,
                       color=F.INDEX_COLOURS.get(str(ix), "grey"),
                       label=F.INDEX_LABELS.get(str(ix), str(ix)))
        a.set(xlabel=m.upper(), ylabel="Plots")
    axes.ravel()[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p1 = F.save_figure(fig, "fig06_lsp_distributions", out)

    wide = lsp.pivot_table(index="plot_id", columns="index",
                           values=[m for m in METRICS if m in lsp])
    wide.columns = [f"{m}_{ix}" for m, ix in wide.columns]
    d = db.merge(wide, left_on="plot_id", right_index=True, how="inner")
    rows, ylab = [], []
    for ix in sorted(lsp["index"].unique()):
        for m in METRICS:
            col = f"{m}_{ix}"
            if col not in d:
                continue
            sub = d[TOPO + [col, "richness"]].replace([np.inf, -np.inf], np.nan)
            c = sub.corr(method="spearman")
            rows.append([c.loc[col, t] for t in TOPO] + [c.loc[col, "richness"]])
            ylab.append(f"{F.INDEX_LABELS.get(str(ix), ix)}  {m.upper()}")
    if rows:
        fig, ax = plt.subplots(figsize=(6.2, max(3.2, 0.20 * len(rows))))
        mesh = F.heatmap(ax, np.array(rows),
                         ["Elevation", "Slope", "Northness", "Heat load", "TPI", "Richness"],
                         ylab, vmin=-.6, vmax=.6, fontsize=5.5)
        fig.colorbar(mesh, ax=ax, label="Spearman ρ")
        ax.set_title("LSP metrics vs terrain and richness", loc="left", fontweight="bold")
        fig.tight_layout()
        F.save_figure(fig, "fig07_lsp_topography", out)
    return p1


def fig_example_plot(D, out):
    """Fig S1. One plot in detail — the only genuinely raster figure, hence PNG at 300 dpi."""
    ids = (D["db"]["plot_id"] if "db" in D else D["plots"]["PlotObservationID"]).tolist()
    pid = next((p for p in ids if (PHENO / f"{p}.nc").exists()), None)
    if pid is None:
        return None
    with xr.open_dataset(PHENO / f"{pid}.nc") as d:
        doy = np.asarray(d["doy"].values)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

        cur = d["phenoshape"].sel(index="ndvi").stack(px=("y", "x"))
        ax[0].plot(doy, cur.values, color="0.7", lw=.6)
        ax[0].plot(doy, cur.mean("px"), color=F.PALETTE[1], lw=1.6, label="5×5 mean")
        ax[0].set(xlabel="Day of year", ylabel="NDVI")
        ax[0].legend()
        ax[0].set_title("a  Within-plot pixel spread", loc="left", fontweight="bold")

        for ix in [str(v) for v in np.atleast_1d(d["index"].values)]:
            c = d["phenoshape"].sel(index=ix)
            ax[1].plot(doy, c.isel(y=c.sizes["y"] // 2, x=c.sizes["x"] // 2),
                       color=F.INDEX_COLOURS.get(ix, "grey"), label=F.INDEX_LABELS.get(ix, ix))
        ax[1].set(xlabel="Day of year", ylabel="Index value")
        ax[1].legend(ncol=2, fontsize=6)
        ax[1].set_title("b  All vegetation indices", loc="left", fontweight="bold")

        # Actual pixel imagery: the 52-step curve reshaped to 8x8, i.e. the CNN substrate
        v = d["phenoshape"].sel(index="ndvi")
        v = v.isel(y=v.sizes["y"] // 2, x=v.sizes["x"] // 2).values
        pad = np.full(64, np.nan); pad[:len(v)] = v
        im = ax[2].imshow(pad.reshape(8, 8), cmap="YlGn", interpolation="nearest")
        fig.colorbar(im, ax=ax[2], shrink=.8, label="NDVI")
        ax[2].set_xticks([]); ax[2].set_yticks([])
        ax[2].set_title("c  8×8 reshape (CNN substrate)", loc="left", fontweight="bold")

        fig.suptitle(f"Plot {pid}  ·  window {d.attrs.get('win_start')}–{d.attrs.get('win_end')}"
                     f"  ·  {d.attrs.get('n_obs')} clear observations", fontsize=8)
        fig.tight_layout()
    return F.save_figure(fig, "figS1_example_plot", out, raster=True)


def fig_cv_folds(D, out):
    """Fig S2. Cross-validation fold composition — publish this next to any score."""
    if "cv" not in D:
        return None
    cv = D["cv"]
    schemes = cv["scheme"].unique()
    fig, axes = plt.subplots(1, len(schemes), figsize=(3.6 * len(schemes), 3.2), squeeze=False)
    for a, s in zip(axes[0], schemes):
        sub = cv[cv["scheme"] == s]
        a.errorbar(range(len(sub)), sub["test_mean"], yerr=sub["test_sd"], fmt="o",
                   ms=3.5, lw=.9, capsize=2, color=F.PALETTE[0], label="Test")
        a.axhline(sub["train_mean"].mean(), color=F.PALETTE[1], ls="--", lw=.9,
                  label="Train mean")
        a.set(xlabel="Fold", ylabel="Richness")
        a.set_title(s, loc="left", fontsize=8, fontweight="bold")
        a.legend(fontsize=6)
    fig.suptitle("Response distribution per fold: R² is not comparable across folds "
                 "with different target variance", fontsize=8)
    fig.tight_layout()
    return F.save_figure(fig, "figS2_cv_folds", out)


FIGURES = {
    "map_study_area": fig_map_study_area,
    "sampling": fig_sampling,
    "predictor_quality": fig_predictor_quality,
    "curves_by_index": fig_curves_by_index,
    "phase": fig_phase,
    "lsp_topography": fig_lsp_topography,
    "example_plot": fig_example_plot,
    "cv_folds": fig_cv_folds,
    "curves_by_project": fig_curves_by_project,
    "environment_maps": fig_environment_maps,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--only", nargs="*", choices=list(FIGURES), default=None)
    p.add_argument("--base-size", type=int, default=9)
    args = p.parse_args()

    F.set_paper_style(args.base_size)
    D = load_all()
    print(f"plots: {len(D['plots'])} | with phenology: {len(D.get('db', []))} | "
          f"lsp rows: {len(D.get('lsp', []))}")

    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        try:
            path = fn(D, args.out_dir)
            print(f"  {name:20s} -> {path.name if path else 'skipped (missing input)'}")
        except Exception as e:
            print(f"  {name:20s} -> FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
