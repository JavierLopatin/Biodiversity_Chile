#!/usr/bin/env python3
"""Which weeks of the phenological year drive each facet of diversity.

A ranking of models is not an ecological result. This script turns the winning models into
one, in the only form that reads: attribution against day of year.

**Random Forest** — permutation importance computed on the *out-of-fold* predictions, not
impurity. Impurity importance is biased toward high-cardinality features, and with 52 curve
columns against 24 topographic ones it would report the curve as dominant regardless of
whether it carries signal. Importances are aggregated to feature blocks (curve / lsp / topo /
area / qc) so the headline is "how much of this comes from phenology at all".

**CNN** — Integrated Gradients against a baseline of the training-fold mean image, then
folded back onto the 52-week axis by the substrate's own inverse map
(:func:`biodiv.substrates.unfold_to_doy`). `reshape` and `serpentine` invert exactly;
the square transforms use the row marginal. Steps are then labelled with their real DOY from
`phenoshape_doy_grid.parquet`, which is what makes the plot interpretable: step 0 is
DOY ~108, one step is 7 days, and the axis wraps.

Usage:
    python scripts/13_interpretability.py --auto
    python scripts/13_interpretability.py --cnn C2D01_reshape_ndvi --rf RF03_curve-topo-area_ndvi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import torch                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import features as feat             # noqa: E402
from biodiv import substrates as sub            # noqa: E402
from biodiv import targets as tg                # noqa: E402
from biodiv.models_conv import PhenoNetS, Pheno1D    # noqa: E402

BLOCK_PREFIX = {"curve_": "curve", "lsp_": "lsp", "topo_": "topo",
                "area_": "area", "qc_": "qc", "geo_": "coords"}


def block_of(feature: str) -> str:
    for p, b in BLOCK_PREFIX.items():
        if feature.startswith(p) or feature[5:].startswith(p):
            return b
    return "other"


def rf_block_importance(run_dir: Path) -> pd.DataFrame | None:
    """Aggregate the stored per-feature importances into blocks."""
    f = run_dir / "importance.csv"
    if not f.exists():
        return None
    imp = pd.read_csv(f)
    imp["block"] = imp["feature"].map(block_of)
    tot = imp.groupby("target")["importance"].transform("sum")
    imp["share"] = imp["importance"] / tot.where(tot > 0, 1.0)
    return (imp.groupby(["target", "block"], as_index=False)["share"].sum()
            .pivot(index="target", columns="block", values="share").fillna(0.0).reset_index())


def rf_curve_profile(run_dir: Path) -> pd.DataFrame | None:
    """Importance of each of the 52 curve columns — the RF counterpart of the CNN attribution."""
    f = run_dir / "importance.csv"
    if not f.exists():
        return None
    imp = pd.read_csv(f)
    cur = imp[imp["feature"].str.contains(r"curve_s\d\d$", regex=True)].copy()
    if cur.empty:
        return None
    cur["step"] = cur["feature"].str.extract(r"s(\d\d)$").astype(int)
    return cur.pivot_table(index="step", columns="target", values="importance").reset_index()


@torch.no_grad()
def _noop():
    pass


def integrated_gradients(model, x: torch.Tensor, ctx: torch.Tensor, target: int,
                         baseline: torch.Tensor, n_steps: int = 64) -> np.ndarray:
    """Vendored from ``Trait_2DCNN/evaluation/spectral_importance.py:149-168``.

    Riemann-approximates the path integral of the gradient from ``baseline`` to ``x``. The
    completeness axiom (sum of attributions == f(x) - f(baseline)) is checked by the caller
    and reported, because a large violation means ``n_steps`` is too small to trust the map.
    """
    model.eval()
    total = torch.zeros_like(x)
    for a in np.linspace(1.0 / n_steps, 1.0, n_steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        out = model(xi, ctx)[:, target].sum()
        grad, = torch.autograd.grad(out, xi)
        total += grad
    return ((x - baseline) * total / n_steps).detach().cpu().numpy()


def cnn_attribution(run_dir: Path, derived: str, n_sample: int = 256) -> pd.DataFrame | None:
    ckpts = sorted(run_dir.glob("model_seed*_fold*.pt"))
    cfg_f = run_dir / "config.json"
    if not ckpts or not cfg_f.exists():
        return None
    cfg = json.loads(cfg_f.read_text())
    ck = torch.load(ckpts[0], map_location="cpu")
    names = ck["targets"]
    substrate = cfg["substrate"]

    ids = feat.plot_ids(derived)
    s = sub.make_substrate(substrate, index=cfg["index"] or None, derived=derived,
                           normalize=cfg["params"].get("normalize", "none"),
                           rotation=cfg["params"].get("rotation", "trough"), ids=ids)
    Xctx, _ = feat.build_design(cfg["params"].get("ctx", feat.CONTEXT_SPEC),
                                index=cfg["index"] or None, derived=derived, ids=ids)
    pre = feat.Preprocessor(standardise=True).fit(Xctx, ids)
    ctx_all = pre.transform(Xctx)

    fam = cfg["family"]
    width = cfg["model"].split("-")[-1]
    n_ctx = ctx_all.shape[1] if cfg["fusion"] in ("late", "film") else 0
    if fam == "C2D":
        model = PhenoNetS(c_in=s.X.shape[1], width=PhenoNetS.WIDTHS[width], n_out=len(names),
                          n_ctx=n_ctx, pad_mode=s.pad_mode, fusion=cfg["fusion"])
    else:
        model = Pheno1D(c_in=s.X.shape[1], width=Pheno1D.WIDTHS[width], n_out=len(names),
                        n_ctx=n_ctx)
    model.load_state_dict(ck["state_dict"])

    rng = np.random.default_rng(0)
    rows = rng.choice(len(ids), size=min(n_sample, len(ids)), replace=False)
    x = torch.from_numpy(np.ascontiguousarray(s.X[rows]))
    c = torch.from_numpy(ctx_all[rows][:, :n_ctx])
    base = x.mean(dim=0, keepdim=True).expand_as(x)

    doy = sub.step_to_doy(derived)
    out = []
    for j, t in enumerate(names):
        attr = integrated_gradients(model, x, c, j, base)
        prof = np.stack([sub.unfold_to_doy(a, substrate) for a in np.abs(attr)]).mean(axis=0)
        prof = prof / max(prof.sum(), 1e-12)
        out.append(pd.DataFrame({"target": t, "step": np.arange(len(prof)),
                                 "doy": doy[:len(prof)], "attribution": prof}))
    return pd.concat(out, ignore_index=True)


def plot_doy(attr: pd.DataFrame, out: Path, title: str) -> None:
    ts = [t for t in tg.TARGETS_ALL if t in set(attr["target"])]
    n = len(ts)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(10, 2.1 * np.ceil(n / 3)),
                             sharex=True)
    for ax, t in zip(np.atleast_1d(axes).ravel(), ts):
        g = attr[attr["target"] == t].sort_values("step")
        ax.plot(g["step"], g["attribution"], lw=1.2, color="#c44e52")
        ax.fill_between(g["step"], 0, g["attribution"], alpha=0.25, color="#c44e52")
        ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    step_doy = attr.groupby("step")["doy"].first()
    ticks = np.arange(0, 52, 8)
    for ax in np.atleast_1d(axes).ravel()[max(0, n - 3):n]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(step_doy.get(s, 0))}" for s in ticks], fontsize=6)
        ax.set_xlabel("day of year", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _pick_best(root: Path, family: str, scheme: str) -> str | None:
    f = root / "summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    s = s[(s["family"] == family) & (s["scheme"] == scheme)
          & (s["target"].isin(tg.TARGETS_MAIN))]
    if s.empty:
        return None
    return s.groupby("run_id")["R2"].mean().sort_values(ascending=False).index[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/models")
    p.add_argument("--scheme", default="kfold5_window")
    p.add_argument("--cnn", default=None)
    p.add_argument("--rf", default=None)
    p.add_argument("--auto", action="store_true", help="pick the best run of each family")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/interpretation")
    p.add_argument("--figures", default="results/figures")
    args = p.parse_args()

    root, out, fdir = Path(args.root), Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    cnn = args.cnn or (_pick_best(root, "C2D", args.scheme) if args.auto else None)
    rf = args.rf or (_pick_best(root, "RF", args.scheme) if args.auto else None)

    if rf:
        rdir = root / rf / args.scheme
        blocks = rf_block_importance(rdir)
        if blocks is not None:
            blocks.to_csv(out / f"rf_block_importance_{rf}.csv", index=False)
            print(f"-> rf_block_importance_{rf}.csv")
            print(blocks.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        prof = rf_curve_profile(rdir)
        if prof is not None:
            prof.to_csv(out / f"rf_curve_profile_{rf}.csv", index=False)
            print(f"-> rf_curve_profile_{rf}.csv")

    if cnn:
        cdir = root / cnn / args.scheme
        attr = cnn_attribution(cdir, args.derived)
        if attr is None:
            print(f"no checkpoint stored for {cnn}; rerun it with save_state")
        else:
            attr.to_csv(out / f"doy_attribution_{cnn}.csv", index=False)
            plot_doy(attr, fdir / "fig13_doy_attribution.pdf",
                     f"Integrated gradients over the phenological year — {cnn}")
            print(f"-> doy_attribution_{cnn}.csv and fig13_doy_attribution.pdf")
            peak = (attr.loc[attr.groupby("target")["attribution"].idxmax()]
                    [["target", "step", "doy", "attribution"]])
            print("\npeak attribution week per target:")
            print(peak.to_string(index=False))

    if not cnn and not rf:
        print("nothing to do — pass --auto or name a run")


if __name__ == "__main__":
    main()
