#!/usr/bin/env python3
"""Pretrain the 2-D trunk by masked autoencoding.

The convolutional family loses on data, not on architecture: 1,082 labelled plots against a
model that has to learn a phenological representation from scratch.

TWO POOLS, AND THE FIRST ONE FAILED. The default reads the 135,250 pixel curves already on
disk -- 1,082 plots x 25 pixels x 5 indices. Measured, they gain nothing: **0.359 against
0.362 unpretrained** (`docs/14` section 4). The method worked (reconstruction MSE 0.0095 ->
0.0031); the data did not. Those curves come from the 25-pixel windows of the same plots,
correlate at +0.844 within a plot, and carry participation dimension 1.3 against the plots'
own 1.2 -- 125x more images spanning 1.08x more space.

`--unlabelled` reads the MapBiomas-masked pool built by `scripts/32_sample_unlabelled.py`
instead: native vegetation only, drawn across every populated 10 km cell of the study region
at >=1 km spacing, with the plots' own year distribution. That pool is what the coverage
argument asks for; the pixel curves are kept only as the baseline it has to beat.

The encoder is the existing `PhenoNetS` trunk, so fine-tuning yields exactly the supervised
model with different initial weights and the comparison isolates pretraining. See
`src/biodiv/mae.py` for why not the ViT from Trait_2DCNN.

**Declare this when reporting.** Pretraining sees no targets, so no response leaks. On the
pixel curves it does see pixels of plots that later land in a test fold -- input-side leakage,
standard in self-supervised learning, and not nothing; `--per-fold` is the strict alternative,
at 5x the cost. On `--unlabelled` there is nothing to defend against: every sample is at least
1 km from any plot, so `--per-fold` does not apply and is refused.

**Measure coverage before spending GPU.** If the new pool's participation dimension does not
clear the 1.3 of the pixel curves, the coverage problem is not solved and no amount of
pretraining will fix it.

Usage:
    python scripts/31_pretrain_mae.py --substrate serpentine --index kndvi
    python scripts/31_pretrain_mae.py --unlabelled data/derived/unlabelled
    python scripts/31_pretrain_mae.py --unlabelled data/derived/unlabelled \\
        --landcover 'Shrubland,Secondary Forest,Primary Forest'
    python scripts/31_pretrain_mae.py --per-fold          # one checkpoint per CV fold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import curves                       # noqa: E402
from biodiv import cv as cvmod                  # noqa: E402
from biodiv import features as feat             # noqa: E402
from biodiv import substrates as sub            # noqa: E402
from biodiv.dl_runner import substrate_builder   # noqa: E402
from biodiv.mae import pretrain                 # noqa: E402

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]


def _rel(f: Path) -> Path:
    return f.relative_to(ROOT) if f.is_relative_to(ROOT) else f


def pixel_images(substrate: str, indices: list[str], derived: str, ids,
                 normalize: str, rotation: str) -> tuple[np.ndarray, np.ndarray]:
    """Every pixel curve, transformed into the model's substrate.

    Returns ``(images, plot_index)`` where `plot_index[i]` says which plot row image `i`
    came from -- needed by `--per-fold` to hold out a fold's plots.
    """
    imgs, owner = [], []
    tf = substrate_builder(substrate, normalize=normalize)
    for ix in indices:
        cube, cids = sub.load_pixel_curves(ix, derived, ids)      # (N, 25, 52)
        n, npx, nt = cube.shape
        flat = cube.reshape(n * npx, 1, nt).astype(np.float32)
        imgs.append(tf(flat))
        owner.append(np.repeat(np.arange(n), npx))
    return np.concatenate(imgs), np.concatenate(owner)


def unlabelled_images(substrate: str, index: str, updir: Path, ngs: int,
                      normalize: str, landcover: str | None = None
                      ) -> tuple[np.ndarray, pd.DataFrame]:
    """The MapBiomas-masked pool, from observation-level series to model substrate.

    `scripts/32` stores observations with their dates rather than a fixed grid, because
    `docs/14` section 2 could only measure that step resolution does not matter (eight of
    them, spread 0.011) by deriving all eight without returning to the cube. The grid is
    therefore built here, with `biodiv.curves` -- the same interpolation the labelled plots
    get, which is the point of it living in a module.

    Each series is gridded over **its own causal window**, so all of them cover three years of
    real time regardless of when their first and last clear observation happened to fall.
    """
    man = pd.read_csv(updir / "manifest.csv")
    if "error" in man:
        man = man[man["error"].isna()]
    if landcover:
        keep = [c.strip().lower() for c in landcover.split(",")]
        man = man[man["mb_class"].str.lower().isin(keep)]
    man = man.reset_index(drop=True)

    ser = pd.read_parquet(updir / "series.parquet")
    ser = ser[ser["sample_id"].isin(set(man["sample_id"]))]
    ser["t"] = ser["time"].values.astype("datetime64[D]").astype(float)

    win = man.set_index("sample_id")[["win_start", "win_end"]]
    rows, kept = [], []
    for sid, g in ser.groupby("sample_id", sort=False):
        w = win.loc[sid]
        lo = np.datetime64(f"{int(w.win_start)}-01-01", "D").astype(float)
        hi = np.datetime64(f"{int(w.win_end)}-12-31", "D").astype(float)
        curve = curves.interp_grid(g["t"].to_numpy(), g[index].to_numpy(), ngs,
                                   t_min=lo, t_max=hi)
        if np.isfinite(curve).all():
            rows.append(curve)
            kept.append(sid)
    if not rows:
        raise SystemExit(f"no usable series in {updir} (index={index!r})")

    arr = np.asarray(rows, dtype=np.float32)[:, None, :]          # (N, 1, ngs)
    tf = substrate_builder(substrate, normalize=normalize)
    return tf(arr), man[man["sample_id"].isin(kept)].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrate", default="serpentine")
    p.add_argument("--index", default="kndvi")
    p.add_argument("--indices", default=None,
                   help="'all' pretrains on the five indices at once, 5x the images")
    p.add_argument("--width", default="B")
    p.add_argument("--arch", default="sep", choices=["sep", "res", "se", "multi"])
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.6, dest="mask_ratio")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-fold", action="store_true", dest="per_fold",
                   help="one checkpoint per fold, pretrained only on that fold's training "
                        "plots. Removes the input-side leakage at 5x the cost")
    p.add_argument("--scheme", default="kfold5_window")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/mae")
    p.add_argument("--unlabelled", default=None, metavar="DIR",
                   help="pretrain on the MapBiomas-masked pool from scripts/32 instead of "
                        "the plots' own pixel curves. The pixel curves come from the 25-pixel "
                        "windows of the same 1,082 plots and gained nothing (0.359 vs 0.362, "
                        "docs/14 section 4): they correlate at +0.844 within a plot and their "
                        "participation dimension is 1.3 against the plots' 1.2")
    p.add_argument("--landcover", default=None,
                   help="comma-separated MapBiomas classes to keep, e.g. "
                        "'Shrubland,Secondary Forest'. This is the ablation the pool was "
                        "built to support, not a filter to guess at")
    p.add_argument("--ngs", type=int, default=curves.NGS,
                   help="steps of the grid built from the observations")
    args = p.parse_args()

    indices = INDICES if args.indices == "all" else [args.index]
    print(f"substrate={args.substrate}  indices={indices}  arch={args.arch} "
          f"width={args.width}  patch={args.patch}  mask={args.mask_ratio}")

    t0 = time.time()
    if args.unlabelled:
        if args.per_fold:
            # the pool carries no plot identity, so there is no fold to hold out. It also
            # needs none: the samples are >=1 km from any plot, so unlike the pixel curves
            # there is no input-side leakage to defend against in the first place
            raise SystemExit("--per-fold does not apply to --unlabelled: the pool contains "
                             "no plot pixels, so no fold can leak into it")
        updir = Path(args.unlabelled)
        imgs, man = unlabelled_images(args.substrate, args.index, updir, args.ngs,
                                      normalize="none", landcover=args.landcover)
        owner = np.arange(len(imgs))
        ids = man["sample_id"]
        print(f"{len(imgs):,} unlabelled images of shape {tuple(imgs.shape[1:])} "
              f"from {_rel(updir)} ({args.index}, {args.ngs} steps), "
              f"{time.time()-t0:.0f} s to build")
        print("by land cover: "
              f"{man['mb_class'].value_counts().to_dict()}")
    else:
        ids = feat.plot_ids(args.derived)
        print(f"curves: BIODIV_CURVES={feat.curve_suffix()!r}")
        imgs, owner = pixel_images(args.substrate, indices, args.derived, ids,
                                   normalize="none", rotation="trough")
        print(f"{len(imgs):,} unlabelled images of shape {tuple(imgs.shape[1:])} "
              f"({len(ids)} plots x 25 pixels x {len(indices)} indices), "
              f"{time.time()-t0:.0f} s to build")

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    # the source belongs in the run id. A checkpoint pretrained on the plots' own pixels and
    # one pretrained on the MapBiomas pool are different objects, and `docs/14` section 3b
    # already cost this project a result by letting two different runs share a name
    src = "px"
    if args.unlabelled:
        src = "mb" + (f"-{args.landcover.replace(',', '+').replace(' ', '')}"
                      if args.landcover else "")
    tag = f"{args.substrate}_{'all' if len(indices) > 1 else indices[0]}_{args.arch}" \
          f"_w{args.width}_p{args.patch}_m{args.mask_ratio:g}_{src}" \
          f"{'' if args.unlabelled else feat.curve_suffix()}"

    kw = dict(width=args.width, arch=args.arch, pad_mode=sub.PAD_MODE.get(args.substrate,
                                                                          "zeros"),
              patch=args.patch, mask_ratio=args.mask_ratio, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr, seed=args.seed)

    if not args.per_fold:
        ck = pretrain(imgs, **kw)
        f = outdir / f"mae_{tag}.pt"
        torch.save(ck, f)
        print(f"\n  -> {_rel(f)}   final masked MSE {ck['history'][-1]:.5f}")
        return

    # one checkpoint per fold, pretrained only on that fold's training plots
    cvf = cvmod.load_schemes(Path(args.derived) / "cv_folds_modelling.parquet")
    pos = {pid: i for i, pid in enumerate(ids)}
    for fold, _, tr_ids, _ in cvmod.iter_folds(cvf, args.scheme):
        keep = np.isin(owner, [pos[i] for i in tr_ids])
        print(f"\nfold {fold}: {keep.sum():,} of {len(imgs):,} images")
        ck = pretrain(imgs[keep], **kw)
        f = outdir / f"mae_{tag}_fold{fold}.pt"
        torch.save(ck, f)
        print(f"  -> {_rel(f)}   final masked MSE {ck['history'][-1]:.5f}")


if __name__ == "__main__":
    main()
