#!/usr/bin/env python3
"""Pretrain the 2-D trunk by masked autoencoding on the pixel curves.

The convolutional family loses on data, not on architecture: 1,082 labelled plots against a
model that has to learn a phenological representation from scratch. The cubes already hold
**135,250 unlabelled pixel curves** -- 1,082 plots x 25 pixels x 5 indices -- from the same
sensor and window. This spends them.

The encoder is the existing `PhenoNetS` trunk, so fine-tuning yields exactly the supervised
model with different initial weights and the comparison isolates pretraining. See
`src/biodiv/mae.py` for why not the ViT from Trait_2DCNN.

**Declare this when reporting.** Pretraining sees no targets, so no response leaks. It does
see pixels belonging to plots that later land in a test fold -- input-side leakage, standard
in self-supervised learning, and not nothing. The strict alternative is `--per-fold`, which
pretrains five times against each training split; it costs 5x and is the check to run if the
gain is worth defending.

Usage:
    python scripts/31_pretrain_mae.py --substrate serpentine --index kndvi
    python scripts/31_pretrain_mae.py --substrate serpentine --indices all --epochs 50
    python scripts/31_pretrain_mae.py --per-fold          # one checkpoint per CV fold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    args = p.parse_args()

    ids = feat.plot_ids(args.derived)
    indices = INDICES if args.indices == "all" else [args.index]
    print(f"substrate={args.substrate}  indices={indices}  arch={args.arch} "
          f"width={args.width}  patch={args.patch}  mask={args.mask_ratio}")
    print(f"curves: BIODIV_CURVES={feat.curve_suffix()!r}")

    t0 = time.time()
    imgs, owner = pixel_images(args.substrate, indices, args.derived, ids,
                               normalize="none", rotation="trough")
    print(f"{len(imgs):,} unlabelled images of shape {tuple(imgs.shape[1:])} "
          f"({len(ids)} plots x 25 pixels x {len(indices)} indices), "
          f"{time.time()-t0:.0f} s to build")

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.substrate}_{'all' if len(indices) > 1 else indices[0]}_{args.arch}" \
          f"_w{args.width}_p{args.patch}_m{args.mask_ratio:g}{feat.curve_suffix()}"

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
