#!/usr/bin/env python3
"""Tiers 3 and 4 — 1-D CNN over the curve, and 2-D CNN over eleven substrates.

The question the whole benchmark is built around: **does a second dimension help, and does it
have to be a real one?**

  C1D*   1-D convolution over the 52 weeks. The control. Any 2-D gain must be a gain over
         this, not over the Random Forest.

  C2D01-09  the manufactured catalogue — reshape, serpentine, hilbert, gaf, mtf, ndi, cwt,
         cos2d, spectrogram. These were designed for time series and applied in `Trait_2DCNN`
         to a *spectral* axis; here they operate in their native domain for the first time.
         `reshape` won that hyperspectral benchmark, which makes it the one to beat.

  C2D10  `stack5`, 5 indices x 52 weeks. The vertical axis is the vegetation index, so a 3x3
         kernel reads greenness-against-moisture contrast within the same week.

  C2D11  `pxcube`, 25 pixels x 52 weeks. The vertical axis is within-plot phenological
         heterogeneity — a mechanistic predictor of exactly the beta-diversity targets.

The last two are the substrates whose second axis is not manufactured, which is the claim
`docs/03_cnn_architecture.md` wanted the (unavailable) year x DOY phenocube to support.

Usage:
    python scripts/11_run_conv.py --substrate reshape --index ndvi --scheme kfold5_owner
    python scripts/11_run_conv.py --stage 4a --index nbr --seeds 5
    python scripts/11_run_conv.py --count-params
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                              # noqa: E402

from biodiv import features as feat             # noqa: E402
from biodiv import runlog                       # noqa: E402
from biodiv import substrates as sub            # noqa: E402
from biodiv.dl_runner import run_dl             # noqa: E402
from biodiv.models_conv import PhenoNetS, Pheno1D, count_params   # noqa: E402
from biodiv.trainer import TrainCfg             # noqa: E402

#: Stage 4a: the transform screen. Ordered so the two non-manufactured substrates come last
#: and are directly comparable against the catalogue that precedes them.
STAGE_4A = ["reshape", "serpentine", "gaf", "mtf", "ndi", "cwt", "hilbert", "cos2d",
            "spectrogram", "stack5", "pxcube"]

SUBSTRATE_ID = {s: f"C2D{i + 1:02d}" for i, s in enumerate(STAGE_4A)}
SUBSTRATE_ID["curve1d"] = "C1D01"
SUBSTRATE_ID["curve5"] = "C1D02"


def family_of(substrate: str) -> str:
    return "C1D" if substrate in ("curve1d", "curve5") else "C2D"


def run(substrate: str, index: str | None, args, width: str | None = None,
        fusion: str | None = None, rotation: str | None = None,
        normalize: str | None = None, mixup: bool | None = None,
        augment: bool | None = None, tag: str = "") -> None:
    width = width or args.width
    fusion = fusion or args.fusion
    rotation = rotation or args.rotation
    normalize = normalize or args.normalize
    fam = family_of(substrate)
    base = SUBSTRATE_ID.get(substrate, fam)
    ident = runlog.make_run_id(base + (f"-{tag}" if tag else ""),
                               substrate, index or "", "", fusion)
    if width != "B":
        ident += f"_w{width}"
    if rotation != "trough":
        ident += f"_{rotation}"
    if normalize != "none":
        ident += f"_n{normalize}"
    if mixup:
        ident += "_mixup"
    if augment is False:
        ident += "_noaug"

    cfg = TrainCfg(max_epochs=args.max_epochs, patience=args.patience,
                   batch_size=args.batch_size, lr=args.lr,
                   augment=args.augment if augment is None else augment,
                   mixup=args.mixup if mixup is None else mixup)
    if args.px == "center":
        ident += "_ctr"
    run_dl(family=fam, run_id=ident, scheme=args.scheme, substrate=substrate, index=index,
           px=args.px,
           width=width, fusion=fusion, rotation=rotation, normalize=normalize,
           seeds=tuple(range(args.seeds)), derived=args.derived, out_root=Path(args.out),
           train_cfg=cfg, force=args.force, save_state=True,
           notes=f"{fam} on {substrate}")


def count_params_table() -> None:
    print(f"{'model':10s} {'width':16s} {'c_in':>5s} {'n_ctx':>6s} {'params':>10s}")
    for w in ("A", "B", "C"):
        for c_in in (1, 5):
            m = Pheno1D(c_in=c_in, width=Pheno1D.WIDTHS[w], n_out=9, n_ctx=28)
            print(f"{'Pheno1D':10s} {str(Pheno1D.WIDTHS[w]):16s} {c_in:5d} {28:6d} "
                  f"{count_params(m):10,d}")
    for w in ("A", "B", "C", "X"):
        for c_in, ctx in ((1, 0), (1, 28), (2, 28), (5, 28)):
            m = PhenoNetS(c_in=c_in, width=PhenoNetS.WIDTHS[w], n_out=9, n_ctx=ctx,
                          fusion="late" if ctx else "none")
            print(f"{'PhenoNetS':10s} {str(PhenoNetS.WIDTHS[w]):16s} {c_in:5d} {ctx:6d} "
                  f"{count_params(m):10,d}")


def substrate_shapes(index: str, derived: str) -> None:
    print(f"{'substrate':14s} {'shape (C,H,W)':20s} {'pad (v,h)':22s} rows")
    for s in STAGE_4A + ["curve1d"]:
        try:
            obj = sub.make_substrate(s, index=index, derived=derived)
            print(f"{s:14s} {str(tuple(obj.shape)):20s} {str(obj.pad_mode):22s} "
                  f"{len(obj.rows)}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"{s:14s} FAILED: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrate", default=None)
    p.add_argument("--stage", choices=["3", "4a", "4b", "4c"], default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--indices", nargs="+", default=None, help="stage 4b: indices to expand over")
    p.add_argument("--top", nargs="+", default=None, help="stage 4b: substrates to expand")
    p.add_argument("--scheme", default="kfold5_owner")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--width", default="B", choices=["A", "B", "C", "X"])
    p.add_argument("--fusion", default="late", choices=["none", "late", "film", "patch"])
    p.add_argument("--rotation", default="trough", choices=["trough", "calendar"])
    p.add_argument("--normalize", default="none", choices=["none", "global", "perSample"])
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/models")
    p.add_argument("--px", default="mean5x5", choices=["mean5x5", "center"],
                   help="pixel level of the curve and of the topographic context")
    p.add_argument("--force", action="store_true")
    p.add_argument("--count-params", action="store_true")
    p.add_argument("--shapes", action="store_true")
    args = p.parse_args()

    if args.count_params:
        count_params_table()
        return
    if args.shapes:
        substrate_shapes(args.index or "ndvi", args.derived)
        return

    if args.substrate:
        multi = args.substrate in ("stack5", "curve5")
        idxs = [args.index] if (args.index or multi) else feat.INDICES
        idxs = [None] if multi else idxs
        for ix in idxs:
            run(args.substrate, ix, args)
        return

    if args.stage == "3":
        for ix in (args.indices or feat.INDICES):
            run("curve1d", ix, args)
        return

    if args.stage == "4a":
        ix = args.index or "ndvi"
        for s in STAGE_4A:
            run(s, None if s == "stack5" else ix, args)
        return

    if args.stage == "4b":
        tops = args.top or ["reshape", "stack5", "pxcube"]
        for s in tops:
            if s == "stack5":
                continue                      # stack5 already uses all five indices
            for ix in (args.indices or feat.INDICES):
                run(s, ix, args)
        return

    if args.stage == "4c":
        s = args.substrate or "reshape"
        ix = args.index or "ndvi"
        for rot in ("trough", "calendar"):
            run(s, ix, args, rotation=rot, tag="R")
        for norm in ("none", "global", "perSample"):
            run(s, ix, args, normalize=norm, tag="N")
        for fus in ("none", "late", "film", "patch"):
            run(s, ix, args, fusion=fus, tag="T")
        for w in ("A", "B", "C", "X"):
            run(s, ix, args, width=w, tag="W")
        run(s, ix, args, mixup=True, tag="M")
        run(s, ix, args, augment=False, tag="A")
        return

    p.error("give --substrate or --stage")


if __name__ == "__main__":
    main()
