#!/usr/bin/env python3
"""Tier 2 — multi-output tabular MLP over LSP metrics or over the 52-week curve.

This tier exists to separate two things the RF tier cannot separate. A forest fits each
target independently and cannot share a representation; an MLP with one trunk and nine heads
can. So the MLP-vs-RF contrast on the *same* features answers "does joint modelling of the
diversity facets help?", while the MLP01-vs-MLP02 contrast answers the same question the RF
tier asks about LSP against the curve — but now with a model that can exploit the ordering of
the 52 steps only through its weights, not through convolution. That makes it the honest
midpoint between the forest and the 1-D CNN.

Usage:
    python scripts/10_run_tabular_dl.py --model MLP01 --index ndvi --scheme kfold5_owner
    python scripts/10_run_tabular_dl.py --all --scheme kfold5_owner --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import features as feat            # noqa: E402
from biodiv import runlog                      # noqa: E402
from biodiv.dl_runner import run_dl            # noqa: E402
from biodiv.trainer import TrainCfg            # noqa: E402

MODELS = {
    "MLP01": dict(spec="lsp", width="B", per_index=True,
                  note="LSP metrics + context, shared trunk"),
    "MLP02": dict(spec="curve", width="B", per_index=True,
                  note="52-week curve + context, shared trunk"),
    "MLP03": dict(spec="lsp+curve+qc", width="B", per_index=True,
                  note="everything from one index"),
    "MLP04": dict(spec="curve_all", width="A", per_index=False,
                  note="curves of all five indices; narrower net for the wider input"),
    "MLP05a": dict(spec="curve", width="A", per_index=True, note="width ablation A"),
    "MLP05c": dict(spec="curve", width="C", per_index=True,
                   note="width ablation C — over the 50k budget, an over-fitting control"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--index", default=None)
    p.add_argument("--scheme", default="kfold5_owner")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/models")
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--no-augment", action="store_true",
                   help="curve augmentation is meaningless for the LSP block; auto-disabled there")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    todo = sorted(MODELS) if args.all else ([args.model] if args.model else [])
    if not todo:
        p.error("give --model or --all")
    seeds = tuple(range(args.seeds))

    for name in todo:
        meta = MODELS[name]
        idxs = ([args.index] if args.index else feat.INDICES) if meta["per_index"] else [None]
        for ix in idxs:
            spec = meta["spec"]
            run_dl(
                family="MLP",
                run_id=runlog.make_run_id(name, spec.replace("+", "-"), ix or ""),
                scheme=args.scheme, substrate="tabular", index=ix, features_spec=spec,
                width=meta["width"], fusion="late", target_set="all", seeds=seeds,
                derived=args.derived, out_root=Path(args.out),
                train_cfg=TrainCfg(max_epochs=args.max_epochs, patience=args.patience,
                                   augment=False),      # tabular rows are not curves
                force=args.force, notes=meta["note"])


if __name__ == "__main__":
    main()
