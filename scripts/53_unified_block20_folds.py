#!/usr/bin/env python3
"""Build `kfold5_block20_unified`: the block-CV scheme over Parcelas-CL + Living Trees.

Reuses `biodiv.cv.add_block_key`/`biodiv.cv_groups.grouped_kfold` unchanged
(`scripts/08_build_modelling_folds.py`'s own recipe) -- the only difference is the
coordinate columns fed in: `add_block_key` reads `plots["X"]`/`plots["Y"]` literally, so
this script hands it the unified LAEA-Chile `X_m`/`Y_m` columns
(`scripts/51_build_unified_dataset.py`) under those names, instead of Parcelas-CL's own
UTM19S X/Y -- which would distort the 20km grid badly for plots near Magallanes.

Usage:
    python scripts/53_unified_block20_folds.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import cv as cvmod          # noqa: E402
from biodiv import cv_groups            # noqa: E402

ID_COL = "PlotObservationID"


def build(args: argparse.Namespace) -> None:
    derived = Path(args.derived)
    plots = pd.read_parquet(derived / "plots_unified.parquet")
    plots = plots.sort_values(ID_COL, kind="stable").reset_index(drop=True)

    grid_input = pd.DataFrame({"X": plots["X_m"], "Y": plots["Y_m"]})
    plots["block20"] = cvmod.add_block_key(grid_input, args.block_km)

    n_blocks = plots["block20"].nunique()
    counts = plots["block20"].value_counts()
    print(f"[block{int(args.block_km)}] {n_blocks} bloques sobre {len(plots)} parcelas, "
          f"el mayor con {int(counts.iloc[0])}, mediana {int(counts.median())}")

    folds = cv_groups.grouped_kfold(plots, "block20", k=args.k, seed=args.seed,
                                    stratify_on=args.stratify)
    scheme = f"kfold5_block{int(args.block_km)}_unified"
    folds = folds.assign(scheme=scheme)

    n_test = int((folds["split"] == "test").sum())
    print(f"folds={folds['fold'].nunique()}  filas de test={n_test}  "
          f"parcelas cubiertas={folds[ID_COL].nunique()} de {len(plots)}")

    for source in sorted(plots["source"].unique()):
        ids = set(plots.loc[plots["source"] == source, ID_COL])
        n_test_src = int((folds[ID_COL].isin(ids) & (folds["split"] == "test")).sum())
        print(f"  {source}: {len(ids)} parcelas totales, {n_test_src} filas de test")

    empty_folds = [
        int(f) for f, g in folds[folds["split"] == "test"].groupby("fold")
        if g[ID_COL].nunique() == 0
    ]
    if empty_folds:
        print(f"AVISO: folds sin ninguna parcela de test: {empty_folds}")

    out_path = derived / "cv_folds_unified_block20.parquet"
    folds.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}  ({len(folds):,} filas, scheme={scheme!r})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--block-km", type=float, default=20.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify", default="richness",
                   help="response variable balanced across folds (empty disables it)")
    build(p.parse_args())


if __name__ == "__main__":
    main()
