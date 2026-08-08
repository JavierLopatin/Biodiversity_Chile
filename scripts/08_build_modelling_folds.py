#!/usr/bin/env python3
"""Build the cross-validation schemes used by every model in this project.

`cv_folds.parquet` (from script 01) already carries three grouped schemes. Two things are
missing for the modelling stage, and one is added as a deliberate reference:

  kfold5_owner     NEW, PRIMARY. Groups by `Owner`, the contributor. `kfold5_dataset` groups
                   by `metadata_id`, but eight of Ovalle's projects, three of Galleguillos's
                   and two of Miranda's are separate metadata_ids from the same person, with
                   the same field protocol and often the same sites. Holding out a dataset
                   therefore does not hold out the protocol. Holding out the owner does.

  kfold5_block20   NEW, spatial. Neither `Owner` nor `metadata_id` nor `Location` is
                   geometry: measured on this subset, 103 pairs of datasets have overlapping
                   bounding boxes, so two "independent" projects can hold plots a few hundred
                   metres apart. A 20 km UTM grid is the only scheme here that actually
                   controls spatial autocorrelation, which is what risk R8 is about.

  lodo_owner       NEW. Leave-one-contributor-out — the hardest transferability test.

  kfold5_random    NEW, and deliberately optimistic. Reported so the gap against the grouped
                   schemes can be quoted. That gap is a result, not a diagnostic:
                   docs/02_innovation_and_impact.md asks for both.

The block size is audited rather than assumed: `--block-km` accepts several values and the
fold report records the composition of each, so the choice is visible to a reader.

Usage:
    python scripts/08_build_modelling_folds.py
    python scripts/08_build_modelling_folds.py --block-km 10 20 25 --primary-block-km 20
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


def build(plots: pd.DataFrame, block_km: list[float], primary_km: float,
          k: int = 5, seed: int = 42, min_lodo: int = 20) -> pd.DataFrame:
    schemes: list[pd.DataFrame] = []

    def add(df: pd.DataFrame, name: str) -> None:
        schemes.append(df.assign(scheme=name))
        n_test = int((df["split"] == "test").sum())
        print(f"  {name:22s} folds={df['fold'].nunique():>2}  test rows={n_test:>5}")

    add(cv_groups.grouped_kfold(plots, "Owner", k=k, seed=seed, stratify_on="richness"),
        "kfold5_owner")

    for km in block_km:
        col = f"block{int(km)}"
        plots[col] = cvmod.add_block_key(plots, km)
        n_blocks = plots[col].nunique()
        largest = int(plots[col].value_counts().iloc[0])
        print(f"  [{col}] {n_blocks} blocks, largest {largest} plots, "
              f"median {int(plots[col].value_counts().median())}")
        add(cv_groups.grouped_kfold(plots, col, k=k, seed=seed, stratify_on="richness"),
            f"kfold5_{col}")

    add(cv_groups.leave_one_group_out(plots, "Owner", min_size=min_lodo), "lodo_owner")
    add(cvmod.random_kfold(plots, k=k, seed=seed), "kfold5_random")

    out = pd.concat(schemes, ignore_index=True)
    # keep the primary block scheme under a stable name regardless of the audited sizes
    primary = f"kfold5_block{int(primary_km)}"
    if primary not in set(out["scheme"]):
        raise SystemExit(f"--primary-block-km {primary_km} not among --block-km {block_km}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--block-km", type=float, nargs="+", default=[10.0, 20.0, 25.0])
    p.add_argument("--primary-block-km", type=float, default=20.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-lodo", type=int, default=20,
                   help="smallest group that may become a leave-one-out test fold")
    args = p.parse_args()

    d = Path(args.derived)
    plots = pd.read_parquet(d / "plots_subset.parquet")
    plots = plots.sort_values(ID_COL, kind="stable").reset_index(drop=True)
    print(f"plots: {len(plots)}  owners: {plots.Owner.nunique()}  "
          f"datasets: {plots.metadata_id.nunique()}  locations: {plots.Location.nunique()}")

    print("\nnew schemes:")
    new = build(plots, args.block_km, args.primary_block_km, k=args.k, seed=args.seed,
                min_lodo=args.min_lodo)

    old_path = d / "cv_folds.parquet"
    print("\nexisting schemes:")
    old = pd.read_parquet(old_path)
    for name, g in old.groupby("scheme"):
        print(f"  {name:22s} folds={g['fold'].nunique():>2}  "
              f"test rows={int((g['split'] == 'test').sum()):>5}")

    allcv = pd.concat([new, old[["fold", "held_out", ID_COL, "split", "scheme"]]],
                      ignore_index=True)

    out_path = d / "cv_folds_modelling.parquet"
    allcv.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}  ({len(allcv):,} rows, {allcv['scheme'].nunique()} schemes)")

    report = cv_groups.fold_report(plots, allcv, response="richness")
    rep_path = d / "cv_fold_report_modelling.csv"
    report.to_csv(rep_path, index=False)
    print(f"-> {rep_path}")

    # The check that matters: a test fold with little response variance returns R2 near zero
    # however good the model is, so a scheme with such a fold is not usable for comparison.
    pooled_sd = float(plots["richness"].std())
    weak = report[report["test_sd"] < 0.5 * pooled_sd]
    print(f"\npooled richness sd = {pooled_sd:.2f}")
    if len(weak):
        print(f"WARNING: {len(weak)} folds with test sd < half the pooled sd "
              "(their per-fold R2 is not comparable; pooled OOF scoring is mandatory):")
        print(weak[["scheme", "fold", "held_out", "n_test", "test_mean", "test_sd"]]
              .to_string(index=False))
    else:
        print("all folds carry at least half the pooled response sd")

    print("\nprimary scheme:    kfold5_owner")
    print(f"spatial complement: kfold5_block{int(args.primary_block_km)}")
    print("optimistic ref:    kfold5_random")


if __name__ == "__main__":
    main()
