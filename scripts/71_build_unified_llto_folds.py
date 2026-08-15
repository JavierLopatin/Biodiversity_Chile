#!/usr/bin/env python3
"""Build `kfold_loc_time_unified`: LLTO (leave-location-and-time-out) over the unified
pool (Parcelas-CL + Living Trees, 3,102 plots).

Same recipe `src/biodiv/cv_groups.py:kfold_loc_time` already uses for the Parcelas-CL-only
LLTO reported in `docs/16_stemp_protocol.md` section 2.5 -- spatial block x temporal block,
test = a (block, period) cell, train = everything outside both AND outside the causal
3-year observation window of the held-out cell (`_buffered_train`, so training never sees
Landsat pixels the test plots' own window would also touch). Two things differ from the
non-unified version because the pool differs:

- ``loc_fold``: `kfold_loc_time` needs the outer **fold assignment** (0..k-1), not the raw
  geometric block key -- `s{int(s)}t{b}` names a (spatial-fold, time-block) cell, and a
  raw block key is a non-numeric string ("-10_-26"). Read straight from the already-built
  `cv_folds_unified_block20.parquet` (`scripts/53_unified_block20_folds.py`, 20x20 km
  blocks over the Chile-centred equal-area `X_m`/`Y_m` projection -- NOT Parcelas-CL's own
  UTM19S `X`/`Y`, which is NaN for every Living Trees row) rather than recomputing
  `add_block_key` here, so the spatial fold used by this LLTO is identical to the one
  `kfold5_block20_unified` itself reports on.
- ``edges``: `TIME_EDGES` (2002..2027) already covers the unified pool's year range
  (2003-2026, Living Trees 2011-2020) unchanged -- no new edges needed.

Usage:
    python scripts/71_build_unified_llto_folds.py
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
SCHEME = "kfold_loc_time_unified"


def build(args: argparse.Namespace) -> None:
    derived = Path(args.derived)
    plots = pd.read_parquet(derived / "plots_unified.parquet")
    plots = plots.sort_values(ID_COL, kind="stable").reset_index(drop=True)

    block20 = pd.read_parquet(derived / "cv_folds_unified_block20.parquet")
    block20_test_fold = block20[block20["split"] == "test"].set_index(ID_COL)["fold"]
    loc_fold = plots[ID_COL].map(block20_test_fold)  # index stays plots' own (positional)
    n_missing = int(loc_fold.isna().sum())
    if n_missing:
        raise SystemExit(f"{n_missing} parcelas sin fold espacial en cv_folds_unified_block20.parquet")
    loc_fold = loc_fold.astype(int)
    n_blocks = loc_fold.nunique()
    print(f"[block20 outer folds] {n_blocks} folds espaciales sobre {len(plots)} parcelas")

    tb = cv_groups.time_block(plots["Year"])
    print(f"[time] {tb.nunique()} bloques temporales, bordes={cv_groups.TIME_EDGES}")
    print(pd.concat([tb.value_counts().sort_index(),
                     plots.groupby(tb.to_numpy())["Year"].agg(["min", "max"])], axis=1))

    folds = cv_groups.kfold_loc_time(plots, loc_fold, min_test=args.min_test)
    if folds.empty:
        raise SystemExit("0 celdas (bloque x tiempo) superan min_test -- LLTO no produce "
                         "ningun fold utilizable en este pool")
    folds = folds.assign(scheme=SCHEME)

    n_cells = folds["held_out"].nunique()
    n_test = int((folds["split"] == "test").sum())
    n_cov = folds.loc[folds["split"] == "test", ID_COL].nunique()
    print(f"\nceldas (bloque x tiempo) con >= {args.min_test} test: {n_cells}")
    print(f"filas de test: {n_test}  parcelas cubiertas: {n_cov} de {len(plots)} "
         f"({100 * n_cov / len(plots):.1f}%)")

    for source in sorted(plots["source"].unique()):
        ids = set(plots.loc[plots["source"] == source, ID_COL])
        n_test_src = int((folds[ID_COL].isin(ids) & (folds["split"] == "test")).sum())
        n_ids_src = len(ids)
        print(f"  {source}: {n_ids_src} parcelas totales, {n_test_src} filas de test "
             f"({100 * n_test_src / n_ids_src:.1f}%)")

    # Coverage against the PG-style facets specifically -- their N is already a fraction of
    # the pool (2,499 / 888 / 895 of 3,102), and LLTO's finer (block x time) partition can
    # starve individual test cells of PG-observed plots well below `min_test` even though
    # the cell itself cleared the threshold on the full pool.
    for name, path in [("lcbd_count_sorensen", "lcbd_count_sorensen_unified_padded.parquet"),
                       ("pd_inext_q0", "pd_inext_coverage_unified_padded.parquet"),
                       ("td_inext_q0", "td_inext_coverage_unified_padded.parquet")]:
        f = derived / path
        if not f.exists():
            continue
        obs = pd.read_parquet(f).set_index(ID_COL)[name].dropna()
        te = folds[folds["split"] == "test"]
        n_te_obs = te[ID_COL].isin(obs.index).sum()
        print(f"  PG coverage {name}: {n_te_obs}/{n_test} filas de test tienen la faceta observada")

    out_path = derived / "cv_folds_unified_llto.parquet"
    folds.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}  ({len(folds):,} filas, scheme={SCHEME!r})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--min-test", type=int, default=5)
    build(p.parse_args())


if __name__ == "__main__":
    main()
