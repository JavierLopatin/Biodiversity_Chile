"""Pad the Perez-Giraldo-style facets to the full 3,102-plot unified pool.

`lcbd_count_sorensen.parquet` (2,499 rows), `pd_inext_coverage.parquet` (888 rows) and
`td_inext_coverage.parquet` (895 rows) only carry rows for plots that cleared the
real-count / species-floor requirements of their own estimator (script outputs from the
Perez-Giraldo-style pipeline, see docs plan). Every other unified facet parquet
(`unified_diversity_responses.parquet` etc.) already ships one row per plot in
`plots_unified.parquet`, with NaN where the underlying R package itself declined to
estimate -- that is the masking contract `src/biodiv/targets.py:load_targets` relies on
(it raises `KeyError` on a plot id genuinely absent from a source's index, on the
assumption that means a real alignment bug, not partial coverage by design).

This script reindexes the three PG-style parquets onto the full unified id list so they
carry the same contract: NaN, not a missing row, where the estimator has no value.
Non-destructive -- writes new `*_unified_padded.parquet` files, leaves the originals
(diagnostic, partial-N) untouched.
"""

from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"

SOURCES = {
    "lcbd_count_sorensen.parquet": "lcbd_count_sorensen_unified_padded.parquet",
    "pd_inext_coverage.parquet": "pd_inext_coverage_unified_padded.parquet",
    "td_inext_coverage.parquet": "td_inext_coverage_unified_padded.parquet",
}


def main() -> None:
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")
    all_ids = plots[ID_COL].astype(str)
    assert all_ids.is_unique
    full_index = pd.Index(all_ids, name=ID_COL)

    for src_name, out_name in SOURCES.items():
        src = DERIVED / src_name
        df = pd.read_parquet(src).set_index(ID_COL)
        df.index = df.index.astype(str)
        n_before = len(df)
        padded = df.reindex(full_index)
        n_after = padded.notna().any(axis=1).sum()
        assert n_after == n_before, f"{src_name}: lost rows on reindex ({n_before} -> {n_after})"
        out = DERIVED / out_name
        padded.reset_index().to_parquet(out, index=False)
        pcl = padded.index.str.startswith("PCL_").sum()
        lt = padded.index.str.startswith("LT_").sum()
        print(f"{src_name}: {n_before} observed rows padded to {len(padded)} "
              f"(full unified pool) -> {out_name}  [pool: PCL={pcl} LT={lt}]")


if __name__ == "__main__":
    main()
