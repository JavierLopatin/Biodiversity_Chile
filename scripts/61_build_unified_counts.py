#!/usr/bin/env python3
"""Build the abundance-only (true individual counts) unified occurrence table.

`occurrences_unified.parquet` mixes 3 incommensurable currencies (Cover, Basal_area,
Abundance). This script pulls out ONLY the subset with genuine individual counts:

- Parcelas-CL: rows already tagged `Abundance_parameter=="Abundance"` in
  `occurrences_unified.parquet` (its `Value` is a real count, confirmed against the raw
  contributor files this session -- Cover/Basal_area strata have no per-individual data
  recoverable anywhere in `data/Parcelas_CL_RAW/`).
- Living Trees: `living_trees_long_counts.parquet` (built by `scripts/50_build_living_trees.py`,
  counting tree-level rows per species x plot -- distinct from its existing
  `Abundance_parameter="Basal_area"` output).

Output is additive: does not touch `occurrences_unified.parquet` or any of the facets
already computed on the full pool (`hill_q0_unified`, `lcbd_*_unified`, etc, see
`docs/19_unified_facets_methodology.md`). This is a smaller, count-only companion table
for facets that specifically require true individual counts (coverage-based / Chao1-type
estimators, e.g. iNEXT.3D) -- see `scripts/62_lcbd_sorensen_pg.R` and
`scripts/63_pd_inext_coverage.R`.

Usage:
    python scripts/61_build_unified_counts.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")


def main() -> None:
    occ = pd.read_parquet(DERIVED / "occurrences_unified.parquet")
    pcl = occ[occ["Abundance_parameter"] == "Abundance"].copy()

    lt = pd.read_parquet(DERIVED / "living_trees_long_counts.parquet")

    unified = pd.concat([pcl, lt], ignore_index=True)
    unified.to_parquet(DERIVED / "occurrences_unified_counts.parquet", index=False)

    print("=== base unificada de conteo real ===")
    for name, sub in [("parcelas_cl", pcl), ("living_trees", lt), ("total", unified)]:
        n_plots = sub["PlotObservationID"].nunique()
        n_sp = sub["species"].nunique()
        rich = sub.groupby("PlotObservationID")["species"].nunique()
        print(f"{name:14s} -- {len(sub):5d} filas, {n_plots:5d} parcelas, {n_sp:4d} especies "
              f"| riqueza media {rich.mean():5.2f}, mediana {rich.median():.0f}, "
              f">=5 spp: {(rich >= 5).sum()}/{n_plots} ({100 * (rich >= 5).mean():.1f}%)")

    print(f"\n-> {DERIVED / 'occurrences_unified_counts.parquet'}")


if __name__ == "__main__":
    main()
