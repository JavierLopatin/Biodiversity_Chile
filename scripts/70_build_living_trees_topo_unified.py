#!/usr/bin/env python3
"""Unify topography for the 3,102-plot pool now that Living Trees' DEM extraction exists
(`data/derived/living_trees/topography/topography.parquet`, 28 columns, same schema as Parcelas-CL's
`data/derived/topography/topography.parquet` -- same variable names, same `_mean`/`_std`
patch-statistic suffixes).

Crosswalk: `site_id` (`sites.parquet`, satellite/topo side) <-> `PlotObservationID`
(`living_trees_plots.parquet`, species side) via exact lat/lon match -- same join already
verified and used by `scripts/67_build_living_trees_curves.py`.

Output is additive: `data/derived/topography_unified.parquet`, indexed by the same
`PlotObservationID` scheme as `plots_unified.parquet` (`PCL_`-prefixed original ids +
`LT_`-prefixed Living Trees ids). The Parcelas-CL-only `topography/topography.parquet` is
never touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
LT_DIR = DERIVED / "living_trees"

#: `scripts/03 --out-dir data/derived/living_trees/topography` writes into a subdirectory, the
#: layout `docs/16` documents. An earlier hand-copy of the same table (same md5) sits flattened
#: beside `sites.parquet` on another machine, so both are accepted and the canonical one wins.
LT_TOPO_PATHS = (LT_DIR / "topography" / "topography.parquet",
                 LT_DIR / "topography.parquet")


def lt_topo_path() -> Path:
    for p in LT_TOPO_PATHS:
        if p.exists():
            return p
    raise SystemExit("no se encontro la topografia de Living Trees en "
                     + " ni ".join(str(p) for p in LT_TOPO_PATHS))


def build_crosswalk() -> pd.DataFrame:
    lt_plots = pd.read_parquet(DERIVED / "living_trees_plots.parquet")
    sites = pd.read_parquet(LT_DIR / "sites.parquet")
    cw = lt_plots.merge(sites[["site_id", "lat", "lon"]], on=["lat", "lon"], how="left")
    n_missing = cw["site_id"].isna().sum()
    if n_missing:
        raise SystemExit(f"{n_missing} Living Trees plots sin site_id -- revisar antes de seguir")
    return cw[["PlotObservationID", "site_id"]]


def main() -> None:
    cw = build_crosswalk()

    lt_topo = pd.read_parquet(lt_topo_path())
    lt_topo = cw.merge(lt_topo, on="site_id", how="inner")  # drops the 1 species-less site
    lt_topo = lt_topo.drop(columns=["site_id"]).rename(columns={"PlotObservationID": "plot_id"})

    pcl_topo = pd.read_parquet(DERIVED / "topography" / "topography.parquet").copy()
    pcl_topo["plot_id"] = "PCL_" + pcl_topo["plot_id"].astype(str)

    assert list(lt_topo.columns) == list(pcl_topo.columns), (
        f"schema mismatch: LT {list(lt_topo.columns)} vs PCL {list(pcl_topo.columns)}")

    unified = pd.concat([pcl_topo, lt_topo], ignore_index=True)
    assert unified["plot_id"].is_unique

    out_path = DERIVED / "topography_unified.parquet"
    unified.to_parquet(out_path, index=False)

    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")
    all_ids = set(plots["PlotObservationID"].astype(str))
    missing = all_ids - set(unified["plot_id"])
    print(f"Parcelas-CL: {len(pcl_topo)} parcelas")
    print(f"Living Trees: {len(lt_topo)} parcelas")
    print(f"-> {out_path} ({len(unified)} filas)")
    print(f"unified plots sin topografia: {len(missing)} (esperado: 1, el sitio "
          f"all-unidentified-species ya diagnosticado)")


if __name__ == "__main__":
    main()
