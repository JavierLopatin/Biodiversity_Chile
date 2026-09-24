"""Targets por estrato para la compuerta estratificada: *_unified_padded_strat_<clave>.parquet.

Para cada estrato de scripts/87, el padded de LCBD lleva el LCBD recalculado dentro del
estrato (NaN fuera). TD y PD van leñosos y enmascarados a las parcelas del estrato; son
por parcela y no cambian, pero se reportan solo como referencia, porque la riqueza ya
quedó descartada. Se selecciona con BIODIV_TARGETS=_strat_<clave>. Imprime las claves.

Uso:
    python scripts/88_stratum_targets.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"
STEMS = ["lcbd_count_sorensen_unified_padded", "pd_inext_coverage_unified_padded",
         "td_inext_coverage_unified_padded"]


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def main() -> None:
    st = pd.read_parquet(DERIVED / "lcbd_count_sorensen_woody_by_stratum.parquet")
    for estrato, g in st.groupby("estrato"):
        key = slug(estrato)
        ids = pd.Index(g[ID_COL])
        for stem in STEMS:
            df = pd.read_parquet(DERIVED / f"{stem}_woody.parquet").set_index(ID_COL)
            out = df.copy()
            out.loc[~out.index.isin(ids)] = float("nan")
            if stem.startswith("lcbd"):
                out["lcbd_count_sorensen"] = g.set_index(ID_COL).lcbd_count_sorensen.reindex(out.index)
            out.reset_index().to_parquet(DERIVED / f"{stem}_strat_{key}.parquet", index=False)
        print(f"{key}\t{estrato}\tLCBD {len(ids)}")


if __name__ == "__main__":
    main()
