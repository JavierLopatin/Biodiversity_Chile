"""Targets PG leñosos restringidos a Parcelas-CL: *_unified_padded_woodypcl.parquet.

Para la compuerta espectral (RFG4-6): cube_predictors.parquet solo cubre Parcelas-CL, así
que sobre el pool mixto el bloque gm llega imputado con la mediana del fold en 3 de cada 4
filas y funciona como etiqueta de fuente. Enmascarando los targets de Living Trees a NaN,
todas las especificaciones se ajustan y evalúan sobre las mismas parcelas de Parcelas-CL
con gm completo, con los mismos folds. Se selecciona con BIODIV_TARGETS=_woodypcl.

Uso:
    python scripts/85_pcl_only_targets.py
"""

from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"
STEMS = ["lcbd_count_sorensen_unified_padded", "pd_inext_coverage_unified_padded",
         "td_inext_coverage_unified_padded"]


def main() -> None:
    for stem in STEMS:
        df = pd.read_parquet(DERIVED / f"{stem}_woody.parquet").set_index(ID_COL)
        out = df.copy()
        out.loc[~df.index.str.startswith("PCL_")] = float("nan")
        c = df.columns[0]
        print(f"{stem}.{c}: leñosa {df[c].notna().sum()} -> solo PCL {out[c].notna().sum()}")
        f = DERIVED / f"{stem}_woodypcl.parquet"
        out.reset_index().to_parquet(f, index=False)
        print(f"-> {f}")


if __name__ == "__main__":
    main()
