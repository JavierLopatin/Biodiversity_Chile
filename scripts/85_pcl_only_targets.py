"""Targets PG leñosos restringidos a Parcelas-CL: *_unified_padded_woodypcl.parquet.

Para la compuerta espectral (RFG4-6): cube_predictors.parquet solo cubre Parcelas-CL, así
que sobre el pool mixto el bloque gm llega imputado con la mediana del fold en 3 de cada 4
filas y funciona como etiqueta de fuente. Enmascarando los targets de Living Trees a NaN,
todas las especificaciones se ajustan y evalúan sobre las mismas parcelas de Parcelas-CL
con gm completo, con los mismos folds. Se selecciona con BIODIV_TARGETS=_woodypcl.

`--own-lcbd` reemplaza el LCBD por el recalculado sobre la matriz de las parcelas de
Parcelas-CL solas (`62 --woody --pcl`) y escribe *_woodypclown: el LCBD de la variante
_woodypcl es la unicidad de cada parcela PCL contra un pool que incluye 2.020 de Living
Trees, evaluada solo en las filas PCL. TD y PD son por parcela y no cambian.

Uso:
    python scripts/85_pcl_only_targets.py
    python scripts/85_pcl_only_targets.py --own-lcbd
"""

import sys
from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"
STEMS = ["lcbd_count_sorensen_unified_padded", "pd_inext_coverage_unified_padded",
         "td_inext_coverage_unified_padded"]


def main() -> None:
    own = "--own-lcbd" in sys.argv[1:]
    tag = "_woodypclown" if own else "_woodypcl"
    for stem in STEMS:
        df = pd.read_parquet(DERIVED / f"{stem}_woody.parquet").set_index(ID_COL)
        out = df.copy()
        out.loc[~df.index.str.startswith("PCL_")] = float("nan")
        if own and stem.startswith("lcbd"):
            pcl = pd.read_parquet(DERIVED / "lcbd_count_sorensen_woody_pcl.parquet")
            pcl = pcl.set_index(ID_COL).reindex(out.index)
            out[pcl.columns] = pcl
            assert out.iloc[:, 0].notna().sum() == len(pd.read_parquet(
                DERIVED / "lcbd_count_sorensen_woody_pcl.parquet"))
        c = df.columns[0]
        print(f"{stem}.{c}: leñosa {df[c].notna().sum()} -> solo PCL {out[c].notna().sum()}")
        f = DERIVED / f"{stem}{tag}.parquet"
        out.reset_index().to_parquet(f, index=False)
        print(f"-> {f}")


if __name__ == "__main__":
    main()
