"""Variante con hierbas restringida a las parcelas que también llevan la faceta leñosa.

El umbral de >=5 especies de iNEXT (scripts/63, 64) deja afuera en la variante leñosa
parcelas que con hierbas sí pasaban: TD 895 -> 881, PD 888 -> 877. Las que caen son sobre
todo parcelas de Becerra con riqueza alta puramente herbácea, así que un R² con hierbas
sobre 895 contra uno leñoso sobre 881 mezcla el cambio del target con la salida de esas
parcelas (docs/24).

Este script enmascara a NaN, en los padded con hierbas, cada valor cuya parcela no lleva
esa columna en la variante leñosa, y escribe *_unified_padded_herbcommon.parquet. Con
BIODIV_TARGETS=_herbcommon el modelo con hierbas se ajusta y evalúa sobre exactamente las
mismas parcelas que el leñoso.

Uso:
    python scripts/84_common_plot_targets.py
"""

from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"
STEMS = ["lcbd_count_sorensen_unified_padded", "pd_inext_coverage_unified_padded",
         "td_inext_coverage_unified_padded"]


def main() -> None:
    for stem in STEMS:
        herb = pd.read_parquet(DERIVED / f"{stem}.parquet").set_index(ID_COL)
        woody = pd.read_parquet(DERIVED / f"{stem}_woody.parquet").set_index(ID_COL)
        assert herb.index.equals(woody.index), stem
        out = herb.where(woody.notna())
        for c in herb.columns:
            extra = int((woody[c].notna() & herb[c].isna()).sum())
            print(f"{stem}.{c}: con hierbas {herb[c].notna().sum()}, leñosa "
                  f"{woody[c].notna().sum()}, común {out[c].notna().sum()}"
                  + (f"  (AVISO: {extra} solo en la leñosa)" if extra else ""))
        f = DERIVED / f"{stem}_herbcommon.parquet"
        out.reset_index().to_parquet(f, index=False)
        print(f"-> {f}")


if __name__ == "__main__":
    main()
