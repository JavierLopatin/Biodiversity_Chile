"""Clase MapBiomas (colección Chile) en cada una de las 3.102 parcelas del pool unificado.

Para estratificar los modelos por tipo de cobertura con una clasificación que existe como
mapa wall-to-wall (propuesta de modelos por tipo, Fassnacht et al. 2021). Usa el mapa
anual del año de censo, o el más cercano disponible (`mapbiomas.year_map`), y guarda el
desplazamiento para que quede auditable. Muestrea el píxel que contiene la coordenada.

Escribe data/derived/mapbiomas_class_unified.parquet.

Uso:
    BIODIV_MAPBIOMAS_DIR=/ruta/a/los/tif python scripts/89_mapbiomas_class_at_plots.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import mapbiomas as mb  # noqa: E402

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"


def main() -> None:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[[ID_COL, "lon", "lat", "Year", "source"]]
    out = []
    for year, g in p.groupby("Year"):
        codes, used, delta = mb.sample_at(g.lon.to_numpy(), g.lat.to_numpy(), int(year))
        out.append(g.assign(mb_code=codes, mb_year=used, mb_delta=delta))
    out = pd.concat(out).sort_values(ID_COL)
    out["mb_class"] = out.mb_code.map(mb.class_name)
    out["mb_native"] = mb.is_native(out.mb_code.to_numpy())
    f = DERIVED / "mapbiomas_class_unified.parquet"
    out[[ID_COL, "mb_code", "mb_class", "mb_native", "mb_year", "mb_delta"]].to_parquet(f, index=False)
    print(f"-> {f} ({len(out)} parcelas); |delta| de año: "
          f"{out.mb_delta.abs().value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
