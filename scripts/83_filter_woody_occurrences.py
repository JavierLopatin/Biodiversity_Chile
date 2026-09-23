"""Variante leñosa de la tabla de conteos unificada (docs/24_woody_harmonisation_spec.md).

Las 60 parcelas de Becerra, P.I. son un censo de flora vascular completa y las otras 835
son censos de leñosas. Esta variante deja solo los registros con `forma == "lenosa"` en
`data/derived/growth_form_lookup.csv` (scripts/80 -> 81 -> 82): descarta las herbáceas y
los 6 taxones sin resolver (`forma` NA). Conserva todas las parcelas; si alguna quedara
vacía, falla en vez de achicar el pool en silencio.

No corrige la nomenclatura (los tres pares de grafías, la tilde de *Nothofagus
antárctica*): la variante con hierbas tampoco la corrige, así que el contraste entre las
dos aísla el filtro de forma de crecimiento.

Uso:
    python scripts/83_filter_woody_occurrences.py
"""

from pathlib import Path

import pandas as pd

DERIVED = Path("data/derived")
ID_COL = "PlotObservationID"


def main() -> None:
    occ = pd.read_parquet(DERIVED / "occurrences_unified_counts.parquet")
    lk = pd.read_csv(DERIVED / "growth_form_lookup.csv")
    assert lk.species.is_unique

    unknown = sorted(set(occ.species) - set(lk.species))
    if unknown:
        raise SystemExit(f"{len(unknown)} taxones sin fila en el lookup: {unknown[:10]}")

    forma = occ.species.map(dict(zip(lk.species, lk.forma)))
    keep = forma.eq("lenosa")
    woody = occ[keep].reset_index(drop=True)

    lost = sorted(set(occ[ID_COL]) - set(woody[ID_COL]))
    if lost:
        raise SystemExit(f"{len(lost)} parcelas quedan sin leñosas, p. ej. {lost[:5]}")

    print(f"registros: {len(occ)} -> {len(woody)} "
          f"(herbacea {forma.eq('herbacea').sum()}, sin resolver {forma.isna().sum()})")
    print(f"taxones: {occ.species.nunique()} -> {woody.species.nunique()}")
    print(f"parcelas: {occ[ID_COL].nunique()} -> {woody[ID_COL].nunique()}")

    out = DERIVED / "occurrences_unified_counts_woody.parquet"
    woody.to_parquet(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
