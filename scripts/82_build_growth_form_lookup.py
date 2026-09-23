#!/usr/bin/env python3
"""Tabla final especie -> forma de crecimiento para el pool unificado.

Combina tres fuentes, en este orden de prioridad:

1. El catalogo de Rodriguez et al. 2018 (scripts/81), cruzado por nombre aceptado,
   por los sinonimos que el propio catalogo lista, y por herencia de genero cuando
   todas las especies del genero coinciden en forma (scripts/80). Resuelve 244 de 261.
2. Once asignaciones manuales para taxones que el catalogo de 2018 no conoce, casi
   todos recombinaciones posteriores. Cada una lleva su motivo.
3. Seis taxones que se dejan SIN RESOLVER por decision del autor (2026-09-23): cinco
   generos que el catalogo registra con las dos formas y un registro que no es un
   taxon. Sus registros no entran al filtro de lenosas, porque no se pueden confirmar
   lenosos. Para revertirlo basta moverlos a MANUAL.

Salida: data/derived/growth_form_lookup.csv, que es lo que consume el paso 1 de
docs/24_woody_harmonisation_spec.md.

Uso:
    python scripts/80_growth_form_audit.py --catalog results/taxa/rodriguez2018_habitos.csv \
        --catalog-name-col species --catalog-habit-col forma
    python scripts/82_build_growth_form_lookup.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Taxones ausentes del catalogo de 2018, con el motivo de la asignacion.
MANUAL = {
    "Pseudopanax laetevirens":   ("lenosa",   "arbol; el catalogo lo trae como Raukaua"),
    "Leucostele chiloensis":     ("lenosa",   "cactus columnar; antes Echinopsis/Trichocereus"),
    "Citronella mucronata":      ("lenosa",   "arbol; el catalogo lo trae como Villaresia"),
    "Solanum crispum":           ("lenosa",   "trepadora lignificada"),
    "Neltuma chilensis":         ("lenosa",   "arbol; antes Prosopis chilensis"),
    "Archidasyphyllum excelsum": ("lenosa",   "arbol; antes Dasyphyllum excelsum"),
    "Oziroe arida":              ("herbacea", "geofita bulbosa"),
    "Adesmia pedicellata":       ("herbacea", "anual; congenere de A. tenella en las mismas parcelas"),
    "Chaetanthera frayjorgensis":("herbacea", "anual"),
    "Hemionitis mollis":         ("herbacea", "helecho"),
    "Hemionitis hypoleuca":      ("herbacea", "helecho"),
}

#: Sin resolver por decision del autor. Motivo por el que no se fuerza.
SIN_RESOLVER = {
    "Adesmia":     "genero con arbustos y anuales; la base no identifica la especie",
    "Haplopappus": "genero con arbustos y hierbas andinas; la base no identifica la especie",
    "Spergularia": "genero mixto; la base no identifica la especie",
    "Verbena":     "genero mixto; la base no identifica la especie",
    "Atriplex":    "genero mixto; la base no identifica la especie",
    "Myrtaceae":   "no es un taxon: una familia en el campo de especie",
}


def main() -> None:
    src = ROOT / "results" / "taxa" / "taxa_roster_vs_catalogo.csv"
    if not src.exists():
        raise SystemExit(f"falta {src}; corre antes scripts/80 con --catalog")
    j = pd.read_csv(src)

    out = j[["species", "family", "becerra", "parcelas_cl_resto", "living_trees",
             "n_parcelas", "en_conteos", "forma", "via"]].copy()
    out = out.rename(columns={"via": "fuente"})
    out["motivo"] = ""

    for sp, (forma, motivo) in MANUAL.items():
        m = out.species == sp
        if not m.any():
            print(f"  aviso: MANUAL menciona '{sp}', ausente del padron; se ignora")
            continue
        out.loc[m, ["forma", "fuente", "motivo"]] = [forma, "manual", motivo]

    for sp, motivo in SIN_RESOLVER.items():
        m = out.species == sp
        if not m.any():
            print(f"  aviso: SIN_RESOLVER menciona '{sp}', ausente del padron; se ignora")
            continue
        out.loc[m, ["forma", "fuente", "motivo"]] = [pd.NA, "sin_resolver", motivo]

    f = ROOT / "data" / "derived" / "growth_form_lookup.csv"
    out.sort_values("n_parcelas", ascending=False).to_csv(f, index=False)
    print(f"-> {f}  ({len(out)} taxones)")
    print("\nfuente de la asignacion:")
    print(out.fuente.value_counts().to_string())
    print("\nforma:")
    print(out.forma.value_counts(dropna=False).to_string())

    sin = out[out.fuente == "sin_resolver"]
    print(f"\nsin resolver: {len(sin)} taxones, {int(sin.n_parcelas.sum())} apariciones en parcelas")


if __name__ == "__main__":
    main()
