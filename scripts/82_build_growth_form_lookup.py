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

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def normalize(name: str) -> str:
    """Igual que scripts/80: sin tildes, sin epiteto infraespecifico, sin hibridos."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+(var|subsp|ssp|f)\.\s+\S+", "", s)
    s = re.sub(r"\s+x\s+", " ", s)
    return re.sub(r"\s+", " ", s.lower().strip())

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


#: Union de las tres fuentes de nombres que el pipeline toca. El padron de scripts/80
#: sale de `occurrences_unified.parquet` (593), pero `scripts/55` (dark diversity) arma el
#: pool de co-ocurrencia desde el zip completo de Parcelas-CL, que trae 675 `Accepted_species`
#: -- 93 de ellos ausentes del parquet. Sin esos, el filtro lenoso sesgaria el pool de
#: co-ocurrencia contra especies que el catalogo si resuelve.
ZIP_CSV = ROOT / "20602096" / "Parcelas_CL.csv"


def extra_species() -> pd.DataFrame:
    """Nombres del zip de Parcelas-CL y de living_trees que no estan en el padron."""
    extra = set()
    if ZIP_CSV.exists():
        z = pd.read_csv(ZIP_CSV, low_memory=False, usecols=["Accepted_species"])
        extra |= set(z.Accepted_species.dropna().astype(str))
    lt = ROOT / "data" / "derived" / "living_trees_long.parquet"
    if lt.exists():
        extra |= set(pd.read_parquet(lt).species.astype(str))
    return pd.DataFrame({"species": sorted(extra)})


def assign_from_catalogue(names: pd.Series) -> pd.DataFrame:
    """Mismo cruce de tres pasadas que scripts/80: nombre aceptado, sinonimo, genero unanime."""
    cat = pd.read_csv(ROOT / "data" / "derived" / "rodriguez2018_habitos.csv")
    cat["nn"] = cat.species.map(normalize)
    lut = cat.drop_duplicates("nn").set_index("nn").forma
    cat["gen"] = cat.nn.str.split().str[0]
    unan = cat.groupby("gen").forma.agg(["nunique", "first"])
    unan = unan[unan["nunique"] == 1]["first"]

    nn = names.map(normalize)
    forma = nn.map(lut)
    via = pd.Series(np.where(forma.notna(), "especie", ""), index=names.index)
    falta = forma.isna()
    heredado = nn[falta].str.split().str[0].map(unan)
    forma[falta] = heredado
    via[falta & heredado.notna()] = "genero"
    return pd.DataFrame({"forma": forma, "via": via})


def main() -> None:
    src = ROOT / "results" / "taxa" / "taxa_roster_vs_catalogo.csv"
    if not src.exists():
        raise SystemExit(f"falta {src}; corre antes scripts/80 con --catalog")
    j = pd.read_csv(src)

    out = j[["species", "family", "becerra", "parcelas_cl_resto", "living_trees",
             "n_parcelas", "en_conteos", "forma", "via"]].copy()

    # Filas nuevas: los nombres del zip y de living_trees ausentes del padron.
    nuevos = extra_species()
    nuevos = nuevos[~nuevos.species.isin(out.species)].reset_index(drop=True)
    if len(nuevos):
        asign = assign_from_catalogue(nuevos.species)
        nuevos = pd.concat([nuevos, asign], axis=1)
        for col in ("family", "becerra", "parcelas_cl_resto", "living_trees", "n_parcelas"):
            nuevos[col] = 0 if col != "family" else pd.NA
        nuevos["en_conteos"] = False
        out = pd.concat([out, nuevos[out.columns]], ignore_index=True)

    out = out.rename(columns={"via": "fuente"})
    out["fuente"] = out.fuente.fillna("").replace("", "sin_resolver")
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

    # Motivo por defecto de los que no resuelven, para que el log del filtro diga por que
    # se descarta cada uno en vez de dejarlos mudos. Los nombres de rango familiar y los
    # taxones ausentes de la flora chilena son errores de emparejamiento de la base fuente,
    # no del cruce: Decarydendron es de Madagascar, Diospyros venosa y D. piscatoria del
    # tropico asiatico. Se registran, no se corrigen.
    sin = out.fuente == "sin_resolver"
    vacio = sin & (out.motivo == "")
    es_familia = out.species.astype(str).str.endswith(("aceae", "Aceae"))
    un_termino = out.species.astype(str).str.split().str.len() == 1
    out.loc[vacio & es_familia, "motivo"] = "no es un taxon: nombre de rango familiar"
    out.loc[vacio & un_termino & ~es_familia, "motivo"] = (
        "solo genero; el catalogo lo registra con mas de una forma")
    out.loc[vacio & (out.motivo == ""), "motivo"] = (
        "ausente del catalogo de flora vascular de Chile (Rodriguez et al. 2018)")

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
