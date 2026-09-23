#!/usr/bin/env python3
"""Padron de taxones de la base unificada, para cruzarlo contra un catalogo oficial de
flora vascular de Chile y decidir la forma de crecimiento de cada uno.

Motivo: las 60 parcelas de Becerra (Coquimbo) son el unico censo de flora completa del
pool; las otras 835 son censos de lenosas. Mezclar ambos targets es lo que sostiene el
R2 de TD0. Para armonizarlos hace falta una asignacion de forma de crecimiento que no
dependa de determinacion a ojo, y para eso el catalogo oficial es la fuente.

Sin `--catalog` exporta el padron con los conteos y las banderas de nomenclatura que
romperian el cruce. Con `--catalog` ademas reporta que taxones casan y cuales no.

Uso:
    python scripts/80_growth_form_audit.py
    python scripts/80_growth_form_audit.py --catalog ruta/al/catalogo.csv \
        --catalog-name-col nombre --catalog-habit-col habito
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "results" / "taxa"

#: Pares que designan el mismo taxon escrito de dos formas. Verificado que ninguno de
#: los dos miembros coexiste con el otro en una misma parcela, asi que fusionarlos no
#: altera la riqueza por parcela, solo el conteo del pool y las puntas del arbol.
SINONIMOS = {
    "Podocarpus saligna": "Podocarpus salignus",
    "Schinus polygama": "Schinus polygamus",
    "Nothofagus x leonii": "Nothofagus leonii",
    "Nothofagus antárctica": "Nothofagus antarctica",
}

#: Registros que no son un taxon utilizable.
NO_TAXON = {"Myrtaceae"}


def normalize(name: str) -> str:
    """Minusculas sin tildes y sin epiteto infraespecifico, para cruzar con el catalogo."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+(var|subsp|ssp|f)\.\s+\S+", "", s)
    s = re.sub(r"\s+x\s+", " ", s)
    return re.sub(r"\s+", " ", s.lower().strip())


def build_roster() -> pd.DataFrame:
    # Superconjunto, no solo el subconjunto de conteos reales: las curvas de acumulacion
    # (scripts/56, 59) arman la comunidad desde Parcelas-CL completo mas living_trees, o
    # sea del estrato de cobertura tambien. Un lookup que solo cubriera las 261 especies
    # con conteo dejaria sin asignar las ~330 que solo aparecen en cobertura.
    occ = pd.read_parquet(DERIVED / "occurrences_unified.parquet")
    conteos = set(pd.read_parquet(DERIVED / "occurrences_unified_counts.parquet").species)
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "Owner", "source"]]
    fam = pd.read_csv(DERIVED / "phylo_species_status_unified.csv")[["species", "family"]]

    occ = occ.merge(plots, on="PlotObservationID", how="left").merge(fam, on="species", how="left")
    occ["grupo"] = np.where(occ.source == "living_trees", "living_trees",
                    np.where(occ.Owner == "Becerra, P.I.", "becerra", "parcelas_cl_resto"))

    wide = (occ.groupby(["species", "grupo"]).PlotObservationID.nunique()
               .unstack(fill_value=0).reset_index())
    for col in ("becerra", "parcelas_cl_resto", "living_trees"):
        if col not in wide:
            wide[col] = 0
    wide["n_parcelas"] = wide[["becerra", "parcelas_cl_resto", "living_trees"]].sum(axis=1)
    wide = wide.merge(fam, on="species", how="left")

    wide["en_conteos"] = wide.species.isin(conteos)
    wide["nombre_normalizado"] = wide.species.map(normalize)
    wide["solo_genero"] = wide.species.str.split().str.len() == 1
    wide["infraespecifico"] = wide.species.str.contains(r" (?:var|subsp|ssp|f)\. ", regex=True)
    wide["no_ascii"] = wide.species != wide.species.map(
        lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())
    wide["sinonimo_de"] = wide.species.map(SINONIMOS).fillna("")
    wide["no_es_taxon"] = wide.species.isin(NO_TAXON)
    #: solo en Becerra y en ninguna otra base: candidato a herbacea
    wide["exclusivo_becerra"] = (wide.becerra > 0) & (wide.n_parcelas == wide.becerra)

    return wide.sort_values("n_parcelas", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", help="CSV o TSV del catalogo oficial de flora vascular")
    ap.add_argument("--catalog-name-col", default="species")
    ap.add_argument("--catalog-habit-col", default=None,
                    help="columna con habito/forma de crecimiento, si el catalogo la trae")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    roster = build_roster()
    f = OUT / "taxa_roster_unified.csv"
    roster.to_csv(f, index=False)
    print(f"-> {f}  ({len(roster)} taxones)")
    print(f"   solo genero {roster.solo_genero.sum()}, infraespecificos "
          f"{roster.infraespecifico.sum()}, no ASCII {roster.no_ascii.sum()}, "
          f"sinonimos {(roster.sinonimo_de != '').sum()}, no-taxon {roster.no_es_taxon.sum()}")
    print(f"   exclusivos de Becerra: {roster.exclusivo_becerra.sum()}")

    if not args.catalog:
        print("\nSin --catalog: solo se exporto el padron. Pasa el catalogo oficial para cruzar.")
        return

    sep = "\t" if Path(args.catalog).suffix.lower() in (".tsv", ".tab") else ","
    cat = pd.read_csv(args.catalog, sep=sep)
    if args.catalog_name_col not in cat.columns:
        raise SystemExit(f"--catalog-name-col '{args.catalog_name_col}' no esta en el catalogo. "
                         f"Columnas: {list(cat.columns)}")
    cat["nombre_normalizado"] = cat[args.catalog_name_col].map(normalize)
    keep = ["nombre_normalizado"] + ([args.catalog_habit_col] if args.catalog_habit_col else [])
    cat = cat[keep].drop_duplicates("nombre_normalizado")

    j = roster.merge(cat, on="nombre_normalizado", how="left", indicator=True)
    j["en_catalogo"] = j._merge == "both"
    j["via"] = np.where(j.en_catalogo, "especie", "")
    j = j.drop(columns="_merge")

    # Segunda pasada por genero. Cubre dos casos que el cruce por especie no puede:
    # los 16 registros identificados solo a nivel de genero, y las recombinaciones
    # posteriores a 2018 que la base usa y el catalogo de 2018 no conoce. Solo se
    # hereda la forma cuando TODAS las especies del genero en el catalogo coinciden
    # en ella; si el genero mezcla lenosas y hierbas, se deja sin resolver.
    if args.catalog_habit_col:
        cat_g = cat.copy()
        cat_g["genero"] = cat_g.nombre_normalizado.str.split().str[0]
        unanime = (cat_g.groupby("genero")[args.catalog_habit_col]
                        .agg(["nunique", "first", "size"]))
        unanime = unanime[unanime["nunique"] == 1]["first"]
        falta = ~j.en_catalogo
        gen = j.loc[falta, "nombre_normalizado"].str.split().str[0]
        heredado = gen.map(unanime)
        j.loc[falta, args.catalog_habit_col] = heredado
        j.loc[falta & heredado.notna(), "via"] = "genero"
        j["en_catalogo"] = j[args.catalog_habit_col].notna()
    f2 = OUT / "taxa_roster_vs_catalogo.csv"
    j.to_csv(f2, index=False)
    print(f"\n-> {f2}")
    print(f"   casan {int(j.en_catalogo.sum())} de {len(j)}; "
          f"sin casar {int((~j.en_catalogo).sum())}")
    if "via" in j:
        print("   resueltos por:", j[j.en_catalogo].via.value_counts().to_dict())
    faltan = j[~j.en_catalogo].sort_values("n_parcelas", ascending=False)
    if len(faltan):
        print("\n   sin casar (mayor a menor ocurrencia):")
        for _, r in faltan.head(40).iterrows():
            print(f"     {r.species:38s} {int(r.n_parcelas):4d} parcelas")


if __name__ == "__main__":
    main()
