#!/usr/bin/env python3
"""Extrae especie -> habito del Catalogo de las plantas vasculares de Chile
(Rodriguez et al. 2018, Gayana Bot. 75(1): 1-430), convertido a markdown.

El catalogo declara su propio esquema: "datos de Habito (Arbol, Arbol pequeno, Arbusto,
Subarbusto, Hierba), Ciclo de vida (anual, bienal, perenne), Estatus (endemico, nativo,
introducido)...". Esa es la fuente autoritativa para separar lenosas de herbaceas en el
pool unificado, en vez de determinacion a ojo (ver docs/24_woody_harmonisation_spec.md).

El parseo NO usa los encabezados markdown: la conversion del PDF los dejo poco fiables.
Se ancla en el patron de la entrada de especie, `_**Genero especie**_ Autor`, que si es
regular, y lee el habito del bloque de texto que sigue hasta la entrada siguiente.

Trampa evitada: "Hierba mora", "Hierba loca", "Hierba del incordio" y companía son
nombres vulgares, no habitos. Se corta el bloque en "Nombre vulgar" antes de buscar.

Uso:
    python scripts/81_parse_rodriguez_catalogue.py \
        --md /ruta/rodriguez-etal-2018-gaya-bot.md
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
#: `results/` esta en el .gitignore y este parseo tiene que viajar entre maquinas
#: (pop-os no tiene el PDF), asi que la tabla derivada va a data/derived, que si
#: esta exceptuado. Es una tabla factual especie->habito, no el articulo.
OUT = ROOT / "data" / "derived"

#: Entrada de especie aceptada: genero + epiteto en negrita-cursiva.
ENTRY = re.compile(r"_\*\*([A-ZÁÉÍÓÚÑ][a-záéíóúñ\-]+)\s+([a-záéíóúñ\-]+)\*\*_")

#: Habito al inicio de oracion. El orden importa: "Arbol pequeno" antes que "Arbol".
HABIT = re.compile(
    r"\b(Árbol pequeño|Árbol|Arbusto|Subarbusto|Hierba|Palma)"
    r"((?:\s+(?:o|u|y)?\s*[a-záéíóúñ]+){0,3}?)\s*\.",
    re.UNICODE)

#: Nombre en cursiva simple dentro del bloque de sinonimos.
SYNONYM = re.compile(r"_([A-ZÁÉÍÓÚÑ][a-záéíóúñ\-]+)\s+([a-záéíóúñ\-]+)_")

#: Subarbusto cuenta como lenosa: los censos de lenosas del pool los registran
#: (Baccharis, Haplopappus, Euphorbia collina, Tetraglochin).
WOODY_HEAD = {"Árbol pequeño", "Árbol", "Arbusto", "Subarbusto", "Palma"}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower().strip())


def classify(head: str, tail: str) -> str:
    """lenosa / herbacea / ambiguo, a partir de la cabeza del habito y sus calificadores."""
    tail_l = tail.lower()
    if head == "Hierba":
        return "ambiguo" if "subarbusto" in tail_l else "herbacea"
    if head in WOODY_HEAD:
        return "lenosa"
    return "ambiguo"


def parse(md_path: Path) -> pd.DataFrame:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    hits = list(ENTRY.finditer(text))
    rows = []
    for i, m in enumerate(hits):
        genus, epithet = m.group(1), m.group(2)
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        block = text[m.end():end]
        # los nombres vulgares contienen "Hierba mora", "Hierba loca", etc.
        block = re.split(r"Nombre[s]? vulgar", block)[0]
        h = HABIT.search(block)
        if not h:
            continue
        head, tail = h.group(1), (h.group(2) or "").strip()
        accepted = f"{genus} {epithet}"
        habito = (head + (" " + tail if tail else "")).strip()
        forma = classify(head, tail)
        estatus = ("introducida" if re.search(r"\bIntroducid", block) else
                   "endemica" if re.search(r"\bEndémic", block) else
                   "nativa" if re.search(r"\bNativ", block) else "")
        rows.append(dict(species=accepted, nombre_normalizado=normalize(accepted),
                         habito_raw=habito, forma=forma, estatus=estatus,
                         es_sinonimo=False, aceptado=accepted))
        # El catalogo lista los sinonimos de cada taxon. La base de parcelas usa varios
        # de ellos (Lithraea/Lithrea, Vachellia/Acacia, Neltuma/Prosopis), asi que se
        # indexan como entradas propias que heredan el habito del nombre aceptado.
        for syn in SYNONYM.finditer(block[:h.start()]):
            sname = f"{syn.group(1)} {syn.group(2)}"
            if sname == accepted:
                continue
            rows.append(dict(species=sname, nombre_normalizado=normalize(sname),
                             habito_raw=habito, forma=forma, estatus=estatus,
                             es_sinonimo=True, aceptado=accepted))
    df = pd.DataFrame(rows)
    # Un taxon puede aparecer mas de una vez (infraespecificos). Se queda el primero,
    # y se marca si sus repeticiones discrepan en forma de crecimiento.
    disagree = (df.groupby("nombre_normalizado").forma.nunique() > 1)
    df["forma_inconsistente"] = df.nombre_normalizado.map(disagree).fillna(False)
    df = df.sort_values("es_sinonimo", kind="stable")
    return df.drop_duplicates("nombre_normalizado", keep="first").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, type=Path)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cat = parse(args.md)
    f = OUT / "rodriguez2018_habitos.csv"
    cat.to_csv(f, index=False)
    print(f"-> {f}  ({len(cat)} taxones con habito)")
    print(f"   de ellos {int(cat.es_sinonimo.sum())} son sinonimos indexados")
    print("\nreparto por forma de crecimiento:")
    print(cat.forma.value_counts().to_string())
    print("\nhabitos crudos mas frecuentes:")
    print(cat.habito_raw.value_counts().head(12).to_string())
    print(f"\ninconsistentes entre repeticiones: {int(cat.forma_inconsistente.sum())}")


if __name__ == "__main__":
    main()
