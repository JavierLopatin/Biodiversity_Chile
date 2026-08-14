#!/usr/bin/env python3
"""Recupera la cobertura real de Zamorano-Elgueta, C. perdida en la curacion de Parcelas-CL.

Hallazgo (sesion de revision de `data/Parcelas_CL_RAW/`): para las 84 parcelas de este
dueno, `Parcelas_CL.csv` tiene `Abundance_parameter=="Cover"` pero `Value` fijo en 1.0 para
TODA especie de TODA parcela (nunique==1) -- la curacion perdio la cobertura real, aunque el
metadato del dueno confirma "estimacion visual de la cobertura" y el archivo crudo
(`Zamorano_data_2013.csv`) SI trae el gradiente real (0-90, multiplos de 10 mayormente).
Efecto practico: hoy estas 84 parcelas aportan a cualquier analisis "de frecuencia"
(`beta_freq`, particion de Hill por abundancia) un vector de pesos plano -- cada especie con
peso 1/riqueza, indistinguible de presencia/ausencia disfrazada de cobertura.

Se comparo fidelidad raw-vs-curado para las 8 fuentes en `data/Parcelas_CL_RAW/` antes de
tocar nada (join por Location+X/Y, no por nombre suelto, para evitar falsos matches):
Altamirano, MirandaDobbs, Miranda_2010, Morales, Ovalle -- 100% identico al crudo, sin
ganancia. Zamorano -- 0% identico (100% perdido). Becerra -- hojas de transecto por punto,
formato con encabezado partido en 2 filas y celdas con tipos mezclados (int/float/str) sin
una columna de PlotObservationID limpia; se descarta por ahora, no vale la pena arriesgar
una mala lectura silenciosa sobre un archivo asi de sucio. Venegas ya es puramente
presencia/ausencia en el crudo (no aplica). Miranda_2017 usa lat/lon en vez de UTM y no se
verifico (bajo impacto, ~21 parcelas, mismo dueno/protocolo que Miranda_2010 que si fue
fiel) -- no se toca en este script.

Salida: `data/derived/zamorano_cover_corrected.parquet`, columnas
(PlotObservationID, Accepted_species, Value_corrected) -- listo para hacer un merge-override
sobre `Value` donde `Owner=="Zamorano-Elgueta, C."` en cualquier loader corriente aguas abajo
(`scripts/51_build_unified_dataset.py`, `scripts/lib/parcelas_comm.R` via un puente).

Uso:
    python scripts/58_recover_zamorano_cover.py
"""

from __future__ import annotations

import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biodiv import io_parcelas  # noqa: E402

ZIP = "data/20602096.zip"
RAW_CSV = "data/Parcelas_CL_RAW/Zamorano_data_2013.csv"
OUT = "data/derived/zamorano_cover_corrected.parquet"
OWNER = "Zamorano-Elgueta, C."


def main() -> None:
    long = io_parcelas.load_long(ZIP)
    sub = long[long["Owner"] == OWNER].copy()
    assert sub["Value"].nunique() == 1 and sub["Value"].iloc[0] == 1.0, (
        "el bug de curacion que motiva este script ya no esta presente -- "
        "revisar si Parcelas_CL.csv cambio antes de aplicar esta correccion"
    )

    raw = pd.read_csv(RAW_CSV)
    raw.columns = [c.strip() for c in raw.columns]
    sp_cols = [c for c in raw.columns if c not in ("X", "Y", "Sitio")]
    raw["raw_row"] = raw.index
    raw_long = raw.melt(id_vars=["X", "Y", "Sitio", "raw_row"], value_vars=sp_cols,
                        var_name="sp_us", value_name="raw_val")
    raw_long = raw_long[raw_long["raw_val"].notna() & (raw_long["raw_val"] != 0)].copy()
    raw_long["sp_us"] = raw_long["sp_us"].str.replace(" ", "_")
    raw_species_by_row = raw_long.groupby("raw_row")["sp_us"].apply(set)

    # "Achibueno_PL" (raw) <-> "Achibueno" (curado): el sufijo de subconjunto de estudio
    # (_PL/_CM/_EG: Persea lingue / Citronella mucronata / Eucryphia glutinosa) no esta en
    # `Location`. Un solo prefijo "RN_" (reserva nacional) tambien se pierde en curado
    # (RN_Malleco_PL -> "Malleco"). Con ambos recortes, varios nombres de sitio quedan
    # compartidos por 2-3 filas crudas -- se resuelven mas abajo por solapamiento de
    # especies, no por posicion ni por conteo.
    base = raw["Sitio"].str.replace(r"^RN_", "", regex=True).str.replace(
        r"_[A-Z]+$", "", regex=True)
    base = base.replace({"Los_Placeres": "Los_Paceres",  # error tipografico en curado
                         "Los_Queñes": "Queñes"})  # prefijo "Los_" perdido en curado

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return s.lower()

    raw_base_by_row = base.map(norm)

    sub["sp_us"] = sub["Original_species_name"].str.replace(" ", "_")
    plot_species = sub.groupby("PlotObservationID")["sp_us"].apply(set)
    plot_locations_check = sub[["PlotObservationID", "Location"]].drop_duplicates()
    assert plot_locations_check["PlotObservationID"].is_unique
    plot_lookup = sub[["PlotObservationID", "Location", "Accepted_species", "sp_us"]]
    plot_locations = sub[["PlotObservationID", "Location"]].drop_duplicates().set_index(
        "PlotObservationID")["Location"]

    # asignacion parcela-crudo por solapamiento de especies dentro de cada grupo de
    # `Location` (1 a 3 candidatos crudos comparten nombre de sitio tras el recorte de
    # arriba) -- se resuelve de forma golosa, del par con mayor Jaccard hacia abajo, para
    # que ninguna fila cruda se use dos veces.
    plot_to_row: dict[str, int] = {}
    for loc, plots_here in plot_locations.groupby(plot_locations.map(norm)):
        cand_rows = raw_base_by_row[raw_base_by_row == loc].index.tolist()
        pairs = []
        for pid in plots_here.index:
            for r in cand_rows:
                cs, rs = plot_species.get(pid, set()), raw_species_by_row.get(r, set())
                jac = len(cs & rs) / len(cs | rs) if (cs | rs) else 0.0
                pairs.append((jac, pid, r))
        pairs.sort(reverse=True)
        used_pid, used_row = set(), set()
        for jac, pid, r in pairs:
            if pid in used_pid or r in used_row:
                continue
            plot_to_row[pid] = r
            used_pid.add(pid)
            used_row.add(r)

    plot_lookup = plot_lookup.copy()
    plot_lookup["raw_row"] = plot_lookup["PlotObservationID"].map(plot_to_row)
    merged = plot_lookup.merge(
        raw_long[["raw_row", "sp_us", "raw_val"]], on=["raw_row", "sp_us"], how="left")

    n_missing = merged["raw_val"].isna().sum()
    print(f"parcelas Zamorano: {plot_lookup['PlotObservationID'].nunique()}")
    print(f"parcelas sin fila cruda asignada: "
          f"{plot_lookup['PlotObservationID'].nunique() - len(plot_to_row)}")
    print(f"pares parcela-especie en curado: {len(merged)}")
    print(f"sin match en crudo: {n_missing} ({100 * n_missing / len(merged):.1f}%)")
    # Estos son pares presentes en el CSV curado (Value=1, luego se sabe que la especie
    # esta) pero sin celda no-cero en el crudo bajo ese nombre de columna exacto -- lo mas
    # probable es un nombre de especie que cambio entre el crudo y `Accepted_species`
    # (sinonimia resuelta en la curacion), o una parcela sin fila cruda asignada. Se
    # conserva Value=1 (el piso ya existente) para esos, no se inventa una cobertura.
    merged["Value_corrected"] = merged["raw_val"].fillna(1.0)

    used_rows = set(plot_to_row.values())
    n_raw_rows_unused = raw_species_by_row.index.difference(used_rows).shape[0]
    print(f"filas crudas de Zamorano sin parcela curada asignada: {n_raw_rows_unused} "
          f"de {len(raw)}")

    out = merged[["PlotObservationID", "Accepted_species", "Value_corrected"]]
    out.to_parquet(OUT, index=False)
    print(f"\nvalores no ya en 1.0 recuperados: {(out['Value_corrected'] != 1.0).sum()} "
          f"de {len(out)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
