#!/usr/bin/env python3
"""Figura 1: la zona de estudio sobre el pool unificado (Parcelas-CL + Living Trees).

Reemplaza la version de `scripts/06_paper_figures.py:fig_map_study_area`, que se construyo
cuando el pool era solo Parcelas-CL: sus tres paneles (proyecto fuente, ano de censo,
riqueza) cubrian 30-38S y el pool de hoy llega a 55S.

Cuatro paneles descriptivos, todo lo que un lector necesita saber sobre donde y cuando se
midio, antes de cualquier resultado:

    a  inventario          -- el corte que separa las dos fuentes, que casi no se solapan
    b  clase MapBiomas     -- la cobertura donde cae cada parcela (coleccion 2, Chile)
    c  ano de censo        -- 2003-2026, el eje que obliga al bloqueo temporal
    d  riqueza lenosa      -- armonizada con `growth_form_lookup.csv`, no el conteo crudo

El panel d usa riqueza lenosa y no la columna `richness` del parquet a proposito: el conteo
crudo incluye el estrato herbaceo, que solo un inventario de los dos registro, asi que un
mapa de `richness` dibujaria el protocolo de muestreo y no la vegetacion.

Uso:
    python scripts/93_study_area_unified.py [--out-dir results/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import figures as F  # noqa: E402

DERIVED = ROOT / "data" / "derived"

#: El pool unificado va de Atacama a Magallanes, no del extent 30-38S de `biodiv.figures`.
EXTENT = (-76.0, -65.5, -56.2, -29.5)

SOURCES = [("parcelas_cl", "#0072B2", "Parcelas-CL"),
           ("living_trees", "#D55E00", "Living Trees Chile")]

#: Las clases con soporte propio en el pool; el resto se agrupa en "Other". El orden es el
#: de la leyenda: bosques, luego lenosa no arborea, luego herbacea y plantacion.
MB_CLASSES = [
    (59, "#1B5E20", "Primary forest"),
    (60, "#4CAF50", "Secondary forest"),
    (3,  "#A5D6A7", "Forest, unspecified"),
    (66, "#BCAA5E", "Shrubland"),
    (12, "#F0D26A", "Grassland"),
    (9,  "#8E6FAF", "Silviculture"),
]
MB_OTHER = ("#9A9A9A", "Other")


def load() -> pd.DataFrame:
    plots = pd.read_parquet(DERIVED / "plots_unified.parquet")

    mb_path = DERIVED / "mapbiomas_class_unified.parquet"
    if not mb_path.exists():
        raise SystemExit(f"falta {mb_path}: correr scripts/89_mapbiomas_class_at_plots.py")
    mb = pd.read_parquet(mb_path)[["PlotObservationID", "mb_code", "mb_class"]]
    plots = plots.merge(mb, on="PlotObservationID", how="left")

    # Riqueza lenosa desde la tabla de ocurrencias, no desde `richness`. Ver el docstring.
    occ = pd.read_parquet(DERIVED / "occurrences_unified.parquet")
    forms = pd.read_csv(DERIVED / "growth_form_lookup.csv").set_index("species")["forma"]
    occ["forma"] = occ["species"].map(forms)
    woody = (occ[occ["forma"] == "lenosa"]
             .groupby("PlotObservationID")["species"].nunique()
             .rename("richness_woody"))
    plots = plots.merge(woody, on="PlotObservationID", how="left")
    return plots


def panel_categorical(ax, fig, df, key, entries, other, title, first: bool):
    """Un panel de categorias discretas: dibuja cada grupo y rotula con su n."""
    F.basemap(ax, extent=EXTENT, label_axes=False)
    shown = {e[0] for e in entries}
    for val, color, lab in entries:
        s = df[df[key] == val]
        if s.empty:
            continue
        ax.scatter(s["lon"], s["lat"], s=3.5, lw=0, color=color, alpha=0.75,
                   label=f"{lab} ({len(s)})")
    rest = df[~df[key].isin(shown) & df[key].notna()]
    if not rest.empty:
        ax.scatter(rest["lon"], rest["lat"], s=3.5, lw=0, color=other[0], alpha=0.75,
                   label=f"{other[1]} ({len(rest)})")
    ax.legend(loc="upper left", fontsize=6.2, markerscale=2.6, framealpha=0.92,
              handletextpad=0.35, borderpad=0.35, labelspacing=0.3)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel("Longitude (°)")
    if first:
        ax.set_ylabel("Latitude (°)")


def panel_continuous(ax, fig, df, col, cmap, label, title, ticks=None, ticklabels=None):
    F.basemap(ax, extent=EXTENT, label_axes=False)
    s = df[df[col].notna()]
    h = ax.scatter(s["lon"], s["lat"], c=s[col], s=3.5, lw=0, cmap=cmap)
    cb = fig.colorbar(h, ax=ax, shrink=0.42, pad=0.02)
    cb.set_label(label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels if ticklabels is not None else ticks)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel("Longitude (°)")


def build(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 7.2))

    panel_categorical(axes[0], fig, df, "source", SOURCES, MB_OTHER,
                      "a  Inventory", first=True)
    F.scalebar(axes[0], length_km=200)

    panel_categorical(axes[1], fig, df, "mb_code", MB_CLASSES, MB_OTHER,
                      "b  MapBiomas class", first=False)

    panel_continuous(axes[2], fig, df, "Year", "viridis", "Census year", "c  Census year")

    tk = np.log10([1, 2, 5, 10, 25])
    df = df.assign(_logw=np.log10(df["richness_woody"].where(df["richness_woody"] > 0)))
    panel_continuous(axes[3], fig, df, "_logw", "magma_r", "Woody species",
                     "d  Woody species richness", ticks=tk, ticklabels=[1, 2, 5, 10, 25])

    fig.tight_layout()
    return F.save_figure(fig, "fig01_study_area", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/figures")
    a = p.parse_args()

    df = load()
    print(f"parcelas: {len(df)}, por fuente {df['source'].value_counts().to_dict()}")
    print(f"clase MapBiomas ausente en {int(df['mb_code'].isna().sum())} parcelas")
    print(f"riqueza lenosa ausente en {int(df['richness_woody'].isna().sum())} parcelas")
    print(f"-> {build(df, Path(a.out_dir))}")


if __name__ == "__main__":
    main()
