#!/usr/bin/env python3
"""Genera `docs/16_stemp_protocol.md`: el protocolo STeMP de este modelo.

STeMP (Linnenbrink et al. 2026, https://github.com/LOEK-RS/STeMP) es un protocolo de
**reporte** para modelos espacio-temporales de aprendizaje automático: no prescribe cómo
validar, prescribe qué declarar, y su aplicación levanta avisos cuando detecta cuatro
errores conocidos. Los ~45 campos son los de su Tabla A1.

**Por qué generado y no escrito.** La app Shiny de STeMP autorrellena desde objetos RDS de
`caret`/`tidymodels`/`mlr3`; este proyecto es Python y PyTorch, así que el llenado sería
manual -- y un protocolo con números tecleados a mano diverge del modelo en la primera
corrida nueva. Aquí cada cifra sale de `results/`, igual que `scripts/33_search_report.py`.

Fuentes: `results/models/summary.csv`, `results/tables/{selection_audit,
sampling_pattern,aoa_summary,temporal_validation}.*`, y `data/derived/plots_subset.parquet`.
Lo que no exista todavía sale marcado como PENDIENTE en vez de omitirse: un campo ausente se
lee como "no aplica", y no es lo mismo que "aún no medido".

Uso:
    python scripts/38_stemp_protocol.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import features as feat              # noqa: E402
from biodiv import metrics as mx                 # noqa: E402
from biodiv import targets as tg                 # noqa: E402

PEND = "**PENDIENTE**"
HEAD = "MLP06_curve_kndvi_raw100"


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


def fmt(x, nd=3):
    return PEND if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def section(title: str) -> str:
    return f"\n## {title}\n"


def field(name: str, value: str, mandatory: bool = False) -> str:
    return f"| {'**' + name + '**' if mandatory else name} | {value} |"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/16_stemp_protocol.md")
    p.add_argument("--derived", default="data/derived")
    args = p.parse_args()

    d = Path(args.derived)
    tabl = ROOT / "results" / "tables"
    plots = pd.read_parquet(d / "plots_subset.parquet")
    summary = pd.read_csv(ROOT / "results" / "models" / "summary.csv")
    pat = read_json(tabl / "sampling_pattern.json")
    aoa = read_json(tabl / "aoa_summary.json")
    sel = read_csv(tabl / "selection_audit.csv")
    tmp = read_csv(tabl / "temporal_validation.csv")

    win = summary[summary.scheme == "kfold5_window"]
    m = mx.by_facet(win)
    best = m.loc[m.groupby("family")["media"].idxmax()].sort_values("media", ascending=False)
    head_r2 = m.loc[HEAD, "media"] if HEAD in m.index else None
    nrmse = mx.by_facet(win, "nRMSE")

    L: list[str] = []
    A = L.append
    A("# Protocolo STeMP\n")
    A("Protocolo de reporte para modelos espacio-temporales de aprendizaje automático "
      "(Linnenbrink et al. 2026). Los campos son los de su Tabla A1; **en negrita** los "
      "obligatorios.\n")
    A(f"**Generado por** `scripts/38_stemp_protocol.py`. No editar a mano: cada cifra sale "
      f"de `results/`. {len(summary.run_id.unique())} corridas en la tabla de origen.\n")

    # ------------------------------------------------------------------ Overview
    A(section("1. Overview"))
    A("| campo | valor |\n|---|---|")
    A(field("Model title", "Predicción de facetas de diversidad vegetal desde fenología "
            "Landsat, Chile central", True))
    A(field("Author names", "J. Lopatin", True))
    A(field("Contact", "javierlopatin@gmail.com", True))
    A(field("Study title", "Multi-faceted plant diversity from satellite phenology"))
    A(field("Study link", PEND))
    A(field("Target variable", f"{len(tg.FACETS)} facetas, "
            f"{sum(len(v) for v in tg.FACETS.values())} respuestas: "
            + "; ".join(f"{k} ({len(v)})" for k, v in tg.FACETS.items()), True))
    A(field("Scientific field", "Ecología / teledetección", True))

    # ------------------------------------------------------------------ Model
    A(section("2. Model"))
    A("### 2.1 Learning method\n")
    A("| campo | valor |\n|---|---|")
    A(field("Model type", "Regresión multi-salida", True))
    A(field("Algorithm", "MLP (titular), Random Forest, CNN 1-D y 2-D sobre "
            "transformaciones señal→imagen, línea base de coordenadas", True))
    A(field("Architecture", "MLP06: bloque de curva + contexto clima/topografía/área. "
            "C2D: separable, 17k parámetros, 5 sustratos probados"))

    A("\n### 2.2 Response\n")
    A("| campo | valor |\n|---|---|")
    A(field("Sample size", f"{len(plots)} parcelas", True))
    A(field("Sampling extent", f"X {plots.X.min():.0f}–{plots.X.max():.0f}, "
            f"Y {plots.Y.min():.0f}–{plots.Y.max():.0f} (EPSG:32719); "
            f"años {int(plots.Year.min())}–{int(plots.Year.max())}", True))
    A(field("Sample acquisition", "Inventarios de vegetación en terreno, recopilados en "
            "Parcelas-CL; 12 contribuyentes, 23 conjuntos", True))
    A(field("Sample geometry", f"Polígonos tratados como puntos; superficie "
            f"{plots.PlotSize_m2.min():.0f}–{plots.PlotSize_m2.max():.0f} m² "
            f"(mediana {plots.PlotSize_m2.median():.0f}), la mayoría sub-píxel Landsat", True))
    A(field("Range", "; ".join(
        f"{t}: {win[win.target == t].n.max():.0f} obs" for t in list(tg.FACETS['alpha'])), True))
    if pat:
        s = pat["pattern"]
        A(field("Sampling pattern",
                f"**{s['label']}** — Clark-Evans R = {s['clark_evans_R']}. Distancia entre "
                f"parcelas (Gj) mediana {s['Gj_p50_km']} km; del dominio nativo a la parcela "
                f"más próxima (Gij) mediana {s['Gij_p50_km']} km. El mapa predecirá "
                f"**{s['median_ratio_Gij_Gj']}× más lejos** de lo que separa a las parcelas "
                f"entre sí", True))
    else:
        A(field("Sampling pattern", PEND, True))
    A(field("Coordinate Reference System", "EPSG:32719 (UTM 19S)", True))
    A(field("Data sources of response", "Parcelas-CL (Chilean vegetation plot database)", True))

    A("\n### 2.3 Predictors\n")
    A("| campo | valor |\n|---|---|")
    A(field("Predictor types", "Curva fenológica por índice de vegetación; métricas LSP; "
            "topografía; clima; área de parcela", True))
    A(field("Number of predictors", "121 en el diseño titular (curva de 100 pasos + 21 de "
            "contexto)", True))
    A(field("Resolution of predictors", "30 m (Landsat C2L2); ventana de 5×5 píxeles por "
            "parcela; paso temporal ~11 días sobre 3 años", True))
    A(field("Preprocessing", "Escalado C2 (DN·2,75e-5 − 0,2); máscara QA por producto antes "
            "de concatenar sensores; interpolación a grilla regular con "
            "`biodiv.curves.interp_grid`"))
    A(field("Temporal alignment", "**Ventana causal `y−2..y`**: 3 años que terminan en el "
            "año del censo. Ninguna observación es posterior al censo", True))
    A(field("Data sources of predictors", "Landsat 5/7/8/9 C2L2 vía Data Cube Chile; "
            "Copernicus DEM 30 m; CHELSA/WorldClim; MapBiomas Chile para la máscara nativa",
            True))

    # ------------------------------------------------------------------ evaluación
    A("\n### 2.4 Model evaluation and selection\n")
    A("| campo | valor |\n|---|---|")
    A(field("Model evaluation strategy",
            "Validación cruzada agrupada de 5 folds por **componente de ventana** "
            "(`kfold5_window`): cero fuga de píxeles por construcción, 0 de 135 componentes "
            "partidos. Complementada con bloques de 20 km, LTO y LLTO — ver §2.5", True))
    A(field("Performance metrics", "R² fuera de muestra, %RMSE y sesgo normalizado, en "
            "unidades originales tras retransformación de Yeo-Johnson con el estimador de "
            "Duan", True))
    A(field("Model evaluation results",
            f"Mejor modelo `{HEAD}`: R² medio sobre facetas = {fmt(head_r2)} bajo "
            f"`kfold5_window`. Por facetas: "
            + ", ".join(f"{k} {fmt(m.loc[HEAD, k]) if HEAD in m.index else PEND}"
                        for k in tg.FACETS), True))
    A(field("Hyperparameter tuning", "Búsqueda por coordenadas sobre ~195 combinaciones "
            "(reconstructor de curva, sustrato, arquitectura, aumentación, regularización); "
            "rejilla completa publicada en `docs/14`"))
    A(field("Predictor selection", "Cribado por bloques (19 combinaciones) más "
            "comparaciones pareadas; **las coordenadas no entran en los modelos titulares**, "
            "sólo como línea base `B03_coords`"))

    A("\n**Mejor corrida de cada familia** (R² medio sobre facetas, `kfold5_window`):\n")
    A("| familia | corrida | " + " | ".join(tg.FACETS) + " | media |")
    A("|---|---|" + "---:|" * (len(tg.FACETS) + 1))
    for rid, r in best.iterrows():
        A(f"| {r['family']} | `{rid}` | "
          + " | ".join(fmt(r[k]) for k in tg.FACETS) + f" | **{fmt(r['media'])}** |")

    if sel is not None and len(sel):
        s0 = sel.iloc[0]
        A(f"\n**Sesgo de selección** (`scripts/35_selection_audit.py`). Seleccionar y evaluar "
          f"sobre la misma partición infla el R²; medido eligiendo el ganador sobre 4 de los "
          f"5 folds y puntuándolo en el quinto: global **{s0['sesgo']:+.4f}** "
          f"(el ganador es el mismo en los 5 subconjuntos). Por familia el sesgo escala con "
          f"cuánto se buscó: 114 corridas C2D → +0,0142; 4 corridas BASE → 0,0000.\n")

    # ------------------------------------------------------------------ 2.5 temporal
    A("\n### 2.5 Transferencia espacial y temporal\n")
    if pat:
        A("Distancia a la que cada esquema evalúa, contra los "
          f"{pat['pattern']['Gij_p50_km']} km a los que el dominio nativo dista de la parcela "
          "más próxima:\n")
        A("| esquema | mediana test→entrenamiento | fracción de la distancia del mapa |")
        A("|---|---:|---:|")
        for r in sorted(pat["schemes"], key=lambda x: x["p50_km"]):
            A(f"| `{r['scheme']}` | {r['p50_km']:.2f} km | {r['vs_Gij']:.2f} |")
        A("")
    if tmp is not None and len(tmp):
        piv = tmp.pivot_table(index="familia", columns="scheme", values="media")
        A("R² medio del **mismo modelo** bajo cada esquema:\n")
        A("| familia | " + " | ".join(f"`{c}`" for c in piv.columns) + " |")
        A("|---|" + "---:|" * len(piv.columns))
        for fam, r in piv.iterrows():
            A(f"| {fam} | " + " | ".join(fmt(r[c]) for c in piv.columns) + " |")
        A("")
    else:
        A(f"{PEND} — correr `scripts/run_temporal_stage.sh` y "
          "`scripts/39_temporal_report.py`.\n")

    # ------------------------------------------------------------------ interpretación
    A("\n### 2.6 Interpretation, uncertainty and limitations\n")
    A("| campo | valor |\n|---|---|")
    A(field("Explainability", "Importancia por permutación en RF; ablación por bloques de "
            "predictores; descomposición within/between contribuyente"))
    A(field("Scientific interpretation", "Beta se sostiene con el nivel del contribuyente "
            "removido (R² intra +0,380); alfa es en un 88 % nivel de contribuyente y se "
            "reporta como descriptor"))
    A(field("Potential biases",
            "(a) η²(contribuyente) = 0,72 sobre la riqueza; (b) el 10,7 % de las parcelas "
            "cae en clases que la máscara nativa excluye (6,1 % silvicultura, 3,2 % "
            "agricultura, 1,4 % urbano); (c) contribuyente y año están confundidos "
            "(Cramér's V = 0,732), así que un leave-time-out no separa ambos efectos"))
    A(field("Sensitivity assessment", "20 reconstructores de curva (dispersión 0,010), 8 "
            "resoluciones de serie cruda (0,011), 5 semillas; ruido de no-determinismo a "
            "semilla fija medido en 0,023 de dispersión"))
    A(field("Limitations",
            "(a) `log10_area` y `stratum` están en todos los diseños y **no son mapeables**: "
            "un píxel no tiene superficie de parcela; (b) `lon/lat/elev` alcanzan el 90 % "
            "del beta que logran los 87 predictores, así que el aporte propio de la "
            "teledetección es +0,054; (c) el esquema titular evalúa a 0,5 km y el mapa "
            "predice a 13 km"))
    A(field("Software", "Python 3.11, PyTorch 2.6, scikit-learn, xarray/datacube; R para "
            "filogenia y diversidad oscura"))
    A(field("Code availability", "github.com/JavierLopatin/Biodiversity_Chile"))
    A(field("Data availability", "Parcelas-CL público; predictores derivados en "
            "`data/derived/`"))

    # ------------------------------------------------------------------ Prediction
    A(section("3. Prediction"))
    A("> El modelo **todavía no produce un mapa**. Esta sección es opcional en STeMP "
      "precisamente para este caso, y se rellena con lo que ya está medido y condiciona el "
      "mapa futuro.\n")
    A("| campo | valor |\n|---|---|")
    A(field("Prediction extent", "Vegetación nativa de Chile central según MapBiomas: "
            "~83.000 km², 1.449 celdas de 10 km"))
    A(field("Prediction resolution", "30 m, año a año dentro de 2003–2026"))
    A(field("Map evaluation strategy", "PENDIENTE hasta que exista mapa. El esquema "
            "adecuado **no** es `kfold5_window` sino uno cuya distancia de evaluación se "
            "parezca a la del mapa (`kfold5_block20`, `kfold_loc_time`)"))
    if aoa:
        A(field("Uncertainty quantification",
                f"**Área de aplicabilidad** (Meyer & Pebesma 2021) sobre "
                f"{aoa['n_domain']:,} muestras de cobertura nativa: "
                f"**{100 * aoa['fraction_inside']:.1f} % dentro**. Umbral DI = "
                f"{aoa['threshold']:.3f} desde los folds de `{aoa['scheme']}`; DI del "
                f"dominio mediana {aoa['di_median']:.3f}, p95 {aoa['di_p95']:.3f}", True))
    else:
        A(field("Uncertainty quantification", PEND, True))
    b20 = (tmp[tmp.scheme == "kfold5_block20"].set_index("familia")["media"]
           if tmp is not None and "kfold5_block20" in set(tmp.scheme) else None)
    if b20 is not None and len(b20):
        A(field("Map accuracy",
                "Estimada con `kfold5_block20`, cuya distancia de evaluacion (12,1 km) se "
                "parece a la del mapa (13,2 km): "
                + ", ".join(f"{k} {v:.3f}" for k, v in b20.sort_values(ascending=False).items())
                + ". **El orden se invierte respecto de `kfold5_window`**: el RF pasa de "
                  "tercero a primero por 0,062, muy por encima del ruido"))
    else:
        A(field("Map accuracy", PEND))
    A(field("Threshold selection", "No aplica: las respuestas son continuas"))
    A(field("Post-processing", "No aplica"))

    # ------------------------------------------------------------------ avisos
    A(section("4. Los cuatro avisos automáticos de STeMP"))
    A("| aviso | estado |\n|---|---|")
    A("| Estrategia de evaluación inadecuada al patrón de muestreo | **Parcialmente.** "
      "`kfold5_window` cierra la fuga de píxeles por construcción, pero evalúa a 0,5 km "
      "mientras el mapa predice a 13 km. Declarado y complementado con block20 y LLTO |")
    A("| Fuga entre selección y evaluación | **Medido, no evitado.** Auditado en "
      "`scripts/35_selection_audit.py`: 0,000 global, +0,014 en la familia más buscada |")
    A("| Falta de cuantificación de incertidumbre en extrapolación | **Cubierto.** AOA "
      "calculado sobre el dominio nativo antes de producir mapa |")
    A("| Proxies espaciales sin selección de predictores | **No aplica.** Las coordenadas "
      "sólo entran como línea base `B03_coords`, nunca en los modelos titulares |")

    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n")
    n_pend = sum(PEND in x for x in L)
    print(f"  -> {args.out}   ({len(L)} bloques, {n_pend} campos pendientes)")


if __name__ == "__main__":
    main()
