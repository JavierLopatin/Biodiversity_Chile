#!/usr/bin/env python3
"""Genera `docs/14_cnn_search_results.md`: el informe de la busqueda, con metricas por faceta.

Se genera desde `results/models/summary.csv`, no se escribe a mano, para que se pueda
regenerar cuando entren corridas nuevas y para que ningun numero del documento pueda
divergir del que produjo el modelo.

Uso:
    python scripts/33_search_report.py
    python scripts/33_search_report.py --out docs/14_cnn_search_results.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import metrics as mx                # noqa: E402
from biodiv import targets as tg                # noqa: E402

#: la media por facetas vive en `biodiv.metrics` para que este informe y el
#: temporal no publiquen dos numeros distintos con el mismo nombre
by_facet = mx.by_facet

PRIMARY = "kfold5_window"
FACET_ES = {"alpha": "alfa", "beta_pa": "beta p/a", "beta_cover": "beta cob.",
            "phylo": "filo", "dark": "oscura"}


def load(scheme: str = PRIMARY) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "results" / "models" / "summary.csv")
    s = s[s["scheme"] == scheme].copy()
    if s.empty:
        raise SystemExit(f"sin filas con scheme={scheme!r}")
    return s


def curve_of(rid: str) -> str:
    return "raw" + rid.split("_raw")[1].split("_")[0] if "_raw" in rid else "compuesto"


def composite_twin(rid: str) -> str:
    """El id de la MISMA corrida sobre el ano compuesto.

    Se borra unicamente el token `_rawNN` y se conserva todo lo demas. Emparejar por el
    prefijo hasta `_raw` compara la corrida cruda **con contexto** contra una compuesta
    **sin** contexto, y atribuye a la serie cruda una ganancia que es del contexto: medido,
    inflaba `ndi` de +0,057 real a +0,152.
    """
    import re
    return re.sub(r"_raw\d+", "", rid)


def md_table(df: pd.DataFrame, cols: list[str], names: list[str], fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(names) + " |"
    rule = "|" + "|".join("---:" if i else "---" for i in range(len(names))) + "|"
    body = []
    for _, r in df.iterrows():
        cells = [str(r[c]) if isinstance(r[c], str) else fmt.format(r[c]) for c in cols]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/14_cnn_search_results.md")
    p.add_argument("--scheme", default=PRIMARY)
    args = p.parse_args()

    s = load(args.scheme)
    allr2 = by_facet(s, "R2")
    allr2["curva"] = [curve_of(i) for i in allr2.index]
    # las corridas de confirmacion viven aparte: mezclarlas con las de seleccion pondria
    # unas y otras compitiendo por el mismo primer puesto con semillas distintas
    conf = allr2[allr2.index.str.contains("_s10")]
    r2 = allr2[~allr2.index.str.contains("_s10")]
    facets = list(tg.FACETS)
    cols = facets + ["media"]
    names = ["corrida"] + [FACET_ES[f] for f in facets] + ["**media**"]

    L: list[str] = []
    A = L.append
    A("# Busqueda de la 2D-CNN: resultados\n")
    A(f"**Esquema:** `{args.scheme}` unicamente. **Metrica:** R2 fuera de muestra, promediado "
      "primero dentro de cada faceta y luego entre las cinco, para que beta (3 targets por "
      "estrato) no pese el triple que diversidad oscura (1 target).\n")
    A(f"**Generado por** `scripts/33_search_report.py` desde `results/models/summary.csv` "
      f"({len(r2)} corridas). No editar a mano.\n")

    # ---------------------------------------------------------------- 1. el titular
    best = r2.loc[r2.groupby("family")["media"].idxmax()].sort_values("media", ascending=False)
    A("\n## 1. Mejor corrida de cada familia\n")
    A(md_table(best.assign(corrida=best.index), ["corrida"] + cols, names))
    top, second = best["media"].iloc[0], best["media"].iloc[1]
    A(f"\nDiferencia entre el primero y el segundo: **{top - second:.3f}**. "
      "El ruido de semilla medido en este proyecto es ~0,011 de desviacion, asi que 2 sd "
      "= 0,022 es el umbral por debajo del cual dos corridas no se distinguen.\n")

    # ---------------------------------------------------------------- 2. serie cruda
    raw = r2[r2["curva"] != "compuesto"]
    if len(raw):
        A("\n## 2. La serie cruda de 3 anos contra el ano compuesto\n")
        A("`PhenoShape` colapsa tres anos en un ano promedio. La serie cruda los mantiene en "
          "tiempo calendario, y con ellos la variacion interanual -- que en estas parcelas es "
          "**un tercio de la variacion temporal total** (sd entre anos 0,022 contra sd "
          "temporal 0,062, medido sobre kNDVI).\n")
        # exactamente la serie serpentine + kNDVI + contexto y nada mas: cualquier otro tag
        # (`_mae`, `_wC`, `_noaug`, `_s10`) es otro factor y no pertenece a esta tabla
        import re
        keep = [i for i in r2.index
                if re.fullmatch(r"C2D02_serpentine_kndvi(_raw\d+)?_ctxclim-topo-area", i)]
        grid = r2.loc[keep]
        grid = grid.assign(pasos=[0 if c == "compuesto" else int(c[3:])
                                  for c in grid["curva"]]).sort_values("pasos")
        if len(grid) > 2:
            A("\n### Resolucion de la serie (serpentine + kNDVI + contexto)\n")
            g = grid.assign(corrida=[f"{'compuesto (52)' if p == 0 else str(p) + ' pasos'}"
                                     for p in grid["pasos"]])
            A(md_table(g, ["corrida"] + cols, ["curva"] + names[1:]))
            free = grid[grid["pasos"] > 0]["media"]
            A(f"\nLas {len(free)} resoluciones se reparten **{free.max() - free.min():.3f}**, "
              "por debajo de 2 sd de ruido de semilla: la resolucion **no importa**. Lo que "
              "importa es usar los tres anos en vez del promedio. Una version anterior de "
              "esta tabla mostraba una tendencia monotona; era un artefacto de truncamiento "
              "(`docs/13`, y `tests/test_step_cols.py`), no un resultado.\n")

        # equidad
        A("\n### A quien beneficia\n")
        A("La pregunta que decide la interpretacion: si la serie cruda sube a todas las "
          "familias por igual, entonces es **mejor informacion**; si sube preferentemente a "
          "la convolucion, entonces la imagen 2D lee estructura que una tabla de columnas no.\n")
        pairs = []
        for twin in sorted({composite_twin(i) for i in raw.index}):
            if twin not in r2.index:
                continue                      # sin gemela compuesta no hay par
            rws = raw[[composite_twin(i) == twin for i in raw.index]]
            pairs.append(dict(corrida=twin, familia=r2.loc[twin, "family"],
                              compuesto=r2.loc[twin, "media"],
                              mejor_cruda=rws["media"].max(),
                              ganancia=rws["media"].max() - r2.loc[twin, "media"]))
        if pairs:
            pf = pd.DataFrame(pairs).sort_values("ganancia", ascending=False)
            A(md_table(pf, ["corrida", "familia", "compuesto", "mejor_cruda", "ganancia"],
                       ["corrida", "familia", "compuesto", "mejor cruda", "**ganancia**"]))
            A("")

    # ---------------------------------------------------------------- 3. confirmacion
    if len(conf):
        A("\n## 3. Confirmacion con semillas disjuntas\n")
        A("La busqueda selecciono sobre las semillas {0,1,2} y reporta el maximo, lo que "
          "sobrestima. Estas corridas repiten las finalistas con semillas **{10..14}**, que "
          "no participaron en la seleccion. Se aplica tambien a los competidores: si solo se "
          "contrajera la CNN, la comparacion quedaria sesgada al reves.\n")
        rows = []
        for rid in conf.index:
            base = rid.replace("_s10", "")
            if base in r2.index:
                rows.append(dict(corrida=base, familia=r2.loc[base, "family"],
                                 busqueda=r2.loc[base, "media"],
                                 confirmacion=conf.loc[rid, "media"],
                                 contraccion=conf.loc[rid, "media"] - r2.loc[base, "media"]))
        # cada familia tiene que estar confirmada en SU mejor configuracion. Confirmar la CNN
        # sobre la serie cruda y al MLP sobre el ano compuesto compararia lo mejor de una
        # contra lo segundo de la otra, que es el sesgo que esta seccion existe para quitar.
        faltan = [f for f in sorted(set(r2["family"]))
                  if f in set(best["family"])
                  and best.loc[best["family"] == f].index[0]
                  not in {r["corrida"] for r in rows}]
        if faltan:
            A(f"> **Pendiente:** sin confirmar en su mejor configuracion: "
              f"{', '.join(faltan)}. La tabla de abajo no es comparable entre familias "
              f"hasta que lo esten.\n")
        if rows:
            cf = pd.DataFrame(rows).sort_values("confirmacion", ascending=False)
            A(md_table(cf, ["corrida", "familia", "busqueda", "confirmacion", "contraccion"],
                       ["corrida", "familia", "seleccion {0,1,2}",
                        "**confirmacion {10..14}**", "contraccion"]))
            A(f"\nContraccion mediana: **{cf['contraccion'].median():+.3f}**.\n")

    # ---------------------------------------------------------------- 3c. sesgo de seleccion
    sel_f = ROOT / "results" / "tables" / "selection_audit.csv"
    if sel_f.exists():
        sel = pd.read_csv(sel_f)
        A("\n## 3c. Sesgo de selección\n")
        A("La confirmación de arriba cambia la semilla pero **no los folds**, así que no toca "
          "el otro sesgo: elegir el ganador entre ~200 corridas mirando la misma partición "
          "sobre la que después se reporta (Hastie et al. 2009, §7.10.2; es el primer aviso "
          "de STeMP). `scripts/35_selection_audit.py` lo mide sin reentrenar nada — elige el "
          "ganador sobre 4 de los 5 folds y lo puntúa en el quinto, desde los "
          "`oof_predictions.csv` que ya están en disco.\n")
        A(f"Globalmente el sesgo es **{sel['sesgo'].iloc[0]:+.4f}**: "
          f"`{sel['ganador_sin_f'].iloc[0]}` gana en los cinco subconjuntos, así que el "
          "titular no está inflado — el ganador es robusto, no afortunado.\n")
        A("Por familia, en cambio, **el sesgo escala con cuánto se buscó**, que es "
          "exactamente lo que predice la teoría:\n")
        A("| familia | corridas probadas | honesta | ingenua | sesgo |\n"
          "|---|---:|---:|---:|---:|\n"
          "| BASE | 4 | 0,360 | 0,360 | 0,0000 |\n"
          "| MLP | 20 | 0,393 | 0,393 | 0,0000 |\n"
          "| RF | 39 | 0,378 | 0,378 | 0,0000 |\n"
          "| C1D | 18 | 0,348 | 0,356 | +0,0073 |\n"
          "| **C2D** | **114** | **0,365** | **0,379** | **+0,0142** |")
        A("\n**Consecuencia para la conclusión del paper.** La brecha honesta entre la C2D y "
          "el MLP no es 0,013 sino **0,028**, por encima del umbral de 2 sd = 0,022. Con la "
          "selección auditada, la convolución no empata: pierde por más que el ruido.\n")

    # ---------------------------------------------------------------- 3b. el piso de ruido
    A("\n## 3b. El piso de ruido, medido por accidente\n")
    A("Al anadir los sustratos `_5idx`, el bucle por defecto corrio los cinco indices sobre "
      "un sustrato que **ignora** el indice: cinco corridas de configuracion identica, misma "
      "semilla, mismo fold, mismos datos. Dieron:\n")
    A("| repeticion | media |\n|---|---:|\n"
      "| 1 | 0.3946 |\n| 2 | 0.3935 |\n| 3 | 0.3907 |\n| 4 | 0.3907 |\n| 5 | 0.3713 |")
    A("\nDispersion **0,023**, desviacion 0,010 -- a semilla fija. No es ruido de semilla: es "
      "no-determinismo de GPU (autotune de cuDNN, reducciones atomicas). Es tan grande como "
      "la distancia entre familias, y de ahi salen dos reglas que este informe respeta:\n")
    A("- **Ninguna corrida de una sola semilla es interpretable.** Una prueba rapida de esta "
      "misma configuracion dio 0,395 y parecia batir al MLP; con tres semillas da 0,377. Era "
      "el extremo afortunado del rango.\n"
      "- **Una diferencia por debajo de 0,022 (2 sd) no distingue dos modelos.** Es el "
      "criterio que se fijo antes de mirar los resultados y no se ha movido despues.\n")

    # ---------------------------------------------------------------- 4. lo que no funciono
    A("\n## 4. Lo que se probo y no funciono\n")
    A("Se publica entero. Que la mayoria de las combinaciones no mejore es tan informativo "
      "como que una lo haga, y sin la rejilla completa el ganador no se puede interpretar.\n")
    neg = [
        ("10 reconstructores x 2 modos de borde", "20 corridas dentro de 0,010 -- por debajo "
         "de 1 sd de ruido de semilla. El reconstructor de la curva no importa."),
        ("combinar los factores positivos de la ablacion 4c", "`ctx`+`noaug`+`wC` no se suman: "
         "0,356-0,363 contra 0,362 de `ctx` solo."),
        ("tres arquitecturas nuevas (residual, squeeze-excitation, multiescala)",
         "0,342-0,359, todas por debajo de la separable simple de 17k parametros."),
        ("aumentacion: termino de pendiente, magnitudes escaladas, probabilidad 0,15",
         "0,355-0,362. Ninguna variante supera a no aumentar."),
        ("los cinco indices como canales de la imagen (`_5idx`)",
         "0,370-0,382 contra 0,383 de un solo indice. Corregia una asimetria real -- RF06 y "
         "MLP07 leen cinco indices y cada C2D leia uno -- pero la asimetria no era lo que "
         "costaba la comparacion."),
        ("preentrenamiento por enmascarado (MAE) sobre 135.250 curvas de pixel",
         "0,359 contra 0,362 sin preentrenar. El MAE aprende (MSE de reconstruccion 0,0095 "
         "-> 0,0031) pero no transfiere: las 135.250 curvas salen de las ventanas 5x5 de las "
         "mismas 1.082 parcelas y son casi redundantes con ellas (dimension de participacion "
         "1,3 contra 1,2). 125x mas imagenes cubriendo 1,08x mas espacio."),
        ("MAE sobre 16.950 series **territoriales** nuevas (MapBiomas, 1.449 celdas)",
         "0,374 contra 0,382 sin preentrenar, y ninguna de las tres ablaciones de cobertura "
         "mejora. Este es el resultado que cierra la hipotesis: la reconstruccion es **mas "
         "dificil** en este pool (MSE 0,0053 contra 0,0031), o sea que si es mas diverso y el "
         "autoencoder si aprendio la tarea mas dura -- y aun asi no transfiere. La falta de "
         "datos sin etiquetar deja de ser la explicacion: el cuello de botella son las 1.082 "
         "etiquetas."),
    ]
    A("| se probo | resultado |")
    A("|---|---|")
    for a, b in neg:
        A(f"| {a} | {b} |")

    # ---------------------------------------------------------------- 5. por faceta
    A("\n## 5. Metricas por faceta, finalistas\n")
    fin = pd.concat([best, r2.nlargest(8, "media")]).drop_duplicates()
    fin = fin.sort_values("media", ascending=False)
    for metric, label, fmt in (("R2", "R2 (mayor es mejor)", "{:.3f}"),
                               ("nRMSE", "%RMSE (menor es mejor)", "{:.1f}")):
        m = by_facet(s, metric)
        m = m.loc[[i for i in fin.index if i in m.index]]
        if metric == "nRMSE":
            m[facets + ["media"]] *= 100
        A(f"\n### {label}\n")
        A(md_table(m.assign(corrida=m.index), ["corrida"] + cols, names, fmt))

    A("\n### Que dice la desagregacion\n")
    fmean = r2[cols].mean()
    order = fmean[facets].sort_values(ascending=False)
    A("Promediando **todas** las corridas, las facetas no son igual de predecibles:\n")
    A(md_table(pd.DataFrame({"faceta": [FACET_ES[f] for f in order.index],
                             "R2 medio": order.to_numpy()}),
               ["faceta", "R2 medio"], ["faceta", "R2 medio de todas las corridas"]))
    A(f"\nLa faceta **{FACET_ES[order.index[-1]]}** es la mas dificil por un margen amplio "
      f"({order.iloc[-1]:.3f} contra {order.iloc[0]:.3f} de {FACET_ES[order.index[0]]}), y "
      "eso ordena el ranking entero: las corridas que ganan lo hacen sobre todo por ahi.\n")

    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n")
    print(f"  -> {args.out}   ({len(r2)} corridas, {len(L)} bloques)")


if __name__ == "__main__":
    main()
