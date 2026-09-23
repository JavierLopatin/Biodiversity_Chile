# Armonización del target a plantas leñosas — especificación para pop-os

Fecha: 2026-09-23. Estado: **esperando el catálogo oficial de flora vascular de Chile**
antes de fijar la lista de formas de crecimiento.

## Por qué

El pool unificado mezcla dos definiciones de comunidad. Las 60 parcelas de Becerra, P.I.
(Coquimbo, 31,3–31,5°S, 2022, 500 m²) son un censo de flora vascular completa; las otras
835 parcelas son censos de leñosas.

Evidencia, medida sobre `occurrences_unified_counts.parquet`:

| | taxones en familias exclusivamente herbáceas | riqueza media/parcela |
|---|---|---|
| Becerra | 52 (23 familias), en 60 de 60 parcelas | 28,2 |
| Becerra, solo leñosas | — | 6,8 |
| resto Parcelas-CL | 0 | 4,7 |
| Living Trees | 0 | 3,6 |

De los 124 taxones de Becerra, 88 son no leñosos (gramíneas, geófitas, anuales, dos
helechos), 28 leñosos, 8 dudosos. 108 de los 261 taxones del pool aparecen solo ahí.

Consecuencia medida: el R² del modelo desplegado cae de 0,815 a −0,422 en TD₀ y de 0,641
a −0,027 en PD₀ al excluir esas 60 parcelas del cálculo. El resultado de riqueza del
manuscrito descansa entero en ellas. La estandarización por cobertura no lo arregla: las
de Becerra quedan en cobertura mediana 0,867 y el resto en 0,99, así que se evalúan en
puntos distintos de la curva de rarefacción.

## Qué hacer

Filtrar los registros no leñosos de las 60 parcelas de Becerra y **conservar las
parcelas**. No hay que tocar las otras 835: ya son solo leñosas.

La asignación de forma de crecimiento ya está resuelta y versionada en
`data/derived/growth_form_lookup.csv` (`scripts/80` → `scripts/81` → `scripts/82`). Cubre
las 593 especies del superconjunto `occurrences_unified.parquet`, con la columna
`en_conteos` marcando las 261 que importan para las facetas y las curvas. Sobre esas 261:
214 por nombre aceptado o sinónimo del catálogo de Rodríguez et al. 2018, 30 por herencia
de género unánime, 11 manuales con su motivo, y 6 sin resolver que no entran al filtro
(51 registros, 0,5%). Los 33 taxones sin asignar del lookup viven todos fuera de
`en_conteos`. Efecto medido sobre la riqueza observada por parcela:

| | antes | después |
|---|---|---|
| Becerra | 28,2 | 7,2 |
| resto Parcelas-CL | 4,7 | 4,6 |
| Living Trees | 3,6 | 3,6 |

Ninguna parcela queda sin leñosas, así que el pool conserva las 3.102.

## Cadena a re-ejecutar

1. **Filtro**. `occurrences_unified_counts.parquet` → versión leñosa. La asignación de
   forma de crecimiento sale del catálogo oficial cruzado con
   `results/taxa/taxa_roster_unified.csv` (`scripts/80_growth_form_audit.py --catalog ...`).
   No usar la clasificación provisional a ojo salvo como control.
2. `scripts/62_lcbd_sorensen_pg.R` → `lcbd_count_sorensen.parquet`
3. `scripts/63_pd_inext_coverage.R` → `pd_inext_coverage.parquet`
4. `scripts/64_td_inext_coverage.R` → `td_inext_coverage.parquet`
5. `scripts/69_pad_pg_facets_unified.py` → versiones padded al pool de 3.102
6. **Curvas de acumulación, recalculadas sobre las parcelas del análisis.** Decisión del
   autor (2026-09-23), y cambia el alcance respecto de lo que hacían `scripts/56` y
   `scripts/59`: esos arman la comunidad desde el zip de Parcelas-CL completo más
   `living_trees_long.parquet` (vía `lib/unified_comm.R` y `lib/beta_freq.R`), o sea 593
   especies incluyendo todo el estrato de cobertura. Las curvas tienen que salir del mismo
   conjunto de parcelas sobre el que se ajusta y valida cada faceta, porque la figura vive
   en §Diversity facets para describir las variables respuesta: si describe un pool que el
   modelo nunca ve, el lector asocia 610 especies efectivas a una predicción hecha sobre
   895 parcelas.

   Panel por panel:
   - TD: las 895 parcelas que llevan TD₀
   - PD: las 888 que llevan PD₀
   - beta: las 2.499 que llevan LCBD

   Substrato común: `occurrences_unified_counts.parquet` filtrado a leñosas, no el zip.
   Con esto el lookup de 261 especies basta y sobra; las 332 que solo aparecen en
   cobertura dejan de importar (de ellas 33 quedaban sin asignar).

   Después `python scripts/79_hill_curve_figures.py` regenera las figuras. Después `python scripts/79_hill_curve_figures.py`
   regenera `fig24_hill_curves_q0` y `figS6_hill_curves_q12`. Las cifras que el manuscrito
   cita hoy en §Diversity facets (610 especies efectivas en q=0, PD media 53,3, beta de 7,6
   a 128,4) salieron del pool sin filtrar y van a bajar.
7. Reajuste con `scripts/11_run_conv.py`, esquema `kfold5_block20_unified`, 5 semillas,
   para **C1D01** (el desplegado) y **C2D02** (control de arquitectura). Salida a
   `results/models_unified_woody/` para no pisar `models_unified_topofix`.
8. **Correr también la variante CON hierbas**, con el arreglo de semilla aplicado. No por
   deriva de entorno (ver abajo, esa sospecha era infundada) sino porque la corrida
   topofix existente se produjo con el bug de inicialización y sus semillas no son
   reproducibles.
9. Comparar las dos corridas frescas faceta por faceta.

## Tres cosas que hay que vigilar

- **LCBD cambia en las 2.499 parcelas, no solo en las 60.** `beta.div.comp` se calcula
  sobre la matriz de comunidad completa, así que quitar especies de 60 parcelas mueve las
  disimilitudes de todas. No es un cambio local como TD/PD.
- **Árbol filogenético.** `phylo_tree_unified.tre` tiene 610 puntas (scripts/54). Al caer
  ~100 taxones herbáceos hay que decidir si se poda o se reconstruye, y verificar que PD
  por parcela no se mueva en las parcelas leñosas — no debería, porque las longitudes de
  rama vienen del megaárbol, pero hay que comprobarlo antes de interpretar deltas.
- **Entorno.** Ejecutar en pop-os, no en el Mac: el Mac no tiene iNEXT.3D, adespatial ni
  V.PhyloMaker2, y arrastra numpy 2.4 / pandas 2.3 / torch 2.12 contra numpy 1.24 /
  pandas 1.5 en pop-os. Corregido 2026-09-23: una versión anterior de este documento
  afirmaba que pop-os había derivado a sklearn 1.8 y que por eso `models_unified_topofix`
  dejaba de servir de línea base. **Es falso.** El entorno de pop-os coincide versión por
  versión con el bloque `versions` del `config.json` de topofix (python 3.11.6, numpy
  1.24.4, pandas 1.5.3, scipy 1.10.0, sklearn 1.3.1, torch 2.0.1+cu117). Los commits
  `6186fd4` y `d32a76c` salieron de una máquina cloud distinta (Tesla T4, python 3.12.13,
  sklearn 1.8/1.9.1) y solo tocaron los checkpoints `C1D01_..._FINAL_alldata/final/` del
  refit de mapas, no los directorios `kfold5_block20_unified`. La deriva no toca la Tabla 3.
- **Bug de inicialización de semilla — este sí obliga a rehacer la línea base.** En
  `src/biodiv/dl_runner.py` se llama a `build_model` antes de `train_one_fold`, y
  `seed_everything` corre dentro de `train_one_fold` (`src/biodiv/trainer.py:124`), o sea
  *después* de inicializar los pesos. El primer modelo del proceso arranca desde la
  entropía del proceso. Bajo validación cruzada el daño es peor que en el refit all-data:
  con parada temprana, el número de épocas del fold 0 depende de esa inicialización, y con
  él las extracciones del RNG, así que el estado que hereda cada (semilla, fold) siguiente
  cambia en cascada. En el all-data las épocas fijas cortaban la cascada.

  Esto no es cosmético para este manuscrito. La Tabla 3 reporta media ± desviación entre
  cinco semillas y se niega explícitamente a interpretar diferencias menores que esa
  dispersión. Si las semillas no están controladas, esa desviación no estima dispersión
  entre semillas: mezcla ruido de entropía del proceso. El criterio de comparación del
  paper depende de que el arreglo se aplique.

  Arreglo: mover `seed_everything` antes de `build_model` en `dl_runner.py` y en
  `scripts/77`. Después correr las dos variantes frescas con el arreglo puesto.

## Defectos de nomenclatura detectados (arreglar en el paso 1)

- `Myrtaceae` en el campo de especie: 1 registro, 1 parcela. No es un taxón.
- 16 registros solo a nivel de género.
- `Nothofagus antárctica` con tilde.
- Tres pares del mismo taxón escrito distinto: *Podocarpus saligna*/*salignus*,
  *Schinus polygama*/*polygamus*, *Nothofagus leonii*/*x leonii*. Verificado que ninguno
  coexiste con su par en la misma parcela, así que fusionarlos no altera la riqueza por
  parcela; sí baja el conteo del pool de 261 a 258 y quita puntas duplicadas del árbol.

## Qué queda bloqueado en el manuscrito hasta que esto corra

El 0,815 de TD₀ y el 0,641 de PD₀ de la Tabla 3 no describen predicción de riqueza. LCBD
(0,436) sobrevive a la exclusión de Becerra con 0,396 y no está en cuestión.

Además quedan bloqueadas las cifras de la Figura 2 y su párrafo en §Diversity facets, y la
Sección S4 del suplemento, porque las curvas de acumulación se calcularon sobre el pool sin
filtrar.
