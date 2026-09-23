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
`data/derived/growth_form_lookup.csv` (`scripts/80` → `scripts/81` → `scripts/82`): 214
taxones por nombre aceptado o sinónimo del catálogo de Rodríguez et al. 2018, 30 por
herencia de género unánime, 11 manuales con su motivo, y 6 sin resolver que no entran al
filtro (51 registros, 0,5%). Efecto medido sobre la riqueza observada por parcela:

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
6. **Curvas de acumulación**, que también salen de la tabla de ocurrencias y por lo tanto
   también están contaminadas: `scripts/56_unified_hill_curves.R` y
   `scripts/59_unified_beta_freq_curve.R`. Después `python scripts/79_hill_curve_figures.py`
   regenera `fig24_hill_curves_q0` y `figS6_hill_curves_q12`. Las cifras que el manuscrito
   cita hoy en §Diversity facets (610 especies efectivas en q=0, PD media 53,3, beta de 7,6
   a 128,4) salieron del pool sin filtrar y van a bajar.
7. Reajuste con `scripts/11_run_conv.py`, esquema `kfold5_block20_unified`, 5 semillas,
   para **C1D01** (el desplegado) y **C2D02** (control de arquitectura). Salida a
   `results/models_unified_woody/` para no pisar `models_unified_topofix`.
8. **Correr también la variante CON hierbas en el mismo entorno de hoy.** Ver la nota de
   entorno más abajo: el `models_unified_topofix` existente ya no sirve de línea base
   limpia, así que el contraste tiene que ser contra una corrida fresca, no contra la
   antigua.
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
  pandas 1.5 en pop-os. Pero **el argumento de constancia de entorno ya no se sostiene
  solo**: el commit `6186fd4` reescribió los checkpoints del C1D01 con sklearn 1.8, contra
  el sklearn 1.3.1 que figura en el `config.json` de la corrida topofix. Es decir, pop-os
  tampoco es hoy el entorno que produjo la Tabla 3. Por eso el paso 8: la comparación
  válida es entre dos corridas frescas del mismo día, no contra `models_unified_topofix`.
- **Reproducibilidad de semillas.** El commit `d32a76c` reporta que la semilla 0 del refit
  all-data no es reproducible. Verificar si eso afecta también al esquema de bloques antes
  de leer diferencias del orden de la dispersión entre semillas.

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
