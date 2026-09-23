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

## Cadena a re-ejecutar

1. **Filtro**. `occurrences_unified_counts.parquet` → versión leñosa. La asignación de
   forma de crecimiento sale del catálogo oficial cruzado con
   `results/taxa/taxa_roster_unified.csv` (`scripts/80_growth_form_audit.py --catalog ...`).
   No usar la clasificación provisional a ojo salvo como control.
2. `scripts/62_lcbd_sorensen_pg.R` → `lcbd_count_sorensen.parquet`
3. `scripts/63_pd_inext_coverage.R` → `pd_inext_coverage.parquet`
4. `scripts/64_td_inext_coverage.R` → `td_inext_coverage.parquet`
5. `scripts/69_pad_pg_facets_unified.py` → versiones padded al pool de 3.102
6. Reajuste con `scripts/11_run_conv.py`, esquema `kfold5_block20_unified`, 5 semillas,
   para **C1D01** (el desplegado) y **C2D02** (control de arquitectura). Salida a
   `results/models_unified_woody/` para no pisar `models_unified_topofix`.
7. Comparar contra `results/models_unified_topofix/` faceta por faceta.

## Tres cosas que hay que vigilar

- **LCBD cambia en las 2.499 parcelas, no solo en las 60.** `beta.div.comp` se calcula
  sobre la matriz de comunidad completa, así que quitar especies de 60 parcelas mueve las
  disimilitudes de todas. No es un cambio local como TD/PD.
- **Árbol filogenético.** `phylo_tree_unified.tre` tiene 610 puntas (scripts/54). Al caer
  ~100 taxones herbáceos hay que decidir si se poda o se reconstruye, y verificar que PD
  por parcela no se mueva en las parcelas leñosas — no debería, porque las longitudes de
  rama vienen del megaárbol, pero hay que comprobarlo antes de interpretar deltas.
- **Entorno.** Ejecutar en pop-os, no en el Mac. El Mac tiene numpy 2.4 / pandas 2.3 /
  torch 2.12 contra numpy 1.24 / pandas 1.5 / torch 2.0.1+cu117 en pop-os, y le faltan
  iNEXT.3D, adespatial y V.PhyloMaker2. Correr allá mantiene el delta atribuible al
  cambio de datos y no al cambio de entorno.

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
