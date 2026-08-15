# Facetas estilo Pérez-Giraldo sobre el pool unificado

Pérez-Giraldo, L.C. et al. (2025). *Environmental heterogeneity mediates plant diversity
and ecosystem stability in mountain ecosystems of the Mediterranean Andes*. Ecography.
Metodología replicada aquí, sobre subset de conteo real de individuos (Parcelas-CL
estrato `Abundance` + conteo de árboles de Living Trees) y no sobre el pool completo
unificado (3,102 parcelas) que usan `hill_q0_unified` / `lcbd_pa_unified` / etc, ver
`docs/19_unified_facets_methodology.md`.

## 1. Facetas: qué son, cómo se calcularon, con qué N

| faceta | script | método | N observado (de 3,102) |
|---|---|---|---|
| `lcbd_count_sorensen` | `scripts/62_lcbd_sorensen_pg.R` | `adespatial::beta.div.comp(coef="S", quant=TRUE)` → `LCBD.comp(sqrt.D=TRUE)` | 2,499 |
| `pd_inext_q0/q1/q2` | `scripts/63_pd_inext_coverage.R` | `iNEXT.3D::estimate3D(diversity="PD", base="coverage", nboot=1, PDtree=...)`, restringido al árbol de 610 tips (`scripts/54`) | 888 |
| `td_inext_q0/q1/q2` | `scripts/64_td_inext_coverage.R` | igual, `diversity="TD"`, todas las especies (sin restricción al árbol) | 895 |

Fuente común: `data/derived/occurrences_unified_counts.parquet` (`scripts/61_build_unified_counts.py`)
— el único subset con conteo real de individuos, no cobertura ni área basal:

- **Parcelas-CL**: filas `Abundance_parameter=="Abundance"` de `occurrences_unified.parquet`
  (479 parcelas). Cover/Basal_area no tienen dato por individuo recuperable en
  `data/Parcelas_CL_RAW/` — verificado leyendo los 6 archivos crudos locales.
- **Living Trees**: `living_trees_long_counts.parquet` — conteo de filas a nivel árbol por
  especie×parcela (`scripts/50_build_living_trees.py`), no la suma de área basal que ya
  existía.

`estimate3D(base="coverage")` exige ≥5 especies por unidad — de ahí la caída de N en
`pd_inext`/`td_inext` frente a `lcbd_count_sorensen`. `pd_inext` además pierde candidatas
por restringirse al árbol filogenético de 610 tips (888 vs 895/2,499).

Padding a pool completo: `scripts/69_pad_pg_facets_unified.py` reindexa los tres parquets
sobre las 3,102 `PlotObservationID` de `plots_unified.parquet`, NaN donde el estimador no
alcanzó su piso — mismo contrato de máscara que el resto de facetas unificadas (una fila
por parcela, nunca fila ausente). Salidas: `lcbd_count_sorensen_unified_padded.parquet`,
`pd_inext_coverage_unified_padded.parquet`, `td_inext_coverage_unified_padded.parquet`.

## 2. Wiring en `src/biodiv/targets.py`

```
TARGETS_PG_LCBD = ["lcbd_count_sorensen"]
TARGETS_PG_PD   = ["pd_inext_q0", "pd_inext_q1", "pd_inext_q2"]
TARGETS_PG_TD   = ["td_inext_q0", "td_inext_q1", "td_inext_q2"]
TARGETS_ALL_PG  = TARGETS_PG_LCBD + TARGETS_PG_PD + TARGETS_PG_TD
```

`TARGET_SETS["pg_all"|"pg_lcbd"|"pg_pd"|"pg_td"]`. Reporting group separado,
`FACETS_PG`/`FACET_OF_PG` — no se mezcla con `FACETS`/`FACETS_UNIFIED` para no romper el
invariante `FACET_OF == TARGETS_ALL` que varios tests guardan.

## 3. Topografía real de Living Trees

`data/derived/living_trees/topography.parquet` (28 columnas, mismo esquema que
`data/derived/topography/topography.parquet`) llegó durante esta sesión. Antes, el bloque
`topo` del contexto (`CONTEXT_SPEC = "topo+area"`, usado por defecto en **todo** modelo,
CNN y RF) imputaba NaN → mediana del fold para las 2,020 parcelas Living Trees.

`scripts/70_build_living_trees_topo_unified.py`: mismo crosswalk lat/lon exacto que
`scripts/67` (`site_id` ↔ `PlotObservationID`), concatena con el topo Parcelas-CL
(re-prefijado `PCL_`) → `data/derived/topography_unified.parquet`, 3,102 filas, 0 NaN
salvo `heat_load` en terreno plano (10 parcelas, comportamiento esperado ya documentado
para Parcelas-CL). `src/biodiv/features.py:_load_tables` usa este archivo automáticamente
cuando existe y `BIODIV_UNIFIED=1` — sin flag nuevo, sin cambio de firma.

Efecto medido (raw ctr, kndvi, `pg_all`): `lcbd_count_sorensen` +0.02–0.04 en todos los
índices; `pd_inext_q0`/`td_inext_q0` neutro. Se deja topo real como default — sin
contraindicación, mejora donde mejora.

## 4. Resultados: crudo vs PhenoShape, center vs mean5x5

`serpentine`, `kndvi`, `kfold5_block20_unified`, `pg_all` (7 facetas):

| faceta | n | raw ctr | raw mean5x5 | phenoshape ctr | phenoshape mean5x5 |
|---|---:|---:|---:|---:|---:|
| lcbd_count_sorensen | 2,499 | 0.417 | 0.415 | 0.412 | 0.335 |
| pd_inext_q0 | 888 | 0.598 | 0.581 | 0.571 | 0.345 |
| pd_inext_q1 | 888 | -0.004 | -0.025 | +0.006 | -0.012 |
| pd_inext_q2 | 888 | 0.053 | 0.040 | 0.059 | 0.026 |
| td_inext_q0 | 895 | 0.769 | 0.735 | 0.686 | 0.329 |
| td_inext_q1 | 895 | -0.126 | -0.099 | +0.005 | -0.126 |
| td_inext_q2 | 895 | -0.060 | -0.062 | -0.030 | +0.007 |

Crudo gana o empata en casi todo; la brecha se abre fuerte en mean5x5 (phenoshape colapsa:
0.335/0.345/0.329 vs 0.415/0.581/0.735). Mismo patrón "cruda > phenoshape" ya visto en
`unified_all` (13 facetas, pool completo), ahora confirmado también con esta metodología
distinta (iNEXT.3D coverage-based, LCBD Sørensen cuantitativa sobre conteo real) y N mucho
menor. **Ganador: raw, center pixel.**

## 5. Barrido de índices (raw ctr, topo real, `pg_all`)

| index | lcbd_count_sorensen | pd_inext_q0 | pd_inext_q1 | pd_inext_q2 | td_inext_q0 | td_inext_q1 | td_inext_q2 | media (R²>0.4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kndvi | 0.434 | 0.608 | 0.011 | 0.065 | 0.772 | -0.178 | -0.079 | **0.6047** |
| ndvi | 0.433 | 0.608 | 0.035 | 0.077 | 0.767 | -0.221 | -0.096 | 0.6027 |
| evi | 0.426 | 0.612 | 0.001 | 0.036 | 0.775 | -0.148 | -0.083 | 0.6043 |
| nbr | 0.435 | 0.604 | 0.033 | 0.084 | 0.725 | -0.150 | -0.066 | 0.5880 |
| savi | 0.424 | 0.615 | -0.020 | 0.025 | 0.763 | -0.198 | -0.097 | 0.6007 |

"media (R²>0.4)" = promedio sobre las únicas 3 facetas con señal real
(`lcbd_count_sorensen`, `pd_inext_q0`, `td_inext_q0`) — `q1`/`q2` no predicen en ningún
índice, promediarlos con las demás escondería la única señal que hay. **kndvi gana** por
margen mínimo sobre evi (0.6047 vs 0.6043); nbr es el peor (td_inext_q0 flojo, 0.725).
Consistente con `unified_all` (13 facetas), donde kndvi también ganó — se fija kndvi como
índice default.

## 6. Comparación de familias de modelo

Mismo diseño ganador (kndvi, curvas crudas `raw100`, pixel centro, `topo+area`,
`kfold5_block20_unified`, `pg_all`):

| faceta | 2D-CNN | RF (`RF03p`) | 1D-CNN | 2D-CNN + MAE |
|---|---:|---:|---:|---:|
| lcbd_count_sorensen | 0.434 | **0.447** | 0.420 | 0.437 |
| pd_inext_q0 | 0.608 | 0.582 | 0.612 | **0.613** |
| pd_inext_q1 | 0.011 | 0.004 | **0.049** | -0.010 |
| pd_inext_q2 | 0.065 | 0.054 | **0.091** | 0.040 |
| td_inext_q0 | 0.772 | 0.666 | 0.780 | **0.786** |
| td_inext_q1 | -0.178 | +0.042 | -0.144 | -0.181 |
| td_inext_q2 | -0.079 | -0.073 | -0.066 | -0.090 |

- **RF** (`scripts/09_run_baselines.py --model RF03p`, `curve+topo+area` centro):
  114 s vs ~8.5 min del CNN. Gana `lcbd_count_sorensen`, pierde riqueza (`td_inext_q0`
  -0.106 frente al 2D-CNN).
- **1D-CNN** (`--substrate curve1d`): 14.5k parámetros (vs 15.2k del 2D), casi empata al
  2D en riqueza y es el único con señal utilizable en `pd_inext_q1/q2` (0.049/0.091).
- **2D-CNN + MAE** (`--init-from results/mae/mae_serpentine_kndvi_sep_wB_p2_m0.6.pt`,
  checkpoint preentrenado sobre el pool Parcelas-CL-only, `docs/14`): las 34 tensores del
  trunk cargaron sin error de forma pese a la imagen 10×10 (raw100) vs el shape original de
  pretraining; da el mejor R² absoluto de toda la sesión en `td_inext_q0` (0.786) y
  `pd_inext_q0` (0.613), pero no traslada ganancia a `lcbd_count_sorensen` ni a q1/q2.

`src/biodiv/targets.py`/`scripts/09_run_baselines.py`: se agregó `--folds` (mismo patrón
que `scripts/11_run_conv.py`) y tag de `target_set`/`unified` al `run_id` — mismo bug de
dedup por `run_id` incompleto que ya se había corregido en `scripts/11`.

## 7. Validación espacio-temporal (LLTO)

`scripts/71_build_unified_llto_folds.py` — mismo esquema `kfold_loc_time` de
`docs/16_stemp_protocol.md` §2.5 (bloque espacial de 20km × bloque temporal, buffer
causal de ventana), extendido al pool unificado. `loc_fold` toma el **fold externo** de
`kfold5_block20_unified` (0..4), no la clave geométrica cruda — `kfold_loc_time` nombra
celdas `s{fold}t{bloque}` y espera un entero. `TIME_EDGES` (2002..2027) no cambia.
Registrado como `kfold_loc_time_unified` en `SCHEME_GROUP` (group_col `time_block`, mismo
que los esquemas temporales no-unificados).

29 celdas superan `min_test=5`, cubren el 100% del pool (3,102/3,102, cada parcela es test
exactamente una vez). Cobertura de facetas PG sin degradarse frente a block20:
`lcbd_count_sorensen` 2,499/3,102 filas de test, `pd_inext_q0` 888/3,102, `td_inext_q0`
895/3,102.

**Bug real encontrado y corregido**: `inner_split` (`src/biodiv/cv.py`) fijaba k=5 para el
split interno (early stopping) sin chequear cuántos grupos distintos había disponibles.
Bajo LLTO, el pool de entrenamiento de algunas celdas (tras excluir el bloque espacial de
test y todo lo que se solapa con su ventana causal) queda con solo 4 bloques temporales
distintos — `s0t4`/`s1t4`: {0,1,2,3}, sin 4 ni 5 — y con k=5 fijo, `seed=4` siempre cae en
un fold vacío (`RuntimeError: degenerate inner split`, reproducible en 1D-CNN, 2D-CNN y
2D-CNN+MAE). Fix: `k = max(2, min(k_wanted, n_grupos_distintos))` — no cambia el
comportamiento cuando hay grupos de sobra, sólo evita pedir más folds de los que hay
grupos. RF no lo sufrió porque no usa split interno (sin early stopping).

**Resultados** (mismo diseño: kndvi, `raw100`, pixel centro, `topo+area`, `pg_all`):

| faceta | RF | 1D-CNN | 2D-CNN | 2D-CNN + MAE |
|---|---:|---:|---:|---:|
| lcbd_count_sorensen | **0.419** | 0.344 | 0.319 | 0.322 |
| pd_inext_q0 | -0.068 | -0.170 | -0.134 | -0.155 |
| pd_inext_q1 | 0.031 | -0.127 | -0.168 | -0.158 |
| pd_inext_q2 | 0.017 | -0.210 | -0.218 | -0.178 |
| td_inext_q0 | -0.103 | -0.736 | **-0.798** | -0.674 |
| td_inext_q1 | -0.097 | -0.130 | -0.166 | -0.181 |
| td_inext_q2 | -0.007 | -0.094 | -0.069 | -0.080 |

Colapso confirmado, más severo que en `docs/16`'s LLTO no-unificado (que ya reportaba
R² negativo pero no de esta magnitud): `td_inext_q0` (0.77 bajo block20) cae a -0.67/-0.80
en las 3 familias neuronales — el modelo extrapola *mal activamente*, no sólo pierde
señal. RF es el único que se mantiene cerca de cero, sin el colapso catastrófico. Único
sobreviviente parcial: `lcbd_count_sorensen`, positivo en las 4 familias (0.32-0.42,
degradado desde 0.42-0.45 en block20 pero no destruido) — beta composicional (Sørensen
cuantitativo) generaliza en el tiempo mejor que riqueza filogenética/taxonómica
coverage-based. Mismo patrón cualitativo que el pool no-unificado documentado en `docs/16`
y `docs/17`: la extrapolación temporal sigue siendo el límite real del proyecto, con más
datos no se resuelve solo.

## 8. Lectura general

`q0` (riqueza taxonómica y filogenética, coverage-standardized) es la única familia
robustamente publicable: R² 0.58–0.79 en las 4 familias de modelo probadas, 5 índices, 2
formatos de curva. `lcbd_count_sorensen` (beta, Sørensen cuantitativo sobre conteo real)
da señal moderada y estable (~0.42–0.45). `q1`/`q2` (Hill ponderado por
abundancia/dominancia) no predicen en ningún modelo, índice o formato de curva probado —
no es limitación de arquitectura, es límite de señal en esas facetas específicas (mismo
patrón ya visto para `hill_q1`/`hill_q2` no-unified, ver `docs/12_phylo_and_rarefaction.md`).

Config ganadora para seguir iterando: **kndvi, curva cruda (`raw100`), pixel centro,
`topo+area` (con topo real de Living Trees), `kfold5_block20_unified`**.
