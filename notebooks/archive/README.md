# Notebooks archivados

Documentan fases que el proyecto ya superó. No se borraron: describen cómo se llegó a
las decisiones que el workflow final da por sentadas, y varios se pueden reproducir o
actualizar contra el pool unificado cuando haga falta.

El workflow vigente es `scripts/50–71` (pool unificado) → `72` (modelo final) →
`73`/`74` (mapas) → fan-out en Argo (`scripts/argo/process_argo.yaml`). Los notebooks
vivos que lo acompañan son `02` (variables respuesta), `09`, `10` y `11` (mapas).

| Notebook | Qué documenta | Por qué está aquí | Para revivirlo |
|---|---|---|---|
| `01_explore_phenology_db.ipynb` | Control de calidad de la adquisición: cobertura DOY, huecos, calidad de predictores sobre el subset Parcelas-CL | Pool superado: 1.082 parcelas y curvas `PhenoShape` de 52 pasos, frente a las 3.102 parcelas y curvas crudas de 100 pasos del pool unificado | Sus insumos (`plots_subset.parquet`, `phenology/manifest.csv`, `topography.parquet`) siguen en `data/derived`, así que corre tal cual. Portarlo al pool unificado significa leer `plots_unified.parquet` y las curvas `raw100` |
| `03_explore_modelling_results.ipynb` | Los 79 modelos sobre 7 esquemas de CV: controles, brecha de optimismo, tests pareados | Roto y superado. Lee `results/models/summary.csv`, un árbol de artefactos por corrida que está en `.gitignore` y ya no existe en disco. Sus números van bajo `kfold5_owner`, que el propio `05` declara no comparables | Hay que volver a correr `scripts/14_run_matrix.py` para regenerar `results/models/`. Actualizarlo de verdad es reapuntarlo a `results/models_unified_topofix/` bajo block20 |
| `05_results_all_facets.ipynb` | Cinco facetas (`alpha`, `beta_pa`, `beta_cover`, `phylo`, `dark`) bajo `kfold5_window`, con tres métricas | Fase Parcelas-CL. `results/tables/results_best.csv` tiene 15 filas y ninguna mención de `unified`, `block20` ni `raw100`; el producto final publica 3 facetas bajo block20 | Sus tablas siguen en `results/tables/`, así que corre. El equivalente vigente para el conjunto unificado es `scripts/60_unified_summary_figures.py` y el notebook `02` |

## Lo que NO se archivó, y por qué

`04_substrates_2d` y `07_unlabelled_pool` parecen de fases cerradas pero sostienen
decisiones vivas: la transformada `serpentine` que el `04` explica es la del modelo
desplegado (`docs/21` §entrada fenológica), y el checkpoint MAE cuyo pool documenta el
`07` es del que se inicializa `scripts/72_train_final_map_model.py`. Ojo: el veredicto
escrito en ambos quedó desfasado respecto de en qué terminaron.

`08_sensor_harmonization` no vuelve a correr —le faltan insumos— pero es la evidencia de
por qué el camino de mapas no armoniza entre TM/ETM+ y OLI pese a cubrir 2000–2026: la
corrección se implementó bien, se verificó, y aun así no mejoró el modelo.

`06_year_boundary` está citado por `docs/13_phenology_year_boundary.md` y su arreglo vive
en el código.
