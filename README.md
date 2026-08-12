# Biodiversity_Chile

Estimación multitemporal de facetas múltiples de biodiversidad vegetal en Chile central a partir de
fenología satelital.

Se predicen simultáneamente diversidad alfa taxonómica, contribución local a la diversidad beta
(LCBD), diversidad filogenética, diversidad funcional y composición florística (ejes de ordenación),
usando la curva fenológica reconstruida desde series temporales Landsat y Sentinel-2 más variables
topográficas derivadas de un DEM.

**Estado:** modelamiento completo; siguiente fase en diseño. El pipeline de adquisición está completo para 1.082
parcelas de Chile central (curvas fenológicas de 52 pasos semanales, 18 métricas LSP y 9
variables topográficas, todo × 5 índices de vegetación) y las variables respuesta de fase 1
(diversidad taxonómica, LCBD, PCoA) están calculadas. La etapa de modelamiento —Random
Forest, MLP tabular, CNN 1D y CNN 2D sobre once sustratos, 79 modelos sobre 7 esquemas de
validación cruzada— está corrida. **Resultado principal: la fenología predice composición
florística y unicidad composicional (`pcoa1_pa` R² = +0,41, `lcbd_pa` +0,21 bajo CV dejando
fuera al contribuyente completo) y no predice diversidad alfa.** Diseño, decisiones y
resultados en [`docs/08_modelling.md`](docs/08_modelling.md); registro de ejecución en
[`docs/08_modelling_progress.md`](docs/08_modelling_progress.md). Registro reproducible de la adquisición en
[`docs/07_run_record.md`](docs/07_run_record.md).

---

## Pipeline

```bash
python scripts/01_build_subset.py --zip data/20602096.zip --out-dir data/derived
python scripts/03_extract_topography.py
python scripts/02_extract_phenology.py --hemisphere auto --resume
python scripts/04_recompute_lsp.py --hemisphere auto --write-back
python scripts/05_flatten_phenoshape.py --csv
Rscript scripts/07_compute_taxonomic_beta_responses.R \
    --zip data/20602096.zip --plots data/derived/plots_subset.parquet \
    --out data/derived/biodiversity_responses.parquet
python scripts/06_paper_figures.py --out-dir results/figures

# --- modelamiento ---
python scripts/08_build_modelling_folds.py
python scripts/14_run_matrix.py --all --deadline-hours 8   # corre la matriz completa
```

| Script | Rol |
|---|---|
| `01_build_subset.py` | Subset espacio-temporal de Parcelas-CL y folds de CV agrupados |
| `02_extract_phenology.py` | Cubo 5×5 px por parcela desde el datacube; `PhenoShape` + LSP |
| `03_extract_topography.py` | DEM y derivadas topográficas sobre el mismo parche |
| `04_recompute_lsp.py` | Re-ancla las LSP sobre las curvas ya guardadas, sin volver al cubo |
| `05_flatten_phenoshape.py` | Aplana las curvas de los cubos a tablas de modelado |
| `07_compute_taxonomic_beta_responses.R` | Variables respuesta fase 1: Hill numbers, LCBD y PCoA (dos niveles PA/cover) |
| `06_paper_figures.py` | Figuras del manuscrito |
| `08_build_modelling_folds.py` | Añade `kfold5_owner` (primario), bloques UTM geométricos, `lodo_owner` y `kfold5_random` |
| `09_run_baselines.py` | Controles sin fenología y grilla de Random Forest |
| `10_run_tabular_dl.py` | MLP tabular multioutput |
| `11_run_conv.py` | CNN 1D sobre la curva y CNN 2D sobre once sustratos |
| `12_model_report.py` | Tablas comparativas, brecha de optimismo y tests pareados |
| `13_interpretability.py` | Integrated Gradients replegado al eje DOY; importancia por bloque del RF |
| `14_run_matrix.py` | Orquestador: corre la matriz entera, reanudable, reparte entre GPUs |
| `15_pixel_ablation.py` | Píxel central contra media 5×5, con emparejamiento exacto |

### Modelamiento

Cinco familias sobre las mismas parcelas, los mismos folds y la misma pérdida enmascarada,
de modo que la única diferencia sea **cómo se lee la curva**:

| Familia | Entrada | Parámetros | Qué prueba |
|---|---|---|---|
| Random Forest | 18 métricas LSP + topografía + área | — | la baseline del campo |
| Random Forest | curva de 52 semanas + topografía + área | — | ¿aporta la curva sobre sus resúmenes? (gap G1) |
| MLP multioutput | LSP o curva + topografía | 6,7 k – 59 k | ¿aporta un tronco compartido entre facetas? |
| `Pheno1D` | curva (1 o 5 canales) × 52 | 7,0 k – 43,8 k | ¿aporta la convolución? |
| `PhenoNetS` | once sustratos 2D | 5,1 k – 55,7 k | ¿aporta una segunda dimensión, y tiene que ser real? |

Los sustratos 2D son las nueve transformaciones señal→imagen de
[Trait_2DCNN](https://github.com/JavierLopatin/Trait_2DCNN) retuneadas para n=52, más dos
cuyo eje vertical **no** es manufacturado: `stack5` (5 índices × 52 semanas) y `pxcube`
(25 píxeles × 52 semanas, heterogeneidad intra-parcela). Detalle completo, incluidas las
cuatro opciones de fusión topográfica y las decisiones no obvias (sesgo de retransformación,
padding circular por eje, jitter sub-paso), en [`docs/08_modelling.md`](docs/08_modelling.md).

Exploración de resultados:
[`notebooks/03_explore_modelling_results.ipynb`](notebooks/03_explore_modelling_results.ipynb).
Compuertas de validación: `python -m pytest tests/test_modelling.py -q`.

### Variables respuesta

**Fase 1 (implementada):** diversidad taxonómica (Hill numbers q=0,1,2 vía `hillR`), LCBD
(`adespatial::beta.div`, Legendre & De Cáceres 2013) y 2 ejes PCoA (`ape::pcoa`, corrección
de Cailliez), en dos niveles por abundancia (presencia/ausencia sobre las 1.082 parcelas;
ponderado por cobertura solo sobre el estrato `cover`, ~546 parcelas — columnas `*_cover` en
`NA` fuera de ese estrato, nunca se descarta ni se imputa una parcela). Salida:
`data/derived/biodiversity_responses.parquet`. Exploración de resultados en
[`notebooks/02_explore_biodiversity_responses.ipynb`](notebooks/02_explore_biodiversity_responses.ipynb).

**Nota sobre el método de composición:** se usa PCoA en vez de NMDS. Con mediana de riqueza
= 5 especies/parcela, NMDS (`vegan::metaMDS`) producía soluciones inestables — parcelas con
muy poca información compositional quedaban mal restringidas y el optimizador iterativo las
disparaba a valores absurdos en un eje (se detectó `PCL0468`, monoespecífica, con un valor
~380 desviaciones estándar por sobre la mediana). PCoA es una descomposición espectral
determinística sin esa patología. El script sigue avisando si alguna parcela domina un eje
(criterio MAD) y si los ejes retenidos explican poca varianza — con ~570 especies en un
espacio muy disperso, los primeros 2 ejes explican ~13% de la varianza, bajo pero esperable
a este nivel de dimensionalidad y dispersión.

**Fase 2 (pendiente):** diversidad filogenética, diversidad funcional y dark diversity —
cada una depende de un insumo aún no resuelto: ninguna filogenia referenciada en este repo
(requiere construir un mega-árbol desde la lista de especies), Rasgos-CL solo aporta 2
rasgos continuos (el resto son categóricos), y la curva de acumulación de especies de
Parcelas-CL no satura (ver `docs/01_state_of_the_art.md` gap G5 y
`docs/02_innovation_and_impact.md` riesgos R2/R7).

---

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/01_state_of_the_art.md`](docs/01_state_of_the_art.md) | Revisión crítica en 7 ejes: Spectral Variation Hypothesis, fenología como predictor de biodiversidad, diversidad funcional y filogenética desde teledetección, regresión sobre ejes de ordenación, deep learning multi-tarea, dark diversity, contexto chileno. Cierra con tabla de 5 gaps. |
| [`docs/02_innovation_and_impact.md`](docs/02_innovation_and_impact.md) | Evaluación graduada de innovación, tabla de 9 riesgos cuantificados, impacto esperado, y sección explícita de lo que el proyecto **no** va a demostrar. |
| [`docs/03_cnn_architecture.md`](docs/03_cnn_architecture.md) | Diseño de `PhenoNet-S`, CNN de ~15 k parámetros para n ≈ 1.000 parcelas. Sustrato *phenocube* (año × DOY), catálogo de transformaciones señal→imagen, régimen de entrenamiento y matriz experimental. |
| [`docs/05_data_acquisition.md`](docs/05_data_acquisition.md) | Estrategia de adquisición satelital: ventana temporal, pooling y no estacionariedad, geometría de extracción, plan de ejecución. |
| [`docs/06_phase_and_2d_transform.md`](docs/06_phase_and_2d_transform.md) | Anclaje de fase de las LSP (`hemisphere="auto"`) y la rotación global para la transformación 2D. |
| [`docs/08_modelling.md`](docs/08_modelling.md) | **Modelamiento:** las tres preguntas del benchmark, las decisiones no obvias (sesgo de retransformación, DOY circular por eje, alfa contra beta, píxel central contra 5×5), arquitecturas con conteos medidos, fusión topográfica, la matriz de ~170 corridas y los resultados con tests pareados. |
| [`docs/09_predictors.md`](docs/09_predictors.md) | **Siguiente fase:** eliminar los NaN de LSP (99,2 % vienen de un `return None` cuando la curva no se rota), agregación explícita por píxel, CV por componente de ventana solapada, y el cribado con Random Forest de alternativas no fenológicas — composites anuales, geomedianas y heterogeneidad espectral. |
| [`docs/07_run_record.md`](docs/07_run_record.md) | **Registro del run final:** comandos, parámetros, versiones, salidas, diagnósticos y advertencias. Con [`docs/run_manifest.json`](docs/run_manifest.json) (inventario con `sha256`). |
| [`docs/10_findings.md`](docs/10_findings.md) | Hallazgos de la fase de diagnóstico y del cribado de predictores, con la descomposición within/between del R² de alfa. |
| [`docs/11_next_steps.md`](docs/11_next_steps.md) | **Traspaso:** qué falta, en qué orden y con qué comando. Incluye el apéndice de deudas técnicas conocidas. |
| [`docs/12_phylo_and_rarefaction.md`](docs/12_phylo_and_rarefaction.md) | Diversidad filogenética con árbol propio, rarefacción iNEXT contra Parcelas-CL y dark diversity con DarkDiv. |
| [`docs/13_phenology_year_boundary.md`](docs/13_phenology_year_boundary.md) | La curva fenológica no cierra el año: diagnóstico, mecanismo con `file:line` y las tres opciones de arreglo. Defecto de `PhenoPY`, no de este repo. |
| [`docs/14_cnn_search_results.md`](docs/14_cnn_search_results.md) | **Resultados de la búsqueda de la 2D-CNN:** 195 corridas, la serie cruda de 3 años como único hallazgo real, el piso de ruido de GPU medido (0,010 de desviación) y lo que se probó y no funcionó. |
| [`docs/15_datacube_extraction_spec.md`](docs/15_datacube_extraction_spec.md) | **Para la máquina del datacube:** muestreo no etiquetado de vegetación nativa enmascarada con MapBiomas, para preentrenar el codificador por enmascarado. Qué sacar, cómo, y cómo verificarlo al llegar. |
| [`docs/refs.bib`](docs/refs.bib) | 42 referencias; todos los DOI resueltos contra la API de Crossref. |

---

## Datos

Ninguna base de datos se versiona en este repositorio. Todas son públicas:

| Base | Contenido | Acceso |
|---|---|---|
| **Parcelas-CL** | 1.485 parcelas de vegetación georreferenciadas, 675 especies leñosas, 1976–2026, 30,25°S–54,82°S | [10.5281/zenodo.20602096](https://doi.org/10.5281/zenodo.20602096) — preprint: [10.21203/rs.3.rs-9986019/v1](https://doi.org/10.21203/rs.3.rs-9986019/v1) |
| **Rasgos-CL** | 662 especies leñosas chilenas, 25.174 registros, 23 rasgos funcionales | [github.com/dylancraven/Rasgos-CL](https://github.com/dylancraven/Rasgos-CL) — paper: [10.1111/geb.13755](https://doi.org/10.1111/geb.13755) |

Descarga esperada en `data/` (ignorada por git).

---

## Métodos y dependencias propias

| Repositorio | Rol |
|---|---|
| [PhenoSensing](https://github.com/JavierLopatin/PhenoSensing) | Reconstrucción de curvas fenológicas (`PhenoShape`) y 18 métricas LSP (`PhenoLSP`) sobre cubos xarray. Aporta el predictor. |
| [Trait_2DCNN](https://github.com/JavierLopatin/Trait_2DCNN) | Transformaciones señal→imagen, pérdida enmascarada multi-target y pretraining MAE. Aporta el marco de modelado. |

---

## Diseño acordado

- **Alcance:** Chile central. Landsat 1999–2026 como serie base; Sentinel-2 / HLS desde ~2017 para mayor
  resolución temporal en el período reciente. El sensor se trata como estrato, no se mezclan las series.
- **Abundancia (dos niveles):** métricas de presencia/ausencia sobre todas las parcelas; métricas
  ponderadas solo sobre el subset con abundancia comparable. Los targets faltantes se manejan con
  pérdida enmascarada, sin descartar parcelas ni imputar.
- **Beta y composición:** LCBD (Legendre & De Cáceres 2013) más ejes de PCoA (originalmente
  planeado como NMDS; cambiado por inestabilidad con la dispersión real de los datos, ver
  "Variables respuesta" más abajo) como respuestas continuas por parcela.
- **Validación:** validación cruzada por bloques espaciales desde el primer experimento. El CV aleatorio
  se reporta solo como referencia optimista.

---

## Financiamiento

ANID FONDECYT Iniciación 11241088 · FSEQ210022 · Fundación Data Observatory.

## Licencia

Por definir.
