# Protocolo STeMP

Protocolo de reporte para modelos espacio-temporales de aprendizaje automático (Linnenbrink et al. 2026). Los campos son los de su Tabla A1; **en negrita** los obligatorios.

**Generado por** `scripts/38_stemp_protocol.py`. No editar a mano: cada cifra sale de `results/`. 222 corridas en la tabla de origen.


## 1. Overview

| campo | valor |
|---|---|
| **Model title** | Predicción de facetas de diversidad vegetal desde fenología Landsat, Chile central |
| **Author names** | J. Lopatin |
| **Contact** | javierlopatin@gmail.com |
| Study title | Multi-faceted plant diversity from satellite phenology |
| Study link | **PENDIENTE** |
| **Target variable** | 5 facetas, 15 respuestas: alpha (3); beta_pa (3); beta_cover (3); phylo (5); dark (1) |
| **Scientific field** | Ecología / teledetección |

## 2. Model

### 2.1 Learning method

| campo | valor |
|---|---|
| **Model type** | Regresión multi-salida |
| **Algorithm** | MLP (titular), Random Forest, CNN 1-D y 2-D sobre transformaciones señal→imagen, línea base de coordenadas |
| Architecture | MLP06: bloque de curva + contexto clima/topografía/área. C2D: separable, 17k parámetros, 5 sustratos probados |

### 2.2 Response

| campo | valor |
|---|---|
| **Sample size** | 1082 parcelas |
| **Sampling extent** | X 126732–407321, Y 5800807–6651332 (EPSG:32719); años 2003–2026 |
| **Sample acquisition** | Inventarios de vegetación en terreno, recopilados en Parcelas-CL; 12 contribuyentes, 23 conjuntos |
| **Sample geometry** | Polígonos tratados como puntos; superficie 78–10000 m² (mediana 314), la mayoría sub-píxel Landsat |
| **Range** | hill_q0: 1082 obs; hill_q1: 1082 obs; hill_q2: 1082 obs |
| **Sampling pattern** | **clustered** — Clark-Evans R = 0.144. Distancia entre parcelas (Gj) mediana 0.149 km; del dominio nativo a la parcela más próxima (Gij) mediana 13.23 km. El mapa predecirá **88.92× más lejos** de lo que separa a las parcelas entre sí |
| **Coordinate Reference System** | EPSG:32719 (UTM 19S) |
| **Data sources of response** | Parcelas-CL (Chilean vegetation plot database) |

### 2.3 Predictors

| campo | valor |
|---|---|
| **Predictor types** | Curva fenológica por índice de vegetación; métricas LSP; topografía; clima; área de parcela |
| **Number of predictors** | 121 en el diseño titular (curva de 100 pasos + 21 de contexto) |
| **Resolution of predictors** | 30 m (Landsat C2L2); ventana de 5×5 píxeles por parcela; paso temporal ~11 días sobre 3 años |
| Preprocessing | Escalado C2 (DN·2,75e-5 − 0,2); máscara QA por producto antes de concatenar sensores; interpolación a grilla regular con `biodiv.curves.interp_grid` |
| **Temporal alignment** | **Ventana causal `y−2..y`**: 3 años que terminan en el año del censo. Ninguna observación es posterior al censo |
| **Data sources of predictors** | Landsat 5/7/8/9 C2L2 vía Data Cube Chile; Copernicus DEM 30 m; CHELSA/WorldClim; MapBiomas Chile para la máscara nativa |

### 2.4 Model evaluation and selection

| campo | valor |
|---|---|
| **Model evaluation strategy** | Validación cruzada agrupada de 5 folds por **componente de ventana** (`kfold5_window`): cero fuga de píxeles por construcción, 0 de 135 componentes partidos. Complementada con bloques de 20 km, LTO y LLTO — ver §2.5 |
| **Performance metrics** | R² fuera de muestra, %RMSE y sesgo normalizado, en unidades originales tras retransformación de Yeo-Johnson con el estimador de Duan |
| **Model evaluation results** | Mejor modelo `MLP06_curve_kndvi_raw100`: R² medio sobre facetas = 0.396 bajo `kfold5_window`. Por facetas: alpha 0.519, beta_pa 0.529, beta_cover 0.342, phylo 0.164, dark 0.427 |
| Hyperparameter tuning | Búsqueda por coordenadas sobre ~195 combinaciones (reconstructor de curva, sustrato, arquitectura, aumentación, regularización); rejilla completa publicada en `docs/14` |
| Predictor selection | Cribado por bloques (19 combinaciones) más comparaciones pareadas; **las coordenadas no entran en los modelos titulares**, sólo como línea base `B03_coords` |

**Mejor corrida de cada familia** (R² medio sobre facetas, `kfold5_window`):

| familia | corrida | alpha | beta_pa | beta_cover | phylo | dark | media |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | `MLP06_curve_kndvi_raw100_s10` | 0.520 | 0.529 | 0.352 | 0.162 | 0.419 | **0.396** |
| C2D | `C2D02_serpentine_kndvi_raw24_ctxclim-topo-area_s10` | 0.502 | 0.505 | 0.328 | 0.170 | 0.413 | **0.383** |
| RF | `RF06_curve_all-topo-area_raw100_s10` | 0.424 | 0.534 | 0.386 | 0.158 | 0.413 | **0.383** |
| BASE | `B03_coords_s10` | 0.491 | 0.508 | 0.347 | 0.089 | 0.380 | **0.363** |
| C1D | `C1D01_curve1d_kndvi_raw36_ctxclim-topo-area_s10` | 0.455 | 0.497 | 0.311 | 0.168 | 0.381 | **0.362** |

**Sesgo de selección** (`scripts/35_selection_audit.py`). Seleccionar y evaluar sobre la misma partición infla el R²; medido eligiendo el ganador sobre 4 de los 5 folds y puntuándolo en el quinto: global **+0.0000** (el ganador es el mismo en los 5 subconjuntos). Por familia el sesgo escala con cuánto se buscó: 114 corridas C2D → +0,0142; 4 corridas BASE → 0,0000.


### 2.5 Transferencia espacial y temporal

Distancia a la que cada esquema evalúa, contra los 13.23 km a los que el dominio nativo dista de la parcela más próxima:

| esquema | mediana test→entrenamiento | fracción de la distancia del mapa |
|---|---:|---:|
| `kfold5_window` | 0.49 km | 0.04 |
| `time_within_owner` | 7.56 km | 0.57 |
| `kfold_time` | 10.07 km | 0.76 |
| `kfold5_block20` | 12.05 km | 0.91 |
| `kfold_loc_time` | 14.09 km | 1.07 |

R² medio del **mismo modelo** bajo cada esquema:

| familia | `kfold5_window` | `kfold_time` |
|---|---:|---:|
| BASE | 0.363 | 0.001 |
| C1D | 0.360 | **PENDIENTE** |
| C2D | 0.383 | -0.121 |
| MLP | 0.396 | -0.082 |
| RF | 0.381 | **PENDIENTE** |


### 2.6 Interpretation, uncertainty and limitations

| campo | valor |
|---|---|
| Explainability | Importancia por permutación en RF; ablación por bloques de predictores; descomposición within/between contribuyente |
| Scientific interpretation | Beta se sostiene con el nivel del contribuyente removido (R² intra +0,380); alfa es en un 88 % nivel de contribuyente y se reporta como descriptor |
| Potential biases | (a) η²(contribuyente) = 0,72 sobre la riqueza; (b) el 10,7 % de las parcelas cae en clases que la máscara nativa excluye (6,1 % silvicultura, 3,2 % agricultura, 1,4 % urbano); (c) contribuyente y año están confundidos (Cramér's V = 0,732), así que un leave-time-out no separa ambos efectos |
| Sensitivity assessment | 20 reconstructores de curva (dispersión 0,010), 8 resoluciones de serie cruda (0,011), 5 semillas; ruido de no-determinismo a semilla fija medido en 0,023 de dispersión |
| Limitations | (a) `log10_area` y `stratum` están en todos los diseños y **no son mapeables**: un píxel no tiene superficie de parcela; (b) `lon/lat/elev` alcanzan el 90 % del beta que logran los 87 predictores, así que el aporte propio de la teledetección es +0,054; (c) el esquema titular evalúa a 0,5 km y el mapa predice a 13 km |
| Software | Python 3.11, PyTorch 2.6, scikit-learn, xarray/datacube; R para filogenia y diversidad oscura |
| Code availability | github.com/JavierLopatin/Biodiversity_Chile |
| Data availability | Parcelas-CL público; predictores derivados en `data/derived/` |

## 3. Prediction

> El modelo **todavía no produce un mapa**. Esta sección es opcional en STeMP precisamente para este caso, y se rellena con lo que ya está medido y condiciona el mapa futuro.

| campo | valor |
|---|---|
| Prediction extent | Vegetación nativa de Chile central según MapBiomas: ~83.000 km², 1.449 celdas de 10 km |
| Prediction resolution | 30 m, año a año dentro de 2003–2026 |
| Map evaluation strategy | PENDIENTE hasta que exista mapa. El esquema adecuado **no** es `kfold5_window` sino uno cuya distancia de evaluación se parezca a la del mapa (`kfold5_block20`, `kfold_loc_time`) |
| **Uncertainty quantification** | **Área de aplicabilidad** (Meyer & Pebesma 2021) sobre 16,950 muestras de cobertura nativa: **88.2 % dentro**. Umbral DI = 0.352 desde los folds de `kfold5_block20`; DI del dominio mediana 0.204, p95 0.429 |
| Map accuracy | **PENDIENTE** |
| Threshold selection | No aplica: las respuestas son continuas |
| Post-processing | No aplica |

## 4. Los cuatro avisos automáticos de STeMP

| aviso | estado |
|---|---|
| Estrategia de evaluación inadecuada al patrón de muestreo | **Parcialmente.** `kfold5_window` cierra la fuga de píxeles por construcción, pero evalúa a 0,5 km mientras el mapa predice a 13 km. Declarado y complementado con block20 y LLTO |
| Fuga entre selección y evaluación | **Medido, no evitado.** Auditado en `scripts/35_selection_audit.py`: 0,000 global, +0,014 en la familia más buscada |
| Falta de cuantificación de incertidumbre en extrapolación | **Cubierto.** AOA calculado sobre el dominio nativo antes de producir mapa |
| Proxies espaciales sin selección de predictores | **No aplica.** Las coordenadas sólo entran como línea base `B03_coords`, nunca en los modelos titulares |
