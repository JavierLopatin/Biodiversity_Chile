# Mapas multitemporales de facetas: especificación de inferencia

Cómo se producen los mapas anuales 2000–2026 de las siete facetas con el modelo final
(`scripts/72_train_final_map_model.py --all-data`), dónde corre cada parte, y qué
decisiones se tomaron. Las decisiones son de J. Lopatin (2026-09-02, sesión coordinada
desde el Mac); la ejecución corre en el pod de Data Cube Chile (`jupyter-jlopatin`).

## 1. Qué modelo y qué entrada

| pieza | valor | fuente |
|---|---|---|
| modelo | 2D-CNN `PhenoNetS` (width B, separable, late fusion) inicializada desde MAE, 15.175 parámetros | `docs/20` §6, `scripts/72` |
| checkpoints | 5 semillas, `results/models_unified/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL_alldata/final/model_seed{0..4}.pt`, refit sobre las 3.102 parcelas, 33 épocas fijas | `logs/81` |
| entrada fenológica | serie **cruda** kNDVI del **píxel central**, ventana causal `y−2..y`, 100 pasos, media móvil de 5, imagen `serpentine` 10×10 | `biodiv.curves.interp_grid`, `biodiv.transforms1d` |
| contexto | 8 variables topográficas del píxel (`features.TOPO_VARS`) + bandera de terreno plano + `log10(área)` + 3 indicadores de estrato; 3 indicadores de faltante ⇒ 16 columnas | `ck["ctx_preprocessor"].columns_` |
| salidas | `lcbd_count_sorensen`, `pd_inext_q0/q1/q2`, `td_inext_q0/q1/q2` | `ck["targets"]` |

El código de inferencia es `src/biodiv/mapinfer.py` (núcleo numérico, probado sin
datacube en `tests/test_mapinfer.py`) y `scripts/73_map_inference.py` (driver por
tesela). `scripts/74_check_map_consistency.py` es la compuerta previa: sobre las 3.102
parcelas de entrenamiento, el camino de mapa debe reproducir **exactamente** el contexto,
la imagen y la salida del modelo del camino de entrenamiento, y el escalador de targets
debe cerrar el viaje de ida y vuelta. No se produce ningún mapa si esa compuerta falla.

## 2. Decisiones

| # | decisión | valor | por qué |
|---|---|---|---|
| D1 | extensión | envolvente de las parcelas unificadas (30,2–55,0°S) + 20 km, **solo vegetación nativa** según MapBiomas | el modelo se entrenó en parcelas de vegetación nativa; fuera de eso no hay soporte. Elección del autor (Q1) |
| D2 | máscara | clases nativas `{3, 59, 60, 61, 11, 12, 63, 66}` del mapa anual **más cercano** (`biodiv.mapbiomas.year_map`); 2025 y 2026 usan 2024 y llevan `map_delta` en los metadatos | MapBiomas termina en 2024 |
| D3 | área de parcela | **900 m²** (un píxel Landsat), constante | elección del autor (Q2): el mapa se lee como "diversidad esperada en una unidad de 900 m²". Está por encima del percentil 90 del pool (500 m²); es extrapolación leve y declarada |
| D4 | estrato | **`basal`** (protocolo de inventario, Living Trees) | 81 % del pool LCBD y 73 % del pool TD/PD; pendiente de confirmación explícita del autor |
| D5 | años | **2000–2026**; la ventana de 2000 (1998–2000) existe (L5 14 + 9 + 2 escenas, L7 13 + 34 en el bbox de prueba); 2026 es parcial (hasta agosto) y se publica con `span_days`/`n_obs` como aviso | lectura del archivo en el pod |
| D6 | grilla de la curva | 100 pasos entre la **primera y la última fecha con algún píxel despejado en la tesela** dentro de la ventana; interpolación lineal por píxel con extrapolación constante a los bordes; NaN con < 5 observaciones | réplica del cubo por parcela (`raw_series`: primera..última fecha del cubo); probada contra `interp_grid` píxel a píxel |
| D7 | teselas | 10 km ajustados a píxeles enteros (333 × 30 m = 9.990 m), origen en múltiplos, EPSG:32719 | mismo retículo que `dc.load`; los mosaicos no remuestrean |
| D8 | lectura Landsat | una sola carga por tesela para 1998–2026 (`red`, `nir`, `qa_pixel`; misma máscara QA por producto que `cube.load_window`), y cada año se corta en memoria | 27 años objetivo comparten 29 años de archivo; cargar por año leería 3× |
| D9 | retransformación | por semilla: clip al rango de entrenamiento en el espacio Yeo-Johnson, **Duan smearing con los residuos OOF del run block20 de la misma configuración** (`--oof-csv`, por defecto), clip al rango observado, y promedio de las 5 semillas | LCBD tiene λ ≈ −4.205 y escala 3,5e-6: sin clip un z fuera de rango explota. Medido en la compuerta (pod, 2026-09-02): la inversa simple pierde 0,25 de R² in-sample en TD₀ (0,50 → 0,74; λ = −2,2) y 0,04 en PD₀; sin smearing no se publica |
| D13 | facetas que se publican | **LCBD, PD₀ y TD₀**; las cuatro facetas ponderadas (PD₁, PD₂, TD₁, TD₂) se escriben en los GeoTIFF por tesela para diagnóstico pero no se mosaican ni se publican | su R² de block-CV es ≤ 0,04 o negativo (referencia recalculada del OOF: PD₁ −0,010, PD₂ +0,040, TD₁ −0,181, TD₂ −0,090): sin habilidad validada. Por defecto salvo indicación del autor |
| D10 | salida | un GeoTIFF por (tesela, año), 10 bandas float32: 7 facetas + `n_obs` + `span_days` + `native`; deflate, tiled; tags con todas las decisiones; `manifest.csv` reanudable | mosaico anual por `gdalbuildvrt` después |
| D11 | cómputo | CPU (el pod no tiene GPU: 32 cores, 123 GB); dask local para el piloto, dask-gateway (workers de 8 cores / 28 GB) para la corrida completa | recon del pod |
| D12 | despliegue | **piloto primero**: una tesela con 52 parcelas (Cauquenes, −36,0/−72,4), todos los años, medida en segundos/tesela y comparada con los valores de parcela; con eso se decide extensión final y resolución | elección del autor (Q3) |

## 3. Tamaño del problema (medido en el pod, MapBiomas 2024, decimado 10×)

277.933 km² de vegetación nativa entre 30° y 55,2°S ⇒ ~4,4·10⁸ píxeles de 30 m por año,
~1,2·10¹⁰ píxel-años, ~6·10¹⁰ pasadas de la CNN (5 semillas). Aproximadamente 2.800
teselas de 10 km. Lo que decide el calendario es la lectura de ~1.400 escenas por tesela
desde S3, no la CNN; el piloto mide ambas. Si el costo obliga, las alternativas —en este
orden— son: (a) restringir la extensión (Chile central primero), (b) grilla de salida de
90 m leyendo el píxel más cercano (misma señal de 30 m, 9× menos píxeles), (c) menos años.

## 4. Compuerta y piloto (qué corre el pod)

```bash
git pull --ff-only origin main
BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python scripts/74_check_map_consistency.py
# solo si termina en ALL PASS:
python scripts/73_map_inference.py --bbox -72.50 -36.10 -72.35 -35.95 --years 2000-2026 \
    --area-m2 900 --stratum basal --mask mapbiomas --workers 8 --out results/maps --tag pilot_cauquenes
```

Antes de la corrida, la compuerta `scripts/74 --oof-csv ...` debe dar ALL PASS con los
checkpoints refit sobre la topografía corregida, y además el R² in-sample con smearing de
TD₀ debe alcanzar al menos el nivel que los checkpoints viejos daban sobre su propia
topografía (0,78): la compuerta por sí sola no distingue topografías (ALL PASS también con
la vieja), así que ese nivel es el criterio de que el refit recuperó lo perdido.

Reportar: segundos de carga por tesela, segundos por año, píxeles predichos por año,
rango de cada faceta, y la comparación píxel-de-parcela vs valor observado para las
parcelas de la tesela en su año de censo.

## 5. Corrección previa obligatoria: convención de aspecto (2026-09-02)

Verificado en el pod recalculando `terrain()` corregido sobre 30 parcelas Parcelas-CL y 10
Living Trees: en las filas `PCL_` de `topography_unified.parquet` `northness` y `eastness`
tienen correlación −1,000 con el recálculo (giro de 180°, las 1.082 filas) y `heat_load`
está espejado; en las filas `LT_` el acuerdo es 100 % (heat_load a 2e-4). El modelo final y
todos los runs de `docs/20` se entrenaron con esa mezcla. Decisión del autor: regenerar la
topografía de Parcelas-CL con el `scripts/03` corregido, reconstruir `topography_unified`,
re-correr block20, LLTO y barridos, refit `--all-data`, y actualizar `docs/20` y el paper.
Hasta que eso termine no se produce ningún mapa. Los valores antiguos quedan en git
(`b921daa`). Hallazgo colateral: `plots_unified.parquet` tiene `X`/`Y` (UTM 19S) en NaN
para las 2.020 filas `LT_`; se corrige en `scripts/51` con reproyección desde lon/lat.

## 6. Piloto medido (2026-09-02, tesela t18_600, Cauquenes)

Tesela de 9.990 m (333 x 333 = 110.889 px), 27 años, 8 workers locales, checkpoints
anteriores a la corrección de topografía (la corrida equivalente con los checkpoints
corregidos está en curso):

| magnitud | valor |
|---|---|
| carga Landsat 1998-2026 (1.811 fechas), una sola vez por tesela | 346 s |
| trabajo por año (curvas + inferencia + escritura) | mediana 31,2 s (21,9-37,7) |
| píxeles predichos por año (nativos con curva completa) | mediana 46.369 (31.901-53.173) |
| GeoTIFF por tesela-año, 10 bandas | 1,5 MB (41 MB los 27 años) |
| RAM | 6-8 GB de 123 disponibles |

Dos lecturas operativas. **La carga se amortiza**: es el 29 % de una corrida de 27 años y
se paga una vez por tesela. **El cuello de botella es el trabajo por año**, que corre en un
solo proceso mientras los ocho workers quedan ociosos: 0,673 ms por píxel-año. Extrapolado
a los 3,1·10⁸ píxeles nativos por año, la corrida completa 2000-2026 a 30 m son ~1.830 h en
serie (1.560 de inferencia + 270 de carga), es decir 57 h con 32 procesos o ~14 h en un
cluster de 128 núcleos. Alternativas medidas sobre la misma base: grilla de salida de 90 m
(441 h en serie), un año de cada tres (787 h), o restringir a 30-38°S (505 h).

**Subdispersión, declararla antes de que alguien lea un máximo como valor real.** En la
tesela piloto TD₀ llega a 42,8 contra un máximo observado de 84,0 en las parcelas, y la
mediana del mapa (6,8) queda bajo la mediana observada (9,3). El modelo comprime la cola
alta incluso con smearing; los mapas se interpretan como superficie relativa, no como
conteos absolutos de especies.

**La comparación con parcelas dentro de una tesela no es validación.** En el piloto caen
7-11 parcelas; la correlación de Spearman con ese n tiene error estándar cercano a 0,4. La
validación del modelo es la de la Sección 4 (bloques espaciales y LLTO); el contraste por
píxel es solo una lectura de coherencia.

## 7. Advertencias operativas

- `dask_gateway.Gateway().cluster_options()` imprime credenciales AWS STS y una
  contraseña de base de datos en su `repr`. No imprimirlo en notebooks, logs ni archivos
  versionados. `scripts/73` solo imprime `cluster.name` y el enlace del dashboard.
- Los pickles de `PowerTransformer` se escribieron con scikit-learn 1.3.1 y se leen con
  1.8.0 (`InconsistentVersionWarning`). La compuerta 1 de `scripts/74` (ida y vuelta del
  escalador) es la que dice si eso importa.
- `ck["rows"]` son las etiquetas de fila de la imagen serpentine, no las columnas de
  contexto; las columnas de contexto salen de `ck["ctx_preprocessor"].columns_`.
