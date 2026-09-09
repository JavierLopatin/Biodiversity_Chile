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
| D11 | cómputo | CPU (no hay GPU). El piloto en el pod; la corrida completa en **dask-gateway, una tesela entera por tarea**. El pod hoy da 36 núcleos y **64 GiB** (no los 123 GB de la primera recon: encogió), y ese es el techo que el gateway rompe | §8 |
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

---

## 8. Dónde corre la inferencia, y por qué ahí (medido 2026-09-03)

La primera versión de `scripts/73 --jobs N` repartía teselas entre procesos hijos del pod y
cada hijo cargaba su tesela. Las tres mediciones de abajo desarmaron dos supuestos de ese
diseño y llevaron la corrida completa al gateway. Ninguna es una estimación.

### 8.1 La carga no escala con hilos: el GIL, no la red

Una tesela de 10 km, 1.812 fechas, span 1998-2026 (`logs/thread_scaling.log`,
`logs/load_scaling.log`):

| hilos | carga | vs 1 hilo | RSS pico |
|---|---|---|---|
| 1 (síncrono) | 1.892 s | 1,0x | — |
| 4 | 643 s | 2,9x | 2,4 GB |
| 8 | 562 s | 3,4x | 3,1 GB |
| 16 | 556 s | 3,4x | 3,7 GB |
| 24 | 559 s | 3,4x | 4,3 GB |
| 8 **procesos** | 304 s | 6,2x | — |

La carga es 96 % costo fijo (1.847 s + 765 s/Mpx): es abrir ~1.800 cabeceras COG, no mover
bytes. Pero **los hilos se aplanan en 4 y no pasan de 3,4x**, mientras ocho *procesos* dan
6,2x sobre la misma tesela: GDAL parsea esas cabeceras sosteniendo el GIL. La consecuencia de
diseño es directa —el paralelismo que paga es **un proceso por tesela**, con apenas 4 hilos
adentro— y corrige lo que este documento y el docstring de `scripts/73` afirmaban antes, que
los hilos escalaban casi linealmente.

### 8.2 Un cluster por tesela acelera la parte chica

Con 10 años objetivo, una tesela son ~300 s de carga contra ~1.100 s de CNN
(`--torch-threads 1`); con 27 años, ~300 s contra ~3.000 s. Un cluster dedicado a la carga
ataca entre el 9 % y el 21 % del trabajo y deja sus workers ociosos el resto —la misma
patología que el piloto ya mostró (§6)—. Por eso el gateway recibe la **tesela entera**
(carga + curvas + CNN + escritura), no la carga.

### 8.3 Qué ve un worker del gateway (`logs/gw_probe.log`)

| | worker |
|---|---|
| recursos por defecto | 16 núcleos, 28 GB |
| home / repo / MapBiomas | **no visibles** |
| `torch` | 2.12.0+cpu |
| índice ODC | accesible (43 productos; el gateway inyecta `DB_*`) |
| versiones | numpy 2.3.5, pandas 3.0.3, rasterio 1.5.0, sklearn 1.8.0, datacube 1.9.18 |

La imagen del worker es **idéntica a la del pod**, versión por versión. Eso cierra la
advertencia de §7 sobre los pickles de `PowerTransformer` escritos con scikit-learn 1.3.1:
se leen en el worker exactamente como en el pod, y no hay una segunda combinación de
versiones que auditar.

Lo que el worker no tiene se le manda: el paquete `biodiv` como zip (`Client.upload_file`)
—con `scripts/03_extract_topography.py` adentro como `biodiv/_terrain_src.py`, porque
`load_terrain` lee `terrain()` de ese script por ruta para que siga siendo la única fuente de
las derivadas topográficas, y una carga por ruta no alcanza el interior de un zip; la copia se
reconstruye del script vivo en cada corrida, así que no pueden divergir dentro de una—,
los cinco checkpoints como bytes en un `Payload` difundido una vez —no viajando con cada una
de las ~5.900 tareas—, MapBiomas leído del bucket de scratch (`BIODIV_MAPBIOMAS_DIR`, ventana
de 557×557 en 0,68 s porque los rasters son tiled 512×512 con overviews) y los GeoTIFF
escritos de vuelta al scratch. El worker devuelve solo la fila del manifest.

**El bucle por tesela vive en `src/biodiv/maptask.py`, no en el script.** La compuerta de
`scripts/74` produce mapas solo si el camino de mapa reproduce exactamente el de
entrenamiento, y esa prueba no vale nada si el código que la compuerta revisó no es el que
corrieron los workers. Driver local y worker importan la misma función.

**El scratch se borra a los 30 días.** Los mosaicos anuales de las tres facetas publicables
(D13: LCBD, PD₀, TD₀) hay que generarlos y bajarlos dentro de esa ventana; los GeoTIFF por
tesela-año con las siete facetas son intermedios y no sobreviven a propósito.

### 8.4 Presupuesto medido, y la corrida que se lanzó (2026-09-03)

Ocho teselas repartidas de 30° a 55°S, 10 años, 8 workers de 2 núcleos
(`logs/calib_gateway.log`), medianas por tesela:

| pieza | mediana | rango |
|---|---|---|
| carga Landsat 1998-2026, una vez por tesela | **1.102 s** | 553-1.777 |
| trabajo por año | **28 s** | 1,6-56 |
| inferencia por píxel-año | 0,52 ms | — |

**La carga es el ~80 % de la tesela, no la CNN.** Eso invierte el supuesto de §6, que salía
de un piloto con torch sobre los 36 núcleos del pod; un worker recibe 2. La consecuencia es
la que decidió el alcance: la carga se paga una vez cubra la tesela 10 años o 27, así que
**27 años cuestan 35 % más que 10, no 2,7x** (3.059 contra 2.268 horas-tesela). Recortar años
ahorra poco y cuesta casi toda la serie temporal.

Cobertura nativa por tesela, sobre la misma grilla decimada de `scripts/76`
(`results/figures/tiles_native_10km_cover.csv`): mediana 1.532 píxeles de ~3.000 posibles.
La sospecha de que muchas teselas retenidas estaban casi vacías **era falsa** —la
distribución está cargada hacia teselas llenas—; t16_455, que pagó 1.102 s de carga para 271
píxeles, es la excepción. Descartar las de menos de 20 píxeles nativos saca 123 teselas
(2,1 %), ahorra ~38 h y pierde 0,01 % del área: se aplicó, y el resto no, porque a partir de
ahí ya se cambia cobertura por tiempo.

Lo que corre (decisiones del autor, 2026-09-03):

```bash
python scripts/73_map_inference.py \
    --tiles-file results/figures/tiles_native_10km_run.csv --years 2000-2026 \
    --area-m2 900 --stratum basal --mask mapbiomas \
    --gw-workers 128 --worker-cores 2 --worker-memory 8 --worker-threads 1 \
    --load-threads 4 --dest s3://<scratch>/biodiv/maps/chile_30m_2000_2026 \
    --mapbiomas-dir s3://<scratch>/biodiv/MapBiomas \
    --out results/maps --tag chile_full_30m --resume
```

5.769 teselas, 27 años, ~191 GB de salida en el scratch. Con los 128 workers concedidos en
pleno es ~1 día; con menos, proporcionalmente más. **Pendiente y con plazo:** los mosaicos
anuales de LCBD, PD₀ y TD₀ hay que construirlos y bajarlos antes de que el scratch expire a
los 30 días.

### 8.5 La corrida se detiene sola, y por qué (2026-09-04)

El contenedor se reinició durante la noche y mató las dos mitades: los 14 procesos del pod
—sus logs terminan a mitad de tesela, sin errores— y el driver del gateway. `setsid` protege
de que termine la sesión que lanzó el trabajo, que es de lo que protegió dos veces; **no
protege de que se reinicie el contenedor**, y nada que corra dentro de él puede hacerlo.

La causa encaja con un **culler por inactividad**: última actividad interactiva a las 20:07,
contenedor levantado de nuevo a las 01:51 cuando algo volvió a abrirlo. No está confirmado
directamente —el hub responde 403 en `/hub/api/services` y `/info` con el token del pod— pero
sí lo está el desajuste de fondo:

> El servidor reporta actividad al hub (`JUPYTERHUB_ACTIVITY_URL`), y JupyterHub la mide por
> **kernels y peticiones HTTP, no por CPU**. Quince procesos saturando 36 núcleos son
> invisibles para esa métrica: a ojos del hub este servidor estaba ocioso mientras corría a
> plena carga.

Decisión del autor (2026-09-04): **aceptarlo y relanzar**, en vez de registrar actividad
artificial contra `JUPYTERHUB_ACTIVITY_URL` —que sería derrotar a propósito un mecanismo de
reparto en infraestructura compartida— o de pedir una exención a los admins. La consecuencia
para el calendario hay que declararla: 4,2 días de cómputo se convierten en 7-10 días de
calendario si se pierden ~6 h por noche.

Lo que se relanza, una sola línea, idempotente (una mitad ya viva se deja en paz, porque
lanzarla dos veces correría las mismas teselas en dos sitios y competiría por el manifiesto):

```bash
cd ~/temp/Biodiversity_Chile && scripts/start_maps.sh
```

`scripts/run_maps_supervised.sh <gateway|pod>` es el supervisor por mitad: cuenta lo que falta
como las teselas sin sus 27 años escritos y relanza con `--resume` hasta que no queda ninguna.
Un relanzamiento solo cuesta las teselas que estaban en vuelo.
